#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "anchor/random.hpp"
#include "anchor/range_query.hpp"
#include "anchor/types.hpp"

namespace anchor {

// A deliberately direct, node-based multi-level range tree.
//
// Every node stores the active weight of its subtree.  At every nonterminal
// coordinate, every node owns a recursively built associated tree over the
// same point subset.  Queries descend to a disjoint family of canonical
// subtrees; sampling chooses one such subtree by weight and then performs
// active-rank selection down its node tree.
//
// This is the generic static/dynamic adapter of the standalone reference
// implementation.  The adapter keeps the benchmark's coordinate
// types, exact UInt128 weights, stable object tie breakers, and RNG streams,
// but intentionally does not use terminal associated arrays or Fenwick trees.
template <Coordinate Coord>
class NodeRangeTree {
 public:
  using Index = std::uint32_t;
  using CoordinateAccessor = std::function<Coord(Index, std::size_t)>;
  using IdAccessor = std::function<std::uint64_t(Index)>;

  NodeRangeTree(std::size_t dimensions, std::vector<Index> points,
                CoordinateAccessor coordinate, IdAccessor id, bool dynamic,
                std::size_t object_universe)
      : dimensions_(dimensions),
        dynamic_(dynamic),
        internal_by_point_(object_universe, npos()) {
    if (dimensions_ == 0) {
      throw std::invalid_argument("zero-dimensional range tree is a dense set");
    }

    points_.reserve(points.size());
    for (Index point : points) {
      if (point >= object_universe) {
        throw std::invalid_argument("point index exceeds range-tree universe");
      }
      if (internal_by_point_[point] != npos()) {
        throw std::invalid_argument("duplicate point in range-tree skeleton");
      }

      Point stored;
      stored.payload = point;
      stored.id = id(point);
      stored.active = !dynamic_;
      stored.coordinates.reserve(dimensions_);
      for (std::size_t axis = 0; axis < dimensions_; ++axis) {
        const Coord value = coordinate(point, axis);
        validate_coordinate(value);
        stored.coordinates.push_back(value);
      }

      internal_by_point_[point] = points_.size();
      points_.push_back(std::move(stored));
    }

    estimated_bytes_ += points_.capacity() * sizeof(Point);
    for (const Point& point : points_) {
      estimated_bytes_ +=
          point.coordinates.capacity() *
          sizeof(typename decltype(point.coordinates)::value_type);
    }
    estimated_bytes_ += internal_by_point_.capacity() * sizeof(std::size_t);

    if (!points_.empty()) {
      std::vector<std::size_t> internal(points_.size());
      for (std::size_t i = 0; i < internal.size(); ++i) internal[i] = i;
      root_level_ = std::make_unique<RangeLevel>(*this, 0, internal);
    }
  }

  NodeRangeTree(const NodeRangeTree&) = delete;
  NodeRangeTree& operator=(const NodeRangeTree&) = delete;
  NodeRangeTree(NodeRangeTree&&) = delete;
  NodeRangeTree& operator=(NodeRangeTree&&) = delete;

  [[nodiscard]] std::size_t dimensions() const noexcept { return dimensions_; }
  [[nodiscard]] bool dynamic() const noexcept { return dynamic_; }
  [[nodiscard]] std::size_t estimated_bytes() const noexcept {
    return estimated_bytes_;
  }
  [[nodiscard]] std::size_t skeleton_nodes() const noexcept {
    return node_count_;
  }
  [[nodiscard]] std::size_t point_references() const noexcept {
    return point_reference_count_;
  }

  [[nodiscard]] UInt128 count(std::span<const AxisRange<Coord>> query) const {
    const QueryState state = validate_query(query);
    if (state == QueryState::Empty || !root_level_) return 0;
    std::vector<Block> blocks;
    root_level_->collect_blocks(query, blocks);
    UInt128 total = 0;
    for (const Block& block : blocks) {
      total = checked_add(total, block.weight, "range count overflow");
    }
    return total;
  }

  // Returns false exactly when a nonempty output was requested from an empty
  // query result.  Sampling is iid with replacement and never changes active
  // state.  The two RNGs preserve the benchmark's block/rank stream split.
  bool sample(std::span<const AxisRange<Coord>> query, std::span<Index> output,
              DeterministicRng& block_rng, DeterministicRng& rank_rng) const {
    const QueryState state = validate_query(query);
    if (output.empty()) return true;
    if (state == QueryState::Empty || !root_level_) return false;

    std::vector<Block> blocks;
    root_level_->collect_blocks(query, blocks);
    std::vector<UInt128> prefix;
    prefix.reserve(blocks.size());
    UInt128 total = 0;
    for (const Block& block : blocks) {
      if (block.weight == 0) continue;
      total = checked_add(total, block.weight,
                          "range-query sample weight overflow");
      prefix.push_back(total);
    }
    if (total == 0) return false;

    // collect_blocks never emits zero-weight blocks, so prefix and blocks
    // have identical indexing.  Keep this invariant explicit.
    if (prefix.size() != blocks.size()) {
      throw std::logic_error("zero-weight canonical range-tree block");
    }

    for (Index& destination : output) {
      const UInt128 draw = uniform_below(block_rng, total);
      const auto iterator =
          std::upper_bound(prefix.begin(), prefix.end(), draw);
      if (iterator == prefix.end()) {
        throw std::logic_error("range-tree block selection escaped prefix");
      }
      const std::size_t block_index =
          static_cast<std::size_t>(iterator - prefix.begin());
      const Block& block = blocks[block_index];
      const UInt128 rank = uniform_below(rank_rng, block.weight);
      destination = select_active_by_rank(block.node, rank);
    }
    return true;
  }

  void set_active(Index point, bool value) {
    if (!dynamic_) {
      throw std::logic_error("static range tree has no active-state updates");
    }
    const std::size_t internal = internal_index(point);
    Point& stored = points_[internal];
    if (stored.active == value) {
      throw std::logic_error(value ? "duplicate range-tree insertion"
                                   : "range-tree deletion of inactive point");
    }
    stored.active = value;
    if (!root_level_) {
      throw std::logic_error("missing range-tree root during update");
    }
    root_level_->activate(internal, value ? +1 : -1);
  }

  [[nodiscard]] bool is_active(Index point) const {
    if (point >= internal_by_point_.size()) return false;
    const std::size_t internal = internal_by_point_[point];
    return internal != npos() && points_[internal].active;
  }

  [[nodiscard]] bool all_inactive() const noexcept {
    return !root_level_ || root_level_->active_count() == 0;
  }

 private:
  struct RangeLevel;

  struct Point {
    Index payload{};
    std::uint64_t id{};
    std::vector<Coord> coordinates;
    bool active{};
  };

  struct Key {
    Coord coordinate{};
    std::uint64_t id{};
    Index payload{};
    std::size_t internal{};
  };

  struct Node {
    Coord min_coordinate{};
    Coord max_coordinate{};
    Key max_key;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
    std::unique_ptr<RangeLevel> associated;
    std::size_t point_index{npos()};
    UInt128 active_count{};

    [[nodiscard]] bool is_leaf() const noexcept {
      return point_index != npos();
    }
  };

  struct Block {
    const Node* node{};
    UInt128 weight{};
  };

  enum class Relation : std::uint8_t { Outside, Inside, Partial };
  enum class QueryState : std::uint8_t { Nonempty, Empty };

  struct RangeLevel {
    RangeLevel(NodeRangeTree& owner, std::size_t dimension,
               const std::vector<std::size_t>& point_indices)
        : owner_(owner), dimension_(dimension) {
      owner_.estimated_bytes_ += sizeof(RangeLevel);
      if (!point_indices.empty()) {
        std::vector<std::size_t> copy = point_indices;
        root_ = build(copy);
      }
    }

    void activate(std::size_t point_index, int delta) {
      update(root_.get(), point_index, delta);
    }

    void collect_blocks(std::span<const AxisRange<Coord>> query,
                        std::vector<Block>& blocks) const {
      collect(root_.get(), query, blocks);
    }

    [[nodiscard]] UInt128 active_count() const noexcept {
      return root_ ? root_->active_count : 0;
    }

   private:
    std::unique_ptr<Node> build(std::vector<std::size_t>& indices) {
      if (indices.empty()) {
        throw std::logic_error("cannot build an empty range-tree node");
      }
      auto node = std::make_unique<Node>();
      ++owner_.node_count_;
      owner_.estimated_bytes_ += sizeof(Node);

      node->min_coordinate =
          owner_.points_[indices.front()].coordinates[dimension_];
      node->max_coordinate = node->min_coordinate;
      for (std::size_t point_index : indices) {
        const Coord coordinate =
            owner_.points_[point_index].coordinates[dimension_];
        if (coordinate < node->min_coordinate) {
          node->min_coordinate = coordinate;
        }
        if (node->max_coordinate < coordinate) {
          node->max_coordinate = coordinate;
        }
      }

      std::sort(indices.begin(), indices.end(),
                [&](std::size_t lhs, std::size_t rhs) {
                  return owner_.key_less(owner_.make_key(lhs, dimension_),
                                         owner_.make_key(rhs, dimension_));
                });
      node->max_key = owner_.make_key(indices.back(), dimension_);
      node->active_count =
          owner_.dynamic_ ? UInt128{0} : static_cast<UInt128>(indices.size());

      if (indices.size() == 1) {
        node->point_index = indices.front();
        ++owner_.point_reference_count_;
      } else {
        const std::size_t middle = indices.size() / 2;
        std::vector<std::size_t> left_indices(
            indices.begin(),
            indices.begin() + static_cast<std::ptrdiff_t>(middle));
        std::vector<std::size_t> right_indices(
            indices.begin() + static_cast<std::ptrdiff_t>(middle),
            indices.end());
        node->left = build(left_indices);
        node->right = build(right_indices);
      }

      if (dimension_ + 1 < owner_.dimensions_) {
        node->associated =
            std::make_unique<RangeLevel>(owner_, dimension_ + 1, indices);
      }
      return node;
    }

    void update(Node* node, std::size_t point_index, int delta) {
      if (!node) {
        throw std::logic_error("missing range-tree node during update");
      }
      if (delta > 0) {
        node->active_count = checked_add(node->active_count, UInt128{1},
                                         "range-tree active-count overflow");
      } else {
        if (node->active_count == 0) {
          throw std::logic_error(
              "corrupted range-tree active count during deletion");
        }
        --node->active_count;
      }

      if (node->associated) {
        node->associated->activate(point_index, delta);
      }
      if (node->is_leaf()) return;

      const Key key = owner_.make_key(point_index, dimension_);
      if (owner_.key_less(node->left->max_key, key)) {
        update(node->right.get(), point_index, delta);
      } else {
        update(node->left.get(), point_index, delta);
      }
    }

    void collect(const Node* node, std::span<const AxisRange<Coord>> query,
                 std::vector<Block>& blocks) const {
      if (!node || node->active_count == 0) return;

      const Relation relation =
          owner_.relation_for_axis(*node, query[dimension_]);
      if (relation == Relation::Outside) return;
      if (relation == Relation::Inside) {
        if (dimension_ + 1 == owner_.dimensions_) {
          blocks.push_back(Block{node, node->active_count});
        } else {
          if (!node->associated) {
            throw std::logic_error(
                "missing associated range-tree level during query");
          }
          node->associated->collect_blocks(query, blocks);
        }
        return;
      }

      if (node->is_leaf()) {
        const Point& point = owner_.points_[node->point_index];
        if (point.active && owner_.point_satisfies(point, query)) {
          blocks.push_back(Block{node, UInt128{1}});
        }
        return;
      }
      collect(node->left.get(), query, blocks);
      collect(node->right.get(), query, blocks);
    }

    NodeRangeTree& owner_;
    std::size_t dimension_{};
    std::unique_ptr<Node> root_;
  };

  static constexpr std::size_t npos() noexcept {
    return std::numeric_limits<std::size_t>::max();
  }

  static void validate_coordinate(Coord value) {
    if constexpr (std::same_as<Coord, double>) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument(
            "range-tree coordinates and bounds must be finite");
      }
    }
  }

  [[nodiscard]] QueryState validate_query(
      std::span<const AxisRange<Coord>> query) const {
    if (query.size() != dimensions_) {
      throw std::invalid_argument("range query dimensionality mismatch");
    }
    for (const AxisRange<Coord>& range : query) {
      if (range.lower) validate_coordinate(*range.lower);
      if (range.upper) validate_coordinate(*range.upper);
      if (!range.lower || !range.upper) continue;
      if (*range.upper < *range.lower) return QueryState::Empty;
      const bool equal =
          !(*range.lower < *range.upper) && !(*range.upper < *range.lower);
      if (equal && (range.lower_strict || range.upper_strict)) {
        return QueryState::Empty;
      }
    }
    return QueryState::Nonempty;
  }

  [[nodiscard]] std::size_t internal_index(Index point) const {
    if (point >= internal_by_point_.size()) {
      throw std::out_of_range("range-tree active point index");
    }
    const std::size_t internal = internal_by_point_[point];
    if (internal == npos()) {
      throw std::invalid_argument(
          "point is not part of the range-tree skeleton");
    }
    return internal;
  }

  [[nodiscard]] Key make_key(std::size_t point_index,
                             std::size_t dimension) const {
    const Point& point = points_[point_index];
    return Key{point.coordinates[dimension], point.id, point.payload,
               point_index};
  }

  [[nodiscard]] bool key_less(const Key& lhs, const Key& rhs) const noexcept {
    if (lhs.coordinate < rhs.coordinate) return true;
    if (rhs.coordinate < lhs.coordinate) return false;
    if (lhs.id != rhs.id) return lhs.id < rhs.id;
    if (lhs.payload != rhs.payload) return lhs.payload < rhs.payload;
    return lhs.internal < rhs.internal;
  }

  [[nodiscard]] Relation relation_for_axis(
      const Node& node, const AxisRange<Coord>& range) const noexcept {
    if (range.lower) {
      const bool outside =
          range.lower_strict
              ? !(static_cast<bool>(*range.lower < node.max_coordinate))
              : node.max_coordinate < *range.lower;
      if (outside) return Relation::Outside;
    }
    if (range.upper) {
      const bool outside =
          range.upper_strict
              ? !(static_cast<bool>(node.min_coordinate < *range.upper))
              : *range.upper < node.min_coordinate;
      if (outside) return Relation::Outside;
    }

    bool inside = true;
    if (range.lower) {
      inside &= range.lower_strict
                    ? static_cast<bool>(*range.lower < node.min_coordinate)
                    : !static_cast<bool>(node.min_coordinate < *range.lower);
    }
    if (range.upper) {
      inside &= range.upper_strict
                    ? static_cast<bool>(node.max_coordinate < *range.upper)
                    : !static_cast<bool>(*range.upper < node.max_coordinate);
    }
    return inside ? Relation::Inside : Relation::Partial;
  }

  [[nodiscard]] bool point_satisfies(
      const Point& point,
      std::span<const AxisRange<Coord>> query) const noexcept {
    for (std::size_t dimension = 0; dimension < dimensions_; ++dimension) {
      const Coord value = point.coordinates[dimension];
      const AxisRange<Coord>& range = query[dimension];
      if (range.lower) {
        const bool accepted = range.lower_strict
                                  ? static_cast<bool>(*range.lower < value)
                                  : !static_cast<bool>(value < *range.lower);
        if (!accepted) return false;
      }
      if (range.upper) {
        const bool accepted = range.upper_strict
                                  ? static_cast<bool>(value < *range.upper)
                                  : !static_cast<bool>(*range.upper < value);
        if (!accepted) return false;
      }
    }
    return true;
  }

  [[nodiscard]] Index select_active_by_rank(const Node* node,
                                            UInt128 rank) const {
    if (!node || rank >= node->active_count) {
      throw std::logic_error("rank outside canonical range-tree block");
    }
    if (node->is_leaf()) {
      const Point& point = points_[node->point_index];
      if (!point.active || rank != 0) {
        throw std::logic_error("corrupted active range-tree leaf");
      }
      return point.payload;
    }

    const UInt128 left_count =
        node->left ? node->left->active_count : UInt128{0};
    if (rank < left_count) {
      return select_active_by_rank(node->left.get(), rank);
    }
    return select_active_by_rank(node->right.get(), rank - left_count);
  }

  std::size_t dimensions_{};
  bool dynamic_{};
  std::vector<Point> points_;
  std::vector<std::size_t> internal_by_point_;
  std::unique_ptr<RangeLevel> root_level_;
  std::size_t estimated_bytes_{};
  std::size_t node_count_{};
  std::size_t point_reference_count_{};
};

}  // namespace anchor
