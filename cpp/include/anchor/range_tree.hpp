#pragma once

#include "anchor/fenwick.hpp"
#include "anchor/random.hpp"
#include "anchor/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace anchor {

template <Coordinate Coord>
struct AxisRange {
  std::optional<Coord> lower;
  std::optional<Coord> upper;
  bool lower_strict{true};
  bool upper_strict{true};

  static AxisRange less_than(Coord value) {
    AxisRange q;
    q.upper = value;
    q.upper_strict = true;
    return q;
  }
  static AxisRange greater_than(Coord value) {
    AxisRange q;
    q.lower = value;
    q.lower_strict = true;
    return q;
  }
};

// A conventional layered range tree. At dimensions 0..D-2 it stores a
// balanced segment/search tree and an associated next-dimensional structure
// at every node. The last-dimensional associated structures are sorted arrays.
// Dynamic mode adds a Fenwick active-bit array to every terminal array and an
// occurrence list for every point; coordinates themselves remain static.
template <Coordinate Coord>
class OrthogonalRangeTree {
 public:
  using Index = std::uint32_t;
  using CoordinateAccessor = std::function<Coord(Index, std::size_t)>;
  using IdAccessor = std::function<std::uint64_t(Index)>;

 private:
  struct Terminal {
    std::vector<Index> values;
    Fenwick active;
  };
  struct Level;
  struct Node {
    std::size_t begin{};
    std::size_t end{};
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
    std::unique_ptr<Level> next;
    std::unique_ptr<Terminal> terminal;
  };
  struct Level {
    std::size_t dimension{};
    std::vector<Index> sorted;
    std::unique_ptr<Node> root;
  };
  struct Occurrence {
    Terminal* terminal{};
    std::size_t position{};
  };

 public:
  struct Block {
    const Terminal* terminal{};
    std::size_t begin{};
    std::size_t end{};
    UInt128 weight{};
  };

  OrthogonalRangeTree(std::size_t dimensions, std::vector<Index> points,
                      CoordinateAccessor coordinate, IdAccessor id,
                      bool dynamic, std::size_t object_universe)
      : dimensions_(dimensions),
        coordinate_(std::move(coordinate)),
        id_(std::move(id)),
        dynamic_(dynamic),
        occurrences_(dynamic ? object_universe : 0),
        point_active_(dynamic ? object_universe : 0, false) {
    if (dimensions_ == 0) {
      throw std::invalid_argument("zero-dimensional range tree is a dense set");
    }
    for (Index p : points) {
      if (dynamic_ && p >= object_universe) {
        throw std::invalid_argument("point index exceeds dynamic universe");
      }
    }
    if (dimensions_ == 1) {
      root_terminal_ = build_terminal(std::move(points), 0);
    } else {
      root_level_ = build_level(0, std::move(points));
    }
    if (dynamic_) {
      estimated_bytes_ += occurrences_.capacity() * sizeof(occurrences_[0]) +
                          (point_active_.capacity() + 7) / 8;
      for (const auto& list : occurrences_) {
        estimated_bytes_ += list.capacity() * sizeof(Occurrence);
      }
    }
  }

  OrthogonalRangeTree(const OrthogonalRangeTree&) = delete;
  OrthogonalRangeTree& operator=(const OrthogonalRangeTree&) = delete;
  OrthogonalRangeTree(OrthogonalRangeTree&&) = delete;
  OrthogonalRangeTree& operator=(OrthogonalRangeTree&&) = delete;

  [[nodiscard]] std::size_t dimensions() const noexcept { return dimensions_; }
  [[nodiscard]] bool dynamic() const noexcept { return dynamic_; }
  [[nodiscard]] std::size_t estimated_bytes() const noexcept {
    return estimated_bytes_;
  }
  [[nodiscard]] std::size_t skeleton_nodes() const noexcept {
    return skeleton_nodes_;
  }
  [[nodiscard]] std::size_t terminal_items() const noexcept {
    return terminal_items_;
  }

  [[nodiscard]] std::vector<Block> canonical_blocks(
      std::span<const AxisRange<Coord>> query) const {
    if (query.size() != dimensions_) {
      throw std::invalid_argument("range query dimensionality mismatch");
    }
    std::vector<Block> out;
    if (dimensions_ == 1) {
      if (root_terminal_) add_terminal_block(*root_terminal_, 0, query, out);
    } else if (root_level_) {
      query_level(*root_level_, query, out);
    }
    return out;
  }

  [[nodiscard]] UInt128 count(std::span<const AxisRange<Coord>> query) const {
    UInt128 total = 0;
    for (const Block& block : canonical_blocks(query)) {
      total = checked_add(total, block.weight, "range count overflow");
    }
    return total;
  }

  // Returns false exactly when the current query result is empty. Sampling is
  // with replacement and does not alter active bits.
  bool sample(std::span<const AxisRange<Coord>> query, std::span<Index> output,
              DeterministicRng& block_rng, DeterministicRng& rank_rng) const {
    if (output.empty()) return true;
    auto blocks = canonical_blocks(query);
    std::vector<Block> positive;
    std::vector<UInt128> weights;
    positive.reserve(blocks.size());
    weights.reserve(blocks.size());
    for (const Block& b : blocks) {
      if (b.weight != 0) {
        positive.push_back(b);
        weights.push_back(b.weight);
      }
    }
    if (positive.empty()) return false;
    IntegerAlias alias(weights);
    for (Index& destination : output) {
      const Block& block = positive[alias.sample(block_rng)];
      if (dynamic_) {
        const std::uint64_t before =
            block.terminal->active.prefix_sum(block.begin);
        const UInt128 rho = uniform_below(rank_rng, block.weight);
        if (rho > std::numeric_limits<std::uint64_t>::max() - before) {
          throw std::overflow_error("active rank overflow");
        }
        const auto target = before + static_cast<std::uint64_t>(rho) + 1;
        const std::size_t position = block.terminal->active.select(target);
        if (position < block.begin || position >= block.end) {
          throw std::logic_error("Fenwick block select escaped query interval");
        }
        destination = block.terminal->values[position];
      } else {
        const std::size_t offset = checked_size(uniform_below(rank_rng, block.weight));
        destination = block.terminal->values[block.begin + offset];
      }
    }
    return true;
  }

  void set_active(Index point, bool value) {
    if (!dynamic_) throw std::logic_error("static range tree has no active bits");
    if (point >= point_active_.size()) throw std::out_of_range("active point index");
    if (occurrences_[point].empty()) {
      throw std::invalid_argument("point is not part of the active skeleton");
    }
    if (point_active_[point] == value) {
      throw std::logic_error(value ? "duplicate range-tree insertion"
                                   : "range-tree deletion of inactive point");
    }
    const int delta = value ? 1 : -1;
    for (const Occurrence& occurrence : occurrences_[point]) {
      occurrence.terminal->active.add(occurrence.position, delta);
    }
    point_active_[point] = value;
  }

  [[nodiscard]] bool is_active(Index point) const {
    if (!dynamic_ || point >= point_active_.size()) return false;
    return point_active_[point];
  }

  [[nodiscard]] bool all_inactive() const {
    return std::none_of(point_active_.begin(), point_active_.end(),
                        [](bool value) { return value; });
  }

 private:
  [[nodiscard]] bool point_less(Index a, Index b, std::size_t dimension) const {
    const Coord x = coordinate_(a, dimension);
    const Coord y = coordinate_(b, dimension);
    if (x < y) return true;
    if (y < x) return false;
    const auto ia = id_(a);
    const auto ib = id_(b);
    if (ia != ib) return ia < ib;
    return a < b;
  }

  void sort_points(std::vector<Index>& points, std::size_t dimension) const {
    std::stable_sort(points.begin(), points.end(), [&](Index a, Index b) {
      return point_less(a, b, dimension);
    });
  }

  std::unique_ptr<Terminal> build_terminal(std::vector<Index> points,
                                           std::size_t dimension,
                                           bool already_sorted = false) {
    if (!already_sorted) sort_points(points, dimension);
    auto terminal = std::make_unique<Terminal>();
    terminal->values = std::move(points);
    terminal_items_ += terminal->values.size();
    ++skeleton_nodes_;  // terminal associated array / D=1 root
    if (dynamic_) terminal->active = Fenwick(terminal->values.size());
    estimated_bytes_ += sizeof(Terminal) +
                        terminal->values.capacity() * sizeof(Index) +
                        (dynamic_ ? (terminal->values.size() + 1) *
                                        sizeof(std::uint64_t)
                                  : 0);
    if (dynamic_) {
      Terminal* stable = terminal.get();
      terminals_.push_back(stable);
      for (std::size_t i = 0; i < stable->values.size(); ++i) {
        occurrences_[stable->values[i]].push_back(Occurrence{stable, i});
      }
    }
    return terminal;
  }

  std::unique_ptr<Level> build_level(std::size_t dimension,
                                     std::vector<Index> points) {
    auto level = std::make_unique<Level>();
    level->dimension = dimension;
    sort_points(points, dimension);
    level->sorted = std::move(points);
    estimated_bytes_ +=
        sizeof(Level) + level->sorted.capacity() * sizeof(Index);
    if (!level->sorted.empty()) {
      level->root = build_node(*level, 0, level->sorted.size());
    }
    return level;
  }

  std::unique_ptr<Node> build_node(const Level& owner, std::size_t begin,
                                   std::size_t end) {
    auto node = std::make_unique<Node>();
    ++skeleton_nodes_;
    node->begin = begin;
    node->end = end;
    estimated_bytes_ += sizeof(Node);
    if (end - begin > 1) {
      const std::size_t middle = begin + (end - begin) / 2;
      node->left = build_node(owner, begin, middle);
      node->right = build_node(owner, middle, end);
    }
    if (owner.dimension + 1 == dimensions_ - 1) {
      // Standard bottom-up associated-array construction. Sorting every node
      // independently would add an erroneous extra logarithmic build factor.
      std::vector<Index> associated;
      associated.reserve(end - begin);
      if (node->left && node->right) {
        const auto& left = node->left->terminal->values;
        const auto& right = node->right->terminal->values;
        std::merge(left.begin(), left.end(), right.begin(), right.end(),
                   std::back_inserter(associated), [&](Index a, Index b) {
                     return point_less(a, b, owner.dimension + 1);
                   });
      } else {
        associated.push_back(owner.sorted[begin]);
      }
      node->terminal =
          build_terminal(std::move(associated), owner.dimension + 1, true);
    } else {
      std::vector<Index> subset(
          owner.sorted.begin() + static_cast<std::ptrdiff_t>(begin),
          owner.sorted.begin() + static_cast<std::ptrdiff_t>(end));
      node->next = build_level(owner.dimension + 1, std::move(subset));
    }
    return node;
  }

  [[nodiscard]] std::pair<std::size_t, std::size_t> rank_interval(
      const std::vector<Index>& sorted, std::size_t dimension,
      const AxisRange<Coord>& range) const {
    auto lower_bound_coord = [&](Coord x) {
      return static_cast<std::size_t>(std::lower_bound(
                 sorted.begin(), sorted.end(), x,
                 [&](Index point, Coord value) {
                   return coordinate_(point, dimension) < value;
                 }) -
             sorted.begin());
    };
    auto upper_bound_coord = [&](Coord x) {
      return static_cast<std::size_t>(std::upper_bound(
                 sorted.begin(), sorted.end(), x,
                 [&](Coord value, Index point) {
                   return value < coordinate_(point, dimension);
                 }) -
             sorted.begin());
    };
    std::size_t begin = 0;
    std::size_t end = sorted.size();
    if (range.lower) {
      begin = range.lower_strict ? upper_bound_coord(*range.lower)
                                 : lower_bound_coord(*range.lower);
    }
    if (range.upper) {
      end = range.upper_strict ? lower_bound_coord(*range.upper)
                               : upper_bound_coord(*range.upper);
    }
    if (begin > end) begin = end;
    return {begin, end};
  }

  void add_terminal_block(const Terminal& terminal, std::size_t dimension,
                          std::span<const AxisRange<Coord>> query,
                          std::vector<Block>& out) const {
    const auto [begin, end] =
        rank_interval(terminal.values, dimension, query[dimension]);
    if (begin == end) return;
    const UInt128 weight = dynamic_
                               ? terminal.active.range_sum(begin, end)
                               : static_cast<UInt128>(end - begin);
    out.push_back(Block{&terminal, begin, end, weight});
  }

  void emit_associated(const Node& node, std::size_t next_dimension,
                       std::span<const AxisRange<Coord>> query,
                       std::vector<Block>& out) const {
    if (node.terminal) {
      add_terminal_block(*node.terminal, next_dimension, query, out);
    } else if (node.next) {
      query_level(*node.next, query, out);
    }
  }

  void cover_nodes(const Node& node, std::size_t query_begin,
                   std::size_t query_end, std::size_t next_dimension,
                   std::span<const AxisRange<Coord>> query,
                   std::vector<Block>& out) const {
    if (query_end <= node.begin || node.end <= query_begin) return;
    if (query_begin <= node.begin && node.end <= query_end) {
      emit_associated(node, next_dimension, query, out);
      return;
    }
    if (node.left) cover_nodes(*node.left, query_begin, query_end,
                               next_dimension, query, out);
    if (node.right) cover_nodes(*node.right, query_begin, query_end,
                                next_dimension, query, out);
  }

  void query_level(const Level& level,
                   std::span<const AxisRange<Coord>> query,
                   std::vector<Block>& out) const {
    if (!level.root) return;
    const auto [begin, end] =
        rank_interval(level.sorted, level.dimension, query[level.dimension]);
    if (begin == end) return;
    cover_nodes(*level.root, begin, end, level.dimension + 1, query, out);
  }

  std::size_t dimensions_{};
  CoordinateAccessor coordinate_;
  IdAccessor id_;
  bool dynamic_{};
  std::unique_ptr<Level> root_level_;
  std::unique_ptr<Terminal> root_terminal_;
  std::vector<std::vector<Occurrence>> occurrences_;
  std::vector<bool> point_active_;
  std::vector<Terminal*> terminals_;
  std::size_t estimated_bytes_{};
  std::size_t skeleton_nodes_{};
  std::size_t terminal_items_{};
};

}  // namespace anchor
