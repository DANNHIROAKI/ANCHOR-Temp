#pragma once

#include "anchor/random.hpp"
#include "anchor/types.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace anchor::detail {

using Index = std::uint32_t;

template <Coordinate Coord>
class Geometry {
 public:
  explicit Geometry(std::shared_ptr<const BoxJoinInstance<Coord>> input)
      : input_(std::move(input)) {
    if (!input_) throw std::invalid_argument("null box-join input");
    dimensions_ = input_->dimensions();
    r_valid_ = input_->r.nonempty_indices();
    s_valid_ = input_->s.nonempty_indices();
  }

  [[nodiscard]] const BoxJoinInstance<Coord>& input() const noexcept {
    return *input_;
  }
  [[nodiscard]] std::size_t dimensions() const noexcept { return dimensions_; }
  [[nodiscard]] const std::vector<Index>& valid(Side side) const noexcept {
    return side == Side::R ? r_valid_ : s_valid_;
  }
  [[nodiscard]] std::size_t universe(Side side) const noexcept {
    return side == Side::R ? input_->r.size() : input_->s.size();
  }
  [[nodiscard]] Coord lower(Side side, Index index, std::size_t dimension) const {
    return boxes(side).lower(index, dimension);
  }
  [[nodiscard]] Coord upper(Side side, Index index, std::size_t dimension) const {
    return boxes(side).upper(index, dimension);
  }
  [[nodiscard]] std::uint64_t id(Side side, Index index) const {
    return boxes(side).id(index);
  }
  [[nodiscard]] const BoxSet<Coord>& boxes(Side side) const noexcept {
    return side == Side::R ? input_->r : input_->s;
  }
  [[nodiscard]] JoinPair pair(Index r, Index s) const {
    return JoinPair{id(Side::R, r), id(Side::S, s)};
  }
  [[nodiscard]] bool intersects(Index r, Index s) const {
    for (std::size_t j = 0; j < dimensions_; ++j) {
      if (!(lower(Side::R, r, j) < upper(Side::S, s, j) &&
            lower(Side::S, s, j) < upper(Side::R, r, j))) {
        return false;
      }
    }
    return true;
  }

 private:
  std::shared_ptr<const BoxJoinInstance<Coord>> input_;
  std::size_t dimensions_{};
  std::vector<Index> r_valid_;
  std::vector<Index> s_valid_;
};

struct SideOrders {
  std::vector<std::vector<Index>> by_lower;
  std::vector<Index> by_upper_d;

  explicit SideOrders(std::size_t dimensions = 0) : by_lower(dimensions) {}

  [[nodiscard]] std::size_t size_at(std::size_t dimension) const {
    return by_lower.at(dimension).size();
  }
};

struct LocalView {
  std::size_t level{};
  SideOrders a;
  SideOrders b;

  explicit LocalView(std::size_t dimensions = 0)
      : a(dimensions), b(dimensions) {}

  [[nodiscard]] bool empty() const {
    return a.by_lower[level].empty() || b.by_lower[level].empty();
  }
  [[nodiscard]] std::size_t size() const {
    return a.by_lower[level].size() + b.by_lower[level].size();
  }
};

template <Coordinate Coord>
std::vector<Index> stable_sort_indices(const Geometry<Coord>& geometry, Side side,
                                       std::span<const Index> members,
                                       std::size_t dimension, bool upper) {
  std::vector<Index> sorted(members.begin(), members.end());
  std::stable_sort(sorted.begin(), sorted.end(), [&](Index x, Index y) {
    const Coord a = upper ? geometry.upper(side, x, dimension)
                          : geometry.lower(side, x, dimension);
    const Coord b = upper ? geometry.upper(side, y, dimension)
                          : geometry.lower(side, y, dimension);
    if (a < b) return true;
    if (b < a) return false;
    const auto ia = geometry.id(side, x);
    const auto ib = geometry.id(side, y);
    if (ia != ib) return ia < ib;
    return x < y;
  });
  return sorted;
}

template <Coordinate Coord>
LocalView make_top_view(const Geometry<Coord>& geometry) {
  const std::size_t d = geometry.dimensions();
  LocalView view(d);
  view.level = 0;
  for (std::size_t j = 0; j < d; ++j) {
    view.a.by_lower[j] =
        stable_sort_indices(geometry, Side::R, geometry.valid(Side::R), j, false);
    view.b.by_lower[j] =
        stable_sort_indices(geometry, Side::S, geometry.valid(Side::S), j, false);
  }
  view.a.by_upper_d = stable_sort_indices(
      geometry, Side::R, geometry.valid(Side::R), d - 1, true);
  view.b.by_upper_d = stable_sort_indices(
      geometry, Side::S, geometry.valid(Side::S), d - 1, true);
  return view;
}

struct RankInterval {
  std::size_t lo{};
  std::size_t hi{};
  [[nodiscard]] bool empty() const noexcept { return lo == hi; }
};

using IntervalMap = std::unordered_map<Index, RankInterval>;

template <Coordinate Coord>
std::size_t lower_rank(const Geometry<Coord>& geometry, Side side,
                       const std::vector<Index>& sorted, std::size_t dimension,
                       Coord value) {
  return static_cast<std::size_t>(std::lower_bound(
             sorted.begin(), sorted.end(), value,
             [&](Index point, Coord boundary) {
               return geometry.lower(side, point, dimension) < boundary;
             }) -
         sorted.begin());
}

template <Coordinate Coord>
std::size_t upper_rank(const Geometry<Coord>& geometry, Side side,
                       const std::vector<Index>& sorted, std::size_t dimension,
                       Coord value) {
  return static_cast<std::size_t>(std::upper_bound(
             sorted.begin(), sorted.end(), value,
             [&](Coord boundary, Index point) {
               return boundary < geometry.lower(side, point, dimension);
             }) -
         sorted.begin());
}

template <Coordinate Coord>
std::pair<IntervalMap, IntervalMap> make_anchor_intervals(
    const Geometry<Coord>& geometry, const LocalView& view) {
  const std::size_t ell = view.level;
  IntervalMap xb;
  IntervalMap xa;
  xb.reserve(view.a.by_lower[ell].size());
  xa.reserve(view.b.by_lower[ell].size());
  for (Index a : view.a.by_lower[ell]) {
    xb.emplace(a, RankInterval{
                      lower_rank(geometry, Side::S, view.b.by_lower[ell], ell,
                                 geometry.lower(Side::R, a, ell)),
                      lower_rank(geometry, Side::S, view.b.by_lower[ell], ell,
                                 geometry.upper(Side::R, a, ell))});
  }
  for (Index b : view.b.by_lower[ell]) {
    xa.emplace(b, RankInterval{
                      upper_rank(geometry, Side::R, view.a.by_lower[ell], ell,
                                 geometry.lower(Side::S, b, ell)),
                      lower_rank(geometry, Side::R, view.a.by_lower[ell], ell,
                                 geometry.upper(Side::S, b, ell))});
  }
  return {std::move(xb), std::move(xa)};
}

enum class RouteOrientation : std::uint8_t { BTree = 0, ATree = 1 };

struct NodeKey {
  RouteOrientation orientation{RouteOrientation::BTree};
  std::size_t begin{};
  std::size_t end{};
  friend bool operator==(const NodeKey&, const NodeKey&) = default;
};

struct TreeLayout {
  static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();
  struct Node {
    NodeKey key;
    std::size_t parent{npos};
    std::size_t left{npos};
    std::size_t right{npos};
  };

  RouteOrientation orientation{RouteOrientation::BTree};
  std::size_t real_size{};
  std::size_t padded_size{};
  std::vector<Node> nodes;
};

inline TreeLayout make_tree_layout(std::size_t real_size,
                                   RouteOrientation orientation) {
  if (real_size == 0) throw std::invalid_argument("RouteTree requires a leaf");
  TreeLayout layout;
  layout.orientation = orientation;
  layout.real_size = real_size;
  layout.padded_size = 1;
  while (layout.padded_size < real_size) {
    if (layout.padded_size > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::overflow_error("RouteTree padding overflow");
    }
    layout.padded_size *= 2;
  }
  struct Pending {
    std::size_t begin, end, parent;
    bool is_left;
  };
  std::deque<Pending> queue;
  queue.push_back({0, layout.padded_size, TreeLayout::npos, false});
  while (!queue.empty()) {
    const Pending pending = queue.front();
    queue.pop_front();
    if (pending.begin >= real_size) continue;
    const std::size_t index = layout.nodes.size();
    layout.nodes.push_back(TreeLayout::Node{
        NodeKey{orientation, pending.begin, pending.end}, pending.parent,
        TreeLayout::npos, TreeLayout::npos});
    if (pending.parent != TreeLayout::npos) {
      if (pending.is_left)
        layout.nodes[pending.parent].left = index;
      else
        layout.nodes[pending.parent].right = index;
    }
    if (pending.end - pending.begin > 1) {
      const std::size_t middle = pending.begin + (pending.end - pending.begin) / 2;
      queue.push_back({pending.begin, middle, index, true});
      queue.push_back({middle, pending.end, index, false});
    }
  }
  return layout;
}

inline SideOrders restrict_orders(const SideOrders& input,
                                  const IntervalMap& intervals,
                                  std::size_t first_dimension) {
  SideOrders out(input.by_lower.size());
  for (std::size_t j = first_dimension; j < input.by_lower.size(); ++j) {
    for (Index point : input.by_lower[j]) {
      const auto it = intervals.find(point);
      if (it != intervals.end() && !it->second.empty()) {
        out.by_lower[j].push_back(point);
      }
    }
  }
  for (Index point : input.by_upper_d) {
    const auto it = intervals.find(point);
    if (it != intervals.end() && !it->second.empty()) {
      out.by_upper_d.push_back(point);
    }
  }
  return out;
}

template <class Predicate>
void partition_order(const std::vector<Index>& input, std::vector<Index>& yes,
                     std::vector<Index>& no, Predicate predicate) {
  yes.reserve(input.size());
  no.reserve(input.size());
  for (Index point : input) {
    (predicate(point) ? yes : no).push_back(point);
  }
}

template <Coordinate Coord, class Callback>
void route_tree(const Geometry<Coord>& geometry, const LocalView& view,
                RouteOrientation orientation, const IntervalMap& intervals,
                const TreeLayout& layout,
                std::span<const std::uint8_t> active,
                std::span<const std::uint8_t> requested, Callback&& callback) {
  if (view.level + 1 >= geometry.dimensions()) {
    throw std::invalid_argument("RouteTree called at terminal dimension");
  }
  if (active.size() != layout.nodes.size() ||
      requested.size() != layout.nodes.size()) {
    throw std::invalid_argument("RouteTree activity vector size mismatch");
  }
  if (active.empty() || !active[0]) return;
  const std::size_t next_level = view.level + 1;
  const bool b_tree = orientation == RouteOrientation::BTree;
  const SideOrders& source_parent = b_tree ? view.a : view.b;
  const SideOrders& lattice_parent = b_tree ? view.b : view.a;
  SideOrders source_root =
      restrict_orders(source_parent, intervals, next_level);
  SideOrders lattice_root(geometry.dimensions());
  for (std::size_t j = next_level; j < geometry.dimensions(); ++j) {
    lattice_root.by_lower[j] = lattice_parent.by_lower[j];
  }
  lattice_root.by_upper_d = lattice_parent.by_upper_d;

  std::unordered_map<Index, std::size_t> position;
  const auto& current_lattice = lattice_parent.by_lower[view.level];
  position.reserve(current_lattice.size());
  for (std::size_t i = 0; i < current_lattice.size(); ++i) {
    position.emplace(current_lattice[i], i);
  }

  struct State {
    std::size_t node{};
    SideOrders source;
    SideOrders lattice;
  };
  std::deque<State> queue;
  queue.push_back(State{0, std::move(source_root), std::move(lattice_root)});
  while (!queue.empty()) {
    State state = std::move(queue.front());
    queue.pop_front();
    const auto& meta = layout.nodes[state.node];
    const std::size_t real_end = std::min(meta.key.end, layout.real_size);

    auto stops_here = [&](Index point) {
      const RankInterval x = intervals.at(point);
      return x.lo <= meta.key.begin && real_end <= x.hi;
    };
    auto intersects_child = [&](Index point, const TreeLayout::Node& child) {
      const RankInterval x = intervals.at(point);
      const std::size_t child_end = std::min(child.key.end, layout.real_size);
      return std::max(x.lo, child.key.begin) < std::min(x.hi, child_end);
    };

    SideOrders stop(geometry.dimensions());
    SideOrders left_source(geometry.dimensions());
    SideOrders right_source(geometry.dimensions());
    SideOrders left_lattice(geometry.dimensions());
    SideOrders right_lattice(geometry.dimensions());
    const bool left_active = meta.left != TreeLayout::npos && active[meta.left];
    const bool right_active = meta.right != TreeLayout::npos && active[meta.right];

    auto split_source_vector = [&](const std::vector<Index>& input,
                                   std::vector<Index>& stop_out,
                                   std::vector<Index>& left_out,
                                   std::vector<Index>& right_out) {
      for (Index point : input) {
        if (stops_here(point)) {
          stop_out.push_back(point);
        } else {
          if (left_active && intersects_child(point, layout.nodes[meta.left]))
            left_out.push_back(point);
          if (right_active && intersects_child(point, layout.nodes[meta.right]))
            right_out.push_back(point);
        }
      }
    };
    auto split_lattice_vector = [&](const std::vector<Index>& input,
                                    std::vector<Index>& left_out,
                                    std::vector<Index>& right_out) {
      const std::size_t middle =
          meta.key.begin + (meta.key.end - meta.key.begin) / 2;
      for (Index point : input) {
        const bool goes_left = position.at(point) < middle;
        if (goes_left && left_active) left_out.push_back(point);
        if (!goes_left && right_active) right_out.push_back(point);
      }
    };

    for (std::size_t j = next_level; j < geometry.dimensions(); ++j) {
      split_source_vector(state.source.by_lower[j], stop.by_lower[j],
                          left_source.by_lower[j], right_source.by_lower[j]);
      split_lattice_vector(state.lattice.by_lower[j],
                           left_lattice.by_lower[j], right_lattice.by_lower[j]);
    }
    split_source_vector(state.source.by_upper_d, stop.by_upper_d,
                        left_source.by_upper_d, right_source.by_upper_d);
    split_lattice_vector(state.lattice.by_upper_d,
                         left_lattice.by_upper_d, right_lattice.by_upper_d);

    if (requested[state.node] && !stop.by_lower[next_level].empty() &&
        !state.lattice.by_lower[next_level].empty()) {
      LocalView child(geometry.dimensions());
      child.level = next_level;
      if (b_tree) {
        child.a = std::move(stop);
        child.b = std::move(state.lattice);
      } else {
        child.a = std::move(state.lattice);
        child.b = std::move(stop);
      }
      callback(state.node, meta.key, child);
    }
    if (left_active) {
      queue.push_back(
          State{meta.left, std::move(left_source), std::move(left_lattice)});
    }
    if (right_active) {
      queue.push_back(
          State{meta.right, std::move(right_source), std::move(right_lattice)});
    }
  }
}

struct BaseAtom {
  Side anchor_side{Side::R};
  Index anchor{};
  const std::vector<Index>* opposite{};
  std::size_t lo{};
  std::size_t hi{};
  UInt128 weight{};
};

template <Coordinate Coord>
std::pair<std::vector<BaseAtom>, UInt128> compute_base_atoms(
    const Geometry<Coord>& geometry, const LocalView& view) {
  const std::size_t d = geometry.dimensions();
  if (view.level != d - 1) {
    throw std::invalid_argument("base atoms require terminal LocalView");
  }
  const auto& al = view.a.by_lower[d - 1];
  const auto& bl = view.b.by_lower[d - 1];
  std::unordered_map<Index, RankInterval> a_bounds;
  std::unordered_map<Index, RankInterval> b_bounds;
  a_bounds.reserve(al.size());
  b_bounds.reserve(bl.size());

  std::size_t pointer = 0;
  for (Index a : al) {
    while (pointer < bl.size() &&
           geometry.lower(Side::S, bl[pointer], d - 1) <
               geometry.lower(Side::R, a, d - 1)) {
      ++pointer;
    }
    a_bounds[a].lo = pointer;
  }
  pointer = 0;
  for (Index a : view.a.by_upper_d) {
    while (pointer < bl.size() &&
           geometry.lower(Side::S, bl[pointer], d - 1) <
               geometry.upper(Side::R, a, d - 1)) {
      ++pointer;
    }
    a_bounds[a].hi = pointer;
  }
  pointer = 0;
  for (Index b : bl) {
    while (pointer < al.size() &&
           !(geometry.lower(Side::S, b, d - 1) <
             geometry.lower(Side::R, al[pointer], d - 1))) {
      ++pointer;  // L_A <= L_B
    }
    b_bounds[b].lo = pointer;
  }
  pointer = 0;
  for (Index b : view.b.by_upper_d) {
    while (pointer < al.size() &&
           geometry.lower(Side::R, al[pointer], d - 1) <
               geometry.upper(Side::S, b, d - 1)) {
      ++pointer;
    }
    b_bounds[b].hi = pointer;
  }

  std::vector<BaseAtom> atoms;
  atoms.reserve(al.size() + bl.size());
  UInt128 total = 0;
  for (Index a : al) {
    const RankInterval x = a_bounds.at(a);
    if (!x.empty()) {
      const UInt128 weight = x.hi - x.lo;
      atoms.push_back(BaseAtom{Side::R, a, &bl, x.lo, x.hi, weight});
      total = checked_add(total, weight, "base atom total overflow");
    }
  }
  for (Index b : bl) {
    const RankInterval x = b_bounds.at(b);
    if (!x.empty()) {
      const UInt128 weight = x.hi - x.lo;
      atoms.push_back(BaseAtom{Side::S, b, &al, x.lo, x.hi, weight});
      total = checked_add(total, weight, "base atom total overflow");
    }
  }
  return {std::move(atoms), total};
}

inline std::uint64_t child_path(std::uint64_t parent, std::size_t level,
                                const NodeKey& key) {
  std::uint64_t value = hash_combine(parent, level);
  value = hash_combine(value, static_cast<std::uint64_t>(key.orientation));
  value = hash_combine(value, key.begin);
  return hash_combine(value, key.end);
}

}  // namespace anchor::detail
