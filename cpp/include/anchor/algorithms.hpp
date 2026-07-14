#pragma once

#include "anchor/anchor_core.hpp"
#include "anchor/node_range_tree.hpp"
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
#include <string_view>
#include <utility>
#include <vector>

namespace anchor {

namespace detail {

template <Coordinate Coord>
void initialize_diagnostics(Diagnostics& diagnostics, std::string algorithm,
                            const Geometry<Coord>& geometry) {
  diagnostics.algorithm = std::move(algorithm);
  diagnostics.original_r = geometry.input().r.size();
  diagnostics.original_s = geometry.input().s.size();
  diagnostics.filtered_r = geometry.valid(Side::R).size();
  diagnostics.filtered_s = geometry.valid(Side::S).size();
}

struct TreeWeights {
  TreeLayout layout;
  std::vector<UInt128> weights;
};

struct NodeWeightResult {
  IntervalMap xb;
  IntervalMap xa;
  TreeWeights b_tree;
  TreeWeights a_tree;
  UInt128 total{};
};

inline std::vector<std::uint8_t> all_nodes(const TreeLayout& layout) {
  return std::vector<std::uint8_t>(layout.nodes.size(), 1);
}

inline void add_subtree_quotas(const TreeLayout& layout,
                               std::span<const std::size_t> direct,
                               std::vector<std::size_t>& subtree) {
  subtree.assign(direct.begin(), direct.end());
  for (std::size_t i = layout.nodes.size(); i-- > 0;) {
    const auto& node = layout.nodes[i];
    if (node.left != TreeLayout::npos) {
      if (subtree[i] > std::numeric_limits<std::size_t>::max() -
                           subtree[node.left]) {
        throw std::overflow_error("subtree quota overflow");
      }
      subtree[i] += subtree[node.left];
    }
    if (node.right != TreeLayout::npos) {
      if (subtree[i] > std::numeric_limits<std::size_t>::max() -
                           subtree[node.right]) {
        throw std::overflow_error("subtree quota overflow");
      }
      subtree[i] += subtree[node.right];
    }
  }
}

template <Coordinate Coord>
UInt128 anchor_count_recursive(const Geometry<Coord>& geometry,
                               const LocalView& view,
                               AlgorithmCounters* counters = nullptr) {
  if (counters) ++counters->recursive_count_calls;
  if (view.empty()) return 0;
  if (view.level + 1 == geometry.dimensions()) {
    return compute_base_atoms(geometry, view).second;
  }
  auto [xb, xa] = make_anchor_intervals(geometry, view);
  UInt128 total = 0;
  const auto process = [&](RouteOrientation orientation,
                           const IntervalMap& intervals,
                           std::size_t lattice_size) {
    const TreeLayout layout = make_tree_layout(lattice_size, orientation);
    const auto active = all_nodes(layout);
    const auto requested = active;
    route_tree(geometry, view, orientation, intervals, layout, active, requested,
               [&](std::size_t, const NodeKey&, const LocalView& child) {
                 total = checked_add(total,
                                     anchor_count_recursive(geometry, child,
                                                            counters),
                                     "ANCHOR count overflow");
               });
  };
  process(RouteOrientation::BTree, xb, view.b.by_lower[view.level].size());
  process(RouteOrientation::ATree, xa, view.a.by_lower[view.level].size());
  return total;
}

template <Coordinate Coord>
NodeWeightResult anchor_node_weights(const Geometry<Coord>& geometry,
                                     const LocalView& view,
                                     AlgorithmCounters* counters = nullptr) {
  auto [xb, xa] = make_anchor_intervals(geometry, view);
  NodeWeightResult result;
  result.xb = std::move(xb);
  result.xa = std::move(xa);
  result.b_tree.layout = make_tree_layout(
      view.b.by_lower[view.level].size(), RouteOrientation::BTree);
  result.a_tree.layout = make_tree_layout(
      view.a.by_lower[view.level].size(), RouteOrientation::ATree);
  result.b_tree.weights.assign(result.b_tree.layout.nodes.size(), 0);
  result.a_tree.weights.assign(result.a_tree.layout.nodes.size(), 0);

  auto fill = [&](TreeWeights& tree, const IntervalMap& intervals,
                  RouteOrientation orientation) {
    const auto active = all_nodes(tree.layout);
    const auto requested = active;
    route_tree(geometry, view, orientation, intervals, tree.layout, active,
               requested,
               [&](std::size_t node, const NodeKey&, const LocalView& child) {
                 tree.weights[node] =
                     anchor_count_recursive(geometry, child, counters);
                 result.total = checked_add(result.total, tree.weights[node],
                                            "node weight total overflow");
               });
  };
  fill(result.b_tree, result.xb, RouteOrientation::BTree);
  fill(result.a_tree, result.xa, RouteOrientation::ATree);
  return result;
}

template <Coordinate Coord>
void base_sample(const Geometry<Coord>& geometry, const LocalView& view,
                 std::span<JoinPair> output, const RngPool& pool,
                 std::uint64_t path) {
  if (output.empty()) return;
  auto [atoms, total] = compute_base_atoms(geometry, view);
  if (total == 0) throw std::logic_error("positive base quota on empty join");
  std::vector<UInt128> weights;
  weights.reserve(atoms.size());
  for (const BaseAtom& atom : atoms) weights.push_back(atom.weight);
  IntegerAlias alias(weights);
  auto atom_rng = pool.stream("AS/base-atom", path);
  auto rank_rng = pool.stream("AS/base-rank", path);
  for (JoinPair& pair : output) {
    const BaseAtom& atom = atoms[alias.sample(atom_rng)];
    const std::size_t offset = uniform_index(rank_rng, atom.hi - atom.lo);
    const Index opposite = atom.opposite->at(atom.lo + offset);
    pair = atom.anchor_side == Side::R ? geometry.pair(atom.anchor, opposite)
                                       : geometry.pair(opposite, atom.anchor);
  }
}

template <Coordinate Coord>
bool anchor_sample_recursive(const Geometry<Coord>& geometry,
                             const LocalView& view,
                             std::span<JoinPair> output, const RngPool& pool,
                             std::uint64_t path, UInt128* observed_total,
                             AlgorithmCounters* counters,
                             std::size_t live_workspace_bytes = 0) {
  if (output.empty()) return true;
  if (view.empty()) return false;
  if (view.level + 1 == geometry.dimensions()) {
    if (counters) {
      const std::size_t local =
          view.size() * (sizeof(Index) * 4 + sizeof(RankInterval));
      counters->max_live_workspace_bytes = std::max(
          counters->max_live_workspace_bytes, live_workspace_bytes + local);
    }
    const UInt128 total = compute_base_atoms(geometry, view).second;
    if (observed_total) *observed_total = total;
    if (total == 0) return false;
    base_sample(geometry, view, output, pool, path);
    return true;
  }
  NodeWeightResult nodes = anchor_node_weights(geometry, view, counters);
  if (observed_total) *observed_total = nodes.total;
  if (nodes.total == 0) return false;

  std::vector<UInt128> combined;
  combined.reserve(nodes.b_tree.weights.size() + nodes.a_tree.weights.size());
  combined.insert(combined.end(), nodes.b_tree.weights.begin(),
                  nodes.b_tree.weights.end());
  combined.insert(combined.end(), nodes.a_tree.weights.begin(),
                  nodes.a_tree.weights.end());
  auto quota_rng = pool.stream("AS/quotas", path);
  const auto quotas = draw_quotas(output.size(), combined, quota_rng);
  const std::size_t split = nodes.b_tree.weights.size();
  const std::span<const std::size_t> b_direct(quotas.data(), split);
  const std::span<const std::size_t> a_direct(quotas.data() + split,
                                             quotas.size() - split);
  std::vector<std::size_t> b_subtree;
  std::vector<std::size_t> a_subtree;
  add_subtree_quotas(nodes.b_tree.layout, b_direct, b_subtree);
  add_subtree_quotas(nodes.a_tree.layout, a_direct, a_subtree);
  const std::size_t local_workspace =
      view.size() * geometry.dimensions() * sizeof(Index) * 2 +
      combined.size() * (sizeof(UInt128) + sizeof(std::size_t) * 2 + 2) +
      (nodes.xb.size() + nodes.xa.size()) *
          (sizeof(Index) + sizeof(RankInterval));
  if (counters) {
    counters->max_live_workspace_bytes =
        std::max(counters->max_live_workspace_bytes,
                 live_workspace_bytes + local_workspace);
    counters->positive_quota_nodes += static_cast<std::size_t>(
        std::count_if(quotas.begin(), quotas.end(),
                      [](std::size_t quota) { return quota != 0; }));
    counters->active_route_nodes += static_cast<std::size_t>(
        std::count_if(b_subtree.begin(), b_subtree.end(),
                      [](std::size_t quota) { return quota != 0; }));
    counters->active_route_nodes += static_cast<std::size_t>(
        std::count_if(a_subtree.begin(), a_subtree.end(),
                      [](std::size_t quota) { return quota != 0; }));
  }

  std::size_t cursor = 0;
  auto replay = [&](const TreeWeights& tree, const IntervalMap& intervals,
                    RouteOrientation orientation,
                    std::span<const std::size_t> direct,
                    const std::vector<std::size_t>& subtree) {
    std::vector<std::uint8_t> active(tree.layout.nodes.size());
    std::vector<std::uint8_t> requested(tree.layout.nodes.size());
    for (std::size_t i = 0; i < tree.layout.nodes.size(); ++i) {
      active[i] = subtree[i] != 0;
      requested[i] = direct[i] != 0;
    }
    route_tree(geometry, view, orientation, intervals, tree.layout, active,
               requested,
               [&](std::size_t node, const NodeKey& key,
                   const LocalView& child) {
                 const std::size_t quota = direct[node];
                 if (cursor + quota > output.size()) {
                   throw std::logic_error("ANCHOR quota exceeds output slice");
                 }
                 const std::uint64_t next_path =
                     child_path(path, view.level, key);
                 if (!anchor_sample_recursive(
                         geometry, child, output.subspan(cursor, quota), pool,
                         next_path, nullptr, counters,
                         live_workspace_bytes + local_workspace)) {
                   throw std::logic_error("positive node quota on empty join");
                 }
                 cursor += quota;
               });
  };
  replay(nodes.b_tree, nodes.xb, RouteOrientation::BTree, b_direct, b_subtree);
  replay(nodes.a_tree, nodes.xa, RouteOrientation::ATree, a_direct, a_subtree);
  if (cursor != output.size()) {
    throw std::logic_error("ANCHOR quotas did not fill output slice");
  }
  auto shuffle_rng = pool.stream("AS/shuffle", path);
  fisher_yates(output, shuffle_rng);
  return true;
}

}  // namespace detail

template <Coordinate Coord>
class AnchorCompiled final : public ISampler<Coord> {
 public:
  explicit AnchorCompiled(std::shared_ptr<const BoxJoinInstance<Coord>> input,
                          SamplerOptions options = {})
      : geometry_(std::move(input)), options_(options) {
    detail::initialize_diagnostics(diagnostics_, "AC", geometry_);
    {
      StageTimer timer(diagnostics_.timings, "top_view");
      top_ = detail::make_top_view(geometry_);
    }
    {
      StageTimer timer(diagnostics_.timings, "compile");
      if (!top_.empty()) compile(top_);
    }
    // AC queries only need compiled atoms and their owned terminal arrays.
    top_ = detail::LocalView{};
    if (total_ != 0) {
      StageTimer timer(diagnostics_.timings, "global_alias");
      std::vector<UInt128> weights;
      weights.reserve(atoms_.size());
      for (const Atom& atom : atoms_) weights.push_back(atom.weight);
      alias_ = std::make_unique<IntegerAlias>(weights);
    }
    diagnostics_.join_count = total_;
    diagnostics_.persistent_bytes_estimate =
        atoms_.capacity() * sizeof(Atom) + persisted_coordinate_indices_ *
                                                sizeof(detail::Index);
    diagnostics_.counters.positive_atom_count = atoms_.size();
    diagnostics_.counters.persistent_terminal_array_items =
        persisted_coordinate_indices_;
    diagnostics_.counters.alias_label_count = atoms_.size();
  }
  AnchorCompiled(const AnchorCompiled&) = delete;
  AnchorCompiled& operator=(const AnchorCompiled&) = delete;
  AnchorCompiled(AnchorCompiled&&) = delete;
  AnchorCompiled& operator=(AnchorCompiled&&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override { return "AC"; }
  UInt128 count() override { return total_; }

  SampleStatus sample(std::span<JoinPair> output,
                      std::uint64_t query_id = 0) override {
    if (output.empty()) return SampleStatus::Ok;
    if (total_ == 0) return SampleStatus::EmptyInstance;
    StageTimer timer(diagnostics_.timings, "sample");
    RngPool pool(options_.seed, query_id);
    auto atom_rng = pool.stream("AC/atom");
    auto rank_rng = pool.stream("AC/rank");
    for (JoinPair& pair : output) {
      const Atom& atom = atoms_.at(alias_->sample(atom_rng));
      const std::size_t offset = uniform_index(rank_rng, atom.hi - atom.lo);
      const detail::Index opposite = atom.opposite->at(atom.lo + offset);
      pair = atom.anchor_side == Side::R
                 ? geometry_.pair(atom.anchor, opposite)
                 : geometry_.pair(opposite, atom.anchor);
    }
    return SampleStatus::Ok;
  }

  [[nodiscard]] const Diagnostics& diagnostics() const noexcept override {
    return diagnostics_;
  }

 private:
  struct Atom {
    Side anchor_side{Side::R};
    detail::Index anchor{};
    std::shared_ptr<const std::vector<detail::Index>> opposite;
    std::size_t lo{};
    std::size_t hi{};
    UInt128 weight{};
  };

  void compile(const detail::LocalView& view) {
    if (view.empty()) return;
    if (view.level + 1 == geometry_.dimensions()) {
      ++diagnostics_.counters.terminal_instance_count;
      auto [base, ignored_total] = detail::compute_base_atoms(geometry_, view);
      (void)ignored_total;
      std::shared_ptr<const std::vector<detail::Index>> persistent_a;
      std::shared_ptr<const std::vector<detail::Index>> persistent_b;
      for (const detail::BaseAtom& atom : base) {
        std::shared_ptr<const std::vector<detail::Index>> opposite;
        if (atom.anchor_side == Side::R) {
          if (!persistent_b) {
            persistent_b = std::make_shared<const std::vector<detail::Index>>(
                view.b.by_lower.back());
            persisted_coordinate_indices_ += persistent_b->size();
          }
          opposite = persistent_b;
        } else {
          if (!persistent_a) {
            persistent_a = std::make_shared<const std::vector<detail::Index>>(
                view.a.by_lower.back());
            persisted_coordinate_indices_ += persistent_a->size();
          }
          opposite = persistent_a;
        }
        atoms_.push_back(Atom{atom.anchor_side, atom.anchor, std::move(opposite),
                              atom.lo, atom.hi, atom.weight});
        total_ = checked_add(total_, atom.weight, "AC total overflow");
      }
      return;
    }
    auto [xb, xa] = detail::make_anchor_intervals(geometry_, view);
    auto process = [&](detail::RouteOrientation orientation,
                       const detail::IntervalMap& intervals,
                       std::size_t lattice_size) {
      const auto layout = detail::make_tree_layout(lattice_size, orientation);
      const auto active = detail::all_nodes(layout);
      const auto requested = active;
      detail::route_tree(geometry_, view, orientation, intervals, layout, active,
                         requested,
                         [&](std::size_t, const detail::NodeKey&,
                             const detail::LocalView& child) { compile(child); });
    };
    process(detail::RouteOrientation::BTree, xb,
            view.b.by_lower[view.level].size());
    process(detail::RouteOrientation::ATree, xa,
            view.a.by_lower[view.level].size());
  }

  detail::Geometry<Coord> geometry_;
  SamplerOptions options_;
  detail::LocalView top_;
  std::vector<Atom> atoms_;
  std::unique_ptr<IntegerAlias> alias_;
  UInt128 total_{};
  std::size_t persisted_coordinate_indices_{};
  Diagnostics diagnostics_;
};

template <Coordinate Coord>
class AnchorStreaming final : public ISampler<Coord> {
 public:
  explicit AnchorStreaming(std::shared_ptr<const BoxJoinInstance<Coord>> input,
                           SamplerOptions options = {})
      : geometry_(std::move(input)), options_(options) {
    detail::initialize_diagnostics(diagnostics_, "AS", geometry_);
    StageTimer timer(diagnostics_.timings, "top_view");
    top_ = detail::make_top_view(geometry_);
    std::size_t items = top_.a.by_upper_d.size() + top_.b.by_upper_d.size();
    for (const auto& order : top_.a.by_lower) items += order.size();
    for (const auto& order : top_.b.by_lower) items += order.size();
    diagnostics_.persistent_bytes_estimate = items * sizeof(detail::Index);
  }
  AnchorStreaming(const AnchorStreaming&) = delete;
  AnchorStreaming& operator=(const AnchorStreaming&) = delete;
  AnchorStreaming(AnchorStreaming&&) = delete;
  AnchorStreaming& operator=(AnchorStreaming&&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override { return "AS"; }

  UInt128 count() override {
    StageTimer timer(diagnostics_.timings, "count");
    diagnostics_.counters.recursive_count_calls = 0;
    diagnostics_.counters.positive_quota_nodes = 0;
    diagnostics_.counters.active_route_nodes = 0;
    diagnostics_.counters.max_live_workspace_bytes = 0;
    diagnostics_.join_count = top_.empty()
                                  ? 0
                                  : detail::anchor_count_recursive(
                                        geometry_, top_, &diagnostics_.counters);
    return diagnostics_.join_count;
  }

  SampleStatus sample(std::span<JoinPair> output,
                      std::uint64_t query_id = 0) override {
    if (output.empty()) return SampleStatus::Ok;
    if (top_.empty()) {
      diagnostics_.join_count = 0;
      return SampleStatus::EmptyInstance;
    }
    StageTimer timer(diagnostics_.timings, "sample");
    diagnostics_.counters.recursive_count_calls = 0;
    diagnostics_.counters.positive_quota_nodes = 0;
    diagnostics_.counters.active_route_nodes = 0;
    diagnostics_.counters.max_live_workspace_bytes = 0;
    RngPool pool(options_.seed, query_id);
    UInt128 observed = 0;
    const bool ok = detail::anchor_sample_recursive(
        geometry_, top_, output, pool, 0x41532f746f70ULL, &observed,
        &diagnostics_.counters);
    diagnostics_.join_count = observed;
    return ok ? SampleStatus::Ok : SampleStatus::EmptyInstance;
  }

  [[nodiscard]] const Diagnostics& diagnostics() const noexcept override {
    return diagnostics_;
  }

 private:
  detail::Geometry<Coord> geometry_;
  SamplerOptions options_;
  detail::LocalView top_;
  Diagnostics diagnostics_;
};

}  // namespace anchor

namespace anchor::detail {

class DenseActiveSet {
 public:
  explicit DenseActiveSet(std::size_t universe)
      : position_(universe, npos), active_(universe, false) {}

  void insert(Index point) {
    check(point);
    if (active_[point]) throw std::logic_error("duplicate dense insertion");
    position_[point] = values_.size();
    values_.push_back(point);
    active_[point] = true;
  }

  void erase(Index point) {
    check(point);
    if (!active_[point]) throw std::logic_error("dense deletion of inactive point");
    const std::size_t position = position_[point];
    const Index last = values_.back();
    values_[position] = last;
    position_[last] = position;
    values_.pop_back();
    position_[point] = npos;
    active_[point] = false;
  }

  [[nodiscard]] std::size_t size() const noexcept { return values_.size(); }
  [[nodiscard]] bool empty() const noexcept { return values_.empty(); }
  [[nodiscard]] bool all_inactive() const noexcept { return values_.empty(); }

  void sample(std::span<Index> output, DeterministicRng& rng) const {
    if (!output.empty() && values_.empty()) {
      throw std::logic_error("sampling empty active dense set");
    }
    for (Index& point : output) point = values_[uniform_index(rng, values_.size())];
  }

 private:
  void check(Index point) const {
    if (point >= position_.size()) throw std::out_of_range("dense active index");
  }
  static constexpr std::size_t npos = std::numeric_limits<std::size_t>::max();
  std::vector<Index> values_;
  std::vector<std::size_t> position_;
  std::vector<bool> active_;
};

}  // namespace anchor::detail

namespace anchor {

template <Coordinate Coord>
class LiftedRangeTree;

template <Coordinate Coord>
class SweepRangeTree final : public ISampler<Coord> {
 public:
  explicit SweepRangeTree(
      std::shared_ptr<const BoxJoinInstance<Coord>> input,
      SamplerOptions options = {})
      : geometry_(std::move(input)),
        options_(options),
        dense_r_(geometry_.universe(Side::R)),
        dense_s_(geometry_.universe(Side::S)) {
    detail::initialize_diagnostics(diagnostics_, "SweepRT", geometry_);
    {
      StageTimer timer(diagnostics_.timings, "event_sort");
      build_events();
    }
    if (!geometry_.valid(Side::R).empty() &&
        !geometry_.valid(Side::S).empty()) {
      if (geometry_.dimensions() > 1) {
        StageTimer timer(diagnostics_.timings, "active_skeleton_build");
        tree_r_ = build_tree(Side::R);
        tree_s_ = build_tree(Side::S);
      }
      {
        StageTimer timer(diagnostics_.timings, "pass1_count");
        pass1();
      }
      if (total_ != 0 && options_.build_sampling_index) {
        StageTimer timer(diagnostics_.timings, "event_alias");
        event_alias_ = std::make_unique<IntegerAlias>(weights_);
      }
    } else {
      weights_.assign(start_count_, 0);
    }
    diagnostics_.join_count = total_;
    diagnostics_.persistent_bytes_estimate =
        events_.capacity() * sizeof(Event) + weights_.capacity() * sizeof(UInt128) +
        (tree_r_ ? tree_r_->estimated_bytes() : 0) +
        (tree_s_ ? tree_s_->estimated_bytes() : 0);
    diagnostics_.counters.nonzero_event_blocks = static_cast<std::size_t>(
        std::count_if(weights_.begin(), weights_.end(),
                      [](UInt128 weight) { return weight != 0; }));
    diagnostics_.counters.range_tree_nodes =
        (tree_r_ ? tree_r_->skeleton_nodes() : 0) +
        (tree_s_ ? tree_s_->skeleton_nodes() : 0);
    diagnostics_.counters.range_tree_point_references =
        (tree_r_ ? tree_r_->point_references() : 0) +
        (tree_s_ ? tree_s_->point_references() : 0);
  }
  SweepRangeTree(const SweepRangeTree&) = delete;
  SweepRangeTree& operator=(const SweepRangeTree&) = delete;
  SweepRangeTree(SweepRangeTree&&) = delete;
  SweepRangeTree& operator=(SweepRangeTree&&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "SweepRT";
  }
  UInt128 count() override {
    diagnostics_.counters.selected_event_blocks = 0;
    return total_;
  }

  SampleStatus sample(std::span<JoinPair> output,
                      std::uint64_t query_id = 0) override {
    if (output.empty()) return SampleStatus::Ok;
    if (total_ == 0) return SampleStatus::EmptyInstance;
    if (!event_alias_) {
      throw std::logic_error("SweepRT sampling index was not built");
    }
    diagnostics_.counters.selected_event_blocks = 0;
    ensure_all_inactive();
    StageTimer timer(diagnostics_.timings, "pass2_sample");
    RngPool pool(options_.seed, query_id);
    auto event_rng = pool.stream("SweepRT/event");
    std::vector<std::vector<std::size_t>> positions(weights_.size());
    for (std::size_t i = 0; i < output.size(); ++i) {
      positions[event_alias_->sample(event_rng)].push_back(i);
    }
    diagnostics_.counters.selected_event_blocks = static_cast<std::size_t>(
        std::count_if(positions.begin(), positions.end(),
                      [](const auto& group) { return !group.empty(); }));

    for (const Event& event : events_) {
      if (event.end) {
        set_active(event.side, event.point, false);
        continue;
      }
      const auto& destinations = positions[event.start_index];
      if (!destinations.empty()) {
        std::vector<detail::Index> partners(destinations.size());
        const std::uint64_t domain = hash_combine(
            static_cast<std::uint64_t>(event.start_index),
            geometry_.id(event.side, event.point));
        if (geometry_.dimensions() == 1) {
          auto rank_rng = pool.stream("SweepRT/dense-rank", domain);
          dense(other(event.side)).sample(partners, rank_rng);
        } else {
          auto block_rng = pool.stream("SweepRT/block", domain);
          auto rank_rng = pool.stream("SweepRT/rank", domain);
          const auto query = make_query(event.side, event.point);
          if (!tree(other(event.side)).sample(query, partners, block_rng,
                                               rank_rng)) {
            throw std::logic_error("positive SweepRT event has empty query");
          }
        }
        for (std::size_t j = 0; j < partners.size(); ++j) {
          output[destinations[j]] =
              event.side == Side::R
                  ? geometry_.pair(event.point, partners[j])
                  : geometry_.pair(partners[j], event.point);
        }
      }
      set_active(event.side, event.point, true);
    }
    ensure_all_inactive();
    return SampleStatus::Ok;
  }

  [[nodiscard]] const Diagnostics& diagnostics() const noexcept override {
    return diagnostics_;
  }

 private:
  struct Event {
    Coord coordinate{};
    bool end{};
    Side side{Side::R};
    detail::Index point{};
    std::size_t start_index{std::numeric_limits<std::size_t>::max()};
  };

  static Side other(Side side) noexcept {
    return side == Side::R ? Side::S : Side::R;
  }

  void build_events() {
    events_.reserve(2 * (geometry_.valid(Side::R).size() +
                         geometry_.valid(Side::S).size()));
    for (Side side : {Side::R, Side::S}) {
      for (detail::Index point : geometry_.valid(side)) {
        events_.push_back(Event{geometry_.lower(side, point, 0), false, side,
                                point, std::numeric_limits<std::size_t>::max()});
        events_.push_back(Event{geometry_.upper(side, point, 0), true, side,
                                point, std::numeric_limits<std::size_t>::max()});
      }
    }
    std::stable_sort(events_.begin(), events_.end(), [&](const Event& a,
                                                         const Event& b) {
      if (a.coordinate < b.coordinate) return true;
      if (b.coordinate < a.coordinate) return false;
      if (a.end != b.end) return a.end;  // END before START.
      if (a.side != b.side) return a.side < b.side;
      const auto ia = geometry_.id(a.side, a.point);
      const auto ib = geometry_.id(b.side, b.point);
      if (ia != ib) return ia < ib;
      return a.point < b.point;
    });
    start_count_ = 0;
    for (Event& event : events_) {
      if (!event.end) event.start_index = start_count_++;
    }
  }

  std::unique_ptr<NodeRangeTree<Coord>> build_tree(Side side) {
    const std::size_t h = geometry_.dimensions() - 1;
    const std::size_t dimensions = 2 * h;
    auto coordinate = [this, side, h](detail::Index point, std::size_t axis) {
      return axis < h ? geometry_.lower(side, point, axis + 1)
                      : geometry_.upper(side, point, axis - h + 1);
    };
    auto id = [this, side](detail::Index point) {
      return geometry_.id(side, point);
    };
    return std::make_unique<NodeRangeTree<Coord>>(
        dimensions, geometry_.valid(side), std::move(coordinate), std::move(id),
        true, geometry_.universe(side));
  }

  [[nodiscard]] std::vector<AxisRange<Coord>> make_query(
      Side side, detail::Index point) const {
    const std::size_t h = geometry_.dimensions() - 1;
    std::vector<AxisRange<Coord>> query(2 * h);
    for (std::size_t j = 0; j < h; ++j) {
      query[j] = AxisRange<Coord>::less_than(
          geometry_.upper(side, point, j + 1));
      query[h + j] = AxisRange<Coord>::greater_than(
          geometry_.lower(side, point, j + 1));
    }
    return query;
  }

  NodeRangeTree<Coord>& tree(Side side) {
    auto& pointer = side == Side::R ? tree_r_ : tree_s_;
    if (!pointer) throw std::logic_error("missing active range tree");
    return *pointer;
  }
  const NodeRangeTree<Coord>& tree(Side side) const {
    const auto& pointer = side == Side::R ? tree_r_ : tree_s_;
    if (!pointer) throw std::logic_error("missing active range tree");
    return *pointer;
  }
  detail::DenseActiveSet& dense(Side side) {
    return side == Side::R ? dense_r_ : dense_s_;
  }
  const detail::DenseActiveSet& dense(Side side) const {
    return side == Side::R ? dense_r_ : dense_s_;
  }

  void set_active(Side side, detail::Index point, bool value) {
    if (geometry_.dimensions() == 1) {
      if (value)
        dense(side).insert(point);
      else
        dense(side).erase(point);
    } else {
      tree(side).set_active(point, value);
    }
  }

  [[nodiscard]] UInt128 partner_count(Side start_side,
                                      detail::Index point) const {
    if (geometry_.dimensions() == 1) {
      return static_cast<UInt128>(dense(other(start_side)).size());
    }
    const auto query = make_query(start_side, point);
    return tree(other(start_side)).count(query);
  }

  void pass1() {
    ensure_all_inactive();
    weights_.assign(start_count_, 0);
    for (const Event& event : events_) {
      if (event.end) {
        set_active(event.side, event.point, false);
      } else {
        const UInt128 weight = partner_count(event.side, event.point);
        weights_[event.start_index] = weight;
        total_ = checked_add(total_, weight, "SweepRT total overflow");
        set_active(event.side, event.point, true);
      }
    }
    ensure_all_inactive();
  }

  void ensure_all_inactive() const {
    const bool inactive =
        geometry_.dimensions() == 1
            ? dense_r_.all_inactive() && dense_s_.all_inactive()
            : (!tree_r_ || (tree_r_->all_inactive() && tree_s_->all_inactive()));
    if (!inactive) throw std::logic_error("SweepRT pass did not end all-inactive");
  }

  detail::Geometry<Coord> geometry_;
  SamplerOptions options_;
  std::vector<Event> events_;
  std::size_t start_count_{};
  std::vector<UInt128> weights_;
  std::unique_ptr<IntegerAlias> event_alias_;
  std::unique_ptr<NodeRangeTree<Coord>> tree_r_;
  std::unique_ptr<NodeRangeTree<Coord>> tree_s_;
  detail::DenseActiveSet dense_r_;
  detail::DenseActiveSet dense_s_;
  UInt128 total_{};
  Diagnostics diagnostics_;
};

template <Coordinate Coord>
using AC = AnchorCompiled<Coord>;
template <Coordinate Coord>
using AS = AnchorStreaming<Coord>;
template <Coordinate Coord>
using LiftedRT = LiftedRangeTree<Coord>;
template <Coordinate Coord>
using SweepRT = SweepRangeTree<Coord>;

template <Coordinate Coord>
std::unique_ptr<ISampler<Coord>> make_sampler(
    AlgorithmKind kind, std::shared_ptr<const BoxJoinInstance<Coord>> input,
    SamplerOptions options = {}) {
  switch (kind) {
    case AlgorithmKind::AC:
      return std::make_unique<AC<Coord>>(std::move(input), options);
    case AlgorithmKind::AS:
      return std::make_unique<AS<Coord>>(std::move(input), options);
    case AlgorithmKind::LiftedRT:
      return std::make_unique<LiftedRT<Coord>>(std::move(input), options);
    case AlgorithmKind::SweepRT:
      return std::make_unique<SweepRT<Coord>>(std::move(input), options);
  }
  throw std::invalid_argument("unknown algorithm kind");
}

}  // namespace anchor

namespace anchor {

template <Coordinate Coord>
class LiftedRangeTree final : public ISampler<Coord> {
 public:
  explicit LiftedRangeTree(
      std::shared_ptr<const BoxJoinInstance<Coord>> input,
      SamplerOptions options = {})
      : geometry_(std::move(input)), options_(options) {
    detail::initialize_diagnostics(diagnostics_, "LiftedRT", geometry_);
    const std::size_t dimensions = 2 * geometry_.dimensions();
    if (!geometry_.valid(Side::R).empty() &&
        !geometry_.valid(Side::S).empty()) {
      {
        StageTimer timer(diagnostics_.timings, "range_tree_build");
        auto coordinate = [this](detail::Index point, std::size_t axis) {
          const std::size_t source_dimension = axis / 2;
          return axis % 2 == 0
                     ? geometry_.lower(Side::S, point, source_dimension)
                     : geometry_.upper(Side::S, point, source_dimension);
        };
        auto id = [this](detail::Index point) {
          return geometry_.id(Side::S, point);
        };
        tree_ = std::make_unique<NodeRangeTree<Coord>>(
            dimensions, geometry_.valid(Side::S), std::move(coordinate),
            std::move(id), false, geometry_.universe(Side::S));
      }
      {
        StageTimer timer(diagnostics_.timings, "degree_count");
        for (detail::Index a : geometry_.valid(Side::R)) {
          ++preprocess_query_count_;
          const auto query = make_query(a);
          const UInt128 degree = tree_->count(query);
          if (degree != 0) {
            anchors_.push_back(a);
            degrees_.push_back(degree);
            total_ = checked_add(total_, degree, "LiftedRT total overflow");
          }
        }
      }
      if (total_ != 0 && options_.build_sampling_index) {
        StageTimer timer(diagnostics_.timings, "outer_alias");
        outer_alias_ = std::make_unique<IntegerAlias>(degrees_);
      }
    }
    diagnostics_.join_count = total_;
    diagnostics_.persistent_bytes_estimate =
        (tree_ ? tree_->estimated_bytes() : 0) +
        anchors_.capacity() * sizeof(detail::Index) +
        degrees_.capacity() * sizeof(UInt128);
    diagnostics_.counters.positive_degree_left_objects = anchors_.size();
    diagnostics_.counters.canonical_block_queries = preprocess_query_count_;
    diagnostics_.counters.range_tree_nodes = tree_ ? tree_->skeleton_nodes() : 0;
    diagnostics_.counters.range_tree_point_references =
        tree_ ? tree_->point_references() : 0;
  }
  LiftedRangeTree(const LiftedRangeTree&) = delete;
  LiftedRangeTree& operator=(const LiftedRangeTree&) = delete;
  LiftedRangeTree(LiftedRangeTree&&) = delete;
  LiftedRangeTree& operator=(LiftedRangeTree&&) = delete;

  [[nodiscard]] std::string_view name() const noexcept override {
    return "LiftedRT";
  }
  UInt128 count() override {
    diagnostics_.counters.selected_left_objects = 0;
    diagnostics_.counters.canonical_block_queries = preprocess_query_count_;
    return total_;
  }

  SampleStatus sample(std::span<JoinPair> output,
                      std::uint64_t query_id = 0) override {
    if (output.empty()) return SampleStatus::Ok;
    if (total_ == 0) return SampleStatus::EmptyInstance;
    if (!outer_alias_) {
      throw std::logic_error("LiftedRT sampling index was not built");
    }
    diagnostics_.counters.selected_left_objects = 0;
    diagnostics_.counters.canonical_block_queries = preprocess_query_count_;
    StageTimer timer(diagnostics_.timings, "sample");
    RngPool pool(options_.seed, query_id);
    auto outer_rng = pool.stream("LiftedRT/outer");
    std::vector<std::vector<std::size_t>> groups(anchors_.size());
    for (std::size_t position = 0; position < output.size(); ++position) {
      groups[outer_alias_->sample(outer_rng)].push_back(position);
    }
    for (std::size_t label = 0; label < groups.size(); ++label) {
      if (groups[label].empty()) continue;
      ++diagnostics_.counters.selected_left_objects;
      ++diagnostics_.counters.canonical_block_queries;
      const detail::Index a = anchors_[label];
      std::vector<detail::Index> partners(groups[label].size());
      const std::uint64_t domain = detail::child_path(
          0x4c69667465645254ULL, 0,
          detail::NodeKey{detail::RouteOrientation::BTree, label,
                          geometry_.id(Side::R, a)});
      auto block_rng = pool.stream("LiftedRT/block", domain);
      auto rank_rng = pool.stream("LiftedRT/rank", domain);
      const auto query = make_query(a);
      if (!tree_->sample(query, partners, block_rng, rank_rng)) {
        throw std::logic_error("positive LiftedRT degree has empty query");
      }
      for (std::size_t i = 0; i < partners.size(); ++i) {
        output[groups[label][i]] = geometry_.pair(a, partners[i]);
      }
    }
    return SampleStatus::Ok;
  }

  [[nodiscard]] const Diagnostics& diagnostics() const noexcept override {
    return diagnostics_;
  }

 private:
  [[nodiscard]] std::vector<AxisRange<Coord>> make_query(
      detail::Index a) const {
    std::vector<AxisRange<Coord>> query(2 * geometry_.dimensions());
    for (std::size_t j = 0; j < geometry_.dimensions(); ++j) {
      // L_S < U_R is a strict prefix. U_S > L_R is a strict suffix.
      // Keeping U in its native order avoids both negation and MIN overflow.
      query[2 * j] =
          AxisRange<Coord>::less_than(geometry_.upper(Side::R, a, j));
      query[2 * j + 1] =
          AxisRange<Coord>::greater_than(geometry_.lower(Side::R, a, j));
    }
    return query;
  }

  detail::Geometry<Coord> geometry_;
  SamplerOptions options_;
  std::unique_ptr<NodeRangeTree<Coord>> tree_;
  std::vector<detail::Index> anchors_;
  std::vector<UInt128> degrees_;
  std::unique_ptr<IntegerAlias> outer_alias_;
  std::size_t preprocess_query_count_{};
  UInt128 total_{};
  Diagnostics diagnostics_;
};

}  // namespace anchor
