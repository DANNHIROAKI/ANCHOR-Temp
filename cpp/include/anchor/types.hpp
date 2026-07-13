#pragma once

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <compare>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace anchor {

using UInt128 = unsigned __int128;

inline std::string to_string(UInt128 value) {
  if (value == 0) return "0";
  std::string out;
  while (value != 0) {
    out.push_back(static_cast<char>('0' + value % 10));
    value /= 10;
  }
  std::reverse(out.begin(), out.end());
  return out;
}

inline UInt128 checked_add(UInt128 a, UInt128 b,
                           std::string_view what = "UInt128 addition") {
  constexpr UInt128 kMax = ~UInt128{0};
  if (b > kMax - a) throw std::overflow_error(std::string(what));
  return a + b;
}

inline UInt128 checked_mul(UInt128 a, UInt128 b,
                           std::string_view what = "UInt128 multiplication") {
  constexpr UInt128 kMax = ~UInt128{0};
  if (a != 0 && b > kMax / a) throw std::overflow_error(std::string(what));
  return a * b;
}

inline std::size_t checked_size(UInt128 value,
                                std::string_view what = "size conversion") {
  if (value > static_cast<UInt128>(std::numeric_limits<std::size_t>::max())) {
    throw std::overflow_error(std::string(what));
  }
  return static_cast<std::size_t>(value);
}

enum class Side : std::uint8_t { R = 0, S = 1 };

template <class Coord>
concept Coordinate = std::same_as<Coord, double> ||
                     std::same_as<Coord, std::int64_t>;

template <Coordinate Coord>
class BoxSet {
 public:
  using coordinate_type = Coord;

  explicit BoxSet(std::size_t dimensions = 0) : dimensions_(dimensions) {}

  BoxSet(std::size_t dimensions, std::vector<std::uint64_t> ids,
         std::vector<Coord> lower, std::vector<Coord> upper)
      : dimensions_(dimensions),
        ids_(std::move(ids)),
        lower_(std::move(lower)),
        upper_(std::move(upper)) {
    validate_shape();
    validate_coordinates();
  }

  void reserve(std::size_t boxes) {
    if (dimensions_ != 0 &&
        boxes > std::numeric_limits<std::size_t>::max() / dimensions_) {
      throw std::overflow_error("BoxSet reserve coordinate count overflow");
    }
    ids_.reserve(boxes);
    lower_.reserve(boxes * dimensions_);
    upper_.reserve(boxes * dimensions_);
  }

  void add(std::uint64_t id, std::span<const Coord> lower,
           std::span<const Coord> upper) {
    if (lower.empty()) throw std::invalid_argument("box dimension must be positive");
    if (dimensions_ == 0) dimensions_ = lower.size();
    if (lower.size() != dimensions_ || upper.size() != dimensions_) {
      throw std::invalid_argument("box dimensionality mismatch");
    }
    for (std::size_t j = 0; j < dimensions_; ++j) {
      validate_coordinate(lower[j]);
      validate_coordinate(upper[j]);
    }
    ids_.push_back(id);
    lower_.insert(lower_.end(), lower.begin(), lower.end());
    upper_.insert(upper_.end(), upper.begin(), upper.end());
  }

  [[nodiscard]] std::size_t dimensions() const noexcept { return dimensions_; }
  [[nodiscard]] std::size_t size() const noexcept { return ids_.size(); }
  [[nodiscard]] bool empty() const noexcept { return ids_.empty(); }

  [[nodiscard]] std::uint64_t id(std::size_t i) const { return ids_.at(i); }
  [[nodiscard]] Coord lower(std::size_t i, std::size_t j) const {
    return lower_.at(i * dimensions_ + j);
  }
  [[nodiscard]] Coord upper(std::size_t i, std::size_t j) const {
    return upper_.at(i * dimensions_ + j);
  }
  [[nodiscard]] std::span<const Coord> lower_row(std::size_t i) const {
    return std::span<const Coord>(lower_).subspan(i * dimensions_, dimensions_);
  }
  [[nodiscard]] std::span<const Coord> upper_row(std::size_t i) const {
    return std::span<const Coord>(upper_).subspan(i * dimensions_, dimensions_);
  }
  [[nodiscard]] const std::vector<std::uint64_t>& ids() const noexcept {
    return ids_;
  }
  [[nodiscard]] const std::vector<Coord>& flat_lower() const noexcept {
    return lower_;
  }
  [[nodiscard]] const std::vector<Coord>& flat_upper() const noexcept {
    return upper_;
  }

  [[nodiscard]] bool nonempty(std::size_t i) const {
    for (std::size_t j = 0; j < dimensions_; ++j) {
      if (!(lower(i, j) < upper(i, j))) return false;
    }
    return true;
  }

  [[nodiscard]] std::vector<std::uint32_t> nonempty_indices() const {
    if (size() > std::numeric_limits<std::uint32_t>::max()) {
      throw std::overflow_error("BoxSet exceeds 32-bit internal index space");
    }
    std::vector<std::uint32_t> out;
    out.reserve(size());
    for (std::size_t i = 0; i < size(); ++i) {
      if (nonempty(i)) out.push_back(static_cast<std::uint32_t>(i));
    }
    return out;
  }

  void validate_unique_ids() const {
    std::unordered_set<std::uint64_t> seen;
    seen.reserve(ids_.size());
    for (auto id_value : ids_) {
      if (!seen.insert(id_value).second) {
        throw std::invalid_argument("object ids must be unique within each side");
      }
    }
  }

 private:
  static void validate_coordinate(Coord value) {
    if constexpr (std::same_as<Coord, double>) {
      if (!std::isfinite(value)) {
        throw std::invalid_argument("double endpoints must be finite");
      }
    }
  }

  void validate_shape() const {
    if (dimensions_ == 0 && !ids_.empty()) {
      throw std::invalid_argument("nonempty BoxSet must have positive dimension");
    }
    if (dimensions_ != 0 &&
        ids_.size() > std::numeric_limits<std::size_t>::max() / dimensions_) {
      throw std::overflow_error("BoxSet coordinate count overflow");
    }
    const std::size_t expected = ids_.size() * dimensions_;
    if (lower_.size() != expected || upper_.size() != expected) {
      throw std::invalid_argument("flat coordinate array has invalid size");
    }
  }

  void validate_coordinates() const {
    for (Coord x : lower_) validate_coordinate(x);
    for (Coord x : upper_) validate_coordinate(x);
  }

  std::size_t dimensions_{};
  std::vector<std::uint64_t> ids_;
  std::vector<Coord> lower_;
  std::vector<Coord> upper_;
};

template <Coordinate Coord>
struct BoxJoinInstance {
  BoxSet<Coord> r;
  BoxSet<Coord> s;

  BoxJoinInstance(BoxSet<Coord> r_boxes, BoxSet<Coord> s_boxes)
      : r(std::move(r_boxes)), s(std::move(s_boxes)) {
    if (r.dimensions() == 0 || s.dimensions() == 0 ||
        r.dimensions() != s.dimensions()) {
      throw std::invalid_argument("both sides must have the same positive dimension");
    }
    r.validate_unique_ids();
    s.validate_unique_ids();
  }

  [[nodiscard]] std::size_t dimensions() const noexcept {
    return r.dimensions();
  }
};

template <Coordinate Coord>
[[nodiscard]] bool boxes_intersect(const BoxSet<Coord>& r, std::size_t ri,
                                   const BoxSet<Coord>& s, std::size_t si) {
  if (r.dimensions() != s.dimensions()) {
    throw std::invalid_argument("intersection dimensionality mismatch");
  }
  if (!r.nonempty(ri) || !s.nonempty(si)) return false;
  for (std::size_t j = 0; j < r.dimensions(); ++j) {
    if (!(r.lower(ri, j) < s.upper(si, j) &&
          s.lower(si, j) < r.upper(ri, j))) {
      return false;
    }
  }
  return true;
}

struct JoinPair {
  std::uint64_t r{};
  std::uint64_t s{};
  friend bool operator==(const JoinPair&, const JoinPair&) = default;
  friend auto operator<=>(const JoinPair&, const JoinPair&) = default;
};

struct SampleBatch {
  bool empty_instance{false};
  std::vector<JoinPair> pairs;
};

enum class SampleStatus { Ok, EmptyInstance };

struct StageTiming {
  std::string stage;
  double seconds{};
};

// Stable superset of algorithm-specific counters used by the experiment raw
// schema. Irrelevant fields remain zero for a given algorithm.
struct AlgorithmCounters {
  // AC
  std::size_t terminal_instance_count{};
  std::size_t positive_atom_count{};
  std::size_t persistent_terminal_array_items{};
  std::size_t alias_label_count{};
  // AS
  std::size_t recursive_count_calls{};
  std::size_t positive_quota_nodes{};
  std::size_t active_route_nodes{};
  std::size_t max_live_workspace_bytes{};
  // SweepRT
  std::size_t nonzero_event_blocks{};
  std::size_t selected_event_blocks{};
  std::size_t skeleton_nodes{};
  std::size_t fenwick_items{};
  // LiftedRT
  std::size_t positive_degree_left_objects{};
  std::size_t selected_left_objects{};
  std::size_t canonical_block_queries{};
  std::size_t range_tree_items{};
};

struct Diagnostics {
  std::string algorithm;
  std::size_t original_r{};
  std::size_t original_s{};
  std::size_t filtered_r{};
  std::size_t filtered_s{};
  UInt128 join_count{};
  std::size_t persistent_bytes_estimate{};
  AlgorithmCounters counters;
  std::vector<StageTiming> timings;
};

class StageTimer {
 public:
  StageTimer(std::vector<StageTiming>& sink, std::string stage)
      : sink_(sink), stage_(std::move(stage)), start_(Clock::now()) {}
  StageTimer(const StageTimer&) = delete;
  StageTimer& operator=(const StageTimer&) = delete;
  ~StageTimer() {
    const auto elapsed = std::chrono::duration<double>(Clock::now() - start_).count();
    sink_.push_back(StageTiming{stage_, elapsed});
  }

 private:
  using Clock = std::chrono::steady_clock;
  std::vector<StageTiming>& sink_;
  std::string stage_;
  Clock::time_point start_;
};

struct SamplerOptions {
  std::array<std::uint64_t, 4> seed{
      0x243f6a8885a308d3ULL, 0x13198a2e03707344ULL,
      0xa4093822299f31d0ULL, 0x082efa98ec4e6c89ULL};
  // Count-only benchmark tasks for range-tree baselines suppress the final
  // outer/event sampling alias. Normal construction keeps it enabled.
  bool build_sampling_index{true};
};

enum class AlgorithmKind { AC, AS, LiftedRT, SweepRT };

template <Coordinate Coord>
class ISampler {
 public:
  virtual ~ISampler() = default;
  [[nodiscard]] virtual std::string_view name() const noexcept = 0;
  virtual UInt128 count() = 0;
  virtual SampleStatus sample(std::span<JoinPair> output,
                              std::uint64_t query_id = 0) = 0;
  SampleBatch sample_allocating(std::size_t t, std::uint64_t query_id = 0) {
    SampleBatch batch;
    batch.pairs.resize(t);
    batch.empty_instance =
        sample(batch.pairs, query_id) == SampleStatus::EmptyInstance;
    if (batch.empty_instance) batch.pairs.clear();
    return batch;
  }
  [[nodiscard]] virtual const Diagnostics& diagnostics() const noexcept = 0;
};

}  // namespace anchor
