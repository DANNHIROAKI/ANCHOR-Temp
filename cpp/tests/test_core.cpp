#include "anchor/algorithms.hpp"
#include "anchor/fenwick.hpp"
#include "anchor/random.hpp"
#include "anchor/range_tree.hpp"
#include "anchor/types.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition)) {                                                         \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK failed: "          \
                << #condition << '\n';                                          \
      ++failures;                                                               \
    }                                                                           \
  } while (false)

template <class Exception, class Callable>
void check_throws(Callable&& callable) {
  bool thrown = false;
  try {
    callable();
  } catch (const Exception&) {
    thrown = true;
  }
  CHECK(thrown);
}

void test_uint128_and_random() {
  using anchor::UInt128;
  constexpr UInt128 max = ~UInt128{0};
  CHECK(anchor::to_string(static_cast<UInt128>(1234567890123456789ULL)) ==
        "1234567890123456789");
  check_throws<std::overflow_error>([&] { (void)anchor::checked_add(max, 1); });
  check_throws<std::overflow_error>([&] { (void)anchor::checked_mul(max, 2); });

  const std::array<std::uint64_t, 4> seed{1, 2, 3, 4};
  anchor::RngPool pool_a(seed, 7);
  anchor::RngPool pool_b(seed, 7);
  auto a = pool_a.stream("uniform", 9);
  auto b = pool_b.stream("uniform", 9);
  for (int i = 0; i < 1000; ++i) {
    const UInt128 x = anchor::uniform_integer(a, 11, 28);
    const UInt128 y = anchor::uniform_integer(b, 11, 28);
    CHECK(x == y);
    CHECK(x >= 11 && x < 28);
  }
  auto wide = pool_a.stream("wide");
  const UInt128 wide_bound = (static_cast<UInt128>(1) << 100U) + 17;
  for (int i = 0; i < 100; ++i) {
    CHECK(anchor::uniform_below(wide, wide_bound) < wide_bound);
  }
  auto different = pool_a.stream("different", 9);
  auto original = pool_a.stream("uniform", 9);
  CHECK(different.next_u64() != original.next_u64());

  const std::vector<UInt128> weights{0, 1, 3, 0, 6};
  anchor::IntegerAlias alias(weights);
  auto alias_rng = pool_a.stream("alias");
  std::array<std::size_t, 5> counts{};
  for (int i = 0; i < 10000; ++i) ++counts[alias.sample(alias_rng)];
  CHECK(counts[0] == 0 && counts[3] == 0);
  CHECK(counts[4] > counts[2] && counts[2] > counts[1]);
}

void test_fenwick() {
  anchor::Fenwick f(8);
  f.add(1, 1);
  f.add(3, 1);
  f.add(7, 1);
  CHECK(f.prefix_sum(8) == 3);
  CHECK(f.range_sum(2, 7) == 1);
  CHECK(f.select(1) == 1);
  CHECK(f.select(2) == 3);
  CHECK(f.select(3) == 7);
  check_throws<std::logic_error>([&] { f.add(3, 1); });
  f.add(3, -1);
  check_throws<std::logic_error>([&] { f.add(3, -1); });
}

void test_boxset_validation() {
  anchor::BoxSet<double> boxes(1);
  const std::array<double, 1> lo{0.0};
  const std::array<double, 1> hi{1.0};
  boxes.add(1, lo, hi);
  CHECK(boxes.nonempty(0));
  const std::array<double, 1> nan{std::numeric_limits<double>::infinity()};
  check_throws<std::invalid_argument>([&] { boxes.add(2, nan, hi); });
}

void test_range_tree() {
  using Tree = anchor::OrthogonalRangeTree<std::int64_t>;
  using Index = Tree::Index;
  const std::vector<std::array<std::int64_t, 3>> points{
      {0, 0, 0}, {1, 4, 2}, {2, 2, 4}, {3, 3, 1}, {4, 1, 3}};
  std::vector<Index> ids{0, 1, 2, 3, 4};
  auto coordinate = [&](Index point, std::size_t dimension) {
    return points[point][dimension];
  };
  auto id = [](Index point) { return static_cast<std::uint64_t>(point); };
  Tree tree(3, ids, coordinate, id, false, points.size());
  std::vector<anchor::AxisRange<std::int64_t>> query(3);
  query[0] = anchor::AxisRange<std::int64_t>::greater_than(0);
  query[1] = anchor::AxisRange<std::int64_t>::less_than(4);
  query[2].lower = 1;
  query[2].upper = 4;
  query[2].lower_strict = false;
  query[2].upper_strict = false;
  CHECK(tree.count(query) == 3);  // points 2,3,4

  Tree dynamic_tree(3, ids, coordinate, id, true, points.size());
  dynamic_tree.set_active(1, true);
  dynamic_tree.set_active(2, true);
  dynamic_tree.set_active(4, true);
  CHECK(dynamic_tree.count(query) == 2);  // points 2,4
  std::array<Index, 200> sampled{};
  const std::array<std::uint64_t, 4> seed{8, 6, 7, 5};
  anchor::RngPool pool(seed);
  auto block_rng = pool.stream("blocks");
  auto rank_rng = pool.stream("ranks");
  CHECK(dynamic_tree.sample(query, sampled, block_rng, rank_rng));
  for (Index point : sampled) CHECK(point == 2 || point == 4);
  dynamic_tree.set_active(1, false);
  dynamic_tree.set_active(2, false);
  dynamic_tree.set_active(4, false);
  CHECK(dynamic_tree.all_inactive());
}

void test_range_tree_random_strict_boundaries() {
  using Tree = anchor::OrthogonalRangeTree<std::int64_t>;
  using Index = Tree::Index;
  constexpr std::size_t d = 4;
  std::vector<std::array<std::int64_t, d>> points;
  for (std::int64_t i = 0; i < 8; ++i) {
    points.push_back({i % 3, (i * 2) % 5, (i + 1) % 3, (i * i) % 4});
  }
  std::vector<Index> ids(points.size());
  for (std::size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<Index>(i);
  auto coordinate = [&](Index point, std::size_t dimension) {
    return points[point][dimension];
  };
  auto id = [](Index point) { return static_cast<std::uint64_t>(point); };
  Tree statik(d, ids, coordinate, id, false, points.size());
  Tree dynamic(d, ids, coordinate, id, true, points.size());
  for (Index point : ids) {
    if (point % 2 == 0) dynamic.set_active(point, true);
  }
  const std::array<std::uint64_t, 4> seed{91, 82, 73, 64};
  anchor::RngPool pool(seed);
  auto rng = pool.stream("strict-query-generation");
  for (int iteration = 0; iteration < 250; ++iteration) {
    std::vector<anchor::AxisRange<std::int64_t>> query(d);
    for (std::size_t axis = 0; axis < d; ++axis) {
      const auto mode = static_cast<unsigned>(anchor::uniform_below(rng, 3));
      const auto x = static_cast<std::int64_t>(anchor::uniform_below(rng, 7)) - 1;
      const auto y = static_cast<std::int64_t>(anchor::uniform_below(rng, 7)) - 1;
      if (mode == 0) {
        query[axis] = anchor::AxisRange<std::int64_t>::less_than(x);
      } else if (mode == 1) {
        query[axis] = anchor::AxisRange<std::int64_t>::greater_than(x);
      } else {
        query[axis].lower = std::min(x, y);
        query[axis].upper = std::max(x, y);
        query[axis].lower_strict = true;
        query[axis].upper_strict = true;
      }
    }
    std::size_t expected_static = 0;
    std::size_t expected_dynamic = 0;
    for (Index point : ids) {
      bool inside = true;
      for (std::size_t axis = 0; axis < d; ++axis) {
        const auto value = points[point][axis];
        if (query[axis].lower) {
          inside &= query[axis].lower_strict
                        ? *query[axis].lower < value
                        : ! (value < *query[axis].lower);
        }
        if (query[axis].upper) {
          inside &= query[axis].upper_strict
                        ? value < *query[axis].upper
                        : ! (*query[axis].upper < value);
        }
      }
      if (inside) {
        ++expected_static;
        if (point % 2 == 0) ++expected_dynamic;
      }
    }
    CHECK(statik.count(query) == expected_static);
    CHECK(dynamic.count(query) == expected_dynamic);
  }
  for (Index point : ids) {
    if (point % 2 == 0) dynamic.set_active(point, false);
  }
}

template <anchor::Coordinate Coord>
std::set<anchor::JoinPair> brute_join(
    const anchor::BoxJoinInstance<Coord>& input) {
  std::set<anchor::JoinPair> result;
  for (std::size_t r = 0; r < input.r.size(); ++r) {
    if (!input.r.nonempty(r)) continue;
    for (std::size_t s = 0; s < input.s.size(); ++s) {
      if (!input.s.nonempty(s)) continue;
      bool intersects = true;
      for (std::size_t j = 0; j < input.dimensions(); ++j) {
        intersects &= input.r.lower(r, j) < input.s.upper(s, j) &&
                      input.s.lower(s, j) < input.r.upper(r, j);
      }
      if (intersects) result.insert({input.r.id(r), input.s.id(s)});
    }
  }
  return result;
}

template <anchor::Coordinate Coord>
void verify_all_algorithms(
    const std::shared_ptr<const anchor::BoxJoinInstance<Coord>>& input) {
  const auto truth = brute_join(*input);
  const std::array kinds{anchor::AlgorithmKind::AC, anchor::AlgorithmKind::AS,
                         anchor::AlgorithmKind::LiftedRT,
                         anchor::AlgorithmKind::SweepRT};
  for (auto kind : kinds) {
    auto sampler = anchor::make_sampler<Coord>(kind, input);
    CHECK(sampler->count() == truth.size());
    std::vector<anchor::JoinPair> zero;
    CHECK(sampler->sample(zero, 1) == anchor::SampleStatus::Ok);
    std::vector<anchor::JoinPair> output(300);
    const auto status = sampler->sample(output, 42);
    if (truth.empty()) {
      CHECK(status == anchor::SampleStatus::EmptyInstance);
    } else {
      CHECK(status == anchor::SampleStatus::Ok);
      for (const auto pair : output) CHECK(truth.contains(pair));
      std::vector<anchor::JoinPair> repeated(300);
      CHECK(sampler->sample(repeated, 42) == anchor::SampleStatus::Ok);
      CHECK(output == repeated);
      const auto& counters = sampler->diagnostics().counters;
      switch (kind) {
        case anchor::AlgorithmKind::AC:
          CHECK(counters.positive_atom_count != 0);
          CHECK(counters.alias_label_count == counters.positive_atom_count);
          break;
        case anchor::AlgorithmKind::AS:
          CHECK(counters.max_live_workspace_bytes != 0);
          break;
        case anchor::AlgorithmKind::LiftedRT:
          CHECK(counters.positive_degree_left_objects != 0);
          CHECK(counters.selected_left_objects != 0);
          CHECK(counters.range_tree_items != 0);
          break;
        case anchor::AlgorithmKind::SweepRT:
          CHECK(counters.nonzero_event_blocks != 0);
          CHECK(counters.selected_event_blocks != 0);
          break;
      }
    }
  }
}

std::shared_ptr<const anchor::BoxJoinInstance<std::int64_t>> make_case(
    std::size_t dimensions, std::uint64_t mask_r, std::uint64_t mask_s) {
  struct Candidate {
    std::array<std::int64_t, 3> lo;
    std::array<std::int64_t, 3> hi;
  };
  const std::array<Candidate, 5> candidates{{
      {{{0, 0, 0}}, {{2, 3, 1}}},
      {{{1, 1, 0}}, {{4, 2, 4}}},
      {{{2, 0, 2}}, {{3, 4, 5}}},
      {{{3, 2, 1}}, {{5, 5, 3}}},
      {{{2, 2, 2}}, {{2, 4, 4}}},  // empty in dimension 0
  }};
  anchor::BoxSet<std::int64_t> r(dimensions);
  anchor::BoxSet<std::int64_t> s(dimensions);
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    if (mask_r & (1ULL << i)) {
      r.add(100 + i,
            std::span<const std::int64_t>(candidates[i].lo).first(dimensions),
            std::span<const std::int64_t>(candidates[i].hi).first(dimensions));
    }
    if (mask_s & (1ULL << i)) {
      const std::size_t shifted = (i + 2) % candidates.size();
      s.add(200 + i,
            std::span<const std::int64_t>(candidates[shifted].lo)
                .first(dimensions),
            std::span<const std::int64_t>(candidates[shifted].hi)
                .first(dimensions));
    }
  }
  return std::make_shared<const anchor::BoxJoinInstance<std::int64_t>>(
      std::move(r), std::move(s));
}

void test_algorithms_exhaustive_small() {
  for (std::size_t d = 1; d <= 3; ++d) {
    for (std::uint64_t mr = 1; mr < 16; ++mr) {
      for (std::uint64_t ms = 1; ms < 16; ++ms) {
        verify_all_algorithms(make_case(d, mr, ms));
      }
    }
  }
}

void test_duplicate_identity_and_touching() {
  anchor::BoxSet<std::int64_t> r(1);
  anchor::BoxSet<std::int64_t> s(1);
  const std::array<std::int64_t, 1> zero{0};
  const std::array<std::int64_t, 1> one{1};
  const std::array<std::int64_t, 1> two{2};
  r.add(1, zero, one);
  r.add(2, zero, one);
  s.add(3, zero, one);
  s.add(4, one, two);  // touching only
  auto input =
      std::make_shared<const anchor::BoxJoinInstance<std::int64_t>>(
          std::move(r), std::move(s));
  verify_all_algorithms(input);
  CHECK(brute_join(*input).size() == 2);
}

void test_filtering_and_double_coordinates() {
  anchor::BoxSet<double> r(2);
  anchor::BoxSet<double> s(2);
  const std::array<double, 2> r0_lo{-1.5, 0.0};
  const std::array<double, 2> r0_hi{2.0, 3.0};
  const std::array<double, 2> empty_lo{4.0, 1.0};
  const std::array<double, 2> empty_hi{4.0, 5.0};
  const std::array<double, 2> s0_lo{1.0, 2.5};
  const std::array<double, 2> s0_hi{5.0, 4.0};
  const std::array<double, 2> touching_lo{2.0, 0.0};
  const std::array<double, 2> touching_hi{3.0, 1.0};
  r.add(10, r0_lo, r0_hi);
  r.add(11, empty_lo, empty_hi);
  s.add(20, s0_lo, s0_hi);
  s.add(21, touching_lo, touching_hi);
  auto input = std::make_shared<const anchor::BoxJoinInstance<double>>(
      std::move(r), std::move(s));
  CHECK(brute_join(*input).size() == 1);
  verify_all_algorithms(input);
}

void test_empty_after_filtering() {
  anchor::BoxSet<std::int64_t> r(1);
  anchor::BoxSet<std::int64_t> s(1);
  const std::array<std::int64_t, 1> zero{0};
  const std::array<std::int64_t, 1> one{1};
  r.add(1, one, one);
  s.add(2, zero, one);
  auto input =
      std::make_shared<const anchor::BoxJoinInstance<std::int64_t>>(
          std::move(r), std::move(s));
  verify_all_algorithms(input);
}

void test_sampling_balance() {
  anchor::BoxSet<std::int64_t> r(1);
  anchor::BoxSet<std::int64_t> s(1);
  const std::array<std::int64_t, 1> lo{0};
  const std::array<std::int64_t, 1> hi{5};
  r.add(1, lo, hi);
  r.add(2, lo, hi);
  s.add(3, lo, hi);
  s.add(4, lo, hi);
  auto input =
      std::make_shared<const anchor::BoxJoinInstance<std::int64_t>>(
          std::move(r), std::move(s));
  for (auto kind : {anchor::AlgorithmKind::AC, anchor::AlgorithmKind::AS,
                    anchor::AlgorithmKind::LiftedRT,
                    anchor::AlgorithmKind::SweepRT}) {
    auto sampler = anchor::make_sampler(kind, input);
    std::vector<anchor::JoinPair> output(40000);
    CHECK(sampler->sample(output, 99) == anchor::SampleStatus::Ok);
    std::map<anchor::JoinPair, std::size_t> frequencies;
    for (auto pair : output) ++frequencies[pair];
    CHECK(frequencies.size() == 4);
    for (const auto& [pair, frequency] : frequencies) {
      (void)pair;
      CHECK(frequency > 9000 && frequency < 11000);
    }
  }
}

void test_algorithms_dimension_four() {
  anchor::BoxSet<std::int64_t> r(4);
  anchor::BoxSet<std::int64_t> s(4);
  const std::array<std::int64_t, 4> r0l{0, 0, 0, 0};
  const std::array<std::int64_t, 4> r0u{3, 3, 3, 3};
  const std::array<std::int64_t, 4> r1l{2, -1, 1, 0};
  const std::array<std::int64_t, 4> r1u{5, 2, 4, 2};
  const std::array<std::int64_t, 4> s0l{1, 1, 1, 1};
  const std::array<std::int64_t, 4> s0u{4, 4, 4, 4};
  const std::array<std::int64_t, 4> s1l{3, 0, 0, 0};
  const std::array<std::int64_t, 4> s1u{6, 1, 2, 2};
  const std::array<std::int64_t, 4> s2l{2, -1, 3, 0};
  const std::array<std::int64_t, 4> s2u{3, 2, 5, 1};
  r.add(1, r0l, r0u);
  r.add(2, r1l, r1u);
  s.add(3, s0l, s0u);
  s.add(4, s1l, s1u);  // touches r0 in dimension 0
  s.add(5, s2l, s2u);  // touches r0 in dimension 2
  auto input =
      std::make_shared<const anchor::BoxJoinInstance<std::int64_t>>(
          std::move(r), std::move(s));
  CHECK(brute_join(*input).size() == 4);
  verify_all_algorithms(input);
}

void test_int64_extremes_without_negation() {
  anchor::BoxSet<std::int64_t> r(1);
  anchor::BoxSet<std::int64_t> s(1);
  const std::array<std::int64_t, 1> min{
      std::numeric_limits<std::int64_t>::min()};
  const std::array<std::int64_t, 1> minus_one{-1};
  const std::array<std::int64_t, 1> zero{0};
  const std::array<std::int64_t, 1> max{
      std::numeric_limits<std::int64_t>::max()};
  r.add(1, min, zero);
  s.add(2, minus_one, max);
  auto input =
      std::make_shared<const anchor::BoxJoinInstance<std::int64_t>>(
          std::move(r), std::move(s));
  verify_all_algorithms(input);
}

}  // namespace

int main() {
  try {
    test_uint128_and_random();
    test_fenwick();
    test_boxset_validation();
    test_range_tree();
    test_range_tree_random_strict_boundaries();
    test_duplicate_identity_and_touching();
    test_filtering_and_double_coordinates();
    test_empty_after_filtering();
    test_sampling_balance();
    test_algorithms_dimension_four();
    test_int64_extremes_without_negation();
    test_algorithms_exhaustive_small();
  } catch (const std::exception& error) {
    std::cerr << "unexpected exception: " << error.what() << '\n';
    return 2;
  }
  if (failures != 0) {
    std::cerr << failures << " test checks failed\n";
    return 1;
  }
  std::cout << "all anchor core tests passed\n";
  return 0;
}
