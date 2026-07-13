#pragma once

#include "anchor/types.hpp"

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace anchor {

inline std::uint64_t mix64(std::uint64_t x) noexcept {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31U);
}

inline std::uint64_t hash_domain(std::string_view domain) noexcept {
  std::uint64_t h = 1469598103934665603ULL;
  for (unsigned char c : domain) {
    h ^= c;
    h *= 1099511628211ULL;
  }
  return mix64(h);
}

inline std::uint64_t hash_combine(std::uint64_t a, std::uint64_t b) noexcept {
  return mix64(a ^ (mix64(b) + 0x9e3779b97f4a7c15ULL + (a << 6U) +
                    (a >> 2U)));
}

class DeterministicRng {
 public:
  explicit DeterministicRng(std::array<std::uint64_t, 4> state)
      : state_(state) {
    if (std::all_of(state_.begin(), state_.end(),
                    [](std::uint64_t word) { return word == 0; })) {
      state_[0] = 1;
    }
  }

  [[nodiscard]] std::uint64_t next_u64() noexcept {
    // xoshiro256** 1.0: a full-period 256-bit-state generator (except for
    // the forbidden all-zero state) with high-quality output scrambling.
    const std::uint64_t result = std::rotl(state_[1] * 5ULL, 7) * 9ULL;
    const std::uint64_t t = state_[1] << 17U;
    state_[2] ^= state_[0];
    state_[3] ^= state_[1];
    state_[1] ^= state_[2];
    state_[0] ^= state_[3];
    state_[2] ^= t;
    state_[3] = std::rotl(state_[3], 45);
    return result;
  }

 private:
  std::array<std::uint64_t, 4> state_;
};

class RngPool {
 public:
  RngPool(const std::array<std::uint64_t, 4>& seed,
          std::uint64_t query_id = 0) {
    // Keep a full 256-bit root. Cross-mixing makes every output state word
    // depend on every master word without collapsing the master seed to a
    // smaller PRNG state.
    constexpr std::array<std::uint64_t, 4> constants{
        0x6a09e667f3bcc909ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL};
    for (std::size_t i = 0; i < seed.size(); ++i) {
      root_[i] = mix64(seed[i] ^ constants[i] ^
                       std::rotl(seed[(i + 1) % 4], static_cast<int>(11 * i + 7)) ^
                       mix64(query_id + constants[(i + 2) % 4]));
    }
    for (std::size_t i = 0; i < root_.size(); ++i) {
      root_[i] ^= mix64(root_[(i + 1) % 4] + root_[(i + 3) % 4]);
    }
  }

  [[nodiscard]] DeterministicRng stream(std::string_view domain,
                                        std::uint64_t instance = 0) const {
    const std::uint64_t domain_hash = hash_domain(domain);
    std::array<std::uint64_t, 4> state{};
    for (std::size_t i = 0; i < state.size(); ++i) {
      state[i] = mix64(root_[i] ^
                       std::rotl(domain_hash, static_cast<int>(13 * i)) ^
                       mix64(instance + 0x9e3779b97f4a7c15ULL * (i + 1)) ^
                       root_[(i + 2) % 4]);
    }
    return DeterministicRng(state);
  }

 private:
  std::array<std::uint64_t, 4> root_{};
};

inline unsigned bit_width(UInt128 x) noexcept {
  if (x == 0) return 0;
  const auto high = static_cast<std::uint64_t>(x >> 64U);
  if (high != 0) return 64U + std::bit_width(high);
  return std::bit_width(static_cast<std::uint64_t>(x));
}

// Exact rejection sampler on [0,bound). It deliberately avoids modulo
// reduction, including for ranges wider than one machine word.
inline UInt128 uniform_below(DeterministicRng& rng, UInt128 bound) {
  if (bound == 0) throw std::invalid_argument("uniform bound must be positive");
  if (bound == 1) return 0;
  const unsigned bits = bit_width(bound - 1);
  for (;;) {
    UInt128 x;
    if (bits <= 64) {
      const std::uint64_t raw = rng.next_u64();
      const std::uint64_t mask =
          bits == 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << bits) - 1);
      x = static_cast<UInt128>(raw & mask);
    } else {
      const unsigned high_bits = bits - 64;
      const std::uint64_t high_mask =
          high_bits == 64 ? ~std::uint64_t{0}
                          : ((std::uint64_t{1} << high_bits) - 1);
      x = (static_cast<UInt128>(rng.next_u64() & high_mask) << 64U) |
          static_cast<UInt128>(rng.next_u64());
    }
    if (x < bound) return x;
  }
}

inline UInt128 uniform_integer(DeterministicRng& rng, UInt128 lo, UInt128 hi) {
  if (lo >= hi) throw std::invalid_argument("uniform interval must be nonempty");
  return checked_add(lo, uniform_below(rng, hi - lo), "uniform result overflow");
}

inline std::size_t uniform_index(DeterministicRng& rng, std::size_t size) {
  if (size == 0) throw std::invalid_argument("cannot sample an empty array");
  return checked_size(uniform_below(rng, static_cast<UInt128>(size)));
}

class IntegerAlias {
 public:
  IntegerAlias() = default;
  explicit IntegerAlias(std::span<const UInt128> weights) { build(weights); }

  void build(std::span<const UInt128> weights) {
    threshold_.clear();
    alias_.clear();
    total_ = 0;
    for (UInt128 w : weights) total_ = checked_add(total_, w, "weight sum overflow");
    if (weights.empty() || total_ == 0) {
      throw std::invalid_argument("alias table requires positive total weight");
    }
    const std::size_t k = weights.size();
    threshold_.resize(k);
    alias_.resize(k);
    std::vector<UInt128> scaled(k);
    std::vector<std::size_t> small;
    std::vector<std::size_t> large;
    small.reserve(k);
    large.reserve(k);
    for (std::size_t i = 0; i < k; ++i) {
      scaled[i] = checked_mul(weights[i], static_cast<UInt128>(k),
                              "alias scaled weight overflow");
      (scaled[i] < total_ ? small : large).push_back(i);
    }
    while (!small.empty() && !large.empty()) {
      const std::size_t s = small.back();
      small.pop_back();
      const std::size_t l = large.back();
      large.pop_back();
      threshold_[s] = scaled[s];
      alias_[s] = l;
      const UInt128 deficit = total_ - scaled[s];
      if (scaled[l] < deficit) {
        throw std::logic_error("alias mass invariant violated");
      }
      scaled[l] -= deficit;
      (scaled[l] < total_ ? small : large).push_back(l);
    }
    for (std::size_t i : small) {
      threshold_[i] = total_;
      alias_[i] = i;
    }
    for (std::size_t i : large) {
      threshold_[i] = total_;
      alias_[i] = i;
    }
  }

  [[nodiscard]] std::size_t size() const noexcept { return threshold_.size(); }
  [[nodiscard]] UInt128 total() const noexcept { return total_; }

  [[nodiscard]] std::size_t sample(DeterministicRng& rng) const {
    if (threshold_.empty()) throw std::logic_error("unbuilt alias table");
    const std::size_t slot = uniform_index(rng, threshold_.size());
    const UInt128 r = uniform_below(rng, total_);
    return r < threshold_[slot] ? slot : alias_[slot];
  }

 private:
  UInt128 total_{};
  std::vector<UInt128> threshold_;
  std::vector<std::size_t> alias_;
};

template <class T>
void fisher_yates(std::span<T> values, DeterministicRng& rng) {
  for (std::size_t i = values.size(); i > 1; --i) {
    const std::size_t j = uniform_index(rng, i);
    std::swap(values[i - 1], values[j]);
  }
}

template <class T>
void fisher_yates(std::vector<T>& values, DeterministicRng& rng) {
  fisher_yates(std::span<T>(values), rng);
}

inline std::vector<std::size_t> draw_quotas(
    std::size_t h, std::span<const UInt128> weights, DeterministicRng& rng) {
  std::vector<std::size_t> quotas(weights.size(), 0);
  if (h == 0) return quotas;
  IntegerAlias alias(weights);
  for (std::size_t i = 0; i < h; ++i) ++quotas[alias.sample(rng)];
  return quotas;
}

}  // namespace anchor
