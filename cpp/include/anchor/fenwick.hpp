#pragma once

#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace anchor {

class Fenwick {
 public:
  Fenwick() = default;
  explicit Fenwick(std::size_t size) : tree_(size + 1, 0) {}

  [[nodiscard]] std::size_t size() const noexcept {
    return tree_.empty() ? 0 : tree_.size() - 1;
  }

  void add(std::size_t position, int delta) {
    if (position >= size()) throw std::out_of_range("Fenwick update position");
    if (delta != 1 && delta != -1) {
      throw std::invalid_argument("active Fenwick only accepts +/-1 updates");
    }
    const auto current = range_sum(position, position + 1);
    if ((delta < 0 && current == 0) || (delta > 0 && current != 0)) {
      throw std::logic_error(delta < 0 ? "deactivating an inactive occurrence"
                                       : "activating an active occurrence");
    }
    for (std::size_t i = position + 1; i < tree_.size(); i += i & -i) {
      if (delta > 0) {
        ++tree_[i];
      } else {
        --tree_[i];
      }
    }
  }

  [[nodiscard]] std::uint64_t prefix_sum(std::size_t end) const {
    if (end > size()) throw std::out_of_range("Fenwick prefix endpoint");
    std::uint64_t sum = 0;
    for (std::size_t i = end; i != 0; i -= i & -i) sum += tree_[i];
    return sum;
  }

  [[nodiscard]] std::uint64_t range_sum(std::size_t begin,
                                        std::size_t end) const {
    if (begin > end) throw std::invalid_argument("reversed Fenwick range");
    return prefix_sum(end) - prefix_sum(begin);
  }

  // Return the zero-based position containing the target-th active item.
  // target is one-based and must be in [1,total()].
  [[nodiscard]] std::size_t select(std::uint64_t target) const {
    const std::uint64_t total = prefix_sum(size());
    if (target == 0 || target > total) {
      throw std::out_of_range("Fenwick select rank");
    }
    std::size_t index = 0;
    std::size_t step = std::bit_floor(size());
    while (step != 0) {
      const std::size_t next = index + step;
      if (next <= size() && tree_[next] < target) {
        index = next;
        target -= tree_[next];
      }
      step >>= 1U;
    }
    if (index >= size()) throw std::logic_error("Fenwick select invariant");
    return index;
  }

  void clear() { std::fill(tree_.begin(), tree_.end(), 0); }

 private:
  std::vector<std::uint64_t> tree_;
};

}  // namespace anchor
