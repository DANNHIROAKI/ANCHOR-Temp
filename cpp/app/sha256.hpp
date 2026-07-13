#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace anchor::app {

class Sha256 {
 public:
  Sha256() = default;

  void update(std::span<const std::byte> bytes);
  [[nodiscard]] std::array<std::uint8_t, 32> finish();

 private:
  void compress(const std::byte* block);

  std::array<std::uint32_t, 8> state_{
      0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
      0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U};
  std::array<std::byte, 64> buffer_{};
  std::size_t buffered_{};
  std::uint64_t total_bytes_{};
  bool finished_{};
};

[[nodiscard]] std::string hex_digest(
    const std::array<std::uint8_t, 32>& digest);

}  // namespace anchor::app
