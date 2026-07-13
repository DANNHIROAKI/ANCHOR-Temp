#include "sha256.hpp"

#include <algorithm>
#include <bit>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace anchor::app {
namespace {

constexpr std::array<std::uint32_t, 64> kRoundConstants{
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U,
    0x3956c25bU, 0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U,
    0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U,
    0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U,
    0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU,
    0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U,
    0xc6e00bf3U, 0xd5a79147U, 0x06ca6351U, 0x14292967U,
    0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U,
    0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U,
    0xd192e819U, 0xd6990624U, 0xf40e3585U, 0x106aa070U,
    0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U,
    0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U,
    0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U};

std::uint32_t load_be32(const std::byte* source) {
  return (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(source[0]))
          << 24U) |
         (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(source[1]))
          << 16U) |
         (static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(source[2]))
          << 8U) |
         static_cast<std::uint32_t>(std::to_integer<std::uint8_t>(source[3]));
}

void store_be32(std::uint32_t value, std::uint8_t* destination) {
  destination[0] = static_cast<std::uint8_t>(value >> 24U);
  destination[1] = static_cast<std::uint8_t>(value >> 16U);
  destination[2] = static_cast<std::uint8_t>(value >> 8U);
  destination[3] = static_cast<std::uint8_t>(value);
}

}  // namespace

void Sha256::compress(const std::byte* block) {
  std::array<std::uint32_t, 64> words{};
  for (std::size_t i = 0; i < 16; ++i) {
    words[i] = load_be32(block + 4 * i);
  }
  for (std::size_t i = 16; i < words.size(); ++i) {
    const std::uint32_t s0 = std::rotr(words[i - 15], 7) ^
                             std::rotr(words[i - 15], 18) ^
                             (words[i - 15] >> 3U);
    const std::uint32_t s1 = std::rotr(words[i - 2], 17) ^
                             std::rotr(words[i - 2], 19) ^
                             (words[i - 2] >> 10U);
    words[i] = words[i - 16] + s0 + words[i - 7] + s1;
  }

  std::uint32_t a = state_[0];
  std::uint32_t b = state_[1];
  std::uint32_t c = state_[2];
  std::uint32_t d = state_[3];
  std::uint32_t e = state_[4];
  std::uint32_t f = state_[5];
  std::uint32_t g = state_[6];
  std::uint32_t h = state_[7];
  for (std::size_t i = 0; i < words.size(); ++i) {
    const std::uint32_t big1 = std::rotr(e, 6) ^ std::rotr(e, 11) ^
                               std::rotr(e, 25);
    const std::uint32_t choose = (e & f) ^ ((~e) & g);
    const std::uint32_t temp1 =
        h + big1 + choose + kRoundConstants[i] + words[i];
    const std::uint32_t big0 = std::rotr(a, 2) ^ std::rotr(a, 13) ^
                               std::rotr(a, 22);
    const std::uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
    const std::uint32_t temp2 = big0 + majority;
    h = g;
    g = f;
    f = e;
    e = d + temp1;
    d = c;
    c = b;
    b = a;
    a = temp1 + temp2;
  }
  state_[0] += a;
  state_[1] += b;
  state_[2] += c;
  state_[3] += d;
  state_[4] += e;
  state_[5] += f;
  state_[6] += g;
  state_[7] += h;
}

void Sha256::update(std::span<const std::byte> bytes) {
  if (finished_) throw std::logic_error("SHA-256 update after finish");
  if (bytes.size() > std::numeric_limits<std::uint64_t>::max() - total_bytes_) {
    throw std::overflow_error("SHA-256 input length overflow");
  }
  total_bytes_ += static_cast<std::uint64_t>(bytes.size());
  while (!bytes.empty()) {
    const std::size_t amount =
        std::min(buffer_.size() - buffered_, bytes.size());
    std::memcpy(buffer_.data() + buffered_, bytes.data(), amount);
    buffered_ += amount;
    bytes = bytes.subspan(amount);
    if (buffered_ == buffer_.size()) {
      compress(buffer_.data());
      buffered_ = 0;
    }
  }
}

std::array<std::uint8_t, 32> Sha256::finish() {
  if (finished_) throw std::logic_error("SHA-256 finish called twice");
  if (total_bytes_ > std::numeric_limits<std::uint64_t>::max() / 8U) {
    throw std::overflow_error("SHA-256 bit length overflow");
  }
  const std::uint64_t bit_length = total_bytes_ * 8U;
  buffer_[buffered_++] = std::byte{0x80};
  if (buffered_ > 56) {
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_),
              buffer_.end(), std::byte{0});
    compress(buffer_.data());
    buffered_ = 0;
  }
  std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_),
            buffer_.begin() + 56, std::byte{0});
  for (std::size_t i = 0; i < 8; ++i) {
    buffer_[63 - i] =
        static_cast<std::byte>(bit_length >> static_cast<unsigned>(8 * i));
  }
  compress(buffer_.data());
  finished_ = true;

  std::array<std::uint8_t, 32> output{};
  for (std::size_t i = 0; i < state_.size(); ++i) {
    store_be32(state_[i], output.data() + 4 * i);
  }
  return output;
}

std::string hex_digest(const std::array<std::uint8_t, 32>& digest) {
  constexpr char kHex[] = "0123456789abcdef";
  std::string output(64, '0');
  for (std::size_t i = 0; i < digest.size(); ++i) {
    output[2 * i] = kHex[digest[i] >> 4U];
    output[2 * i + 1] = kHex[digest[i] & 0x0fU];
  }
  return output;
}

}  // namespace anchor::app
