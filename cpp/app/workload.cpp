#include "workload.hpp"

#include "sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace anchor::app {
namespace {

constexpr std::size_t kHeaderSize = 128;
constexpr std::uint64_t kAlignment = 64;
constexpr std::array<std::byte, 8> kMagic{
    std::byte{'A'}, std::byte{'N'}, std::byte{'C'}, std::byte{'H'},
    std::byte{'O'}, std::byte{'R'}, std::byte{'W'}, std::byte{0}};

class FileDescriptor {
 public:
  explicit FileDescriptor(const std::filesystem::path& path) {
    descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (descriptor_ < 0) {
      throw WorkloadFormatError("cannot open workload: " +
                                std::string(std::strerror(errno)));
    }
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  ~FileDescriptor() {
    if (descriptor_ >= 0) ::close(descriptor_);
  }
  [[nodiscard]] int get() const noexcept { return descriptor_; }

 private:
  int descriptor_{-1};
};

std::uint16_t read_le16(const std::byte* source) {
  return static_cast<std::uint16_t>(
      static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(source[0])) |
      (static_cast<std::uint16_t>(std::to_integer<std::uint8_t>(source[1]))
       << 8U));
}

std::uint32_t read_le32(const std::byte* source) {
  std::uint32_t value = 0;
  for (std::size_t i = 0; i < 4; ++i) {
    value |= static_cast<std::uint32_t>(
                 std::to_integer<std::uint8_t>(source[i]))
             << static_cast<unsigned>(8 * i);
  }
  return value;
}

std::uint64_t read_le64(const std::byte* source) {
  std::uint64_t value = 0;
  for (std::size_t i = 0; i < 8; ++i) {
    value |= static_cast<std::uint64_t>(
                 std::to_integer<std::uint8_t>(source[i]))
             << static_cast<unsigned>(8 * i);
  }
  return value;
}

std::uint64_t checked_add64(std::uint64_t left, std::uint64_t right,
                            std::string_view what) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw WorkloadFormatError(std::string(what) + " overflows uint64");
  }
  return left + right;
}

std::uint64_t checked_mul64(std::uint64_t left, std::uint64_t right,
                            std::string_view what) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw WorkloadFormatError(std::string(what) + " overflows uint64");
  }
  return left * right;
}

std::size_t as_size(std::uint64_t value, std::string_view what) {
  if (value > std::numeric_limits<std::size_t>::max()) {
    throw WorkloadFormatError(std::string(what) + " exceeds size_t");
  }
  return static_cast<std::size_t>(value);
}

void pread_exact(int descriptor, std::uint64_t offset,
                 std::span<std::byte> destination, std::string_view what) {
  if (offset > static_cast<std::uint64_t>(
                   std::numeric_limits<off_t>::max())) {
    throw WorkloadFormatError(std::string(what) + " offset exceeds off_t");
  }
  std::size_t completed = 0;
  while (completed < destination.size()) {
    const std::size_t request = std::min<std::size_t>(
        destination.size() - completed,
        static_cast<std::size_t>(std::numeric_limits<ssize_t>::max()));
    const std::uint64_t position =
        checked_add64(offset, static_cast<std::uint64_t>(completed), what);
    if (position > static_cast<std::uint64_t>(
                       std::numeric_limits<off_t>::max())) {
      throw WorkloadFormatError(std::string(what) + " offset exceeds off_t");
    }
    const ssize_t result = ::pread(descriptor, destination.data() + completed,
                                   request, static_cast<off_t>(position));
    if (result < 0) {
      if (errno == EINTR) continue;
      throw WorkloadFormatError(std::string("cannot read ") +
                                std::string(what) + ": " +
                                std::strerror(errno));
    }
    if (result == 0) {
      throw WorkloadFormatError(std::string("truncated ") + std::string(what));
    }
    completed += static_cast<std::size_t>(result);
  }
}

template <class T>
std::vector<T> read_array(int descriptor, std::uint64_t offset,
                          std::uint64_t count, Sha256& digest,
                          std::string_view name) {
  const std::size_t size = as_size(count, name);
  std::vector<T> values(size);
  auto bytes = std::as_writable_bytes(std::span<T>(values));
  pread_exact(descriptor, offset, bytes, name);
  digest.update(std::as_bytes(std::span<const T>(values)));
  return values;
}

void validate_padding(int descriptor, std::uint64_t begin,
                      std::uint64_t end) {
  if (end < begin) throw WorkloadFormatError("overlapping workload arrays");
  std::array<std::byte, 64 * 1024> bytes{};
  std::uint64_t cursor = begin;
  while (cursor < end) {
    const std::uint64_t remaining = end - cursor;
    const std::size_t amount = static_cast<std::size_t>(
        std::min<std::uint64_t>(remaining, bytes.size()));
    auto chunk = std::span<std::byte>(bytes).first(amount);
    pread_exact(descriptor, cursor, chunk, "alignment padding");
    if (std::any_of(chunk.begin(), chunk.end(),
                    [](std::byte value) { return value != std::byte{0}; })) {
      throw WorkloadFormatError("nonzero workload alignment padding");
    }
    cursor = checked_add64(cursor, amount, "padding cursor");
  }
}

template <class Coord>
void validate_positive_finite(std::span<const Coord> lower,
                              std::span<const Coord> upper,
                              std::string_view side) {
  if (lower.size() != upper.size()) {
    throw WorkloadFormatError(std::string(side) + " coordinate size mismatch");
  }
  for (std::size_t i = 0; i < lower.size(); ++i) {
    if constexpr (std::same_as<Coord, double>) {
      if (!std::isfinite(lower[i]) || !std::isfinite(upper[i])) {
        throw NumericDegeneracyError(std::string(side) +
                                     " contains a non-finite endpoint");
      }
    }
    if (!(lower[i] < upper[i])) {
      throw NumericDegeneracyError(std::string(side) +
                                   " contains an empty or inverted box");
    }
  }
}

template <class Coord>
std::variant<DoubleInstance, Int64Instance> make_instance(
    std::size_t dimension, std::vector<std::uint64_t> r_ids,
    std::vector<Coord> r_lower, std::vector<Coord> r_upper,
    std::vector<std::uint64_t> s_ids, std::vector<Coord> s_lower,
    std::vector<Coord> s_upper) {
  validate_positive_finite<Coord>(r_lower, r_upper, "R");
  validate_positive_finite<Coord>(s_lower, s_upper, "S");
  try {
    BoxSet<Coord> r(dimension, std::move(r_ids), std::move(r_lower),
                    std::move(r_upper));
    BoxSet<Coord> s(dimension, std::move(s_ids), std::move(s_lower),
                    std::move(s_upper));
    auto instance = std::make_shared<const BoxJoinInstance<Coord>>(
        std::move(r), std::move(s));
    return std::variant<DoubleInstance, Int64Instance>{std::move(instance)};
  } catch (const std::invalid_argument& error) {
    throw WorkloadFormatError(std::string("invalid workload arrays: ") +
                              error.what());
  }
}

}  // namespace

LoadedWorkload load_workload(const std::filesystem::path& path) {
  if constexpr (std::endian::native != std::endian::little) {
    throw WorkloadFormatError(
        "canonical workload loading requires a little-endian host");
  }

  FileDescriptor file(path);
  struct stat metadata {};
  if (::fstat(file.get(), &metadata) != 0) {
    throw WorkloadFormatError("cannot stat workload: " +
                              std::string(std::strerror(errno)));
  }
  if (!S_ISREG(metadata.st_mode) || metadata.st_size < 0) {
    throw WorkloadFormatError("workload must be a regular file");
  }
  const std::uint64_t actual_size = static_cast<std::uint64_t>(metadata.st_size);
  std::array<std::byte, kHeaderSize> header{};
  pread_exact(file.get(), 0, header, "workload header");
  if (!std::equal(kMagic.begin(), kMagic.end(), header.begin())) {
    throw WorkloadFormatError("invalid workload magic");
  }
  if (read_le16(header.data() + 8) != 1 ||
      std::to_integer<std::uint8_t>(header[10]) != 1 ||
      read_le32(header.data() + 20) != kHeaderSize) {
    throw WorkloadFormatError("unsupported workload format or byte order");
  }
  const auto endpoint_code =
      std::to_integer<std::uint8_t>(header[11]);
  if (endpoint_code != static_cast<std::uint8_t>(EndpointType::Float64) &&
      endpoint_code != static_cast<std::uint8_t>(EndpointType::Int64)) {
    throw WorkloadFormatError("unknown endpoint type code");
  }
  const std::uint64_t dimension = read_le32(header.data() + 12);
  if (dimension == 0 || read_le32(header.data() + 16) != 0) {
    throw WorkloadFormatError("invalid dimension or unsupported header flags");
  }
  const std::uint64_t n_r = read_le64(header.data() + 24);
  const std::uint64_t n_s = read_le64(header.data() + 32);
  std::array<std::uint64_t, 6> offsets{};
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    offsets[i] = read_le64(header.data() + 40 + 8 * i);
  }
  const std::uint64_t declared_size = read_le64(header.data() + 88);
  if (declared_size != actual_size) {
    throw WorkloadFormatError("file size differs from workload header");
  }
  std::array<std::uint8_t, 32> expected_digest{};
  for (std::size_t i = 0; i < expected_digest.size(); ++i) {
    expected_digest[i] = std::to_integer<std::uint8_t>(header[96 + i]);
  }

  const std::uint64_t r_coordinates =
      checked_mul64(n_r, dimension, "R coordinate count");
  const std::uint64_t s_coordinates =
      checked_mul64(n_s, dimension, "S coordinate count");
  const std::array<std::uint64_t, 6> counts{
      n_r, r_coordinates, r_coordinates, n_s, s_coordinates, s_coordinates};
  std::array<std::uint64_t, 6> byte_sizes{};
  std::uint64_t logical_payload_bytes = 0;
  std::uint64_t previous_end = kHeaderSize;
  for (std::size_t i = 0; i < offsets.size(); ++i) {
    byte_sizes[i] = checked_mul64(counts[i], 8, "array byte size");
    if (offsets[i] < previous_end || offsets[i] % kAlignment != 0) {
      throw WorkloadFormatError("array offset is overlapping or unaligned");
    }
    validate_padding(file.get(), previous_end, offsets[i]);
    previous_end = checked_add64(offsets[i], byte_sizes[i], "array end");
    if (previous_end > declared_size) {
      throw WorkloadFormatError("array extends beyond end of workload");
    }
    logical_payload_bytes = checked_add64(
        logical_payload_bytes, byte_sizes[i], "logical payload size");
  }
  if (previous_end != declared_size) {
    throw WorkloadFormatError("unexpected trailing workload bytes");
  }

  Sha256 digest;
  auto r_ids = read_array<std::uint64_t>(file.get(), offsets[0], counts[0],
                                         digest, "R id array");
  const EndpointType endpoint_type = static_cast<EndpointType>(endpoint_code);
  std::variant<DoubleInstance, Int64Instance> instance;
  if (endpoint_type == EndpointType::Float64) {
    auto r_lower = read_array<double>(file.get(), offsets[1], counts[1], digest,
                                      "R lower array");
    auto r_upper = read_array<double>(file.get(), offsets[2], counts[2], digest,
                                      "R upper array");
    auto s_ids = read_array<std::uint64_t>(file.get(), offsets[3], counts[3],
                                           digest, "S id array");
    auto s_lower = read_array<double>(file.get(), offsets[4], counts[4], digest,
                                      "S lower array");
    auto s_upper = read_array<double>(file.get(), offsets[5], counts[5], digest,
                                      "S upper array");
    if (digest.finish() != expected_digest) {
      throw WorkloadFormatError("logical payload SHA-256 mismatch");
    }
    instance = make_instance<double>(
        as_size(dimension, "dimension"), std::move(r_ids), std::move(r_lower),
        std::move(r_upper), std::move(s_ids), std::move(s_lower),
        std::move(s_upper));
  } else {
    auto r_lower = read_array<std::int64_t>(file.get(), offsets[1], counts[1],
                                            digest, "R lower array");
    auto r_upper = read_array<std::int64_t>(file.get(), offsets[2], counts[2],
                                            digest, "R upper array");
    auto s_ids = read_array<std::uint64_t>(file.get(), offsets[3], counts[3],
                                           digest, "S id array");
    auto s_lower = read_array<std::int64_t>(file.get(), offsets[4], counts[4],
                                            digest, "S lower array");
    auto s_upper = read_array<std::int64_t>(file.get(), offsets[5], counts[5],
                                            digest, "S upper array");
    if (digest.finish() != expected_digest) {
      throw WorkloadFormatError("logical payload SHA-256 mismatch");
    }
    instance = make_instance<std::int64_t>(
        as_size(dimension, "dimension"), std::move(r_ids), std::move(r_lower),
        std::move(r_upper), std::move(s_ids), std::move(s_lower),
        std::move(s_upper));
  }
  // Do not keep workload page-cache pages charged to a memory-measurement process.
  (void)::posix_fadvise(file.get(), 0, 0, POSIX_FADV_DONTNEED);

  return LoadedWorkload{path,
                        endpoint_type,
                        as_size(dimension, "dimension"),
                        as_size(n_r, "R object count"),
                        as_size(n_s, "S object count"),
                        declared_size,
                        logical_payload_bytes,
                        expected_digest,
                        std::move(instance)};
}

std::string_view endpoint_type_name(EndpointType type) noexcept {
  switch (type) {
    case EndpointType::Float64:
      return "float64";
    case EndpointType::Int64:
      return "int64";
  }
  return "unknown";
}

}  // namespace anchor::app
