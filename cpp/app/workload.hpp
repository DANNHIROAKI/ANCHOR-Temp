#pragma once

#include "anchor/types.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>

namespace anchor::app {

class WorkloadFormatError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class NumericDegeneracyError : public WorkloadFormatError {
 public:
  using WorkloadFormatError::WorkloadFormatError;
};

enum class EndpointType : std::uint8_t { Float64 = 1, Int64 = 2 };

using DoubleInstance = std::shared_ptr<const BoxJoinInstance<double>>;
using Int64Instance = std::shared_ptr<const BoxJoinInstance<std::int64_t>>;

struct LoadedWorkload {
  std::filesystem::path path;
  EndpointType endpoint_type{};
  std::size_t dimension{};
  std::size_t n_r{};
  std::size_t n_s{};
  std::uint64_t file_size_bytes{};
  std::uint64_t logical_payload_bytes{};
  std::array<std::uint8_t, 32> payload_sha256{};
  std::variant<DoubleInstance, Int64Instance> instance;
};

// Reads each of the six logical arrays directly into the vector ultimately
// owned by its BoxSet. No mmap or coordinate conversion is performed.
[[nodiscard]] LoadedWorkload load_workload(
    const std::filesystem::path& path);

[[nodiscard]] std::string_view endpoint_type_name(EndpointType type) noexcept;

}  // namespace anchor::app
