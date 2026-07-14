#include "anchor/algorithms.hpp"
#include "sha256.hpp"
#include "workload.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <signal.h>
#include <sys/resource.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace anchor::app {
namespace {

constexpr std::string_view kSchema = "anchor-benchmark-v1";
constexpr char kSetupTimeoutJson[] =
    "{\"schema_version\":\"anchor-benchmark-v1\",\"status\":\"TO\","
    "\"failure_stage\":\"setup-interval\","
    "\"timeout_source\":\"benchmark-setup-interval-sigalrm\","
    "\"error_message\":\"setup interval exceeded its independent timeout\"}\n";

void setup_timeout_handler(int) noexcept {
  const ssize_t ignored = ::write(STDOUT_FILENO, kSetupTimeoutJson,
                                  sizeof(kSetupTimeoutJson) - 1U);
  (void)ignored;
  ::_exit(124);
}

class CliError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class MemoryMeasurementError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

enum class MeasurementMode { Time, Memory };
enum class Task { OneShot, CountOnly, PreparedQuery };

struct Options {
  std::filesystem::path workload;
  AlgorithmKind algorithm{};
  std::string algorithm_name;
  std::size_t samples{};
  std::string seed_hex;
  SamplerOptions sampler_options;
  MeasurementMode measurement_mode{};
  Task task{};
  std::uint64_t timeout_seconds{};
  std::uint64_t setup_timeout_seconds{900};
  std::optional<int> memory_event_fd;
  std::optional<int> memory_ack_fd;
  std::optional<std::filesystem::path> dump_output;
};

struct RunResult {
  std::string status{"OK"};
  std::string failure_stage;
  std::string error_message;
  UInt128 join_count{};
  bool has_join_count{};
  SampleStatus sample_status{SampleStatus::Ok};
  std::size_t output_length{};
  std::string output_sha256;
  bool membership_checked{};
  double primary_seconds{};
  std::optional<double> prepare_seconds;
  std::optional<double> query_seconds;
  Diagnostics diagnostics;
};

class JsonObject {
 public:
  void string(std::string_view key, std::string_view value) {
    add(key, quote(value));
  }
  template <std::unsigned_integral Integer>
  void number(std::string_view key, Integer value) {
    add(key, std::to_string(static_cast<unsigned long long>(value)));
  }
  void number(std::string_view key, double value) {
    if (!std::isfinite(value)) throw std::logic_error("non-finite JSON number");
    std::array<char, 64> buffer{};
    const auto [end, error] = std::to_chars(
        buffer.data(), buffer.data() + buffer.size(), value,
        std::chars_format::general, std::numeric_limits<double>::max_digits10);
    if (error != std::errc{}) throw std::logic_error("cannot encode JSON number");
    add(key, std::string(buffer.data(), end));
  }
  void boolean(std::string_view key, bool value) {
    add(key, value ? "true" : "false");
  }
  void null(std::string_view key) { add(key, "null"); }
  void raw(std::string_view key, std::string value) {
    add(key, std::move(value));
  }
  [[nodiscard]] std::string finish() const {
    std::string output{"{"};
    for (std::size_t i = 0; i < fields_.size(); ++i) {
      if (i != 0) output.push_back(',');
      output += fields_[i];
    }
    output += "}\n";
    return output;
  }
  [[nodiscard]] static std::string quote(std::string_view value) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string output;
    output.reserve(value.size() + 2);
    output.push_back('"');
    for (unsigned char byte : value) {
      switch (byte) {
        case '"':
          output += "\\\"";
          break;
        case '\\':
          output += "\\\\";
          break;
        case '\b':
          output += "\\b";
          break;
        case '\f':
          output += "\\f";
          break;
        case '\n':
          output += "\\n";
          break;
        case '\r':
          output += "\\r";
          break;
        case '\t':
          output += "\\t";
          break;
        default:
          if (byte < 0x20U) {
            output += "\\u00";
            output.push_back(kHex[byte >> 4U]);
            output.push_back(kHex[byte & 0x0fU]);
          } else {
            output.push_back(static_cast<char>(byte));
          }
      }
    }
    output.push_back('"');
    return output;
  }

 private:
  void add(std::string_view key, std::string value) {
    fields_.push_back(quote(key) + ":" + std::move(value));
  }
  std::vector<std::string> fields_;
};

std::string_view measurement_mode_name(MeasurementMode mode) {
  return mode == MeasurementMode::Time ? "time" : "memory";
}

std::string_view task_name(Task task) {
  switch (task) {
    case Task::OneShot:
      return "oneshot";
    case Task::CountOnly:
      return "count-only";
    case Task::PreparedQuery:
      return "prepared-query";
  }
  return "unknown";
}

std::uint64_t parse_u64(std::string_view text, std::string_view option,
                        bool allow_zero = true) {
  std::uint64_t value = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), value);
  if (text.empty() || error != std::errc{} || end != text.data() + text.size() ||
      (!allow_zero && value == 0)) {
    throw CliError(std::string(option) + " requires " +
                   (allow_zero ? "a nonnegative" : "a positive") +
                   " decimal integer");
  }
  return value;
}

int parse_fd(std::string_view text) {
  const std::uint64_t value = parse_u64(text, "--start-gate-fd");
  if (value > static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
    throw CliError("--start-gate-fd is outside the descriptor range");
  }
  return static_cast<int>(value);
}

std::uint8_t hex_nibble(char value) {
  if (value >= '0' && value <= '9') return static_cast<std::uint8_t>(value - '0');
  if (value >= 'a' && value <= 'f') {
    return static_cast<std::uint8_t>(value - 'a' + 10);
  }
  if (value >= 'A' && value <= 'F') {
    return static_cast<std::uint8_t>(value - 'A' + 10);
  }
  throw CliError("--seed-hex must contain exactly 64 hexadecimal digits");
}

SamplerOptions parse_seed(std::string_view value) {
  if (value.size() != 64) {
    throw CliError("--seed-hex must contain exactly 64 hexadecimal digits");
  }
  SamplerOptions result;
  for (std::size_t word = 0; word < result.seed.size(); ++word) {
    std::uint64_t parsed = 0;
    for (std::size_t digit = 0; digit < 16; ++digit) {
      parsed = (parsed << 4U) | hex_nibble(value[word * 16 + digit]);
    }
    result.seed[word] = parsed;
  }
  return result;
}

Options parse_options(int argc, char** argv) {
  if (argc < 2 || std::string_view(argv[1]) != "run") {
    throw CliError("expected the 'run' subcommand (use --help for usage)");
  }
  Options options;
  std::unordered_set<std::string> seen;
  for (int i = 2; i < argc; ++i) {
    const std::string name(argv[i]);
    if (name == "--help" || name == "-h") {
      throw CliError("__HELP__");
    }
    constexpr std::array<std::string_view, 11> kKnown{
        "--workload",       "--algorithm",     "--samples",
        "--seed-hex",       "--measurement-mode", "--task",
        "--timeout-seconds", "--setup-timeout-seconds", "--memory-event-fd",
        "--memory-ack-fd", "--dump-output"};
    if (std::find(kKnown.begin(), kKnown.end(), name) == kKnown.end()) {
      throw CliError("unknown option: " + name);
    }
    if (!seen.insert(name).second) throw CliError("duplicate option: " + name);
    if (++i >= argc) throw CliError("missing value for " + name);
    const std::string_view value(argv[i]);
    if (name == "--workload") {
      options.workload = std::filesystem::path(value);
    } else if (name == "--algorithm") {
      if (value == "ac") {
        options.algorithm = AlgorithmKind::AC;
        options.algorithm_name = "AC";
      } else if (value == "as") {
        options.algorithm = AlgorithmKind::AS;
        options.algorithm_name = "AS";
      } else if (value == "sweeprt") {
        options.algorithm = AlgorithmKind::SweepRT;
        options.algorithm_name = "SweepRT";
      } else if (value == "liftedrt") {
        options.algorithm = AlgorithmKind::LiftedRT;
        options.algorithm_name = "LiftedRT";
      } else {
        throw CliError("--algorithm must be ac, as, sweeprt, or liftedrt");
      }
    } else if (name == "--samples") {
      const auto parsed = parse_u64(value, "--samples");
      if (parsed > std::numeric_limits<std::size_t>::max()) {
        throw CliError("--samples exceeds size_t");
      }
      options.samples = static_cast<std::size_t>(parsed);
    } else if (name == "--seed-hex") {
      options.seed_hex = std::string(value);
      options.sampler_options = parse_seed(value);
    } else if (name == "--measurement-mode") {
      if (value == "time") {
        options.measurement_mode = MeasurementMode::Time;
      } else if (value == "memory") {
        options.measurement_mode = MeasurementMode::Memory;
      } else {
        throw CliError("--measurement-mode must be time or memory");
      }
    } else if (name == "--task") {
      if (value == "oneshot") {
        options.task = Task::OneShot;
      } else if (value == "count-only") {
        options.task = Task::CountOnly;
      } else if (value == "prepared-query") {
        options.task = Task::PreparedQuery;
      } else {
        throw CliError("--task must be oneshot, count-only, or prepared-query");
      }
    } else if (name == "--timeout-seconds") {
      options.timeout_seconds = parse_u64(value, "--timeout-seconds", false);
    } else if (name == "--setup-timeout-seconds") {
      options.setup_timeout_seconds =
          parse_u64(value, "--setup-timeout-seconds", false);
    } else if (name == "--memory-event-fd") {
      options.memory_event_fd = parse_fd(value);
    } else if (name == "--memory-ack-fd") {
      options.memory_ack_fd = parse_fd(value);
    } else if (name == "--dump-output") {
      options.dump_output = std::filesystem::path(value);
    }
  }

  constexpr std::array<std::string_view, 7> kRequired{
      "--workload", "--algorithm", "--samples", "--seed-hex",
      "--measurement-mode", "--task", "--timeout-seconds"};
  for (std::string_view required : kRequired) {
    if (!seen.contains(std::string(required))) {
      throw CliError("missing required option: " + std::string(required));
    }
  }
  if (options.measurement_mode == MeasurementMode::Memory &&
      (!options.memory_event_fd || !options.memory_ack_fd)) {
    throw MemoryMeasurementError(
        "memory mode requires --memory-event-fd and --memory-ack-fd");
  }
  if (options.measurement_mode == MeasurementMode::Time &&
      (options.memory_event_fd || options.memory_ack_fd)) {
    throw CliError("memory protocol descriptors are forbidden in time mode");
  }
  if (options.task == Task::PreparedQuery &&
      options.algorithm == AlgorithmKind::AS) {
    throw CliError("AS has no persistent prepared-query task");
  }
  if (options.samples >
      std::numeric_limits<std::size_t>::max() / sizeof(JoinPair)) {
    throw CliError("requested output buffer size overflows size_t");
  }
  return options;
}

void print_help() {
  std::cout
      << "Usage:\n"
      << "  anchor_bench run --workload PATH --algorithm "
         "ac|as|sweeprt|liftedrt\n"
      << "    --samples T --seed-hex 64HEX --measurement-mode time|memory\n"
      << "    --task oneshot|count-only|prepared-query --timeout-seconds N\n"
      << "    [--setup-timeout-seconds N]\n"
      << "    [--memory-event-fd FD --memory-ack-fd FD]\n"
      << "    [--dump-output PATH]\n\n"
      << "The run subcommand emits exactly one JSON object. Memory mode uses "
         "an external /proc VmRSS polling monitor and inherited event pipes.\n";
}

class MemoryProtocol {
 public:
  MemoryProtocol(int event_fd, int ack_fd)
      : event_fd_(event_fd), ack_fd_(ack_fd) {}
  MemoryProtocol(const MemoryProtocol&) = delete;
  MemoryProtocol& operator=(const MemoryProtocol&) = delete;
  ~MemoryProtocol() {
    if (event_fd_ >= 0) ::close(event_fd_);
    if (ack_fd_ >= 0) ::close(ack_fd_);
  }

  void notify(char event) {
    for (;;) {
      const ssize_t result = ::write(event_fd_, &event, 1);
      if (result == 1) break;
      if (result < 0 && errno == EINTR) continue;
      throw MemoryMeasurementError(
          "cannot write memory protocol event: " +
          std::string(std::strerror(errno)));
    }
    char acknowledgement = 0;
    for (;;) {
      const ssize_t result = ::read(ack_fd_, &acknowledgement, 1);
      if (result == 1) break;
      if (result < 0 && errno == EINTR) continue;
      if (result == 0) {
        throw MemoryMeasurementError(
            "memory monitor closed acknowledgement pipe");
      }
      throw MemoryMeasurementError(
          "cannot read memory protocol acknowledgement: " +
          std::string(std::strerror(errno)));
    }
  }

 private:
  int event_fd_{-1};
  int ack_fd_{-1};
};

class SetupIntervalTimeout {
 public:
  explicit SetupIntervalTimeout(std::uint64_t seconds) {
    if (seconds > static_cast<std::uint64_t>(
                      std::numeric_limits<decltype(itimerval{}.it_value.tv_sec)>::max())) {
      throw CliError("--setup-timeout-seconds exceeds setitimer range");
    }
    struct sigaction action {};
    action.sa_handler = setup_timeout_handler;
    if (::sigemptyset(&action.sa_mask) != 0 ||
        ::sigaction(SIGALRM, &action, nullptr) != 0) {
      throw std::runtime_error("cannot install setup SIGALRM handler: " +
                               std::string(std::strerror(errno)));
    }
    sigset_t alarm_set;
    if (::sigemptyset(&alarm_set) != 0 ||
        ::sigaddset(&alarm_set, SIGALRM) != 0 ||
        ::sigprocmask(SIG_UNBLOCK, &alarm_set, nullptr) != 0) {
      throw std::runtime_error("cannot unblock setup SIGALRM: " +
                               std::string(std::strerror(errno)));
    }
    itimerval timer{};
    timer.it_value.tv_sec =
        static_cast<decltype(timer.it_value.tv_sec)>(seconds);
    if (::setitimer(ITIMER_REAL, &timer, nullptr) != 0) {
      throw std::runtime_error("cannot arm setup timeout: " +
                               std::string(std::strerror(errno)));
    }
    armed_ = true;
  }
  SetupIntervalTimeout(const SetupIntervalTimeout&) = delete;
  SetupIntervalTimeout& operator=(const SetupIntervalTimeout&) = delete;
  ~SetupIntervalTimeout() { disarm_noexcept(); }
  void cancel() {
    if (!armed_) return;
    itimerval timer{};
    if (::setitimer(ITIMER_REAL, &timer, nullptr) != 0) {
      throw std::runtime_error("cannot cancel setup timeout: " +
                               std::string(std::strerror(errno)));
    }
    armed_ = false;
  }

 private:
  void disarm_noexcept() noexcept {
    if (!armed_) return;
    itimerval timer{};
    (void)::setitimer(ITIMER_REAL, &timer, nullptr);
  }
  bool armed_{};
};

class MainIntervalTimeout {
 public:
  explicit MainIntervalTimeout(std::uint64_t seconds) {
    if (seconds > static_cast<std::uint64_t>(
                      std::numeric_limits<decltype(itimerval{}.it_value.tv_sec)>::max())) {
      throw CliError("--timeout-seconds exceeds setitimer range");
    }
    struct sigaction action {};
    action.sa_handler = SIG_DFL;
    if (::sigemptyset(&action.sa_mask) != 0 ||
        ::sigaction(SIGALRM, &action, nullptr) != 0) {
      throw std::runtime_error("cannot install default SIGALRM disposition: " +
                               std::string(std::strerror(errno)));
    }
    sigset_t alarm_set;
    if (::sigemptyset(&alarm_set) != 0 ||
        ::sigaddset(&alarm_set, SIGALRM) != 0 ||
        ::sigprocmask(SIG_UNBLOCK, &alarm_set, nullptr) != 0) {
      throw std::runtime_error("cannot unblock SIGALRM: " +
                               std::string(std::strerror(errno)));
    }
    itimerval timer{};
    timer.it_value.tv_sec =
        static_cast<decltype(timer.it_value.tv_sec)>(seconds);
    if (::setitimer(ITIMER_REAL, &timer, nullptr) != 0) {
      throw std::runtime_error("cannot arm algorithm timeout: " +
                               std::string(std::strerror(errno)));
    }
    armed_ = true;
  }
  MainIntervalTimeout(const MainIntervalTimeout&) = delete;
  MainIntervalTimeout& operator=(const MainIntervalTimeout&) = delete;
  ~MainIntervalTimeout() { disarm_noexcept(); }
  void cancel() {
    if (!armed_) return;
    itimerval timer{};
    if (::setitimer(ITIMER_REAL, &timer, nullptr) != 0) {
      throw std::runtime_error("cannot cancel algorithm timeout: " +
                               std::string(std::strerror(errno)));
    }
    armed_ = false;
  }

 private:
  void disarm_noexcept() noexcept {
    if (!armed_) return;
    itimerval timer{};
    (void)::setitimer(ITIMER_REAL, &timer, nullptr);
  }
  bool armed_{};
};

timespec raw_now() {
  timespec value{};
  if (::clock_gettime(CLOCK_MONOTONIC_RAW, &value) != 0) {
    throw std::runtime_error("CLOCK_MONOTONIC_RAW failed: " +
                             std::string(std::strerror(errno)));
  }
  return value;
}

double elapsed_seconds(const timespec& begin, const timespec& end) {
  const auto seconds = static_cast<long double>(end.tv_sec - begin.tv_sec);
  const auto nanoseconds = static_cast<long double>(end.tv_nsec - begin.tv_nsec);
  return static_cast<double>(seconds + nanoseconds / 1'000'000'000.0L);
}

template <class T>
void pretouch_vector(const std::vector<T>& values, std::size_t page_size,
                     volatile std::uint8_t& sink) {
  const auto bytes = std::as_bytes(std::span<const T>(values));
  for (std::size_t offset = 0; offset < bytes.size(); offset += page_size) {
    sink = static_cast<std::uint8_t>(sink ^
                                    std::to_integer<std::uint8_t>(bytes[offset]));
  }
  if (!bytes.empty()) {
    sink = static_cast<std::uint8_t>(
        sink ^ std::to_integer<std::uint8_t>(bytes.back()));
  }
}

template <Coordinate Coord>
void pretouch_instance(const BoxJoinInstance<Coord>& instance) {
  const long page = ::sysconf(_SC_PAGESIZE);
  const std::size_t page_size =
      page > 0 ? static_cast<std::size_t>(page) : std::size_t{4096};
  volatile std::uint8_t sink = 0;
  for (const BoxSet<Coord>* side : {&instance.r, &instance.s}) {
    pretouch_vector(side->ids(), page_size, sink);
    pretouch_vector(side->flat_lower(), page_size, sink);
    pretouch_vector(side->flat_upper(), page_size, sink);
  }
  (void)sink;
}

void pretouch_output(std::span<JoinPair> output) {
  for (JoinPair& pair : output) pair = JoinPair{};
}

template <Coordinate Coord>
bool intersects(const BoxJoinInstance<Coord>& instance, std::size_t r,
                std::size_t s) {
  for (std::size_t axis = 0; axis < instance.dimensions(); ++axis) {
    if (!(instance.r.lower(r, axis) < instance.s.upper(s, axis) &&
          instance.s.lower(s, axis) < instance.r.upper(r, axis))) {
      return false;
    }
  }
  return true;
}

template <Coordinate Coord>
bool validate_membership(const BoxJoinInstance<Coord>& instance,
                         std::span<JoinPair> output) {
  using Index = std::uint32_t;
  if (instance.r.size() > std::numeric_limits<Index>::max() ||
      instance.s.size() > std::numeric_limits<Index>::max()) {
    throw std::overflow_error("membership index exceeds uint32");
  }
  // Hashing/dumping has already consumed the original IDs. Reuse each output
  // word for its resolved input index so post-run validation needs only one
  // compact uint32 input permutation at a time.
  std::vector<Index> index(instance.r.size());
  for (std::size_t i = 0; i < index.size(); ++i) {
    index[i] = static_cast<Index>(i);
  }
  const auto sort_by_ids = [](std::vector<Index>& permutation,
                              const std::vector<std::uint64_t>& ids) {
    std::sort(permutation.begin(), permutation.end(),
              [&](Index left, Index right) { return ids[left] < ids[right]; });
  };
  const auto locate = [](const std::vector<Index>& permutation,
                         const std::vector<std::uint64_t>& ids,
                         std::uint64_t id) -> std::optional<Index> {
    const auto found = std::lower_bound(
        permutation.begin(), permutation.end(), id,
        [&](Index position, std::uint64_t key) { return ids[position] < key; });
    if (found == permutation.end() || ids[*found] != id) return std::nullopt;
    return *found;
  };
  sort_by_ids(index, instance.r.ids());
  for (JoinPair& pair : output) {
    const auto resolved = locate(index, instance.r.ids(), pair.r);
    if (!resolved) return false;
    pair.r = *resolved;
  }
  std::vector<Index>().swap(index);
  index.resize(instance.s.size());
  for (std::size_t i = 0; i < index.size(); ++i) {
    index[i] = static_cast<Index>(i);
  }
  sort_by_ids(index, instance.s.ids());
  for (JoinPair& pair : output) {
    const auto resolved = locate(index, instance.s.ids(), pair.s);
    if (!resolved) return false;
    pair.s = *resolved;
    if (!intersects(instance, static_cast<std::size_t>(pair.r),
                    static_cast<std::size_t>(pair.s))) {
      return false;
    }
  }
  return true;
}

void write_all(int descriptor, std::span<const std::byte> bytes,
               std::string_view what) {
  while (!bytes.empty()) {
    const ssize_t result = ::write(descriptor, bytes.data(), bytes.size());
    if (result < 0 && errno == EINTR) continue;
    if (result <= 0) {
      throw std::runtime_error("cannot write " + std::string(what) + ": " +
                               std::strerror(errno));
    }
    bytes = bytes.subspan(static_cast<std::size_t>(result));
  }
}

std::string hash_and_maybe_dump_output(
    std::span<const JoinPair> output,
    const std::optional<std::filesystem::path>& dump_path) {
  int descriptor = -1;
  if (dump_path) {
    descriptor = ::open(dump_path->c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC,
                        0644);
    if (descriptor < 0) {
      throw std::runtime_error("cannot open output dump: " +
                               std::string(std::strerror(errno)));
    }
  }
  try {
    Sha256 digest;
    std::array<std::byte, 64 * 1024> buffer{};
    std::size_t used = 0;
    const auto flush = [&] {
      const auto bytes = std::span<const std::byte>(buffer).first(used);
      digest.update(bytes);
      if (descriptor >= 0) write_all(descriptor, bytes, "output dump");
      used = 0;
    };
    for (const JoinPair& pair : output) {
      if (used + 16 > buffer.size()) flush();
      for (std::size_t i = 0; i < 8; ++i) {
        buffer[used + i] =
            static_cast<std::byte>(pair.r >> static_cast<unsigned>(8 * i));
        buffer[used + 8 + i] =
            static_cast<std::byte>(pair.s >> static_cast<unsigned>(8 * i));
      }
      used += 16;
    }
    flush();
    if (descriptor >= 0) {
      if (::close(descriptor) != 0) {
        descriptor = -1;
        throw std::runtime_error("cannot close output dump: " +
                                 std::string(std::strerror(errno)));
      }
      descriptor = -1;
    }
    return hex_digest(digest.finish());
  } catch (...) {
    if (descriptor >= 0) ::close(descriptor);
    throw;
  }
}

std::size_t count_integer_bits(UInt128 value) {
  std::size_t bits = 0;
  while (value != 0) {
    ++bits;
    value >>= 1U;
  }
  return bits;
}

double realized_alpha(UInt128 count, std::size_t total_objects) {
  if (total_objects == 0) return 0.0;
  const std::uint64_t low = static_cast<std::uint64_t>(count);
  const std::uint64_t high = static_cast<std::uint64_t>(count >> 64U);
  const long double wide =
      std::ldexp(static_cast<long double>(high), 64) +
      static_cast<long double>(low);
  return static_cast<double>(wide / static_cast<long double>(total_objects));
}

std::uint64_t process_max_rss_bytes() {
  rusage usage{};
  if (::getrusage(RUSAGE_SELF, &usage) != 0) {
    throw MemoryMeasurementError("getrusage failed: " +
                                 std::string(std::strerror(errno)));
  }
  if (usage.ru_maxrss < 0 ||
      static_cast<std::uint64_t>(usage.ru_maxrss) >
          std::numeric_limits<std::uint64_t>::max() / 1024U) {
    throw MemoryMeasurementError("ru_maxrss is outside the byte range");
  }
  // Linux defines ru_maxrss in KiB.
  return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024U;
}

template <Coordinate Coord>
RunResult run_algorithm(const Options& options,
                        std::shared_ptr<const BoxJoinInstance<Coord>> input,
                        std::span<JoinPair> output,
                        MemoryProtocol* memory_protocol,
                        SetupIntervalTimeout* setup_timeout) {
  RunResult result;
  std::unique_ptr<ISampler<Coord>> sampler;
  const bool has_prepared_state = options.algorithm != AlgorithmKind::AS;
  SamplerOptions sampler_options = options.sampler_options;
  if (options.task == Task::CountOnly &&
      (options.algorithm == AlgorithmKind::LiftedRT ||
       options.algorithm == AlgorithmKind::SweepRT)) {
    sampler_options.build_sampling_index = false;
  }

  if (options.task == Task::PreparedQuery) {
    sampler = make_sampler<Coord>(options.algorithm, input, sampler_options);
    result.join_count = sampler->count();
    result.has_join_count = true;
    setup_timeout->cancel();
    MainIntervalTimeout timeout(options.timeout_seconds);
    const timespec begin = raw_now();
    result.sample_status = sampler->sample(output, 0);
    const timespec end = raw_now();
    timeout.cancel();
    result.primary_seconds = elapsed_seconds(begin, end);
  } else {
    setup_timeout->cancel();
    MainIntervalTimeout timeout(options.timeout_seconds);
    const timespec begin = raw_now();
    const timespec prepare_begin = begin;
    sampler = make_sampler<Coord>(options.algorithm, input, sampler_options);
    const timespec prepare_end = raw_now();
    if (options.task == Task::CountOnly) {
      result.join_count = sampler->count();
      result.has_join_count = true;
    } else {
      if (has_prepared_state && memory_protocol != nullptr) {
        memory_protocol->notify('P');
      }
      const timespec query_begin = raw_now();
      result.sample_status = sampler->sample(output, 0);
      // AS sample(t=0) intentionally has no sampling work; count inside this
      // same one-shot interval so every successful record still reports exact W.
      if (options.algorithm == AlgorithmKind::AS && output.empty()) {
        result.join_count = sampler->count();
      } else {
        result.join_count = sampler->diagnostics().join_count;
      }
      result.has_join_count = true;
      const timespec query_end = raw_now();
      if (has_prepared_state) {
        result.prepare_seconds = elapsed_seconds(prepare_begin, prepare_end);
        result.query_seconds = elapsed_seconds(query_begin, query_end);
      }
    }
    const timespec end = raw_now();
    timeout.cancel();
    result.primary_seconds = elapsed_seconds(begin, end);
  }
  result.diagnostics = sampler->diagnostics();
  return result;
}

std::string stage_timings_json(const Diagnostics& diagnostics) {
  std::string output{"["};
  for (std::size_t i = 0; i < diagnostics.timings.size(); ++i) {
    if (i != 0) output.push_back(',');
    JsonObject stage;
    stage.string("stage", diagnostics.timings[i].stage);
    stage.number("seconds", diagnostics.timings[i].seconds);
    std::string object = stage.finish();
    if (!object.empty() && object.back() == '\n') object.pop_back();
    output += object;
  }
  output.push_back(']');
  return output;
}

std::string diagnostics_json(const Diagnostics& diagnostics) {
  JsonObject object;
  object.string("algorithm", diagnostics.algorithm);
  object.number("original_r", diagnostics.original_r);
  object.number("original_s", diagnostics.original_s);
  object.number("filtered_r", diagnostics.filtered_r);
  object.number("filtered_s", diagnostics.filtered_s);
  object.string("join_count", to_string(diagnostics.join_count));
  object.number("persistent_bytes_estimate",
                diagnostics.persistent_bytes_estimate);
  std::string result = object.finish();
  if (!result.empty() && result.back() == '\n') result.pop_back();
  return result;
}

void add_algorithm_counters(JsonObject& json, const Diagnostics& diagnostics,
                            AlgorithmKind algorithm) {
  const AlgorithmCounters& c = diagnostics.counters;
  switch (algorithm) {
    case AlgorithmKind::AC:
      json.number("terminal_instance_count", c.terminal_instance_count);
      json.number("positive_atom_count", c.positive_atom_count);
      json.number("persistent_terminal_array_items",
                  c.persistent_terminal_array_items);
      json.number("alias_label_count", c.alias_label_count);
      break;
    case AlgorithmKind::AS:
      json.number("recursive_count_calls", c.recursive_count_calls);
      json.number("positive_quota_nodes", c.positive_quota_nodes);
      json.number("active_route_nodes", c.active_route_nodes);
      json.number("max_live_workspace_bytes", c.max_live_workspace_bytes);
      break;
    case AlgorithmKind::SweepRT:
      json.number("nonzero_event_blocks", c.nonzero_event_blocks);
      json.number("selected_event_blocks", c.selected_event_blocks);
      json.number("range_tree_nodes", c.range_tree_nodes);
      json.number("range_tree_point_references",
                  c.range_tree_point_references);
      break;
    case AlgorithmKind::LiftedRT:
      json.number("positive_degree_left_objects",
                  c.positive_degree_left_objects);
      json.number("selected_left_objects", c.selected_left_objects);
      json.number("canonical_block_queries", c.canonical_block_queries);
      json.number("range_tree_nodes", c.range_tree_nodes);
      json.number("range_tree_point_references",
                  c.range_tree_point_references);
      break;
  }
}

std::string base_error_json(std::string_view status, std::string_view stage,
                            std::string_view message) {
  JsonObject json;
  json.string("schema_version", kSchema);
  json.string("status", status);
  json.string("failure_stage", stage);
  json.string("error_message", message);
  return json.finish();
}

int run(const Options& options) {
  SetupIntervalTimeout setup_timeout(options.setup_timeout_seconds);

  std::unique_ptr<MemoryProtocol> memory_protocol;
  if (options.measurement_mode == MeasurementMode::Memory) {
    memory_protocol = std::make_unique<MemoryProtocol>(
        *options.memory_event_fd, *options.memory_ack_fd);
  }

  const LoadedWorkload workload = load_workload(options.workload);
  std::visit(
      [](const auto& instance) { pretouch_instance(*instance); },
      workload.instance);
  if (memory_protocol) memory_protocol->notify('I');

  const std::size_t output_size =
      options.task == Task::CountOnly ? 0 : options.samples;
  std::vector<JoinPair> output(output_size);
  pretouch_output(output);
  if (memory_protocol) memory_protocol->notify('B');

  RunResult result = std::visit(
      [&](const auto& instance) {
        return run_algorithm(options, instance, std::span<JoinPair>(output),
                             memory_protocol.get(), &setup_timeout);
      },
      workload.instance);
  if (memory_protocol) memory_protocol->notify('D');

  // Post-processing, membership validation, hashing, and optional result dump
  // have a fresh setup budget and are never charged to OneShotTime.
  SetupIntervalTimeout postprocess_timeout(options.setup_timeout_seconds);

  if (result.sample_status == SampleStatus::EmptyInstance &&
      output_size > 0) {
    result.status = "EMPTY-JOIN";
    result.output_length = 0;
    result.membership_checked = true;
    result.output_sha256 = hash_and_maybe_dump_output({}, options.dump_output);
  } else if (options.task == Task::CountOnly) {
    result.output_length = 0;
    result.output_sha256 = hash_and_maybe_dump_output({}, options.dump_output);
  } else {
    result.output_length = output.size();
    result.output_sha256 =
        hash_and_maybe_dump_output(output, options.dump_output);
    const bool valid = std::visit(
        [&](const auto& instance) {
          return validate_membership(*instance, std::span<JoinPair>(output));
        },
        workload.instance);
    result.membership_checked = true;
    if (!valid) {
      result.status = "MEMBERSHIP-MISMATCH";
      result.failure_stage = "post-run-membership";
      result.error_message = "sampled pair is absent or does not strictly overlap";
    }
  }

  if (result.status == "OK" && result.has_join_count &&
      result.join_count == 0 && output_size > 0 &&
      options.task != Task::CountOnly) {
    result.status = "EMPTY-JOIN";
    result.output_length = 0;
  }

  std::optional<std::uint64_t> max_rss;
  if (memory_protocol) max_rss = process_max_rss_bytes();

  JsonObject json;
  json.string("schema_version", kSchema);
  json.string("status", result.status);
  json.string("algorithm", options.algorithm_name);
  json.string("measurement_mode",
              measurement_mode_name(options.measurement_mode));
  json.string("task", task_name(options.task));
  json.number("timeout_seconds", options.timeout_seconds);
  json.number("setup_timeout_seconds", options.setup_timeout_seconds);
  json.number("t", options.samples);
  json.number("N_total", workload.n_r + workload.n_s);
  json.number("n_R", workload.n_r);
  json.number("n_S", workload.n_s);
  json.number("d", workload.dimension);
  json.string("endpoint_type", endpoint_type_name(workload.endpoint_type));
  json.string("payload_sha256", hex_digest(workload.payload_sha256));
  json.string("seed_to_state_version", "anchor-seed-words-be-v1");
  if (result.has_join_count) {
    json.string("W", to_string(result.join_count));
    json.number("alpha_realized",
                realized_alpha(result.join_count, workload.n_r + workload.n_s));
    json.number("count_integer_bits", count_integer_bits(result.join_count));
  }
  switch (options.task) {
    case Task::OneShot:
      json.number("OneShotTime", result.primary_seconds);
      if (result.prepare_seconds) {
        json.number("PrepareTime", *result.prepare_seconds);
      }
      if (result.query_seconds) json.number("QueryTime", *result.query_seconds);
      break;
    case Task::CountOnly:
      json.number("CountOnlyTime", result.primary_seconds);
      break;
    case Task::PreparedQuery:
      json.number("PreparedQueryTime", result.primary_seconds);
      break;
  }
  if (options.measurement_mode == MeasurementMode::Memory) {
    json.boolean("timing_is_diagnostic", true);
    json.number("input_payload_bytes", workload.logical_payload_bytes);
    json.number("output_buffer_bytes",
                static_cast<std::uint64_t>(output_size * sizeof(JoinPair)));
    json.boolean("memory_protocol_handshake", true);
    json.number("ProcessMaxRSS", *max_rss);
  }
  json.number("output_length", result.output_length);
  json.string("output_sha256", result.output_sha256);
  json.boolean("membership_checked", result.membership_checked);
  json.boolean("count_consistency_checked", false);
  if (!result.failure_stage.empty()) {
    json.string("failure_stage", result.failure_stage);
  }
  if (!result.error_message.empty()) {
    json.string("error_message", result.error_message);
  }
  json.raw("stage_timings", stage_timings_json(result.diagnostics));
  json.raw("diagnostics", diagnostics_json(result.diagnostics));
  add_algorithm_counters(json, result.diagnostics, options.algorithm);
  postprocess_timeout.cancel();
  std::cout << json.finish();
  std::cout.flush();
  return result.status == "OK" || result.status == "EMPTY-JOIN" ? 0 : 3;
}

}  // namespace
}  // namespace anchor::app

int main(int argc, char** argv) {
  using namespace anchor::app;
  if (argc == 2 &&
      (std::string_view(argv[1]) == "--help" ||
       std::string_view(argv[1]) == "-h")) {
    print_help();
    return 0;
  }
  try {
    const Options options = parse_options(argc, argv);
    return run(options);
  } catch (const CliError& error) {
    if (std::string_view(error.what()) == "__HELP__") {
      print_help();
      return 0;
    }
    std::cout << base_error_json("BENCHMARK-ERROR", "argument-parsing",
                                 error.what());
    return 2;
  } catch (const MemoryMeasurementError& error) {
    std::cout << base_error_json("MEMORY-MEASUREMENT-UNAVAILABLE",
                                 "memory-measurement", error.what());
    return 78;
  } catch (const NumericDegeneracyError& error) {
    std::cout << base_error_json("NUMERIC-DEGENERACY", "workload-validation",
                                 error.what());
    return 2;
  } catch (const WorkloadFormatError& error) {
    std::cout << base_error_json("BENCHMARK-ERROR", "workload-loading",
                                 error.what());
    return 2;
  } catch (const std::overflow_error& error) {
    std::cout << base_error_json("INTEGER-OVERFLOW", "algorithm", error.what());
    return 4;
  } catch (const std::bad_alloc& error) {
    std::cout << base_error_json("OOM", "allocation", error.what());
    return 137;
  } catch (const std::exception& error) {
    std::cout << base_error_json("BENCHMARK-ERROR", "benchmark", error.what());
    return 2;
  }
}
