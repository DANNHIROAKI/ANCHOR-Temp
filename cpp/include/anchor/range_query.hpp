#pragma once

#include <optional>

#include "anchor/types.hpp"

namespace anchor {

template <Coordinate Coord>
struct AxisRange {
  std::optional<Coord> lower;
  std::optional<Coord> upper;
  bool lower_strict{true};
  bool upper_strict{true};

  static AxisRange less_than(Coord value) {
    AxisRange query;
    query.upper = value;
    query.upper_strict = true;
    return query;
  }

  static AxisRange greater_than(Coord value) {
    AxisRange query;
    query.lower = value;
    query.lower_strict = true;
    return query;
  }
};

}  // namespace anchor
