#pragma once

#include <cmath>
#include <iostream>
#include <stdexcept>

inline void require_near(double actual, double expected, double tolerance,
                         const char *message) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << message << ": actual=" << actual << " expected=" << expected
              << '\n';
    throw std::runtime_error(message);
  }
}
