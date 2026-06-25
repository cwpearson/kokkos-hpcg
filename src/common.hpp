#pragma once

#include "kokkos_hpcg.hpp"

#include <stdexcept>
#include <string>

namespace kokkos_hpcg {

inline void validate_matching_extents(const Kokkos::View<double*>& a,
                                      const Kokkos::View<double*>& b,
                                      const char* op) {
  if (a.extent(0) != b.extent(0)) {
    throw std::invalid_argument(std::string(op) + " vector extents differ");
  }
}

}  // namespace kokkos_hpcg
