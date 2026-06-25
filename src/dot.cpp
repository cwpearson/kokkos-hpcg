#include "common.hpp"

#include <cmath>

namespace kokkos_hpcg {

double dot(const Kokkos::View<double*>& x, const Kokkos::View<double*>& y) {
  validate_matching_extents(x, y, "dot");
  const int n = static_cast<int>(x.extent(0));
  double result = 0.0;
  Kokkos::parallel_reduce(
      "hpcg_dot", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const int i, double& local) { local += x(i) * y(i); },
      result);
  return result;
}

double norm2(const Kokkos::View<double*>& x) {
  return std::sqrt(dot(x, x));
}

}  // namespace kokkos_hpcg
