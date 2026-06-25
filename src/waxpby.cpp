#include "common.hpp"

namespace kokkos_hpcg {

void waxpby(double alpha, const Kokkos::View<double*>& x, double beta,
            const Kokkos::View<double*>& y, const Kokkos::View<double*>& w) {
  validate_matching_extents(x, y, "waxpby");
  validate_matching_extents(x, w, "waxpby");
  const int n = static_cast<int>(x.extent(0));
  Kokkos::parallel_for(
      "hpcg_waxpby", Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(const int i) { w(i) = alpha * x(i) + beta * y(i); });
}

}  // namespace kokkos_hpcg
