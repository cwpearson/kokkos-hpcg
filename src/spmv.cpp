#include "spmv.hpp"
#include "common.hpp"

#include <stdexcept>

namespace kokkos_hpcg {

void spmv(const SparseMatrix& A, const Kokkos::View<double*>& x,
          const Kokkos::View<double*>& y) {
  validate_matching_extents(x, y, "spmv");
  if (static_cast<int>(x.extent(0)) != A.nrows) {
    throw std::invalid_argument("spmv vector extent does not match matrix");
  }

  auto row_length = A.row_length;
  auto col_idx = A.col_idx;
  auto values = A.values;
  Kokkos::parallel_for(
      "hpcg_spmv", Kokkos::RangePolicy<>(0, A.nrows),
      KOKKOS_LAMBDA(const int row) {
        double sum = 0.0;
        for (int slot = 0; slot < row_length(row); ++slot) {
          sum += values(row, slot) * x(col_idx(row, slot));
        }
        y(row) = sum;
      });
}

}  // namespace kokkos_hpcg
