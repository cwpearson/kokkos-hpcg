#include "kokkos_hpcg.hpp"

#include <stdexcept>

namespace kokkos_hpcg {

void symgs(const SparseMatrix& A, const Kokkos::View<double*>& b,
           const Kokkos::View<double*>& x) {
  if (static_cast<int>(b.extent(0)) != A.nrows ||
      static_cast<int>(x.extent(0)) != A.nrows) {
    throw std::invalid_argument("symgs vector extent does not match matrix");
  }

  auto row_length_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.row_length);
  auto col_idx_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.col_idx);
  auto values_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.values);
  auto diag_slot_h =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A.diag_slot);
  auto b_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), b);
  auto x_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x);

  auto sweep = [&](int first, int last, int step) {
    for (int row = first; row != last; row += step) {
      double sum = b_h(row);
      for (int slot = 0; slot < row_length_h(row); ++slot) {
        if (slot != diag_slot_h(row)) {
          sum -= values_h(row, slot) * x_h(col_idx_h(row, slot));
        }
      }
      x_h(row) = sum / values_h(row, diag_slot_h(row));
    }
  };

  sweep(0, A.nrows, 1);
  sweep(A.nrows - 1, -1, -1);
  Kokkos::deep_copy(x, x_h);
}

}  // namespace kokkos_hpcg
