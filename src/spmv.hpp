#pragma once

#include <Kokkos_Core.hpp>

namespace kokkos_hpcg {

struct SparseMatrix;

void spmv(const SparseMatrix &A, const Kokkos::View<double *> &x,
          const Kokkos::View<double *> &y);

} // namespace kokkos_hpcg
