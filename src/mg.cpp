#include "mg.hpp"

namespace kokkos_hpcg {

void restrict_injection(const SparseMatrix& fine, const SparseMatrix& coarse,
                        const Kokkos::View<double*>& fine_residual,
                        const Kokkos::View<double*>& coarse_rhs) {
  const int cnx = coarse.nx;
  const int cny = coarse.ny;
  const int fnx = fine.nx;
  const int fny = fine.ny;
  Kokkos::parallel_for(
      "hpcg_mg_restrict", Kokkos::RangePolicy<>(0, coarse.nrows),
      KOKKOS_LAMBDA(const int coarse_row) {
        const int ci = coarse_row % cnx;
        const int cj = (coarse_row / cnx) % cny;
        const int ck = coarse_row / (cnx * cny);
        const int fine_row = (2 * ci) + fnx * ((2 * cj) + fny * (2 * ck));
        coarse_rhs(coarse_row) = fine_residual(fine_row);
      });
}

void prolongate_and_add(const SparseMatrix& fine, const SparseMatrix& coarse,
                        const Kokkos::View<double*>& coarse_x,
                        const Kokkos::View<double*>& fine_x) {
  const int cnx = coarse.nx;
  const int cny = coarse.ny;
  const int fnx = fine.nx;
  const int fny = fine.ny;
  Kokkos::parallel_for(
      "hpcg_mg_prolongate", Kokkos::RangePolicy<>(0, coarse.nrows),
      KOKKOS_LAMBDA(const int coarse_row) {
        const int ci = coarse_row % cnx;
        const int cj = (coarse_row / cnx) % cny;
        const int ck = coarse_row / (cnx * cny);
        const int fine_row = (2 * ci) + fnx * ((2 * cj) + fny * (2 * ck));
        fine_x(fine_row) += coarse_x(coarse_row);
      });
}

MultigridHierarchy build_hierarchy(const SparseMatrix& fine) {
  MultigridHierarchy hierarchy;
  hierarchy.levels.push_back(fine);
  int nx = fine.nx;
  int ny = fine.ny;
  int nz = fine.nz;
  while (nx % 2 == 0 && ny % 2 == 0 && nz % 2 == 0 && nx > 2 && ny > 2 &&
         nz > 2) {
    nx /= 2;
    ny /= 2;
    nz /= 2;
    hierarchy.levels.push_back(generate_problem(nx, ny, nz));
  }
  return hierarchy;
}

void mg_vcycle(const MultigridHierarchy& hierarchy, std::size_t level,
               const Kokkos::View<double*>& b, const Kokkos::View<double*>& x) {
  const SparseMatrix& A = hierarchy.levels[level];
  symgs(A, b, x);
  if (level + 1 == hierarchy.levels.size()) {
    symgs(A, b, x);
    return;
  }

  Kokkos::View<double*> Ax("mg_Ax", A.nrows);
  Kokkos::View<double*> residual("mg_residual", A.nrows);
  spmv(A, x, Ax);
  waxpby(1.0, b, -1.0, Ax, residual);

  const SparseMatrix& coarse = hierarchy.levels[level + 1];
  Kokkos::View<double*> coarse_b("mg_coarse_b", coarse.nrows);
  Kokkos::View<double*> coarse_x("mg_coarse_x", coarse.nrows);
  Kokkos::deep_copy(coarse_x, 0.0);
  restrict_injection(A, coarse, residual, coarse_b);
  mg_vcycle(hierarchy, level + 1, coarse_b, coarse_x);
  prolongate_and_add(A, coarse, coarse_x, x);
  symgs(A, b, x);
}

}  // namespace kokkos_hpcg
