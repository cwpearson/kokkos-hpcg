#pragma once

#include "kokkos_hpcg.hpp"

#include <cstddef>
#include <vector>

namespace kokkos_hpcg {

struct MultigridHierarchy {
  std::vector<SparseMatrix> levels;
};

MultigridHierarchy build_hierarchy(const SparseMatrix& fine);
void restrict_injection(const SparseMatrix& fine, const SparseMatrix& coarse,
                        const Kokkos::View<double*>& fine_residual,
                        const Kokkos::View<double*>& coarse_rhs);
void prolongate_and_add(const SparseMatrix& fine, const SparseMatrix& coarse,
                        const Kokkos::View<double*>& coarse_x,
                        const Kokkos::View<double*>& fine_x);
void mg_vcycle(const MultigridHierarchy& hierarchy, std::size_t level,
               const Kokkos::View<double*>& b, const Kokkos::View<double*>& x);

}  // namespace kokkos_hpcg
