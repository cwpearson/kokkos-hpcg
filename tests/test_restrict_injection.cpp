#include "mg.hpp"
#include "test_common.hpp"

#include <Kokkos_Core.hpp>

#include <iostream>
#include <stdexcept>

namespace {

constexpr int fine_nx = 4;
constexpr int fine_ny = 6;
constexpr int fine_nz = 8;
constexpr int coarse_nx = fine_nx / 2;
constexpr int coarse_ny = fine_ny / 2;
constexpr int coarse_nz = fine_nz / 2;

int fine_row_for_coarse_row(int coarse_row) {
  const int ci = coarse_row % coarse_nx;
  const int cj = (coarse_row / coarse_nx) % coarse_ny;
  const int ck = coarse_row / (coarse_nx * coarse_ny);
  return kokkos_hpcg::linear_index(2 * ci, 2 * cj, 2 * ck, fine_nx,
                                   fine_ny);
}

void test_restrict_injection() {
  const auto fine =
      kokkos_hpcg::generate_problem(fine_nx, fine_ny, fine_nz);
  const auto coarse =
      kokkos_hpcg::generate_problem(coarse_nx, coarse_ny, coarse_nz);
  Kokkos::View<double*> fine_residual("fine_residual", fine.nrows);
  Kokkos::View<double*> coarse_rhs("coarse_rhs", coarse.nrows);

  auto fine_host = Kokkos::create_mirror_view(fine_residual);
  for (int row = 0; row < fine.nrows; ++row) {
    fine_host(row) = 0.25 + static_cast<double>(row);
  }
  Kokkos::deep_copy(fine_residual, fine_host);

  kokkos_hpcg::restrict_injection(fine, coarse, fine_residual, coarse_rhs);

  const auto coarse_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), coarse_rhs);
  for (int coarse_row = 0; coarse_row < coarse.nrows; ++coarse_row) {
    const double expected =
        0.25 + static_cast<double>(fine_row_for_coarse_row(coarse_row));
    require_near(coarse_host(coarse_row), expected, 0.0,
                 "restriction injects the matching even-grid value");
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  try {
    test_restrict_injection();
    Kokkos::finalize();
    std::cout << "test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
}
