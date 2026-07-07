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

void test_prolongate_and_add() {
  const auto fine =
      kokkos_hpcg::generate_problem(fine_nx, fine_ny, fine_nz);
  const auto coarse =
      kokkos_hpcg::generate_problem(coarse_nx, coarse_ny, coarse_nz);
  Kokkos::View<double*> coarse_x("coarse_x", coarse.nrows);
  Kokkos::View<double*> fine_x("fine_x", fine.nrows);

  auto coarse_host = Kokkos::create_mirror_view(coarse_x);
  for (int row = 0; row < coarse.nrows; ++row) {
    coarse_host(row) = 1000.0 + static_cast<double>(row);
  }
  Kokkos::deep_copy(coarse_x, coarse_host);
  Kokkos::deep_copy(fine_x, 7.0);

  kokkos_hpcg::prolongate_and_add(fine, coarse, coarse_x, fine_x);

  const auto fine_host =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), fine_x);
  for (int coarse_row = 0; coarse_row < coarse.nrows; ++coarse_row) {
    const int fine_row = fine_row_for_coarse_row(coarse_row);
    require_near(fine_host(fine_row),
                 7.0 + 1000.0 + static_cast<double>(coarse_row), 0.0,
                 "prolongation adds at the matching even-grid point");
  }

  for (int k = 0; k < fine_nz; ++k) {
    for (int j = 0; j < fine_ny; ++j) {
      for (int i = 0; i < fine_nx; ++i) {
        if (i % 2 == 0 && j % 2 == 0 && k % 2 == 0) {
          continue;
        }
        const int row = kokkos_hpcg::linear_index(i, j, k, fine_nx, fine_ny);
        require_near(fine_host(row), 7.0, 0.0,
                     "prolongation leaves non-coarse grid points unchanged");
      }
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  try {
    test_prolongate_and_add();
    Kokkos::finalize();
    std::cout << "test passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "test failure: " << error.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
}
