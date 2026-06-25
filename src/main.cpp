#include "kokkos_hpcg.hpp"

#include <Kokkos_Core.hpp>

#include <exception>
#include <iostream>

int main(int argc, char** argv) {
  Kokkos::initialize(argc, argv);
  try {
    const auto options = kokkos_hpcg::parse_options(argc, argv);
    const auto result = kokkos_hpcg::run(options);
    kokkos_hpcg::print_result(std::cout, result);
    Kokkos::finalize();
    return result.valid ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "kokkos-hpcg: " << error.what() << '\n';
    Kokkos::finalize();
    return 1;
  }
}
