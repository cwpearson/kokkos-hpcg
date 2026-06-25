#include "kokkos_hpcg.hpp"

#include <cstdlib>
#include <stdexcept>
#include <string>

namespace kokkos_hpcg {
namespace {

int parse_positive_int(const char* text, const char* name) {
  const int value = std::atoi(text);
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  return value;
}

}  // namespace

Options parse_options(int argc, char** argv) {
  Options options;
  if (argc == 2 || argc == 3) {
    throw std::invalid_argument(
        "usage: kokkos-hpcg [nx ny nz [runtime_seconds]]");
  }
  if (argc >= 4) {
    options.nx = parse_positive_int(argv[1], "nx");
    options.ny = parse_positive_int(argv[2], "ny");
    options.nz = parse_positive_int(argv[3], "nz");
  }
  if (argc >= 5) {
    options.runtime_seconds = std::atof(argv[4]);
    if (options.runtime_seconds < 0.0) {
      throw std::invalid_argument("runtime_seconds must be nonnegative");
    }
  }
  if (argc > 5) {
    throw std::invalid_argument(
        "usage: kokkos-hpcg [nx ny nz [runtime_seconds]]");
  }
  return options;
}

}  // namespace kokkos_hpcg
