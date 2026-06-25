# kokkos-hpcg

A small, single-process Kokkos research implementation of the HPCG 3.1 operator and solver.
This is not intended to represent a high-performance HPCG implementation.

## Build

Install Kokkos 5 separately, then point CMake at the install prefix via `Kokkos_ROOT`.

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DKokkos_ROOT=/path/to/kokkos/install
cmake --build build -j
ctest --test-dir build --output-on-failure
```

## Run

```bash
./build/kokkos-hpcg 16 16 16
```

The executable accepts:

```text
kokkos-hpcg [nx ny nz [runtime_seconds]]
```

When dimensions are omitted it runs a small `16 16 16` problem.
Output is a line-oriented key/value format:

```text
HPCG.problem.nx=16
HPCG.result.final_residual=...
HPCG.result.validation=PASSED
```

## Compare With HPCG 3.1 Reference Code

Download the official HPCG 3.1 reference package from:

https://www.hpcg-benchmark.org/software/view.html%3Fid=262.html

The reference page identifies the 2019-03-28 HPCG 3.1 tarball and publishes this SHA256:

```text
33a434e716b79e59e745f77ff72639c32623e7f928eeb7977655ffcaade0f4a4
```

A typical comparison flow is:

```bash
curl -L -o hpcg-3.1.tar.gz \
  https://www.hpcg-benchmark.org/downloads/hpcg-3.1.tar.gz
echo "33a434e716b79e59e745f77ff72639c32623e7f928eeb7977655ffcaade0f4a4  hpcg-3.1.tar.gz" | sha256sum -c -
tar xf hpcg-3.1.tar.gz
```

Build the reference code using one of its serial Linux setup targets, then run it on the same single-rank dimensions as this implementation.
Save both logs and compare them with:

```bash
bash scripts/compare_reference.sh \
  --reference reference.log \
  --candidate kokkos.log
```

The comparison checks that this implementation reports validation success and that both runs use the same matrix size and nonzero count.
It then checks that the Kokkos run converged below the requested residual tolerance.
The reference HPCG report exposes an official reproducibility scaled residual, which is not the same quantity as this implementation's single-solve final residual.
This implementation reports the same reproducibility key names after a fixed `max_iterations` CG pass, and the comparison script checks the scaled residual means with a separate `--scaled-rtol`/`--scaled-atol` tolerance.

The reference code accepts smaller input dimensions but reports/runs at least `16 16 16`.

## Limitations and Simplifications

- no MPI and no halo exchange
