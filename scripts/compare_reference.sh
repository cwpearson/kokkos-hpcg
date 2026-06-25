#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: compare_reference.sh --reference reference.log --candidate kokkos.log [--rtol value] [--atol value] [--scaled-rtol value] [--scaled-atol value]
EOF
}

reference=""
candidate=""
rtol="1.0e-6"
atol="1.0e-10"
scaled_rtol="1.0e-2"
scaled_atol="1.0e-10"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reference)
      reference="${2:-}"
      shift 2
      ;;
    --candidate)
      candidate="${2:-}"
      shift 2
      ;;
    --rtol)
      rtol="${2:-}"
      shift 2
      ;;
    --atol)
      atol="${2:-}"
      shift 2
      ;;
    --scaled-rtol)
      scaled_rtol="${2:-}"
      shift 2
      ;;
    --scaled-atol)
      scaled_atol="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "$reference" || -z "$candidate" ]]; then
  usage
  exit 2
fi

if [[ ! -r "$reference" ]]; then
  echo "reference log is not readable: $reference" >&2
  exit 2
fi

if [[ ! -r "$candidate" ]]; then
  echo "candidate log is not readable: $candidate" >&2
  exit 2
fi

extract_value() {
  local file="$1"
  shift
  awk -v keys="$*" '
    BEGIN {
      n = split(keys, wanted, "\034")
      for (i = 1; i <= n; ++i) {
        lower_wanted[i] = tolower(wanted[i])
      }
    }
    {
      line = $0
      eq = index(line, "=")
      colon = index(line, ":")
      if (eq > 0) {
        key = substr(line, 1, eq - 1)
        value = substr(line, eq + 1)
      } else if (colon > 0) {
        key = substr(line, 1, colon - 1)
        value = substr(line, colon + 1)
      } else {
        next
      }
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", key)
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      lower_key = tolower(key)
      for (i = 1; i <= n; ++i) {
        if (lower_key == lower_wanted[i]) {
          split(value, fields, /[[:space:]]+/)
          print fields[1]
          found = 1
          exit
        }
      }
    }
    END { exit(found ? 0 : 1) }
  ' "$file"
}

validation_passed() {
  local file="$1"
  awk '
    {
      line = tolower($0)
      if ((line ~ /valid/ || line ~ /validation/) &&
          (line ~ /pass/ || line ~ /valid/ || line ~ /true/ || line ~ /(^|[^0-9])1([^0-9]|$)/)) {
        found = 1
      }
      if (line ~ /result/ && line ~ /passed/) {
        found = 1
      }
    }
    END { exit(found ? 0 : 1) }
  ' "$file"
}

if ! validation_passed "$reference"; then
  echo "reference validation did not pass or could not be found" >&2
  exit 1
fi

if ! validation_passed "$candidate"; then
  echo "candidate validation did not pass or could not be found" >&2
  exit 1
fi

reference_rows="$(extract_value "$reference" \
  "Linear System Information::Number of Equations"$'\034'"HPCG.problem.rows")" || {
  echo "missing reference row count" >&2
  exit 1
}
candidate_rows="$(extract_value "$candidate" "HPCG.problem.rows")" || {
  echo "missing candidate row count" >&2
  exit 1
}

if [[ "$reference_rows" != "$candidate_rows" ]]; then
  echo "row-count mismatch: reference=${reference_rows} candidate=${candidate_rows}" >&2
  exit 1
fi

reference_nnz="$(extract_value "$reference" \
  "Linear System Information::Number of Nonzero Terms"$'\034'"HPCG.problem.nonzeros")" || {
  echo "missing reference nonzero count" >&2
  exit 1
}
candidate_nnz="$(extract_value "$candidate" "HPCG.problem.nonzeros")" || {
  echo "missing candidate nonzero count" >&2
  exit 1
}

if [[ "$reference_nnz" != "$candidate_nnz" ]]; then
  echo "nonzero-count mismatch: reference=${reference_nnz} candidate=${candidate_nnz}" >&2
  exit 1
fi

candidate_residual="$(extract_value "$candidate" \
  "HPCG.result.final_residual"$'\034'"Result.final_residual"$'\034'"final_residual"$'\034'"Final Residual")" || {
  echo "missing candidate final residual" >&2
  exit 1
}
candidate_relative_residual="$(extract_value "$candidate" \
  "HPCG.result.final_relative_residual"$'\034'"Result.final_relative_residual"$'\034'"final_relative_residual")" || {
  echo "missing candidate final relative residual" >&2
  exit 1
}

awk -v final="$candidate_residual" \
    -v relative="$candidate_relative_residual" \
    -v rtol="$rtol" \
    -v atol="$atol" '
  BEGIN {
    if (final + 0 != final || relative + 0 != relative) {
      print "candidate residual is not numeric" > "/dev/stderr"
      exit 1
    }
    if (final != final || relative != relative) {
      print "candidate residual is not finite" > "/dev/stderr"
      exit 1
    }
    if (relative > rtol && final > atol) {
      printf "candidate did not converge within tolerance: final=%.17g relative=%.17g rtol=%.3e atol=%.3e\n",
             final, relative, rtol, atol > "/dev/stderr"
      exit 1
    }
  }
'

reference_scaled_mean="$(extract_value "$reference" \
  "Reproducibility Information::Scaled residual mean")" || {
  echo "missing reference reproducibility scaled residual mean" >&2
  exit 1
}
candidate_scaled_mean="$(extract_value "$candidate" \
  "Reproducibility Information::Scaled residual mean")" || {
  echo "missing candidate reproducibility scaled residual mean" >&2
  exit 1
}

awk -v reference="$reference_scaled_mean" \
    -v candidate="$candidate_scaled_mean" \
    -v rtol="$scaled_rtol" \
    -v atol="$scaled_atol" '
  BEGIN {
    if (reference + 0 != reference || candidate + 0 != candidate) {
      print "reproducibility scaled residual mean is not numeric" > "/dev/stderr"
      exit 1
    }
    if (reference != reference || candidate != candidate) {
      print "reproducibility scaled residual mean is not finite" > "/dev/stderr"
      exit 1
    }
    diff = candidate - reference
    if (diff < 0.0) {
      diff = -diff
    }
    abs_reference = reference < 0.0 ? -reference : reference
    tolerance = atol + rtol * abs_reference
    if (diff > tolerance) {
      printf "reproducibility scaled residual mismatch: reference=%.17g candidate=%.17g difference=%.3e tolerance=%.3e\n",
             reference, candidate, diff, tolerance > "/dev/stderr"
      exit 1
    }
  }
'

printf 'comparison passed: rows=%s nnz=%s candidate_final=%.17g candidate_relative=%.17g reference_scaled_mean=%.17g candidate_scaled_mean=%.17g\n' \
  "$candidate_rows" "$candidate_nnz" "$candidate_residual" \
  "$candidate_relative_residual" "$reference_scaled_mean" \
  "$candidate_scaled_mean"
