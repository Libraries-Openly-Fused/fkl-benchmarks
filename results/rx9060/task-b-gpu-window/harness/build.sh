#!/usr/bin/env bash
# build.sh <FKL tree> <output binary>: builds harness/filters_before_after.hip against the FKL tree's include/, with the flags of FKL's HIP tests
# (tests/image_processing at 24478f0: -O3 -DNDEBUG -std=c++17 --offload-arch=gfx1200, -Werror=c++20-extensions
# -pedantic-errors -Wno-gnu-anonymous-struct -ftemplate-depth=1000) and this repository's common/ (hip_timing.h, hip_wallclock.h).
# The FKL tree must be a clean git checkout; its HEAD is compiled into the binary (FILTERS_FKL_COMMIT) and printed by every run.
# Compile and link only; nothing runs.
set -eu
FKL=$(cd "${1:?usage: build.sh <FKL tree> <output binary>}" && pwd)
OUT=${2:?usage: build.sh <FKL tree> <output binary>}
HERE=$(cd "$(dirname "$0")" && pwd)
FKLB=$(cd "$HERE/../../../.." && pwd)
ROCM=/opt/rocm/core-10.1
[ -z "$(cd "$FKL" && git status --porcelain --untracked-files=no)" ] || { echo "$FKL has changes: not building"; exit 1; }
COMMIT=$(cd "$FKL" && git rev-parse HEAD)
mkdir -p "$(dirname "$OUT")"
( ulimit -v 16000000
  export TMPDIR=/home/cudeiro/rocm-fkl-benchmarks/tmp/sys
  "$ROCM/bin/amdclang++" -x hip --offload-arch=gfx1200 -O3 -DNDEBUG -std=c++17 -Werror=c++20-extensions -pedantic-errors \
      -Wno-gnu-anonymous-struct -ftemplate-depth=1000 "-DFILTERS_FKL_COMMIT=\"$COMMIT\"" -I"$FKLB" -I"$FKL/include" \
      "$HERE/filters_before_after.hip" -o "$OUT" --hip-link )
echo "built $OUT against FKL $COMMIT: sha256 $(sha256sum "$OUT" | cut -d' ' -f1)"
