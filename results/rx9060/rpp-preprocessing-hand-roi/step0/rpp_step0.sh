#!/usr/bin/env bash
# rpp_step0.sh <name> <source worktree> <build dir>: the step-0 facts of one RPP binary (nothing is run)
set -u
NAME=$1; SRC=$2; B=$3; BIN=$B/bin/benchmark_rpp_preprocessing_hip
echo "== $NAME"
echo "source worktree: $SRC"
echo "  repo HEAD $(cd $SRC && git rev-parse HEAD), tracked changes: $(cd $SRC && git status --porcelain --untracked-files=no | wc -l)"
echo "  external/fkl HEAD $(cd $SRC/external/fkl && git rev-parse HEAD), tracked changes: $(cd $SRC/external/fkl && git status --porcelain --untracked-files=no | wc -l)"
echo "build info (generated/fklb_build_info.h, printed in the CSV header):"
grep -E '#define' $B/generated/fklb_build_info.h | sed 's/^#define /  /'
echo "commit strings compiled into the binary:"
strings $BIN | grep -E '^[0-9a-f]{40}(-dirty)?$|not checked out' | sort -u | sed 's/^/  /'
echo "sha256: $(sha256sum $BIN | cut -d' ' -f1)  ($(stat -c %s $BIN) bytes)"
echo "compile command of the HIP object (paths normalised):"
ninja -C $B -t commands bin/benchmark_rpp_preprocessing_hip 2>/dev/null | grep -E -- "-x hip" | sed -E "s#$SRC#<src>#g; s#$B#<build>#g" | sed 's/^/  /'
echo "compile command of common/main.cpp:"
ninja -C $B -t commands bin/benchmark_rpp_preprocessing_hip 2>/dev/null | grep -E "common/main.cpp" | sed -E "s#$SRC#<src>#g; s#$B#<build>#g" | sed 's/^/  /'
echo "link command:"
ninja -C $B -t commands bin/benchmark_rpp_preprocessing_hip 2>/dev/null | tail -1 | sed -E "s#$SRC#<src>#g; s#$B#<build>#g" | cut -c1-400 | sed 's/^/  /'
