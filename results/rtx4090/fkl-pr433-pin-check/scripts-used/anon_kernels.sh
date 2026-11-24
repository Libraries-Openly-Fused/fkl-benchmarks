#!/bin/bash
# Per *_cu program of a build: the kernel entries of its sm_89 cubin (cuobjdump -symbols: STT_FUNC with STO_ENTRY), how many of them carry
# nvcc's anonymous-namespace module id (_GLOBAL__N__) in their name, and the pairs and hash_only renames that rename_causes.py reported for it.
#   anon_kernels.sh <bin folder> <rename-causes report>
# Reads the binaries only; nothing is run.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh > /dev/null 2>&1
BIN=$1; RC=$2
cd "$BIN" || exit 1
for b in *_cu; do
  s=$(cuobjdump -symbols -arch sm_89 "$b" 2>/dev/null | awk '$1 == "STT_FUNC" && $3 == "STO_ENTRY"')
  k=$(printf '%s\n' "$s" | grep -c .); n=$(printf '%s\n' "$s" | grep -c '_GLOBAL__N__')
  p=$(grep -E "^  $b: " "$RC" | grep -o -E 'pairs [0-9]+' | awk '{print $2}'); h=$(grep -E "^  $b: " "$RC" | grep -o -E 'hash_only [0-9]+' | awk '{print $2}')
  echo "$b: kernel entries $k, pairs ${p:-?}; with the module id $n, hash_only renames ${h:-?}"
done | awk '{print} {gsub(/[,;]/, ""); k += $4; n += $11; h += $14; if ($11 > 0) c++; if ($14 > 0) d++; if ($11 != $14 || $4 != $6) m++}
            END {print "TOTAL: kernel entries " k "; with the module id " n " in " c " programs; hash_only renames " h " in " d " programs; programs where a count differs: " m + 0}'
