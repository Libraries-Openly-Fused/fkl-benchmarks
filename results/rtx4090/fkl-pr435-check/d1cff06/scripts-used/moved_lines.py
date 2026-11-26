#!/usr/bin/env python3
"""C-PM-CUDA: the error-check lines that moved between two versions of source files (CPU only).

  moved_lines.py <old file> <new file> [<old file> <new file> ...]

The two versions are aligned line by line (difflib.SequenceMatcher, the equal blocks); for every line of the old file inside an equal block that
calls gpuErrchk( or FKLB_*_CHECK( and whose number changes, prints old:new (the form of host_compare435.py --line-pairs), comma separated.
"""
import difflib, re, sys
pairs = set()
for o, n in zip(sys.argv[1::2], sys.argv[2::2]):
    a, b = open(o).read().splitlines(), open(n).read().splitlines()
    for blk in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
        for k in range(blk.size):
            i, j = blk.a + k, blk.b + k
            if i != j and re.search(r"gpuErrchk\(|FKLB_[A-Z]+_CHECK\(", a[i]):
                pairs.add((i + 1, j + 1))
print(",".join(f"{x}:{y}" for x, y in sorted(pairs)))
