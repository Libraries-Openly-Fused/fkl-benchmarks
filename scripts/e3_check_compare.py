#!/usr/bin/env python3
# Copyright 2026 Oscar Amoros Huguet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Compare two raw float32 outputs of the E3 check binaries (the GPU and the CPU back-end on the same deterministic input, or two GPU builds).

  python scripts/e3_check_compare.py a.bin b.bin
Prints the number of values, the number of bit-identical values, the maximum absolute and relative difference, and the number of values that
differ by more than 1e-3 relative. Exit code 0 when the files have the same size and no value is NaN or infinite where the other is not.
"""
import sys

import numpy as np


def main():
    a = np.fromfile(sys.argv[1], dtype=np.float32)
    b = np.fromfile(sys.argv[2], dtype=np.float32)
    if a.size != b.size or a.size == 0:
        print(f"different sizes or empty: {a.size} and {b.size}")
        sys.exit(1)
    finite = np.isfinite(a) & np.isfinite(b)
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    rel = d / np.maximum(np.abs(b.astype(np.float64)), 1e-30)
    print(f"{a.size} values; bit-identical {int(np.sum(a.view(np.uint32) == b.view(np.uint32)))}; non-finite in a {int(np.sum(~np.isfinite(a)))}, in b {int(np.sum(~np.isfinite(b)))}; "
          f"max abs diff {np.max(d[finite]) if finite.any() else float('nan'):.6g}; max rel diff {np.max(rel[finite]) if finite.any() else float('nan'):.6g}; "
          f"values with rel diff > 1e-3: {int(np.sum(rel[finite] > 1e-3))}; range of b [{np.min(b[np.isfinite(b)]):.6g}, {np.max(b[np.isfinite(b)]):.6g}]")
    sys.exit(0 if np.array_equal(np.isfinite(a), np.isfinite(b)) else 1)


if __name__ == "__main__":
    main()
