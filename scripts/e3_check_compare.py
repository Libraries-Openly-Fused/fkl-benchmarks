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
import array
import math
import struct
import sys


def read(path):
    a = array.array("f")
    with open(path, "rb") as f:
        a.fromfile(f, len(open(path, "rb").read()) // 4)
    return a


def main():
    a, b = read(sys.argv[1]), read(sys.argv[2])
    if len(a) != len(b) or len(a) == 0:
        print(f"different sizes or empty: {len(a)} and {len(b)}")
        sys.exit(1)
    identical = sum(1 for x, y in zip(a, b) if struct.pack("f", x) == struct.pack("f", y))
    nonfinite_a = sum(1 for x in a if not math.isfinite(x))
    nonfinite_b = sum(1 for x in b if not math.isfinite(x))
    pairs = [(x, y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    maxabs = max((abs(x - y) for x, y in pairs), default=float("nan"))
    maxrel = max((abs(x - y) / max(abs(y), 1e-30) for x, y in pairs), default=float("nan"))
    over = sum(1 for x, y in pairs if abs(x - y) / max(abs(y), 1e-30) > 1e-3)
    finite_b = [y for y in b if math.isfinite(y)]
    print(f"{len(a)} values; bit-identical {identical}; non-finite in a {nonfinite_a}, in b {nonfinite_b}; max abs diff {maxabs:.6g}; max rel diff {maxrel:.6g}; "
          f"values with rel diff > 1e-3: {over}; range of b [{min(finite_b, default=float('nan')):.6g}, {max(finite_b, default=float('nan')):.6g}]")
    sys.exit(0 if nonfinite_a == nonfinite_b else 1)


if __name__ == "__main__":
    main()
