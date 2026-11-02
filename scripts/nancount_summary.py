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
"""NaN review: summarise the NaN and Inf counts printed by the counting copies of the E4 and E16 programs (results/rtx4090/e4-preprocessing/nan-count).

  python scripts/nancount_summary.py results/rtx4090/e4-preprocessing/nan-count

Reads e4/stdout.txt and e16/stdout.txt (lines "NANCOUNT ..."), and, for E4, e4/e4_preprocessing_agreement.csv of the same run to check that the run reproduces the committed check
(162 outputs, ok_own = 1, the same maximum error). Writes nancount_e4.csv, nancount_e16.csv and nancount_summary.md: per output the counts, per version the totals and the largest count of one output.
"""
import collections
import csv
import os
import re
import sys

E4 = re.compile(r"^NANCOUNT (\S+) batch (\d+) (\S+) \((\S+)\): (\d+) NaN, (\d+) Inf, (\d+) zeros of (\d+) values")
E16 = re.compile(r"^NANCOUNT E16 (\S+) D=(\d+) S=(\d+) BH=(\d+) causal=(\d): (\d+) NaN, (\d+) Inf, (\d+) zeros of (\d+) output values")


def read(path, pattern):
    rows = []
    if os.path.exists(path):
        for line in open(path):
            m = pattern.match(line)
            if m:
                rows.append(m.groups())
    return rows


def table(header, rows):
    return "\n".join(["| " + " | ".join(header) + " |", "|---|" + "---:|" * (len(header) - 1)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def main():
    folder = sys.argv[1]
    out = []
    e4 = read(os.path.join(folder, "e4", "stdout.txt"), E4)
    if e4:
        with open(os.path.join(folder, "nancount_e4.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["experiment", "batch", "version", "convention", "nan", "inf", "zeros", "values"])
            w.writerows(e4)
        per = collections.OrderedDict()
        for exp, batch, version, conv, nan, inf, zeros, values in e4:
            p = per.setdefault((version, conv), [0, 0, 0, 0, 0, 0, 0])   # outputs, values, nan, inf, zeros, max nan, max inf
            p[0] += 1
            p[1] += int(values)
            p[2] += int(nan)
            p[3] += int(inf)
            p[4] += int(zeros)
            p[5] = max(p[5], int(nan))
            p[6] = max(p[6], int(inf))
        rows = [[v, c, p[0], f"{p[1]:,}", p[2], p[3], p[4], p[5], p[6]] for (v, c), p in per.items()]
        tot = [sum(p[i] for p in per.values()) for i in range(5)]
        rows.append(["all", "", tot[0], f"{tot[1]:,}", tot[2], tot[3], tot[4], max(p[5] for p in per.values()), max(p[6] for p in per.values())])
        out += ["## E4: NaN and Inf values per version (every output of the sweep, E4_PROFILE=1)", "", table(["version", "convention", "outputs", "values", "NaN", "Inf", "zeros", "most NaN in one output", "most Inf in one output"], rows), ""]
        agr = os.path.join(folder, "e4", "e4_preprocessing_agreement.csv")
        if os.path.exists(agr):
            ag = list(csv.DictReader([l for l in open(agr) if not l.startswith("#")]))
            errs = collections.Counter(r["max_abs_err_own_reference"] for r in ag)
            oks = collections.Counter(r["ok_own"] for r in ag)
            out += [f"The agreement CSV of this run has {len(ag)} rows, ok_own {dict(oks)}, max_abs_err_own_reference {dict(errs)} (the committed runs: 162 rows, ok_own = 1, 5.35117255e-06 in every row).", ""]
    e16 = read(os.path.join(folder, "e16", "stdout.txt"), E16)
    if e16:
        with open(os.path.join(folder, "nancount_e16.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["variant", "d", "seq", "bh", "causal", "nan", "inf", "zeros", "values"])
            w.writerows(e16)
        per = collections.OrderedDict()
        for variant, d, s, bh, c, nan, inf, zeros, values in e16:
            p = per.setdefault(variant, [0, 0, 0, 0, 0, 0, 0])
            p[0] += 1
            p[1] += int(values)
            p[2] += int(nan)
            p[3] += int(inf)
            p[4] += int(zeros)
            p[5] = max(p[5], int(nan))
            p[6] = max(p[6], int(inf))
        rows = [[v, p[0], f"{p[1]:,}", p[2], p[3], p[4], p[5], p[6]] for v, p in per.items()]
        tot = [sum(p[i] for p in per.values()) for i in range(5)]
        rows.append(["all", tot[0], f"{tot[1]:,}", tot[2], tot[3], tot[4], max(p[5] for p in per.values()), max(p[6] for p in per.values())])
        out += ["## E16: NaN and Inf values per variant (all BH x S x D output values; D = 64 and 128, S = 512 and 8192, both masks, E16_PROFILE=1)", "",
                table(["variant", "outputs", "values", "NaN", "Inf", "zeros", "most NaN in one output", "most Inf in one output"], rows), ""]
    text = "\n".join(out)
    open(os.path.join(folder, "nancount_summary.md"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
