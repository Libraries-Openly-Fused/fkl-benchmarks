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
"""Per-target result of one `make -k` build of scripts/check_fkl_tree.sh.

  check_fkl_tree_report.py GROUP BUILD_LOG TARGETS_FILE OUT_CSV

For every target of TARGETS_FILE (one name per line): OK when the log has its 'Built target' line, FAIL with the first error line of its compile / link output when one of its recipes failed, NOT_BUILT otherwise
(a dependency failed, or the build was interrupted). The first error line is the first line with 'error', 'fatal error' or 'undefined reference' after the 'Building ... <target>.dir' / 'Linking ... <target>' line of
the target (make --output-sync=target keeps the output of a recipe together). Prints one line per target and the totals; writes OUT_CSV (group, target, status, first_error).
"""
import csv
import re
import sys

ERR = re.compile(r"(\bfatal error\b|\berror\b\s*[:#]|: error|\bundefined reference\b|static assertion failed)", re.I)
BUILD_OBJ = re.compile(r"Building (?:CUDA|CXX|C) object (?:\S*/)?CMakeFiles/([^/\s]+)\.dir/")
LINK = re.compile(r"Linking (?:CUDA|CXX|C) executable (?:\S*/)?([^/\s]+)\s*$")
BUILT = re.compile(r"Built target (\S+)")
MAKE_FAIL = re.compile(r"\*\*\* \[.*CMakeFiles/([^/\s]+)\.dir/")


def main():
    group, log_path, targets_path, out_csv = sys.argv[1:5]
    targets = [t.strip() for t in open(targets_path) if t.strip()]
    lines = open(log_path, errors="replace").read().splitlines()
    built = {m.group(1) for l in lines for m in [BUILT.search(l)] if m}
    failed = {m.group(1) for l in lines for m in [MAKE_FAIL.search(l)] if m}
    first_error = {}
    current = None
    for l in lines:
        m = BUILD_OBJ.search(l) or LINK.search(l)
        if m:
            current = m.group(1)
            continue
        if l.startswith("["):          # another recipe's header (Built target, Generating, ...)
            current = None
            continue
        if l.startswith(("make", "gmake")):
            continue
        if current and current not in first_error and ERR.search(l):
            first_error[current] = l.strip()[:400]
    rows = []
    for t in targets:
        if t in built and t not in failed:
            rows.append((group, t, "OK", ""))
        elif t in failed:
            rows.append((group, t, "FAIL", first_error.get(t, "(no error line found; see the build log)")))
        else:
            rows.append((group, t, "NOT_BUILT", first_error.get(t, "")))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "target", "status", "first_error"])
        w.writerows(rows)
    for g, t, s, e in rows:
        print(f"{s:9s} {t}" + (f"  {e}" if e else ""))
    n = {s: sum(1 for r in rows if r[2] == s) for s in ("OK", "FAIL", "NOT_BUILT")}
    print(f"{group}: {len(rows)} targets, {n['OK']} compiled, {n['FAIL']} failed, {n['NOT_BUILT']} not built")
    return 0 if n["FAIL"] == 0 and n["NOT_BUILT"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
