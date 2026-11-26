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
"""Helpers of scripts/check_fkl_tree.sh.

  check_fkl_tree_report.py report GROUP BUILD_LOG TARGETS_FILE OUT_CSV [REJECTED_LOG]
  check_fkl_tree_report.py targets BUILD_DIR

report       Per-target result of one `make -k` build. For every target of TARGETS_FILE (one name per line): OK when the log has its 'Built target' line, FAIL with the first error line of its compile / link output
             when one of its recipes failed, NOT_BUILT otherwise (a dependency failed, or the build was interrupted). The first error line is the first line with 'error', 'fatal error' or 'undefined reference'
             after the 'Building ... <target>.dir' / 'Linking ... <target>' line of the target (make --output-sync=target keeps the output of a recipe together).
             With REJECTED_LOG (the output of `ctest -V -R '_rejected$'` in the build tree) a target that has a ctest test '<target>_rejected' is an EXPECTED REJECTION: such a target is not built by the group, and its
             result is the result of that test, which builds the target and passes only if the build fails, with the message of the check and with no other error: REJECTED_OK (the test passed) or REJECTED_FAIL
             (it did not; the line says why). Prints one line per target and the totals; writes OUT_CSV (group, target, status, first_error or the message of the rejection test). Exit status 1 when a target is FAIL,
             NOT_BUILT or REJECTED_FAIL.
targets      The names of the targets of the configured CMake build tree BUILD_DIR that compile something (type EXECUTABLE, SHARED_LIBRARY, STATIC_LIBRARY, MODULE_LIBRARY or OBJECT_LIBRARY; not the INTERFACE
             libraries and not the custom targets), one per line, from the CMake file API (the empty query file .cmake/api/v1/query/codemodel-v2 has to exist in BUILD_DIR before the configure step).
"""
import csv
import glob
import json
import os
import re
import sys

ERR = re.compile(r"(\bfatal error\b|\berror\b\s*[:#]|: error|\bundefined reference\b|static assertion failed)", re.I)
BUILD_OBJ = re.compile(r"Building (?:CUDA|CXX|C) object (?:\S*/)?CMakeFiles/([^/\s]+)\.dir/")
LINK = re.compile(r"Linking (?:CUDA|CXX|C) executable (?:\S*/)?([^/\s]+)\s*$")
BUILT = re.compile(r"Built target (\S+)")
MAKE_FAIL = re.compile(r"\*\*\* \[.*CMakeFiles/([^/\s]+)\.dir/")
CTEST_RESULT = re.compile(r"Test\s+#(\d+):\s+(\S+)_rejected\s+\.+\s*(.*?)\s+[\d.]+ sec")
CTEST_PREFIX = re.compile(r"^(\d+): ?(.*)$")
COMPILED_TYPES = ("EXECUTABLE", "SHARED_LIBRARY", "STATIC_LIBRARY", "MODULE_LIBRARY", "OBJECT_LIBRARY")   # the target types that compile something (not INTERFACE_LIBRARY, not custom targets)


def compiled_targets(build_dir):
    replies = os.path.join(build_dir, ".cmake", "api", "v1", "reply")
    index = sorted(glob.glob(os.path.join(replies, "index-*.json")))
    if not index:
        sys.exit(f"no CMake file API reply in {replies} (the query file .cmake/api/v1/query/codemodel-v2 has to exist before the configure step)")
    codemodel = next(o for o in json.load(open(index[-1]))["objects"] if o["kind"] == "codemodel")
    model = json.load(open(os.path.join(replies, codemodel["jsonFile"])))
    names = set()
    for conf in model["configurations"]:
        for t in conf["targets"]:
            if json.load(open(os.path.join(replies, t["jsonFile"])))["type"] in COMPILED_TYPES:
                names.add(t["name"])
    print("\n".join(sorted(names)))
    return 0


def rejected_results(path):
    """target -> (passed, message), from the output of `ctest -V -R '_rejected$'` (ctest -V prefixes the output of test N with 'N: ')."""
    lines = open(path, errors="replace").read().splitlines()
    own = {}
    for l in lines:
        m = CTEST_PREFIX.match(l)
        if m:
            own.setdefault(int(m.group(1)), []).append(m.group(2).rstrip())
    res = {}
    for l in lines:
        m = CTEST_RESULT.search(l)
        if not m:
            continue
        number, target, status = int(m.group(1)), m.group(2), m.group(3).strip()
        out = own.get(number, [])
        if status == "Passed":
            msg = next((x.strip()[3:] if x.strip().startswith("-- ") else x.strip() for x in out if "rejected with the message" in x), "")
        else:
            i = next((k for k, x in enumerate(out) if "CMake Error" in x), None)
            text = [x.strip() for x in out[i + 1:] if x.strip()][:3] if i is not None else []
            msg = (" | ".join(text) if text else status)[:400]
        res[target] = (status == "Passed", msg)
    return res


def report(group, log_path, targets_path, out_csv, rejected_path=None):
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
    rejected = rejected_results(rejected_path) if rejected_path and os.path.exists(rejected_path) else {}
    rows = []
    for t in targets:
        if t in rejected:
            rows.append((group, t, "REJECTED_OK" if rejected[t][0] else "REJECTED_FAIL", rejected[t][1]))
        elif t in built and t not in failed:
            rows.append((group, t, "OK", ""))
        elif t in failed:
            rows.append((group, t, "FAIL", first_error.get(t, "(no error line found; see the build log)")))
        else:
            rows.append((group, t, "NOT_BUILT", first_error.get(t, "")))
    for t in sorted(set(rejected) - set(targets)):       # a rejection test whose target is not in the list of the group
        rows.append((group, t, "REJECTED_OK" if rejected[t][0] else "REJECTED_FAIL", rejected[t][1]))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "target", "status", "first_error"])
        w.writerows(rows)
    for g, t, s, e in rows:
        print(f"{s:13s} {t}" + (f"  {e}" if e else ""))
    n = {s: sum(1 for r in rows if r[2] == s) for s in ("OK", "FAIL", "NOT_BUILT", "REJECTED_OK", "REJECTED_FAIL")}
    line = f"{group}: {len(rows)} targets, {n['OK']} compiled, {n['FAIL']} failed, {n['NOT_BUILT']} not built"
    if rejected:
        line += f"; {n['REJECTED_OK']} of {len(rejected)} expected rejections held (the ctest test <target>_rejected passed), {n['REJECTED_FAIL']} did not"
    print(line)
    return 0 if n["FAIL"] == 0 and n["NOT_BUILT"] == 0 and n["REJECTED_FAIL"] == 0 else 1


def main():
    if len(sys.argv) >= 6 and sys.argv[1] == "report":
        return report(*sys.argv[2:7])
    if len(sys.argv) == 3 and sys.argv[1] == "targets":
        return compiled_targets(sys.argv[2])
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
