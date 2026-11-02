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
"""Check the repair of scripts/csv_repair.py independently of it: every repaired file against its original.

  python scripts/csv_repair_verify.py --original-ref <git ref of the commit before the repair> [--listing results/rtx4090/csv-repair/listing.csv]

For each file of the listing, with the original taken from git (`git show <ref>:<path>`):
  - the sha256 of the original and of the repaired file are the ones of the listing;
  - both have the same number of lines, and every line that differs is a data row of the original with more fields than the header;
  - the repaired row, read with a standard CSV reader, has exactly as many fields as the header, and joining its fields with commas gives back the original line character by character
    (so no character of any value changed, only the quoting was added);
  - every other line (comments, header, the other rows) is identical.
Exit code 0 when everything holds. This script does not import scripts/csv_repair.py and does not use its splitting rules.
"""
import argparse
import csv
import hashlib
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--original-ref", required=True)
    ap.add_argument("--listing", default="results/rtx4090/csv-repair/listing.csv")
    a = ap.parse_args()
    failures = 0
    files = list(csv.DictReader(open(a.listing, newline="")))
    for item in files:
        path = item["path"]
        original = subprocess.run(["git", "show", f"{a.original_ref}:{path}"], capture_output=True, check=True).stdout
        repaired = open(path, "rb").read()
        problems = []
        if hashlib.sha256(original).hexdigest() != item["sha256_original"]:
            problems.append("sha256 of the original differs from the listing")
        if hashlib.sha256(repaired).hexdigest() != item["sha256_repaired"]:
            problems.append("sha256 of the repaired file differs from the listing")
        if int(item["rows_rewritten"]) == 0:
            if original != repaired:
                problems.append("a file listed as kept differs from its original")
            print(f"{path}: {'OK, kept as it is' if not problems else 'FAILED: ' + '; '.join(problems)}")
            failures += bool(problems)
            continue
        o_lines, r_lines = original.decode().split("\n"), repaired.decode().split("\n")
        if len(o_lines) != len(r_lines):
            problems.append(f"{len(o_lines)} lines against {len(r_lines)}")
        else:
            width = None
            changed = 0
            for n, (o, r) in enumerate(zip(o_lines, r_lines), start=1):
                is_data = bool(o.strip()) and not o.startswith("#")
                if is_data and width is None:
                    width = len(o.split(","))   # the header line
                    if o != r:
                        problems.append(f"line {n}: the header changed")
                    continue
                if o == r:
                    continue
                changed += 1
                if not is_data:
                    problems.append(f"line {n}: a comment or empty line changed")
                    continue
                if len(o.split(",")) == width:
                    problems.append(f"line {n}: a row with the right number of fields was rewritten")
                parsed = next(csv.reader([r]))
                if len(parsed) != width:
                    problems.append(f"line {n}: {len(parsed)} fields after the repair for {width} columns")
                if ",".join(parsed) != o:
                    problems.append(f"line {n}: the values changed")
            if changed != int(item["rows_rewritten"]):
                problems.append(f"{changed} rows differ, the listing says {item['rows_rewritten']}")
        print(f"{path}: {'OK, ' + item['rows_rewritten'] + ' row(s)' if not problems else 'FAILED: ' + '; '.join(problems[:3])}")
        failures += bool(problems)
    print(f"{len(files)} files checked, {failures} failed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
