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
"""Quote the text fields of the committed CSV files whose rows have more fields than the header because a text value contains commas.

  python scripts/csv_repair.py [--root .] [--listing results/rtx4090/csv-repair/listing.csv]

The writers of these files (common/fklb_csv.py before its fix, common/csv_writer.h before its fix) wrote text values unquoted. Only the rows whose field count differs from the header are rewritten;
every other line (comments, header, the other rows) is kept byte for byte, and no value changes: the pieces of the text field are joined with the commas they were split at, and the field is quoted.
Where the text field sits is known per experiment (RULES): `tail` = the text is the column NAME and the columns after it are fixed, so the text is everything between the columns before it and the columns after it;
`parentheses` = every text column of the file has balanced parentheses and a comma inside parentheses belongs to the text.
Writes a listing (path, sha256 of the original, sha256 of the repaired file, rows rewritten, rule); scripts/csv_repair_verify.py checks every repaired file against its original independently.
"""
import argparse
import csv
import hashlib
import io
import os
import sys

RULES = [
    ("results/rtx4090/e3-compile", "*.csv", ("tail", "why")),
    ("results/rtx4090/e16-attention/precheck", "e16_attention.csv", ("tail", "schedule")),
    ("results/rtx4090/e6-resize-convention", "resize_convention.csv", ("parentheses", None)),
]
# files with the same defect that are NOT repaired, listed with the reason
KEPT = [
    ("results/rtx4090/e16-attention/original-csv", "*.csv", "kept: the originals of the earlier repair of results/rtx4090/e16-attention/run*/e16_attention.csv, as the program wrote them; the README of that folder shows the repair that gives the repaired files"),
]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def split_parentheses(line, width):
    """Split a line on the commas outside parentheses; the result must have `width` fields."""
    fields, depth, cur = [], 0, []
    for ch in line.rstrip("\n"):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            fields.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    fields.append("".join(cur))
    return fields


def quote_row(fields):
    out = io.StringIO()
    csv.writer(out, lineterminator="\n").writerow(fields)
    return out.getvalue()


def repair(path, rule):
    kind, name = rule
    lines = open(path, newline="").read().split("\n")
    trailing_newline = lines[-1] == ""
    if trailing_newline:
        lines = lines[:-1]
    head = None
    changed = 0
    out = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            out.append(line)
            continue
        if head is None:
            head = line.split(",")   # the files were written without any quoting: a plain split is what the writer produced
            if kind == "tail" and name not in head:
                return "\n".join(lines) + ("\n" if trailing_newline else ""), 0   # not a file of this writer
            out.append(line)
            continue
        fields = line.split(",")
        if len(fields) == len(head):
            out.append(line)
            continue
        if kind == "tail":
            k = head.index(name)
            after = len(head) - k - 1
            new = fields[:k] + [",".join(fields[k:len(fields) - after])] + fields[len(fields) - after:]
        else:
            new = split_parentheses(line, len(head))
        if len(new) != len(head):
            raise SystemExit(f"{path}: cannot repair the row {line[:80]!r} ({len(new)} fields for {len(head)} columns)")
        out.append(quote_row(new).rstrip("\n"))
        changed += 1
    return "\n".join(out) + ("\n" if trailing_newline else ""), changed


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=".")
    ap.add_argument("--listing", default="results/rtx4090/csv-repair/listing.csv")
    a = ap.parse_args()
    import glob
    rows = []
    for folder, pattern, rule in RULES:
        for path in sorted(glob.glob(os.path.join(a.root, folder, pattern))):
            before = sha256(path)
            text, changed = repair(path, rule)
            if changed == 0:
                continue
            open(path, "w", newline="").write(text)
            rows.append([os.path.relpath(path, a.root), before, sha256(path), changed, f"{rule[0]}:{rule[1] or 'all text columns'}"])
            print(f"{rows[-1][0]}: {changed} row(s) rewritten")
    for folder, pattern, why in KEPT:
        for path in sorted(glob.glob(os.path.join(a.root, folder, pattern))):
            rows.append([os.path.relpath(path, a.root), sha256(path), sha256(path), 0, why])
    os.makedirs(os.path.dirname(os.path.join(a.root, a.listing)), exist_ok=True)
    with open(os.path.join(a.root, a.listing), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["path", "sha256_original", "sha256_repaired", "rows_rewritten", "rule"])
        w.writerows(rows)
    print(f"{sum(1 for r in rows if r[3])} files repaired, {sum(1 for r in rows if not r[3])} kept as they are; listing {a.listing}")


if __name__ == "__main__":
    main()
