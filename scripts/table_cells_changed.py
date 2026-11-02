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
"""Compare derived tables (.csv, and the pipe tables of .md files) between a git ref and the working tree, cell by cell.

  python scripts/table_cells_changed.py --ref HEAD --out results/rtx4090/csv-repair/e3_table_cells_changed.csv results/rtx4090/e3-compile/e3_table.md results/rtx4090/e3-compile/e3_table_chains.csv ...

Both versions must have the same number of rows (and, for the markdown, the same number of cells per row). Prints, per file, the number of cells compared, the number that differ and the rows that contain a difference, and writes every changed cell
(file, table, row, row label, column index, column, old value, new value) to --out; table = the n-th pipe table of a markdown file (0 for a CSV), row = the n-th data row of that table, both counted from 0.
Exit code 1 when the shapes differ.
"""
import argparse
import csv
import subprocess
import sys


def old_text(ref, path):
    return subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True, text=True, check=True).stdout


def csv_tables(text):
    lines = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    rows = list(csv.reader(lines))
    return [(rows[0], rows[1:])]


def md_tables(text):
    """[(header cells, [rows])] for every pipe table of a markdown file."""
    tables, header, rows = [], None, []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("|"):
            if header is not None:
                tables.append((header, rows))
                header, rows = None, []
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if header is None:
            header = cells
        elif set("".join(cells)) <= set("-: "):
            continue
        else:
            rows.append(cells)
    if header is not None:
        tables.append((header, rows))
    return tables


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    changes, shape_ok = [], True
    for path in a.files:
        parse = md_tables if path.endswith(".md") else csv_tables
        old, new = parse(old_text(a.ref, path)), parse(open(path).read())
        if len(old) != len(new):
            print(f"{path}: {len(old)} tables against {len(new)}")
            shape_ok = False
            continue
        compared = changed_cells = 0
        changed_rows = set()
        for t, ((oh, orows), (nh, nrows)) in enumerate(zip(old, new)):
            if oh != nh or len(orows) != len(nrows):
                print(f"{path}: table {t} changed shape ({len(orows)} against {len(nrows)} rows, header equal: {oh == nh})")
                shape_ok = False
                continue
            for r, (o, n) in enumerate(zip(orows, nrows)):
                if len(o) != len(n):
                    print(f"{path}: table {t} row {r}: {len(o)} against {len(n)} cells")
                    shape_ok = False
                    continue
                for c, (x, y) in enumerate(zip(o, n)):
                    compared += 1
                    if x != y:
                        changed_cells += 1
                        changed_rows.add((t, r))
                        label = " | ".join(v for i, v in enumerate(n[:4]) if o[i] == v and v)   # the first cells of the row that did not change identify it
                        changes.append([path, t, r, label, c, nh[c] if c < len(nh) else f"cell {c}", x, y])
        print(f"{path}: {compared} cells compared, {changed_cells} changed, in {len(changed_rows)} rows; {compared - changed_cells} cells identical")
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["file", "table", "row", "row_label", "column_index", "column", "old_value", "new_value"])
        w.writerows(changes)
    print(f"{len(changes)} changed cells written to {a.out}")
    sys.exit(0 if shape_ok else 1)


if __name__ == "__main__":
    main()
