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
"""Quote the text that has a comma in the agreement CSVs of the CPU benchmarks (results/cpu-5800x3d/) and of the RPP benchmark (results/rx9060/rpp-preprocessing/).

The first versions of the benchmarks wrote text without quotes although some of it has a comma, so a reader that splits on commas saw more fields in those rows than in the header.
Two layouts have the defect, and the script tells them apart by the header:

  separator ", " (opencv-cpu/preprocessing/): the name of the comparison is the second field ("opencv_default vs scalar reference, HALF_PIXEL sampling (<= 1e-4)"), so it is the text
  between the first field and the last (columns - 2) fields.
  separator "," and a header that ends in "note" (rpp/benchmark_rpp_preprocessing.h): the note ("FKL samples ASYMMETRIC, RPP HALF_PIXEL") is the last column, so it is the text after the
  (columns - 1)th comma of a row that has more fields than the header. The columns before the note must be numbers, or text columns without a comma (comparison, region): if the extra comma
  were in an earlier column the numbers would shift, and the row is an error and is not "repaired".

The text is put in double quotes (RFC 4180) when it has a comma or a quote (a text that is already in quotes is left as it is, so running the script again changes nothing), and nothing
else changes: every other byte of the file stays, the line endings and a missing final newline included. With the separator ", " a reader has to skip the space after the separator to
read the quoted name as one field (Python csv.reader(..., skipinitialspace=True), pandas.read_csv(..., skipinitialspace=True)); with the separator "," any csv reader does.
Files with another layout are not touched, but their rows must have as many fields as the header. A file with a byte order mark, with no header row, or with rows whose number of fields
cannot be explained is an error.

  python scripts/e5_quote_agreement_csv.py --check FILE...   for each file: how many rows have each number of fields, and how many rows need quotes. Exit status 0 when no file
                                                             needs quotes and every file could be processed, 1 otherwise
  python scripts/e5_quote_agreement_csv.py --write FILE...   rewrites the files that need it and prints the sha256 of each before and after. If any file is an error NOTHING is written
                                                             (exit status 1)
"""
import argparse
import csv
import hashlib
import io
import sys

BOM = b'\xef\xbb\xbf'
TEXT_COLUMNS = ('comparison', 'region', 'note')  # layout with the separator ",": the columns that hold text; the others hold numbers


class Problem(Exception):
    """A file or a row that the script does not repair."""


def split_lines(text):
    """[(body, line ending)] for every line: the ending is kept, so that nothing but the quotes changes (a lone CR or CRLF stays, the last line may have none)."""
    out = []
    for line in io.StringIO(text, newline=''):
        body = line.rstrip('\r\n')
        out.append((body, line[len(body):]))
    return out


def is_quoted(text):
    """Already one RFC 4180 field in double quotes (every quote inside is doubled): a second run must not quote it again."""
    return len(text) >= 2 and text[0] == '"' and text[-1] == '"' and '"' not in text[1:-1].replace('""', '')


def quote(text):
    if is_quoted(text):
        return text
    if ',' in text or '"' in text or '\n' in text or '\r' in text:
        return '"' + text.replace('"', '""') + '"'
    return text


def is_row(body):
    return bool(body.strip()) and not body.startswith('#')


def header_index(lines):
    for i, (body, _) in enumerate(lines):
        if is_row(body):
            return i
    raise Problem('no header row (the file is empty or has only comments)')


def fix_name_row(body, columns):
    """Separator ", ": the name of the comparison (second field) in quotes when it has a comma or a quote."""
    parts = body.split(', ')
    if len(parts) < columns:
        raise Problem(f'fewer than {columns} fields: {body!r}')
    head, tail = parts[0], parts[len(parts) - (columns - 2):]
    middle = ', '.join(parts[1:len(parts) - (columns - 2)])
    new = ', '.join([head, quote(middle)] + tail)
    if new != body:  # read back with a csv reader: the same fields, and the name in one piece
        if next(csv.reader([new], skipinitialspace=True)) != [head, middle] + tail:
            raise Problem(f'the repaired row does not read back as the same fields: {body!r}')
    return new


def fix_note_row(body, columns, names):
    """Separator ",": the note (last column) in quotes when the row has more fields than the header."""
    n = len(next(csv.reader([body])))
    if n < columns:
        raise Problem(f'fewer than {columns} fields: {body!r}')
    if n == columns:
        return body
    if '"' in body:
        raise Problem(f'a row with more fields than the header and a double quote is not rewritten: {body!r}')
    parts = body.split(',')
    head = parts[:columns - 1]
    for name, value in zip(names, head):
        if name not in TEXT_COLUMNS:
            try:
                float(value)
            except ValueError:
                raise Problem(f'column {name!r} holds {value!r}, not a number: the extra comma is not in the note: {body!r}') from None
    note = ','.join(parts[columns - 1:])
    new = ','.join(head + [quote(note)])
    if next(csv.reader([new])) != head + [note]:
        raise Problem(f'the repaired row does not read back as the same fields: {body!r}')
    return new


def field_counts(lines, h, skip_space):
    counts = {}
    for body, _ in lines[h + 1:]:
        if is_row(body):
            n = len(next(csv.reader([body], skipinitialspace=skip_space)))
            counts[n] = counts.get(n, 0) + 1
    return counts


def process(path):
    raw = open(path, 'rb').read()
    if raw.startswith(BOM):
        raise Problem('starts with a byte order mark: not handled')
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError as e:
        raise Problem(f'not UTF-8: {e}') from None
    lines = split_lines(text)
    h = header_index(lines)
    header = lines[h][0]
    names = [n.strip() for n in header.split(',')]
    columns = len(names)
    comma_space = ', ' in header
    if comma_space and names[-1] == 'note':
        raise Problem('a header that ends in "note" and has the separator ", " is not a layout that this script knows')
    layout = 'name' if comma_space else ('note' if names[-1] == 'note' else 'other')
    out, changed = [], 0
    for i, (body, eol) in enumerate(lines):
        new = body
        if i > h and is_row(body):
            if layout == 'name':
                new = fix_name_row(body, columns)
            elif layout == 'note':
                new = fix_note_row(body, columns, names)
        changed += new != body
        out.append(new + eol)
    new_text = ''.join(out)
    before = field_counts(lines, h, comma_space)
    after = field_counts(split_lines(new_text), h, comma_space)
    wrong = sum(n for k, n in after.items() if k != columns)
    if wrong:
        why = 'this layout cannot be repaired by this script' if layout == 'other' else 'after the repair'
        raise Problem(f'{wrong} rows do not have the {columns} fields of the header ({why}): rows by number of fields {after}')
    return {'path': path, 'raw': raw, 'new': new_text.encode('utf-8'), 'changed': changed, 'columns': columns, 'layout': layout, 'before': before, 'after': after}


def describe(r):
    if r['layout'] == 'name':
        return (f"{r['path']}: header {r['columns']} columns; rows by number of fields (csv reader, skipinitialspace): before {r['before']}, after {r['after']}; "
                f"{r['changed']} rows need quotes")
    if r['layout'] == 'note':
        return (f"{r['path']}: separator \",\", header {r['columns']} columns (last: note); rows by number of fields (csv reader): before {r['before']}, after {r['after']}; "
                f"{r['changed']} rows need quotes")
    return f"{r['path']}: separator \",\", header {r['columns']} columns, rows by number of fields {r['before']}: not touched"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--write', action='store_true')
    ap.add_argument('files', nargs='+')
    args = ap.parse_args()
    results, problems = [], []
    for path in args.files:
        try:
            results.append(process(path))
        except (Problem, OSError, csv.Error) as e:
            problems.append((path, str(e)))
    if args.write and problems:
        for r in results:
            print(describe(r))
        for path, message in problems:
            print(f'{path}: ERROR: {message}')
        print(f'nothing was written: {len(problems)} of {len(args.files)} files could not be processed', file=sys.stderr)
        return 1
    for r in results:
        line = describe(r)
        if args.write and r['changed']:
            with open(r['path'], 'wb') as f:
                f.write(r['new'])
            line += f"; sha256 before {hashlib.sha256(r['raw']).hexdigest()}, after {hashlib.sha256(r['new']).hexdigest()}"
        print(line)
    for path, message in problems:
        print(f'{path}: ERROR: {message}')
    if args.check:
        return 1 if problems or any(r['changed'] for r in results) else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
