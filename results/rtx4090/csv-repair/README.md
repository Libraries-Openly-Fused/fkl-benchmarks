# Repair of CSV files whose text fields were written without quotes

**What was wrong.** A text value that contains a comma was written unquoted, so the row has more fields than the header: 12 files of `e3-compile` (the free text `why` of one failed compile each, column 6 of 26: the 20 columns after it (19 numbers and `host_status`) shift for a reader that splits on commas),
`e6-resize-convention/resize_convention.csv` (84 of 84 rows: the case and candidate names have commas inside parentheses; the three numbers are the last three columns) and `e16-attention/precheck/e16_attention.csv` (2 rows: the comma of the schedule string of the first FKL binary).
The writers were `common/fklb_csv.py` (Python) and `common/csv_writer.h` (C++, fixed by the AMD session in #13); `common/fklb_csv.py` quotes now.

**What was done.** `scripts/csv_repair.py` rewrote only the rows whose field count differs from the header (the text field is quoted; its pieces are joined with the commas they were split at); every other line, comments and header included, is byte for byte the original.
`scripts/csv_repair_verify.py`, written independently of it, checks every file against git: the sha256 of the original and of the repaired file are the ones of `listing.csv`; same number of lines; each rewritten row, read by a standard CSV reader, has the width of the header and joins back to the original line character by character.
`listing.csv` has the path, the sha256 of the original and of the repaired file, the number of rewritten rows and the rule. **No value changes.**

**Left as it is, on purpose:** `e16-attention/original-csv/` (3 files, 40 rows each with the same defect): they are the originals, as the program wrote them, of the earlier repair of `e16-attention/run*/e16_attention.csv`; their sha256 and the repair are in `e16-attention/README.md`. They are listed with 0 rewritten rows and the verifier checks that they are identical to git.

**Which numbers can be affected.** The numbers of the paper that come from `e3-compile` are taken from rows with `status` ok (`scripts/plot_compiletime.py` takes only those and the timeouts): those rows were never affected. The 11 rows with `status` failed were read with shifted columns by `scripts/e3_table.py`: the cells of the derived tables for them are regenerated in the next commit, with the list of the changed cells.
Nothing reads `resize_convention.csv` or the precheck file in a script; the numbers of the resize convention were taken from the program output.

## The derived E3 tables (second commit, can be dropped)

`results/rtx4090/e3-compile/e3_table.md`, `e3_table_chains.csv`, `e3_table_trees.csv`, `e3_table_before_after.csv` and `e3_table_staticloop.csv` are written by `scripts/e3_table.py` from the CSV files of that folder. Regenerated from the **original** files they reproduce the previously committed tables byte for byte; regenerated from the **repaired** files, 53 cells change, all of them in the 11 rows of failed compiles (variant B at N = 3,848 and N = 4,000; nine failures of the older trees at N = 245 to 300): the columns after the text were read one or more positions to the left (e.g. the system CPU time as the peak memory). The status `ok` rows and `e3_table_staticloop.csv` are identical.
`scripts/table_cells_changed.py` compares the committed and the regenerated tables cell by cell; `e3_table_cells_changed.csv` lists every changed cell with the file, table, row, column, old value and new value, complete. The compile-time figure reads only `ok` rows and the StaticLoop table: drawn from the regenerated tables it is identical to the one drawn from the committed tables (the PNG byte for byte, the PDF except the folder names in its metadata).
These are re-readings of values that are in the committed CSV files, not measurements.
