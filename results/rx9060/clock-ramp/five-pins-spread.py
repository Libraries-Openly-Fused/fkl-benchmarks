#!/usr/bin/env python3
"""Spread of the timing of pipeline (a) between the five FKL pins, from the CSVs of five-pins-three-rounds/ (written after the review of fkl-benchmarks #6).

  python results/rx9060/clock-ramp/five-pins-spread.py > results/rx9060/clock-ramp/five-pins-spread.txt     (from the root of the repository)

For every variant and batch: the mean over the 3 rounds of the median_ms of each pin, then (max - min) / median over the 5 pins."""
import csv
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "five-pins-three-rounds")
PINS = ["38", "b44725e", "d776713", "bbc5a7b", "c8"]
data = {}
for pin in PINS:
    for rnd in (1, 2, 3):
        path = os.path.join(BASE, f"r{rnd}_{pin}", "benchmark_rpp_preprocessing_a_GPU_AMD.csv")
        with open(path, newline="") as f:
            lines = [l for l in f if l.strip() and not l.startswith("#")]
        for row in csv.DictReader(lines, skipinitialspace=True):
            data.setdefault((row["variant"], int(row["batch"])), {}).setdefault(pin, []).append(float(row["median_ms"]))
cells = []
for (variant, batch), per_pin in sorted(data.items()):
    means = [statistics.mean(per_pin[p]) for p in PINS]
    cells.append(((max(means) - min(means)) / statistics.median(means), variant, batch, means))
print("pins:", ", ".join(PINS), "(38 = 38c4f32, b44725e = merge of #411, d776713 = #414, bbc5a7b = #416, c8 = c8fcbe8); mean of 3 rounds each; spread = (max - min) / median of the 5 pin means")
print(f"{'variant':22s} {'cells':>5s} {'median %':>9s} {'max %':>7s} {'at batch':>9s} {'> 1 %':>6s} {'> 2 %':>6s} {'> 5 %':>6s}")
for variant in sorted({c[1] for c in cells}):
    cs = [c for c in cells if c[1] == variant]
    worst = max(cs)
    print(f"{variant:22s} {len(cs):5d} {100 * statistics.median(c[0] for c in cs):9.2f} {100 * worst[0]:7.2f} {worst[2]:9d} {sum(c[0] > 0.01 for c in cs):6d} {sum(c[0] > 0.02 for c in cs):6d} {sum(c[0] > 0.05 for c in cs):6d}")
print(f"{'all':22s} {len(cells):5d} {100 * statistics.median(c[0] for c in cells):9.2f} {100 * max(c[0] for c in cells):7.2f} {'':9s} {sum(c[0] > 0.01 for c in cells):6d} {sum(c[0] > 0.02 for c in cells):6d} {sum(c[0] > 0.05 for c in cells):6d}")
print("\nbatches 100, 200, 300 (the means in ms of the 5 pins, in the order above):")
for batch in (100, 200, 300):
    for variant in sorted({c[1] for c in cells}):
        c = next(c for c in cells if c[1] == variant and c[2] == batch)
        print(f"  batch {batch} {variant:22s} spread {100 * c[0]:5.2f} %   " + "  ".join(f"{m:.4f}" for m in c[3]))
print("\nthe final pin (c8) against the first (38), fkl_fused:")
for batch in (100, 200, 300):
    c = next(c for c in cells if c[1] == "fkl_fused" and c[2] == batch)
    print(f"  batch {batch}: {100 * (c[3][4] / c[3][0] - 1):+.2f} %")
print("\nthe ten largest cells:")
for c in sorted(cells, reverse=True)[:10]:
    print(f"  {100 * c[0]:6.2f} %  {c[1]:22s} batch {c[2]}")
