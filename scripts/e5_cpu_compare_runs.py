#!/usr/bin/env python3
"""E5 on the CPU (Ryzen 7 5800X3D): compare repeated sweeps of one pipeline and write the median-of-three files.

  e5_cpu_compare_runs.py a OLD_DIR NEW_DIR [--write]
      Pipeline (a), the new rows (FKL c8fcbe8, three sweeps per compiler) against the old rows (FKL fb9ac3a, one sweep).
        OLD_DIR = results/cpu-5800x3d/e5-preprocessing-a
                  <compiler>/e5_preprocessing_a_CPU.csv  and  e5_preprocessing_a_agreement_CPU.csv
        NEW_DIR = results/cpu-5800x3d/e5-preprocessing-a-c8fcbe8
                  <compiler>/                                       sweep 1
                  repeats/<compiler>/run2, run3                     sweeps 2 and 3
                  repeats/<compiler>/check-nothing-else-running     one more sweep, only printed (not part of the median)
  e5_cpu_compare_runs.py b REPEATS_DIR [--write]
      Pipeline (b), three sweeps per compiler at FKL c8fcbe8.
        REPEATS_DIR = results/cpu-5800x3d/e5-preprocessing-b-c8fcbe8-repeats
                      <compiler>/run1 (= the rows of pull request #9), run2, run3

  With --write, <compiler>/e5_preprocessing_{a,b}_CPU_median_of_3.csv is written in NEW_DIR (a) or REPEATS_DIR (b): for every point and variant the median of the
  three sweep medians (median_ms), the smallest and the largest of the three medians (medians_min_ms, medians_max_ms).

Definitions used in the output (a):
  O = old median;  N1, N2, N3 = medians of the three new sweeps;  N = median(N1, N2, N3)
  D = N / O - 1;   S = (max Ni - min Ni) / N          (run-to-run spread of the new sweeps at that point)
  'outside the window':  O < min Ni - w  or  O > max Ni + w,  with w = max Ni - min Ni
  'outside the old range': N < old min  or  N > old max   (minimum and maximum of the 30 timed runs of the old row)
Percentiles are sorted[int(0.9 * n)].
"""
import csv
import os
import statistics
import sys

COMPILERS = ["g++-13", "clang++-23"]


# ---------------------------------------------------------------------------------------------------------------- reading
class Sweep:
    def __init__(self, path):
        self.path = path
        self.comments = []
        data = []
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    self.comments.append(line.rstrip("\n"))
                else:
                    data.append(line)
        reader = csv.DictReader(data, skipinitialspace=True)
        self.first = reader.fieldnames[0]          # 'batch' (a) or 'crops' (b)
        self.rows = {}
        self.order = []
        for r in reader:
            key = (int(r[self.first]), r["variant"])
            self.order.append(key)
            self.rows[key] = dict(threads=int(r["threads"]), median=float(r["median_ms"]), min=float(r["min_ms"]), max=float(r["max_ms"]),
                                  mean=float(r["mean_ms"]), iters=int(r["iters"]))


def read_agreement(path):
    """The names of the comparisons contain commas: split the first field off the left and the five numbers off the right."""
    rows = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or line.startswith(("batch", "crops")):
                continue
            n, rest = line.rstrip("\n").split(",", 1)
            parts = rest.rsplit(",", 5)
            name = parts[0].strip()
            rows[(int(n), name)] = tuple(p.strip() for p in parts[1:])
    return rows


def log_passed(path):
    with open(path) as f:
        return "all numerical checks passed" in f.read()


def pct(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))]


def fmt(x, nd=2):
    return f"{100 * x:.{nd}f} %"


# ---------------------------------------------------------------------------------------------------------------- median of three
def write_median3(out_path, sweeps, source_note):
    ref = sweeps[0]
    for s in sweeps[1:]:
        assert s.order == ref.order, (s.path, "different points")
    # the lines that identify the build must be the same in the three sweeps
    for tag in ("# repo:", "# fkl:", "# host compiler:", "# opencv:", "# cpu:"):
        lines = {next((c for c in s.comments if c.startswith(tag)), None) for s in sweeps}
        assert len(lines) == 1, (tag, lines)
    with open(out_path, "w") as f:
        for c in ref.comments:
            f.write(c + "\n")
        f.write(f"# derived by scripts/e5_cpu_compare_runs.py: {source_note}; for every point and variant median_ms is the median of the three sweep medians, "
                f"medians_min_ms and medians_max_ms the smallest and the largest of the three medians\n")
        f.write(f"{ref.first}, variant, threads, median_ms, medians_min_ms, medians_max_ms\n")
        for key in ref.order:
            m = [s.rows[key]["median"] for s in sweeps]
            f.write(f"{key[0]}, {key[1]}, {ref.rows[key]['threads']}, {statistics.median(m):g}, {min(m):g}, {max(m):g}\n")


# ---------------------------------------------------------------------------------------------------------------- pipeline (a)
def group_of(variant, threads):
    kind = "FKL" if variant.startswith("fkl") else "OpenCV"
    return f"{kind}, {'1 thread' if threads == 1 else '16 threads'}"


GROUP_ORDER = ["FKL, 1 thread", "FKL, 16 threads", "OpenCV, 1 thread", "OpenCV, 16 threads"]
HEADLINE_A = [("fkl_fused_cpu", "FKL fused, 1 thread"), ("fkl_fused_cpu_mt", "FKL fused, 16 threads"), ("opencv_1thread", "OpenCV, 1 thread"), ("opencv_default", "OpenCV, 16 threads")]
BATCH_A = [10, 50, 100, 150, 200, 250, 300]


def run_a(old_dir, new_dir, write):
    results = {}
    for comp in COMPILERS:
        old = Sweep(os.path.join(old_dir, comp, "e5_preprocessing_a_CPU.csv"))
        paths = [os.path.join(new_dir, comp), os.path.join(new_dir, "repeats", comp, "run2"), os.path.join(new_dir, "repeats", comp, "run3")]
        new = [Sweep(os.path.join(p, "e5_preprocessing_a_CPU.csv")) for p in paths]
        check = Sweep(os.path.join(new_dir, "repeats", comp, "check-nothing-else-running", "e5_preprocessing_a_CPU.csv"))
        for s in new + [check]:
            assert set(s.rows) == set(old.rows), (s.path, "different points")
        results[comp] = (old, new, check, paths)
        if write:
            write_median3(os.path.join(new_dir, comp, "e5_preprocessing_a_CPU_median_of_3.csv"), new,
                          f"the sweeps {comp}/ (sweep 1), repeats/{comp}/run2 and repeats/{comp}/run3 of the same binary")

    print("## Numerical checks")
    print("| compiler | sweep | comparisons | differ from the old rows | required checks failed | log says all passed |")
    print("|---|---|--:|--:|--:|---|")
    for comp in COMPILERS:
        old, new, check, paths = results[comp]
        old_ag = read_agreement(os.path.join(old_dir, comp, "e5_preprocessing_a_agreement_CPU.csv"))
        names = ["1", "2", "3", "check (nothing else running)"]
        dirs = paths + [os.path.join(new_dir, "repeats", comp, "check-nothing-else-running")]
        for nm, d in zip(names, dirs):
            ag = read_agreement(os.path.join(d, "e5_preprocessing_a_agreement_CPU.csv"))
            assert set(ag) == set(old_ag)
            differ = sum(1 for k in old_ag if old_ag[k] != ag[k])
            failed = sum(1 for k, v in ag.items() if v[4] == "1" and v[3] != "1")
            print(f"| {comp} | {nm} | {len(ag)} | {differ} | {failed} | {'yes' if log_passed(os.path.join(d, 'stdout.log')) else 'NO'} |")

    print("\n## Change from the old rows, by group of rows (new = median of the three sweeps)")
    print("| compiler | rows | points | change \\|D\\|: median / 90th percentile / max | spread S of the three sweeps: median / max | old (max-min)/median: median | outside the window of the three sweeps | and outside the old range |")
    print("|---|---|--:|--|--|--:|--:|--:|")
    points = []
    for comp in COMPILERS:
        old, new, check, paths = results[comp]
        for key in old.order:
            o = old.rows[key]
            ns = [s.rows[key]["median"] for s in new]
            n = statistics.median(ns)
            w = max(ns) - min(ns)
            points.append(dict(comp=comp, key=key, variant=key[1], group=group_of(key[1], o["threads"]), O=o["median"], Omin=o["min"], Omax=o["max"], Ns=ns, N=n,
                               D=n / o["median"] - 1, S=w / n, out_window=not (min(ns) - w <= o["median"] <= max(ns) + w),
                               out_old=not (o["min"] <= n <= o["max"]), old_wide=(o["max"] - o["min"]) / o["median"]))
    for comp in COMPILERS:
        for g in GROUP_ORDER:
            ps = [p for p in points if p["comp"] == comp and p["group"] == g]
            ad = [abs(p["D"]) for p in ps]
            print(f"| {comp} | {g} | {len(ps)} | {fmt(statistics.median(ad))} / {fmt(pct(ad, 0.9))} / {fmt(max(ad))} | {fmt(statistics.median(p['S'] for p in ps))} / {fmt(max(p['S'] for p in ps))} "
                  f"| {fmt(statistics.median(p['old_wide'] for p in ps), 1)} | {sum(p['out_window'] for p in ps)} | {sum(p['out_window'] and p['out_old'] for p in ps)} |")
    for comp in COMPILERS:
        ps = [p for p in points if p["comp"] == comp]
        fk = [p for p in ps if p["variant"].startswith("fkl")]
        print(f"\n{comp}: {sum(p['out_window'] for p in ps)} of {len(ps)} points lie outside the window of the three sweeps ({sum(p['out_window'] for p in fk)} FKL, "
              f"{sum(p['out_window'] for p in ps if not p['variant'].startswith('fkl'))} OpenCV); outside the window and the old range: "
              f"{sum(p['out_window'] and p['out_old'] for p in fk)} FKL, {sum(p['out_window'] and p['out_old'] for p in ps if not p['variant'].startswith('fkl'))} OpenCV.")

    print("\n## FKL points outside the window of the three sweeps AND outside the old range (old median [old min, old max] -> the three new medians; D; S)")
    for comp in COMPILERS:
        for p in sorted((p for p in points if p["comp"] == comp and p["variant"].startswith("fkl") and p["out_window"] and p["out_old"]), key=lambda q: (q["key"][1], q["key"][0])):
            print(f"- {comp}, batch {p['key'][0]}, {p['variant']}: {p['O']:.4f} [{p['Omin']:.4f}, {p['Omax']:.4f}] -> {p['Ns'][0]:.4f} {p['Ns'][1]:.4f} {p['Ns'][2]:.4f}, D {100 * p['D']:+.2f} %, S {fmt(p['S'])}")

    print("\n## Largest changes (old median [old min, old max] -> the three new medians; D; S)")
    for comp in COMPILERS:
        for label, sel in (("FKL", lambda p: p["variant"].startswith("fkl")), ("OpenCV", lambda p: not p["variant"].startswith("fkl"))):
            top = sorted((p for p in points if p["comp"] == comp and sel(p)), key=lambda q: -abs(q["D"]))[:4]
            for p in top:
                print(f"- {comp}, {label}, batch {p['key'][0]}, {p['variant']}: {p['O']:.4f} [{p['Omin']:.4f}, {p['Omax']:.4f}] -> {p['Ns'][0]:.4f} {p['Ns'][1]:.4f} {p['Ns'][2]:.4f}, D {100 * p['D']:+.2f} %, S {fmt(p['S'])}")

    print("\n## Headline rows: old (FKL fb9ac3a) -> new (FKL c8fcbe8, median of the three sweeps), change, spread S of the three sweeps")
    for comp in COMPILERS:
        print(f"\n### {comp}, median ms of 30 runs")
        print("| batch | " + " | ".join(n for _, n in HEADLINE_A) + " |")
        print("|--:|" + "--:|" * len(HEADLINE_A))
        for b in BATCH_A:
            cells = []
            for v, _ in HEADLINE_A:
                p = next(q for q in points if q["comp"] == comp and q["key"] == (b, v))
                cells.append(f"{p['O']:.3f} -> {p['N']:.3f} ({100 * p['D']:+.1f} %, S {100 * p['S']:.1f} %)")
            print(f"| {b} | " + " | ".join(cells) + " |")

    print("\n## FKL fused pipeline against OpenCV at 300 and 100 images (new rows, sweeps 1 to 3)")
    names = [("fkl_fused_cpu", "FKL ParArch::CPU, 1 thread, one executeOperations for the batch"),
             ("fkl_fused_cpu_mt", "FKL ParArch::CPU_MULTITHREAD, 16 threads, one executeOperations for the batch"),
             ("opencv_1thread", "OpenCV, whole-batch loop, 1 thread"),
             ("opencv_default", "OpenCV, whole-batch loop, default threads (cv::getNumThreads() = 16)"),
             ("opencv_percrop_parallel", "OpenCV, crops processed in parallel with cv::parallel_for_, 16 threads")]
    print("### Time, ms: median of the three sweeps (smallest to largest of the three medians)")
    print("| images | compiler | variant in the CSV | what it is | time |")
    print("|--:|---|---|---|--:|")
    for b in (300, 100):
        for comp in COMPILERS:
            new = results[comp][1]
            for v, what in names:
                ms = [sw.rows[(b, v)]["median"] for sw in new]
                print(f"| {b} | {comp} | {v} | {what} | {statistics.median(ms):.3f} ({min(ms):.3f} to {max(ms):.3f}) |")
    print("\n### Speedup = OpenCV time / FKL time, computed in each sweep: median over the three sweeps (smallest to largest); above 1 FKL is faster")
    print("| images | compiler | FKL variant | OpenCV variant | speedup |")
    print("|--:|---|---|---|--:|")
    for b in (300, 100):
        for comp in COMPILERS:
            new = results[comp][1]
            for f, o in (("fkl_fused_cpu", "opencv_1thread"), ("fkl_fused_cpu", "opencv_default"), ("fkl_fused_cpu_mt", "opencv_percrop_parallel"),
                         ("fkl_fused_cpu_mt", "opencv_default"), ("fkl_fused_cpu_mt", "opencv_1thread"), ("fkl_fused_cpu", "opencv_percrop_parallel")):
                r = [sw.rows[(b, o)]["median"] / sw.rows[(b, f)]["median"] for sw in new]
                print(f"| {b} | {comp} | {f} | {o} | {statistics.median(r):.2f} ({min(r):.2f} to {max(r):.2f}) |")

    print("\n## Sweep against sweep: |a / b - 1| of the medians over the points (median / 90th percentile; for the 16-thread FKL rows also the maximum)")
    print("| compiler | sweeps | all rows | FKL, 1 thread | FKL, 16 threads |")
    print("|---|---|--|--|--|")
    for comp in COMPILERS:
        old, new, check, paths = results[comp]
        sw = {"1": new[0], "2": new[1], "3": new[2], "check": check}
        names = list(sw)
        sels = [(lambda k: True, False),
                (lambda k: group_of(k[1], old.rows[k]["threads"]) == "FKL, 1 thread", False),
                (lambda k: group_of(k[1], old.rows[k]["threads"]) == "FKL, 16 threads", True)]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                cells = []
                for sel, with_max in sels:
                    d = [abs(sw[a].rows[k]["median"] / sw[b].rows[k]["median"] - 1) for k in old.order if sel(k)]
                    cells.append(f"{fmt(statistics.median(d))} / {fmt(pct(d, 0.9))}" + (f" / max {fmt(max(d))}" if with_max else ""))
                print(f"| {comp} | {a} - {b} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------------------------------------------------------- pipeline (b)
# the columns of the tables of pull request #9 (README of e5-preprocessing-b-c8fcbe8)
HEADLINE_B = [("fkl_convert_cpu", "FKL convert 1 thread"), ("fkl_convert_cpu_mt", "FKL convert 16 threads"), ("opencv_convert_1thread", "OpenCV convert 1 thread"),
              ("opencv_convert_default", "OpenCV convert 16 threads"), ("fkl_chain_fused_cpu", "FKL chain only (fused) 1 thread"),
              ("fkl_chain_fused_cpu_mt", "FKL chain only (fused) 16 threads"), ("opencv_pipeline_percrop_1thread", "OpenCV whole pipeline 1 thread"),
              ("fkl_pipeline_fused_cpu", "FKL whole pipeline (fused) 1 thread"), ("fkl_pipeline_fused_cpu_mt", "FKL whole pipeline (fused) 16 threads"),
              ("opencv_pipeline_percrop_parallel", "OpenCV whole pipeline, crops in parallel (16 threads)")]
CROPS_B = [1, 8, 16, 32, 64]


def run_b(rep_dir, write):
    results = {}
    for comp in COMPILERS:
        runs = [os.path.join(rep_dir, comp, f"run{i}") for i in (1, 2, 3)]
        sw = [Sweep(os.path.join(r, "e5_preprocessing_b_CPU.csv")) for r in runs]
        for s in sw[1:]:
            assert s.order == sw[0].order
        results[comp] = (sw, runs)
        if write:
            write_median3(os.path.join(rep_dir, comp, "e5_preprocessing_b_CPU_median_of_3.csv"), sw,
                          f"the sweeps {comp}/run1 (= the rows of pull request #9), {comp}/run2 and {comp}/run3 of the same binary")

    print("## Numerical checks")
    print("| compiler | sweep | comparisons | differ from run1 | required checks failed | log says all passed |")
    print("|---|---|--:|--:|--:|---|")
    for comp in COMPILERS:
        sw, runs = results[comp]
        base = read_agreement(os.path.join(runs[0], "e5_preprocessing_b_agreement_CPU.csv"))
        for i, r in enumerate(runs, 1):
            ag = read_agreement(os.path.join(r, "e5_preprocessing_b_agreement_CPU.csv"))
            assert set(ag) == set(base)
            print(f"| {comp} | {i} | {len(ag)} | {sum(1 for k in base if base[k] != ag[k])} | {sum(1 for v in ag.values() if v[4] == '1' and v[3] != '1')} | "
                  f"{'yes' if log_passed(os.path.join(r, 'stdout.log')) else 'NO'} |")

    print("\n## Median of the three sweeps against run1 (the rows of pull request #9)")
    print("| compiler | points | change \\|D\\| = \\|median of three / run1 - 1\\|: median / 90th percentile / max | spread S = (max-min)/median of the three sweeps: median / 90th percentile / max |")
    print("|---|--:|--|--|")
    pts = []
    for comp in COMPILERS:
        sw, runs = results[comp]
        for key in sw[0].order:
            ms = [s.rows[key]["median"] for s in sw]
            n = statistics.median(ms)
            pts.append(dict(comp=comp, key=key, ms=ms, N=n, D=n / ms[0] - 1, S=(max(ms) - min(ms)) / n))
    for comp in COMPILERS:
        ps = [p for p in pts if p["comp"] == comp]
        ad = [abs(p["D"]) for p in ps]
        ss = [p["S"] for p in ps]
        print(f"| {comp} | {len(ps)} | {fmt(statistics.median(ad))} / {fmt(pct(ad, 0.9))} / {fmt(max(ad))} | {fmt(statistics.median(ss))} / {fmt(pct(ss, 0.9))} / {fmt(max(ss))} |")

    print("\n## Largest differences between the median of three and run1 (the three sweeps; D; S)")
    for comp in COMPILERS:
        for p in sorted((p for p in pts if p["comp"] == comp), key=lambda q: -abs(q["D"]))[:6]:
            print(f"- {comp}, {p['key'][0]} crops, {p['key'][1]}: {p['ms'][0]:.4f} {p['ms'][1]:.4f} {p['ms'][2]:.4f}, median of three {p['N']:.4f}, D {100 * p['D']:+.2f} %, S {fmt(p['S'])}")

    print("\n## Points where the median of three differs from run1 by more than the spread (|D| > S)")
    for comp in COMPILERS:
        bad = [p for p in pts if p["comp"] == comp and abs(p["D"]) > p["S"] + 1e-12]
        if not bad:
            print(f"- {comp}: none")
        for p in bad:
            print(f"- {comp}, {p['key'][0]} crops, {p['key'][1]}: {p['ms'][0]:.4f} {p['ms'][1]:.4f} {p['ms'][2]:.4f}, median of three {p['N']:.4f}, D {100 * p['D']:+.2f} %, S {fmt(p['S'])}")

    print("\n## Points where the largest of the three sweep medians is more than twice the smallest")
    for comp in COMPILERS:
        wide = [p for p in pts if p["comp"] == comp and max(p["ms"]) > 2 * min(p["ms"])]
        if not wide:
            print(f"- {comp}: none")
        for p in wide:
            print(f"- {comp}, {p['key'][0]} crops, {p['key'][1]}: {p['ms'][0]:.4f} {p['ms'][1]:.4f} {p['ms'][2]:.4f}, median of three {p['N']:.4f}")

    print("\n## FKL conversion and whole pipeline (fused): median of three against run1 over the five crop counts")
    print("| compiler | variant | max \\|D\\| over the crop counts | median of three over the crop counts: smallest to largest (ms) | S: median / max |")
    print("|---|---|--:|--|--|")
    for comp in COMPILERS:
        for v in ("fkl_convert_cpu", "fkl_convert_cpu_mt", "fkl_pipeline_fused_cpu", "fkl_pipeline_fused_cpu_mt"):
            ps = [p for p in pts if p["comp"] == comp and p["key"][1] == v]
            assert len(ps) == len(CROPS_B)
            nd = 1 if min(p["N"] for p in ps) >= 20 else 2
            print(f"| {comp} | {v} | {fmt(max(abs(p['D']) for p in ps))} | {min(p['N'] for p in ps):.{nd}f} to {max(p['N'] for p in ps):.{nd}f} "
                  f"| {fmt(statistics.median(p['S'] for p in ps))} / {fmt(max(p['S'] for p in ps))} |")

    print("\n## Median of the three sweeps, ms (smallest to largest of the three medians in brackets)")
    for comp in COMPILERS:
        print(f"\n### {comp}")
        print("| crops | " + " | ".join(n for _, n in HEADLINE_B) + " |")
        print("|--:|" + "--:|" * len(HEADLINE_B))
        for c in CROPS_B:
            cells = []
            for v, _ in HEADLINE_B:
                p = next(q for q in pts if q["comp"] == comp and q["key"] == (c, v))
                nd = 1 if p["N"] >= 20 else (2 if p["N"] >= 1 else 3)
                cells.append(f"{p['N']:.{nd}f} ({min(p['ms']):.{nd}f} to {max(p['ms']):.{nd}f})")
            print(f"| {c} | " + " | ".join(cells) + " |")


# ---------------------------------------------------------------------------------------------------------------- main
def main(argv):
    write = "--write" in argv
    args = [a for a in argv if a != "--write"]
    if len(args) >= 4 and args[1] == "a":
        run_a(args[2], args[3], write)
    elif len(args) >= 3 and args[1] == "b":
        run_b(args[2], write)
    else:
        print(__doc__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
