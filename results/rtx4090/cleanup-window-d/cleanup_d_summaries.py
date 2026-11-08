#!/usr/bin/env python3
"""Summary files of the cleaned build for the paper (cleanup windows C and D, 2026-11-08). Reads only the result folders; writes:
  cleanup-window-d/cleaned/e1_kernel_only_ratios.csv  E1: FKL's time over each hand-written kernel's, per point, from the three cleaned runs of window C
      (cleanup-window-c/cleaned/e1-*/run1..3): the kernel-only columns (*_kernel_mean_ms) and, for comparison, the normal columns (*_mean_ms); per run, then the
      median of the three, the smallest, the largest and the spread ((largest - smallest) / median, %).
  cleanup-window-d/cleaned/e2_per_k_summary.csv  E2: per k and version, warm with the launch (gpu_mean_ms), warm kernel-only (gpu_kernel_mean_ms) and L2-flushed
      (gpu_flushed_mean_ms), each the median of three runs with the spread; k = 2, 10, 100, 500 from window C (cleanup-window-c/cleaned/e2-python/run1..3), k = 5,002
      from window D (cleanup-window-d/cleaned/e2-python-k5002/run1..3). The fkl rows are the FKL program's times (fkl_mean_ms, fkl_kernel_mean_ms,
      fkl_flushed_mean_ms of the cleaned FKL run n, merged into the driver's run n)."""
import csv
import statistics

R = "/home/oscar/tfkl-bench/fkl-benchmarks/results/rtx4090"
C = R + "/cleanup-window-c"
D = R + "/cleanup-window-d"


def rows(path, expect):
    lines = [l for l in open(path) if l.strip() and not l.startswith("#")]
    out = list(csv.DictReader(lines))
    assert len(out) == expect, (path, len(out))
    assert not any("INCOMPLETE" in l for l in open(path)), path
    return out


def summary(values):
    med = statistics.median(values)
    return med, min(values), max(values), (max(values) - min(values)) / med * 100


# E1
E1 = (("E1a", "e1-vf-chain", "e1_vf_chain.csv", "ops", 6, ("hand_params", "hand_imm")),
      ("E1b", "e1-hf-batch", "e1_hf_batch.csv", "batch", 5, ("hand", "hand_generic")),
      ("E1c", "e1-preproc", "e1_preproc.csv", "batch", 5, ("hand", "hand_generic", "hand_padded")))
out = []
for exp, name, fname, key, n, refs in E1:
    runs = [{int(r[key]): r for r in rows(f"{C}/cleaned/{name}/run{i}/{fname}", n)} for i in (1, 2, 3)]
    for point in sorted(runs[0]):
        for ref in refs:
            k = summary([float(x[point]["fkl_kernel_mean_ms"]) / float(x[point][f"{ref}_kernel_mean_ms"]) for x in runs])
            m = summary([float(x[point]["fkl_mean_ms"]) / float(x[point][f"{ref}_mean_ms"]) for x in runs])
            out.append({"exp": exp, "point": point, "reference": ref,
                        "kernel_only_ratio_median": f"{k[0]:.4f}", "kernel_only_ratio_min": f"{k[1]:.4f}", "kernel_only_ratio_max": f"{k[2]:.4f}", "kernel_only_ratio_spread_pct": f"{k[3]:.2f}",
                        "normal_ratio_median": f"{m[0]:.4f}", "normal_ratio_min": f"{m[1]:.4f}", "normal_ratio_max": f"{m[2]:.4f}", "normal_ratio_spread_pct": f"{m[3]:.2f}",
                        "runs": "cleanup-window-c/cleaned/" + name + "/run1..3"})
with open(f"{D}/cleaned/e1_kernel_only_ratios.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print(f"e1_kernel_only_ratios.csv: {len(out)} rows")

# E2
VERSIONS = ("fkl", "triton_hand", "inductor_default", "inductor_wide", "inductor_graphs", "eager", "eager_graphs")
sets = [(k, [f"{C}/cleaned/e2-python/run{i}/e2_chain.csv" for i in (1, 2, 3)], 35, "cleanup-window-c/cleaned/e2-python/run1..3") for k in (2, 10, 100, 500)] + \
       [(5002, [f"{D}/cleaned/e2-python-k5002/run{i}/e2_chain.csv" for i in (1, 2, 3)], 7, "cleanup-window-d/cleaned/e2-python-k5002/run1..3")]
out = []
for k, paths, n, src in sets:
    runs = [{(int(r["ops"]), r["version"]): r for r in rows(p, n)} for p in paths]
    found = sorted({v for (kk, v) in runs[0] if kk == k})
    assert set(found) == set(VERSIONS), (k, found)
    for v in VERSIONS:
        assert all(x[(k, v)]["status"] == "ok" for x in runs), (k, v)
        row = {"k": k, "version": v}
        for col, name in (("gpu_mean_ms", "warm"), ("gpu_kernel_mean_ms", "kernel_only"), ("gpu_flushed_mean_ms", "flushed")):
            s = summary([float(x[(k, v)][col]) for x in runs])
            row[f"{name}_ms_median"] = f"{s[0]:.6g}"
            row[f"{name}_spread_pct"] = f"{s[3]:.2f}"
        row["runs"] = src
        out.append(row)
with open(f"{D}/cleaned/e2_per_k_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader()
    w.writerows(out)
print(f"e2_per_k_summary.csv: {len(out)} rows")
