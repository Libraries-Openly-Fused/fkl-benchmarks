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
"""Which points of the long vertical-fusion runs are not smooth, and when were they measured?

  python scripts/audit_smoothness.py --results results/rtx4090 --events results/rtx4090/e0-load-audit/load_events.csv --out results/rtx4090/e0-load-audit

In the plain-launch pass of the VF benchmarks the OpenCV-CUDA baseline launches 2 kernels per fused operation (per crop) and the time per call is the time the host needs to enqueue them: the OpenCV time
per fused operation is a constant of the machine (here 0.098 ms per operation at batch 50), whatever the number of operations. A point whose time per operation differs from the median of its 8 neighbours
(4 on each side) by more than 2 % is irregular. The time at which each point was measured is reconstructed from the point's own times: a run executes, for every pass (plain warm-up with 1 iteration,
plain with 100, graphs warm-up with 1, graphs with 100), the points in order of increasing number of operations, and the duration of a point is its iterations times (OpenCV + cvGS mean time) plus a
setup per point that is the same for all (estimated so that the total equals the run's recorded duration). The result is an estimate (a few minutes), good enough to tell which of the recorded commands of the session
(load_events.csv) were running while an irregular point was being measured.
"""
import argparse
import csv
import datetime as dt
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_common as pc  # noqa: E402

RUNS = [  # label, folder, plain csv, graphs csv, window start, window end (from e0-run.log)
    ("e0-vf-mul-add", "e0-vf-mul-add", "benchmark_vertical_fusion_loopMulAdd.csv", "benchmark_vertical_fusion_loopMulAdd_CUDAGraphs.csv", "2026-10-30T15:11:47+01:00", "2026-10-30T21:39:24+01:00"),
    ("e0-vf-mul-add1 (run 1)", "e0-vf-mul-add1-run1", "benchmark_vertical_fusion_loopMulAdd1.csv", "benchmark_vertical_fusion_loopMulAdd1_CUDAGraphs.csv", "2026-10-30T10:45:47+01:00", "2026-10-30T12:53:50+01:00"),
    ("e0-vf-mul-add1 (run 2)", "e0-vf-mul-add1", "benchmark_vertical_fusion_loopMulAdd1.csv", "benchmark_vertical_fusion_loopMulAdd1_CUDAGraphs.csv", "2026-10-30T21:56:49+01:00", "2026-10-31T00:04:32+01:00"),
    ("e0-vf-mul1", "e0-vf-mul1", "benchmark_vertical_fusion_loopMul1.csv", "benchmark_vertical_fusion_loopMul1_CUDAGraphs.csv", "2026-10-30T13:00:57+01:00", "2026-10-30T15:10:00+01:00"),
]


def series(path):
    _, blocks = pc.read_blocks(path)
    b = blocks[0]
    h = b[0]
    return dict(ops=pc.column(b, h[0]), ocv=pc.column(b, "OpenCV MeanTime"), omin=pc.column(b, "OpenCV MinTime"), omax=pc.column(b, "OpenCV MaxTime"), ovar=pc.column(b, "OpenCV TimeVariance"),
                cvgs=pc.column(b, "cvGS MeanTime"), cmin=pc.column(b, "cvGS MinTime"), cmax=pc.column(b, "cvGS MaxTime"), speedup=pc.column(b, "Mean Speedup"))


def irregular(per, k=4, thr=0.02):
    out = []
    for i in range(len(per)):
        nb = per[max(0, i - k):i] + per[i + 1:i + 1 + k]
        out.append(per[i] / st.median(nb) - 1)
    return [(i, d) for i, d in enumerate(out) if abs(d) > thr], out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    events = list(csv.DictReader(open(args.events)))
    for e in events:
        e["s"] = dt.datetime.fromisoformat(e["start"])
        e["e"] = dt.datetime.fromisoformat(e["end"]) if e["end"] else None
    lines = []
    rows = []
    for label, folder, plain, graphs, w0, w1 in RUNS:
        t0, t1 = dt.datetime.fromisoformat(w0), dt.datetime.fromisoformat(w1)
        try:
            sp, sg = series(os.path.join(args.results, folder, plain)), series(os.path.join(args.results, folder, graphs))
        except (OSError, ValueError, IndexError) as exc:
            lines.append(f"{label}: cannot read ({exc})")
            continue
        n = len(sp["ops"])
        # cost model of the four passes (seconds): iterations * (OpenCV + cvGS) per point, in ms -> s
        passes = [("plain warm-up", sp, 1), ("plain", sp, 100), ("graphs warm-up", sg, 1), ("graphs", sg, 100)]
        costs = [[it * (s["ocv"][i] + s["cvgs"][i]) / 1000.0 for i in range(n)] for _, s, it in passes]
        total_model = sum(sum(c) for c in costs)
        total_run = (t1 - t0).total_seconds()
        setup = max(0.0, (total_run - total_model) / (4 * n))
        starts = {}
        t = 0.0
        for (pname, _, _), c in zip(passes, costs):
            for i in range(n):
                starts[(pname, i)] = t
                t += c[i] + setup
        per = [o / x for o, x in zip(sp["ocv"], sp["ops"])]
        bad, dev = irregular(per)
        perg = [o / x for o, x in zip(sg["ocv"], sg["ops"])]
        badg, devg = irregular(perg)
        lines.append(f"== {label}: {n} points, window {t0:%m-%d %H:%M} to {t1:%H:%M} ({int(total_run)} s); model total {int(total_model)} s, setup per point and pass {setup:.2f} s")
        lines.append(f"   plain pass: OpenCV time per operation median {st.median(per):.5f} ms; {len(bad)} irregular points (more than 2 % from the median of the 8 neighbours); graphs pass: {len(badg)}")
        for i, d in bad:
            st_, en_ = t0 + dt.timedelta(seconds=starts[("plain", i)]), t0 + dt.timedelta(seconds=starts[("plain", i)] + costs[1][i] + setup)
            ov = [e for e in events if e["s"] < en_ and (e["e"] or e["s"]) > st_ and (e["e"] or e["s"]) - e["s"] >= dt.timedelta(seconds=0)]
            ovs = "; ".join(sorted({f"{e['start'][11:16]} {e['relation_to_e0_cores'].split(':')[0]} {e['cores'][:10]} -j{e['parallel_jobs']} {e['kind'][:14]}" for e in ov}))[:200]
            nb = per[max(0, i - 4):i] + per[i + 1:i + 5]
            corrected = st.median(nb) * sp["ops"][i] / sp["cvgs"][i]
            lines.append(f"   ops {int(sp['ops'][i]):6d}  {d * 100:+6.1f} %  measured {st_:%H:%M}-{en_:%H:%M}  speedup {sp['speedup'][i]:9.2f} (OpenCV time at the neighbours' rate: {corrected:9.2f})  inside-point max/min of OpenCV {sp['omax'][i] / sp['omin'][i]:.2f}  overlapping commands: {ovs or 'none recorded'}")
            rows.append(dict(run=label, pass_="plain", ops=int(sp["ops"][i]), deviation_pct=f"{d * 100:.1f}", measured_from=st_.isoformat(timespec="minutes"), measured_to=en_.isoformat(timespec="minutes"),
                             speedup=f"{sp['speedup'][i]:.2f}", speedup_at_neighbours_rate=f"{corrected:.2f}", overlapping=ovs))
        for i, d in badg[:10]:
            lines.append(f"   (graphs pass) ops {int(sg['ops'][i]):6d}  {d * 100:+6.1f} %")
        top = max(range(n), key=lambda i: sp["speedup"][i])
        lines.append(f"   largest speedup of the plain pass: {sp['speedup'][top]:.2f} at {int(sp['ops'][top])} operations ({'irregular' if any(i == top for i, _ in bad) else 'regular'} point)")
    text = "\n".join(lines)
    print(text)
    os.makedirs(args.out, exist_ok=True)
    open(os.path.join(args.out, "smoothness.txt"), "w").write(text + "\n")
    if rows:
        with open(os.path.join(args.out, "irregular_points.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
