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
"""The controlled compile repeat against the values the paper quotes today, in the standing form: old value, new value (median of the compiles), the values in the order they ran,
the run-to-run spread, whether the change exceeds the spread and the 5 % flag.

  python scripts/e3_repeat_compare.py results/rtx4090/e3-compile/repeat-c8fcbe8

Reads the compiles of e3_repeat_A.csv / e3_repeat_extra_A.csv and e3_repeat_staticloop.csv / e3_repeat_extra_staticloop.csv and paper-values-before.csv (the values the paper quotes) of the same folder;
writes e3_repeat_vs_paper.csv and e3_repeat_vs_paper.md. Only compiles with status ok enter a value.
spread = (max - min) / median of the compiles of that quantity (for a quantity that is a range over several lengths: the largest spread of those lengths).
change = (new - old) / old; for an old value given as a range (600 to 700 s) the distance of the new median from the range (0 when it is inside); for a new range (StaticLoop over its lengths) against a
single old value the distance of the old value from the range (0 when it is inside). The change exceeds the spread when |change| > spread; the flag is 5 % (a compile is host-bound: nvcc, cicc, ptxas and
g++ on one core).
Units: the CSV columns *_mb are MiB (GNU time reports KiB and the scripts divide by 1024). The paper's "GB" is MiB / 1000 (13,433 MiB -> 13.4); the exact readings, GiB = MiB / 1024 and GB = MiB x 1.048576 / 1000,
are in the note column. Cubin sizes are in MB = 10^6 bytes, as the paper has them.
"""
import csv
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import e3_table as t3  # noqa: E402  (read_csv, num)

REQUESTED_LOOP = [2, 10002, 19902]


def load(d):
    dist, loop = {}, {}
    for fn in ("e3_repeat_A.csv", "e3_repeat_extra_A.csv"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            for r in t3.read_csv(p):
                if r["status"] == "ok":
                    dist.setdefault(int(r["n"]), []).append(r)
    for fn in ("e3_repeat_staticloop.csv", "e3_repeat_extra_staticloop.csv"):
        p = os.path.join(d, fn)
        if os.path.exists(p):
            for r in t3.read_csv(p):
                if r["status"] == "ok":
                    loop.setdefault(int(r["ops"]), []).append(r)
    for g in (dist, loop):
        for k in g:
            g[k].sort(key=lambda r: int(r["rep"]))
    return dist, loop


def col(rows, c, scale=1.0):
    return [t3.num(r[c]) * scale for r in rows if t3.num(r.get(c)) is not None]


def host_share(rows):
    return [100.0 * t3.num(r["phase_gcc_s"]) / t3.num(r["cuda_wall_s"]) for r in rows]


def half_unit(text):
    """half a unit of the last digit of the first number printed in text ("13.3 MB" -> 0.05, "197" -> 0.5): the paper's value is rounded to that"""
    tok = text.replace(",", "").split()[0]
    return 0.5 * 10 ** (-len(tok.split(".")[1])) if "." in tok else 0.5


def spread_pct(v):
    m = statistics.median(v)
    return (max(v) - min(v)) / m * 100 if m else 0.0


def fmt(x):
    return f"{x:,.3f}".rstrip("0").rstrip(".") if abs(x) < 1000 else f"{x:,.1f}"


def main():
    d = sys.argv[1]
    dist, loop = load(d)
    old = {r["key"]: r for r in csv.DictReader(open(os.path.join(d, "paper-values-before.csv"), newline=""))}
    R = REQUESTED_LOOP
    # key -> a scalar: list of the values of the compiles of one length; or a range: (list of per-length value lists)
    scalar = {
        "dist_wall_1": lambda: col(dist[1], "cuda_wall_s"),
        "dist_wall_1000": lambda: col(dist[1000], "cuda_wall_s"),
        "dist_wall_2000": lambda: col(dist[2000], "cuda_wall_s"),
        "dist_wall_3000": lambda: col(dist[3000], "cuda_wall_s"),
        "dist_wall_4000": lambda: col(dist[4000], "cuda_wall_s"),
        "dist_mem_4000": lambda: col(dist[4000], "cuda_peak_rss_mb", 1 / 1000.0),
        "dist_cicc_1000": lambda: col(dist[1000], "phase_cicc_s"),
        "dist_wall_1000_phase": lambda: col(dist[1000], "cuda_wall_s"),
        "dist_cicc_4000": lambda: col(dist[4000], "phase_cicc_s"),
        "dist_wall_4000_phase": lambda: col(dist[4000], "cuda_wall_s"),
        "dist_host_share_4000": lambda: host_share(dist[4000]),
        "dist_ptxas_4000": lambda: col(dist[4000], "phase_ptxas_s"),
        "dist_cubin_1000": lambda: col(dist[1000], "cubin_bytes", 1e-6),
        "dist_cubin_4000": lambda: col(dist[4000], "cubin_bytes", 1e-6),
        "sl_cubin": lambda: col(loop[19902], "cubin_bytes", 1e-6),
    }
    ranged = {   # per requested length
        "sl_wall": lambda n: col(loop[n], "cuda_wall_s"),
        "sl_mem": lambda n: col(loop[n], "cuda_peak_rss_mb", 1 / 1000.0),
        "sl_cudafe": lambda n: col(loop[n], "phase_cudafe_s"),
        "sl_host": lambda n: col(loop[n], "phase_gcc_s"),
    }
    notes = {
        "dist_mem_4000": lambda: "the paper's GB is MiB / 1000; exact: " + " / ".join(f"{v / 1024:.2f} GiB = {v * 1.048576 / 1000:.2f} GB" for v in col(dist[4000], "cuda_peak_rss_mb")) + " (compiles in order)",
        "sl_mem": lambda: "MiB / 1000 as in the paper; exact range in GiB: " + " to ".join(f"{x / 1024:.2f}" for x in (min(statistics.median(col(loop[n], 'cuda_peak_rss_mb')) for n in R), max(statistics.median(col(loop[n], 'cuda_peak_rss_mb')) for n in R))),
    }
    out = []
    for key, o in old.items():
        low = float(o["old_low"]) if o["old_low"] else None
        high = float(o["old_high"]) if o["old_high"] else None
        val = float(o["old_value"]) if o["old_value"] else None
        try:
            if key in scalar:
                v = scalar[key]()
                med, spread = statistics.median(v), spread_pct(v)
                new_text, vals = fmt(med), " / ".join(fmt(x) for x in v)
                if key == "dist_mem_4000":
                    new_text = f"{fmt(med)} ({med * 1000:,.0f} MiB / 1000)"
                if val is not None:
                    change = (med - val) / val * 100
                else:
                    change = 0.0 if low <= med <= high else ((med - high) / high * 100 if med > high else (med - low) / low * 100)
            else:
                per = {n: ranged[key](n) for n in R}
                meds = {n: statistics.median(per[n]) for n in R}
                lo, hi = min(meds.values()), max(meds.values())
                spread = max(spread_pct(per[n]) for n in R)
                new_text = f"{fmt(lo)} to {fmt(hi)}" + (" (MiB / 1000)" if key == "sl_mem" else "")
                vals = "; ".join(f"{n:,} ops: " + " / ".join(fmt(x) for x in per[n]) for n in R)
                if val is not None:
                    change = 0.0 if lo <= val <= hi else ((val - hi) / val * 100 * -1 if val > hi else (lo - val) / val * 100)
                else:
                    ch_lo, ch_hi = (lo - low) / low * 100, (hi - high) / high * 100
                    change = ch_lo if abs(ch_lo) > abs(ch_hi) else ch_hi
        except (KeyError, IndexError, statistics.StatisticsError):
            out.append({"key": key, "quantity": o["quantity"], "paper_old": o["old_text"], "new": "no data", "values_in_order": "", "spread_pct": "", "change_pct": "", "within_rounding_of_old": "",
                        "exceeds_spread": "", "exceeds_5pct": "", "note": ""})
            continue
        # the paper prints its value rounded: a difference smaller than half a unit of its last digit is rounding, not a change
        h = half_unit(o["old_text"])
        new_lo, new_hi = (med, med) if key in scalar else (lo, hi)
        if val is not None:
            within = new_lo - h <= val <= new_hi + h
        elif key in scalar:
            within = low - h <= med <= high + h
        else:
            within = abs(new_lo - low) <= h and abs(new_hi - high) <= h
        out.append({"key": key, "quantity": o["quantity"], "paper_old": o["old_text"], "new": new_text, "values_in_order": vals, "spread_pct": f"{spread:.2f}", "change_pct": f"{change:+.1f}",
                    "within_rounding_of_old": "yes" if within else "no", "exceeds_spread": "no" if (within or abs(change) <= spread) else "yes", "exceeds_5pct": "yes" if (abs(change) > 5 and not within) else "no",
                    "note": notes[key]() if key in notes else ""})
    cols = list(out[0].keys())
    with open(os.path.join(d, "e3_repeat_vs_paper.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(out)
    md = ["# E3 controlled repeat against the values the paper quotes today (generated by scripts/e3_repeat_compare.py)", "",
          "new = median of the compiles; spread = (max - min) / median; change = (new - old) / old, or the distance from the old range; within rounding = the new value is within half a unit of the last digit the paper prints;",
          "exceeds the spread = |change| > spread and not within rounding; flag = 5 % (not within rounding). The memory rows are in the paper's unit, MiB / 1000.", "",
          "| quantity | paper (old) | new | the compiles in order | spread % | change % | within rounding | exceeds the spread | exceeds 5 % | note |", "|---|---|---|---|---:|---:|---|---|---|---|"]
    for r in out:
        md.append(f"| {r['quantity']} | {r['paper_old']} | {r['new']} | {r['values_in_order']} | {r['spread_pct']} | {r['change_pct']} | {r['within_rounding_of_old']} | {r['exceeds_spread']} | {r['exceeds_5pct']} | {r['note']} |")
    open(os.path.join(d, "e3_repeat_vs_paper.md"), "w").write("\n".join(md) + "\n")
    print(f"{len(out)} quantities -> {d}/e3_repeat_vs_paper.csv, .md")


if __name__ == "__main__":
    main()
