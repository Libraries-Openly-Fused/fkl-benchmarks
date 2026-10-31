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
"""Audit of the load on the benchmark machine during the E0 (and E1/E2) measurement windows, from the record of the session that ran them.

  python scripts/audit_load.py --transcript <session>.jsonl --run-log results/rtx4090/e0-run.log --out results/rtx4090/e0-load-audit

Nothing recorded the load average while the E0 experiments ran (the window scripts record it from 2026-10-31 on). What exists is (1) results/rtx4090/e0-run.log: the start and end of every E0 experiment
and of every GPU window, with the cores the runner was pinned to (0-3,16-19 = four cores of the first CCD of the Ryzen 9 5950X and their SMT siblings), and (2) the transcript of the session, which holds
every shell command that was run, with the time it was sent and the time its result came back. This script lists the commands that load the CPU (compilation, builds, test suites, sweeps, profilers, Python
with Triton or PyTorch), with the cores they were pinned to (taskset -c), the number of parallel jobs (-j), whether they ran in the foreground (start and end from the transcript), in the background
(end from the completion notice) or detached (nohup ... &: the end is the last modification time of the log file that the command writes, when that file still exists), and intersects them with the
windows of the E0 experiments. It cannot see anything that did not go through the session (the desktop session, other users); it is a record of what the session ran, not a measurement of the load.
CCD of a core number on this CPU: cores 0-7 and 16-23 are the first CCD, 8-15 and 24-31 the second.
"""
import argparse
import csv
import datetime as dt
import json
import os
import re
import sys

ZONE = dt.timezone(dt.timedelta(hours=1))      # CET: the logs are +01:00 (the clocks changed on 2026-10-25)
CCD0 = set(range(0, 8)) | set(range(16, 24))
E0_CORES = set(range(0, 4)) | set(range(16, 20))
HEAVY = re.compile(r"cmake\s+(--build|-S|--preset)|\bmake\b|\bninja\b|\bnvcc\b|\bg\+\+|\bclang|\bgcc\b|e3_sweep|gen_chain|build[-_a-z0-9]*\.sh|\bctest\b|\bncu\b|\bnsys\b|compute-sanitizer|"
                   r"e2_chain|triton|torch|pytest|pip install|\bcuobjdump\b|sass_compare|nvdisasm|bin/[A-Za-z0-9_]+_cu\b|run_e0|pin_transfer|regression_build|git\s+(clone|submodule)")
LIGHT = re.compile(r"^\s*(cat|ls|grep|head|tail|sed|echo|wc|python3? -c|git (log|status|diff|show|fetch|ls-tree)|date|stat|sleep)\b")
CORES_RX = re.compile(r"taskset\s+-c\s+([0-9,\-]+)")
J_RX = re.compile(r"(?:-j\s*|--parallel\s+|--jobs\s+)(\d+)")
REDIR_RX = re.compile(r">\s*(/[^\s;&|\"']+\.(?:log|txt|csv|out))")


KINDS = [   # (kind, regex, counts as CPU load)
    ("measurement script (not a load)", re.compile(r"run_e0\.sh|gpu-window[-_a-z0-9]*\.sh|queue-v\d\.sh|e0-run\.log"), False),
    ("waiting or polling (not a load)", re.compile(r"^\s*(until|while)\b|\bsleep \d|ReadNotifications|\bwait\b\s*$"), False),
    ("E3 compile sweep (one nvcc, --threads 1, plus the host compiler)", re.compile(r"e3_sweep|e3_trees|e3_check|gen_chain"), True),
    ("build (cmake --build, make, ninja)", re.compile(r"cmake\s+--build|\bmake\b|\bninja\b|cmake\s+(-S|--preset)"), True),
    ("ctest", re.compile(r"\bctest\b"), True),
    ("nvcc / host compiler directly", re.compile(r"\bnvcc\b|\bg\+\+|\bclang\+\+|\bgcc\b"), True),
    ("profiler (ncu, nsys)", re.compile(r"\bncu\b|\bnsys\b|compute-sanitizer"), True),
    ("python with torch or triton", re.compile(r"venv-torch|e2_chain|triton|torch\."), True),
    ("cuobjdump / SASS tools", re.compile(r"cuobjdump|nvdisasm|sass_compare|sass_tools|sass_count"), True),
    ("GPU program run by hand", re.compile(r"bin/[A-Za-z0-9_]+_cu\b|build/[A-Za-z0-9_\-]+/bin/"), True),
    ("git clone or submodule", re.compile(r"git\s+(clone|submodule)"), True),
]
CORE_ARG_RX = re.compile(r"--core\s+(\d+)")


def classify_command(cmd):
    for kind, rx, load in KINDS:
        if rx.search(cmd):
            return kind, load
    return "other", False


def parse_cores(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(ZONE)


def load_windows(path):
    """(name, start, end) of the E0 experiments from e0-run.log."""
    wins, open_ = [], {}
    rx_start = re.compile(r"^(\S+)\s+(e[0-9][a-z0-9\-]*): (?!exit|paused|\S*is not)(.+)$")
    rx_end = re.compile(r"^(\S+)\s+(e[0-9][a-z0-9\-]*)(?: \([^)]*\))?: exit (\d+) after (\d+) s")
    for line in open(path):
        m = rx_end.match(line)
        if m:
            name = m.group(2)
            end = dt.datetime.fromisoformat(m.group(1))
            start = end - dt.timedelta(seconds=int(m.group(4)))
            wins.append((name, start, end, m.group(1)))
            open_.pop(name, None)
    return sorted(wins, key=lambda w: w[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--transcript", required=True)
    ap.add_argument("--run-log", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--from", dest="t0", default="2026-10-29T00:00:00+01:00")
    ap.add_argument("--to", dest="t1", default="2026-10-31T01:00:00+01:00")
    args = ap.parse_args()
    t0, t1 = dt.datetime.fromisoformat(args.t0), dt.datetime.fromisoformat(args.t1)
    uses, results, notices = {}, {}, {}
    with open(args.transcript, errors="replace") as f:
        for line in f:
            if '"type":"assistant"' not in line and '"type":"user"' not in line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            when = ts(d["timestamp"]) if "timestamp" in d else None
            msg = d.get("message") or {}
            content = msg.get("content")
            if not isinstance(content, list):
                if isinstance(content, str) and "<task-notification>" in content and when:
                    m = re.search(r"<tool-use-id>(toolu_\w+)</tool-use-id>", content)
                    if m:
                        notices[m.group(1)] = when
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                if c.get("type") == "tool_use" and c.get("name") == "Bash" and when:
                    uses[c["id"]] = (when, c.get("input", {}))
                elif c.get("type") == "tool_result" and when:
                    results[c.get("tool_use_id")] = when
                elif c.get("type") == "text" and "<task-notification>" in (c.get("text") or "") and when:
                    m = re.search(r"<tool-use-id>(toolu_\w+)</tool-use-id>", c["text"])
                    if m:
                        notices[m.group(1)] = when
    events = []
    for tid, (start, inp) in uses.items():
        cmd = inp.get("command", "")
        if not (t0 <= start <= t1) or not HEAVY.search(cmd):
            continue
        kind, is_load = classify_command(cmd)
        if not is_load:
            continue
        detached = bool(re.search(r"nohup|\)\s*&\s*$|&\s*\)\s*$|setsid", cmd)) and not inp.get("run_in_background")
        background = bool(inp.get("run_in_background"))
        end, how = results.get(tid), "foreground (result time)"
        if background:
            end, how = notices.get(tid, results.get(tid)), "background (completion notice)" if tid in notices else "background (end not recorded)"
        elif detached:
            end, how = None, "detached"
            for p in REDIR_RX.findall(cmd):
                if os.path.exists(p):
                    mt = dt.datetime.fromtimestamp(os.path.getmtime(p), ZONE)
                    if mt > start and (end is None or mt > end):
                        end, how = mt, f"detached (last write of {os.path.basename(p)})"
        cores = CORES_RX.findall(cmd) or ["%s" % m for m in CORE_ARG_RX.findall(cmd)]
        cs = parse_cores(cores[0]) if cores else None
        j = J_RX.findall(cmd)
        events.append(dict(kind=kind, start=start, end=end, how=how, cores=cores[0] if cores else "unpinned", ccd0_overlap=("E0 cores" if cs and cs & E0_CORES else "CCD0 other cores" if cs and cs & CCD0 else "CCD1" if cs else "unpinned: any core"),
                           jobs=max(map(int, j)) if j else "", command=re.sub(r"\s+", " ", cmd)[:240]))
    events.sort(key=lambda e: e["start"])
    wins = load_windows(args.run_log)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "load_events.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["start", "end", "kind", "how_the_end_is_known", "cores", "relation_to_e0_cores", "parallel_jobs", "command"])
        for e in events:
            w.writerow([e["start"].isoformat(timespec="seconds"), e["end"].isoformat(timespec="seconds") if e["end"] else "", e["kind"], e["how"], e["cores"], e["ccd0_overlap"], e["jobs"], e["command"]])
    rows = []
    for name, ws, we, _ in wins:
        if (we - ws).total_seconds() < 0:
            continue
        for e in events:
            es, ee = e["start"], e["end"] or e["start"]
            if ee <= ws or es >= we:
                continue
            ov = (min(we, ee) - max(ws, es)).total_seconds()
            rows.append(dict(kind=e["kind"], experiment=name, window_start=ws.isoformat(timespec="seconds"), window_end=we.isoformat(timespec="seconds"), window_s=int((we - ws).total_seconds()), overlap_s=int(ov),
                             share_pct=f"{100 * ov / (we - ws).total_seconds():.0f}", end_known=("yes" if e["end"] else "no"), cores=e["cores"], relation=e["ccd0_overlap"], jobs=e["jobs"], command=e["command"][:160]))
    with open(os.path.join(args.out, "overlaps.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["experiment"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(uses)} shell commands in the transcript, {len(events)} of them load the CPU between {args.t0} and {args.t1}, {len(wins)} E0 experiment windows, {len(rows)} overlaps")

    def union_seconds(intervals):
        total, cur_s, cur_e = 0.0, None, None
        for a, b in sorted(intervals):
            if cur_e is None or a > cur_e:
                if cur_e is not None:
                    total += (cur_e - cur_s).total_seconds()
                cur_s, cur_e = a, b
            else:
                cur_e = max(cur_e, b)
        if cur_e is not None:
            total += (cur_e - cur_s).total_seconds()
        return total

    summary = []
    for name, ws, we, _ in wins:
        dur = (we - ws).total_seconds()
        evs = [e for e in events if e["end"] and e["end"] > ws and e["start"] < we]
        noend = [e for e in events if not e["end"] and ws <= e["start"] < we]
        row = dict(experiment=name, start=ws.isoformat(timespec="seconds"), end=we.isoformat(timespec="seconds"), seconds=int(dur))
        for rel in ("E0 cores", "CCD0 other cores", "CCD1", "unpinned: any core"):
            iv = [(max(ws, e["start"]), min(we, e["end"])) for e in evs if e["ccd0_overlap"] == rel]
            row["loaded_s_" + rel.split(":")[0].replace(" ", "_")] = int(union_seconds(iv))
        iv_all = [(max(ws, e["start"]), min(we, e["end"])) for e in evs]
        row["loaded_s_any"] = int(union_seconds(iv_all))
        row["loaded_share_pct"] = f"{100 * row['loaded_s_any'] / dur:.0f}"
        row["max_parallel_jobs"] = max([int(e["jobs"]) for e in evs if str(e["jobs"]).isdigit()] or [0])
        row["kinds"] = "; ".join(sorted({e["kind"] for e in evs}))
        row["commands_without_known_end_that_started_inside"] = len(noend)
        summary.append(row)
    with open(os.path.join(args.out, "window_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for r in summary:
        print(f"{r['experiment']:24s} {r['start'][5:16]}-{r['end'][11:16]} {r['seconds']:6d} s: loaded {r['loaded_share_pct']:>3s} % (E0 cores {r['loaded_s_E0_cores']} s, CCD0 other {r['loaded_s_CCD0_other_cores']} s, CCD1 {r['loaded_s_CCD1']} s, unpinned {r['loaded_s_unpinned']} s); max -j {r['max_parallel_jobs']}; {r['kinds'][:110]}")


if __name__ == "__main__":
    main()
