# rpp_hand_roi_compare.py: the comparison of results/rx9060/rpp-preprocessing-hand-roi/ (benchmark_rpp_preprocessing_hip with the hand-written
# kernels hand_batch, hand_padded and hand_roi, fkl-benchmarks 151d517, 2026-11-08; every variant timed as the whole call, its build included)
# with the runs of results/rx9060/rpp-preprocessing-hand-batch/ (6dc0541, the same day).
#
# usage: python3 scripts/rpp_hand_roi_compare.py [repository root]   (writes comparison.csv into the folder and prints the tables of its README)
#
# For every source (old: rpp-preprocessing-hand-batch/noise/run0..2; new: rpp-preprocessing-hand-roi/noise/run0..2), batch and variant: the
# median_ms of each of the three runs, their median ("value") and their spread (max - min). A change exceeds the spread when |new - old| >
# max(spread_new, spread_old). The flag threshold is 3 % for a GPU-bound variant and 5 % for a host-bound one (host-bound: the host part,
# build_median_ms, is at least half of the run), as in scripts/rpp_hand_batch_compare.py. The ratios to fkl_fused are taken per run; their value
# is the median of the three runs and their spread the min .. max.
# The program has eight variants now: the rotation of the variants of a point (the first is variant (index of the point + E5_ROTATE_OFFSET) mod
# the number of variants) puts the six variants of the hand_batch runs at other positions than in those runs.
# The pcie_* columns read the GPU function's own link, which is inside the card (to the card's PCIe switch), not the link to the host.
# The fast state: the points of a run in which fkl_fused is more than 4 % faster than the median of the crop-batch runs
# (results/rx9060/rpp-preprocessing-crop-batch/noise/new-run0..2, in which it was rare) at its batch.
import csv
import math
import os
import re
import statistics
import sys
from collections import Counter

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
NEWDIR = os.path.join(ROOT, 'results', 'rx9060', 'rpp-preprocessing-hand-roi')
OLDDIR = os.path.join(ROOT, 'results', 'rx9060', 'rpp-preprocessing-hand-batch')
CROPDIR = os.path.join(ROOT, 'results', 'rx9060', 'rpp-preprocessing-crop-batch')
TIMING = 'benchmark_rpp_preprocessing_a_GPU_AMD.csv'
AGREE = 'benchmark_rpp_preprocessing_a_agreement_GPU_AMD.csv'
OLD_VARIANTS = ['fkl_fused', 'fkl_fused_halfpixel', 'fkl_perop_batch', 'rpp_crop_batch', 'rpp_crop_batch_unsplit', 'hand_batch']
NEW_VARIANTS = OLD_VARIANTS + ['hand_padded', 'hand_roi']
HAND = ['hand_roi', 'hand_batch', 'hand_padded']
SHOWN = (10, 50, 100, 150, 200, 250, 300)
RUNS = ['noise/run0', 'noise/run1', 'noise/run2']


def rows(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(line for line in f if not line.startswith('#')))


def load(dirs):
    """{(batch, variant): [run dicts]} of the timing CSVs of the runs, in run order"""
    out = {}
    for d in dirs:
        for r in rows(os.path.join(d, TIMING)):
            out.setdefault((int(r['batch']), r['variant']), []).append(r)
    return out


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def center(values):
    vals = [v for v in values if not math.isnan(v)]
    return (statistics.median(vals), max(vals) - min(vals)) if vals else (math.nan, math.nan)


SOURCES = {
    'old': [os.path.join(OLDDIR, r) for r in RUNS],
    'new': [os.path.join(NEWDIR, r) for r in RUNS],
    'crop': [os.path.join(CROPDIR, 'noise', f'new-run{k}') for k in range(3)],
    'old natural': [os.path.join(OLDDIR, 'natural-image')],
    'new natural': [os.path.join(NEWDIR, 'natural-image')],
}
DATA = {name: load(dirs) for name, dirs in SOURCES.items()}
BATCHES = sorted({b for (b, _) in DATA['new']})


def stat(source, variant, batch, column='median_ms'):
    runs = DATA[source].get((batch, variant), [])
    return center([num(r[column]) for r in runs]) + (len(runs),)


def fmt(v, digits=4):
    return 'nan' if math.isnan(v) else f'{v:.{digits}f}'


def pct(a, b):
    return math.nan if (math.isnan(a) or math.isnan(b) or b == 0) else 100.0 * (a - b) / b


def host_bound(variant, batch):
    build = stat('new', variant, batch, 'build_median_ms')[0]
    total = stat('new', variant, batch, 'median_ms')[0]
    return (not math.isnan(build)) and build >= 0.5 * total


summary = []

# ---- 1. The hand-written kernels against fkl_fused, per run
print('\n### The hand-written kernels against fkl_fused (per run, then the median of the three runs, min .. max)\n')
print('| batch | hand_roi / fkl_fused | the same, compute parts | hand_batch / fkl_fused | the same, compute | hand_padded / fkl_fused | the same, compute |')
print('|---|---|---|---|---|---|---|')
ranges = {}
for b in BATCHES:
    cells = []
    for v in HAND:
        for column in ('median_ms', 'compute_median_ms'):
            r = [num(x[column]) / num(y[column]) for x, y in zip(DATA['new'].get((b, v), []), DATA['new'].get((b, 'fkl_fused'), []))]
            med = statistics.median(r)
            ranges.setdefault((v, column), []).append(med)
            cells.append(f'{med:.3f} ({min(r):.3f} .. {max(r):.3f})' if column == 'median_ms' else f'{med:.3f} ({min(r):.3f} .. {max(r):.3f})')
            summary.append({'comparison': 'ratio', 'what': f'{v} / fkl_fused ({column})', 'batch': b, 'ratio_median': med, 'ratio_min': min(r),
                            'ratio_max': max(r), 'runs': f'{len(r)}'})
    print(f'| {b} | ' + ' | '.join(cells) + ' |')
print('\nRange over the batches of the median ratio, 10 .. 150 and 10 .. 300:\n')
print('| ratio | 10 .. 150 | 10 .. 300 |')
print('|---|---|---|')
for (v, c), m in ranges.items():
    print(f'| {v} / fkl_fused, {"total" if c == "median_ms" else "compute parts"} | {min(m[:15]):.3f} .. {max(m[:15]):.3f} | {min(m):.3f} .. {max(m):.3f} |')

# ---- 2. The parts
print('\n### fkl_fused and the hand-written kernels: the run and its parts (medians over the three runs of the per-run medians, ms)\n')
print('| batch | fkl_fused total | build | compute | hand_roi total | build | compute | hand_batch compute | hand_padded compute |')
print('|---|---|---|---|---|---|---|---|---|')
for b in BATCHES:
    if b not in SHOWN:
        continue
    vals = [stat('new', 'fkl_fused', b, c)[0] for c in ('median_ms', 'build_median_ms', 'compute_median_ms')]
    vals += [stat('new', 'hand_roi', b, c)[0] for c in ('median_ms', 'build_median_ms', 'compute_median_ms')]
    vals += [stat('new', 'hand_batch', b, 'compute_median_ms')[0], stat('new', 'hand_padded', b, 'compute_median_ms')[0]]
    print(f'| {b} | ' + ' | '.join(fmt(x) for x in vals) + ' |')

# ---- 3. RPP against fkl_fused
print('\n### rpp_crop_batch_unsplit / fkl_fused (per run, then the median of the three runs, min .. max)\n')
print('| batch | these runs | the hand_batch runs |')
print('|---|---|---|')
rr = {}
for b in BATCHES:
    cells = []
    for source in ('new', 'old'):
        r = [num(x['median_ms']) / num(y['median_ms']) for x, y in zip(DATA[source].get((b, 'rpp_crop_batch_unsplit'), []), DATA[source].get((b, 'fkl_fused'), []))]
        med = statistics.median(r)
        rr.setdefault(source, []).append(med)
        cells.append(f'{med:.3f} ({min(r):.3f} .. {max(r):.3f})')
        summary.append({'comparison': 'ratio', 'what': f'{source}: rpp_crop_batch_unsplit / fkl_fused', 'batch': b, 'ratio_median': med,
                        'ratio_min': min(r), 'ratio_max': max(r), 'runs': f'{len(r)}'})
    if b in SHOWN:
        print(f'| {b} | ' + ' | '.join(cells) + ' |')
print('\nRange of the median ratio: ' + '; '.join(f'{s} {min(m[:15]):.3f} .. {max(m[:15]):.3f} at 10 .. 150, {min(m):.3f} .. {max(m):.3f} at 10 .. 300'
                                             for s, m in rr.items()))

# ---- 4. The six variants of the hand_batch runs, old against new
lines = []
for v in OLD_VARIANTS:
    for b in BATCHES:
        n, spn, nn = stat('new', v, b)
        o, spo, no = stat('old', v, b)
        ch = pct(n, o)
        exceeds = abs(n - o) > max(spn, spo)
        hb = host_bound(v, b)
        flag = abs(ch) > (5.0 if hb else 3.0)
        lines.append((v, b, o, spo, n, spn, ch, exceeds, 'host' if hb else 'GPU', flag))
        summary.append({'comparison': 'new vs old', 'what': v, 'batch': b, 'old_ms': o, 'old_spread_ms': spo, 'new_ms': n, 'new_spread_ms': spn,
                        'runs': f'{no}+{nn}', 'change_pct': ch, 'exceeds_spread': int(exceeds), 'class': 'host' if hb else 'GPU', 'flag': int(flag)})
print('\n### The six variants of the hand_batch runs: old (rpp-preprocessing-hand-batch) against new (these runs)\n')
print('| variant | batch | old ms (spread) | new ms (spread) | change | exceeds spread | class | flag |')
print('|---|---|---|---|---|---|---|---|')
for v, b, o, spo, n, spn, ch, exceeds, cls, flag in lines:
    if b in SHOWN:
        print(f'| {v} | {b} | {fmt(o)} ({fmt(spo)}) | {fmt(n)} ({fmt(spn)}) | {ch:+.1f} % | {"yes" if exceeds else "no"} | {cls} | {"**yes**" if flag else "no"} |')
print(f'\nOver the {len(BATCHES)} batches:\n')
print('| variant | change, min .. median .. max | batches beyond the spread | flagged |')
print('|---|---|---|---|')
for v in OLD_VARIANTS:
    sel = [x for x in lines if x[0] == v]
    chs = [x[6] for x in sel]
    print(f'| {v} | {min(chs):+.1f} .. {statistics.median(chs):+.1f} .. {max(chs):+.1f} % | {sum(x[7] for x in sel)} of {len(sel)} | '
          f'{sum(x[9] for x in sel)} of {len(sel)} |')

# ---- 5. The fast state
print('\n### The fast state (the points in which fkl_fused is more than 4 % below the median of the crop-batch runs at its batch)\n')
pts = []
for k in range(3):
    marks = ''
    for b in BATCHES:
        f = num(DATA['new'][(b, 'fkl_fused')][k]['median_ms'])
        fast = f < 0.96 * statistics.median([num(x['median_ms']) for x in DATA['crop'][(b, 'fkl_fused')]])
        marks += 'F' if fast else '.'
        if fast:
            pts.append((k, b))
    print(f'    run{k}, N = 10 .. 300: {marks}')
print(f'\n{len(pts)} of {3 * len(BATCHES)} points. Each variant in those points and in the others, against the median of its three runs at its batch:\n')
print('| variant | in the fast points | in the other points |')
print('|---|---|---|')
for v in NEW_VARIANTS:
    med = {b: statistics.median([num(x['median_ms']) for x in DATA['new'][(b, v)]]) for b in BATCHES}
    fr = [num(DATA['new'][(b, v)][k]['median_ms']) / med[b] - 1 for (k, b) in pts]
    orr = [num(DATA['new'][(b, v)][k]['median_ms']) / med[b] - 1 for k in range(3) for b in BATCHES if (k, b) not in pts]
    print(f'| {v} | {100 * statistics.median(fr):+.1f} % | {100 * statistics.median(orr):+.1f} % |')
    summary.append({'comparison': 'fast state', 'what': v, 'fast_points_pct': 100 * statistics.median(fr), 'other_points_pct': 100 * statistics.median(orr),
                    'runs': f'{len(pts)} fast points'})
for name, sel in (('fast points', pts), ('other points', [(k, b) for k in range(3) for b in BATCHES if (k, b) not in pts])):
    rs = [DATA['new'][(b, 'fkl_fused')][k] for (k, b) in sel]
    print(f'\nfkl_fused rows in the {name}: mclk at the start of the timed runs {dict(Counter(int(num(r["mclk_timed_start_mhz"])) for r in rs))}, '
          f'after them {dict(Counter(int(num(r["mclk_end_mhz"])) for r in rs))}; card-internal PCIe link {dict(Counter((num(r["pcie_gts_timed_start"]), num(r["pcie_width_timed_start"])) for r in rs))}; '
          f'sclk median {statistics.median(num(r["sclk_timed_start_mhz"]) for r in rs):.0f} MHz')

# ---- 6. RPP's ROI padding and pinned arrays (stdout of every run)
print('\n### RPP\'s ROI padding and its other pinned arrays (stdout of every run)\n')
print('| run | batches | "other pinned parameter arrays are unchanged" | padding ROIs rewritten = 256 ceil(N / 256) - N |')
print('|---|---|---|---|')
pad = re.compile(r"RPP's other pinned parameter arrays are unchanged after its calls; RPP rewrote (\d+) of the (\d+) padding ROIs")
for name in RUNS + ['natural-image']:
    text = open(os.path.join(NEWDIR, name, 'stdout.log')).read()
    batches = [int(x) for x in re.findall(r'^\(a\) batch (\d+)$', text, re.M)]
    found = [(int(a), int(p)) for a, p in pad.findall(text)]
    exact = sum(1 for (a, p), b in zip(found, batches) if a == p == 256 * math.ceil(b / 256) - b)
    print(f'| {name} | {len(batches)} | {len(found)} of {len(batches)} | {exact} of {len(batches)} |')

# ---- 7. Agreement
print('\n### Agreement (maximum over the runs and batches of max_abs_diff; required rows passed)\n')
print('| comparison | max | required rows passed | bitwise equal (batches x runs) |')
print('|---|---|---|---|')
agg = {}
for name in RUNS + ['natural-image']:
    for r in rows(os.path.join(NEWDIR, name, AGREE)):
        m, p, n, bits = agg.get(r['comparison'], (0.0, 0, 0, 0))
        d = num(r['max_abs_diff'])
        agg[r['comparison']] = (math.nan if (math.isnan(d) or math.isnan(m)) else max(m, d), p + (r['pass'] == '1'), n + (r['pass'] != '-1'),
                                bits + ('bitwise equal' in r['note'] and r['region'] == 'interior'))
for comp, (m, p, n, bits) in agg.items():
    print(f'| {comp} | {m:.3g} | {f"{p} of {n}" if n else "informational"} | {bits if comp.endswith("vs fkl_fused") and comp.startswith("hand") else ""} |')

# ---- 8. The natural image
print('\n### The natural-image run (one run each; no spread)\n')
nat = {v: [stat('new natural', v, b)[0] / stat('new natural', 'fkl_fused', b)[0] for b in BATCHES] for v in HAND}
for v, m in nat.items():
    print(f'- {v} / fkl_fused over the 30 batches: {min(m):.3f} .. {statistics.median(m):.3f} .. {max(m):.3f} (min .. median .. max)')
ch = [pct(stat('new natural', v, b)[0], stat('old natural', v, b)[0]) for v in OLD_VARIANTS for b in BATCHES]
print(f'- the six variants against the hand_batch natural-image run: {min(ch):+.1f} .. {statistics.median(ch):+.1f} .. {max(ch):+.1f} %')

# ---- 9. Clocks and warm-up
print('\n### Clock, card-internal PCIe link and warm-up columns (min .. max over every row of a source)\n')
print('| source | sclk at the start of the timed runs, MHz | mclk at the start of the timed runs, MHz | card-internal PCIe link, GT/s x width | warm-up, s | rows with warmup_steady = 0 |')
print('|---|---|---|---|---|---|')
for source in ('new', 'new natural'):
    rs = [r for v in DATA[source].values() for r in v]
    ts = [num(r['sclk_timed_start_mhz']) for r in rs]
    ms = [num(r['mclk_timed_start_mhz']) for r in rs]
    pc = sorted({(num(r['pcie_gts_start']), num(r['pcie_width_start'])) for r in rs} | {(num(r['pcie_gts_end']), num(r['pcie_width_end'])) for r in rs})
    ws = [num(r['warmup_s']) for r in rs]
    print(f'| {source} | {min(ts):.0f} .. {max(ts):.0f} | {min(ms):.0f} .. {max(ms):.0f} | {", ".join(f"{g:.0f} x{w:.0f}" for g, w in pc)} | '
          f'{min(ws):.2f} .. {max(ws):.2f} | {sum(1 for r in rs if r.get("warmup_steady") == "0")} of {len(rs)} |')

keys = []
for s in summary:
    for k in s:
        if k not in keys:
            keys.append(k)
with open(os.path.join(NEWDIR, 'comparison.csv'), 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for s in summary:
        w.writerow({k: (f'{v:.6g}' if isinstance(v, float) else v) for k, v in s.items()})
