# rpp_built_once_compare.py: the analysis of results/rx9060/rpp-preprocessing-built-once/ (benchmark_rpp_preprocessing_hip at fkl-benchmarks bdc7d19,
# 2026-11-08: the whole-call variants of #36 and the built-once variants fkl_fused_halfpixel_once and hand_roi_halfpixel_once of #38), the RX 9060
# dataset of Figure 15, and its comparison with the whole-call record results/rx9060/rpp-preprocessing-hand-roi/ (151d517, the same day).
#
# usage: python3 scripts/rpp_built_once_compare.py [repository root]   (writes comparison.csv into the folder and prints the tables of its README)
#
# The series of the figure, per batch: rpp_crop_batch_unsplit / fkl_fused_halfpixel_once and hand_roi_halfpixel_once / fkl_fused_halfpixel_once,
# each taken per run; its value is the median of the three runs and its spread the min .. max. A ratio departs from 1 beyond the spread when
# all three runs lie on the same side of 1.
# Old against new (the eight variants of the whole-call record): value = the median of the median_ms of the three runs, spread = max - min; a
# change exceeds the spread when |new - old| > max(spread_new, spread_old); flag threshold 3 % for a GPU-bound variant, 5 % for a host-bound one
# (build_median_ms at least half of the run), as in scripts/rpp_hand_roi_compare.py. The program has ten variants now: the rotation of the
# variants of a point (the first is variant (index of the point + E5_ROTATE_OFFSET) mod the number of variants) puts the eight variants of the
# whole-call record at other positions than in those runs.
# The fast state: the points of a run in which fkl_fused is more than 4 % below the median of the crop-batch runs
# (results/rx9060/rpp-preprocessing-crop-batch/noise/new-run0..2) at its batch.
import csv
import math
import os
import re
import statistics
import sys
from collections import Counter

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
NEWDIR = os.path.join(ROOT, 'results', 'rx9060', 'rpp-preprocessing-built-once')
OLDDIR = os.path.join(ROOT, 'results', 'rx9060', 'rpp-preprocessing-hand-roi')
CROPDIR = os.path.join(ROOT, 'results', 'rx9060', 'rpp-preprocessing-crop-batch')
TIMING = 'benchmark_rpp_preprocessing_a_GPU_AMD.csv'
AGREE = 'benchmark_rpp_preprocessing_a_agreement_GPU_AMD.csv'
OLD_VARIANTS = ['fkl_fused', 'fkl_fused_halfpixel', 'fkl_perop_batch', 'rpp_crop_batch', 'rpp_crop_batch_unsplit', 'hand_batch', 'hand_padded', 'hand_roi']
ONCE = ['fkl_fused_halfpixel_once', 'hand_roi_halfpixel_once']
NEW_VARIANTS = OLD_VARIANTS + ONCE
SHOWN = (10, 50, 100, 150, 200, 250, 300)
RUNS = ['noise/run0', 'noise/run1', 'noise/run2']
SERIES = [('rpp_crop_batch_unsplit', 'fkl_fused_halfpixel_once'), ('hand_roi_halfpixel_once', 'fkl_fused_halfpixel_once')]


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
    'new natural': [os.path.join(NEWDIR, 'natural-image')],
}
DATA = {name: load(dirs) for name, dirs in SOURCES.items()}
BATCHES = sorted({b for (b, _) in DATA['new']})


def stat(source, variant, batch, column='median_ms'):
    runs = DATA[source].get((batch, variant), [])
    return center([num(r[column]) for r in runs]) + (len(runs),)


def ratios(source, a, b, batch, column='median_ms'):
    return [num(x[column]) / num(y[column]) for x, y in zip(DATA[source].get((batch, a), []), DATA[source].get((batch, b), []))]


def fmt(v, digits=4):
    return 'nan' if math.isnan(v) else f'{v:.{digits}f}'


def pct(a, b):
    return math.nan if (math.isnan(a) or math.isnan(b) or b == 0) else 100.0 * (a - b) / b


def host_bound(variant, batch):
    build = stat('new', variant, batch, 'build_median_ms')[0]
    total = stat('new', variant, batch, 'median_ms')[0]
    return (not math.isnan(build)) and build >= 0.5 * total


summary = []

# ---- 1. The series of the figure
print('\n### The series of Figure 15 (per run, then the median of the three runs, min .. max)\n')
print('| batch | rpp_crop_batch_unsplit / fkl_fused_halfpixel_once | hand_roi_halfpixel_once / fkl_fused_halfpixel_once | all three runs on one side of 1 |')
print('|---|---|---|---|')
meds = {s: [] for s in SERIES}
beyond = []
for b in BATCHES:
    cells = []
    side = ''
    for a, c in SERIES:
        r = ratios('new', a, c, b)
        m = statistics.median(r)
        meds[(a, c)].append(m)
        cells.append(f'{m:.3f} ({min(r):.3f} .. {max(r):.3f})')
        summary.append({'comparison': 'series', 'what': f'{a} / {c}', 'batch': b, 'ratio_median': m, 'ratio_min': min(r), 'ratio_max': max(r),
                        'runs': f'{len(r)}'})
        if a == 'hand_roi_halfpixel_once':
            side = 'below' if max(r) < 1 else ('above' if min(r) > 1 else 'no')
            if side != 'no':
                beyond.append(b)
    print(f'| {b} | ' + ' | '.join(cells) + f' | {side} |')
print('\nRange of the median ratio:\n')
print('| series | 10 .. 150 | 10 .. 300 |')
print('|---|---|---|')
for (a, c), m in meds.items():
    print(f'| {a} / {c} | {min(m[:15]):.3f} .. {max(m[:15]):.3f} | {min(m):.3f} .. {max(m):.3f} |')
print(f'\nhand_roi_halfpixel_once / fkl_fused_halfpixel_once: all three runs on the same side of 1 at {len(beyond)} of {len(BATCHES)} batches: '
      f'{", ".join(str(b) for b in beyond)}')

# ---- 2. The times
print('\n### The times (medians over the three runs of the per-run medians, ms)\n')
print('| batch | fkl_fused_halfpixel_once | hand_roi_halfpixel_once | rpp_crop_batch_unsplit | fkl_fused_halfpixel (whole call) | whole call / built once |')
print('|---|---|---|---|---|---|')
for b in BATCHES:
    if b not in SHOWN:
        continue
    f1 = stat('new', 'fkl_fused_halfpixel_once', b)[0]
    h1 = stat('new', 'hand_roi_halfpixel_once', b)[0]
    u = stat('new', 'rpp_crop_batch_unsplit', b)[0]
    fw = stat('new', 'fkl_fused_halfpixel', b)[0]
    r = ratios('new', 'fkl_fused_halfpixel', 'fkl_fused_halfpixel_once', b)
    print(f'| {b} | {fmt(f1)} | {fmt(h1)} | {fmt(u)} | {fmt(fw)} | {statistics.median(r):.3f} ({min(r):.3f} .. {max(r):.3f}) |')
    summary.append({'comparison': 'whole call / built once', 'what': 'fkl_fused_halfpixel / fkl_fused_halfpixel_once', 'batch': b,
                    'ratio_median': statistics.median(r), 'ratio_min': min(r), 'ratio_max': max(r), 'runs': f'{len(r)}'})
bo = [stat('new', v, b, 'build_median_ms')[0] for v in ONCE for b in BATCHES]
print(f'\nThe build part of the built-once variants (build_median_ms): at most {max(bo) * 1000:.2f} us.')

# ---- 3. Agreement
print('\n### Agreement (maximum over the runs and batches of max_abs_diff; required rows passed)\n')
print('| comparison | max | required rows passed | bitwise equal (batch-runs) |')
print('|---|---|---|---|')
agg = {}
noted = set()   # the comparisons whose rows say whether the two tensors are the same bits
for name in RUNS + ['natural-image']:
    for r in rows(os.path.join(NEWDIR, name, AGREE)):
        m, p, n, bits = agg.get(r['comparison'], (0.0, 0, 0, 0))
        d = num(r['max_abs_diff'])
        agg[r['comparison']] = (math.nan if (math.isnan(d) or math.isnan(m)) else max(m, d), p + (r['pass'] == '1'), n + (r['pass'] != '-1'),
                                bits + ('bitwise equal' in r['note'] and r['region'] == 'interior'))
        if 'the whole tensors' in r['note']:
            noted.add(r['comparison'])
for comp, (m, p, n, bits) in agg.items():
    print(f'| {comp} | {m:.3g} | {f"{p} of {n}" if n else "informational"} | {f"{bits} of 120" if comp in noted else ""} |')

# ---- 4. RPP's ROI padding and pinned arrays
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

# ---- 5. The PCIe links and the clocks
print('\n### The PCIe links and the clocks (every reading of every row: at the start of the warm-up, of the timed runs, and after them)\n')
print('| source | host link GT/s x lanes | card-internal link GT/s x lanes | mclk at the start of the timed runs, MHz | sclk at the start of the timed runs, MHz | warm-up, s | rows with warmup_steady = 0 |')
print('|---|---|---|---|---|---|---|')
for source in ('new', 'new natural'):
    rs = [r for v in DATA[source].values() for r in v]
    host = Counter((num(r[f'host_link_gts_{m}']), num(r[f'host_link_width_{m}'])) for r in rs for m in ('start', 'timed_start', 'end'))
    card = Counter((num(r[f'card_link_gts_{m}']), num(r[f'card_link_width_{m}'])) for r in rs for m in ('start', 'timed_start', 'end'))
    mc = Counter(int(num(r['mclk_timed_start_mhz'])) for r in rs)
    ts = [num(r['sclk_timed_start_mhz']) for r in rs]
    ws = [num(r['warmup_s']) for r in rs]
    print(f'| {source} | {", ".join(f"{g:.0f} x{w:.0f}: {c}" for (g, w), c in sorted(host.items()))} | '
          f'{", ".join(f"{g:.0f} x{w:.0f}: {c}" for (g, w), c in sorted(card.items()))} | {dict(sorted(mc.items()))} | {min(ts):.0f} .. {max(ts):.0f} | '
          f'{min(ws):.2f} .. {max(ws):.2f} | {sum(1 for r in rs if r.get("warmup_steady") == "0")} of {len(rs)} |')

# ---- 6. The fast state
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
    print(f'| {v} | {100 * statistics.median(fr):+.1f} % | {100 * statistics.median(orr):+.1f} % |' if fr else f'| {v} | n/a | n/a |')
for name, sel in (('fast points', pts), ('other points', [(k, b) for k in range(3) for b in BATCHES if (k, b) not in pts])):
    if not sel:
        continue
    rs = [DATA['new'][(b, 'fkl_fused')][k] for (k, b) in sel]
    print(f'\nfkl_fused rows in the {name}: mclk at the start of the timed runs {dict(Counter(int(num(r["mclk_timed_start_mhz"])) for r in rs))}, '
          f'host link {dict(Counter((num(r["host_link_gts_timed_start"]), num(r["host_link_width_timed_start"])) for r in rs))}, '
          f'sclk median {statistics.median(num(r["sclk_timed_start_mhz"]) for r in rs):.0f} MHz')

# ---- 7. The eight variants of the whole-call record, old against new
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
print('\n### The eight variants of the whole-call record: old (rpp-preprocessing-hand-roi) against new (these runs)\n')
print('| variant | change over the 30 batches, min .. median .. max | batches beyond the spread | flagged |')
print('|---|---|---|---|')
for v in OLD_VARIANTS:
    sel = [x for x in lines if x[0] == v]
    chs = [x[6] for x in sel]
    print(f'| {v} | {min(chs):+.1f} .. {statistics.median(chs):+.1f} .. {max(chs):+.1f} % | {sum(x[7] for x in sel)} of {len(sel)} | '
          f'{sum(x[9] for x in sel)} of {len(sel)} |')
print('\nThe rows beyond the spread and flagged:\n')
print('| variant | batch | old ms (spread) | new ms (spread) | change | class |')
print('|---|---|---|---|---|---|')
for v, b, o, spo, n, spn, ch, exceeds, cls, flag in lines:
    if exceeds and flag:
        print(f'| {v} | {b} | {fmt(o)} ({fmt(spo)}) | {fmt(n)} ({fmt(spn)}) | {ch:+.1f} % | {cls} |')

# ---- 8. The natural image
print('\n### The natural-image run (one run; no spread)\n')
for a, c in SERIES:
    m = [stat('new natural', a, b)[0] / stat('new natural', c, b)[0] for b in BATCHES]
    print(f'- {a} / {c} over the 30 batches: {min(m):.3f} .. {statistics.median(m):.3f} .. {max(m):.3f} (min .. median .. max); '
          f'10 .. 150: {min(m[:15]):.3f} .. {max(m[:15]):.3f}')

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
