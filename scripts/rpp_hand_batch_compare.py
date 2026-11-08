# rpp_hand_batch_compare.py: the comparison of results/rx9060/rpp-preprocessing-hand-batch/ (benchmark_rpp_preprocessing_hip with the hand-written
# kernel hand_batch, fkl-benchmarks 6dc0541, 2026-11-08) with the new runs of results/rx9060/rpp-preprocessing-crop-batch/ (9461389, the same day).
#
# usage: python3 scripts/rpp_hand_batch_compare.py [repository root]   (writes comparison.csv into the folder and prints the tables of its README)
#
# For every source (old: rpp-preprocessing-crop-batch/noise/new-run0..2; new: rpp-preprocessing-hand-batch/noise/run0..2), batch and variant: the
# median_ms of each of the three runs, their median ("value") and their spread (max - min). A change exceeds the spread when |new - old| >
# max(spread_new, spread_old). The flag threshold is 3 % for a GPU-bound variant and 5 % for a host-bound one (host-bound: the host part,
# build_median_ms, is at least half of the run), as in scripts/rpp_crop_batch_compare.py. The ratios to fkl_fused are taken per run (hand_batch,
# rpp_crop_batch_unsplit); their value is the median of the three runs and their spread the min .. max.
# The program has six variants now: the rotation of the variants of a point (the first is variant (index of the point + E5_ROTATE_OFFSET) mod the
# number of variants) puts the five variants of the crop-batch runs at other positions than in those runs.
import csv
import math
import os
import statistics
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
NEWDIR = os.path.join(ROOT, 'results', 'rx9060', 'rpp-preprocessing-hand-batch')
OLDDIR = os.path.join(ROOT, 'results', 'rx9060', 'rpp-preprocessing-crop-batch')
TIMING = 'benchmark_rpp_preprocessing_a_GPU_AMD.csv'
AGREE = 'benchmark_rpp_preprocessing_a_agreement_GPU_AMD.csv'
OLD_VARIANTS = ['fkl_fused', 'fkl_fused_halfpixel', 'fkl_perop_batch', 'rpp_crop_batch', 'rpp_crop_batch_unsplit']
NEW_VARIANTS = OLD_VARIANTS + ['hand_batch']
SHOWN = (10, 50, 100, 150, 200, 250, 300)


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
    'old': [os.path.join(OLDDIR, 'noise', f'new-run{k}') for k in range(3)],
    'new': [os.path.join(NEWDIR, 'noise', f'run{k}') for k in range(3)],
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

# ---- 1. The five variants of the crop-batch runs, old (those runs) against new (these runs)
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
print('\n### The five variants of the crop-batch runs: old (rpp-preprocessing-crop-batch, new-run0..2) against new (these runs)\n')
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

# ---- 2. The parts of fkl_fused and hand_batch
print('\n### fkl_fused and hand_batch: the run and its parts (medians over the three runs of the per-run medians, ms)\n')
print('| batch | fkl_fused total | fkl_fused build | fkl_fused compute | hand_batch total | hand_batch build | hand_batch compute |')
print('|---|---|---|---|---|---|---|')
for b in BATCHES:
    vals = [stat('new', v, b, c)[0] for v in ('fkl_fused', 'hand_batch') for c in ('median_ms', 'build_median_ms', 'compute_median_ms')]
    for v in ('fkl_fused', 'hand_batch'):
        for c in ('median_ms', 'build_median_ms', 'compute_median_ms'):
            m, sp, n = stat('new', v, b, c)
            summary.append({'comparison': 'parts', 'what': f'{v} {c}', 'batch': b, 'new_ms': m, 'new_spread_ms': sp, 'runs': f'{n}'})
    if b in SHOWN:
        print(f'| {b} | ' + ' | '.join(fmt(x) for x in vals) + ' |')

# ---- 3. The ratios to fkl_fused, per run
print('\n### The ratios to fkl_fused per batch (median of the three runs, min .. max)\n')
print('| batch | hand_batch / fkl_fused | hand_batch / fkl_fused, compute parts | rpp_crop_batch_unsplit / fkl_fused | the same in the old runs |')
print('|---|---|---|---|---|')
RATIOS = [('new', 'hand_batch', 'median_ms'), ('new', 'hand_batch', 'compute_median_ms'), ('new', 'rpp_crop_batch_unsplit', 'median_ms'),
          ('old', 'rpp_crop_batch_unsplit', 'median_ms')]
ranges = {}
for b in BATCHES:
    cells = []
    for source, v, column in RATIOS:
        r = [num(x[column]) / num(y[column]) for x, y in zip(DATA[source].get((b, v), []), DATA[source].get((b, 'fkl_fused'), []))]
        med = statistics.median(r) if r else math.nan
        ranges.setdefault((source, v, column), []).append(med)
        cells.append(f'{med:.3f} ({min(r):.3f} .. {max(r):.3f})' if r else 'n/a')
        summary.append({'comparison': 'ratio', 'what': f'{source}: {v} / fkl_fused ({column})', 'batch': b, 'ratio_median': med,
                        'ratio_min': min(r) if r else math.nan, 'ratio_max': max(r) if r else math.nan, 'runs': f'{len(r)}'})
    print(f'| {b} | ' + ' | '.join(cells) + ' |')
print('\nRange over the batches of the median ratio: ' + '; '.join(
    f'{s} {v} ({c}) {min(m):.3f} .. {max(m):.3f}' for (s, v, c), m in ranges.items()))
print('Range over the batches 10 .. 150 of the median ratio: ' + '; '.join(
    f'{s} {v} ({c}) {min(m[:15]):.3f} .. {max(m[:15]):.3f}' for (s, v, c), m in ranges.items()))

# ---- 4. The natural image, one run each
print('\n### The natural-image runs (one run each; no spread): new against old\n')
print('| variant | batch | old ms | new ms | change |')
print('|---|---|---|---|---|')
for v in OLD_VARIANTS:
    for b in SHOWN:
        o = stat('old natural', v, b)[0]
        n = stat('new natural', v, b)[0]
        print(f'| {v} | {b} | {fmt(o)} | {fmt(n)} | {pct(n, o):+.1f} % |')
        summary.append({'comparison': 'natural new vs old', 'what': v, 'batch': b, 'old_ms': o, 'new_ms': n, 'change_pct': pct(n, o), 'runs': '1+1'})
nat = [stat('new natural', 'hand_batch', b)[0] / stat('new natural', 'fkl_fused', b)[0] for b in BATCHES]
print(f'\nNatural image, hand_batch / fkl_fused over the {len(BATCHES)} batches: {min(nat):.3f} .. {statistics.median(nat):.3f} .. {max(nat):.3f} '
      '(min .. median .. max)')

# ---- 5. Agreement
print('\n### Agreement (maximum over the 30 batches of max_abs_diff; pass: required rows)\n')
print('| run | comparison | region | max | required rows passed | note |')
print('|---|---|---|---|---|---|')
for name in ['noise/run0', 'noise/run1', 'noise/run2', 'natural-image']:
    agg = {}
    for r in rows(os.path.join(NEWDIR, name, AGREE)):
        k = (r['comparison'], r['region'])
        m, p, n, notes = agg.get(k, (0.0, 0, 0, {}))
        d = num(r['max_abs_diff'])
        notes[r['note']] = notes.get(r['note'], 0) + 1
        agg[k] = (max(m, d) if not (math.isnan(d) or math.isnan(m)) else math.nan, p + (r['pass'] == '1'), n + (r['pass'] != '-1'), notes)
    for (comp, region), (m, p, n, notes) in agg.items():
        note = '; '.join(f'{t} ({c} batches)' for t, c in notes.items()) if comp.startswith('hand_batch vs fkl_fused') else ''
        print(f'| {name} | {comp} | {region} | {m:.3g} | {f"{p} of {n}" if n else "informational"} | {note} |')

# ---- 6. Clocks and warm-up
print('\n### Clock and warm-up columns (min .. max over every row of a source)\n')
print('| source | sclk at the start of the timed runs, MHz | sclk after them, MHz | warm-up, s | rows with warmup_steady = 0 |')
print('|---|---|---|---|---|')
for source in ('old', 'new', 'old natural', 'new natural'):
    rs = [r for v in DATA[source].values() for r in v]
    ts = [num(r.get('sclk_timed_start_mhz')) for r in rs]
    te = [num(r.get('sclk_end_mhz')) for r in rs]
    ws = [num(r.get('warmup_s')) for r in rs]
    print(f'| {source} | {min(ts):.0f} .. {max(ts):.0f} | {min(te):.0f} .. {max(te):.0f} | {min(ws):.2f} .. {max(ws):.2f} | '
          f'{sum(1 for r in rs if r.get("warmup_steady") == "0")} of {len(rs)} |')

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
