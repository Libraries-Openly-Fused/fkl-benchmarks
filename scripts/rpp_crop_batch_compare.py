# rpp_crop_batch_compare.py: the comparison of results/rx9060/rpp-preprocessing-crop-batch/ (RPP's documented batched form, 2026-11-08) with the
# reference runs of the program before the rewrite in the same window and with the published runs behind Figure 15 (results/rx9060/rpp-preprocessing/).
#
# usage: python3 scripts/rpp_crop_batch_compare.py [repository root]   (writes comparison.csv into the folder and prints the tables of its README)
#
# For every source (published: rpp-preprocessing/noise/run0..2; reference and new: rpp-preprocessing-crop-batch/noise/{reference,new}-run0..2), batch
# and variant: the median_ms of each of the three runs, their median ("value") and their spread (max - min). A change between two sources exceeds the
# spread when |a - b| > max(spread_a, spread_b). The flag threshold is 3 % for a GPU-bound variant and 5 % for a host-bound one (host-bound: the host
# part, build_median_ms, is at least half of the run). The ratio of the paper is rpp / fkl_fused per run (rpp_crop_batch_unsplit for the new program,
# rpp_fused_api for the published and reference ones); its value is the median of the three runs and its spread their min..max.
import csv
import math
import os
import statistics
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
NEWDIR = os.path.join(ROOT, 'results', 'rx9060', 'rpp-preprocessing-crop-batch')
PUBDIR = os.path.join(ROOT, 'results', 'rx9060', 'rpp-preprocessing')
TIMING = 'benchmark_rpp_preprocessing_a_GPU_AMD.csv'
AGREE = 'benchmark_rpp_preprocessing_a_agreement_GPU_AMD.csv'


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
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v


def center(values):
    vals = [v for v in values if not math.isnan(v)]
    return (statistics.median(vals), max(vals) - min(vals)) if vals else (math.nan, math.nan)


SOURCES = {
    'published': [os.path.join(PUBDIR, 'noise', f'run{k}') for k in range(3)],
    'reference': [os.path.join(NEWDIR, 'noise', f'reference-run{k}') for k in range(3)],
    'new': [os.path.join(NEWDIR, 'noise', f'new-run{k}') for k in range(3)],
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


# (label, new source, new variant, new column, old source, old variant) for the comparisons
PAIRS_NEW_REF = [
    ('fkl_fused', 'new', 'fkl_fused', 'median_ms', 'reference', 'fkl_fused'),
    ('fkl_fused_halfpixel', 'new', 'fkl_fused_halfpixel', 'median_ms', 'reference', 'fkl_fused_halfpixel'),
    ('fkl_perop_batch', 'new', 'fkl_perop_batch', 'median_ms', 'reference', 'fkl_perop_batch'),
    ('rpp_crop_batch_unsplit vs rpp_fused_api', 'new', 'rpp_crop_batch_unsplit', 'median_ms', 'reference', 'rpp_fused_api'),
    ('rpp_crop_batch vs rpp_fused_api', 'new', 'rpp_crop_batch', 'median_ms', 'reference', 'rpp_fused_api'),
    ('rpp_crop_batch compute vs rpp_fused_api', 'new', 'rpp_crop_batch', 'compute_median_ms', 'reference', 'rpp_fused_api'),
]
PAIRS_REF_PUB = [(v, 'reference', v, 'median_ms', 'published', v) for v in ('fkl_fused', 'fkl_fused_halfpixel', 'fkl_perop_batch', 'rpp_fused_api')]


def host_bound(variant, batch):
    build = stat('new', variant, batch, 'build_median_ms')[0]
    total = stat('new', variant, batch, 'median_ms')[0]
    return (not math.isnan(build)) and build >= 0.5 * total


def compare(pairs, out):
    lines = []
    for label, sa, va, ca, sb, vb in pairs:
        for b in BATCHES:
            a, spa, na = stat(sa, va, b, ca)
            o, spo, no = stat(sb, vb, b)
            if na == 0 or no == 0:
                continue
            ch = pct(a, o)
            exceeds = abs(a - o) > max(spa, spo)
            hb = sa == 'new' and ca == 'median_ms' and host_bound(va, b)
            flag = abs(ch) > (5.0 if hb else 3.0)
            out.append({'comparison': f'{sa} vs {sb}', 'what': label, 'batch': b, f'{sa}_ms': a, f'{sa}_spread_ms': spa, f'{sb}_ms': o,
                        f'{sb}_spread_ms': spo, 'change_pct': ch, 'exceeds_spread': int(exceeds), 'class': 'host' if hb else 'GPU', 'flag': int(flag)})
            lines.append((label, b, a, spa, o, spo, ch, exceeds, 'host' if hb else 'GPU', flag))
    return lines


def table(title, lines, a_name, b_name, every=(10, 50, 100, 150, 200, 250, 300)):
    print(f'\n### {title}\n')
    print(f'| what | batch | {a_name} ms (spread) | {b_name} ms (spread) | change | exceeds spread | class | flag |')
    print('|---|---|---|---|---|---|---|---|')
    for label, b, a, spa, o, spo, ch, exceeds, cls, flag in lines:
        if b in every:
            print(f'| {label} | {b} | {fmt(a)} ({fmt(spa)}) | {fmt(o)} ({fmt(spo)}) | {ch:+.1f} % | {"yes" if exceeds else "no"} | {cls} | {"**yes**" if flag else "no"} |')
    # the summary over all 30 batches
    by = {}
    for label, b, a, spa, o, spo, ch, exceeds, cls, flag in lines:
        by.setdefault(label, []).append((ch, exceeds, flag))
    print(f'\nOver the {len(BATCHES)} batches:\n')
    print('| what | change, min .. median .. max | batches beyond the spread | flagged |')
    print('|---|---|---|---|')
    for label, v in by.items():
        chs = [c for c, _, _ in v]
        print(f'| {label} | {min(chs):+.1f} .. {statistics.median(chs):+.1f} .. {max(chs):+.1f} % | {sum(e for _, e, _ in v)} of {len(v)} | {sum(f for _, _, f in v)} of {len(v)} |')


summary = []
new_ref = compare(PAIRS_NEW_REF, summary)
ref_pub = compare(PAIRS_REF_PUB, summary)
table('New against reference (the effect of the rewrite)', new_ref, 'new', 'reference')
table('Reference against published (everything else since the published runs)', ref_pub, 'reference', 'published')

# FKL's build part next to its total
print('\n### The parts of the new variants (medians over the three runs of the per-run medians)\n')
print('| batch | fkl_fused total | fkl_fused build | build share | fkl_perop_batch build | rpp_crop_batch total | rpp_crop_batch build | rpp_crop_batch compute | rpp_crop_batch_unsplit |')
print('|---|---|---|---|---|---|---|---|---|')
for b in BATCHES:
    if b not in (10, 50, 100, 150, 200, 250, 300):
        continue
    ft = stat('new', 'fkl_fused', b)[0]
    fb = stat('new', 'fkl_fused', b, 'build_median_ms')[0]
    pb = stat('new', 'fkl_perop_batch', b, 'build_median_ms')[0]
    rt = stat('new', 'rpp_crop_batch', b)[0]
    rb = stat('new', 'rpp_crop_batch', b, 'build_median_ms')[0]
    rc = stat('new', 'rpp_crop_batch', b, 'compute_median_ms')[0]
    ru = stat('new', 'rpp_crop_batch_unsplit', b)[0]
    print(f'| {b} | {fmt(ft)} | {fmt(fb)} | {100 * fb / ft:.1f} % | {fmt(pb)} | {fmt(rt)} | {fmt(rb)} | {fmt(rc)} | {fmt(ru)} |')

# The ratio of the paper
print('\n### The ratio RPP / fkl_fused per batch (median of the three runs, min .. max)\n')
print('| batch | new: rpp_crop_batch_unsplit / fkl_fused | reference: rpp_fused_api / fkl_fused | published: rpp_fused_api / fkl_fused |')
print('|---|---|---|---|')
ratios = {}
for b in BATCHES:
    row = []
    for source, rpp in (('new', 'rpp_crop_batch_unsplit'), ('reference', 'rpp_fused_api'), ('published', 'rpp_fused_api')):
        r = [num(x['median_ms']) / num(y['median_ms']) for x, y in zip(DATA[source].get((b, rpp), []), DATA[source].get((b, 'fkl_fused'), []))]
        ratios.setdefault(source, []).append(statistics.median(r) if r else math.nan)
        row.append(f'{statistics.median(r):.2f} ({min(r):.2f} .. {max(r):.2f})' if r else 'n/a')
        summary.append({'comparison': 'ratio', 'what': f'{source}: {rpp} / fkl_fused', 'batch': b, 'ratio_median': statistics.median(r) if r else math.nan,
                        'ratio_min': min(r) if r else math.nan, 'ratio_max': max(r) if r else math.nan})
    if b in (10, 50, 100, 150, 200, 250, 300):
        print(f'| {b} | ' + ' | '.join(row) + ' |')
print('\nRange over the 30 batches (of the median ratio): ' + '; '.join(
    f'{s} {min(v):.2f} .. {max(v):.2f}' for s, v in ratios.items()))

# Agreement
print('\n### Agreement (maximum over the 30 batches of max_abs_diff; pass: required rows)\n')
print('| run | comparison | region | max | required rows passed |')
print('|---|---|---|---|---|')
for name in ['noise/new-run0', 'noise/new-run1', 'noise/new-run2', 'natural-image', 'noise/reference-run0', 'noise/reference-run1', 'noise/reference-run2']:
    agg = {}
    for r in rows(os.path.join(NEWDIR, name, AGREE)):
        k = (r['comparison'], r['region'])
        m, p, n = agg.get(k, (0.0, 0, 0))
        d = num(r['max_abs_diff'])
        agg[k] = (max(m, d), p + (r['pass'] == '1'), n + (r['pass'] != '-1'))
    for (comp, region), (m, p, n) in agg.items():
        print(f'| {name} | {comp} | {region} | {m:.3g} | {f"{p} of {n}" if n else "informational"} |')

# Clocks
print('\n### Clock and warm-up columns (min .. max over every row of a source)\n')
print('| source | sclk at the start of the timed runs, MHz | sclk after them, MHz | warm-up, s | rows with warmup_steady = 0 |')
print('|---|---|---|---|---|')
for source in ('published', 'reference', 'new'):
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
