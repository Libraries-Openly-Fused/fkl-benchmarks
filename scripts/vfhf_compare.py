# vfhf_compare.py <results dir>: the tables of results/rx9060/vfhf-max-speedup/README.md from the three runs of benchmark_vfhf_max_speedup,
# written to stdout, and <results dir>/comparison.csv (a row per run and point). <results dir> is the results folder itself (for example
# results/rx9060/vfhf-max-speedup), not the repository root. It reads <results dir>/runs/run0, run1, run2 (run0 and run2 fused first, run1
# baseline first) and, when it exists, <results dir>/runs-extra/run0, run1, run2: the extra points of a binary of their own, run in the same
# order. Each extra run joins the run of the same name, and every table marks its points with "(extra)".
#   The three speedups of a point: speedup_block_paired_mean_of_ratios (the estimator of the 2025 harness's "Mean Speedup", the one comparable
#   with systems 1 to 5 of the GPU-size figure), speedup_ratio_of_means and speedup_ratio_of_medians, each as the median of the three runs
#   (min .. max), the spread (max - min) / median and the 3 % and 5 % flags.
#   The maximum of each estimator over NUM_OPS: of the per-point median of the runs, and of each run; the plateau: the points whose median is
#   within 5 % of that maximum.
#   The slow state of the fused kernel: a fused repetition is slow when it is more than SLOW (6 %) above the fastest fused repetition of the
#   three runs at its point; per block, the slow repetitions and the first timed repetition over the median of the block.
#   The order: run1 (baseline first) over the mean of run0 and run2 (fused first), per point.
#   The checks (the agreement CSVs), the clocks and links of the timing CSVs and the machine files, the warm-ups and the repetitions.
import csv
import os
import re
import statistics
import sys

SLOW = 0.06
RUNS = ('run0', 'run1', 'run2')
EST = (('speedup_block_paired_mean_of_ratios', 'block-paired mean of ratios'), ('speedup_ratio_of_means', 'ratio of means'),
       ('speedup_ratio_of_medians', 'ratio of medians'))


def load(path):
    return list(csv.DictReader([line for line in open(path) if not line.startswith('#') and line.strip()]))


def med_range(values, fmt):
    return f'{fmt.format(statistics.median(values))} ({fmt.format(min(values))} .. {fmt.format(max(values))})'


def main():
    top = sys.argv[1]
    sets = [s for s in ('runs', 'runs-extra') if os.path.isdir(os.path.join(top, s))]
    runs = {(s, r): os.path.join(top, s, r) for s in sets for r in RUNS}
    S, T, P, A = ({r: [] for r in RUNS} for _ in range(4))
    of_set = {}   # NUM_OPS -> the set of runs that measured it
    for (s, r), p in runs.items():
        part = load(os.path.join(p, 'benchmark_vfhf_max_speedup_speedup_GPU_AMD.csv'))
        for x in part:
            if of_set.setdefault(int(x['num_ops']), s) != s:
                sys.exit(f"NUM_OPS {x['num_ops']} is in more than one set of runs")
        S[r] += part
        T[r] += load(os.path.join(p, 'benchmark_vfhf_max_speedup_GPU_AMD.csv'))
        P[r] += load(os.path.join(p, 'benchmark_vfhf_max_speedup_reps_GPU_AMD.csv'))
        A[r] += load(os.path.join(p, 'benchmark_vfhf_max_speedup_agreement_GPU_AMD.csv'))
    points = sorted(int(x['num_ops']) for x in S['run0'])
    if any(sorted(int(x['num_ops']) for x in S[r]) != points for r in RUNS):
        sys.exit('the runs do not have the same points')
    label = {n: f'{n} (extra)' if of_set[n] == 'runs-extra' else str(n) for n in points}
    sp = {(r, int(x['num_ops'])): {k: float(x[k]) for k, _ in EST} for r in RUNS for x in S[r]}
    tm = {(r, int(x['num_ops']), x['side']): x for r in RUNS for x in T[r]}
    reps = {}
    for r in RUNS:
        for x in P[r]:
            reps.setdefault((r, int(x['num_ops']), x['side']), []).append(float(x['ms']))

    # ---- The speedups per point
    print('### The speedups per point (the median of the three runs, min .. max; spread = (max - min) / median)\n')
    print('| NUM_OPS | block-paired mean of ratios | ratio of means | ratio of medians | spread of the block-paired | flag |')
    print('|---|---|---|---|---|---|')
    rows = []
    for n in points:
        cells, spreads = [], {}
        for k, _ in EST:
            v = [sp[(r, n)][k] for r in RUNS]
            cells.append(med_range(v, '{:.0f}'))
            spreads[k] = (max(v) - min(v)) / statistics.median(v)
        s = spreads[EST[0][0]]
        flag = '> 5 %' if s > 0.05 else ('> 3 %' if s > 0.03 else '')
        print(f'| {label[n]} | ' + ' | '.join(cells) + f' | {100 * s:.1f} % | {flag} |')
        rows.append((n, spreads))

    # ---- The maximum over NUM_OPS
    print('\n### The maximum over NUM_OPS\n')
    print('| estimator | maximum of the per-point median of the runs (NUM_OPS) | run0 (NUM_OPS) | run1 (NUM_OPS) | run2 (NUM_OPS) | the plateau: points within 5 % of the maximum |')
    print('|---|---|---|---|---|---|')
    for k, name in EST:
        medians = {n: statistics.median(sp[(r, n)][k] for r in RUNS) for n in points}
        nmax = max(points, key=lambda n: medians[n])
        per_run = []
        for r in RUNS:
            nr = max(points, key=lambda n: sp[(r, n)][k])
            per_run.append(f'{sp[(r, nr)][k]:.0f} ({label[nr]})')
        plateau = [n for n in points if medians[n] >= 0.95 * medians[nmax]]
        print(f'| {name} | {medians[nmax]:.0f} ({label[nmax]}) | ' + ' | '.join(per_run) + f' | {", ".join(label[n] for n in plateau)} |')

    # ---- The times
    print('\n### The times (the mean of the timed repetitions of a side; the median of the three runs, min .. max)\n')
    print('| NUM_OPS | fused, ms | baseline, ms | baseline per call, us | baseline repetitions (rule) |')
    print('|---|---|---|---|---|')
    for n in points:
        f = [float(tm[(r, n, 'fused')]['mean_ms']) for r in RUNS]
        b = [float(tm[(r, n, 'baseline')]['mean_ms']) for r in RUNS]
        calls = int(tm[('run0', n, 'baseline')]['calls_per_rep'])
        rp = sorted({f"{tm[(r, n, 'baseline')]['reps']} ({tm[(r, n, 'baseline')]['reps_rule']})" for r in RUNS})
        print(f'| {label[n]} | {med_range(f, "{:.4f}")} | {med_range(b, "{:.2f}")} | {1000 * statistics.median(b) / calls:.3f} | {", ".join(rp)} |')

    # ---- The slow state of the fused kernel
    print(f'\n### The fused blocks: slow repetitions (more than {100 * SLOW:.0f} % above the fastest fused repetition of the three runs at the point), '
          'and the first timed repetition over the median of its block\n')
    print('| NUM_OPS | run0 | run1 | run2 |')
    print('|---|---|---|---|')
    slow_blocks, first_slow = [], []
    for n in points:
        fastest = min(min(reps[(r, n, 'fused')]) for r in RUNS)
        cells = []
        for r in RUNS:
            f = reps[(r, n, 'fused')]
            slow = sum(1 for x in f if x > (1 + SLOW) * fastest)
            first = f[0] / statistics.median(f)
            cells.append(f'{slow} slow, first {first:.3f}')
            if slow >= len(f) / 2:
                slow_blocks.append((r, n, slow))
            elif f[0] > (1 + SLOW) * fastest:
                first_slow.append((r, n))
        print(f'| {label[n]} | ' + ' | '.join(cells) + ' |')
    print(f'\nBlocks with at least half of their repetitions slow: {len(slow_blocks)} of {len(points) * len(RUNS)}: '
          + ', '.join(f'{r} at {label[n]} ({s} of 100)' for r, n, s in slow_blocks))
    print(f'\nOther blocks whose first timed repetition is slow: {len(first_slow)}: ' + ', '.join(f'{r} at {label[n]}' for r, n in first_slow))

    # ---- The order
    print('\n### The order: run1 (baseline first) over the mean of run0 and run2 (fused first)\n')
    print('| NUM_OPS | block-paired mean of ratios | fused mean | baseline mean |')
    print('|---|---|---|---|')
    order = []
    for n in points:
        q = [sp[('run1', n)][EST[0][0]] / statistics.mean([sp[('run0', n)][EST[0][0]], sp[('run2', n)][EST[0][0]]]),
             float(tm[('run1', n, 'fused')]['mean_ms']) / statistics.mean([float(tm[(r, n, 'fused')]['mean_ms']) for r in ('run0', 'run2')]),
             float(tm[('run1', n, 'baseline')]['mean_ms']) / statistics.mean([float(tm[(r, n, 'baseline')]['mean_ms']) for r in ('run0', 'run2')])]
        order.append(q)
        print(f'| {label[n]} | ' + ' | '.join(f'{100 * (x - 1):+.1f} %' for x in q) + ' |')
    for i, name in enumerate(('block-paired mean of ratios', 'fused mean', 'baseline mean')):
        v = [q[i] for q in order]
        print(f'\n{name}: median {100 * (statistics.median(v) - 1):+.1f} %, range {100 * (min(v) - 1):+.1f} .. {100 * (max(v) - 1):+.1f} %', end='')
    print()

    # ---- The checks
    print('\n### The checks (tolerance 1e-4 absolute; the maximum of max_abs_diff over the runs and points)\n')
    print('| comparison | max | required rows passed |')
    print('|---|---|---|')
    for comp in sorted({x['comparison'] for r in RUNS for x in A[r]}):
        xs = [x for r in RUNS for x in A[r] if x['comparison'] == comp]
        req = [x for x in xs if x['pass'] != '-1']
        passed = f"{sum(1 for x in req if x['pass'] == '1')} of {len(req)}" if req else 'informational'
        print(f"| {comp} | {max(float(x['max_abs_diff']) for x in xs):.3g} | {passed} |")

    # ---- Clocks, links, warm-ups
    print('\n### The clocks, the links and the warm-ups (every timing row of the runs)\n')
    allrows = [x for r in RUNS for x in T[r]]
    for side in ('fused', 'baseline'):
        xs = [x for x in allrows if x['side'] == side]
        sclk = [float(x['sclk_timed_start_mhz']) for x in xs] + [float(x['sclk_end_mhz']) for x in xs]
        mclk = {}
        for x in xs:
            mclk[x['mclk_timed_start_mhz']] = mclk.get(x['mclk_timed_start_mhz'], 0) + 1
        print(f'- {side}: sclk at the start of the timed repetitions and after them {min(sclk):.0f} .. {max(sclk):.0f} MHz; mclk at the start of the '
              f'timed repetitions {dict(sorted(mclk.items(), key=lambda t: float(t[0])))} (rows); warm-up {min(float(x["warmup_s"]) for x in xs):.2f} .. '
              f'{max(float(x["warmup_s"]) for x in xs):.2f} s; rows with warmup_steady = 0: {sum(1 for x in xs if x["warmup_steady"] == "0")} of {len(xs)}')
    links = {}
    for x in allrows:
        for m in ('start', 'timed_start', 'end'):
            key = (f"host {float(x[f'host_link_gts_{m}']):g} x{float(x[f'host_link_width_{m}']):g}",
                   f"card {float(x[f'card_link_gts_{m}']):g} x{float(x[f'card_link_width_{m}']):g}")
            links[key] = links.get(key, 0) + 1
    print(f'- the links at all {3 * len(allrows)} readings: ' + '; '.join(f'{h}, {c}: {k}' for (h, c), k in links.items()))
    for (st, r), path in runs.items():
        temps = []
        for when in ('before', 'after'):
            m = re.search(r'edge (\d+) C', open(os.path.join(path, f'machine-{when}.txt')).read())
            temps.append(m.group(1) if m else '?')
        print(f'- {st}/{r}: GPU edge temperature {temps[0]} C before the run, {temps[1]} C after it')

    # ---- comparison.csv
    with open(os.path.join(top, 'comparison.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['set', 'run', 'baseline_first', 'num_ops'] + [k for k, _ in EST] +
                   ['fused_mean_ms', 'fused_median_ms', 'baseline_mean_ms', 'baseline_median_ms', 'baseline_reps', 'fused_slow_reps',
                    'fused_first_over_median', 'fused_sclk_timed_start_mhz', 'fused_mclk_timed_start_mhz', 'baseline_sclk_timed_start_mhz',
                    'baseline_mclk_timed_start_mhz'])
        for r in RUNS:
            for n in points:
                f, b = tm[(r, n, 'fused')], tm[(r, n, 'baseline')]
                fr = reps[(r, n, 'fused')]
                fastest = min(min(reps[(q, n, 'fused')]) for q in RUNS)
                w.writerow([of_set[n], r, 1 if r == 'run1' else 0, n] + [f'{sp[(r, n)][k]:.6g}' for k, _ in EST] +
                           [f['mean_ms'], f['median_ms'], b['mean_ms'], b['median_ms'], b['reps'], sum(1 for x in fr if x > (1 + SLOW) * fastest),
                            f'{fr[0] / statistics.median(fr):.4f}', f['sclk_timed_start_mhz'], f['mclk_timed_start_mhz'], b['sclk_timed_start_mhz'],
                            b['mclk_timed_start_mhz']])


if __name__ == '__main__':
    main()
