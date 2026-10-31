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
"""Shared helpers of the figure scripts (scripts/plot_*.py): the paper's style, reading of the benchmark CSVs, and saving a figure
as a vector PDF + a PNG preview + a sidecar .txt that records which CSV it was drawn from.

Style (revision-2 figure specification): white background, no title inside the plot, line width 1 pt, text of at least 7 pt at the
printed width (so every script draws at the printed width of its figure), light dotted grid, distinct markers per series, and the
Okabe-Ito colours in a fixed order. The matplotlib "tab10" blue/orange/green/red that the specification suggests fails the colour-blind
check (orange and green are indistinguishable under protanopia, Delta E 0.7); the three colours below pass it (adjacent Delta E >= 11
under deuteranopia/protanopia, >= 3:1 contrast on white; checked with the dataviz skill's validate_palette.js, 2- and 3-slot subsets).
"""
import csv
import datetime
import io
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SERIES_COLORS = ["#0072B2", "#D55E00", "#009E73"]   # blue, vermillion, bluish green: always in this order, never cycled
SERIES_MARKERS = ["o", "s", "^"]
INK = "#222222"
GRID = "#bbbbbb"

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES = os.path.join(REPO, "figures")


def apply_style(text_pt=7.5):
    """text_pt: size of the tick labels, legends and annotations (>= 7 at the printed width); axis labels are 0.5 pt larger."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
        "font.size": text_pt, "axes.labelsize": text_pt + 0.5, "xtick.labelsize": text_pt, "ytick.labelsize": text_pt,
        "legend.fontsize": text_pt, "legend.frameon": False, "legend.handlelength": 1.8, "legend.borderaxespad": 0.3,
        "axes.linewidth": 0.6, "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK, "text.color": INK,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.0, "lines.markersize": 3.0, "lines.solid_joinstyle": "round",
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4, "grid.linestyle": (0, (1, 2)), "axes.axisbelow": True,
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "pdf.fonttype": 42, "ps.fonttype": 42, "mathtext.default": "regular",
    })


def read_blocks(path):
    """The CSVs of fkl-benchmarks: '# key: value' lines first, then one or more tables, each with its own header line.
    Returns (meta, blocks): meta is the dict of the '# ' lines, blocks a list of (header, rows) with numeric cells as floats."""
    meta, blocks, header, rows = {}, [], None, []
    with open(path, newline="") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if line.startswith("#"):
                if ":" in line:
                    k, v = line[1:].split(":", 1)
                    meta[k.strip()] = v.strip()
                continue
            cells = [c.strip() for c in next(csv.reader(io.StringIO(line)))]
            try:
                row = [float(c) for c in cells]
            except ValueError:
                if header is not None and rows:
                    blocks.append((header, rows))
                header, rows = cells, []
                continue
            rows.append(row)
    if header is not None and rows:
        blocks.append((header, rows))
    return meta, blocks


def column(block, name):
    header, rows = block
    i = header.index(name)
    return [r[i] for r in rows]


def resolve(path, name):
    """--csv may be the file itself or the experiment directory that holds it."""
    return os.path.join(path, name) if os.path.isdir(path) else path


def git_line(path, what):
    d = os.path.dirname(os.path.abspath(path))
    try:
        return subprocess.run(["git", "-C", d] + what, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def provenance(paths):
    """One text line per source CSV: path, the commit of the results repository that last changed it (and whether the working copy
    differs), and the '# repo' / '# fkl' / '# nvcc' / '# gpu' lines of the CSV header (the build that produced the numbers)."""
    lines = []
    for p in paths:
        ap = os.path.abspath(p)
        meta, _ = read_blocks(ap)
        commit = git_line(ap, ["log", "-1", "--format=%H", "--", ap]) or "not committed"
        dirty = git_line(ap, ["status", "--porcelain", "--", ap])
        rel = os.path.relpath(ap, REPO) if ap.startswith(REPO) else ap
        lines.append(f"csv: {rel}")
        lines.append(f"  results commit that last changed it: {commit}{' (working copy differs from it)' if dirty else ''}")
        for k in ("repo", "fkl", "nvcc", "gpu"):
            if k in meta:
                lines.append(f"  produced by {k}: {meta[k]}")
    return lines


def save(fig, out, sources, script, notes=()):
    """<out>.pdf (the vector figure, provenance in its Subject/Keywords), <out>.png (preview, 220 dpi), <out minus .pdf>.txt (sidecar)."""
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    prov = provenance(sources)
    stem = os.path.splitext(out)[0]
    fig.savefig(out, metadata={"Title": os.path.basename(stem), "Author": "fkl-benchmarks", "Creator": f"scripts/{script} (matplotlib {matplotlib.__version__})",
                               "Subject": "; ".join(l.strip() for l in prov), "Keywords": "fkl-benchmarks, revision 2", "CreationDate": None})
    fig.savefig(stem + ".png", dpi=220)
    with open(stem + ".txt", "w") as f:
        f.write(f"{os.path.basename(out)}: drawn by scripts/{script}\n")
        f.write(f"size: {fig.get_figwidth():.2f} x {fig.get_figheight():.2f} in (the printed width)\n")
        f.write("\n".join(prov) + "\n")
        for n in notes:
            f.write(n + "\n")
    print(f"{out} (+ .png, .txt)")
    for n in notes:
        print("  " + n)
    plt.close(fig)


def arg_parser(description, default_out, extra=None):
    import argparse
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--csv", required=True, help="the CSV, or the experiment directory that holds it")
    ap.add_argument("--out", default=os.path.join(FIGURES, default_out))
    return ap
