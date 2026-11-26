#!/usr/bin/env python3
"""C-PM-CUDA: where the code of the programs behind the paper's RTX 4090 numbers differs between the commit that measured them and main 976e42b
(CPU only: git and file reads; nothing is built or run).

  source_history.py <fkl-benchmarks checkout> [--out report.txt]

For each program: its source file and every repository header it includes (the #include closure inside the repository: common/, handwritten-cuda/
and the program's own folder; the submodules are not followed), at main 976e42b. For each measured commit of the program: the FKL, cvGPUSpeedup
and FastNPP gitlinks at that commit and at main, and `git diff --stat` and the commits (git log --oneline) between that commit and main for those
files. The commits are listed so that every change can be read; what a change does to a row is said in the README, from the diff.
"""
import argparse
import os
import re
import subprocess

ap = argparse.ArgumentParser()
ap.add_argument("repo"); ap.add_argument("--out")
a = ap.parse_args()
MAIN = "976e42bfad0ba61c3bb22a555cf72f4da6896aeb"
PROGRAMS = [   # (record, program, source, measured commit)
    ("cleanup-window-b (VI-K, E4)", "e4_preprocessing_cu", "opencv-cuda/batched-preprocessing/e4_preprocessing.cu", "836a98e"),
    ("e4-hand-roi (Figure 15, whole call)", "e4_preprocessing_cu", "opencv-cuda/batched-preprocessing/e4_preprocessing.cu", "469aa6d"),
    ("e4-build-once (Figure 15, build once)", "e4_preprocessing_cu", "opencv-cuda/batched-preprocessing/e4_preprocessing.cu", "f534a38"),
    ("cleanup-window-a (NPP)", "benchmark_npp_batchresize_x_split3D_cu", "npp/batchresize/benchmark_npp_batchresize_x_split3D.cu", "836a98e"),
    ("cleanup-window-a (CPU time)", "benchmark_CPU_NPP_vs_FastNPP_cu", "npp/CPU/benchmark_CPU_NPP_vs_FastNPP.cu", "836a98e"),
    ("cleanup-window-a (CPU time)", "benchmark_CPU_OpenCV_vs_cvGS_cu", "opencv-cuda/benchmark_CPU_OpenCV_vs_cvGS.cu", "836a98e"),
    ("cleanup-window-d/cleaned/e1 (VI-A, E1)", "e1_vf_chain_cu", "handwritten-cuda/e1_vf_chain.cu", "1a2535c"),
    ("cleanup-window-d/cleaned/e1 (VI-A, E1)", "e1_hf_batch_cu", "handwritten-cuda/e1_hf_batch.cu", "1a2535c"),
    ("cleanup-window-d/cleaned/e1 (VI-A, E1)", "e1_preproc_cu", "handwritten-cuda/e1_preproc.cu", "1a2535c"),
    ("cleanup-window-e (E2)", "e2_fkl_chain_cu", "handwritten-cuda/e2_fkl_chain.cu", "9a6a88b"),
    ("cleanup-window-d/cleaned/e16-summary (E16)", "e16_attention_cu", "fkl-self/attention/e16_attention.cu", "f27c605"),
    ("e19-crop-sweep (E19)", "e19_crop_sweep_cu", "opencv-cuda/stencil/e19_crop_sweep.cu", "246b1c8"),
]
INC = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)


def git(*args):
    return subprocess.run(["git", "-C", a.repo, *args], capture_output=True, text=True, check=True).stdout


def show(path, rev=MAIN):
    try:
        return git("show", f"{rev}:{path}")
    except subprocess.CalledProcessError:
        return None


def closure(src):
    seen, todo = [], [src]
    while todo:
        f = todo.pop(0)
        if f in seen:
            continue
        text = show(f)
        if text is None:
            continue
        seen.append(f)
        for inc in INC.findall(text):
            for cand in (inc, os.path.join(os.path.dirname(f), inc), os.path.join("handwritten-cuda", inc)):
                cand = os.path.normpath(cand)
                if show(cand) is not None and cand not in seen:
                    todo.append(cand)
                    break
    return seen


def link(rev, sub):
    out = git("ls-tree", rev, sub).split()
    return out[2][:8] if len(out) > 2 else "-"


lines = [f"main = {MAIN}; the include closure is taken at main"]
for record, prog, src, rev in PROGRAMS:
    full = git("rev-parse", rev).strip()
    files = closure(src)
    lines.append("")
    lines.append(f"== {record}: {prog}, measured at {rev} ({full})")
    lines.append(f"   gitlinks at {rev}: fkl {link(full, 'external/fkl')}, cvGPUSpeedup {link(full, 'external/cvGPUSpeedup')}, FastNPP {link(full, 'external/FastNPP')}; "
                 f"at main: fkl {link(MAIN, 'external/fkl')}, cvGPUSpeedup {link(MAIN, 'external/cvGPUSpeedup')}, FastNPP {link(MAIN, 'external/FastNPP')}")
    lines.append(f"   files ({len(files)}): {', '.join(files)}")
    stat = git("diff", "--stat=200", f"{full}..{MAIN}", "--", *files).rstrip()
    lines.append("   git diff --stat " + rev + "..main: " + ("no change" if not stat else ""))
    if stat:
        lines += ["     " + l for l in stat.splitlines()]
        log = git("log", "--oneline", "--no-merges", f"{full}..{MAIN}", "--", *files).rstrip()
        lines.append("   the commits that change them:")
        lines += ["     " + l for l in log.splitlines()]
text = "\n".join(lines)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
