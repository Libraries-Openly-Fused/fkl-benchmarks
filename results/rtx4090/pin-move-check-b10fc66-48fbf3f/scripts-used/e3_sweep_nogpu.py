#!/usr/bin/env python3
"""C-PM-CUDA: scripts/e3_sweep.py without the GPU query of its CSV header (compile only: nothing is executed and no GPU is visible).

  e3_sweep_nogpu.py <fkl-benchmarks checkout> <e3_sweep.py arguments...>

e3_sweep.py writes its CSVs with common/fklb_csv.py's CsvWriter, whose header (system_info_lines) imports torch and asks the GPU for its
properties, and nvidia-smi for the driver, clocks, power and temperature. This wrapper replaces system_info_lines with one that writes the same
repository, library, python and system lines, and "gpu: not queried" in place of the GPU lines; then it runs e3_sweep.py unchanged (runpy), so the
compile commands and the parsing of their output are those of the script.
"""
import os
import platform
import runpy
import sys

repo = os.path.abspath(sys.argv[1])
sys.path.insert(0, os.path.join(repo, "scripts"))
sys.path.insert(0, os.path.join(repo, "common"))
import fklb_csv  # noqa: E402


def system_info_lines(experiment, device_index=None, notes=None):
    lines = [f"experiment: {experiment}"]
    if notes:
        lines.append(f"notes: {notes}")
    fkl_tree = os.environ.get("FKLB_FKL_TREE") or f"{repo}/external/fkl"
    for name, path in (("repo", repo), ("fkl", fkl_tree), ("cvGPUSpeedup", f"{repo}/external/cvGPUSpeedup"),
                       ("FastNPP", f"{repo}/external/FastNPP")):
        head = fklb_csv._run(["git", "-C", path, "rev-parse", "HEAD"])
        dirty = fklb_csv._run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"])
        lines.append(f"{name}: {head}{'-dirty' if dirty not in ('', 'unavailable') else ''}")
    lines.append(f"python: {platform.python_version()} (torch not imported)")
    try:
        os_name = platform.freedesktop_os_release()["PRETTY_NAME"]
    except (OSError, KeyError, AttributeError):
        os_name = "unavailable"
    lines.append(f"os: {os_name}")
    lines.append(f"kernel: {os.uname().sysname} {os.uname().release}")
    lines.append(f"gpu: not queried (compile only; CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}'; "
                 "scratch/pin-435/e3_sweep_nogpu.py in place of fklb_csv.system_info_lines)")
    return lines


fklb_csv.system_info_lines = system_info_lines
script = os.path.join(repo, "scripts", "e3_sweep.py")
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
