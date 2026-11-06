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
"""One-block CSV writer for the Python benchmarks (triton/, torch-inductor/, pytorch-attention/).

Same layout as common/csv_writer.h: the system info as '# key: value' comment lines, one header row, one row per
measurement. For the Python baselines the header records torch, torch.version.cuda, triton, flash-attn, and the ptxas that
Triton uses (it must be the CUDA 13.4 one, see scripts/triton_env.sh; Triton silently falls back to its bundled 12.9 ptxas
if the override path is unusable, so the header states the version it resolved).

    with CsvWriter("sdpa.csv", "E16 SDPA flash", ["seq", "tflops"]) as csv:
        csv.row(1024, 123.4)
"""
import os
import platform
import re
import subprocess


def _run(cmd):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
        return out.splitlines()[0] if out else "unavailable"
    except Exception:  # noqa: BLE001 - the header must never stop a benchmark
        return "unavailable"


def system_info_lines(experiment, device_index=None, notes=None):
    import torch

    lines = [f"experiment: {experiment}"]
    if notes:
        lines.append(f"notes: {notes}")
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # FKLB_FKL_TREE: a library tree other than the pin of this repository (a compile sweep against the head of a PR, scripts/e3_sweep.py --include): the header then names THAT tree
    fkl_tree = os.environ.get("FKLB_FKL_TREE") or f"{repo}/external/fkl"
    for name, path in (("repo", repo), ("fkl", fkl_tree), ("cvGPUSpeedup", f"{repo}/external/cvGPUSpeedup"),
                       ("FastNPP", f"{repo}/external/FastNPP")):
        head = _run(["git", "-C", path, "rev-parse", "HEAD"])
        dirty = _run(["git", "-C", path, "status", "--porcelain", "--untracked-files=no"])
        lines.append(f"{name}: {head}{'-dirty' if dirty not in ('', 'unavailable') else ''}")
    if os.environ.get("FKLB_FKL_TREE"):
        lines.append(f"fkl tree (FKLB_FKL_TREE, not the pin external/fkl of this repository): {fkl_tree}")
    lines.append(f"python: {platform.python_version()}, torch: {torch.__version__}, torch.version.cuda: {torch.version.cuda}")
    try:
        import triton
        from triton import knobs
        tools = []
        for n in ("ptxas", "ptxas_blackwell", "cuobjdump", "nvdisasm"):
            t = getattr(knobs.nvidia, n)
            tools.append(f"{n} {t.path} (release {t.version})")
        lines.append(f"triton: {triton.__version__}; " + "; ".join(tools))
    except Exception as e:  # noqa: BLE001
        lines.append(f"triton: unavailable ({e})")
    try:
        import flash_attn
        lines.append(f"flash_attn: {flash_attn.__version__}")
    except Exception:  # noqa: BLE001
        lines.append("flash_attn: not installed")
    try:
        os_name = platform.freedesktop_os_release()["PRETTY_NAME"]
    except (OSError, KeyError, AttributeError):
        os_name = "unavailable"
    lines.append(f"os: {os_name}")   # the system the run was made on, read at run time (as common/system_info.h does for the C++ programs)
    lines.append(f"kernel: {os.uname().sysname} {os.uname().release}")
    idx = torch.cuda.current_device() if device_index is None else device_index
    p = torch.cuda.get_device_properties(idx)
    lines.append(f"gpu: {p.name}, cc {p.major}.{p.minor}, {p.multi_processor_count} SMs, {p.total_memory >> 20} MiB")
    bus = _run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader", "-i", str(idx)])
    smi = ["nvidia-smi", "--format=csv,noheader", "-i", str(idx), "--query-gpu="]
    lines.append(f"cuda_visible_devices: {os.environ.get('CUDA_VISIBLE_DEVICES', '')} (device {idx}, pci {bus})")
    lines.append("nvidia-smi driver: " + _run(smi[:-1] + [smi[-1] + "driver_version"]))
    lines.append("clocks (sm, mem, max sm, max mem, MHz): " + _run(smi[:-1] + [smi[-1] + "clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem"]))
    lines.append("power limit, temperature at start: " + _run(smi[:-1] + [smi[-1] + "power.limit,temperature.gpu"]))
    return lines


class CsvWriter:
    def __init__(self, path, experiment, columns, device_index=None, notes=None):
        self._f = open(path, "w")
        for line in system_info_lines(experiment, device_index, notes):
            self._f.write(f"# {line}\n")
        self._f.write(",".join(columns) + "\n")
        self._n = len(columns)

    @staticmethod
    def _cell(value):
        text = repr(value) if isinstance(value, float) else str(value)
        if any(c in text for c in ',"\r\n'):   # a text with a comma, a quote or a line break is written quoted, so that every row has as many fields as the header
            return '"' + text.replace('"', '""') + '"'
        return text

    def row(self, *values):
        assert len(values) == self._n, "the number of values is not the number of columns"
        self._f.write(",".join(self._cell(v) for v in values) + "\n")
        self._f.flush()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()
