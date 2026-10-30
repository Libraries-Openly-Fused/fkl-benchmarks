#!/usr/bin/env bash
# Dumps what an experiment ran on into results/<system>/system-info/ (committed with the CSVs):
#   scripts/system_info.sh <system> <nvidia-smi index>      e.g.  scripts/system_info.sh rtx4090 1
#   scripts/system_info.sh rtx5050 0
# nvidia-smi indices are PCI-bus ordered (CUDA_DEVICE_ORDER=PCI_BUS_ID is forced here). Source scripts/env-cuda13.4.sh first.
set -u
SYSTEM=${1:?system name, e.g. rtx4090}; IDX=${2:?nvidia-smi index of the GPU}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT=$ROOT/results/$SYSTEM/system-info
mkdir -p "$OUT"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$IDX
command -v nvcc >/dev/null || { echo "nvcc not found: source scripts/env-cuda13.4.sh"; exit 1; }

{
  echo "# system-info of $SYSTEM, $(date -Iseconds)"
  echo "nvidia-smi index: $IDX   CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "nvcc: $(command -v nvcc)"; nvcc --version | tail -2
  echo "ptxas: $(ptxas --version | tail -2 | head -1)"
  echo "cuobjdump: $(cuobjdump --version | tail -2 | head -1)"
  echo "host compiler: $(c++ --version | head -1)"
  echo "kernel: $(uname -srvm)"
  echo "os: $(. /etc/os-release && echo "$PRETTY_NAME")"
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')  ($(nproc) threads)"
  echo "memory: $(free -g | awk '/Mem:/ {print $2" GiB"}')"
  for s in fkl cvGPUSpeedup FastNPP; do echo "submodule $s: $(git -C "$ROOT/external/$s" rev-parse HEAD)"; done
  echo "fkl-benchmarks: $(git -C "$ROOT" rev-parse HEAD)$(git -C "$ROOT" status --porcelain --untracked-files=no | grep -q . && echo -dirty)"
  if [ -x "$ROOT/../venv-torch/bin/python" ]; then
    "$ROOT/../venv-torch/bin/python" - <<'PY' 2>/dev/null
import torch, triton
print("torch:", torch.__version__, "| torch.version.cuda:", torch.version.cuda, "| triton:", triton.__version__)
try:
    import flash_attn; print("flash_attn:", flash_attn.__version__)
except Exception: print("flash_attn: not installed")
PY
  fi
} > "$OUT/system.txt"

# the UUID and serial numbers are left out: these files are committed to a public repository
nvidia-smi -i "$IDX" -q | grep -v -E "UUID|Serial Number" > "$OUT/nvidia-smi-q.txt"
nvidia-smi -i "$IDX" --query-gpu=name,driver_version,pci.bus_id,compute_cap,vbios_version,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.limit,memory.total --format=csv > "$OUT/nvidia-smi.csv"
lscpu > "$OUT/lscpu.txt"

# Device attributes: built for the architectures of this repository so that it runs on either GPU
nvcc -std=c++17 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_120,code=sm_120 \
     "$ROOT/common/device_attributes.cu" -o "${TMPDIR:-/tmp}/fklb_device_attributes_$$" && \
  "${TMPDIR:-/tmp}/fklb_device_attributes_$$" > "$OUT/device-attributes.txt"; rm -f "${TMPDIR:-/tmp}/fklb_device_attributes_$$"

# What the assembler accepts for each architecture (the same for any GPU; stored next to the GPU it explains)
python3 "$ROOT/scripts/probe_ptx_features.py" --out-dir "$OUT" > /dev/null
echo "written to $OUT:"; ls "$OUT"
