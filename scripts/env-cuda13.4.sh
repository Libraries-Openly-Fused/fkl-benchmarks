# Source this before configuring, building or running anything in this repository:  source scripts/env-cuda13.4.sh
# It selects the CUDA 13.4 toolkit (the CMake presets and the configure check refuse any other nvcc) and makes sure the
# 13.4 libraries are the ones loaded at run time: a CUDA lib64 directory that an older toolkit left in LD_LIBRARY_PATH would
# otherwise take precedence over the RUNPATH of the binaries and silently load another cudart or NPP.
#   FKLB_CUDA_HOME   the toolkit root (default /usr/local/cuda-13.4); set it before sourcing for another location
#   CUDA_VISIBLE_DEVICES   the GPU to measure; indices follow CUDA_DEVICE_ORDER=PCI_BUS_ID, i.e. nvidia-smi
export FKLB_CUDA_HOME=${FKLB_CUDA_HOME:-/usr/local/cuda-13.4}
if [ ! -x "$FKLB_CUDA_HOME/bin/nvcc" ]; then
    echo "scripts/env-cuda13.4.sh: no nvcc in $FKLB_CUDA_HOME/bin (set FKLB_CUDA_HOME to the CUDA 13.4 toolkit root)" >&2
    return 1 2>/dev/null || exit 1
fi
export PATH=$FKLB_CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$FKLB_CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export CUDA_DEVICE_ORDER=PCI_BUS_ID
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "scripts/env-cuda13.4.sh: CUDA_VISIBLE_DEVICES is not set; set it to the nvidia-smi index of the GPU to measure" >&2
fi
