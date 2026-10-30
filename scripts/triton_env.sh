# Source this before any Triton / TorchInductor / PyTorch run: it makes Triton use the CUDA 13.4 toolkit's ptxas, cuobjdump
# and nvdisasm instead of the CUDA 12.9 ones bundled in the Triton wheel.
#   source scripts/triton_env.sh
# Triton falls back to the bundled tools without a word if an override path is not usable, so verify with
#   python scripts/triton_ptxas_check.py
# which compiles a kernel and prints the toolkit string stamped in its cubin.
export FKLB_CUDA_HOME=${FKLB_CUDA_HOME:?source scripts/env-cuda13.4.sh first (it sets FKLB_CUDA_HOME)}
export TRITON_PTXAS_PATH=$FKLB_CUDA_HOME/bin/ptxas
export TRITON_PTXAS_BLACKWELL_PATH=$FKLB_CUDA_HOME/bin/ptxas   # sm_120 has its own variable
export TRITON_CUOBJDUMP_PATH=$FKLB_CUDA_HOME/bin/cuobjdump
export TRITON_NVDISASM_PATH=$FKLB_CUDA_HOME/bin/nvdisasm
