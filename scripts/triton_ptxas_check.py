# Compiles a trivial Triton kernel with the ptxas override of scripts/triton_env.sh and prints the toolkit string stamped
# in the cubin. Run it on each GPU:  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 python scripts/triton_ptxas_check.py
import os, re, subprocess, sys, tempfile
C = "/home/oscar/tfkl-bench/toolchains/cuda-13.4.3/bin"
os.environ["TRITON_PTXAS_PATH"] = f"{C}/ptxas"
os.environ["TRITON_PTXAS_BLACKWELL_PATH"] = f"{C}/ptxas"
os.environ["TRITON_CUOBJDUMP_PATH"] = f"{C}/cuobjdump"
os.environ["TRITON_NVDISASM_PATH"] = f"{C}/nvdisasm"
os.environ["TRITON_CACHE_DIR"] = tempfile.mkdtemp(prefix="triton-cache-")
import torch, triton, triton.language as tl
from triton import knobs
for n in ("ptxas", "ptxas_blackwell", "cuobjdump", "nvdisasm"):
    t = getattr(knobs.nvidia, n)
    print(f"{n:16s} {t.path}  release {t.version}")
@triton.jit
def axpy(x, y, o, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(o + i, tl.load(x + i, mask=m) * 2.0 + tl.load(y + i, mask=m), mask=m)
x = torch.randn(1 << 20, device="cuda"); y = torch.randn_like(x); o = torch.empty_like(x)
k = axpy[(x.numel() // 1024,)](x, y, o, x.numel(), BLOCK=1024)
assert torch.allclose(o, x * 2 + y)
p = torch.cuda.get_device_properties(0)
cub = k.asm["cubin"]
f = os.path.join(os.environ["TRITON_CACHE_DIR"], "k.cubin"); open(f, "wb").write(cub)
tk = subprocess.run(["strings", "-a", f], capture_output=True, text=True).stdout
m = re.findall(r"Cuda compilation tools, release [0-9.]+, V[0-9.]+", tk)
print("device", p.name, f"sm_{p.major}{p.minor}", "| torch", torch.__version__, "cuda", torch.version.cuda, "| triton", triton.__version__)
print("cubin built by:", sorted(set(m)) or "no toolkit string found")
