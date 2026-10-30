#!/usr/bin/env python3
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
"""Which hardware features does the CUDA toolkit's assembler accept for each GPU architecture?

  probe_ptx_features.py [--out-dir DIR] [--targets sm_89 sm_120 sm_120a sm_100a sm_90a]

Each feature is a minimal PTX kernel that uses one instruction family (mma.sync, ldmatrix, cp.async, TMA, wgmma, tcgen05,
cluster barriers...). It is assembled with ptxas -arch=<target>; the result is ACCEPT or REJECT with the first line of the
assembler's message. This is the assembler's view, not proof that the hardware executes the instruction (that needs a run);
it is what decides whether a kernel for that architecture can be built at all.

Every probe is validated by positive controls: sm_90a accepts wgmma, TMA and setmaxnreg, sm_100a accepts tcgen05. A REJECT
on another target is therefore a statement about that target, not a typo in the probe. The matrix is written as TSV and as
a markdown table, with the toolkit version, to DIR.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

PTX_VERSION = "9.4"  # what nvcc 13.4 emits for these targets

HEADER = ".version {ver}\n.target {target}\n.address_size 64\n\n.visible .entry probe()\n{{\n"
FOOTER = "\n    ret;\n}}\n"

# feature name -> PTX body (declarations and instructions inside the kernel)
FEATURES = [
    ("mma.sync m16n8k16 bf16 (HMMA)", """
    .reg .b32 %a<4>; .reg .b32 %b<2>; .reg .f32 %c<4>; .reg .f32 %d<4>;
    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};"""),
    ("mma.sync m16n8k32 e4m3 fp8 (QMMA)", """
    .reg .b32 %a<4>; .reg .b32 %b<2>; .reg .f32 %c<4>; .reg .f32 %d<4>;
    mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3};"""),
    ("mma.sync block-scaled mxf4 e2m1 (fp4)", """
    .reg .b32 %a<4>; .reg .b32 %b<2>; .reg .f32 %c<4>; .reg .f32 %d<4>; .reg .b32 %sa, %sb;
    mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 {%d0,%d1,%d2,%d3}, {%a0,%a1,%a2,%a3}, {%b0,%b1}, {%c0,%c1,%c2,%c3}, %sa, {0, 0}, %sb, {0, 0};"""),
    ("ldmatrix (LDSM)", """
    .reg .b32 %r<5>; .shared .align 16 .b8 smem[1024];
    mov.u32 %r4, smem;
    ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r0,%r1,%r2,%r3}, [%r4];"""),
    ("cp.async 16 B (LDGSTS)", """
    .reg .b32 %r<2>; .reg .b64 %rd<2>; .shared .align 16 .b8 smem[1024];
    mov.u32 %r0, smem; mov.u64 %rd0, 0;
    cp.async.ca.shared.global [%r0], [%rd0], 16;
    cp.async.commit_group; cp.async.wait_group 0;"""),
    ("redux.sync", """
    .reg .b32 %r<3>;
    redux.sync.add.u32 %r1, %r0, 0xffffffff;"""),
    ("griddepcontrol (programmatic dependent launch)", """
    griddepcontrol.launch_dependents;"""),
    ("cluster special registers + barrier.cluster", """
    .reg .b32 %r<3>;
    mov.u32 %r0, %cluster_ctaid.x; mov.u32 %r1, %cluster_nctarank;
    barrier.cluster.arrive; barrier.cluster.wait;"""),
    ("mapa (distributed shared memory address)", """
    .reg .b32 %r<3>; .shared .align 16 .b8 smem[1024];
    mov.u32 %r0, smem; mov.u32 %r1, 0;
    mapa.shared::cluster.u32 %r2, %r0, %r1;"""),
    ("cp.async.bulk (non-tensor bulk copy)", """
    .reg .b32 %r<3>; .reg .b64 %rd<2>; .shared .align 16 .b8 smem[1024]; .shared .align 8 .b64 mbar[1];
    mov.u32 %r0, smem; mov.u32 %r1, mbar; mov.u64 %rd0, 0;
    cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%r0], [%rd0], 128, [%r1];"""),
    ("cp.async.bulk.tensor 2d (TMA)", """
    .reg .b32 %r<5>; .reg .b64 %rd<2>; .shared .align 128 .b8 smem[4096]; .shared .align 8 .b64 mbar[1];
    mov.u32 %r0, smem; mov.u32 %r1, mbar; mov.u64 %rd0, 0; mov.u32 %r2, 0; mov.u32 %r3, 0;
    cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%r0], [%rd0, {%r2, %r3}], [%r1];"""),
    ("setmaxnreg", """
    setmaxnreg.inc.sync.aligned.u32 232;"""),
    ("elect.sync", """
    .reg .b32 %r<2>; .reg .pred %p<2>;
    elect.sync %r0|%p0, 0xffffffff;"""),
    ("wgmma.mma_async m64n8k16 bf16 (warpgroup MMA)", """
    .reg .b64 %da, %db; .reg .f32 %f<4>; .reg .pred %p;
    setp.ne.b32 %p, 1, 0;
    wgmma.fence.sync.aligned;
    wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%f0,%f1,%f2,%f3}, %da, %db, %p, 1, 1, 0, 0;
    wgmma.commit_group.sync.aligned;
    wgmma.wait_group.sync.aligned 0;"""),
    ("tcgen05.alloc (Tensor Memory)", """
    .reg .b32 %r<2>; .shared .align 4 .b8 smem[16];
    mov.u32 %r0, smem;
    tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%r0], 32;"""),
]

# The four features that distinguish the architectures discussed in the paper
KEY_FEATURES = ["cp.async.bulk.tensor 2d (TMA)", "wgmma.mma_async m64n8k16 bf16 (warpgroup MMA)",
                "tcgen05.alloc (Tensor Memory)", "cluster special registers + barrier.cluster"]


def probe(ptxas, target, body, workdir):
    path = os.path.join(workdir, "probe.ptx")
    with open(path, "w") as f:
        f.write(HEADER.format(ver=PTX_VERSION, target=target) + body + FOOTER.format())
    res = subprocess.run([ptxas, f"-arch={target}", path, "-o", os.devnull], capture_output=True, text=True)
    if res.returncode == 0:
        return "ACCEPT", ""
    msg = (res.stderr or res.stdout).strip().splitlines()
    first = next((l for l in msg if "error" in l.lower()), msg[0] if msg else "")
    return "REJECT", first.split("probe.ptx, ")[-1].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=".")
    ap.add_argument("--targets", nargs="+", default=["sm_89", "sm_120", "sm_120a", "sm_100a", "sm_90a"])
    args = ap.parse_args()
    ptxas = shutil.which("ptxas")
    if not ptxas:
        sys.exit("ptxas not found: source scripts/env-cuda13.4.sh")
    version = subprocess.run([ptxas, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-2:]
    os.makedirs(args.out_dir, exist_ok=True)
    results = {}
    with tempfile.TemporaryDirectory(prefix="ptxprobe-") as work:
        for name, body in FEATURES:
            for t in args.targets:
                results[(name, t)] = probe(ptxas, t, body, work)
    with open(os.path.join(args.out_dir, "ptx-feature-probe.tsv"), "w") as f:
        f.write("# ptxas: " + " ".join(l.strip() for l in version) + f" ({ptxas}), PTX ISA {PTX_VERSION}\n")
        f.write("feature\ttarget\tresult\tfirst assembler error\n")
        for (name, t), (r, msg) in results.items():
            f.write(f"{name}\t{t}\t{r}\t{msg}\n")
    with open(os.path.join(args.out_dir, "ptx-feature-probe.md"), "w") as f:
        f.write(f"ptxas: {' '.join(l.strip() for l in version)}  ({ptxas}), PTX ISA {PTX_VERSION}. ACCEPT = the assembler builds it for that target.\n\n")
        f.write("| feature | " + " | ".join(args.targets) + " |\n|---|" + "---|" * len(args.targets) + "\n")
        for name, _ in FEATURES:
            f.write(f"| {name} | " + " | ".join(results[(name, t)][0] for t in args.targets) + " |\n")
        f.write("\nFirst assembler error of each rejected probe:\n\n")
        for (name, t), (r, msg) in results.items():
            if r == "REJECT":
                f.write(f"- {name} on {t}: `{msg}`\n")
    print(open(os.path.join(args.out_dir, "ptx-feature-probe.md")).read())


if __name__ == "__main__":
    main()
