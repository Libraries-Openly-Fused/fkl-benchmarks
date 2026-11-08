# vfhf_devcode.py <scripts dir> <binary A> [<binary B>] [--llvm DIR]: the gfx1200 device code of a multi-TU HIP binary, whose .hip_fatbin holds
# one offload bundle per translation unit (scripts/devcode_compare.py reads only the first): every bundle is unbundled, its functions are listed
# with the rule of scripts/isa_pipeline_kernels.py (devcode_compare's grouping, labels folded, __hip_cuid_ normalised), and for each fused kernel
# (a name with StaticLoop) the FMA operations are counted (each operation of a VOPD pair on its own). With B: the per-function comparison.
import collections
import os
import re
import subprocess
import sys
import tempfile

args = [a for a in sys.argv[1:] if not a.startswith('--llvm')]
llvm = next((a.split('=', 1)[1] for a in sys.argv[1:] if a.startswith('--llvm=')), '/opt/rocm/core-10.1/lib/llvm/bin')
sys.path.insert(0, args[0])
import devcode_compare as dc  # noqa: E402
import isa_pipeline_kernels as ipk  # noqa: E402

MAGIC = b'__CLANG_OFFLOAD_BUNDLE__'
FMA = re.compile(r'^v_(dual_)?fma(c|k|mk|ak)?_f32\b')


def device_functions(exe, tmp):
    fb = os.path.join(tmp, 'fb.bin')
    subprocess.run([llvm + '/llvm-objcopy', '--dump-section', '.hip_fatbin=' + fb, exe, os.devnull], check=True)
    data = open(fb, 'rb').read()
    starts = [m.start() for m in re.finditer(re.escape(MAGIC), data)]
    funcs = collections.defaultdict(list)   # name -> [lines per bundle that defines it]
    for i, s in enumerate(starts):
        part, co = os.path.join(tmp, f'b{i}.bin'), os.path.join(tmp, f'co{i}.o')
        open(part, 'wb').write(data[s:starts[i + 1] if i + 1 < len(starts) else len(data)])
        subprocess.run([llvm + '/clang-offload-bundler', '--unbundle', '--type=o', '--targets=hipv4-amdgcn-amd-amdhsa--gfx1200',
                        '--input=' + part, '--output=' + co], check=True, capture_output=True)
        for name, lines in ipk.instruction_lines(llvm, co).items():
            funcs[name].append(lines)
    return len(starts), funcs


def fma_ops(lines):
    return sum(1 for line in lines for op in line.split('::') if FMA.match(op.strip()))


with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
    nA, A = device_functions(args[1], tmp)
    dupA = sorted(n for n, v in A.items() if len(v) > 1)
    print(f'A: {nA} bundles, {len(A)} functions, defined in more than one bundle: {len(dupA)}')
    fused = sorted(((fma_ops(v[0]), len(v[0])) for n, v in A.items() if 'StaticLoop' in n and 'launchTransformDPP_Kernel' in n))
    print(f'fused kernels: {len(fused)}; (FMA operations, instructions): {fused}')
    other = sorted(n for n in A if 'StaticLoop' not in n)
    print(f'other kernels: {len(other)}; their FMA operations: {sorted(fma_ops(A[n][0]) for n in other)}')
    if len(args) > 2:
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp2:
            nB, B = device_functions(args[2], tmp2)
        same = sum(1 for n in A if n in B and [dc.digest(x) for x in A[n]] == [dc.digest(x) for x in B[n]])
        print(f'B: {nB} bundles, {len(B)} functions; identical {same}, different {sum(1 for n in A if n in B) - same}, '
              f'only in A {len(set(A) - set(B))}, only in B {len(set(B) - set(A))}')
