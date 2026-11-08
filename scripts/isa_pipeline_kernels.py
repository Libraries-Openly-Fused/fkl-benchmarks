# isa_pipeline_kernels.py: the gfx1200 ISA review of the pipeline (a) kernels of benchmark_rpp_preprocessing_hip, from the code object of the
# measured binary: the counterpart on the RX 9060 of the SASS review of Table III. Compile-only: nothing is run on the GPU.
#
# usage: python3 -I scripts/isa_pipeline_kernels.py <binary> <remark build log> <out dir> [--llvm <dir>] [--arch gfx1200]
#   <binary>            the measured benchmark_rpp_preprocessing_hip; its .hip_fatbin must hold one offload bundle (one HIP translation unit)
#   <remark build log>  the build log of the same sources compiled with -Rpass-analysis=kernel-resource-usage, whose device code is the same as
#                       the binary's (scripts/devcode_compare.py): the compiler's own occupancy of every kernel
# For fkl_fused, fkl_fused_halfpixel (also the kernel of fkl_fused_halfpixel_once), hand_batch, hand_padded, hand_roi and hand_roi_halfpixel
# (e1c::handPreprocRoiHalfPixel, the kernel of hand_roi_halfpixel_once) at N = 10, 150 and 300:
#   instructions    the lines of the function's disassembly (llvm-objdump -d --symbolize-operands), grouped as scripts/devcode_compare.py groups
#                   them (the function with its branch-target blocks), without the '...' lines that stand for zero bytes
#   vgprs, sgprs    .vgpr_count and .sgpr_count of the code-object metadata (NT_AMDGPU_METADATA); vgprs_allocated is .amdhsa_next_free_vgpr of
#                   the kernel descriptor, the allocation in whole granules
#   scratch_bytes   the private segment: .private_segment_fixed_size of the metadata, which must equal .amdhsa_private_segment_fixed_size of the
#                   descriptor; dynamic_stack: .uses_dynamic_stack
#   lds_bytes       the group segment: .group_segment_fixed_size of the metadata, which must equal .amdhsa_group_segment_fixed_size
#   kernarg_bytes   .kernarg_segment_size, which must equal .amdhsa_kernarg_size
#   waves_per_simd  the compiler's "Occupancy [waves/SIMD]" (the kernel-resource-usage remark); waves_formula is
#                   min(16, floor(1536 / (24 * ceil(vgprs / 24)))), gfx1200 in wave32 with no LDS and no scratch (the limits are in the README);
#                   the two must agree
# Writes <out dir>/isa-table.csv and <out dir>/listings/<variant>-N<N>.txt: the instruction lines with the branch labels folded and no newline
# after the last, so that the sha256 of the file is the hash scripts/devcode_compare.py computes for the function. Prints the README's table.
import argparse
import csv
import math
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import devcode_compare as dc  # noqa: E402  (code_object, the patterns and the hash of scripts/devcode_compare.py)

BATCHES = (10, 150, 300)
FKL = re.compile(r'BatchReadILNS_11PlanePolicyE0ELm(\d+)E')
VARIANTS = [
    ('fkl_fused', lambda n: 'launchTransformDPP_Kernel' in n and 'TensorSplit' in n and 'Resize' in n and 'ResizeSamplingE0' in n, FKL),
    ('fkl_fused_halfpixel', lambda n: 'launchTransformDPP_Kernel' in n and 'TensorSplit' in n and 'Resize' in n and 'ResizeSamplingE1' in n, FKL),
    ('hand_batch', lambda n: n.startswith('_ZN8rppbench3e1c11handPreprocILi'), re.compile(r'^_ZN8rppbench3e1c11handPreprocILi(\d+)E')),
    ('hand_padded', lambda n: n.startswith('_ZN8rppbench3e1c17handPreprocPaddedILi'), re.compile(r'^_ZN8rppbench3e1c17handPreprocPaddedILi(\d+)E')),
    ('hand_roi', lambda n: n.startswith('_ZN8rppbench3e1c14handPreprocRoiILi'), re.compile(r'^_ZN8rppbench3e1c14handPreprocRoiILi(\d+)E')),
    ('hand_roi_halfpixel', lambda n: n.startswith('_ZN8rppbench3e1c23handPreprocRoiHalfPixelILi'),
     re.compile(r'^_ZN8rppbench3e1c23handPreprocRoiHalfPixelILi(\d+)E')),
]
META_KEYS = ('group_segment_fixed_size', 'kernarg_segment_size', 'private_segment_fixed_size', 'sgpr_count', 'sgpr_spill_count', 'vgpr_count',
             'vgpr_spill_count', 'wavefront_size', 'uses_dynamic_stack', 'symbol')
KD_KEYS = ('amdhsa_group_segment_fixed_size', 'amdhsa_private_segment_fixed_size', 'amdhsa_kernarg_size', 'amdhsa_next_free_vgpr',
           'amdhsa_next_free_sgpr', 'amdhsa_wavefront_size32', 'amdhsa_workgroup_processor_mode', 'amdhsa_enable_private_segment')


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, errors='replace', check=True).stdout


def metadata(llvm, co):
    """{kernel name: {key: value}} from the code-object metadata printed by llvm-readelf --notes"""
    text = run([llvm + '/llvm-readelf', '--notes', co])
    out = {}
    # the entries of amdhsa.kernels start with '  - .args:'; their own fields are indented by 4 spaces, the fields of their arguments deeper
    for entry in re.split(r'\n  - \.args:', text)[1:]:
        values = {}
        for key in META_KEYS:
            m = re.search(r'^    \.' + key + r':\s+(\S+)', entry, re.M)
            if m:
                values[key] = m.group(1)
        if values.get('symbol', '').endswith('.kd'):
            out[values['symbol'][:-3]] = values
    return out


def descriptor(llvm, co, name):
    """{key: value} of the kernel descriptor <name>.kd, as llvm-objdump prints it"""
    text = run([llvm + '/llvm-objdump', '-D', '--disassemble-symbols=' + name + '.kd', co])
    return {k: int(m.group(1)) for k in KD_KEYS for m in [re.search(r'\.' + k + r'\s+(-?\d+)', text)] if m}


def instruction_lines(llvm, co):
    """{function name: [instruction lines]}, grouped as scripts/devcode_compare.py groups them (labels folded, trailing '...' padding removed)"""
    groups = []
    for line in run([llvm + '/llvm-objdump', '-d', '--symbolize-operands', co]).split('\n'):
        m = dc.HEADER.match(line)
        if m:
            name = dc.CUID.sub('__hip_cuid_@', m.group(1))
            if dc.LABEL_NAME.fullmatch(name) and groups:
                groups[-1][1].append(None)   # a branch-target block of the current function: its lines join the function's
            else:
                groups.append([name, []])
            continue
        if not groups:
            continue
        code = line.split('//', 1)[0].strip()
        if code:
            groups[-1][1].append(dc.LABEL.sub('L@', dc.CUID.sub('__hip_cuid_@', code)))
    out = {}
    for name, lines in groups:
        lines = [x for x in lines if x is not None]
        while lines and lines[-1] == '...':
            lines.pop()
        out[name] = lines
    return out


def remarks(path):
    """{function name: {field: value}} of the kernel-resource-usage remarks of a build log"""
    out, cur = {}, None
    for line in open(path, errors='replace'):
        m = re.search(r'Function Name: (\S+)', line)
        if m:
            cur = out.setdefault(m.group(1), {})
            continue
        m = re.search(r'^\S+: +(TotalSGPRs|VGPRs|ScratchSize \[bytes/lane\]|Dynamic Stack|Occupancy \[waves/SIMD\]|SGPRs Spill|VGPRs Spill|'
                      r'LDS Size \[bytes/block\]): (\S+)', line)
        if m and cur is not None:
            cur[m.group(1)] = m.group(2)
    return out


def waves_formula(vgprs):
    return min(16, 1536 // (24 * math.ceil(vgprs / 24)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('binary')
    ap.add_argument('remark_log')
    ap.add_argument('out')
    ap.add_argument('--llvm', default='/opt/rocm/core-10.1/lib/llvm/bin')
    ap.add_argument('--arch', default='gfx1200')
    a = ap.parse_args()
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        fb = os.path.join(tmp, 'fb.bin')
        run([a.llvm + '/llvm-objcopy', '--dump-section', '.hip_fatbin=' + fb, a.binary, os.devnull])
        bundles = open(fb, 'rb').read().count(b'__CLANG_OFFLOAD_BUNDLE__')
        if bundles != 1:
            sys.exit(f'{a.binary}: {bundles} offload bundles in .hip_fatbin; this script reads one HIP translation unit')
        co = dc.code_object(a.llvm, a.arch, a.binary, tmp)
        meta = metadata(a.llvm, co)
        ins = instruction_lines(a.llvm, co)
        rem = remarks(a.remark_log)
        os.makedirs(os.path.join(a.out, 'listings'), exist_ok=True)
        rows = []
        for variant, match, number in VARIANTS:
            found = {}
            for name in meta:
                m = number.search(name) if match(name) else None
                if m and int(m.group(1)) in BATCHES:
                    if int(m.group(1)) in found:
                        sys.exit(f'{variant}: two kernels at N = {m.group(1)}')
                    found[int(m.group(1))] = name
            for n in BATCHES:
                name = found.get(n)
                if name is None:
                    sys.exit(f'{variant}: no kernel at N = {n}')
                md, kd, r, lines = meta[name], descriptor(a.llvm, co, name), rem.get(name, {}), ins[name]
                checks = [int(md['private_segment_fixed_size']) == kd['amdhsa_private_segment_fixed_size'],
                          int(md['group_segment_fixed_size']) == kd['amdhsa_group_segment_fixed_size'],
                          int(md['kernarg_segment_size']) == kd['amdhsa_kernarg_size'],
                          r.get('VGPRs') == md['vgpr_count'], r.get('TotalSGPRs') == md['sgpr_count'],
                          int(r.get('Occupancy [waves/SIMD]', -1)) == waves_formula(int(md['vgpr_count']))]
                if not all(checks):
                    sys.exit(f'{variant} N = {n}: the metadata, the descriptor and the remark disagree: {checks}')
                listing = '\n'.join(lines)
                path = os.path.join(a.out, 'listings', f'{variant}-N{n}.txt')
                with open(path, 'w') as f:
                    f.write(listing)
                rows.append({'variant': variant, 'batch': n, 'instructions': sum(1 for x in lines if x != '...'), 'vgprs': int(md['vgpr_count']),
                             'vgprs_allocated': kd['amdhsa_next_free_vgpr'], 'sgprs': int(md['sgpr_count']),
                             'scratch_bytes': int(md['private_segment_fixed_size']), 'dynamic_stack': md.get('uses_dynamic_stack', ''),
                             'lds_bytes': int(md['group_segment_fixed_size']), 'kernarg_bytes': int(md['kernarg_segment_size']),
                             'vgpr_spills': int(md['vgpr_spill_count']), 'sgpr_spills': int(md['sgpr_spill_count']),
                             'wavefront_size': int(md['wavefront_size']), 'workgroup_processor_mode': kd.get('amdhsa_workgroup_processor_mode', ''),
                             'waves_per_simd': int(r['Occupancy [waves/SIMD]']), 'waves_formula': waves_formula(int(md['vgpr_count'])),
                             'listing_sha256': dc.digest(lines), 'kernel': name})
    with open(os.path.join(a.out, 'isa-table.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print('| variant | N | instructions | VGPRs (allocated) | SGPRs | scratch, bytes | LDS, bytes | kernarg, bytes | waves per SIMD |')
    print('|---|---|---|---|---|---|---|---|---|')
    for r in rows:
        print(f"| {r['variant']} | {r['batch']} | {r['instructions']} | {r['vgprs']} ({r['vgprs_allocated']}) | {r['sgprs']} | {r['scratch_bytes']} | "
              f"{r['lds_bytes']} | {r['kernarg_bytes']} | {r['waves_per_simd']} |")
    print(f"\nscratch in any of the {len(rows)} kernels: {any(r['scratch_bytes'] for r in rows)}; dynamic stack: "
          f"{sorted({r['dynamic_stack'] for r in rows})}; spills: {any(r['vgpr_spills'] or r['sgpr_spills'] for r in rows)}; LDS: "
          f"{any(r['lds_bytes'] for r in rows)}; wavefront sizes: {sorted({r['wavefront_size'] for r in rows})}")


if __name__ == '__main__':
    main()
