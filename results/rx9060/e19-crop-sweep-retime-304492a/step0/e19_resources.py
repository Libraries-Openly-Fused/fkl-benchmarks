# e19_resources.py: the resources of the fkl_batch kernel of e19_crop_sweep_hip (the batched MorphologyDPP box blur of runCrops<N>) at every N
# of the crop sweep, from the gfx1200 code object of each binary given. Compile products only: nothing is run on the GPU.
#
# usage: python3 -I e19_resources.py <out.csv> <label>=<e19_crop_sweep_hip> [<label>=<e19_crop_sweep_hip> ...] [--scripts <dir>] [--llvm <dir>]
#   The kernel is fk::morphologyDPPKernel<...BatchRead<PROCESS_ALL, N, ...>...> before FKL #435 and
#   fk::launchDPP_Kernel<fk::MorphologyDPP<...>, ...BatchRead<PROCESS_ALL, N, ...>...> after it; exactly one per N must exist.
#   The fields come from isa_pipeline_kernels.py and devcode_compare.py of this repository's scripts/ (--scripts), as for Figure 15's ISA table:
#     instructions    the lines of the kernel's disassembly, grouped with its branch-target blocks, without the '...' zero-byte lines
#     vgprs, sgprs    .vgpr_count and .sgpr_count of the code-object metadata; vgprs_allocated is .amdhsa_next_free_vgpr of the descriptor
#     lds_bytes       .group_segment_fixed_size (must equal the descriptor's .amdhsa_group_segment_fixed_size)
#     scratch_bytes   .private_segment_fixed_size (must equal .amdhsa_private_segment_fixed_size); dynamic_stack: .uses_dynamic_stack
#     kernarg_bytes   .kernarg_segment_size (must equal .amdhsa_kernarg_size); vgpr_spills, sgpr_spills: the metadata's spill counts
#     listing_sha256  the sha256 of the listing as devcode_compare.py hashes it, so equal hashes mean identical code
import argparse
import csv
import hashlib
import os
import re
import sys
import tempfile

SWEEP = (1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150)
BATCH = re.compile(r'BatchReadILNS_11PlanePolicyE0ELm(\d+)E')


def is_batch_kernel(name):
    return name.startswith('_ZN2fk19morphologyDPPKernel') or name.startswith('_ZN2fk16launchDPP_KernelINS_13MorphologyDPPI')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('binaries', nargs='+')
    ap.add_argument('--scripts', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', '..', 'scripts'))
    ap.add_argument('--llvm', default='/opt/rocm/core-10.1/lib/llvm/bin')
    ap.add_argument('--arch', default='gfx1200')
    a = ap.parse_args()
    sys.path.insert(0, os.path.abspath(a.scripts))
    import devcode_compare as dc            # noqa: E402
    import isa_pipeline_kernels as ipk      # noqa: E402
    rows = []
    for spec in a.binaries:
        label, path = spec.split('=', 1)
        with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
            co = dc.code_object(a.llvm, a.arch, path, tmp)
            meta = ipk.metadata(a.llvm, co)
            ins = ipk.instruction_lines(a.llvm, co)
            found = {}
            for name in meta:
                m = BATCH.search(name) if is_batch_kernel(name) else None
                if m:
                    n = int(m.group(1))
                    if n in found:
                        sys.exit(f'{label}: two batch kernels at N = {n}')
                    found[n] = name
            if sorted(found) != list(SWEEP):
                sys.exit(f'{label}: batch kernels at N = {sorted(found)}, not the sweep')
            for n in SWEEP:
                name = found[n]
                md, kd, lines = meta[name], ipk.descriptor(a.llvm, co, name), ins[name]
                checks = [int(md['private_segment_fixed_size']) == kd['amdhsa_private_segment_fixed_size'],
                          int(md['group_segment_fixed_size']) == kd['amdhsa_group_segment_fixed_size'],
                          int(md['kernarg_segment_size']) == kd['amdhsa_kernarg_size']]
                if not all(checks):
                    sys.exit(f'{label} N = {n}: the metadata and the descriptor disagree: {checks}')
                rows.append({'binary': label, 'n': n, 'kernel': name[:60] + '...',
                             'instructions': sum(1 for x in lines if x != '...'), 'vgprs': int(md['vgpr_count']),
                             'vgprs_allocated': kd['amdhsa_next_free_vgpr'], 'sgprs': int(md['sgpr_count']),
                             'lds_bytes': int(md['group_segment_fixed_size']), 'scratch_bytes': int(md['private_segment_fixed_size']),
                             'dynamic_stack': md.get('uses_dynamic_stack', ''), 'kernarg_bytes': int(md['kernarg_segment_size']),
                             'vgpr_spills': int(md['vgpr_spill_count']), 'sgpr_spills': int(md['sgpr_spill_count']),
                             'listing_sha256': hashlib.sha256('\n'.join(lines).encode()).hexdigest()})
    with open(a.out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    fields = ('instructions', 'vgprs', 'vgprs_allocated', 'sgprs', 'lds_bytes', 'scratch_bytes', 'kernarg_bytes', 'vgpr_spills', 'sgpr_spills')
    labels = [s.split('=', 1)[0] for s in a.binaries]
    print('| N | binary | ' + ' | '.join(fields) + ' | listing sha256 |')
    print('|' + '---|' * (len(fields) + 3))
    for n in SWEEP:
        for r in [r for r in rows if r['n'] == n]:
            print(f"| {n} | {r['binary']} | " + ' | '.join(str(r[k]) for k in fields) + f" | {r['listing_sha256'][:12]} |")
    for x, y in zip(labels, labels[1:]):
        same = sum(1 for n in SWEEP if next(r for r in rows if r['binary'] == x and r['n'] == n)['listing_sha256']
                   == next(r for r in rows if r['binary'] == y and r['n'] == n)['listing_sha256'])
        print(f'{x} against {y}: {same} of {len(SWEEP)} kernels identical')


if __name__ == '__main__':
    main()
