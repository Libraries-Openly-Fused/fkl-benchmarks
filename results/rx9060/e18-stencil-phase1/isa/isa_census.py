# isa_census.py <dis.s> <notes.txt> <remarks.txt> <cxxfilt>: one row per kernel of the gfx1200 code object of e18_stencil_hip.
# instructions: lines of the kernel's body in llvm-objdump -d (every instruction line, from its symbol to the next kernel symbol; labels and the
# s_code_end padding after the last kernel are not counted). The rest from the kernel's metadata note (.kernarg_segment_size, the explicit
# arguments = .args without the hidden_ ones) and from -Rpass-analysis=kernel-resource-usage.
import re, subprocess, sys
dis, notes, remarks, filt = sys.argv[1:5]
def dem(s): return subprocess.run([filt], input=s, capture_output=True, text=True).stdout.strip()
def role(d):
    if 'gpuSpin' in d: return 'gpuSpin'
    samp = 'HALF_PIXEL' if 'ResizeSampling)1' in d else 'ASYMMETRIC'
    if 'linearFilterDPPKernel' in d:
        return ('fused ' if 'SaturateCast' in d else 'identity-check ') + samp
    if 'TransformDPP' in d: return 'resize-alone ' + samp
    return '?'
# instructions
counts, cur = {}, None
for line in open(dis):
    m = re.match(r'^[0-9a-f]+ <(.+)>:$', line)
    if m:
        n = m.group(1)
        if not re.match(r'^L\d+$', n): cur = n if (n.startswith('_Z') and ('Kernel' in n or 'gpuSpin' in n)) else None; counts.setdefault(cur, 0) if cur else None
        continue
    if cur and re.match(r'^\s+\S', line) and not line.strip().startswith(';'):
        if 's_code_end' in line: continue
        counts[cur] += 1
# notes: per kernel block
meta, blk = {}, {}
explicit = 0; inargs = False; argsize = None; argkind = None
for line in open(notes):
    s = line.strip()
    if s.startswith('- .args:') or s == '.args:': inargs = True
    m = re.match(r'-?\s*\.size:\s+(\d+)', s)
    if inargs and m: argsize = int(m.group(1))
    m = re.match(r'\.value_kind:\s+(\S+)', s)
    if inargs and m:
        if not m.group(1).startswith('hidden_'): explicit += argsize or 0
        argsize = None
    for k in ('group_segment_fixed_size', 'kernarg_segment_size', 'private_segment_fixed_size', 'sgpr_count', 'vgpr_count'):
        m = re.match(r'\.' + k + r':\s+(\d+)', s)
        if m: blk[k] = int(m.group(1))
    m = re.match(r'\.name:\s+(\S+)', s)
    if m: blk['name'] = m.group(1)
    if s.startswith('.wavefront_size:'):
        blk['explicit'] = explicit; meta[blk['name']] = blk; blk = {}; explicit = 0; inargs = False
# remarks
rem, cur = {}, None
for line in open(remarks):
    m = re.search(r'Function Name: (\S+)', line)
    if m: cur = m.group(1); rem[cur] = {}; continue
    m = re.search(r':\s{4,}([^:]+?):\s+(\S+)$', line)
    if cur and m: rem[cur][m.group(1).strip()] = m.group(2)
cols = ['role', 'instructions', 'VGPRs', 'SGPRs (total)', 'LDS B/block', 'scratch B/lane', 'spills (SGPR/VGPR)', 'kernarg B (explicit)', 'occupancy waves/SIMD', 'symbol']
print('\t'.join(cols))
for n in counts:
    d = dem(n); r = rem.get(n, {}); k = meta.get(n, {})
    print('\t'.join(str(x) for x in [role(d), counts[n], r.get('VGPRs'), r.get('TotalSGPRs'), r.get('LDS Size [bytes/block]'), r.get('ScratchSize [bytes/lane]'),
                                      f"{r.get('SGPRs Spill')}/{r.get('VGPRs Spill')}", f"{k.get('kernarg_segment_size')} ({k.get('explicit')})",
                                      r.get('Occupancy [waves/SIMD]'), n]))
