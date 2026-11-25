# label_owners.py <devcode_compare_renamed.py> <bin dir A> <bin dir B> <functions-labels dir> <program>...: for every symbol that differs in the
# --labels-as-symbols table of a program, the function that holds it in A and in B (the function header before it in llvm-objdump's
# --symbolize-operands disassembly of the gfx1200 code object). Uses the code-object extraction of the comparison script itself.
import importlib.util
import os
import re
import subprocess
import sys
import tempfile

spec = importlib.util.spec_from_file_location('dc', sys.argv[1])
dc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dc)
A, B, TABLES = sys.argv[2], sys.argv[3], sys.argv[4]
LLVM = os.path.join(os.environ.get('ROCM_PATH', '/opt/rocm'), 'lib', 'llvm', 'bin')


def owners(exe, tmp):
    cos = dc.code_objects(LLVM, 'gfx1200', exe, tmp)
    if len(cos) != 1:
        raise SystemExit(f'{exe}: {len(cos)} code objects, this check handles one')
    own, cur = {}, None
    out = subprocess.run([LLVM + '/llvm-objdump', '-d', '--symbolize-operands', cos[0][1]], capture_output=True, text=True).stdout
    for line in out.splitlines():
        m = dc.HEADER.match(line)
        if not m:
            continue
        name = m.group(1)
        if dc.LABEL_NAME.fullmatch(name):
            own[name] = cur
        else:
            cur = name
            own[name] = name
    dem = dc.demangled(LLVM, sorted(set(own.values())))
    return {k: dc.key_of(dem.get(v, v))[0] for k, v in own.items()}   # key_of gives (key, number of removals)


for prog in sys.argv[5:]:
    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        oa, ob = owners(os.path.join(A, prog), ta), owners(os.path.join(B, prog), tb)
    counts = {}
    with open(os.path.join(TABLES, prog + '.tsv')) as f:
        next(f)
        for line in f:
            key, _, _, _, _, same = line.rstrip('\n').split('\t')
            if same == 'yes':
                continue
            wa = oa.get(key, '(none)') if dc.LABEL_NAME.fullmatch(key) else key
            wb = ob.get(key, '(none)') if dc.LABEL_NAME.fullmatch(key) else key
            cls = 'linearFilterDPPKernel' if 'fk::linearFilterDPPKernel<' in wa and 'fk::linearFilterDPPKernel<' in wb else 'OTHER'
            counts[cls] = counts.get(cls, 0) + 1
            if cls == 'OTHER':
                print(f'{prog}\t{key}\tA: {wa[:120]}\tB: {wb[:120]}')
    print(f'{prog}: differing symbols in the labels table, by the function that holds them in A and in B: {counts}')
