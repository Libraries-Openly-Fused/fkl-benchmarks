# Device code of the HIP executables of two build trees (for example the same sources built against two FKL commits), compared
# function by function.
#
# usage: python3 -I scripts/devcode_compare.py <bin dir A> <bin dir B> [--glob '*_hip'] [--only name1,name2] [--arch gfx1200]
#                [--llvm <dir of llvm-objcopy, clang-offload-bundler, llvm-objdump>] [--table <dir>] [--labels-as-symbols]
#
# For every executable that both directories have: llvm-objcopy --dump-section .hip_fatbin, clang-offload-bundler --unbundle (the
# code object of --arch), llvm-objdump -d --symbolize-operands. The instruction text of each function is hashed (sha256), with the
# comments (addresses, encodings) removed and the __hip_cuid_<hash> names normalised. llvm-objdump prints the local branch
# targets as symbols L<n>, numbered per code object: by default their text belongs to the function that contains them and the
# references are normalised (a renumbering is not a code difference); --labels-as-symbols keeps them as separate symbols with
# their numbers, the stricter form. The '...' lines with which llvm-objdump stands for a run of zero bytes are not compared when they come after
# the last instruction of a function (the padding before the next one); a '...' inside a function is compared. --table writes
# <dir>/<executable>.tsv: function, sha256 in A, sha256 in B, same.
# Exit 0 = the same executables in both directories and every function of every executable identical.
import argparse
import fnmatch
import hashlib
import os
import re
import subprocess
import sys
import tempfile

HEADER = re.compile(r'^[0-9a-f]+ <(.+)>:$')
CUID = re.compile(r'__hip_cuid_[0-9a-f]+')
LABEL = re.compile(r'\bL\d+\b')
LABEL_NAME = re.compile(r'L\d+')


def code_object(llvm, arch, exe, tmpdir):
    fb = os.path.join(tmpdir, 'fb.bin')
    co = os.path.join(tmpdir, 'co.o')
    for p in (fb, co):
        if os.path.exists(p):
            os.remove(p)
    subprocess.run([llvm + '/llvm-objcopy', '--dump-section', '.hip_fatbin=' + fb, exe, os.devnull], capture_output=True, text=True)
    if not os.path.exists(fb):
        return None
    for target in ('hip-amdgcn-amd-amdhsa--' + arch, 'hipv4-amdgcn-amd-amdhsa--' + arch):
        subprocess.run([llvm + '/clang-offload-bundler', '--unbundle', '--type=o', '--targets=' + target, '--input=' + fb,
                        '--output=' + co], capture_output=True, text=True)
        if os.path.exists(co) and os.path.getsize(co) > 0:
            return co
    return None


def digest(lines):
    """sha256 of the instruction text of a function, without the trailing '...' lines (llvm-objdump prints '...' for a run of 8 or more zero
    bytes that it does not disassemble; after the last instruction of a function that is the padding before the next one, which moves with the
    layout of the code object). A '...' inside a function is kept and compared."""
    end = len(lines)
    while end > 0 and lines[end - 1] == '...':
        end -= 1
    return hashlib.sha256('\n'.join(lines[:end]).encode()).hexdigest()


def functions(llvm, co, labels_as_symbols):
    """{name: sha256 of the instruction text} of the code object."""
    out = {}
    cur, name = None, None
    prev_name, prev_cur = None, None
    p = subprocess.Popen([llvm + '/llvm-objdump', '-d', '--symbolize-operands', co], stdout=subprocess.PIPE, text=True,
                         errors='replace')
    for line in p.stdout:
        line = line.rstrip('\n')
        m = HEADER.match(line)
        if m:
            if name is not None:
                out[name] = digest(cur)
            name, cur = CUID.sub('__hip_cuid_@', m.group(1)), []
            if not labels_as_symbols and LABEL_NAME.fullmatch(name):
                # a local branch target, not a function: its text belongs to the function before it
                name, cur = prev_name, prev_cur
            else:
                prev_name, prev_cur = name, cur
            continue
        if name is None:
            continue
        code = line.split('//', 1)[0].strip()
        if code:
            code = CUID.sub('__hip_cuid_@', code)
            cur.append(code if labels_as_symbols else LABEL.sub('L@', code))
    if name is not None:
        out[name] = digest(cur)
    p.wait()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--glob', default='*_hip')
    ap.add_argument('--only', default='')
    ap.add_argument('--arch', default='gfx1200')
    ap.add_argument('--llvm', default=os.path.join(os.environ.get('ROCM_PATH', '/opt/rocm'), 'lib', 'llvm', 'bin'))
    ap.add_argument('--table', default='')
    ap.add_argument('--labels-as-symbols', action='store_true')
    args = ap.parse_args()
    names_a = sorted(n for n in os.listdir(args.a) if fnmatch.fnmatch(n, args.glob))
    names_b = sorted(n for n in os.listdir(args.b) if fnmatch.fnmatch(n, args.glob))
    if args.only:
        keep = set(args.only.split(','))
        names_a = [n for n in names_a if n in keep]
        names_b = [n for n in names_b if n in keep]
    if args.table:
        os.makedirs(args.table, exist_ok=True)
    common = [n for n in names_a if n in set(names_b)]
    print(f'binaries: A {len(names_a)}, B {len(names_b)}, in both {len(common)}')
    for n in sorted(set(names_a) - set(names_b)):
        print(f'  only in A: {n}')
    for n in sorted(set(names_b) - set(names_a)):
        print(f'  only in B: {n}')
    tot = dict(funcs=0, ident=0, diff=0, onlyA=0, onlyB=0, nocode=0, nodev=0)
    bad = []
    with tempfile.TemporaryDirectory(dir=os.environ.get('TMPDIR')) as tmp:
        for n in common:
            os.makedirs(tmp + '/a', exist_ok=True)
            os.makedirs(tmp + '/b', exist_ok=True)
            ca = code_object(args.llvm, args.arch, os.path.join(args.a, n), tmp + '/a')
            cb = code_object(args.llvm, args.arch, os.path.join(args.b, n), tmp + '/b')
            if ca is None and cb is None:
                tot['nodev'] += 1       # no device code at all in either executable (no kernel in the program)
                print(f'  {n}: no {args.arch} device code in either')
                continue
            if ca is None or cb is None:
                tot['nocode'] += 1
                print(f'  {n}: {args.arch} code object only in {"B" if ca is None else "A"}')
                continue
            fa, fb = functions(args.llvm, ca, args.labels_as_symbols), functions(args.llvm, cb, args.labels_as_symbols)
            same = sum(1 for k in fa if k in fb and fa[k] == fb[k])
            diff = [k for k in fa if k in fb and fa[k] != fb[k]]
            oa, ob = [k for k in fa if k not in fb], [k for k in fb if k not in fa]
            tot['funcs'] += len(fa)
            tot['ident'] += same
            tot['diff'] += len(diff)
            tot['onlyA'] += len(oa)
            tot['onlyB'] += len(ob)
            flag = '' if not (diff or oa or ob) else '   <-- DIFFERENCES'
            print(f'  {n}: functions A {len(fa)}, B {len(fb)}, identical {same}, different {len(diff)}, only A {len(oa)}, '
                  f'only B {len(ob)}{flag}')
            if diff or oa or ob:
                bad.append(n)
                for k in (diff + oa + ob)[:3]:
                    print('       ', k[:140])
            if args.table:
                with open(os.path.join(args.table, n + '.tsv'), 'w') as t:
                    t.write('function\tsha256_A\tsha256_B\tsame\n')
                    for k in sorted(set(fa) | set(fb)):
                        ha, hb = fa.get(k, '-'), fb.get(k, '-')
                        t.write(f'{k}\t{ha}\t{hb}\t{"yes" if ha == hb else "no"}\n')
    print(f"TOTAL: {len(common)} binaries, {tot['funcs']} functions in A: identical {tot['ident']}, different {tot['diff']}, "
          f"only in A {tot['onlyA']}, only in B {tot['onlyB']}, code object in only one of them {tot['nocode']}; "
          f"{tot['nodev']} binaries have no device code in either")
    ok = not bad and not tot['nocode'] and set(names_a) == set(names_b)
    print('DEVICE CODE ' + ('IDENTICAL' if ok else 'DIFFERENCES'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
