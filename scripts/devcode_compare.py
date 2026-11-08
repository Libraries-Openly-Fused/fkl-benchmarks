# Device code of the HIP executables of two build trees (for example the same sources built against two FKL commits), compared
# function by function.
#
# usage: python3 -I scripts/devcode_compare.py <bin dir A> <bin dir B> [--glob '*_hip'] [--only name1,name2] [--arch gfx1200]
#                [--llvm <dir of llvm-objcopy, clang-offload-bundler, llvm-objdump>] [--table <dir>] [--labels-as-symbols]
#
# For every executable that both directories have: llvm-objcopy --dump-section .hip_fatbin, then clang-offload-bundler --unbundle (the
# code object of --arch) of every offload bundle in it, then llvm-objdump -d --symbolize-operands of every code object.
#   .hip_fatbin holds one offload bundle per HIP translation unit of the executable. The bundles are read one after the other, with only zero
#   bytes (the linker's alignment padding) allowed between them; the script stops, naming the executable, at anything else (a compressed
#   bundle, for example), and when only some of the bundles have a code object of --arch. An executable with one bundle is unbundled from the
#   section as a whole, as before.
#   The instruction text of each function is hashed (sha256), with the comments (addresses, encodings) removed and the __hip_cuid_<hash> names
#   normalised. llvm-objdump prints the local branch targets as symbols L<n>, numbered per code object: by default their text belongs to the
#   function that contains them and the references are normalised (a renumbering is not a code difference); --labels-as-symbols keeps them as
#   separate symbols with their numbers, the stricter form. The '...' lines with which llvm-objdump stands for a run of zero bytes are not
#   compared when they come after the last instruction of a function (the padding before the next one); a '...' inside a function is compared.
#   With several code objects, the functions of all of them are compared by name. A name defined in more than one of them (the runtime stubs
#   that every translation unit holds, a kernel instantiated in two translation units) is compared by the set of the distinct hashes of its
#   definitions, not by their number; with --labels-as-symbols the branch targets are then named <function>:L<n>, since their numbers are per
#   code object. The line of such an executable ends with the number of its code objects in A and in B.
# --table writes <dir>/<executable>.tsv: function, sha256 in A, sha256 in B, same; a function with several distinct definitions has its sorted
# hashes joined by ','.
# Exit 0 = the same executables in both directories and every function of every executable identical.
import argparse
import fnmatch
import hashlib
import os
import re
import struct
import subprocess
import sys
import tempfile

HEADER = re.compile(r'^[0-9a-f]+ <(.+)>:$')
CUID = re.compile(r'__hip_cuid_[0-9a-f]+')
LABEL = re.compile(r'\bL\d+\b')
LABEL_NAME = re.compile(r'L\d+')
MAGIC = b'__CLANG_OFFLOAD_BUNDLE__'
PART = re.compile(r'^(b|co)\d+\.(bin|o)$')


def bundle_ranges(data):
    """[(start, end)] of the offload bundles of a .hip_fatbin, in order: uncompressed bundles one after the other, only zero bytes between and
    after them. A bundle is the magic, the number of its entries (u64), then per entry its offset from the start of the bundle, its size and
    the length of its target triple (u64 each) and the triple; it ends where its last entry ends. Raises ValueError on anything else."""
    out, pos = [], 0
    while True:
        i = data.find(MAGIC, pos)
        gap = data[pos:] if i < 0 else data[pos:i]
        if gap.strip(b'\0'):
            at = pos + len(gap) - len(gap.lstrip(b'\0'))
            raise ValueError(f'unrecognised data at offset {at} of .hip_fatbin, not an uncompressed offload bundle')
        if i < 0:
            return out
        p = i + len(MAGIC)
        try:
            (n,) = struct.unpack_from('<Q', data, p)
            p += 8
            if n > 1000:
                raise ValueError(f'an offload bundle header at offset {i} of .hip_fatbin claims {n} entries')
            end = p
            for _ in range(n):
                off, size, tlen = struct.unpack_from('<QQQ', data, p)
                p += 24 + tlen
                end = max(end, i + off + size)
        except struct.error:
            raise ValueError(f'a truncated offload bundle header at offset {i} of .hip_fatbin')
        if end > len(data):
            raise ValueError(f'the offload bundle at offset {i} of .hip_fatbin ends beyond the section')
        out.append((i, end))
        pos = end


def unbundle(llvm, arch, fatbin, out):
    """The code object of --arch of one offload bundle (the file fatbin), written to out, or None."""
    for target in ('hip-amdgcn-amd-amdhsa--' + arch, 'hipv4-amdgcn-amd-amdhsa--' + arch):
        subprocess.run([llvm + '/clang-offload-bundler', '--unbundle', '--type=o', '--targets=' + target, '--input=' + fatbin,
                        '--output=' + out], capture_output=True, text=True)
        if os.path.exists(out) and os.path.getsize(out) > 0:
            return out
    return None


def code_objects(llvm, arch, exe, tmpdir):
    """The code objects of --arch of an executable, one per offload bundle of its .hip_fatbin, in the order of the bundles; None when it has no
    .hip_fatbin or none of its bundles has a code object of --arch. Exits, naming the executable, when .hip_fatbin cannot be read as offload
    bundles or when only some of them have a code object of --arch."""
    fb = os.path.join(tmpdir, 'fb.bin')
    for f in os.listdir(tmpdir):
        if f in ('fb.bin', 'co.o') or PART.match(f):
            os.remove(os.path.join(tmpdir, f))
    subprocess.run([llvm + '/llvm-objcopy', '--dump-section', '.hip_fatbin=' + fb, exe, os.devnull], capture_output=True, text=True)
    if not os.path.exists(fb):
        return None
    with open(fb, 'rb') as f:
        data = f.read()
    try:
        ranges = bundle_ranges(data)
    except ValueError as e:
        sys.exit(f'{exe}: {e}')
    if len(ranges) <= 1:   # one bundle (one HIP translation unit): unbundled from the section as a whole, as before
        co = unbundle(llvm, arch, fb, os.path.join(tmpdir, 'co.o'))
        return None if co is None else [co]
    cos = []
    for k, (start, end) in enumerate(ranges):
        part = os.path.join(tmpdir, f'b{k}.bin')
        with open(part, 'wb') as f:
            f.write(data[start:end])
        cos.append(unbundle(llvm, arch, part, os.path.join(tmpdir, f'co{k}.o')))
    have = sum(1 for c in cos if c)
    if have == 0:
        return None
    if have < len(cos):
        sys.exit(f'{exe}: {len(cos) - have} of the {len(cos)} offload bundles of its .hip_fatbin have no {arch} code object')
    return cos


def code_object(llvm, arch, exe, tmpdir):
    """The code object of --arch of an executable with one offload bundle (one HIP translation unit), or None. Exits, naming the executable,
    when its .hip_fatbin holds more than one bundle: code_objects reads them all."""
    cos = code_objects(llvm, arch, exe, tmpdir)
    if cos and len(cos) > 1:
        sys.exit(f'{exe}: its .hip_fatbin holds {len(cos)} offload bundles (one per HIP translation unit); code_object reads only one, '
                 'use code_objects')
    return cos[0] if cos else None


def digest(lines):
    """sha256 of the instruction text of a symbol."""
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def functions(llvm, co, labels_as_symbols, qualify_labels=False):
    """{name: sha256 of the instruction text} of the code object.

    The symbols of the disassembly are grouped: a function, then the local branch targets (L<n>) inside it, up to the next function. Only the
    '...' lines after the last instruction of a group are left out: llvm-objdump prints '...' for a run of 8 or more zero bytes that it does not
    disassemble, and at the end of a function that is the padding before the next one, which moves with the layout of the code object. A '...'
    anywhere else, also at the end of a branch-target block inside a function, is kept and compared. By default a function is hashed with the text
    of its branch targets and their references normalised; with labels_as_symbols every symbol is hashed on its own, with its numbers, and with
    qualify_labels as well the branch targets are named <function>:L<n>."""
    groups = []   # [[name, lines], ...] per function: the function first, then its branch targets
    p = subprocess.Popen([llvm + '/llvm-objdump', '-d', '--symbolize-operands', co], stdout=subprocess.PIPE, text=True,
                         errors='replace')
    for line in p.stdout:
        line = line.rstrip('\n')
        m = HEADER.match(line)
        if m:
            name = CUID.sub('__hip_cuid_@', m.group(1))
            if LABEL_NAME.fullmatch(name) and groups:
                groups[-1].append([name, []])
            else:
                groups.append([[name, []]])
            continue
        if not groups:
            continue
        code = line.split('//', 1)[0].strip()
        if code:
            code = CUID.sub('__hip_cuid_@', code)
            groups[-1][-1][1].append(code if labels_as_symbols else LABEL.sub('L@', code))
    p.wait()
    out = {}
    for group in groups:
        last = group[-1][1]
        while last and last[-1] == '...':   # the padding after the last instruction of the function
            last.pop()
        if labels_as_symbols:
            for k, (name, lines) in enumerate(group):
                out[f'{group[0][0]}:{name}' if k and qualify_labels else name] = digest(lines)
        else:
            out[group[0][0]] = digest([code for _, lines in group for code in lines])
    return out


def functions_of(llvm, cos, labels_as_symbols):
    """{name: hash} of the functions of all the code objects of an executable. With one code object, functions() of it. With several, a name
    defined in more than one of them has the sorted distinct hashes of its definitions, joined by ','."""
    if len(cos) == 1:
        return functions(llvm, cos[0], labels_as_symbols)
    seen = {}
    for co in cos:
        for name, h in functions(llvm, co, labels_as_symbols, qualify_labels=True).items():
            seen.setdefault(name, set()).add(h)
    return {name: ','.join(sorted(hs)) for name, hs in seen.items()}


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
            ca = code_objects(args.llvm, args.arch, os.path.join(args.a, n), tmp + '/a')
            cb = code_objects(args.llvm, args.arch, os.path.join(args.b, n), tmp + '/b')
            if ca is None and cb is None:
                tot['nodev'] += 1       # no device code at all in either executable (no kernel in the program)
                print(f'  {n}: no {args.arch} device code in either')
                continue
            if ca is None or cb is None:
                tot['nocode'] += 1
                print(f'  {n}: {args.arch} code object only in {"B" if ca is None else "A"}')
                continue
            fa, fb = functions_of(args.llvm, ca, args.labels_as_symbols), functions_of(args.llvm, cb, args.labels_as_symbols)
            same = sum(1 for k in fa if k in fb and fa[k] == fb[k])
            diff = [k for k in fa if k in fb and fa[k] != fb[k]]
            oa, ob = [k for k in fa if k not in fb], [k for k in fb if k not in fa]
            tot['funcs'] += len(fa)
            tot['ident'] += same
            tot['diff'] += len(diff)
            tot['onlyA'] += len(oa)
            tot['onlyB'] += len(ob)
            flag = '' if not (diff or oa or ob) else '   <-- DIFFERENCES'
            objects = f' (code objects A {len(ca)}, B {len(cb)})' if len(ca) > 1 or len(cb) > 1 else ''
            print(f'  {n}: functions A {len(fa)}, B {len(fb)}, identical {same}, different {len(diff)}, only A {len(oa)}, '
                  f'only B {len(ob)}{objects}{flag}')
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
