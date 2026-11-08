# Device code of the HIP executables of two build trees (for example the same sources built against two FKL commits), compared
# function by function.
#
# usage: python3 -I scripts/devcode_compare.py <bin dir A> <bin dir B> [--glob '*_hip'] [--only name1,name2] [--arch gfx1200]
#                [--llvm <dir of llvm-objcopy, llvm-objdump>] [--table <dir>] [--labels-as-symbols]
#
# For every executable that both directories have: llvm-objcopy --dump-section .hip_fatbin, the code object of --arch of every offload
# bundle in it, then llvm-objdump -d --symbolize-operands of every code object.
#   .hip_fatbin holds one offload bundle per HIP translation unit. The bundles are read from their headers, one after the other, with only zero
#   bytes (the linker's alignment padding) allowed between them, and the code object of --arch (the entry hip- or hipv4-amdgcn-amd-amdhsa--<arch>)
#   is copied out of each bundle as its header gives it. The script stops, naming the executable, at anything else: a compressed bundle, data
#   that is not a bundle, a bundle with two entries of --arch, or only some of the bundles with a code object of --arch.
#   The instruction text of each function is hashed (sha256), with the comments (addresses, encodings) removed and the __hip_cuid_<hash> names
#   normalised. llvm-objdump prints the local branch targets as symbols L<n>, numbered per code object: by default their text belongs to the
#   function that contains them and the references are normalised (a renumbering is not a code difference); --labels-as-symbols keeps them as
#   separate symbols with their numbers, the stricter form. The '...' lines with which llvm-objdump stands for a run of zero bytes are not
#   compared when they come after the last instruction of a function (the padding before the next one); a '...' inside a function is compared.
#   With one code object on each side, the functions are compared by name, as they always were. With more than one on a side:
#   - when both sides have the same number of code objects, with the same triples, they are paired by position (the order of the bundles is
#     the link order) and each function is compared code object by code object: the same name must have the same instruction text in the same
#     code object;
#   - otherwise each function is compared by the multiset of the hashes of its definitions over all the code objects (a name such as the
#     runtime stubs that every translation unit holds then also differs when the number of its definitions differs);
#   and with --labels-as-symbols the branch targets of both sides are named <function>:L<n>, since their numbers are per code object. The line
#   of such an executable ends with the numbers of code objects and the mode.
# --table writes <dir>/<executable>.tsv: function, sha256 in A, sha256 in B, same. With several code objects a hash cell is
# "<position>:<sha256> ..." (paired by position) or "<sha256>*<count> ..." (multisets).
# Exit 0 = the same executables in both directories and every function of every executable identical.
import argparse
import collections
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
PART = re.compile(r'^co\d+\.o$')


def bundle_entries(data):
    """The offload bundles of a .hip_fatbin in order, each as the list of its entries (target triple, start, size), start counted from the
    start of the section. The section must be uncompressed bundles one after the other, with only zero bytes between and after them. A bundle
    is the magic, the number of its entries (u64), then per entry its offset from the start of the bundle, its size and the length of its
    triple (u64 each) and the triple; it ends where its header or its last entry ends, whichever is later. Raises ValueError on anything else."""
    bundles, pos = [], 0
    while True:
        i = data.find(MAGIC, pos)
        gap = data[pos:] if i < 0 else data[pos:i]
        if gap.strip(b'\0'):
            at = pos + len(gap) - len(gap.lstrip(b'\0'))
            raise ValueError(f'unrecognised data at offset {at} of .hip_fatbin, not an uncompressed offload bundle')
        if i < 0:
            return bundles
        p = i + len(MAGIC)
        try:
            (n,) = struct.unpack_from('<Q', data, p)
            p += 8
            if n > 1000:
                raise ValueError(f'an offload bundle header at offset {i} of .hip_fatbin claims {n} entries')
            entries = []
            for _ in range(n):
                off, size, tlen = struct.unpack_from('<QQQ', data, p)
                p += 24
                if p + tlen > len(data):
                    raise struct.error
                triple = data[p:p + tlen].decode('ascii', 'replace')
                p += tlen
                if i + off + size > len(data):
                    raise ValueError(f'an entry of the offload bundle at offset {i} of .hip_fatbin ends beyond the section')
                entries.append((triple, i + off, size))
        except struct.error:
            raise ValueError(f'a truncated offload bundle header at offset {i} of .hip_fatbin')
        bundles.append(entries)
        pos = max([p] + [start + size for _, start, size in entries])


def code_objects(llvm, arch, exe, tmpdir):
    """[(triple, path)]: the code object of --arch of every offload bundle of an executable's .hip_fatbin, in the order of the bundles, each
    copied out of its bundle into tmpdir; None when the executable has no .hip_fatbin or none of its bundles has a code object of --arch.
    Exits, naming the executable, when .hip_fatbin cannot be read as offload bundles, when a bundle has two entries of --arch, or when only some
    of the bundles have one."""
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
        bundles = bundle_entries(data)
    except ValueError as e:
        sys.exit(f'{exe}: {e}')
    wanted = ('hip-amdgcn-amd-amdhsa--' + arch, 'hipv4-amdgcn-amd-amdhsa--' + arch)
    cos = []
    for k, entries in enumerate(bundles):
        match = [(t, start, size) for t, start, size in entries if t in wanted and size > 0]
        if len(match) > 1:
            sys.exit(f'{exe}: offload bundle {k} of its .hip_fatbin has {len(match)} entries for {arch}')
        if not match:
            cos.append(None)
            continue
        triple, start, size = match[0]
        path = os.path.join(tmpdir, f'co{k}.o')
        with open(path, 'wb') as f:
            f.write(data[start:start + size])
        cos.append((triple, path))
    have = sum(1 for c in cos if c)
    if have == 0:
        return None
    if have < len(cos):
        sys.exit(f'{exe}: {len(cos) - have} of the {len(cos)} offload bundles of its .hip_fatbin have no {arch} code object')
    return cos


def code_object(llvm, arch, exe, tmpdir):
    """The path of the code object of --arch of an executable with one offload bundle (one HIP translation unit), or None. Exits, naming the
    executable, when its .hip_fatbin holds more than one bundle with such a code object: code_objects reads them all."""
    cos = code_objects(llvm, arch, exe, tmpdir)
    if cos and len(cos) > 1:
        sys.exit(f'{exe}: its .hip_fatbin holds {len(cos)} offload bundles (one per HIP translation unit); code_object reads only one, '
                 'use code_objects')
    return cos[0][1] if cos else None


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


def compared(llvm, ca, cb, labels_as_symbols):
    """({name: value} of A, {name: value} of B, mode) for the code objects ca and cb of one executable in A and B: two values are equal exactly
    when the function's device code is the same. One code object on each side: the hashes, by name. Otherwise the branch targets are qualified
    on both sides, and the values are the hashes per position when both sides have the same triples in the same order ("paired by position"),
    or the multisets of the hashes over all the code objects ("multisets by name")."""
    if len(ca) == 1 and len(cb) == 1:
        return functions(llvm, ca[0][1], labels_as_symbols), functions(llvm, cb[0][1], labels_as_symbols), ''
    ta = [functions(llvm, path, labels_as_symbols, qualify_labels=True) for _, path in ca]
    tb = [functions(llvm, path, labels_as_symbols, qualify_labels=True) for _, path in cb]
    if [t for t, _ in ca] == [t for t, _ in cb]:
        def paired(tables):
            out = {}
            for k, table in enumerate(tables):
                for name, h in table.items():
                    out.setdefault(name, []).append(f'{k}:{h}')
            return {name: ' '.join(v) for name, v in out.items()}
        return paired(ta), paired(tb), 'paired by position'

    def multisets(tables):
        out = {}
        for table in tables:
            for name, h in table.items():
                out.setdefault(name, collections.Counter())[h] += 1
        return {name: ' '.join(f'{h}*{c}' for h, c in sorted(cnt.items())) for name, cnt in out.items()}
    return multisets(ta), multisets(tb), 'multisets by name'


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
            fa, fb, mode = compared(args.llvm, ca, cb, args.labels_as_symbols)
            same = sum(1 for k in fa if k in fb and fa[k] == fb[k])
            diff = [k for k in fa if k in fb and fa[k] != fb[k]]
            oa, ob = [k for k in fa if k not in fb], [k for k in fb if k not in fa]
            tot['funcs'] += len(fa)
            tot['ident'] += same
            tot['diff'] += len(diff)
            tot['onlyA'] += len(oa)
            tot['onlyB'] += len(ob)
            flag = '' if not (diff or oa or ob) else '   <-- DIFFERENCES'
            objects = f' (code objects A {len(ca)}, B {len(cb)}, {mode})' if mode else ''
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
