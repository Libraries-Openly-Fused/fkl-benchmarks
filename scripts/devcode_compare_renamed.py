# Device code of the HIP executables of two build trees, compared function by function, with the functions PAIRED BY A NORMALISED NAME.
#
# scripts/devcode_compare.py as of 8f4a95b (called "the 8f4a95b script" below), changed for the pin move from FKL 64ad9195 to 1da77899,
# the merge of FKL #431, whose arithmetic.h gives fk::Sub, fk::Mul and fk::Div a 4th template parameter IT = BinaryType: every kernel with
# one of them in its name gets a new mangled name. Results: results/rx9060/pin-move-devcode-64ad9195-1da77899/. The changes:
#   - the key of a function is its name demangled (llvm-cxxfilt of --llvm), with all whitespace removed and ",fk::BinaryType" removed where it
#     is the 4th top-level template argument of fk::Sub, fk::Mul or fk::Div (drop_default below: the NVIDIA session's rule from the
#     sass_identity.py of its CUDA check of #431, copied verbatim); nothing else in the name changes. The functions of A and B are paired by that key.
#   - a mangled name inside the instruction text (after the comments are removed) is replaced by its key in the same way before the text is
#     hashed; the lines that held one are counted.
#   - key collisions (two different names of one code object, or of one side of a multi-bundle executable, with the same key) are counted and
#     make the result DIFFERENCES.
# The rest is the 8f4a95b script: the code objects, the disassembly, the hashing, and the line per program, which is the same byte for byte
# when no name changes. Each such line of an executable with device code is followed by a 'names:' line, and the TOTAL line by a
# 'NAMES TOTAL:' line. --no-normalize uses the raw names as the keys (the 8f4a95b pairing). The --table file has the columns key, name in A,
# name in B, sha256 in A, sha256 in B, same; with several code objects a hash cell is "<position>:<sha256> ..." (paired by position) or
# "<sha256>*<count> ..." (multisets).
#
# usage: python3 -I scripts/devcode_compare_renamed.py <bin dir A> <bin dir B> [--glob '*_hip'] [--only name1,name2] [--arch gfx1200]
#                [--llvm <dir of llvm-objcopy, llvm-objdump, llvm-cxxfilt, llvm-nm>] [--table <dir>] [--labels-as-symbols] [--no-normalize]
#
# From the 8f4a95b script:
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
#   With one code object on each side, the functions are compared by name (here: by key), as they always were. With more than one on a side:
#   - when both sides have the same number of code objects, with the same triples, they are paired by position (the order of the bundles is
#     the link order) and each function is compared code object by code object: the same name must have the same instruction text in the same
#     code object;
#   - otherwise each function is compared by the multiset of the hashes of its definitions over all the code objects (a name such as the
#     runtime stubs that every translation unit holds then also differs when the number of its definitions differs);
#   and with --labels-as-symbols the branch targets of both sides are named <function>:L<n>, since their numbers are per code object. The line
#   of such an executable ends with the numbers of code objects and the mode.
# Exit 0 = the same executables in both directories, every function of every executable identical, and no key collision.
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
MANGLED = re.compile(r'(?<![A-Za-z0-9_.$])_Z[A-Za-z0-9_.$]+')   # an Itanium mangled name inside the instruction text
WHITESPACE = re.compile(r'\s+')

# ---- the NVIDIA session's normalisation, verbatim from the sass_identity.py of its CUDA check of #431 ----
TEMPLATES = ("fk::Sub<", "fk::Mul<", "fk::Div<")
OPEN, CLOSE = "<({[", ">)}]"
def top_level_args(s, start):
    depth, begin, spans, i = 0, start, [], start
    while i < len(s):
        c = s[i]
        if c in OPEN: depth += 1
        elif c in CLOSE:
            if depth == 0:
                spans.append((begin, i)); return spans, i
            depth -= 1
        elif c == "," and depth == 0:
            spans.append((begin, i)); begin = i + 1
        i += 1
    raise ValueError("unbalanced template argument list")
def drop_default(name):   # name = demangled, whitespace removed
    removed, i = 0, 0
    while True:
        hits = [(name.find(t, i), t) for t in TEMPLATES]
        hits = [(p, t) for p, t in hits if p >= 0 and (p == 0 or not (name[p-1].isalnum() or name[p-1] == "_"))]
        if not hits: return name, removed
        p, t = min(hits); start = p + len(t)
        spans, close = top_level_args(name, start)
        if len(spans) == 4 and name[spans[3][0]:spans[3][1]] == "fk::BinaryType":
            name = name[:spans[2][1]] + name[spans[3][1]:]; removed += 1
        i = start
# ---- end of the NVIDIA session's normalisation ----


def key_of(demangled_name):
    """(key, number of ',fk::BinaryType' removed): the demangled name without whitespace, through drop_default."""
    return drop_default(WHITESPACE.sub('', demangled_name))


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


def digest(lines):
    """sha256 of the instruction text of a symbol."""
    return hashlib.sha256('\n'.join(lines).encode()).hexdigest()


def demangled(llvm, names):
    """{name: its demangled form} by one llvm-cxxfilt call (a name that is not a mangled name comes back unchanged)."""
    names = sorted(set(names))
    if not names:
        return {}
    r = subprocess.run([llvm + '/llvm-cxxfilt'], input=''.join(n + '\n' for n in names), capture_output=True, text=True)
    out = r.stdout.split('\n')
    if out and out[-1] == '':
        out.pop()
    if r.returncode != 0 or len(out) != len(names):
        sys.exit(f'llvm-cxxfilt: exit {r.returncode}, {len(out)} lines for {len(names)} names: {r.stderr.strip()[:300]}')
    return dict(zip(names, out))


def kernel_names(llvm, co):
    """The names of the kernels of a code object: the symbols <name>.kd (kernel descriptors), with __hip_cuid_<hash> normalised."""
    r = subprocess.run([llvm + '/llvm-nm', '--defined-only', co], capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f'llvm-nm {co}: exit {r.returncode}: {r.stderr.strip()[:300]}')
    out = set()
    for line in r.stdout.splitlines():
        f = line.split()
        if f and f[-1].endswith('.kd'):
            out.add(CUID.sub('__hip_cuid_@', f[-1][:-3]))
    return out


class Table:
    """The symbols of one code object, or of several merged: hashes {key: value}; per key of a function the set of its raw names (raw); per key
    the key that the 8f4a95b script would have used (old), for the example lines; the keys of the kernels; the counts of the naming; and the
    key collisions {key: set of raw names}."""
    def __init__(self):
        self.hashes, self.raw, self.old, self.kernels = {}, {}, {}, set()
        self.c = collections.Counter()
        self.collisions = {}


def table_of(groups, dem, kernels, labels_as_symbols, qualify_labels, normalize):
    """The Table of the symbol groups of one code object ([[name, lines], ...] per function: the function first, then its branch targets),
    with dem = {name: demangled name} for the function names and the mangled names in the text, and kernels = the raw names of the kernels."""
    t = Table()
    keys = {}
    for group in groups:
        raw = group[0][0]
        if not normalize:
            keys[raw] = raw
            continue
        d = dem[raw]
        if raw.startswith('_Z') and d == raw:
            t.c['not_demangled'] += 1
        keys[raw], removed = key_of(d)
        t.c['removed'] += removed
        t.c['names_with_removal'] += 1 if removed else 0

    def text(code):
        found = MANGLED.findall(code) if normalize else []
        if found:
            t.c['text_lines_mangled'] += 1
            hit = [0]

            def sub(m):
                d = dem.get(m.group(0), m.group(0))
                if d == m.group(0):
                    return d
                k, removed = key_of(d)
                hit[0] += removed
                return k
            code = MANGLED.sub(sub, code)
            if hit[0]:
                t.c['text_lines_removal'] += 1
                t.c['text_removed'] += hit[0]
        return code if labels_as_symbols else LABEL.sub('L@', code)

    def put(key, old, value, raw=None):
        if raw is not None:
            names = t.raw.setdefault(key, set())
            if names and raw not in names:
                t.collisions.setdefault(key, set(names)).add(raw)
            elif raw in names:
                t.c['duplicate_names'] += 1
            names.add(raw)
            if raw in kernels:
                t.kernels.add(key)
        t.hashes[key] = value          # (the last one wins, as in the 8f4a95b script)
        t.old.setdefault(key, old)

    for group in groups:
        last = group[-1][1]
        while last and last[-1] == '...':   # the padding after the last instruction of the function
            last.pop()
        raw = group[0][0]
        fkey = keys[raw]
        lines_of = [[text(code) for code in lines] for _, lines in group]
        if labels_as_symbols:
            for k, (name, _) in enumerate(group):
                if k == 0:
                    put(fkey, raw, digest(lines_of[0]), raw)
                else:
                    put(f'{fkey}:{name}' if qualify_labels else name, f'{raw}:{name}' if qualify_labels else name, digest(lines_of[k]))
        else:
            put(fkey, raw, digest([code for lines in lines_of for code in lines]), raw)
    return t


def functions(llvm, co, labels_as_symbols, qualify_labels=False, normalize=True):
    """The Table of a code object.

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
            groups[-1][-1][1].append(CUID.sub('__hip_cuid_@', code))
    p.wait()
    dem = {}
    if normalize:
        tokens = {tok for group in groups for _, lines in group for code in lines for tok in MANGLED.findall(code)}
        dem = demangled(llvm, [group[0][0] for group in groups] + sorted(tokens))
    return table_of(groups, dem, kernel_names(llvm, co), labels_as_symbols, qualify_labels, normalize)


def merged(tables):
    """One Table of several code objects of one side: raw names, kernels, counts and collisions together (hashes and old keys of the first
    definition); a key that stands for different raw names in different code objects is a collision too."""
    m = Table()
    for t in tables:
        for key, names in t.raw.items():
            have = m.raw.setdefault(key, set())
            if have and not names <= have:
                m.collisions.setdefault(key, set(have)).update(names)
            have.update(names)
        for key, value in t.hashes.items():
            m.hashes.setdefault(key, value)
        for key, old in t.old.items():
            m.old.setdefault(key, old)
        m.kernels |= t.kernels
        m.c.update(t.c)
        for key, names in t.collisions.items():
            m.collisions.setdefault(key, set()).update(names)
    return m


def compared(llvm, ca, cb, labels_as_symbols, normalize):
    """({key: value} of A, {key: value} of B, mode, Table of A, Table of B) for the code objects ca and cb of one executable in A and B: two
    values are equal exactly when the function's device code is the same. One code object on each side: the hashes, by key. Otherwise the branch
    targets are qualified on both sides, and the values are the hashes per position when both sides have the same triples in the same order
    ("paired by position"), or the multisets of the hashes over all the code objects ("multisets by name")."""
    if len(ca) == 1 and len(cb) == 1:
        ta = functions(llvm, ca[0][1], labels_as_symbols, normalize=normalize)
        tb = functions(llvm, cb[0][1], labels_as_symbols, normalize=normalize)
        return ta.hashes, tb.hashes, '', ta, tb
    ta = [functions(llvm, path, labels_as_symbols, qualify_labels=True, normalize=normalize) for _, path in ca]
    tb = [functions(llvm, path, labels_as_symbols, qualify_labels=True, normalize=normalize) for _, path in cb]
    if [t for t, _ in ca] == [t for t, _ in cb]:
        def paired(tables):
            out = {}
            for k, table in enumerate(tables):
                for name, h in table.hashes.items():
                    out.setdefault(name, []).append(f'{k}:{h}')
            return {name: ' '.join(v) for name, v in out.items()}
        return paired(ta), paired(tb), 'paired by position', merged(ta), merged(tb)

    def multisets(tables):
        out = {}
        for table in tables:
            for name, h in table.hashes.items():
                out.setdefault(name, collections.Counter())[h] += 1
        return {name: ' '.join(f'{h}*{c}' for h, c in sorted(cnt.items())) for name, cnt in out.items()}
    return multisets(ta), multisets(tb), 'multisets by name', merged(ta), merged(tb)


def names_line(prefix, pairs, renamed, ia, ib):
    return (f"{prefix}pairs {pairs}, renamed {renamed}; kernels A {ia['kernels']}, B {ib['kernels']}; ',fk::BinaryType' removed from the "
            f"names A {ia['removed']}, B {ib['removed']} (names with a removal A {ia['names_with_removal']}, B {ib['names_with_removal']}); "
            f"key collisions A {ia['collisions']}, B {ib['collisions']}; not demangled A {ia['not_demangled']}, B {ib['not_demangled']}; "
            f"instruction lines with a mangled name A {ia['text_lines_mangled']}, B {ib['text_lines_mangled']} (with a removal "
            f"A {ia['text_lines_removal']}, B {ib['text_lines_removal']})")


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
    ap.add_argument('--no-normalize', action='store_true')
    args = ap.parse_args()
    normalize = not args.no_normalize
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
    ntot = dict(pairs=0, renamed=0)
    sides = {s: collections.Counter() for s in 'AB'}
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
            fa, fb, mode, ta, tb = compared(args.llvm, ca, cb, args.labels_as_symbols, normalize)
            same = sum(1 for k in fa if k in fb and fa[k] == fb[k])
            diff = [k for k in fa if k in fb and fa[k] != fb[k]]
            oa, ob = [k for k in fa if k not in fb], [k for k in fb if k not in fa]
            pairs = sum(1 for k in fa if k in fb)
            renamed = sum(1 for k in fa if k in fb and k in ta.raw and k in tb.raw and ta.raw[k] != tb.raw[k])
            tot['funcs'] += len(fa)
            tot['ident'] += same
            tot['diff'] += len(diff)
            tot['onlyA'] += len(oa)
            tot['onlyB'] += len(ob)
            ntot['pairs'] += pairs
            ntot['renamed'] += renamed
            stats = {}
            for s, t in (('A', ta), ('B', tb)):
                stats[s] = dict(t.c, kernels=len(t.kernels), collisions=len(t.collisions))
                stats[s] = collections.Counter(stats[s])
                sides[s].update(stats[s])
            flag = '' if not (diff or oa or ob) else '   <-- DIFFERENCES'
            objects = f' (code objects A {len(ca)}, B {len(cb)}, {mode})' if mode else ''
            print(f'  {n}: functions A {len(fa)}, B {len(fb)}, identical {same}, different {len(diff)}, only A {len(oa)}, '
                  f'only B {len(ob)}{objects}{flag}')
            if diff or oa or ob:
                bad.append(n)
                for k in (diff + oa + ob)[:3]:
                    print('       ', (ta.old.get(k) or tb.old.get(k) or k)[:140])
            print(names_line('      names: ', pairs, renamed, stats['A'], stats['B']) +
                  ('   <-- KEY COLLISIONS' if ta.collisions or tb.collisions else ''))
            for s, t in (('A', ta), ('B', tb)):
                for key, raws in sorted(t.collisions.items())[:3]:
                    print(f'        collision in {s}: {len(raws)} names for the key {key[:140]}')
            if ta.collisions or tb.collisions:
                bad.append(n)
            if args.table:
                with open(os.path.join(args.table, n + '.tsv'), 'w') as t:
                    t.write('key\tname_A\tname_B\tsha256_A\tsha256_B\tsame\n')
                    for k in sorted(set(fa) | set(fb)):
                        ha, hb = fa.get(k, '-'), fb.get(k, '-')
                        ra = ' '.join(sorted(ta.raw.get(k, ()))) or ('-' if k not in fa else '')
                        rb = ' '.join(sorted(tb.raw.get(k, ()))) or ('-' if k not in fb else '')
                        t.write(f'{k}\t{ra}\t{rb}\t{ha}\t{hb}\t{"yes" if ha == hb else "no"}\n')
    print(f"TOTAL: {len(common)} binaries, {tot['funcs']} functions in A: identical {tot['ident']}, different {tot['diff']}, "
          f"only in A {tot['onlyA']}, only in B {tot['onlyB']}, code object in only one of them {tot['nocode']}; "
          f"{tot['nodev']} binaries have no device code in either")
    print(names_line('NAMES TOTAL: ', ntot['pairs'], ntot['renamed'], sides['A'], sides['B']))
    ok = not bad and not tot['nocode'] and set(names_a) == set(names_b)
    print('DEVICE CODE ' + ('IDENTICAL' if ok else 'DIFFERENCES'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
