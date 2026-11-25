# Host (x86-64) code of one executable in two builds, compared function by function, and the functions that the timed calls of its rows
# reach. Written for the E19 programs of the pin move from FKL 24478f0 to b10fc66 (results/rx9060/pin-move-devcode-24478f0-b10fc66/).
#
# usage: python3 -I host_compare.py <executable A> <executable B> --out <dir> [--objdump objdump]
#                 [--root <row>=<regex>]... [--harness <regex>]... [--rodata-bytes N [--same-path <folder of A>=<folder of B>]...]
#
# The disassembly: objdump -d -C --no-show-raw-insn -j .text (GNU objdump), one function per symbol header '<address> <name>:'.
# The text of each instruction is normalised before it is hashed:
#   - the address at the start of the line is removed;
#   - a target 'addr <symbol+off>' (calls, jumps, the comments of RIP-relative operands) is resolved against the function table of .text:
#     inside the same function it becomes a local label 'L<n>' (n = the rank of the target offset among the targets of that function), at the
#     start of another function it becomes '<name>', inside another function '<name+OFF>'; outside .text (the PLT, data) the address is
#     removed and objdump's '+0x..' offset becomes '+OFF';
#   - a RIP-relative displacement 0x..(%rip) becomes X(%rip);
#   - the names __hip_gpubin_handle_<hash> and __hip_cuid_<hash> (per translation unit, derived from the build paths) become
#     __hip_gpubin_handle_@ and __hip_cuid_@;
#   - the padding after the last instruction (nop forms, int3, '...') is not compared.
#   With --rodata-bytes N (the stricter form) a RIP-relative operand inside a .rodata section is written as '<section>=<what is there>'
#   instead of '<nearest symbol+OFF>' (rodata_value: the entries of a jump table as offsets in the function, a C string, the bytes of a
#   load's operand size, or N bytes for any other address): the constants are compared, and the nearest symbol (which moves with the layout)
#   is not. --same-path replaces a folder of build A by the folder of build B in A's strings (the source paths of __FILE__).
# Functions are paired by demangled name; a name defined more than once gets '#<k>' in address order.
# The call graph: a direct call or jump to another .text function is an edge; calls through the PLT and indirect calls are listed, not
# followed. For each --root the closure of its edges is computed, and for the union of the --harness functions; the closures of A and B are
# both reported. A function in the closure of more than one root is 'shared'.
# Exit 0 always; the summary says IDENTICAL or lists the differences.
import argparse
import collections
import hashlib
import os
import re
import subprocess
import sys

HEADER = re.compile(r'^([0-9a-f]+) <(.*)>:$')
INSN = re.compile(r'^\s*([0-9a-f]+):\t(.*)$')
TARGET = re.compile(r'(?:(#\s*))?\b([0-9a-f]+) <(.*)>$')   # the last 'addr <symbol>' of a line: call/jmp operand or the # comment
RIPDISP = re.compile(r'-?0x[0-9a-f]+\(%rip\)')
OFFSET = re.compile(r'\+0x[0-9a-f]+$')
PADDING = re.compile(r'^(nop[a-z]*\b.*|(data16 |cs |ds )+nop[a-z]*\b.*|int3|xchg\s+%ax,%ax|\.\.\.)\s*$')
HIPHASH = re.compile(r'__hip_(gpubin_handle|cuid)_[0-9a-f]+')   # per translation unit names derived from the build paths
BRANCH = re.compile(r'^(call|jmp|j[a-z]{1,3})[a-z]*\s')


def disassemble(path, objdump):
    out = subprocess.run([objdump, '-d', '-C', '--no-show-raw-insn', '-j', '.text', path], check=True, capture_output=True, text=True).stdout
    funcs = []   # [addr, name, [(addr, text)]]
    for line in out.splitlines():
        m = HEADER.match(line)
        if m:
            funcs.append([int(m.group(1), 16), m.group(2), []])
            continue
        m = INSN.match(line)
        if m and funcs:
            funcs[-1][2].append((int(m.group(1), 16), m.group(2).rstrip()))
        elif line.strip() == '...' and funcs:
            funcs[-1][2].append((None, '...'))
    return funcs


def rodata(path):
    """[(start, end, name, bytes)] of the sections whose name starts with .rodata, from the file."""
    secs = []
    for line in subprocess.run(['readelf', '-SW', path], check=True, capture_output=True, text=True).stdout.splitlines():
        m = re.match(r'^\s*\[\s*\d+\]\s+(\S+)\s+(\S+)\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)', line)
        if m and m.group(1).startswith('.rodata') and m.group(2) != 'NOBITS':
            secs.append((m.group(1), int(m.group(3), 16), int(m.group(4), 16), int(m.group(5), 16)))
    with open(path, 'rb') as f:
        data = f.read()
    sized = {}   # the data symbols with a size: address -> size
    for line in subprocess.run(['nm', '-S', '--defined-only', path], check=True, capture_output=True, text=True).stdout.splitlines():
        m = re.match(r'^([0-9a-f]+) ([0-9a-f]+) [rRvVdD] ', line)
        if m and int(m.group(2), 16) > 0:
            sized[int(m.group(1), 16)] = int(m.group(2), 16)
    return [(a, a + size, n, data[off:off + size], sized) for n, a, off, size in secs]


SCALAR4 = {'movss', 'movd', 'addss', 'subss', 'mulss', 'divss', 'minss', 'maxss', 'ucomiss', 'comiss', 'cvtss2sd', 'sqrtss'}
SCALAR8 = {'movsd', 'movq', 'addsd', 'subsd', 'mulsd', 'divsd', 'minsd', 'maxsd', 'ucomisd', 'comisd', 'cvtsd2ss', 'sqrtsd', 'movhps', 'movlps',
           'movhpd', 'movlpd', 'movddup'}


def rodata_value(text, sec, target, faddr, insns, n, paths=()):
    """What an instruction takes from a .rodata section, for --rodata-bytes:
    - lea of a data symbol with a size (nm -S): 'o:' and its bytes;
    - lea of a jump table (the int32 entries, relative to the table, land in the function): 'jt:' and the entries as offsets in the function;
    - lea of a C string (printable bytes up to a NUL): 's:' and the string in hex;
    - any other lea: 'b:' and n bytes (the size of the object is not known);
    - a load: 'b:' and the bytes of its operand size (16 for a packed xmm operand, 8 or 4 for a scalar one, else by the suffix)."""
    data, at = sec[3], target - sec[0]
    mnem = text.split()[0]
    if mnem == 'lea' and target in sec[4]:   # a data symbol with a size: its bytes
        return 'o:' + data[at:at + sec[4][target]].hex()
    if mnem == 'lea':
        fend = max(i[0] for i in insns if i[0] is not None)
        entries = []
        for j in range(256):
            if at + 4 * j + 4 > len(data):
                break
            e = int.from_bytes(data[at + 4 * j:at + 4 * j + 4], 'little', signed=True)
            dest = target + e
            if not faddr <= dest <= fend:
                break
            entries.append(dest - faddr)
        if entries:
            return 'jt:' + ','.join(f'+0x{x:x}' for x in entries)
        end = data.find(b'\0', at)
        if end > at and all(32 <= c < 127 or c in (9, 10, 13) for c in data[at:end]):
            string = data[at:end]
            for old, new in paths:   # --same-path: the folder of build A read as the folder of build B
                string = string.replace(old, new)
            return 's:' + string.hex()
        return 'b:' + data[at:at + n].hex()
    if '%xmm' in text or '%ymm' in text:
        size = 4 if mnem in SCALAR4 else 8 if mnem in SCALAR8 else 16
    else:
        size = {'q': 8, 'l': 4, 'w': 2, 'b': 1}.get(mnem[-1], 8)
    return 'b:' + data[at:at + size].hex()


def analyse(path, objdump, rodata_bytes=0, paths=()):
    funcs = disassemble(path, objdump)
    ro = rodata(path) if rodata_bytes else []
    starts = sorted(f[0] for f in funcs)
    text_end = max((i[0] for f in funcs for i in f[2] if i[0] is not None), default=0) + 16
    by_start = {f[0]: f for f in funcs}
    counts = collections.Counter()
    for f in funcs:
        counts[f[1]] += 1
        key = f[1] if counts[f[1]] == 1 else f'{f[1]}#{counts[f[1]]}'
        f.append(key)
    total = collections.Counter(f[1] for f in funcs)
    for f in funcs:   # the first definition of a name defined more than once gets '#1'
        if total[f[1]] > 1 and f[3] == f[1]:
            f[3] = f'{f[1]}#1'
    import bisect

    def owner(addr):
        i = bisect.bisect_right(starts, addr) - 1
        if i < 0 or addr >= text_end:
            return None
        return by_start[starts[i]]

    result = {}
    for addr, name, insns, key in funcs:
        body = list(insns)
        while body and PADDING.match(body[-1][1]):
            body.pop()
        local = sorted({int(m.group(2), 16) - addr for _, t in body for m in [TARGET.search(t)] if m and owner(int(m.group(2), 16)) is not None
                        and owner(int(m.group(2), 16))[0] == addr})
        rank = {off: n for n, off in enumerate(local)}
        lines, callees, plt, indirect = [], [], [], []
        for _, t in body:
            m = TARGET.search(t)
            if m:
                target = int(m.group(2), 16)
                f = owner(target)
                head = t[:m.start()]
                hashmark = m.group(1) or ''
                if f is not None and f[0] == addr:
                    t = f'{head}{hashmark}L{rank[target - addr]}'
                elif f is not None:
                    t = f'{head}{hashmark}<{f[3]}>' if target == f[0] else f'{head}{hashmark}<{f[3]}+OFF>'
                    if BRANCH.match(t) and not hashmark:
                        callees.append(f[3])
                else:
                    sec = next((r for r in ro if r[0] <= target < r[1]), None)
                    sym = OFFSET.sub('+OFF', m.group(3))
                    if hashmark and sec is not None:   # --rodata-bytes: the section and what is there, not the nearest symbol
                        t = f'{head}{hashmark}<{sec[2]}>={rodata_value(t, sec, target, addr, insns, rodata_bytes, paths)}'
                    else:
                        t = f'{head}{hashmark}<{sym}>'
                    if BRANCH.match(t) and sym.endswith('@plt'):
                        plt.append(sym)
            if re.match(r'^(call|jmp)\s+\*', t):
                indirect.append(t)
            t = RIPDISP.sub('X(%rip)', t)
            t = HIPHASH.sub(r'__hip_\1_@', t)
            t = re.sub(r'\s+', ' ', t)
            lines.append(t)
        result[key] = {'name': name, 'addr': addr, 'lines': lines, 'sha': hashlib.sha256('\n'.join(lines).encode()).hexdigest(),
                       'n': len(lines), 'callees': callees, 'plt': plt, 'indirect': indirect}
    return result


_RELOCS = {}


def relative_refs(path, addr):
    """The R_X86_64_RELATIVE relocations of an executable whose addend is addr, as '<section>+0x<offset> (<data symbol>+0x<offset>)'."""
    if path not in _RELOCS:
        secs = []
        for line in subprocess.run(['readelf', '-SW', path], check=True, capture_output=True, text=True).stdout.splitlines():
            m = re.match(r'^\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+([0-9a-f]+)\s+[0-9a-f]+\s+([0-9a-f]+)', line)
            if m:
                secs.append((int(m.group(2), 16), int(m.group(3), 16), m.group(1)))
        syms = []
        for line in subprocess.run(['nm', '-C', '--defined-only', path], check=True, capture_output=True, text=True).stdout.splitlines():
            m = re.match(r'^([0-9a-f]+) ([dDbBrRvV]) (.*)$', line)
            if m:
                syms.append((int(m.group(1), 16), m.group(3)))
        syms.sort()
        rel = collections.defaultdict(list)
        for line in subprocess.run(['readelf', '-rW', path], check=True, capture_output=True, text=True).stdout.splitlines():
            m = re.match(r'^([0-9a-f]+)\s+[0-9a-f]+\s+R_X86_64_RELATIVE\s+([0-9a-f]+)\s*$', line)
            if not m:
                continue
            off, add = int(m.group(1), 16), int(m.group(2), 16)
            sec = next((f'{n}+0x{off - a:x}' for a, sz, n in secs if a <= off < a + sz), f'0x{off:x}')
            prev = [sy for sy in syms if sy[0] <= off]
            near = f' ({prev[-1][1]}+0x{off - prev[-1][0]:x})' if prev else ''
            rel[add].append(sec + near)
        _RELOCS[path] = rel
    return _RELOCS[path].get(addr, [])


def closure(funcs, roots):
    seen, todo = set(), list(roots)
    while todo:
        k = todo.pop()
        if k in seen or k not in funcs:
            continue
        seen.add(k)
        todo.extend(funcs[k]['callees'])
    return seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--out', required=True)
    ap.add_argument('--objdump', default='objdump')
    ap.add_argument('--root', action='append', default=[])
    ap.add_argument('--harness', action='append', default=[])
    ap.add_argument('--rodata-bytes', type=int, default=0)
    ap.add_argument('--same-path', action='append', default=[])
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    prog = os.path.basename(args.b)
    paths = [tuple(x.encode() for x in p.split('=', 1)) for p in args.same_path]
    A, B = analyse(args.a, args.objdump, args.rodata_bytes, paths), analyse(args.b, args.objdump, args.rodata_bytes)
    keys = sorted(set(A) | set(B))
    diff = [k for k in keys if k in A and k in B and A[k]['sha'] != B[k]['sha']]
    onlyA = [k for k in keys if k not in B]
    onlyB = [k for k in keys if k not in A]
    same = [k for k in keys if k in A and k in B and A[k]['sha'] == B[k]['sha']]
    with open(os.path.join(args.out, f'{prog}.functions.tsv'), 'w') as f:
        f.write('function\tinstructions_A\tinstructions_B\tsha256_A\tsha256_B\tsame\n')
        for k in keys:
            a, b = A.get(k), B.get(k)
            f.write(f"{k}\t{a['n'] if a else ''}\t{b['n'] if b else ''}\t{a['sha'] if a else ''}\t{b['sha'] if b else ''}\t"
                    f"{'yes' if a and b and a['sha'] == b['sha'] else 'no'}\n")
    rep = []
    rep.append(f'{prog}: host functions in .text A {len(A)}, B {len(B)}; identical {len(same)}, different {len(diff)}, only A {len(onlyA)}, '
               f'only B {len(onlyB)}')
    for label, ks in (('different', diff), ('only in A', onlyA), ('only in B', onlyB)):
        for k in ks:
            a, b = A.get(k), B.get(k)
            rep.append(f"  {label}: {k}  (instructions A {a['n'] if a else '-'}, B {b['n'] if b else '-'})")
    changed = set(diff) | set(onlyA) | set(onlyB)
    # the rows and the harness
    roots = []
    for r in args.root:
        row, rx = r.split('=', 1)
        roots.append((row, re.compile(rx)))
    harness = [re.compile(h) for h in args.harness]
    clos = {}
    for side, F in (('A', A), ('B', B)):
        for row, rx in roots:
            rs = [k for k in F if rx.search(k)]
            clos[(side, row)] = (rs, closure(F, rs))
        hs = [k for k in F if any(h.search(k) for h in harness)]
        clos[(side, 'harness')] = (hs, closure(F, hs))
    names = [r for r, _ in roots] + ['harness']
    for row in names:
        for side, F in (('A', A), ('B', B)):
            rs, c = clos[(side, row)]
            shared = sorted(k for k in c if any(k in clos[(side, o)][1] for o in names if o != row))
            plt = sorted({p for k in c for p in F[k]['plt']})
            ind = sum(len(F[k]['indirect']) for k in c)
            hit = sorted(c & changed)
            rep.append(f'  {row} [{side}]: roots {len(rs)}, functions reached {len(c)}, shared with another row or the harness {len(shared)}, '
                       f'PLT targets {len(plt)}, indirect calls/jumps {ind}; changed functions reached: {len(hit)}')
            for k in rs:
                rep.append(f'      root: {k}')
            for k in hit:
                rep.append(f'      CHANGED: {k}')
            with open(os.path.join(args.out, f'{prog}.reach-{row}-{side}.txt'), 'w') as f:
                f.write(f'# {prog} build {side}: the .text functions reached from the {row} roots by direct calls and jumps\n')
                for k in rs:
                    f.write(f'root\t{k}\n')
                for k in sorted(c):
                    f.write(f"function\t{'same' if k not in changed else 'CHANGED'}\t{'shared' if k in shared else 'own'}\t{k}\n")
                for p in plt:
                    f.write(f'plt\t{p}\n')
                for k in sorted(c):
                    for t in F[k]['indirect']:
                        f.write(f'indirect\t{k}\t{t}\n')
    # who refers to a changed function: in .text (calls, jumps, address loads), and in data (an R_X86_64_RELATIVE relocation whose addend is
    # the function's start: a function pointer in a table, a vtable or a handle)
    for k in sorted(changed):
        for side, F, path in (('A', A, args.a), ('B', B, args.b)):
            if k not in F:
                continue
            refs = sorted({o for o, v in F.items() if o != k and any(f'<{k}>' in t or f'<{k}+OFF>' in t for t in v['lines'])})
            data = relative_refs(path, F[k]['addr'])
            rep.append(f'  referred to in {side} by {len(refs)} functions and {len(data)} data relocations: {k}')
            for o in refs:
                rep.append(f'      {o}')
            for d in data:
                rep.append(f'      data: {d}')
    import difflib
    with open(os.path.join(args.out, f'{prog}.diffs.txt'), 'w') as f:
        for k in diff:
            f.write(f'=== {k}\n')
            f.writelines(l + '\n' for l in difflib.unified_diff(A[k]['lines'], B[k]['lines'], 'A', 'B', n=2, lineterm=''))
    rep.append(f'{prog}: ' + ('HOST CODE IDENTICAL' if not changed else f'{len(changed)} FUNCTIONS DIFFER'))
    text = '\n'.join(rep) + '\n'
    sys.stdout.write(text)
    with open(os.path.join(args.out, f'{prog}.summary.txt'), 'w') as f:
        f.write(text)


if __name__ == '__main__':
    main()
