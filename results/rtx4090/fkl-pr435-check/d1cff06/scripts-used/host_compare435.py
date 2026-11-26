#!/usr/bin/env python3
"""C-PM-CUDA: the host code of one program in two builds, base (fkl-benchmarks 976e42b, FKL b10fc66) against head (the pin move to FKL #435,
48fbf3f), function by function (CPU only: the binaries are only read with objdump; nothing is run).

  host_compare435.py <binary A> <binary B> --tools <dir of pr435_sass_identity.py> --sass-tools <dir of sass_tools.py> [--roots REGEX ...] [--out report.txt] [--csv functions.csv]
                     [--diffs diffs.txt]

Disassembly and normalisation: those of results/rtx4090/fkl-pr433-pin-check/scripts-used/host_compare.py (objdump -d -C --no-show-raw-insn -w
-j .text; addresses dropped; padding removed; targets inside a function as instruction indices, into another .text function as that function's
key and index; RIP displacements masked; nvcc's anonymous-namespace module ids masked), and two more, the ones the HIP side used:
  - the kernels' host stubs (the host functions named after a kernel, and their clones) are keyed by their kernel instantiation: the
    template part of the name, A's rewritten by FKL #435's rename rule (pr435_sass_identity.rewrite, as the device code was paired), then
    "(PARAMS)" and the clone suffix (a stub's parameter list follows from its template arguments; its const qualifiers differ from the
    kernel's). The static locals of the inlined __device_stub__ functions and their guard variables (_ZZL<len>__device_stub__<mangled
    kernel>... and _ZGVZL..., the kernel handles; objdump does not demangle them) are keyed by the same instantiation. Every other name of
    A gets the attention DPPs' leading ParArch (pr435_sass_identity.insert_pa);
  - the __LINE__ arguments of the error-message functions: the immediate moved into the register of the int line parameter by the last
    such instruction before a call is masked, in both builds: fklb::statusMessage (common/status_check.h; %r8d, its 4th parameter after the
    returned string's pointer) and fk::gpuAssert (FKL's gpuErrchk(ans) = gpuAssert(ans, __FILE__, __LINE__); %edx). A port moves the
    lines of a program, and FKL #435 moves those of executors.h.
Pairing: by the key (the demangled, masked and, for A, rewritten name, whitespace removed); a key found twice gets #2, #3, ... in address order.
Roots: every function whose demangled name matches one of the --roots regular expressions (default: lambdas, std::function invokers and
fklb::time*): the timed calls. For every function that differs or is unpaired, the roots that reach it through direct calls and jumps and
through functions whose address is taken (a superset); indirect calls are not followed.
--no-rename: the rename rule is not applied (two builds whose FKL has the same kernel names, e.g. a measured binary against main).
--path-delta N: an immediate larger by exactly N in B is a build-location constant (the length of a __FILE__ path, or a bound from it).
--drop-binarytype-default: FKL #432's defaulted 4th template argument fk::BinaryType of fk::Sub, fk::Mul and fk::Div is dropped from the
names of both builds (FKL before 24478f0 against after).
--intervals: the timed intervals, see the option's help.
Exit 0 when every function pairs and is identical.
"""
import argparse
import bisect
import collections
import csv
import difflib
import hashlib
import re
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--tools", required=True)
ap.add_argument("--sass-tools", required=True, help="the folder of sass_tools.py, which pr435_sass_identity.py imports")
ap.add_argument("--roots", nargs="*", default=[r"\{lambda", r"_Function_handler<", r"^void fklb::time", r"fklb::time\w*<", r"timeCall<"])
ap.add_argument("--no-rename", action="store_true", help="do not apply the rename rule (two builds with the same FKL kernel names)")
ap.add_argument("--drop-binarytype-default", action="store_true",
                help="drop ',fk::BinaryType' where it is the 4th template argument of fk::Sub, fk::Mul or fk::Div, in both builds (FKL #432's "
                     "defaulted parameter, as results/rtx4090/fkl-pr432-check/ paired the device code)")
ap.add_argument("--out"); ap.add_argument("--csv"); ap.add_argument("--diffs")
ap.add_argument("--path-delta", type=int, default=None, help="B's build path is this many characters longer than A's: an immediate that "
                "is larger by exactly this much in B (the length of a __FILE__ string, or a bound derived from it) is a build-location constant")
ap.add_argument("--row", action="append", default=[], metavar="NAME=REGEX",
                help="a row of the program: a timed-call lambda belongs to NAME when the functions and the external calls it reaches match REGEX "
                     "(and no other row's); the report lists per row the functions its lambdas and the harness reach, and those that differ")
ap.add_argument("--row-priority", action="store_true", help="a lambda belongs to the FIRST row (in --row order) that matches, not to the only one")
ap.add_argument("--harness", default=r"^void fklb::time|fklb::time\w*<|timeCall<", help="the timing harness: functions whose name matches")
ap.add_argument("--intervals", action="store_true", help="compare the timed intervals: in every paired function that calls cudaEventRecord "
                "or std::chrono steady_clock::now at least twice, the instructions between consecutive such calls, and the functions they call")
ap.add_argument("--intervals-csv", help="the timed intervals, one row per function")
ap.add_argument("--line-pairs", default="", help="A:B,... source lines that moved (A's line, B's line): a function whose only differences are "
                "these lines' encodings is reported as differing only in __LINE__")
args = ap.parse_args()
sys.argv = [sys.argv[0], "--tools", args.sass_tools, "--a", "x=/nonexistent", "--b", "y=/nonexistent"]   # pr435_sass_identity parses its arguments when it is loaded
src = open(f"{args.tools}/pr435_sass_identity.py").read()
src = src.split("\ntag_a, dir_a = ")[0]   # the definitions only (rewrite, insert_pa, split_args), not the comparison it runs
psi = {}
exec(compile(src, "pr435_sass_identity.py", "exec"), psi)
rewrite = psi["rewrite"]

ANON = re.compile(r"_GLOBAL__N__[0-9a-f]{8}_(\d+)_([A-Za-z0-9_]+?)_[0-9a-f]{8}")
HDR = re.compile(r"^([0-9a-f]+) <(.*)>:$")
INS = re.compile(r"^\s*([0-9a-f]+):\t(.*)$")
TGT = re.compile(r"(?:^|\s)([0-9a-f]+) <(.+)>$")
RIP = re.compile(r"-?0x[0-9a-f]+\(%rip\)")
PAD = re.compile(r"^(nop|xchg\s+%ax,%ax|data16|cs nop|int3|\.\.\.)")
OFF = re.compile(r"\+0x[0-9a-f]+$")
STUB = re.compile(r"__device_stub__(Z\S+?)(?=[(>+]|$)")
LINE_MOV = re.compile(r"^mov \$0x[0-9a-f]+,(%r8d|%edx)$")
LINE_ARGS = {"fklb::statusMessage": "%r8d", "fk::gpuAssert": "%edx"}   # the function, and the register of its int line parameter
_cache = {}


def demangle(m):
    if m not in _cache:
        r = subprocess.run(["c++filt"], input="_" + m, capture_output=True, text=True).stdout.strip()
        if r == "_" + m:
            r = subprocess.run(["/usr/lib/llvm-21/bin/llvm-cxxfilt"], input="_" + m, capture_output=True, text=True).stdout.strip()
        _cache[m] = r
    return _cache[m]


def mask(s):
    return ANON.sub(lambda m: f"_GLOBAL__N__H_{m.group(1)}_{m.group(2)}_H", s)


KERNELS = set(psi["RENAMED"]) | {"launchDPP_Kernel"}
LOCAL = re.compile(r"^(_ZGVZ|_ZZ|_Z)L(\d+)(__device_stub__)")


def stub_key(k, side):
    """a host function named after a kernel (nvcc's host stub, and its clones): the key is its kernel instantiation, i.e. its template part
    (A's rewritten by the rename rule), then '(PARAMS)' and the suffix (e.g. [clone.cold]); the parameter list of a stub (with its const
    qualifiers) follows from the template arguments and is not compared. None for any other name"""
    m = re.match(r"^voidfk::(\w+)<", k)
    if not m or m.group(1) not in KERNELS:
        return None
    _, close = psi["split_args"](k, m.end())
    tpl, rest = k[:close + 1], k[close + 1:]
    suffix = rest
    if rest.startswith("("):
        depth = 0
        for i, c in enumerate(rest):
            depth += c in "(<[{"
            depth -= c in ")>]}"
            if depth == 0:
                break
        suffix = rest[i + 1:]
    if side == "A" and not args.no_rename and m.group(1) != "launchDPP_Kernel":
        new = rewrite(tpl + "()")[0]
        m2 = re.match(r"^voidfk::launchDPP_Kernel<", new)
        _, c2 = psi["split_args"](new, m2.end())
        tpl = new[:c2 + 1]
    return "KERNEL:" + tpl + "(PARAMS)" + suffix


BINOPS = ("fk::Sub<", "fk::Mul<", "fk::Div<")


def drop_binarytype(k):
    """removes ',fk::BinaryType' where it is the 4th template argument of fk::Sub/Mul/Div (whitespace-free names)"""
    i = 0
    while True:
        hits = [(k.find(t, i), t) for t in BINOPS]
        hits = [(q, t) for q, t in hits if q >= 0 and (q == 0 or not (k[q - 1].isalnum() or k[q - 1] == "_"))]
        if not hits:
            return k
        q, t = min(hits)
        start = q + len(t)
        try:
            a, close = psi["split_args"](k, start)
        except ValueError:
            return k
        if len(a) == 4 and a[3] == "fk::BinaryType":
            k = k[:start] + ",".join(a[:3]) + k[close:]
        i = start


def key_of(name, side):
    k = key_of_(name, side)
    return drop_binarytype(k) if args.drop_binarytype_default else k


def key_of_(name, side):
    """the pairing key of a name: anonymous ids masked, whitespace removed; kernel stubs by their instantiation (stub_key); the static locals
    of a __device_stub__ (_ZZL<len>__device_stub__<mangled kernel>..., and their guard variables _ZGVZL...) by the kernel's instantiation;
    with the rename rule (side A), every other name with the attention DPPs' leading ParArch inserted"""
    lm = LOCAL.match(name)
    if lm:
        # _ZL<len>__device_stub__<kernel>: the static stub function itself; _ZZL / _ZGVZL: a static local of it (and its guard). The rest
        # after the <len> characters is the stub's mangled parameter list, whose substitution indices follow the kernel's name: only the
        # local's own name (after the last 'E') is kept
        n = int(lm.group(2))
        start = lm.end(2)
        ident, rest = name[start:start + n], name[start + n:]
        kern = re.sub(r"\s+", "", mask(demangle(ident[len("__device_stub__"):])))
        sk = stub_key(kern, side) or kern
        local = "" if lm.group(1) == "_Z" else rest.rsplit("E", 1)[-1]
        return f"{lm.group(1)}STUB:{sk}:{local}"
    dm = re.match(r"^(guard variable for )?__device_stub__(Z[^(\s]*)(.*)$", name)
    if dm:
        # objdump's partly demangled forms: __device_stub__<mangled kernel>(<parameters>)[::<local>][ [clone ...]], with or without
        # "guard variable for "
        kern = re.sub(r"\s+", "", mask(demangle(dm.group(2))))
        sk = stub_key(kern, side) or kern
        rest = dm.group(3)
        local = rest.rsplit(")::", 1)[1] if ")::" in rest else ""
        clone = " ".join(re.findall(r"\[clone [^\]]*\]", rest))
        return ("GUARD:" if dm.group(1) else "") + "__device_stub__:" + sk + ("::" + local if local else "") + re.sub(r"\s+", "", clone)
    k = re.sub(r"\s+", "", mask(name))
    sk = stub_key(k, side)
    if sk is not None:
        return sk
    if side == "A" and not args.no_rename and "fk::" in k:
        k = psi["insert_pa"](k)
    return k


def disassemble(path, side):
    out = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", "-w", "-j", ".text", path], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"objdump failed on {path}: {out.stderr[:500]}")
    funcs = []
    for line in out.stdout.splitlines():
        h = HDR.match(line)
        if h:
            funcs.append([int(h.group(1), 16), h.group(2), []]); continue
        m = INS.match(line)
        if m and funcs:
            funcs[-1][2].append((int(m.group(1), 16), m.group(2)))
        elif line.strip() == "..." and funcs:
            funcs[-1][2].append((None, "..."))
    starts = [f[0] for f in funcs]
    seen = collections.Counter()
    keys = []
    for f in funcs:
        k = key_of(f[1], side); seen[k] += 1
        keys.append(k if seen[k] == 1 else f"{k}#{seen[k]}")
    collisions = sum(n - 1 for n in seen.values() if n > 1)

    def func_of(addr):
        i = bisect.bisect_right(starts, addr) - 1
        if i < 0: return None
        if i + 1 < len(starts) and addr >= starts[i + 1]: return None
        last = [a for a, _ in funcs[i][2] if a is not None]
        if not last or addr > last[-1] + 16: return None
        return i

    real, index = [], []
    for start, name, ins in funcs:
        r = [(a, t) for a, t in ins if a is not None and not PAD.match(" ".join(t.split()))]
        idx, k = {}, 0
        for a, t in ins:
            if a is None: continue
            idx[a] = k
            if not PAD.match(" ".join(t.split())): k += 1
        real.append(r); index.append(idx)

    res = {}
    for i, (start, name, _) in enumerate(funcs):
        norm, calls, ext, taken = [], set(), set(), set()
        last_line_mov = {}
        for addr, text in real[i]:
            t = " ".join(text.split())
            mnem = t.split(" ", 1)[0]
            m = TGT.search(t)
            line_call = None
            if m:
                target, sym = int(m.group(1), 16), m.group(2)
                j = func_of(target)
                if j == i:
                    rep = f"<SELF@{index[i].get(target, '?')}>"
                elif j is not None:
                    rep = f"<F:{keys[j]}@{index[j].get(target, '?')}>"
                    if mnem.startswith("j") or mnem.startswith("call"):
                        calls.add(j)
                    elif mnem.startswith("lea"):
                        taken.add(j)
                    if mnem.startswith("call"):
                        line_call = next((r for f, r in LINE_ARGS.items() if f in funcs[j][1]), None)
                else:
                    s = key_of(OFF.sub("", sym), side) + ("+OFF" if OFF.search(sym) else "")
                    rep = "<" + s + ">"
                    if (mnem.startswith("j") or mnem.startswith("call")) and "@plt" in s:
                        ext.add(s)
                    if mnem.startswith("call"):
                        line_call = next((r for f, r in LINE_ARGS.items() if f in sym), None)
                t = t[:m.start()] + (" " if m.start() > 0 else "") + rep
            t = RIP.sub("D(%rip)", t)
            t = mask(t)
            lm = LINE_MOV.match(t)
            if lm:
                last_line_mov[lm.group(1)] = len(norm)
            if line_call is not None and line_call in last_line_mov:
                norm[last_line_mov.pop(line_call)] = f"mov $LINE,{line_call}"
            norm.append(t)
        res[keys[i]] = dict(name=name, norm=norm, calls=calls, ext=ext, taken=taken)
    for v in res.values():
        v["calls"] = {keys[j] for j in v["calls"]}; v["taken"] = {keys[j] for j in v["taken"]}
    return res, collisions


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


A, ca = disassemble(args.a, "A")
B, cb = disassemble(args.b, "B")
common = sorted(set(A) & set(B))
same = {k for k in common if A[k]["norm"] == B[k]["norm"]}
diff = sorted(set(common) - same)
only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
roots_re = [re.compile(r) for r in args.roots]


def roots(G):
    return [k for k, v in G.items() if any(r.search(v["name"]) for r in roots_re)]


def reverse_reach(G, targets):
    """for each target, the roots that reach it"""
    rev = collections.defaultdict(set)
    for k, v in G.items():
        for c in v["calls"] | v["taken"]:
            rev[c].add(k)
    rs = set(roots(G))
    out = {}
    for t in targets:
        seen, todo = set(), [t]
        while todo:
            k = todo.pop()
            if k in seen: continue
            seen.add(k)
            todo.extend(rev.get(k, ()))
        out[t] = sorted(seen & rs)
    return out


# the encodings of a moved line: the immediate; inlined std::to_string of a 3-digit line writes the first two digits as one 16-bit
# immediate (little endian) and the third as a byte
LINE_PAIRS = [tuple(int(x) for x in p.split(":")) for p in args.line_pairs.split(",") if p]
ALLOWED = set()
for la, lb in LINE_PAIRS:
    ALLOWED.add((f"${hex(la)}", f"${hex(lb)}"))
    sa, sb = str(la), str(lb)
    if len(sa) == len(sb) == 3:
        ALLOWED.add((f"${hex(ord(sa[0]) | ord(sa[1]) << 8)}", f"${hex(ord(sb[0]) | ord(sb[1]) << 8)}"))
        ALLOWED.add((f"${hex(ord(sa[2]))}", f"${hex(ord(sb[2]))}"))
IMM = re.compile(r"\$0x[0-9a-f]+")


def line_only(a, b):
    """True when a and b have the same length and every differing instruction pair differs only in immediates that are an allowed (A, B)
    encoding of a moved line"""
    if (not ALLOWED and args.path_delta is None) or len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x == y:
            continue
        ix, iy = IMM.findall(x), IMM.findall(y)
        if IMM.sub("$I", x) != IMM.sub("$I", y) or len(ix) != len(iy):
            return False
        if any(p != q and (p, q) not in ALLOWED and not (args.path_delta is not None and int(q[1:], 16) - int(p[1:], 16) == args.path_delta)
               for p, q in zip(ix, iy)):
            return False
    return True


line_diff = sorted(k for k in diff if line_only(A[k]["norm"], B[k]["norm"]))
reach_a = reverse_reach(A, [k for k in diff + only_a])
reach_b = reverse_reach(B, [k for k in diff + only_b])
lines = [f"A = {args.a} (sha256 {sha(args.a)})", f"B = {args.b} (sha256 {sha(args.b)})",
         subprocess.run(["objdump", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
         "normalised as described in host_compare435.py (host_compare.py's normalisation, " + ("no rename rule" if args.no_rename else "FKL #435's rename rule on A's names") + ", fklb::statusMessage's __LINE__ masked)",
         f"TOTAL: functions A {len(A)}, B {len(B)}; pairs {len(common)}; identical {len(same)}, different {len(diff)}; only in A {len(only_a)}, "
         f"only in B {len(only_b)}; name collisions A {ca}, B {cb}",
         f"roots (timed calls; regular expressions {args.roots}): A {len(roots(A))}, B {len(roots(B))}",
         f"of the different: {len(line_diff)} differ only in the __LINE__ of the moved lines {args.line_pairs or '(none given)'}"
         + (f" or by the build path's length (immediates larger by {args.path_delta} in B)" if args.path_delta is not None else "") + ", "
         f"{len(diff) - len(line_diff)} otherwise"]
lines.append("every function that differs (first differing instruction; the roots that reach it in A | in B):")
for k in diff:
    a, b = A[k]["norm"], B[k]["norm"]
    i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
    lines.append(f"  DIFFERENT{' (only __LINE__' + (' / build path' if args.path_delta is not None else '') + ')' if k in line_diff else ''} {k[:400]}")
    lines.append(f"      {len(a)} -> {len(b)} instructions; first difference at {i}: A: {a[i][:220] if i < len(a) else '(end)'} | B: {b[i][:220] if i < len(b) else '(end)'}")
    lines.append(f"      roots A ({len(reach_a[k])}): {'; '.join(r[:200] for r in reach_a[k]) or 'none'}")
    lines.append(f"      roots B ({len(reach_b[k])}): {'; '.join(r[:200] for r in reach_b[k]) or 'none'}")
for k in only_a:
    lines.append(f"  ONLY IN A {k[:400]}  roots: {'; '.join(r[:200] for r in reach_a[k]) or 'none'}")
for k in only_b:
    lines.append(f"  ONLY IN B {k[:400]}  roots: {'; '.join(r[:200] for r in reach_b[k]) or 'none'}")
# rows: the lambdas (their invokers and bodies) classified by what they reach
LAMBDA = re.compile(r"\{lambda\(")


def closure(G, roots_):
    seen, todo = set(), [r for r in roots_ if r in G]
    while todo:
        k = todo.pop()
        if k in seen: continue
        seen.add(k)
        todo.extend(G[k]["calls"] | G[k]["taken"])
    return seen


if args.row:
    rows_ = [(r.split("=", 1)[0], re.compile(r.split("=", 1)[1])) for r in args.row]
    hre = re.compile(args.harness)
    lines.append("")
    lines.append(f"rows (a lambda belongs to the one row whose regular expression matches a function or an external call it reaches; harness: {args.harness}):")
    for side, G in (("B", B),):
        harness = [k for k, v in G.items() if hre.search(v["name"])]
        hreach = closure(A, [k for k in harness if k in A]) | closure(B, [k for k in harness if k in B])
        byrow = collections.defaultdict(list)
        for k, v in G.items():
            if not LAMBDA.search(v["name"]):
                continue
            reach = closure(A, [k] if k in A else []) | closure(B, [k] if k in B else [])
            text_ = " ".join(reach) + " " + " ".join(e for r in reach for H in (A, B) if r in H for e in H[r]["ext"])
            hit = [n for n, rx in rows_ if rx.search(text_)]
            if args.row_priority and hit:
                hit = hit[:1]
            byrow[hit[0] if len(hit) == 1 else ("AMBIGUOUS:" + "+".join(hit) if hit else "other")].append((k, reach))
        for n, _ in rows_ + [("other", None)]:
            lams = byrow.get(n, [])
            reach = set().union(*[r for _, r in lams]) if lams else set()
            dif = sorted(k for k in reach if k in diff and k not in line_diff)
            ldif = sorted(k for k in reach if k in line_diff)
            unp = sorted(k for k in reach if k not in A or k not in B)
            hd = sorted(k for k in hreach if (k in diff and k not in line_diff) or k not in A or k not in B)
            lines.append(f"  {n}: {len(lams)} lambdas; they reach {len(reach)} functions: different {len(dif)}, only __LINE__ / build path {len(ldif)}, "
                         f"unpaired {len(unp)}; the harness reaches {len(hreach)}, of which different or unpaired {len(hd)}")
            for k in dif + unp:
                lines.append(f"      {'DIFFERENT' if k in diff else 'UNPAIRED '} {k[:300]}")
        hd_all = sorted(k for k in hreach if (k in diff and k not in line_diff) or k not in A or k not in B)
        lines.append(f"  the harness ({len(harness)} functions) reaches {len(hreach)}; different or unpaired:")
        for k in hd_all:
            lines.append(f"      {'DIFFERENT' if k in diff else 'UNPAIRED '} {k[:300]}")
        for n in [x for x in byrow if x.startswith("AMBIGUOUS")]:
            lines.append(f"  {n}: {len(byrow[n])} lambdas: {'; '.join(k[:120] for k, _ in byrow[n])}")
# timed intervals
MARKS = {"event": re.compile(r"^call <cudaEventRecord@plt>$"), "clock": re.compile(r"^call <std::chrono::_V2::steady_clock::now\(\)@plt>$")}
CALLF = re.compile(r"<F:(.+)@\d+>$")
REGX = re.compile(r"%([a-z][a-z0-9]*)")


def reg_family(r):
    m = re.fullmatch(r"[xyz]mm(\d+)", r)
    if m: return ("v" + m.group(1), "vec")
    m = re.fullmatch(r"r(\d+)([dwb]?)", r)
    if m: return ("r" + m.group(1), {"": 64, "d": 32, "w": 16, "b": 8}[m.group(2)])
    for fam in ("a", "b", "c", "d"):
        w = {f"r{fam}x": 64, f"e{fam}x": 32, f"{fam}x": 16, f"{fam}l": 8, f"{fam}h": 8}
        if r in w: return (fam, w[r])
    for fam in ("si", "di", "bp", "sp"):
        w = {f"r{fam}": 64, f"e{fam}": 32, fam: 16, fam + "l": 8}
        if r in w: return (fam, w[r])
    return (r, None)


def equal_up_to_registers(a, b):
    """a and b are the same instructions under one consistent one-to-one renaming of the general-purpose and vector registers (%rsp, %rbp and
    %rip are kept); immediates and offsets must be equal or an allowed __LINE__ / build path pair"""
    if len(a) != len(b):
        return False
    fwd, back = {}, {}
    for x, y in zip(a, b):
        if x == y:
            continue
        rx, ry = REGX.findall(x), REGX.findall(y)
        if len(rx) != len(ry) or REGX.sub("%R", x) != REGX.sub("%R", y):
            if not line_only([REGX.sub("%R", x)], [REGX.sub("%R", y)]):
                return False
        for p, q in zip(rx, ry):
            (fp, wp), (fq, wq) = reg_family(p), reg_family(q)
            if wp != wq or (fp in ("sp", "bp", "rip") or fq in ("sp", "bp", "rip")) and fp != fq:
                return False
            if fwd.setdefault(fp, fq) != fq or back.setdefault(fq, fp) != fp:
                return False
    return True


def intervals_of(norm, kind):
    idx = [i for i, t in enumerate(norm) if MARKS[kind].match(t)]
    return [norm[i + 1:j] for i, j in zip(idx, idx[1:])]


def callees(seq):
    return {m.group(1) for t in seq for m in [CALLF.search(t)] if m and (t.startswith("call") or t.startswith("j"))}


EXTCALL = re.compile(r"^(?:call|jmp) <([^<>]*@plt)>$")


def ext_of(seq):
    return collections.Counter(m.group(1) for t in seq for m in [EXTCALL.match(t)] if m)


def closure_status(keys_a, keys_b=None):
    """the closure (through direct calls, jumps and taken addresses) of A's functions keys_a in A and of keys_b in B: how many are identical,
    differ only in __LINE__ / build path, differ, are unpaired"""
    cl = closure(A, keys_a) | closure(B, keys_a if keys_b is None else keys_b)
    st = collections.Counter()
    bad = []
    for k in cl:
        if k not in A or k not in B: st["unpaired"] += 1; bad.append(("UNPAIRED", k))
        elif k in same: st["identical"] += 1
        elif k in line_diff: st["line/path"] += 1
        elif equal_up_to_registers(A[k]["norm"], B[k]["norm"]): st["registers"] += 1; bad.append(("REGISTERS", k))
        else: st["different"] += 1; bad.append(("DIFFERENT", k))
    return st, bad


if args.intervals:
    lines.append("")
    lines.append("timed intervals (the instructions between consecutive calls of cudaEventRecord or steady_clock::now in one function; "
                 "a lambda body inlined into the timing function is inside its interval):")
    tot = collections.Counter()
    icsv = []
    for kind in ("event", "clock"):
      for k in sorted(set(A) | set(B)):
        na = len(intervals_of(A[k]["norm"], kind)) if k in A else 0
        nb = len(intervals_of(B[k]["norm"], kind)) if k in B else 0
        if na == 0 and nb == 0:
            continue
        if k not in A or k not in B:
            tot[(kind, "unpaired")] += 1
            lines.append(f"  [{kind}] UNPAIRED ({'A' if k in A else 'B'} only, {max(na, nb)} intervals) {k[:300]}")
            icsv.append([kind, k, "unpaired " + ("A only" if k in A else "B only"), "", "", "", "", ""])
            continue
        ia, ib = intervals_of(A[k]["norm"], kind), intervals_of(B[k]["norm"], kind)
        if len(ia) != len(ib):
            tot[(kind, "different")] += 1
            lines.append(f"  [{kind}] DIFFERENT (intervals {len(ia)} -> {len(ib)}) {k[:300]}")
            icsv.append([kind, k, f"different ({len(ia)} -> {len(ib)} intervals)", "", "", "", "", ""])
            continue
        sts = []
        for x, y in zip(ia, ib):
            if x == y: sts.append("identical")
            elif line_only(x, y): sts.append("line/path")
            elif equal_up_to_registers(x, y): sts.append("registers")
            else: sts.append("different")
        worst = max(sts, key=["identical", "line/path", "registers", "different"].index) if sts else "identical"
        ka = sorted(set().union(*[callees(x) for x in ia]) if ia else set())
        kb = sorted(set().union(*[callees(x) for x in ib]) if ib else set())
        cs, bad = closure_status(ka, kb)
        # the external functions (PLT calls) that the intervals and the functions they reach call, as sets
        ea = set(e for x in ia for e in ext_of(x)) | set(e for f in closure(A, ka) for e in A[f]["ext"])
        eb = set(e for x in ib for e in ext_of(x)) | set(e for f in closure(B, kb) for e in B[f]["ext"])
        ext_note = "same" if ea == eb else f"A only: {', '.join(sorted(ea - eb)) or '-'}; B only: {', '.join(sorted(eb - ea)) or '-'}"
        cworst = "different" if cs["different"] or cs["unpaired"] else ("registers" if cs["registers"] else ("line/path" if cs["line/path"] else "identical"))
        tot[(kind, worst, cworst)] += 1
        lines.append(f"  [{kind}] interval {worst}, its callees {cworst} ({dict(cs)}), external calls {'same' if ea == eb else 'differ'}: {len(ia)} intervals, "
                     f"{sum(map(len, ia))} -> {sum(map(len, ib))} instructions: {k[:260]}")
        if ea != eb:
            lines.append(f"      external calls {ext_note[:600]}")
        for what, kk in bad[:12]:
            lines.append(f"      callee {what} {kk[:240]}")
        icsv.append([kind, k, worst, cworst, f"{sum(map(len, ia))} -> {sum(map(len, ib))}", ";".join(f"{x}={n}" for x, n in sorted(cs.items())),
                     " | ".join(f"{w} {kk} ({len(A[kk]['norm']) if kk in A else '-'} -> {len(B[kk]['norm']) if kk in B else '-'})" for w, kk in bad),
                     ext_note])
    lines.append("  TOTAL: " + "; ".join(f"[{x[0]}] " + (x[1] if len(x) == 2 else f"interval {x[1]}, callees {x[2]}") + f": {n}"
                                       for x, n in sorted(tot.items(), key=str)))
    if args.intervals_csv:
        with open(args.intervals_csv, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["kind", "function", "interval", "callees", "interval_instructions_a_b", "callee_counts", "callees_not_identical", "external_calls"])
            wr.writerows(icsv)
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["function", "result", "instructions_a", "instructions_b", "roots_reaching_it"])
        for k in sorted(set(A) | set(B)):
            r = "only_a" if k not in B else "only_b" if k not in A else ("identical" if k in same else
                                                                        "different_line_only" if k in line_diff else "different")
            rr = reach_a.get(k, []) if r in ("different", "different_line_only", "only_a") else reach_b.get(k, []) if r == "only_b" else []
            w.writerow([k, r, len(A[k]["norm"]) if k in A else "", len(B[k]["norm"]) if k in B else "", ";".join(rr)])
if args.diffs:
    with open(args.diffs, "w") as f:
        f.write(f"A = {args.a}\nB = {args.b}\nper function that differs: the unified diff of its normalised instructions\n")
        for k in diff:
            f.write(f"\n== {k}\n   {len(A[k]['norm'])} -> {len(B[k]['norm'])} instructions\n")
            for l in difflib.unified_diff(A[k]["norm"], B[k]["norm"], "A", "B", n=3, lineterm=""):
                f.write(l + "\n")
sys.exit(0 if not diff and not only_a and not only_b else 1)
