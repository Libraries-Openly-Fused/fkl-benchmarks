#!/usr/bin/env python3
"""Host code of one E19 program in two builds, function by function (CPU only: the binaries are only read with objdump; nothing is run).

  host_compare.py <binary A> <binary B> [--out report.txt] [--csv functions.csv] [--reach reach.txt] [--diffs diffs.txt]

Disassembly: `objdump -d -C --no-show-raw-insn -w -j .text` of each binary. A function is the code between two symbol headers of .text
(its .cold clone is a function of its own). Each instruction is normalised:
  - the address of the instruction is dropped;
  - padding is removed wherever it is (nop forms: nop, nopw, nopl, xchg %ax,%ax, data16 ... nopw; int3; `...`): the assembler aligns loop
    heads and functions relative to the section, so their number depends on where the function lands, not on its code;
  - a direct target or a RIP-relative reference (`<hex> <symbol+off>` at the end of the line) becomes `<SELF@i>` when it points inside the
    same function and `<F:key@i>` when it points into another .text function (key as below), i = the index of the target instruction among
    the function's instructions without padding (a target on padding takes the next instruction); otherwise `<symbol>` or `<symbol+OFF>`
    (PLT entries, data); the hex address itself is dropped;
  - RIP-relative displacements `0x..(%rip)` become `D(%rip)`;
  - the module id of the anonymous namespace that nvcc puts in names (`_GLOBAL__N__<8 hex>_<len>_<file>_<8 hex>`) is masked, in function
    names and in operands, as are the 8-hex ids of `_GLOBAL__sub_I_` style names;
Functions are paired by their demangled name with those ids masked; a name that occurs more than once in a binary gets #2, #3, ... in
address order (counted as a collision). Reported: functions in A and B, pairs, identical, different, only in A or B.

Rows. The timed call of each row is a lambda stored in a std::function (`Row::call`) and run by fklb::timeIt and fklb::timePreparedLaunch.
For every `std::_Function_handler<void (), CTX::{lambda()#k}>::_M_invoke` the direct call graph is followed: direct calls and jumps, and
.text functions whose address is taken by a RIP-relative lea (counted as reached: a superset). Indirect calls and jumps (`call *...`, `jmp *...`) are counted,
not followed; PLT calls are listed as external (the shared libraries are the same files for both builds and are not compared). Each such
lambda is classified by what it reaches: a function whose name contains morphologyDPPKernel -> fkl_batch, linearFilterDPPKernel ->
fkl_loop, an external nppi* -> npp, an external cv::cuda:: -> opencv_cuda, anything else -> other. CTX is the enclosing function (launch()
or one instantiation launch<N>()). The harness of a context: fklb::timeIt<std::function<void ()> > (shared by every row and every context)
and every fklb::timePreparedLaunch<CTX::...>. For each row of each context: the functions reached from its lambda and from the harness, those
of them that differ, and those that the row shares with another row of its context (reached from both lambdas).
Outputs: --out the report; --csv every function of A and B (instructions in each, result, the rows whose timed calls reach it); --reach per
context and row, every function its timed call reaches (from its lambda, or from the harness only), with its result and the other rows of
the context that reach it from their lambdas; --diffs per function that differs, the unified diff of its normalised instructions (A, B).
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
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--out", default=None); ap.add_argument("--csv", default=None)
ap.add_argument("--reach", default=None); ap.add_argument("--diffs", default=None)
args = ap.parse_args()

ANON = re.compile(r"_GLOBAL__N__[0-9a-f]{8}_(\d+)_([A-Za-z0-9_]+?)_[0-9a-f]{8}")
HDR = re.compile(r"^([0-9a-f]+) <(.*)>:$")
INS = re.compile(r"^\s*([0-9a-f]+):\t(.*)$")
TGT = re.compile(r"(?:^|\s)([0-9a-f]+) <(.+)>$")
RIP = re.compile(r"-?0x[0-9a-f]+\(%rip\)")
PAD = re.compile(r"^(nop|xchg\s+%ax,%ax|data16|cs nop|int3|\.\.\.)")
OFF = re.compile(r"\+0x[0-9a-f]+$")


def mask(s):
    return ANON.sub(lambda m: f"_GLOBAL__N__H_{m.group(1)}_{m.group(2)}_H", s)


def disassemble(path):
    out = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", "-w", "-j", ".text", path], capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"objdump failed on {path}: {out.stderr[:500]}")
    funcs = []   # [start, name, [(addr, text)]]
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
        k = mask(f[1]); seen[k] += 1
        keys.append(k if seen[k] == 1 else f"{k}#{seen[k]}")
    collisions = sum(n - 1 for n in seen.values() if n > 1)

    def func_of(addr):
        i = bisect.bisect_right(starts, addr) - 1
        if i < 0: return None
        if i + 1 < len(starts) and addr >= starts[i + 1]: return None
        last = [a for a, _ in funcs[i][2] if a is not None]
        if not last or addr > last[-1] + 16: return None
        return i

    # per function: the instructions without padding, and the index of the instruction at (or, for padding, after) each address
    real, index = [], []
    for start, name, ins in funcs:
        r = [(a, t) for a, t in ins if a is not None and not PAD.match(" ".join(t.split()))]
        idx, k = {}, 0
        for a, t in ins:
            if a is None: continue
            idx[a] = k
            if not PAD.match(" ".join(t.split())): k += 1
        real.append(r); index.append(idx)

    def at(j, addr):
        return index[j].get(addr, "?")

    res = {}
    for i, (start, name, _) in enumerate(funcs):
        ins = real[i]
        norm, calls, ext, indirect, taken = [], set(), set(), 0, set()
        ind_lines = []
        for addr, text in ins:
            t = " ".join(text.split())
            mnem = t.split(" ", 1)[0]
            m = TGT.search(t)
            if m:
                target, sym = int(m.group(1), 16), m.group(2)
                j = func_of(target)
                if j == i:
                    rep = f"<SELF@{at(i, target)}>"
                elif j is not None:
                    rep = f"<F:{keys[j]}@{at(j, target)}>"
                    if mnem.startswith("j") or mnem.startswith("call"):
                        calls.add(j)
                    elif mnem.startswith("lea"):
                        taken.add(j)
                else:
                    s = mask(sym); rep = "<" + OFF.sub("+OFF", s) + ">"
                    if (mnem.startswith("j") or mnem.startswith("call")) and "@plt" in s:
                        ext.add(s)
                t = t[:m.start()] + (" " if m.start() > 0 else "") + rep
            ops = t.split(" ", 1)[1] if " " in t else ""
            if (mnem.startswith("call") or mnem.startswith("jmp") or mnem == "notrack") and (ops.startswith("*") or ops.startswith("jmp *")):
                indirect += 1; ind_lines.append(t)
            t = RIP.sub("D(%rip)", t)
            norm.append(mask(t))
        res[keys[i]] = dict(name=name, start=start, norm=norm, calls=calls, ext=ext, indirect=indirect, taken=taken, idx=i, ind_lines=ind_lines)
    # translate callee indices to keys
    for k, v in res.items():
        v["calls"] = {keys[j] for j in v["calls"]}; v["taken"] = {keys[j] for j in v["taken"]}
    return res, collisions, len(funcs)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


A, ca, na = disassemble(args.a)
B, cb, nb = disassemble(args.b)
lines = [f"A = {args.a} (sha256 {sha(args.a)})", f"B = {args.b} (sha256 {sha(args.b)})",
         subprocess.run(["objdump", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
         "objdump -d -C --no-show-raw-insn -w -j .text; normalised as described in host_compare.py"]
common = sorted(set(A) & set(B))
same = {k for k in common if A[k]["norm"] == B[k]["norm"]}
diff = sorted(set(common) - same)
only_a, only_b = sorted(set(A) - set(B)), sorted(set(B) - set(A))
lines.append(f"TOTAL: functions A {len(A)}, B {len(B)}; pairs {len(common)}; identical {len(same)}, different {len(diff)}; only in A {len(only_a)}, "
             f"only in B {len(only_b)}; name collisions A {ca}, B {cb}")


def closure(G, roots):
    seen, todo = set(), [r for r in roots if r in G]
    while todo:
        k = todo.pop()
        if k in seen: continue
        seen.add(k)
        todo.extend(G[k]["calls"] | G[k]["taken"])
    return seen


INV = re.compile(r"^std::_Function_handler<void \(\), (.+)::\{lambda\(\)#(\d+)\}>::_M_invoke\(std::_Any_data const&\)$")


def rows_of(G):
    out = collections.defaultdict(dict)   # ctx -> {row: [keys]}
    for k in G:
        m = INV.match(k)
        if not m: continue
        ctx, num = m.group(1), int(m.group(2))
        reach = closure(G, [k])
        names = " ".join(reach); ext = " ".join(e for r in reach for e in G[r]["ext"])
        kinds = [r for r, hit in (("fkl_batch", "morphologyDPPKernel" in names), ("fkl_loop", "linearFilterDPPKernel" in names),
                                  ("npp", re.search(r"(^|\s|<)nppi", ext) is not None), ("opencv_cuda", "cv::cuda::" in ext)) if hit]
        row = kinds[0] if len(kinds) == 1 else ("other" if not kinds else "AMBIGUOUS:" + "+".join(kinds))
        out[ctx].setdefault(row, []).append((num, k))
    return out


def harness(G, ctx):
    return [k for k in G if k.startswith("void fklb::timeIt<std::function<void ()> >(") or k.startswith(f"void fklb::timePreparedLaunch<{ctx}::")]


rows_a, rows_b = rows_of(A), rows_of(B)
problems = []
if {c: {r: [n for n, _ in v] for r, v in d.items()} for c, d in rows_a.items()} != {c: {r: [n for n, _ in v] for r, v in d.items()} for c, d in rows_b.items()}:
    problems.append("the row lambdas are not classified the same way in A and B")
reach_rows = collections.defaultdict(set)   # key -> {"ctx:row"}
reach_lists = []
summary, per_row = [], []
for ctx in sorted(rows_a):
    d = rows_a[ctx]
    for row in ("fkl_loop", "fkl_batch", "opencv_cuda", "npp"):
        if len(d.get(row, [])) != 1:
            problems.append(f"{ctx}: {len(d.get(row, []))} lambdas classified {row}")
    for r in d:
        if r.startswith("AMBIGUOUS"):
            problems.append(f"{ctx}: lambdas {[n for n, _ in d[r]]} are {r}")
    own = {}
    for row in ("fkl_loop", "fkl_batch", "opencv_cuda", "npp"):
        if len(d.get(row, [])) != 1: continue
        root = d[row][0][1]
        own[row] = closure(A, [root]) | closure(B, [root])
    hs = harness(A, ctx)
    hreach = closure(A, hs) | closure(B, hs)
    for row, reach in own.items():
        total = reach | hreach
        for k in total: reach_rows[k].add(f"{ctx}:{row}")
        shared = sorted(k for k in reach if any(k in own[o] for o in own if o != row))
        dif = sorted(k for k in total if k in diff)
        unpaired = sorted(k for k in total if k not in A or k not in B)
        ind = sum(A[k]["indirect"] for k in total if k in A)
        ind_by = sorted((A[k]["indirect"], k) for k in total if k in A and A[k]["indirect"])
        ext = sorted({e for k in total for G in (A, B) if k in G for e in G[k]["ext"]})
        reach_lists.append((ctx, row, d[row][0][0], sorted(reach), sorted(hreach - reach), own))
        per_row.append(dict(ctx=ctx, row=row, lam=d[row][0][0], own=len(reach), harness=len(hreach - reach), dif=dif, shared=shared,
                            unpaired=unpaired, indirect=ind, ext=ext, ind_by=ind_by))
        summary.append(f"  {ctx} {row} (lambda()#{d[row][0][0]}): reaches {len(reach)} functions from its lambda + {len(hreach - reach)} more "
                       f"from the harness; different {len(dif)}; unpaired {len(unpaired)}; shared with another row {len(shared)}; "
                       f"indirect calls {ind}; external {len(ext)}")
lines.append("rows (classified by what the lambda reaches; harness = fklb::timeIt<std::function<void ()> > and fklb::timePreparedLaunch<CTX::...>):")
lines += summary
lines.append("problems: " + ("; ".join(problems) if problems else "none"))
# the question: fkl_batch
fb = [p for p in per_row if p["row"] == "fkl_batch"]
lines.append("")
lines.append("FKL_BATCH: functions inside the timed call (its lambda and everything it reaches, and the harness) that differ: "
             + ("NONE in any context" if fb and all(not p["dif"] for p in fb) and not problems else
                ("; ".join(f"{p['ctx']}: {len(p['dif'])}" for p in fb) or "no fkl_batch row found")))
for p in per_row:
    lines.append(f"{p['ctx']} {p['row']}:")
    for k in p["dif"]:
        lines.append(f"    DIFFERENT  {k[:300]}  ({len(A[k]['norm'])} -> {len(B[k]['norm'])} instructions)")
    for k in p["unpaired"]:
        lines.append(f"    UNPAIRED   {k[:300]}")
    for k in p["shared"]:
        lines.append(f"    shared with another row: {k[:300]}{'  (DIFFERENT)' if k in diff else ''}")
    lines.append(f"    external calls ({len(p['ext'])}): {', '.join(e[:120] for e in p['ext'])}")
    lines.append(f"    indirect calls and jumps (not followed), by function, in A: " + "; ".join(f"{n} in {k[:160]} ({', '.join(sorted(set(A[k]['ind_lines'])))})" for n, k in reversed(p['ind_by'])))
lines.append("")
lines.append("every function that differs (first differing instruction, and which rows' timed calls reach it):")
for k in diff:
    a, b = A[k]["norm"], B[k]["norm"]
    i = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
    lines.append(f"  {k[:400]}")
    lines.append(f"      {len(a)} -> {len(b)} instructions; first difference at {i}: A: {a[i][:200] if i < len(a) else '(end)'} | B: {b[i][:200] if i < len(b) else '(end)'}")
    lines.append(f"      reached by: {', '.join(sorted(reach_rows.get(k, []))) or 'no timed call'}")
for k in only_a:
    lines.append(f"  ONLY IN A  {k[:400]}  reached by: {', '.join(sorted(reach_rows.get(k, []))) or 'no timed call'}")
for k in only_b:
    lines.append(f"  ONLY IN B  {k[:400]}  reached by: {', '.join(sorted(reach_rows.get(k, []))) or 'no timed call'}")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["function", "result", "instructions_a", "instructions_b", "reached_by"])
        for k in sorted(set(A) | set(B)):
            r = "only_a" if k not in B else "only_b" if k not in A else ("identical" if k in same else "different")
            w.writerow([k, r, len(A[k]["norm"]) if k in A else "", len(B[k]["norm"]) if k in B else "", ";".join(sorted(reach_rows.get(k, [])))])


def result(k):
    return "only_a" if k not in B else "only_b" if k not in A else ("identical" if k in same else "DIFFERENT")


if args.reach:
    with open(args.reach, "w") as f:
        f.write(f"A = {args.a}\nB = {args.b}\nper context and row: every function reached by the row's timed call; "
                "'lambda' = reached from the row's lambda (its _M_invoke), 'harness' = reached only from the harness; "
                "'also' = the other rows of the context whose lambdas reach it\n")
        for ctx, row, lam, own_keys, harness_keys, own in reach_lists:
            f.write(f"\n== {ctx} {row} (lambda()#{lam}): {len(own_keys)} from the lambda, {len(harness_keys)} from the harness only; "
                    f"different {sum(1 for k in own_keys + harness_keys if result(k) != 'identical')}\n")
            for tag, ks in (("lambda", own_keys), ("harness", harness_keys)):
                for k in ks:
                    also = sorted(o for o in own if o != row and k in own[o])
                    f.write(f"  {tag:7s} {result(k):9s} {k}{'  also: ' + ','.join(also) if also else ''}\n")
if args.diffs:
    with open(args.diffs, "w") as f:
        f.write(f"A = {args.a}\nB = {args.b}\nper function that differs: the unified diff of its normalised instructions (see host_compare.py)\n")
        for k in diff:
            f.write(f"\n== {k}\n   {len(A[k]['norm'])} -> {len(B[k]['norm'])} instructions; reached by: {', '.join(sorted(reach_rows.get(k, []))) or 'no timed call'}\n")
            for l in difflib.unified_diff(A[k]["norm"], B[k]["norm"], "A", "B", n=3, lineterm=""):
                f.write(l + "\n")
sys.exit(0 if not diff and not only_a and not only_b and not problems else 1)
