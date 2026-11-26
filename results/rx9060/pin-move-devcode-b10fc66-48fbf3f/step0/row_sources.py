# row_sources.py <repo> <measured commit> <commit to compare> <row>...: for each row of rpp/benchmark_rpp_preprocessing.h, the source of its
# timed lambda (the variable that the timeRotated list gives that row: `auto <var> = [&](fklb::WallClock::time_point...) { ... };`) at the two
# commits, compared after collapsing whitespace; and the same for the program's timing functions (timeVariant, timeRotated) and for
# fklb::measureWall and the functions it calls in common/ (hip_wallclock.h). Prints one line per item: same / DIFFERS / missing, with a diff.
import difflib
import re
import subprocess
import sys

repo, measured, other, rows = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
PROG = "rpp/benchmark_rpp_preprocessing.h"


def show(commit, path):
    r = subprocess.run(["git", "show", f"{commit}:{path}"], cwd=repo, capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def block_from(text, start):
    # from text[start], through the first '{' to its matching '}'
    i = text.index("{", start)
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[start:j + 1]
    raise ValueError("unbalanced")


def rowvars(text):
    m = re.search(r"timeRotated\(out, N, [^,]+,\s*\{(.*?)\}\);", text, re.S)
    return dict(re.findall(r'\{"([a-z_]+)", [^,]+, (\w+)\}', m.group(1)))


def lam(text, var):
    m = re.search(r"auto " + var + r" = \[&\]\(fklb::WallClock::time_point&", text)
    return block_from(text, m.start()) if m else None


def func(text, signature_rx):
    m = re.search(signature_rx, text)
    return block_from(text, m.start()) if m else None


def norm(s):
    return re.sub(r"\s+", " ", s).strip() if s is not None else None


def compare(label, a, b):
    if a is None or b is None:
        print(f"{label}: missing at {'the measured commit' if a is None else 'the other commit'}")
        return
    if norm(a) == norm(b):
        print(f"{label}: same")
    else:
        print(f"{label}: DIFFERS")
        for l in difflib.unified_diff(a.splitlines(), b.splitlines(), measured, other, n=1, lineterm=""):
            print("    " + l)


pa, pb = show(measured, PROG), show(other, PROG)
va, vb = rowvars(pa), rowvars(pb)
for r in rows:
    compare(f"row {r} (lambda {va.get(r)} / {vb.get(r)})", lam(pa, va[r]) if r in va else None, lam(pb, vb[r]) if r in vb else None)
for name, rx in (("timeVariant", r"inline void timeVariant\("), ("timeRotated", r"inline void timeRotated\(")):
    compare(f"program function {name}", func(pa, rx), func(pb, rx))
wa, wb = show(measured, "common/hip_wallclock.h"), show(other, "common/hip_wallclock.h")
for name, rx in (("fklb::measureWall", r"\n[^\n]*\bmeasureWall\("),):
    compare(f"harness function {name} (common/hip_wallclock.h)", func(wa, rx), func(wb, rx))
compare("common/hip_wallclock.h, the whole file", wa, wb)
