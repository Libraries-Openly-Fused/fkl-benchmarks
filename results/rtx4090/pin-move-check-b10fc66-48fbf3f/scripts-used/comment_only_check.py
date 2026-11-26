#!/usr/bin/env python3
"""C-PM-CUDA, finding E: does a change of a checkout touch comments and prose only? (CPU only: git and file reads.)

  comment_only_check.py <checkout> <old revision> [<new revision>, default: the working tree] [--out report.txt]

For every file that differs between the two:
  - the number of lines must be the same (so no __LINE__ of a later line moves);
  - a C or C++ file (.cu .cuh .h .hpp .cpp .cc .c): the comments are removed with a lexer that knows string, character and raw string
    literals (a comment is replaced by nothing, its newlines kept), and then every line must be the same in both, character for character.
    So the compiler sees the same tokens on the same lines: no compiled token and no __LINE__ value changes;
  - any other file (.md, .txt, ...): prose, not compiled; reported as such.
Exit 0 when every differing file passes.
"""
import argparse
import subprocess
import sys

CODE = (".cu", ".cuh", ".h", ".hpp", ".cpp", ".cc", ".c")
ap = argparse.ArgumentParser()
ap.add_argument("repo"); ap.add_argument("old"); ap.add_argument("new", nargs="?"); ap.add_argument("--out")
a = ap.parse_args()


def git(*args):
    return subprocess.run(["git", "-C", a.repo, *args], capture_output=True, text=True, check=True).stdout


def strip_comments(s):
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if s.startswith("//", i):
            j = s.find("\n", i)
            i = n if j < 0 else j
        elif s.startswith("/*", i):
            j = s.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("\n" * s.count("\n", i, j))
            i = j
        elif c == "R" and s.startswith('R"', i) and (i == 0 or not (s[i - 1].isalnum() or s[i - 1] == "_")):
            k = s.find("(", i + 2)
            delim = s[i + 2:k]
            j = s.find(")" + delim + '"', k)
            j = n if j < 0 else j + len(delim) + 2
            out.append(s[i:j]); i = j
        elif c in "\"'":
            j = i + 1
            while j < n and s[j] != c:
                j += 2 if s[j] == "\\" else 1
            out.append(s[i:j + 1]); i = j + 1
        else:
            out.append(c); i += 1
    return "".join(out)


def text_at(rev, path):
    if rev is None:
        return open(f"{a.repo}/{path}").read()
    return git("show", f"{rev}:{path}")


files = [f for f in (git("diff", "--name-only", a.old, a.new) if a.new else git("diff", "--name-only", a.old)).split("\n") if f]
lines, bad = [f"old {git('rev-parse', a.old).strip()}, new {a.new and git('rev-parse', a.new).strip() or 'the working tree'}; files that differ: {len(files)}"], 0
for f in files:
    x, y = text_at(a.old, f), text_at(a.new, f)
    nx, ny = x.count("\n"), y.count("\n")
    if nx != ny:
        bad += 1; lines.append(f"  FAIL {f}: {nx} -> {ny} lines"); continue
    if f.endswith(CODE):
        cx, cy = strip_comments(x).split("\n"), strip_comments(y).split("\n")
        diff = [i + 1 for i, (p, q) in enumerate(zip(cx, cy)) if p != q]
        if len(cx) != len(cy) or diff:
            bad += 1; lines.append(f"  FAIL {f}: the code differs at lines {diff[:20]}"); continue
        changed = [i + 1 for i, (p, q) in enumerate(zip(x.split("\n"), y.split("\n"))) if p != q]
        lines.append(f"  ok   {f}: {nx} lines in both; lines {changed} differ in comments only; the code without comments is the same on every line")
    else:
        changed = [i + 1 for i, (p, q) in enumerate(zip(x.split("\n"), y.split("\n"))) if p != q]
        lines.append(f"  prose {f}: {nx} lines in both; lines {changed} differ; not compiled")
lines.append(f"result: {'every differing file passes' if not bad else f'{bad} file(s) FAIL'}")
text = "\n".join(lines)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
sys.exit(1 if bad else 0)
