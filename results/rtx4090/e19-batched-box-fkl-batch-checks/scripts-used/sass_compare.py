#!/usr/bin/env python3
"""The SASS of every function of two cuobjdump -sass dumps (kernel_census.py's .sass files), compared function by function (nothing is run).
gpuSpin is in an anonymous namespace, whose hash in the mangled name follows the file's content, so it is also compared by its bare name.

  sass_compare.py <old.sass> <new.sass>
"""
import re, sys
def funcs(path):
    s = open(path).read()
    out = {}
    for f in re.split(r'\n\s*Function : ', s)[1:]:
        name, body = f.split('\n', 1)
        out[name.strip()] = body
    return out
old, new = funcs(sys.argv[1]), funcs(sys.argv[2])
print("old:", len(old), "functions; new:", len(new), "functions")
for name in old:
    if name in new:
        print(("IDENTICAL " if old[name] == new[name] else "DIFFERENT ") + name[:80], f"({old[name].count(chr(10))} lines)")
    else:
        print("MISSING in new:", name[:80])
for name in new:
    if name not in old:
        print("NEW:", name[:100], f"({new[name].count(chr(10))} lines)")
o = [v for k, v in old.items() if 'gpuSpin' in k]
n = [v for k, v in new.items() if 'gpuSpin' in k]
if o and n:
    print("gpuSpin body (the anonymous namespace hash in its mangled name aside):", "IDENTICAL" if o[0] == n[0] else "DIFFERENT")
