#!/usr/bin/env python3
"""C-NV-A: pr435_sass_identity.py WITHOUT FKL #435's rename rule, for two builds whose kernels already have the new names (48fbf3f against
d1cff06). The rename rule would insert the attention DPPs' leading ParArch a second time into A's names, and they would not pair.

  sass_identity_norename.py <pr435_sass_identity.py> <its arguments...>

It runs pr435_sass_identity.py unchanged except that its rewrite(key) returns (key, "") (the kernels are paired by their demangled names as they
are, with the module ids masked as before). Everything else, the comparison and the report, is the script's own.
"""
import sys

path = sys.argv[1]
src = open(path).read()
marker = "def rewrite(key):"
assert src.count(marker) == 1, "pr435_sass_identity.py has no single rewrite(key)"
src = src.replace(marker, "def rewrite(key):\n    return key, \"\"  # sass_identity_norename.py: no rename rule (the key as it is, no rule)\n\n\ndef _rewrite_with_the_rule(key):")
sys.argv = [path] + sys.argv[2:]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
