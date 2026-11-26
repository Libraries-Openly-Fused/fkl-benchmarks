#!/usr/bin/env python3
"""C-MAIN-NV: pr435_sass_identity.py with FKL #435's rename rule and one more rule for FKL main's softmax kernel (FKL #436).

  sass_identity_main.py <pr435_sass_identity.py> <its arguments...>

FKL main before #436 (ebb3891) already launched its softmax through a kernel named launchDPP_Kernel, with five template parameters:
  launchDPP_Kernel<fk::SoftmaxDPP, Details, InIOp, NullType, OutIOp>    (softmax.h:160 at ebb3891; NullType: no compute IOp)
#436 makes SoftmaxDPP a template on ParArch and its details (Oscar's M1) and launches the common kernel without the NullType:
  launchDPP_Kernel<fk::SoftmaxDPP<(fk::ParArch)1, Details>, Details, InIOp, OutIOp>
The extra rule rewrites A's names of the first form into the second (the template arguments, and the parameter list from them, as
pr435_sass_identity.rewrite builds it), then pr435_sass_identity.rewrite applies #435's rule as before. A softmax kernel of A with a
compute IOp other than NullType is left as it is (#436 has no such form; none is instantiated by FKL main's tests).
Everything else, the comparison and the report, is pr435_sass_identity.py's own.
"""
import sys

path = sys.argv[1]
src = open(path).read()
marker = "def rewrite(key):"
assert src.count(marker) == 1, "pr435_sass_identity.py has no single rewrite(key)"
extra = '''def rewrite(key):
    """C-MAIN-NV: FKL main's softmax kernel first (sass_identity_main.py), then FKL #435's rule."""
    head = "voidfk::launchDPP_Kernel<"
    if key.startswith(head + "fk::SoftmaxDPP,"):
        a, close = split_args(key, len(head))
        rest = key[close + 1:]
        if len(a) == 5 and a[0] == "fk::SoftmaxDPP" and a[3] in ("NullType", "fk::NullType") and rest.startswith("(") and rest.endswith(")"):
            new = [f"fk::SoftmaxDPP<(fk::ParArch)1,{a[1]}>", a[1], a[2], a[4]]
            return f"{head}{','.join(new)}>({','.join(new[1:])})", "main softmax (NullType dropped)"
    return _rewrite_435(key)


def _rewrite_435(key):'''
src = src.replace(marker, extra)
sys.argv = [path] + sys.argv[2:]
exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
