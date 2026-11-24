#!/usr/bin/env python3
"""The checks of the E19 crop sweep's checks window, per N (nothing is run).

  checks_summary.py <window folder> [--out report.txt] [--csv between.csv]

Reads <window folder>/checks/N<NNN>/e19_batched_box_checks.csv (one run per N, E19_NO_TIMING=1). Per N: the gated rows (criterion other than
"reported") and how many passed; the column crops (it must be N in every row); the main numbers of every row; the between rows of fkl_batch
against opencv_cuda and npp (the values that differ, the largest difference in output units and in grey levels).
"""
import argparse, csv, glob, os
ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--out", default=None); ap.add_argument("--csv", default=None)
args = ap.parse_args()
lines, between = [], []
for d in sorted(glob.glob(os.path.join(args.folder, "checks", "N*"))):
    n = int(os.path.basename(d)[1:])
    with open(os.path.join(d, "e19_batched_box_checks.csv")) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    gated = [r for r in rows if r["criterion"] != "reported"]
    passed = sum(r["ok"] == "yes" for r in gated)
    cols = {r["crops"] for r in rows}
    values = {r["values"] for r in rows if r["step"] != "buffers"}
    get = lambda v, s, c, rule=None: [r for r in rows if r["version"] == v and r["step"] == s and r["compared_with"] == c and (rule is None or rule in r["rule"])]
    one = lambda v, s, c, rule=None: get(v, s, c, rule)[0] if get(v, s, c, rule) else None
    fb = one("fkl_batch", "float", "fkl_batch_reference"); fw = one("fkl_batch", "written", "fill"); fl = one("fkl_loop", "float", "fkl_reference")
    oc = one("opencv_cuda", "float", "conversion_of_own_box"); np_ = one("npp", "float", "conversion_of_own_box")
    npp_grey = [r for r in get("npp", "grey", "host_reference") if r["criterion"] != "reported"]
    buf = one("opencv_cuda", "buffers", "before_the_call")
    lines.append(f"N = {n}: {len(rows)} rows; gated {passed} of {len(gated)} passed{'' if passed == len(gated) else ' (FAILED)'}; column crops {sorted(cols)}"
                 f"{'' if cols == {str(n)} else ' (NOT N)'}; values per comparison {sorted(values)} (65,536 N = {65536 * n})")
    lines.append(f"  fkl_batch vs its reference: max {float(fb['max_abs_difference']):.3g}, beyond 1e-5 {fb['differing_values']}, non-finite {fb['non_finite']}; "
                 f"left from the fill {fw['differing_values']}; fkl_loop vs its reference: max {float(fl['max_abs_difference']):.3g}, non-finite {fl['non_finite']}")
    lines.append(f"  opencv_cuda float vs own box {float(oc['max_abs_difference']):.3g}; npp float vs own box {float(np_['max_abs_difference']):.3g}; "
                 f"npp grey rule: {npp_grey[0]['rule'] if npp_grey else '?'} ({npp_grey[0]['differing_values'] if npp_grey else '?'} differ); "
                 f"opencv buffers moved {buf['differing_values'] if buf else '?'} of {buf['values'] if buf else '?'}")
    fbl = one("fkl_batch", "float", "fkl_loop")
    lines.append(f"  fkl_batch vs fkl_loop: max {float(fbl['max_abs_difference']):.3g} (reported)")
    for other in ("opencv_cuda", "npp"):
        u, g = one("fkl_batch", "float", other), one("fkl_batch", "float_grey_levels", other)
        if u is None or g is None:
            lines.append(f"  fkl_batch vs {other}: MISSING")
            continue
        lines.append(f"  between fkl_batch and {other}: {u['differing_values']} of {u['values']} differ ({100 * int(u['differing_values']) / int(u['values']):.2f} %); "
                     f"largest difference {float(u['max_abs_difference']):.6g} in output units = {float(g['max_abs_difference']):.6g} grey levels; non-finite {u['non_finite']}")
        between.append({"crops": n, "against": other, "values": u["values"], "differing_values": u["differing_values"],
                        "max_abs_output_units": u["max_abs_difference"], "max_abs_grey_levels": g["max_abs_difference"], "non_finite": u["non_finite"]})
t = "\n".join(lines); print(t)
if args.out: open(args.out, "w").write(t + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(between[0].keys())); w.writeheader(); w.writerows(between)
