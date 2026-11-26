#!/usr/bin/env python3
"""C-PM-CUDA: ports the calls of the five E18/E19 CUDA programs to FKL #435 (the code only; the comments are edited by hand).

  port_e18_e19.py <worktree>

- if (!fk::executeLinearFilter(ARGS, fkStream)) { throw ...; }
      -> fk::executeOperations<fk::LinearFilterDPP<fk::ParArch::GPU_NVIDIA, D>>(fkStream, ARGS);
- const bool okX = fk::executeLinearFilter(ARGS, fkStream);  ...  if (!okAsym || !okHalf) throw ...;
      -> the same call, and the check line goes
- valid = fk::executeLinearFilter(ARGS, fkStream);  (bool valid = false; ... if (!valid) throw ...;)
      -> the same call, and the flag and the check go
- if (!fk::executeBoxBlur<1>(details, image, output, fkStream)) { throw ...; }
      -> fk::executeOperations<fk::MorphologyDPP<fk::ParArch::GPU_NVIDIA, FklBatchDetails>>(fkStream, details, image, output,
         fk::boxBlurReducers<float, 1>(details));
The bool checks go because the executors throw std::invalid_argument on invalid details (FKL #435). Every call stays where it is.
Lines longer than 150 columns are wrapped after a comma, the continuation indented by 4. Prints what it changed per file.
"""
import re
import sys

W = sys.argv[1]
FILES = {"opencv-cuda/stencil/e18_box.cu": "Details", "opencv-cuda/stencil/e18_box_split.cu": "Details",
         "opencv-cuda/stencil/e18_stencil.cu": "Details", "opencv-cuda/stencil/e19_batched_box.cu": "FklDetails",
         "opencv-cuda/stencil/e19_crop_sweep.cu": "FklDetails"}
LIMIT = 150


def split_top(s):
    """the top-level comma-separated parts of s"""
    depth, out, cur = 0, [], ""
    for c in s:
        if c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        if c == "," and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += c
    out.append(cur.strip())
    return out


def call(ind, head, args, tail=";"):
    """ind + head + '(' + args joined + ')' + tail, wrapped at LIMIT after top-level commas"""
    parts = args
    lines, cur = [], ind + head + "("
    for i, p in enumerate(parts):
        piece = p + ("," if i < len(parts) - 1 else ")" + tail)
        sep = "" if cur.endswith("(") else " "
        if len(cur) + len(sep) + len(piece) > LIMIT and not cur.endswith("("):
            lines.append(cur)
            cur = ind + "    " + piece
        else:
            cur += sep + piece
    lines.append(cur)
    return "\n".join(lines)


def norm_args(a):
    return split_top(re.sub(r"\s*\n\s*", " ", a))


for rel, D in FILES.items():
    path = f"{W}/{rel}"
    s = open(path).read()
    n = {}
    lf = f"fk::executeOperations<fk::LinearFilterDPP<fk::ParArch::GPU_NVIDIA, {D}>>"
    # A
    pat = re.compile(r"^(?P<ind>[ \t]*)if \(!fk::executeLinearFilter\((?P<args>.*?),\s*fkStream\)\) \{\n(?P=ind)    throw std::runtime_error\(\"[^\"]*\"\);\n(?P=ind)\}\n",
                     re.M | re.S)
    s, n["A"] = pat.subn(lambda m: call(m["ind"], lf, ["fkStream", *norm_args(m["args"])]) + "\n", s)
    # B
    pat = re.compile(r"^(?P<ind>[ \t]*)const bool ok(?:Asym|Half) = fk::executeLinearFilter\((?P<args>.*?),\s*fkStream\);\n", re.M | re.S)
    s, n["B"] = pat.subn(lambda m: call(m["ind"], lf, ["fkStream", *norm_args(m["args"])]) + "\n", s)
    s, n["B-check"] = re.subn(r"^[ \t]*if \(!okAsym \|\| !okHalf\) throw std::runtime_error\(\"[^\"]*\"\);\n", "", s, flags=re.M)
    # C
    pat = re.compile(r"^(?P<ind>[ \t]*)valid = fk::executeLinearFilter\((?P<args>.*?),\s*fkStream\);\n", re.M | re.S)
    s, n["C"] = pat.subn(lambda m: call(m["ind"], lf, ["fkStream", *norm_args(m["args"])]) + "\n", s)
    if n["C"]:
        s, n["C-flag"] = re.subn(r"^[ \t]*bool valid = false;\n", "", s, flags=re.M)
        s, n["C-check"] = re.subn(r"^[ \t]*if \(!valid\) throw std::runtime_error\(\"identity check: the details are not valid\"\);\n", "", s, flags=re.M)
    # D
    pat = re.compile(r"^(?P<ind>[ \t]*)if \(!fk::executeBoxBlur<1>\(details, image, output, fkStream\)\) \{\n(?P=ind)    throw std::runtime_error\(\"[^\"]*\"\);\n(?P=ind)\}\n",
                     re.M)
    s, n["D"] = pat.subn(lambda m: call(m["ind"], "fk::executeOperations<fk::MorphologyDPP<fk::ParArch::GPU_NVIDIA, FklBatchDetails>>",
                                        ["fkStream", "details", "image", "output", "fk::boxBlurReducers<float, 1>(details)"]) + "\n", s)
    left = [l for l in s.splitlines() if re.search(r"fk::executeLinearFilter\(|fk::executeBoxBlur<", l) and not l.lstrip().startswith("//")]
    open(path, "w").write(s)
    print(rel, {k: v for k, v in n.items() if v}, "code calls left:", left)
