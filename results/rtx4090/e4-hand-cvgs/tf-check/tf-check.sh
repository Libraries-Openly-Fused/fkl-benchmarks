#!/bin/bash
# Thread-fusion check of cvgs_fused, compile only (no GPU work): tfcheck.cu built twice with the nvcc command line of e4_preprocessing_cu (its defines, its
# include response file, its flags) from the build of branch e4-hand-cvgs at e7a85d4: once with the default cvGS::executeOperations (thread fusion enabled), once
# with cvGS::executeOperations<false>. Then the SASS (scripts/sass_compare_builds.py, normalised per kernel):
#   - default against <false>, matched by demangled name without the (fk::TF) argument: the instruction streams must be identical;
#   - default against e4_preprocessing_cu of that build: the kernels of the default call are those of cvgs_fused (same names, same SASS);
# plus the instruction count of each kernel and the sha256 of each normalised listing.
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; W=$T/worktrees/fkl-benchmarks-e4-series; B=$W/build/local-rtx4090; X=$T/scratch/tf-check; O=$X/out
fail() { echo "# $(date -Iseconds) STOP: $*"; exit 1; }
[ -e $O ] && { echo "$O exists: not starting"; exit 1; }
mkdir -p $O; exec > >(tee -a $O/console.txt) 2>&1
export TMPDIR=$O/tmp; mkdir -p $TMPDIR
echo "# $(date -Iseconds) thread-fusion check starts (compile only)"
cp "$0" $X/tfcheck.cu $O/
[ "$(git -C $W rev-parse HEAD)" = e7a85d4e3ca23240381f0a317750cba29178335f ] || fail "the worktree is not at e7a85d4"
D=$B/opencv-cuda/CMakeFiles/e4_preprocessing_cu.dir
defs=$(sed -n 's/^CUDA_DEFINES = //p' $D/flags.make); flags=$(sed -n 's/^CUDA_FLAGS = //p' $D/flags.make)
echo "defines: $defs"; echo "flags: $flags"; echo "includes: $D/includes_CUDA.rsp"
cp $D/includes_CUDA.rsp $O/includes_CUDA.rsp
mkdir -p $O/default $O/tf-false $O/e4
for v in default tf-false; do
  sw=0; [ $v = tf-false ] && sw=1
  echo "# $(date -Iseconds) build $v (TF_CHECK_EXECUTE_FALSE=$sw)"
  ( cd $B/opencv-cuda && eval nvcc -forward-unknown-to-host-compiler $defs --options-file $D/includes_CUDA.rsp $flags -DTF_CHECK_EXECUTE_FALSE=$sw \
      -x cu -c $X/tfcheck.cu -o $O/$v/tfcheck_cu ) > $O/build-$v.log 2>&1 || fail "build $v (build-$v.log)"
done
echo "# $(date -Iseconds) builds ended"
cp $B/bin/e4_preprocessing_cu $O/e4/tfcheck_cu
S=$W/scripts/sass_compare_builds.py
r=$(python3 $S $O/default $O/tf-false --arch sm_89 --glob 'tfcheck_cu' --match-demangled --ignore '\(fk::TF\)[01]' --out $O/sass-default-vs-tf-false.txt | grep '^kernels identical')
echo "default vs <false>, without the TF argument: $r"
[ "$r" = "kernels identical: 3, different: 0, only in A: 0, only in B: 0" ] || echo "NOT IDENTICAL: see sass-default-vs-tf-false.txt"
r2=$(python3 $S $O/e4 $O/default --arch sm_89 --glob 'tfcheck_cu' --match-demangled --out $O/sass-e4-vs-default.txt | grep '^kernels identical')
echo "e4_preprocessing_cu vs default: $r2"
echo "$r2" | grep -q -E '^kernels identical: 3, different: 0, only in A: [0-9]+, only in B: 0$' || echo "THE DEFAULT KERNELS ARE NOT cvgs_fused's: see sass-e4-vs-default.txt"
python3 - "$W/scripts" "$O" <<'EOF'
import hashlib, os, subprocess, sys
sys.path.insert(0, sys.argv[1])
import sass_tools as st
out = sys.argv[2]
rows = []
for v in ("default", "tf-false"):
    res = subprocess.run(["cuobjdump", "-arch", "sm_89", "-sass", os.path.join(out, v, "tfcheck_cu")], capture_output=True, text=True, check=True)
    funcs = st.split_functions(res.stdout)
    dem = subprocess.run(["cu++filt"], input="\n".join(funcs), capture_output=True, text=True).stdout.splitlines()
    for (name, body), d in zip(funcs.items(), dem):
        lines = st.normalise_sass(body, rename=False)
        instr, _ = st.classify(lines)
        import re
        found = sorted({m for m in re.findall(r"Lm(\d+)E", name)} & {"10", "150", "300"}, key=int)   # the unsigned long template arguments of the mangled name
        batch = found[0] if len(found) == 1 else "?"
        tf = "ENABLED" if "(fk::TF)1" in d.replace(" ", "") else ("DISABLED" if "(fk::TF)0" in d.replace(" ", "") else "?")
        text = "\n".join(lines) + "\n"
        fn = f"sass-{v}-batch{batch}.txt"
        open(os.path.join(out, fn), "w").write(text)
        rows.append((v, batch, tf, instr, len(lines), hashlib.sha256(text.encode()).hexdigest(), fn, name))
with open(os.path.join(out, "kernels.tsv"), "w") as f:
    f.write("build\tbatch\tTF\tinstructions\tnormalised_lines\tsha256_of_normalised_listing\tlisting\tmangled_name\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
for r in rows:
    print(f"{r[0]:9s} batch {r[1]:>3s} TF::{r[2]:8s} {r[3]:5d} instructions  sha256 {r[5][:16]}  {r[6]}")
EOF
for b in 10 150 300; do
  if cmp -s $O/sass-default-batch$b.txt $O/sass-tf-false-batch$b.txt; then echo "batch $b: the normalised listings are byte-identical"; else echo "batch $b: DIFFERENT"; diff $O/sass-default-batch$b.txt $O/sass-tf-false-batch$b.txt | head -20; fi
done
rmdir $TMPDIR 2>/dev/null
echo "# $(date -Iseconds) thread-fusion check ended"
