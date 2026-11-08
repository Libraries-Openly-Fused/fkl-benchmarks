#!/bin/bash
# Before window D: untimed GPU work, nothing of it is a measurement.
# (a) The E16 input files. The cleaned e16_attention_cu of step 0, with E16_PROFILE=1 (one call per variant, no timing) and E16_INPUT_DIR=e16-work/inputs, writes
#     the files of the published configurations (D 64 and 128, S 512 to 8192, BH = 262144 / S; the mask does not change the inputs): 10 files.
#     - Their sizes and SHA-256 are recorded here (sha256sum) and checked against the sizes and hashes in the notes of the program's CSV.
#     - The reference e16_attention_cu, which makes the same bytes with its device generator, runs the same profile pass; the FKL error columns of the two
#       (max_abs_err, mean_abs_err of every variant and configuration) must be equal.
# (b) The k = 5002 warm-up: one untimed run of the cleaned Python driver at k = 5002, as the window runs it (here with the outputs and the CSV of window C's
#     cleaned FKL run 1). It warms the file cache and the Python imports: every (version, k) gets fresh Triton and Inductor caches. Its CSV is not reported.
# Pinned to cores 0-3,16-19 as the window. Every check stops the script. Output: scratch/cleanup-window-d-pre/.
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; WD=$T/worktrees/fkl-benchmarks-window-d; WC=$T/worktrees/fkl-benchmarks-window-c
STEP0=$T/scratch/cleanup-window-d-build; O=$T/scratch/cleanup-window-d-pre; INPUTS=$T/e16-work/inputs
REF_E16=$FB/build/local-rtx4090/bin/e16_attention_cu; NEW_E16=$WD/build/local-rtx4090/bin/e16_attention_cu
INPUT=$T/e2-work/e2_input.u8; RC=$FB/results/rtx4090/cleanup-window-c; PY=$T/venv-torch/bin/python
fail() { echo "# $(date -Iseconds) STOP: $*"; exit 1; }
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
refuse_blocked_binaries $REF_E16 $NEW_E16 || exit 1
grep -q 'all checks passed' $STEP0/console.txt 2>/dev/null || { echo "step 0 of window D has not passed: not starting"; exit 1; }
sha256sum -c --quiet $STEP0/binary-sha256.txt || { echo "an E16 binary differs from step 0's record: not starting"; exit 1; }
[ "$(sha256sum < $INPUT | cut -d' ' -f1)" = "726b9ef38a8277d6021009d344446bf5915fe75f746d07950e608f63614a8995" ] || { echo "the E2 input is not the published one: not starting"; exit 1; }
[ -e $O ] && { echo "$O exists: not starting"; exit 1; }
[ -e $INPUTS ] && { echo "$INPUTS exists: not starting (nothing is overwritten)"; exit 1; }
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
gpuapps=$(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E ', /usr/')
[ -n "$gpuapps" ] && { echo "a program uses the RTX 4090: not starting"; echo "$gpuapps"; exit 1; }
mkdir -p $O $INPUTS
exec > >(tee -a $O/console.txt) 2>&1
cp "$0" $O/pre-script.sh
echo "# $(date -Iseconds) pre-window steps of window D start (untimed; no measurement)"
GENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
# (a) the input files, written by the cleaned program's profile pass
d=$O/e16-profile-cleaned; mkdir -p $d; start=$(date +%s)
( cd $d && env $GENV E16_PROFILE=1 E16_INPUT_DIR=$INPUTS timeout 1800 taskset -c 0-3,16-19 $NEW_E16 > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
echo "# $(date -Iseconds) cleaned profile pass (writes the inputs): exit $(cat $d/exit_code.txt) after $(( $(date +%s) - start )) s"
[ "$(cat $d/exit_code.txt)" = 0 ] || fail "the cleaned profile pass failed"
( cd $INPUTS && sha256sum e16_input_*.bin ) > $O/inputs-sha256.txt
( cd $INPUTS && stat -c '%s %n' e16_input_*.bin ) > $O/inputs-sizes.txt
cat $O/inputs-sizes.txt; echo "total bytes: $(awk '{s += $1} END {print s}' $O/inputs-sizes.txt)"
python3 - "$O" "$d/e16_attention.csv" <<'EOF' || fail "the input files do not match the expected names, sizes or the hashes in the CSV notes"
import re, sys
O, csv_path = sys.argv[1], sys.argv[2]
sizes = {l.split()[1]: int(l.split()[0]) for l in open(f"{O}/inputs-sizes.txt")}
hashes = {l.split()[1]: l.split()[0] for l in open(f"{O}/inputs-sha256.txt")}
expected = {}
for d in (64, 128):
    for s in (512, 1024, 2048, 4096, 8192):
        bh = 262144 // s
        expected[f"e16_input_d{d}_s{s}_bh{bh}.bin"] = 10 * bh * s * d + 16 * bh * s
notes = dict((n, (int(b), h)) for n, b, h in re.findall(r"(e16_input_d\d+_s\d+_bh\d+\.bin) (\d+) bytes sha256 ([0-9a-f]{64})", open(csv_path).read()))
ok = set(sizes) == set(expected) and all(sizes[n] == expected[n] for n in expected) and set(notes) == set(expected) \
     and all(notes[n] == (sizes[n], hashes[n]) for n in expected)
print(f"input files: {len(sizes)} (expected {len(expected)}); sizes as 10 n + 16 BH S and as in the CSV notes, hashes as in the CSV notes: {ok}")
sys.exit(0 if ok else 1)
EOF
# the reference program's profile pass (its own device generator), for the error columns
d=$O/e16-profile-reference; mkdir -p $d; start=$(date +%s)
( cd $d && env $GENV E16_PROFILE=1 timeout 1800 taskset -c 0-3,16-19 $REF_E16 > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
echo "# $(date -Iseconds) reference profile pass: exit $(cat $d/exit_code.txt) after $(( $(date +%s) - start )) s"
[ "$(cat $d/exit_code.txt)" = 0 ] || fail "the reference profile pass failed"
python3 - "$O/e16-profile-reference/e16_attention.csv" "$O/e16-profile-cleaned/e16_attention.csv" <<'EOF' || fail "the FKL error columns differ from the reference's"
import csv, sys
def rows(p):
    return {(r["variant"], r["d"], r["seq"], r["bh"], r["causal"]): (r["max_abs_err"], r["mean_abs_err"])
            for r in csv.DictReader(l for l in open(p) if l.strip() and not l.startswith("#"))}
ref, new = rows(sys.argv[1]), rows(sys.argv[2])
same = sum(1 for k in ref if new.get(k) == ref[k])
print(f"FKL error columns equal to the reference's in {same} of {len(ref)} rows (cleaned rows: {len(new)})")
sys.exit(0 if same == len(ref) == len(new) and len(ref) > 0 else 1)
EOF
# (b) the k = 5002 warm-up of the cleaned Python driver
d=$O/e2-python-warmup; mkdir -p $d; start=$(date +%s)
( cd $d && source $FB/scripts/triton_env.sh > /dev/null 2>&1 && env $GENV timeout 3600 taskset -c 0-3,16-19 $PY $WC/triton/e2_chain.py --ks 5002 --out e2_chain.csv --input $INPUT \
    --fkl-dir $T/e2-work-cleanup/cleaned-run1 --fkl-csv $RC/cleaned/e2-fkl/run1/e2_fkl_chain.csv > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
echo "# $(date -Iseconds) k = 5002 warm-up of the Python driver: exit $(cat $d/exit_code.txt) after $(( $(date +%s) - start )) s; rows ok $(grep -c ',ok,' $d/e2_chain.csv 2>/dev/null)"
[ "$(cat $d/exit_code.txt)" = 0 ] || fail "the warm-up run failed"
echo "# $(date -Iseconds) pre-window steps ended: all checks passed"
