#!/bin/bash
# C-PM-CUDA: the host-code comparisons of the programs behind the paper's RTX 4090 numbers (CPU only: objdump; nothing runs).
#   main-vs-head:     main 976e42b (FKL b10fc66, the base build) against the pin move 2518c75 (FKL 48fbf3f, the head build), with FKL #435's
#                     rename rule (host_compare435.py)
#   measured-vs-main: the binary that made the record (its sha256 checked against the record) against the base build, without the rename
#                     rule (the same FKL kernel names)
set -u
T=/home/oscar/tfkl-bench; O=$T/scratch/pin-435; W=$T/worktrees
BA=$O/base-b10fc66/bench-build/bin; BB=$O/head-48fbf3f/bench-build/bin
HC="python3 -I $O/host_compare435.py --tools $T/scratch/pr435/analysis --sass-tools $W/fkl-benchmarks-pin-435/scripts"
mkdir -p $O/out/host/main-vs-head $O/out/host/measured-vs-main
echo "# $(date -Iseconds) host comparisons start; host_compare435.py sha256 $(sha256sum < $O/host_compare435.py | cut -c1-64)"
PROGS="e4_preprocessing_cu benchmark_npp_batchresize_x_split3D_cu benchmark_CPU_NPP_vs_FastNPP_cu benchmark_CPU_OpenCV_vs_cvGS_cu e1_vf_chain_cu e1_hf_batch_cu e1_preproc_cu e2_fkl_chain_cu e16_attention_cu e19_crop_sweep_cu"
# the moved error-check lines (moved_lines.py): executors.h for the programs whose source the pin move does not change, and also the
# program's source and the FKL headers of its DPPs for E16 and E19
for p in $PROGS; do
  d=$O/out/host/main-vs-head
  case $p in e16_attention_cu) LP=$(cat $O/out/lp-e16.txt);; e19_crop_sweep_cu) LP=$(cat $O/out/lp-e19.txt);; *) LP=$(cat $O/out/lp-unchanged.txt);; esac
  # the rows of E16 and E19: a timed lambda belongs to the first row whose expression matches what it reaches (E16: the K/V reads and the
  # epilogue name the variant; E19: the library each call lambda calls)
  ROWS=()
  case $p in
    e16_attention_cu) ROWS=(--row-priority --row 'simt_raw=fk::FlashAttentionDPP<' --row 'mma_int8kv_unfused=dequantKV<\d+,false>'
                            --row 'mma_fp8kv_unfused=dequantKV<\d+,true>' --row 'mma_int8kv_fused=Int8TokenDequantRead'
                            --row 'mma_fp8kv_fused=Fp8TokenDequantRead' --row 'mma_epilogue_separate=affineBf16'
                            --row 'mma_epilogue_fused=FlashAttentionMmaDPP<[^ ]*Mul<float' --row 'mma_raw=FlashAttentionMmaDPP<');;
    e19_crop_sweep_cu) ROWS=(--row 'fkl_loop=fk::LinearFilterDPP<' --row 'fkl_batch=fk::MorphologyDPP<' --row 'npp=nppi' --row 'opencv_cuda=cv::cuda::');;
  esac
  ( $HC $BA/$p $BB/$p --line-pairs "$LP" "${ROWS[@]}" --intervals --intervals-csv $d/$p-intervals.csv --out $d/$p.txt --csv $d/$p.csv --diffs $d/$p-diffs.txt > /dev/null; echo "main-vs-head $p exit $?: $(sed -n 5p $d/$p.txt) | $(sed -n 7p $d/$p.txt | sed -E 's/lines [0-9:,]+/lines (given)/') | $(grep '^  TOTAL' $d/$p.txt)" ) &
done
wait
# measured-vs-main: label binary sha256
M="cleanup-window-b:836a98e:e4_preprocessing_cu:$W/fkl-benchmarks-window/build/local-rtx4090/bin/e4_preprocessing_cu:6e64d7e9c36fbfec9b43bad329a2c1c94ccda23c487da218ce3a2647a6f475c8
e4-hand-roi:469aa6d:e4_preprocessing_cu:$W/fkl-benchmarks-window-g/build/local-rtx4090/bin/e4_preprocessing_cu:121a088fd6ad888103246a267f48814b384a6254dae706958c577e7aead0588f
e4-build-once:f534a38:e4_preprocessing_cu:$W/fkl-benchmarks-window-h/build/local-rtx4090/bin/e4_preprocessing_cu:1387fcb9633403085e128e0287f6f47c8d8b03275dbc17c576fb89e5562fb5a3
cleanup-window-a:836a98e:benchmark_npp_batchresize_x_split3D_cu:$W/fkl-benchmarks-window/build/local-rtx4090/bin/benchmark_npp_batchresize_x_split3D_cu:44b06dcf92ffcd7b70ced3ed50458b5a3401f0e5bb3792a8bda4c834433df32b
cleanup-window-a:836a98e:benchmark_CPU_NPP_vs_FastNPP_cu:$W/fkl-benchmarks-window/build/local-rtx4090/bin/benchmark_CPU_NPP_vs_FastNPP_cu:50bf1ed262d39d66310e317ea31348cc37a8fe4357483bde2bf7ee678faad1fa
cleanup-window-a:836a98e:benchmark_CPU_OpenCV_vs_cvGS_cu:$W/fkl-benchmarks-window/build/local-rtx4090/bin/benchmark_CPU_OpenCV_vs_cvGS_cu:90aa59d9e94257d9027df3366f8fd439cdca36fcf11b38c7de79af1313f85e00
cleanup-window-c-e1:1a2535c:e1_vf_chain_cu:$W/fkl-benchmarks-window-c/build/rtx4090-e1/bin/e1_vf_chain_cu:23e09697400335cda7a71e0456adf35f8292c8cd456e621ada8c0292d67932b4
cleanup-window-c-e1:1a2535c:e1_hf_batch_cu:$W/fkl-benchmarks-window-c/build/rtx4090-e1/bin/e1_hf_batch_cu:19e78db076f4a2bbaf420c409695e377471556a1c38c9434e5bfc2218982e0dc
cleanup-window-c-e1:1a2535c:e1_preproc_cu:$W/fkl-benchmarks-window-c/build/rtx4090-e1/bin/e1_preproc_cu:85c031d961c7bb9878b8a21c3ac98cd9a4bdceaef8e9df82ba401d19f577fc68
cleanup-window-e:9a6a88b:e2_fkl_chain_cu:$W/fkl-benchmarks-window-e/build/local-rtx4090/bin/e2_fkl_chain_cu:4603f1a4eca2cb43691e88b666b43e83b90510ffe13e877fb817f670493046e9
cleanup-window-d:f27c605:e16_attention_cu:$W/fkl-benchmarks-window-d/build/local-rtx4090/bin/e16_attention_cu:268c78adf599b94b519c88cb49f91031ca7de486a21358d59600e6062ab18ccc"
echo "$M" | { while IFS=: read rec rev p bin want; do
  got=$(sha256sum < $bin | cut -c1-64)
  [ "$got" = "$want" ] || { echo "measured-vs-main $rec $p: sha256 $got is not the record's $want: skipped"; continue; }
  d=$O/out/host/measured-vs-main; n=$rec-$rev-$p
  # FKL 64ad9195 (the measured builds) against b10fc66: #432's defaulted BinaryType, and the moved error-check lines of FKL and of the program
  LP=$(cat $O/out/lp-fkl-64ad9195-b10fc66.txt)
  case $rev in 836a98e) LP="$LP,$(cat $O/out/lp-common-836a98e.txt)";; esac
  # the base build's source path (scratch/pin-435/base-b10fc66/bench-src) against the measured build's worktree: longer by PD characters
  case $bin in */fkl-benchmarks-window/*) PD=7;; *) PD=5;; esac
  ( $HC $bin $BA/$p --no-rename --drop-binarytype-default --line-pairs "$LP" --path-delta $PD --intervals --intervals-csv $d/$n-intervals.csv --out $d/$n.txt --csv $d/$n.csv --diffs $d/$n-diffs.txt > /dev/null
    echo "measured-vs-main $n exit $?: $(sed -n 5p $d/$n.txt) | $(sed -n 7p $d/$n.txt | sed -E 's/lines [0-9:,]+/lines (given)/') | $(grep '^  TOTAL' $d/$n.txt)" ) &
done; wait; }
echo "# $(date -Iseconds) host comparisons end"
