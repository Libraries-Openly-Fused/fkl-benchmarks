#!/usr/bin/env bash
# dump_prefix.sh <dump dir>: is every N's dump the first N planes of N = 150's (cmp of the first N x 65,536 values of every file)?
cd "${1:?}" || exit 1
for d in n*; do n=$((10#${d#n})); r=""
  for f in rpp_grey.u8 rpp_box.u8; do cmp -s -n $((n * 65536)) $d/$f n150/$f && r="$r $f=prefix" || r="$r $f=DIFFERENT"; done
  for f in rpp_tensor.f32 fkl_loop_tensor.f32 fkl_batch_tensor.f32; do cmp -s -n $((n * 262144)) $d/$f n150/$f && r="$r $f=prefix" || r="$r $f=DIFFERENT"; done
  echo "$d$r"; done
