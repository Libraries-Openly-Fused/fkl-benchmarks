Kernel time of the programs written by fkl-self/compiletime/gen_chain.py (variant A: N distinct float operations in one fused kernel, 1920x1080 float image of
zeros, mean of 20 launches per run, 5 runs per binary; the median is in kernel_time_by_binary.csv), RTX 4090, run in GPU window 9 of 2026-10-30 (the E0 runner held).
tree: current = the library at LTS-C++17 38c4f32; patched = a copy of its include tree with get_arg reimplemented without the tuple (repro/get_arg_patch.diff);
getarg-pr = the include tree of the local branch LTS-C++17-get-arg-no-tuple (4f71c94). raw_window9_kernel_time.txt is the output of the window.
