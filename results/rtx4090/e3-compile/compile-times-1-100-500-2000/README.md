Compile time of one fused kernel of N distinct operations (variant A: float Mul/Add/Sub/Div/Max/Min, each with its own parameter; fkl-self/compiletime/gen_chain.py),
without and with the get_arg change, measured on 2026-10-30 with scripts/e3_sweep.py: nvcc 13.4.92, -O3 -DNDEBUG -std=c++17 -arch sm_89 (+PTX) -ftemplate-depth=1000 -lineinfo --threads 1, each
compile pinned to ONE core (core 8 without the fix, core 9 with it, at the same time, nothing else compiling; the E0 runner was measuring on other cores), 5 repetitions per point.
without_fix.csv: the library at LTS-C++17 38c4f32 (external/fkl/include). with_fix.csv: the include tree of the local branch LTS-C++17-get-arg-no-tuple (4f71c94): the two trees differ only in
core/utils/parameter_pack_utils.h. The programs are not executed. host_wall_s is the same source built as plain C++ with g++ 15 (CPU back-end), first repetition only.
