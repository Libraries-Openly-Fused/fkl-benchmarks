cvt-test (2026-11-13): how V_CVT_PK_U8_F32 (__builtin_amdgcn_cvt_pk_u8_f32, behind RPP's rpp_hip_pack) rounds on the RX 9060 (gfx1200), and what
RPP 3.2.0's rppt_box_filter 3x3 U8 returns. Coordinator's go, 2026-11-13. Nothing timed; no CI condition applies.
- cvt_pk_u8_test.hip: the source (its header says what it does and how it was built). build.log is empty (no warnings with -Wall -Wextra).
- output.txt: the console. exit_code.txt: exit 0. stderr.txt: empty.
- part1_builtin.csv: every input with the GPU result and four host rules.
- part2_box3x3.csv: every value of the single-image (n=1) call: its exact 3x3 sum, RPP's output, round(sum/9), floor(sum/9), the float product and the builtin's prediction.
- sha256.txt: the binary, the source and librpp.so.3.2.0.
Toolchain: amdclang++ 24.0.0 (ROCm 10.1.0), -O2, gfx1200; RPP 3.2.0.
Result: the instruction rounds to nearest, ties to even, saturating to [0, 255]; NaN -> 0, -inf -> 0, +inf -> 255 (58/58 non-NaN inputs agree with RNE,
39 with truncation, 54 with round-half-up/away). rpp_hip_pack's nested form equals the single calls. rppt_box_filter 3x3 U8 = round(sum/9) at all
6,144 values (n=1) and 12,288 values (n=2, both images identical and equal to n=1); 255*9*0.1111111f = 254.99997 -> 255, so constant regions are kept.
