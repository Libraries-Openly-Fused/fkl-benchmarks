# E18 box re-time at FKL 304492a (the merge of FKL #435): RT-A2, prepared, window not run yet

E18 (`rpp/e18_box.h`) is the 3 x 3 box filter after a crop and a resize, RPP against FKL, on the RX 9060. Its last record is
`../e18-box-retime-b10fc66/` (#62, 2026-11-24), whose B is FKL b10fc66. The pin move to FKL #435 (#66, merged as af96aef with FKL 304492a)
ported E18's five FKL calls to `fk::executeOperations<fk::LinearFilterDPP<GPU_AMD, D>>`. Oscar ordered this re-time on 2026-11-26,
through the coordinator session. The paper reports no E18 number.

**This folder holds, so far,** the binaries' checks and the window script. Nothing has run on the GPU. It runs in the same window as the E19
crop sweep re-time, after it (`../e19-crop-sweep-retime-304492a/launch.sh`).

Decision: Oscar's re-time order of 2026-11-26 (RT-A1 and RT-A2), relayed by the coordinator session that day.

## The two binaries

| | sources | e18_box_hip sha256 |
|---|---|---|
| **A** | fkl-benchmarks 976e42b, FKL b10fc66: #67's base build (a git archive; records fkl-benchmarks `unknown`) | `f32f98e2f84b22fc599bdbe89cabcc957c3ed6df94948e2bf93a053bb6d584ff`, as in #67's `binaries-sha256.txt` |
| **B** | fkl-benchmarks main 73a9650, FKL 304492a: the clean-worktree build of `../e19-crop-sweep-retime-304492a/step0/` | `93dd323e7debcf7d7d8a56d12987b00d28c8936714b4f5c6798e50c774864097` |

B's four build commands of `bin/e18_box_hip` and its CMake cache equal A's, with the folders replaced.

## The code (compile only; `code-checks/`, from `../e19-crop-sweep-retime-304492a/step0/code_checks.sh`)

- **A against #62's B** (3007b72e..., 2160c70 with FKL b10fc66, the binary that `../e18-box-retime-b10fc66/` timed as B): device code 6 of
  6 functions identical, and host code 128 of 128. So A repeats #62's B.
- **B against #67's head build** (a4b493c with FKL 48fbf3f): host code 128 of 129 functions identical, the exception being
  `fklb::writeSystemInfoHeader` (the build strings). The device code is identical: `../e19-crop-sweep-retime-304492a/code-checks/`, 6 of 6.
- **A against B:** the device code is identical (6 of 6 with the task C rename, #67). The host code differs in the FKL rows' calls:
  - `launch()` and the lambdas of the FKL rows;
  - FKL's three `executeLinearFilter` instantiations against three `executeOperations<LinearFilterDPP<GPU_AMD, ...>>`, plus
    `requireArguments`;
  - `writeSystemInfoHeader`.

## The window (`window.sh`)

It is #62's `window.sh`, with the guards of RT-A1 added:
- B must also carry fkl-benchmarks 73a9650;
- before every run, the GPU-process list (`/sys/class/kfd/kfd/proc`) must be empty.

The protocol is unchanged.
- Every run is complete, in its own process, with no environment variable. It runs the checks, then the 2 s clock warm-up, then `rpp`,
  `rpp_ltrb`, `fkl_box_halfpixel` and `fkl_half_once`, each whole_call then kernel_only.
- The order: `unused-A`, `unused-B`, `quiet-run1-A`, `-B`, `quiet-run2-A`, `-B`, `quiet-run3-A`, `-B`, `rocprofv3-A`, `rocprofv3-B`.
- The first failure ends the window.

`diff ../e18-box-retime-b10fc66/window.sh window.sh` shows only those additions and the names. The guard-only run and the mock test are
in `../e19-crop-sweep-retime-304492a/`.

## After the window

The summary will follow #62's, without the GPU: old (#62's B) / A / B, the spread, B against A and A against old, exceeds, and a flag above
3 %, with the check rows compared with #62's.
