# The first two E4 sweeps: taken while compile jobs ran on the machine (NOT for the paper)

Both sweeps are of the same binary (fkl-benchmarks e9e599d, library c8fcbe8, `e4_preprocessing_cu` without the host-time columns) and ran in a GPU window while compile jobs of the session were running on other cores of the CPU
(`scripts/pin_transfer.sh`: nvcc and g++ processes, up to 12 at once, niced, pinned to cores 8-15,24-31; the benchmark was pinned to cores 0-3,16-19). The two CCDs of the Ryzen 9 5950X share the memory controller and the power budget,
and the versions that are bound by the host (OpenCV-CUDA per crop and in batch, NPP: hundreds of kernel launches per call) are the ones that moved.

- `sweep-a/stdout.txt` (00:17): the console output only; its CSV was overwritten by the profiler runs that followed in the same window.
- `sweep-b/` (00:21): the complete output (CSV, agreement CSV, console).

Same binary, same points: sweep b was 4 to 13 % slower than sweep a for the OpenCV-CUDA and NPP versions (medians over the 30 batches of E4a: `opencv_percrop` +8.6 %, `opencv_batch` +9.1 %, `npp` +10.0 %) and 3 to 4 % slower for
`fkl_fused` and `fkl_fused_half_pixel`. The final E4 numbers are the quiet runs in `../quiet-run1`, `quiet-run2`, `quiet-run3` and their summary in `../e4_summary.csv`.

Against the quiet medians (`../e4_summary.csv`, medians over the 30 batches of E4a): sweep a was slower by 6.7 % (`opencv_percrop`), 3.0 % (`opencv_batch`), 5.0 % (`npp`), 0.9 % (`fkl_fused`) and 1.3 % (`fkl_fused_half_pixel`); sweep b by 16.2 %, 12.6 %, 15.4 %,
5.3 % and 4.6 %.
