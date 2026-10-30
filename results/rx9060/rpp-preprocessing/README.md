# FKL (HIP) against AMD RPP on the pre-processing pipelines: RX 9060 (gfx1200), ROCm 10.1, RPP 3.2.0

`benchmark_rpp_preprocessing_hip` (`rpp/benchmark_rpp_preprocessing.h`, build with `-DFKLB_BUILD_RPP=ON`). RPP is the ROCm package
`amdrocm-rpp10.1` 10.1.0-3 (`librpp.so.3.2.0`), the `rppt_*` tensor API with the HIP back-end. Wall clock of one run including the
synchronisation of the stream, 3 warm-up runs and at least 0.5 s of warm-up, 30 timed runs, median/min/max/mean; idle machine.

- `noise/`: seeded noise as the image (pipeline (a)) and as the NV12 frame (pipeline (b)). `natural-image/`: the Kodak image kodim23 tiled over
  the image / converted to NV12; its timing rows are 5 runs and are not meant, only the agreement.
- Pipeline (a): `benchmark_rpp_preprocessing_a_*.csv`, batch 10 to 300 step 10. Pipeline (b): `..._b_*.csv`, 1, 8, 16, 32, 64 crops.
- Variants: `fkl_fused`, `fkl_perop_batch` (FKL, ASYMMETRIC sampling, the default), `fkl_fused_halfpixel` (FKL fused with `ResizeHalfPixel`, the sampling of RPP, from FKL
  38c4f32), `rpp_fused_api` (RPP: `rppt_resize_mirror_normalize` + `rppt_channel_permute`, two calls; in
  pipeline (b) there is no channel swap: one call), `rpp_convert` (`rppt_yuv_to_rgb`, NV12 8 bit, BT.601, studio swing, default chroma upsampling) and
  `fkl_convert`, `*_chain_*` (the crops, without the conversion; also `fkl_chain_fused_halfpixel`), `*_pipeline_*` (conversion and chain; also `fkl_pipeline_fused_halfpixel`), `mixed_fkl_convert_rpp_chain`.

RPP calls: the crops of one image are a batch with a source batch stride of 0 and one ROI (XYWH) per crop; source `U8 NHWC`, destination `F32 NCHW`;
`mean = sub / 0.3`, `std = div / 0.3` per channel (pipeline (a)), `mean = 255 * mean`, `std = 255 * std` (pipeline (b)), mirror off; the channel
permutation (2, 1, 0) for BGR. RPP has no separate per-channel multiply, subtract and divide: its normalize calls cover the three (the
`rppt_*_scalar` functions take one scalar per image), so there is no one-call-per-operation RPP chain; `rpp_fused_api` is RPP's own pre-fused path.
`rppt_resize_mirror_normalize` changes the array of ROIs in place (XYWH to LTRB): the benchmark writes it again before each call and that is part of the
timed work. One RPP handle per call. Output widths are multiples of 8.

Numerics (`..._agreement_*.csv`, interior and last two columns and rows of the output planes reported separately): RPP follows the HALF_PIXEL
sampling of `common/resize_reference.h` in the interior to 1.9e-6, but it clamps the source coordinate to `size - 2` instead of `size - 1`, so the last
two columns and rows of each plane differ from that reference (e.g. a 60 pixel wide ROI resized to 64: the outputs 62 and 63 read the coordinate 58.0, the
reference 58.09 and 59.0). FKL's `Resize` samples ASYMMETRIC (FKL fb9ac3a), so FKL against RPP differs by the sampling convention; the RGB frames of
`rppt_yuv_to_rgb` and FKL's `ConvertYUVToRGB` differ by at most 1 level (mean 2.8e-6 levels on the noise).

With `ResizeHalfPixel` FKL and RPP use the same sampling and agree in the interior to rounding: pipeline (a), 7.6e-6 (max) on noise and on the Kodak image;
pipeline (b), 0.0095 and 0.0088 (max; one level of the RGB frames, once normalised), mean 1.4e-7, a fraction of 2e-5 of the values over 1e-4. The remaining
difference is in the last two columns and rows of each plane (RPP's clamp to `size - 2`). The HALF_PIXEL variant costs the same as the default (a 0 to 5% difference
in the medians, see the CSVs).
