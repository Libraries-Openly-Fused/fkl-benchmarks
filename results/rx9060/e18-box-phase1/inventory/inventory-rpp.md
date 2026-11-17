PART 2: RPP 3.2.0 (source tag therock-10.1 = the installed 3.2.0; API headers identical). HIP back-end. Paths under rpp-src/projects/rpp/. The key facts were checked by me in the source: rpp_hip_pack, the box constants, the resize sampling and edge clamp, the XYWH->LTRB conversion, the Gaussian weights and the box border.

| function | dataTypes in -> out (HIP) | layouts | ROI | kernel sizes | border | U8 output rounding | weights / sampling |
|---|---|---|---|---|---|---|---|
| rppt_resize | in == out required (-13 otherwise): U8, F16, F32, I8 | NHWC/NCHW; NHWC = 3 channels hard-coded | XYWH or LTRB per image; the ROI IS the crop (no separate crop call needed); XYWH converted to LTRB IN PLACE on the device | - | bilinear taps clamped to [l, r-1] with r inclusive, so the LAST ROI column/row is never read (the "size-2 clamp" of Figure 15); left/top replicate | bilinear/NN: rpp_hip_pack = __builtin_amdgcn_cvt_pk_u8_f32 (hardware V_CVT_PK_U8_F32), no clamp before it: rounding NOT in the source (UNKNOWN). The generic path (bicubic, lanczos, ...): clamp to [0,255] then (uchar), i.e. TRUNCATION | BILINEAR = HALF_PIXEL: src = dst*ratio + (ratio-1)/2 + l, ratio = roiW/dstW, float fmaf chain; NN = roundf on the same coordinate, outside -> 0 |
| rppt_box_filter | in == out: U8, F16, F32, I8 | NHWC 3ch OK; NCHW 1/3ch | XYWH/LTRB per image (LTRB -> XYWH in place) | HIP: 3, 5, 7, 9 only (-2 otherwise; the header's "any >0" is HOST only) | REPLICATE only (-6 otherwise), clamped at the ROI edge; pixels outside the ROI never read | rpp_hip_pack, no clamp: UNKNOWN | U8: integer sums, times the float constant 0.1111111f / 0.04f / 0.02040816f / 0.01234568f (slightly BELOW 1/9 and 1/49). If the pack truncates, a constant region loses 1 level (255*9*0.1111111f = 254.99997); if it rounds to nearest, the result equals round(sum/k^2) exactly (k^2 odd: no ties, nearest boundary 1/18 away) |
| rppt_gaussian_filter | in == out: U8, F16, F32, I8 | as box | as box | 3, 5, 7, 9 | REPLICATE only, ROI edge | rpp_hip_pack: UNKNOWN | one stdDev PER IMAGE (device/pinned memory); an extra kernel builds the weights on EVERY call: w(i,j) = __expf(-(i^2+j^2)/(2 sigma^2)), normalised by 1/sum; full 2D k x k (not separable), fmaf in float. 3x3: W = [[b,a,b],[a,1,a],[b,a,b]] / (1+4a+4b), a = e^(-1/(2 sigma^2)), b = e^(-1/sigma^2). sigma = 1.0: centre 0.2041800, edge 0.1238414, corner 0.0751136 (= outer product of [0.2740686, 0.4518628, 0.2740686]); on the GPU a few ulp off (__expf). sigma = 0 gives NaN |
| rppt_median_filter | in == out: U8, F16, F32, I8 | as box | as box | 3, 5, 7, 9 | REPLICATE only, ROI edge | exact input value (no rounding) | full k x k per channel; 3x3 by min/med3/max of rows, 5x5 sorting network, 7x7/9x9 quickselect |
| rppt_erode / rppt_dilate | U8, F16, F32, I8; HIP does NOT check in == out (a mismatch launches nothing and returns SUCCESS) | as box | as box | 3, 5, 7, 9 | no border argument; replicate clamp at the ROI edge (= "ignore outside" for min/max) | exact input value | full square; erode starts at 255, dilate at 0 (so I8 dilate never gives a negative value) |
| rppt_crop | in == out: U8, F16, F32, I8 | as box | per image, LTRB -> XYWH in place | - | - | exact (unpack/repack) | - |
| rppt_crop_mirror_normalize | U8->U8, F16, F32, I8 same type, and U8->F32, U8->F16 | as box | per image | - | - | U8->U8 clamped to [0,255], then rpp_hip_pack: UNKNOWN | dst = src*mult[n*C+c] + offset[n*C+c] (the arrays need batch x C entries, the header says batch); U8->F32 on raw 0..255, no clamp, no /255 |
| rppt_resize_mirror_normalize | as CMN (U8->F32/F16 included), BILINEAR only | as box | XYWH -> LTRB in place | - | the same r-1 clamp as rppt_resize | U8->U8 clamped then pack: UNKNOWN | (bilinear - mean[n*C+c]) * (1/std[n*C+c]); F32->F32 clamps to [0,1]; every call does a hipMemsetAsync of the whole dst plus a hipStreamSynchronize |

RPP notes:
1. **No user-weight 2D filter and NO F32/F16 -> U8 conversion in any rppt function.**
   - rppt_sobel_filter has fixed kernels (3/5/7), NCHW 1-channel output.
   - rppt_emboss builds its kernel from a strength value of at most 2.
   - The other "filter" names are audio functions.
   - U8 -> F32 is possible through CMN, RMN, rppt_lut (a user 256-entry LUT), rppt_log or rppt_tensor_divide_tensor.
   - So a common pipeline ending in U8 must stay U8 inside RPP: resize U8->U8 (with the crop as its ROI), then a filter U8->U8.
2. **THE OPEN FACT: how V_CVT_PK_U8_F32 rounds** (to nearest even, or truncation; with or without saturation).
   - It decides every RPP U8 result of resize (bilinear), box, Gaussian and CMN/RMN U8.
   - It is not in the source, and RPP has no tests in the tree.
   - A one-kernel test on the RX 9060 settles it: about 1 minute, a few seconds of GPU, nothing timed. I have not run it, because this task allows no GPU use; I can run it on your go.
   - The HOST back-end rounds to nearest even (nearbyintf, cvtps_epi32), so HIP and HOST may differ.
3. **Matching RPP from FKL:**
   - The resize sampling is HALF_PIXEL, but with the last ROI column/row never read. That is an edge difference FKL's Resize does not reproduce: in the last output column(s)/row(s) only (Figure 15: an edge difference up to 126.4 in its F32 output, interior 1.9e-6).
   - Gaussian weights come from __expf on the device, so FKL can match them only to a few ulp.
   - Box U8: if the pack rounds to nearest, RPP equals round(sum/k^2) exactly, so FKL can match it bitwise. If the pack truncates, RPP is a floor that is one level low at every exact multiple of k^2, and it differs from any rounding baseline at about half the values.
   - Median, erode, dilate and crop return exact input values, so they can match bitwise given the same input.
4. **Other behaviours to know:**
   - The ROI buffers are rewritten in place on every call: LTRB->XYWH for filters/crop/CMN, XYWH->LTRB for resize/RMN. A repeated call with the same buffer is safe only if it passes the type the call converts TO: LTRB for resize/RMN, XYWH for filters/crop/CMN. Otherwise the conversion is applied again and the ROI drifts (with x = 0, each XYWH resize call shrinks the width by 1). The Figure 15 program refills the ROIs on every call. The two conversion kernels have no bounds check and write up to 256 entries.
   - Outputs are stored 8 pixels at a time: up to 7 pixels past the ROI width are written.
   - The host entry points discard launch errors and return SUCCESS. Unsupported layout pairs also return SUCCESS without launching anything.
   - Filters/erode/dilate need srcDesc offsetInBytes >= 12*(k/2), else LOW_OFFSET (-3).
   - The batch size is the handle's, not desc->n.
