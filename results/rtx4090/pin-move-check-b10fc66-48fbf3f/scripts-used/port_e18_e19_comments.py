#!/usr/bin/env python3
"""C-PM-CUDA: the comments of the five E18/E19 CUDA programs that name the calls, and the kernel named in e18_box.cu's attribute report,
moved to FKL #435's names (run after port_e18_e19.py). Each replacement is exact text and must occur at least once.

  port_e18_e19_comments.py <worktree>
"""
import sys

W = sys.argv[1] + "/opencv-cuda/stencil/"
E18_COMMON = [
    ("""// The arguments of one fk::executeLinearFilter, as the kernel-only interval""",
     """// The arguments of one LinearFilterDPP call, as the kernel-only interval"""),
]
EDITS = {
    "e18_box.cu": E18_COMMON + [
        ("""//   fkl_box_asym        FKL, one kernel: LinearFilterDPP (one fk::executeLinearFilter) with the arguments that the body of fk::executeBoxFilter
//                       builds for scale 1, and LinearFilterDPPDetails<float3> with its defaults (a 16 x 8 tile, a largest kernel of 7 x 7):""",
         """//   fkl_box_asym        FKL, one kernel: LinearFilterDPP (one fk::executeOperations<fk::LinearFilterDPP<GPU_NVIDIA, Details>>) with the
//                       arguments that the body of fk::executeBoxFilter built for scale 1 (FKL #435 removed it), and
//                       LinearFilterDPPDetails<float3> with its defaults (a 16 x 8 tile, a largest kernel of 7 x 7):"""),
        ("""//                                      as executeBoxFilter computes it)""",
         """//                                      as executeBoxFilter computed it)"""),
        ("""//                       fk::executeBoxFilter itself does not compile with float3 details at FKL 24478f0: it computes the coefficient as""",
         """//                       fk::executeBoxFilter itself did not compile with float3 details at FKL 24478f0: it computed the coefficient as"""),
        ("""the box that OpenCV-CUDA and NPP compute; one fk::executeLinearFilter, the same""",
         """the box that OpenCV-CUDA and NPP compute; one such call, the same"""),
        ("""built once; every call is fk::executeLinearFilter""",
         """built once; every call is that fk::executeOperations"""),
        ("""//            FKL's read chain of each convention (the image Read IOp above, through fk::executeLinearFilter with the identity weights 0 0 0 / 0 1 0 /
//            0 0 0 as a 3 x 3 fk::Ptr2D<float3>, and the rounding write)""",
         """//            FKL's read chain of each convention (the image Read IOp above, through LinearFilterDPP with the identity weights 0 0 0 / 0 1 0 /
//            0 0 0 as a 3 x 3 fk::Ptr2D<float3>, and the rounding write)"""),
        ("""constexpr float SCALE = 1.f;   // the scale argument of fk::executeBoxFilter: the coefficient is SCALE / (BOX_W * BOX_H)""",
         """constexpr float SCALE = 1.f;   // the scale argument of the removed fk::executeBoxFilter: the coefficient is SCALE / (BOX_W * BOX_H)"""),
        ("""    // instantiation that fk::executeLinearFilter launches for these argument types (linear_filter.h, linearFilterDPPKernel).""",
         """    // instantiation that fk::executeOperations<LinearFilterDPP> launches for these argument types (fk::launchDPP_Kernel, FKL #435)."""),
        ("""            const auto kernel = fk::linearFilterDPPKernel<fk::ParArch::GPU_NVIDIA, Details, decltype(A::reads), decltype(A::multiply),
                                                          decltype(A::accumulate), decltype(A::output)>;""",
         """            const auto kernel = fk::launchDPP_Kernel<fk::LinearFilterDPP<fk::ParArch::GPU_NVIDIA, Details>, Details, decltype(A::reads),
                                                     decltype(A::multiply), decltype(A::accumulate), decltype(A::output)>;"""),
        ("""        // ---- (a) The resize alone. FKL's read chain of each convention, through fk::executeLinearFilter with the identity weights""",
         """        // ---- (a) The resize alone. FKL's read chain of each convention, through LinearFilterDPP with the identity weights"""),
    ],
    "e18_box_split.cu": E18_COMMON + [
        ("""// and the box in one fk::executeLinearFilter) and NPP's two calls are repeated with the code of e18_box.cu, and the resize and the box are timed
// alone: NPP's two calls one by one, and FKL's resize in one fk::executeOperations and FKL's box in one fk::executeLinearFilter.""",
         """// and the box in one fk::executeOperations<LinearFilterDPP>) and NPP's two calls are repeated with the code of e18_box.cu, and the resize and
// the box are timed alone: NPP's two calls one by one, and FKL's resize in one fk::executeOperations<TransformDPP> and FKL's box in one
// fk::executeOperations<LinearFilterDPP>."""),
        ("""e18_box.cu's fkl_box_asym, with its code: one fk::executeLinearFilter, LinearFilterDPPDetails<float3> with its defaults""",
         """e18_box.cu's fkl_box_asym, with its code: one fk::executeOperations<LinearFilterDPP>, LinearFilterDPPDetails<float3> with its defaults"""),
        ("""//   resize   FKL's read chain of each convention (the image Read IOp of the fused kernels, through fk::executeLinearFilter with the identity weights,""",
         """//   resize   FKL's read chain of each convention (the image Read IOp of the fused kernels, through LinearFilterDPP with the identity weights,"""),
        ("""constexpr float SCALE = 1.f;   // the scale argument of fk::executeBoxFilter: the coefficient is SCALE / (BOX_W * BOX_H)""",
         """constexpr float SCALE = 1.f;   // the scale argument of the removed fk::executeBoxFilter: the coefficient is SCALE / (BOX_W * BOX_H)"""),
        ("""        // ---- The resize. FKL's read chain of each convention, through fk::executeLinearFilter with the identity weights (e18_box.cu)""",
         """        // ---- The resize. FKL's read chain of each convention, through LinearFilterDPP with the identity weights (e18_box.cu)"""),
    ],
    "e18_stencil.cu": E18_COMMON + [
        ("""//   fkl_stencil_asym              FKL, one kernel: LinearFilterDPP (fk::executeLinearFilter). Its image Read IOp is the read of the crop view followed by""",
         """//   fkl_stencil_asym              FKL, one kernel: LinearFilterDPP (fk::executeOperations). Its image Read IOp is the read of the crop view followed by"""),
        ("""built once; every call is fk::executeLinearFilter""",
         """built once; every call is that fk::executeOperations"""),
    ],
}
E19 = [
    ("""fkl_batch: its box blur (MorphologyDPP, fk::executeBoxBlur) takes a""",
     """fkl_batch: its box blur (MorphologyDPP, fk::boxBlurReducers) takes a"""),
    ("""//   fkl_loop     per crop, one fk::executeLinearFilter (FKL's linear_filter.h, its GPU_NVIDIA version), one kernel:""",
     """//   fkl_loop     per crop, one fk::executeOperations<fk::LinearFilterDPP<GPU_NVIDIA, FklDetails>> (FKL's linear_filter.h), one kernel:"""),
    ("""//                               fk::executeBoxFilter computes it), as E18's fkl_box_* rows (e18_box.cu, fkl_box_asym)""",
     """//                               the removed fk::executeBoxFilter computed it), as E18's fkl_box_* rows (e18_box.cu, fkl_box_asym)"""),
    ("""//   fkl_batch    per frame, one fk::executeBoxBlur<1> (FKL's morphology.h, MorphologyDPP, its GPU_NVIDIA version), one kernel: one block per""",
     """//   fkl_batch    per frame, one fk::executeOperations<fk::MorphologyDPP<GPU_NVIDIA, FklBatchDetails>> with fk::boxBlurReducers<float, 1>
//                (FKL's morphology.h), one kernel: one block per"""),
    ("""the scale argument of fk::executeBoxFilter: FKL's coefficient is SCALE / (BOX_W * BOX_H), as in E18""",
     """the scale argument of the removed fk::executeBoxFilter: FKL's coefficient is SCALE / (BOX_W * BOX_H), as in E18"""),
    ("""// fkl_batch's reference in double: FKL's pipeline as fk::executeBoxBlur<1> computes it (morphology.h at 24478f0).""",
     """// fkl_batch's reference in double: FKL's pipeline as its box blur computes it (fk::executeBoxBlur<1>, morphology.h at 24478f0; MorphologyDPP
// with fk::boxBlurReducers<float, 1> since FKL #435)."""),
    ("""    // ---- FKL (fkl_loop), at the pin 24478f0: one fk::executeLinearFilter per crop, the whole pipeline of the crop in its one kernel""",
     """    // ---- FKL (fkl_loop): one fk::executeOperations<LinearFilterDPP> per crop (fk::executeLinearFilter until FKL #435), the whole pipeline of
    // the crop in its one kernel"""),
]
EDITS["e19_batched_box.cu"] = E19 + [
    ("""    // ---- FKL (fkl_batch), at the pin 24478f0: one fk::executeBoxBlur<1> per frame, the 30 crops in its one kernel (grid z = 30). It shares""",
     """    // ---- FKL (fkl_batch): one fk::executeOperations<MorphologyDPP> per frame (fk::executeBoxBlur<1> until FKL #435), the 30 crops in its one
    // kernel (grid z = 30). It shares"""),
]
EDITS["e19_crop_sweep.cu"] = E19 + [
    ("""    // ---- FKL (fkl_batch), at the pin 24478f0: one fk::executeBoxBlur<1> per frame, the N crops in its one kernel (grid z = N). It shares""",
     """    // ---- FKL (fkl_batch): one fk::executeOperations<MorphologyDPP> per frame (fk::executeBoxBlur<1> until FKL #435), the N crops in its one
    // kernel (grid z = N). It shares"""),
]
for f, reps in EDITS.items():
    s = open(W + f).read()
    for a, b in reps:
        if a not in s:
            sys.exit(f"{f}: not found: {a[:90]}")
        s = s.replace(a, b)
    open(W + f, "w").write(s)
    print(f, len(reps), "edits")
