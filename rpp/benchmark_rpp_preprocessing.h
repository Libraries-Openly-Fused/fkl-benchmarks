/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// FKL on HIP against AMD's RPP (Radeon Performance Primitives, the rppt tensor API with the HIP back-end) on the two pre-processing
// pipelines of the paper, on an AMD GPU. RPP is the counterpart of NPP.
//
// Pipeline (a): the batched DNN pre-processing of cvGPUSpeedup test_batchresize_x_split3D. A 3840x2160 8UC3 image in device memory,
// BATCH crops of 60x120 at (i, i), bilinear resize to 64x128, multiply by 0.3, RGB to BGR, subtract {1, 4, 3.2}, divide by
// {3.2, 0.6, 11.8}, an NCHW float tensor. Batches 10 to 300 step 10.
// Pipeline (b): a 3840x2160 NV12 frame converted to RGB (BT.601 studio swing), then N = 1, 8, 16, 32, 64 detection crops resized
// to 64x128, (x / 255 - mean) / std, an NCHW float tensor (fkl-self/pipelines/nv12_pipeline.h).
//
// Variants (wall clock including the synchronisation of the stream; 3 warm-up runs and at least 0.5 s of warm-up, E5_ITERS (30) timed
// runs; median, min, max, mean):
//   fkl_fused           FKL, one executeOperations for the whole chain (pipeline (b): after the conversion kernel)
//   fkl_fused_halfpixel the same with ResizeHalfPixel (FKL ResizeSampling::HALF_PIXEL: the sampling of cv::resize, NPP and RPP)
//   fkl_perop_batch     FKL, one executeOperations per operation for all the crops
//   rpp_fused_api       RPP, its own fused function: rppt_resize_mirror_normalize (ROI per crop of one image, 8-bit input, float NCHW
//                       output, per channel mean and standard deviation) and, in pipeline (a), rppt_channel_permute (RGB to BGR): two
//                       calls (pipeline (b) has no channel swap: one call)
// RPP has no separate per-channel multiply, subtract and divide: the normalize of rppt_resize_mirror_normalize covers the three, with
// mean = sub / 0.3 and std = div / 0.3. The crops of one image are a batch with a source batch stride of 0 and one ROI per crop.
// Output widths are multiples of 8.
// Environment: E5_ITERS, E5_WARMUP, E5_IMAGE (a binary PPM tiled over the frame / image instead of seeded noise), E5_IMAGE_INFO.
// CSVs in the working directory: benchmark_rpp_preprocessing_{a,b}_GPU_AMD.csv and ..._agreement_GPU_AMD.csv.
#include <common/csv_writer.h>
#include <common/main.h>
#include <fkl-self/pipelines/nv12_pipeline.h>

#include <hip/hip_runtime.h>
#include <rpp.h>
#include <rppdefs.h>
#include <rpp_version.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace rppbench {

using nv12::F3;
using nv12::U3;

constexpr int SRC_W = 3840, SRC_H = 2160, CROP_W = 60, CROP_H = 120, OUT_W = 64, OUT_H = 128, PLANE = OUT_W * OUT_H;
constexpr double TOLERANCE = 1e-4;
constexpr float ALPHA = 0.3f;
constexpr float SUB[3] = {1.f, 4.f, 3.2f}, DIV[3] = {3.2f, 0.6f, 11.8f};

inline int envInt(const char* name, const int fallback) {
    const char* value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

#define RPPB_CHECK_HIP(x) do { const hipError_t e_ = (x); if (e_ != hipSuccess) throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(e_)); } while (0)
#define RPPB_CHECK_RPP(x) do { const int s_ = static_cast<int>(x); if (s_ != 0) throw std::runtime_error("RPP error " + std::to_string(s_) + " in " #x); } while (0)

template <typename T>
class Pinned {
    T* m_data{nullptr};
public:
    explicit Pinned(const size_t n) { RPPB_CHECK_HIP(hipHostMalloc(reinterpret_cast<void**>(&m_data), n * sizeof(T))); }
    ~Pinned() { (void)hipHostFree(m_data); }
    Pinned(const Pinned&) = delete;
    Pinned& operator=(const Pinned&) = delete;
    T* get() const { return m_data; }
    T& operator[](const size_t i) const { return m_data[i]; }
};

struct Region {
    double maxAbs{0}, meanAbs{0}, fractionOver{0};
    size_t count{0};
};

// max, mean and fraction over TOLERANCE of |a - b| on the values of an NCHW tensor of N images, in the interior of each plane or in its last two columns and rows
inline Region compareTensors(const float* a, const float* b, const int n, const int w, const int h, const bool edge) {
    Region r;
    double sum = 0;
    size_t over = 0;
    for (int i = 0; i < n * 3; ++i) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const bool isEdge = x >= w - 2 || y >= h - 2;
                if (isEdge != edge) {
                    continue;
                }
                const size_t k = (static_cast<size_t>(i) * h + y) * w + x;
                const double d = std::fabs(static_cast<double>(a[k]) - static_cast<double>(b[k]));
                if (!(d <= r.maxAbs)) {
                    r.maxAbs = d;
                }
                sum += d;
                over += !(d <= TOLERANCE);
                ++r.count;
            }
        }
    }
    r.meanAbs = r.count ? sum / static_cast<double>(r.count) : 0;
    r.fractionOver = r.count ? static_cast<double>(over) / static_cast<double>(r.count) : 0;
    return r;
}

// RPP: its own fused function for crops of one 8-bit RGB image: rppt_resize_mirror_normalize (resize inside the ROI of each crop,
// U8 -> F32, NHWC -> NCHW, (x - mean) / std per channel) and rppt_channel_permute (RGB -> BGR). The image is in device memory, srcPitch
// bytes between rows; the crops are given as x, y, width, height. The array of ROIs that rppt_resize_mirror_normalize gets is changed by
// the call (XYWH to LTRB in place), so a copy of it is written before each call and that copy is part of the timed work.
class RppFusedApi {
public:
    RppFusedApi(const hipStream_t stream, const void* srcDevice, const size_t srcPitch, const int srcW, const int srcH,
                const std::vector<std::array<int, 4>>& crops, const float alpha, const float (&sub)[3], const float (&div)[3], const bool bgr)
        : bgr_(bgr), n_(static_cast<int>(crops.size())), srcDevice_(const_cast<void*>(srcDevice)), sizes_(n_), roi_(n_), mean_(3 * n_), std_(3 * n_),
          perm_(3 * n_), mirror_(n_), roiMaster_(n_), srcDesc_(n_, 3, srcH, srcW, U8, NHWC), dstDesc_(n_, 3, OUT_H, OUT_W, F32, NCHW) {
        // One handle per call: the parameters of a call are copied asynchronously to buffers of its handle, which two calls on one handle
        // would overwrite while the first kernel may still read them
        RPPB_CHECK_RPP(rppCreate(&handle_, n_, 0, stream, RppBackend::RPP_HIP_BACKEND));
        RPPB_CHECK_RPP(rppCreate(&permuteHandle_, n_, 0, stream, RppBackend::RPP_HIP_BACKEND));
        srcDesc_.strides.nStride = 0; // every crop is a region of the same image
        srcDesc_.strides.hStride = static_cast<Rpp32u>(srcPitch);
        const size_t bytes = static_cast<size_t>(n_) * PLANE * 3 * sizeof(float);
        RPPB_CHECK_HIP(hipMalloc(&resized_, bytes));
        RPPB_CHECK_HIP(hipMalloc(&out_, bytes));
        for (int i = 0; i < n_; ++i) {
            sizes_[i].width = OUT_W;
            sizes_[i].height = OUT_H;
            roi_[i].xywhROI = {{crops[i][0], crops[i][1]}, crops[i][2], crops[i][3]};
            roiMaster_[i] = roi_[i];
            mirror_[i] = 0;
            for (int c = 0; c < 3; ++c) {
                // With bgr, the output channel 2 - c is (alpha * v_c - sub[2 - c]) / div[2 - c] = (v_c - sub / alpha) / (div / alpha); without, the
                // output channel c is (alpha * v_c - sub[c]) / div[c]
                const int k = bgr ? 2 - c : c;
                mean_[i * 3 + c] = sub[k] / alpha;
                std_[i * 3 + c] = div[k] / alpha;
                perm_[i * 3 + c] = 2 - c;
            }
        }
    }
    ~RppFusedApi() {
        (void)hipFree(resized_);
        (void)hipFree(out_);
        (void)rppDestroy(handle_, RppBackend::RPP_HIP_BACKEND);
        (void)rppDestroy(permuteHandle_, RppBackend::RPP_HIP_BACKEND);
    }
    RppFusedApi(const RppFusedApi&) = delete;
    RppFusedApi& operator=(const RppFusedApi&) = delete;

    void resizeNormalize() {
        // rppt_resize_mirror_normalize converts the ROIs from XYWH to LTRB in place: they are written again before each call
        std::memcpy(roi_.get(), roiMaster_.data(), roiMaster_.size() * sizeof(RpptROI));
        RPPB_CHECK_RPP(rppt_resize_mirror_normalize(srcDevice_, &srcDesc_, resized_, &dstDesc_, sizes_.get(), RpptInterpolationType::BILINEAR,
                                                    mean_.get(), std_.get(), mirror_.get(), roi_.get(), RpptRoiType::XYWH, handle_,
                                                    RppBackend::RPP_HIP_BACKEND));
    }
    void permute() {
        RPPB_CHECK_RPP(rppt_channel_permute(resized_, &dstDesc_, out_, &dstDesc_, perm_.get(), permuteHandle_, RppBackend::RPP_HIP_BACKEND));
    }
    void run() {
        resizeNormalize();
        if (bgr_) {
            permute();
        }
    }
    // The BGR NCHW float tensor, copied to the host
    std::vector<float> download() const {
        std::vector<float> host(static_cast<size_t>(n_) * PLANE * 3);
        RPPB_CHECK_HIP(hipDeviceSynchronize());
        RPPB_CHECK_HIP(hipMemcpy(host.data(), bgr_ ? out_ : resized_, host.size() * sizeof(float), hipMemcpyDeviceToHost));
        return host;
    }

private:
    bool bgr_;
    int n_;
    void* srcDevice_;
    rppHandle_t handle_{}, permuteHandle_{};
    void *resized_{nullptr}, *out_{nullptr};
    Pinned<RpptImagePatch> sizes_;
    Pinned<RpptROI> roi_;
    Pinned<Rpp32f> mean_, std_;
    Pinned<Rpp32u> perm_, mirror_;
    std::vector<RpptROI> roiMaster_;
    RpptDesc srcDesc_, dstDesc_;
};

// ------------------------------------------------------------------------------------------------------------------------------
// Pipeline (a)

inline fk::RawPtr<fk::ND::_2D, U3> cropRoi(const fk::RawPtr<fk::ND::_2D, U3>& full, const int x, const int y, const int w, const int h) {
    return nv12::roiOf(full, fk::Rect(static_cast<uint>(x), static_cast<uint>(y), w, h));
}

// FKL: the chain of pipeline (a) fused and one execution per operation, on ParArch::GPU_AMD
template <int N, typename ResizeT = nv12::LinearResize>
class FklA {
public:
    using Dpp = fk::TransformDPP<fk::ParArch::GPU_AMD>;
    FklA(fk::Stream_<fk::ParArch::GPU_AMD>& stream, fk::Ptr2D<U3>& source)
        : stream_(stream), fusedOut_(OUT_W, OUT_H, N, 3), batchOut_(OUT_W, OUT_H, N, 3), tmpA_(OUT_W, OUT_H, N, 1), tmpB_(OUT_W, OUT_H, N, 1),
          resize_(buildResize(source)), color_(nv12ColorOp()), mul_(fk::Mul<F3>::build(F3{ALPHA, ALPHA, ALPHA})),
          sub_(fk::Sub<F3>::build(F3{SUB[0], SUB[1], SUB[2]})), div_(fk::Div<F3>::build(F3{DIV[0], DIV[1], DIV[2]})),
          splitFused_(nv12::split3D(fusedOut_.ptr())), splitBatch_(nv12::split3D(batchOut_.ptr())), readA_(nv12::read3D(tmpA_.ptr())),
          readB_(nv12::read3D(tmpB_.ptr())), writeA_(nv12::write3D(tmpA_.ptr())), writeB_(nv12::write3D(tmpB_.ptr())) {}
    void fused() { fk::executeOperations<Dpp>(stream_, resize_, color_, mul_, sub_, div_, splitFused_); }
    void perOpBatch() {
        fk::executeOperations<Dpp>(stream_, resize_, writeA_);
        fk::executeOperations<Dpp>(stream_, readA_, mul_, writeB_);
        fk::executeOperations<Dpp>(stream_, readB_, color_, writeA_);
        fk::executeOperations<Dpp>(stream_, readA_, sub_, writeB_);
        fk::executeOperations<Dpp>(stream_, readB_, div_, writeA_);
        fk::executeOperations<Dpp>(stream_, readA_, splitBatch_);
    }
    void sync() { stream_.sync(); }
    void download() {
        fusedOut_.download(stream_);
        batchOut_.download(stream_);
    }
    const float* fusedOut() const { return nv12::hostRaw(fusedOut_).data; }
    const float* batchOut() const { return nv12::hostRaw(batchOut_).data; }

private:
    static auto nv12ColorOp() { return fk::ColorConversion<fk::ColorConversionCodes::COLOR_RGB2BGR, F3, F3>::build(); }
    static auto buildResize(fk::Ptr2D<U3>& source) {
        std::array<fk::RawPtr<fk::ND::_2D, U3>, N> rois;
        for (int i = 0; i < N; ++i) {
            rois[i] = cropRoi(source.ptr(), i, i, CROP_W, CROP_H);
        }
        return ResizeT::build(fk::PerThreadRead<fk::ND::_2D, U3>::build_batch(rois), fk::make_set_std_array<N>(fk::Size(OUT_W, OUT_H)));
    }
    fk::Stream_<fk::ParArch::GPU_AMD>& stream_;
    fk::Tensor<float> fusedOut_, batchOut_;
    fk::Tensor<F3> tmpA_, tmpB_;
    decltype(buildResize(std::declval<fk::Ptr2D<U3>&>())) resize_;
    decltype(nv12ColorOp()) color_;
    decltype(fk::Mul<F3>::build(F3{})) mul_;
    decltype(fk::Sub<F3>::build(F3{})) sub_;
    decltype(fk::Div<F3>::build(F3{})) div_;
    nv12::SplitOp splitFused_, splitBatch_;
    nv12::Read3DOp readA_, readB_;
    nv12::Write3DOp writeA_, writeB_;
};

// Scalar reference of pipeline (a) in double precision, from the host copy of the source, with the sampling of a resize
inline std::vector<float> referenceA(const fk::RawPtr<fk::ND::_2D, U3>& hostSource, const int n, const fklb::ResizeSampling sampling) {
    std::vector<float> out(static_cast<size_t>(n) * 3 * PLANE);
    std::vector<double> resized(static_cast<size_t>(PLANE) * 3);
    for (int i = 0; i < n; ++i) {
        const auto roi = cropRoi(hostSource, i, i, CROP_W, CROP_H);
        fklb::resizeBilinear(reinterpret_cast<const uchar*>(roi.data), roi.dims.pitch, CROP_W, CROP_H, 3, OUT_W, OUT_H, sampling, resized.data());
        for (int p = 0; p < PLANE; ++p) {
            for (int c = 0; c < 3; ++c) {
                const int oc = 2 - c; // RGB -> BGR
                out[(static_cast<size_t>(i) * 3 + oc) * PLANE + p] = static_cast<float>(
                    (resized[static_cast<size_t>(p) * 3 + c] * static_cast<double>(ALPHA) - static_cast<double>(SUB[oc])) / static_cast<double>(DIV[oc]));
            }
        }
    }
    return out;
}

struct Output {
    fklb::CsvWriter* timing;
    fklb::CsvWriter* agreement;
    bool allPass{true};
    int warmup, iters;
};

template <typename Work>
void timeVariant(Output& out, const int batch, const std::string& variant, const int calls, Work work) {
    const nv12::Stats s = nv12::measure(work, out.warmup, out.iters, 0.5);
    out.timing->row(batch, variant, calls, s.median, s.min, s.max, s.mean, s.iters);
    std::cout << "  " << variant << ": median " << s.median << " ms (min " << s.min << ", max " << s.max << ")" << std::endl;
}

inline void agreementRows(Output& out, const int batch, const std::string& what, const float* a, const float* b, const int n, const bool required,
                          const char* note) {
    for (const bool edge : {false, true}) {
        const Region r = compareTensors(a, b, n, OUT_W, OUT_H, edge);
        const bool pass = r.maxAbs <= TOLERANCE;
        out.agreement->row(batch, what, edge ? "last 2 columns and rows" : "interior", r.maxAbs, r.meanAbs, r.fractionOver, required ? (pass ? 1 : 0) : -1, note);
        out.allPass &= pass || !required;
        std::cout << "  " << what << " [" << (edge ? "edge" : "interior") << "]: max " << r.maxAbs << ", mean " << r.meanAbs << ", over 1e-4 " << r.fractionOver
                  << (required ? (pass ? " ok" : " FAIL") : " (informational)") << std::endl;
    }
}

template <int N>
void runA(fk::Stream_<fk::ParArch::GPU_AMD>& stream, fk::Ptr2D<U3>& source, Output& out) {
    std::cout << "(a) batch " << N << std::endl;
    using HalfPixel = fk::ResizeHalfPixel<fk::InterpolationType::INTER_LINEAR>;
    FklA<N> fkl(stream, source);
    FklA<N, HalfPixel> fklHalf(stream, source);
    std::vector<std::array<int, 4>> crops(N);
    for (int i = 0; i < N; ++i) {
        crops[i] = {i, i, CROP_W, CROP_H};
    }
    const auto src = source.ptr();
    RppFusedApi rpp(stream.getHIPStream(), src.data, src.dims.pitch, SRC_W, SRC_H, crops, ALPHA, SUB, DIV, true);

    // Agreement
    fkl.fused();
    fkl.perOpBatch();
    fkl.download();
    fklHalf.fused();
    fklHalf.download();
    fkl.sync();
    rpp.run();
    const std::vector<float> rppOut = rpp.download();
    const auto hostSource = nv12::hostRaw(source);
    const std::vector<float> refAsym = referenceA(hostSource, N, fklb::ResizeSampling::ASYMMETRIC);
    const std::vector<float> refHalf = referenceA(hostSource, N, fklb::ResizeSampling::HALF_PIXEL);
    agreementRows(out, N, "fkl_fused vs scalar reference ASYMMETRIC", fkl.fusedOut(), refAsym.data(), N, true, "");
    agreementRows(out, N, "fkl_perop_batch vs fkl_fused", fkl.batchOut(), fkl.fusedOut(), N, true, "");
    agreementRows(out, N, "rpp_fused_api vs scalar reference HALF_PIXEL", rppOut.data(), refHalf.data(), N, false,
                  "RPP clamps the source coordinate to size-2: the last 2 columns and rows differ from the reference");
    agreementRows(out, N, "rpp_fused_api vs fkl_fused", rppOut.data(), fkl.fusedOut(), N, false, "FKL samples ASYMMETRIC, RPP HALF_PIXEL");
    agreementRows(out, N, "fkl_fused_halfpixel vs scalar reference HALF_PIXEL", fklHalf.fusedOut(), refHalf.data(), N, true, "");
    agreementRows(out, N, "rpp_fused_api vs fkl_fused_halfpixel", rppOut.data(), fklHalf.fusedOut(), N, false,
                  "both HALF_PIXEL; RPP clamps the source coordinate to size-2: the last 2 columns and rows differ");

    // Timing
    timeVariant(out, N, "fkl_fused", 1, [&] { fkl.fused(); fkl.sync(); });
    timeVariant(out, N, "fkl_fused_halfpixel", 1, [&] { fklHalf.fused(); fklHalf.sync(); });
    timeVariant(out, N, "fkl_perop_batch", 6, [&] { fkl.perOpBatch(); fkl.sync(); });
    timeVariant(out, N, "rpp_fused_api", 2, [&] { rpp.run(); RPPB_CHECK_HIP(hipStreamSynchronize(stream.getHIPStream())); });
    out.timing->flush();
    out.agreement->flush();
}

template <int... Ns>
void runAll(fk::Stream_<fk::ParArch::GPU_AMD>& stream, fk::Ptr2D<U3>& source, Output& out, std::integer_sequence<int, Ns...>) {
    (runA<(Ns + 1) * 10>(stream, source, out), ...);
}

// ------------------------------------------------------------------------------------------------------------------------------
// Pipeline (b)

template <int N>
void runB(fk::Stream_<fk::ParArch::GPU_AMD>& stream, nv12::Frame& frame, fk::Ptr2D<U3>& rgbFkl, fk::Ptr2D<U3>& rgbRpp, rppHandle_t convertHandle, Output& out) {
    std::cout << "(b) crops " << N << std::endl;
    const auto crops = nv12::makeCrops<N>();
    using HalfPixel = fk::ResizeHalfPixel<fk::InterpolationType::INTER_LINEAR>;
    nv12::Pipeline<fk::ParArch::GPU_AMD, N> fkl(stream, frame, rgbFkl, crops);
    nv12::Pipeline<fk::ParArch::GPU_AMD, N, HalfPixel> fklHalf(stream, frame, rgbFkl, crops);
    std::vector<std::array<int, 4>> roiList(N);
    for (int i = 0; i < N; ++i) {
        roiList[i] = {static_cast<int>(crops[i].x), static_cast<int>(crops[i].y), crops[i].width, crops[i].height};
    }
    const auto rgbRaw = rgbRpp.ptr();
    RppFusedApi rpp(stream.getHIPStream(), rgbRaw.data, rgbRaw.dims.pitch, nv12::FRAME_W, nv12::FRAME_H, roiList, 1.f / 255.f, nv12::MEAN, nv12::STD, false);

    const auto frameRaw = frame.getData().ptr();
    uchar* yPlane = reinterpret_cast<uchar*>(frameRaw.data);
    uchar* uvPlane = yPlane + static_cast<size_t>(frameRaw.dims.pitch) * nv12::FRAME_H;
    RpptDesc yuvDesc(1, 3, nv12::FRAME_H, nv12::FRAME_W, U8, NHWC), rgbDesc(1, 3, nv12::FRAME_H, nv12::FRAME_W, U8, NHWC);
    rgbDesc.strides.hStride = static_cast<Rpp32u>(rgbRaw.dims.pitch);
    const auto convertRpp = [&] {
        RPPB_CHECK_RPP(rppt_yuv_to_rgb(yPlane, uvPlane, &yuvDesc, rgbRaw.data, &rgbDesc, frameRaw.dims.pitch, frameRaw.dims.pitch, rgbRaw.dims.pitch,
                                       nv12::FRAME_W, nv12::FRAME_H, RpptColorStandard_BT601, RpptColorRange_STUDIO, convertHandle,
                                       RppBackend::RPP_HIP_BACKEND));
    };

    // Agreement
    fkl.convert();
    fkl.fused();
    fkl.perOpBatch();
    fkl.download();
    fklHalf.fused();
    fklHalf.download();
    fkl.sync();
    convertRpp();
    rpp.run();
    const std::vector<float> rppOut = rpp.download();
    rgbRpp.download(stream);
    stream.sync();
    {
        // The RGB frames of FKL and RPP, in levels
        const auto a = nv12::hostRaw(rgbFkl);
        const auto b = nv12::hostRaw(rgbRpp);
        int worst = 0;
        double sum = 0;
        for (int y = 0; y < nv12::FRAME_H; ++y) {
            const U3* ra = reinterpret_cast<const U3*>(reinterpret_cast<const char*>(a.data) + static_cast<size_t>(y) * a.dims.pitch);
            const U3* rb = reinterpret_cast<const U3*>(reinterpret_cast<const char*>(b.data) + static_cast<size_t>(y) * b.dims.pitch);
            for (int x = 0; x < nv12::FRAME_W; ++x) {
                const int d = std::max({std::abs(int(ra[x].x) - int(rb[x].x)), std::abs(int(ra[x].y) - int(rb[x].y)), std::abs(int(ra[x].z) - int(rb[x].z))});
                worst = std::max(worst, d);
                sum += std::abs(int(ra[x].x) - int(rb[x].x)) + std::abs(int(ra[x].y) - int(rb[x].y)) + std::abs(int(ra[x].z) - int(rb[x].z));
            }
        }
        out.agreement->row(N, "RGB frame: rppt_yuv_to_rgb vs FKL ConvertYUVToRGB (levels)", "frame", worst, sum / (3.0 * nv12::FRAME_W * nv12::FRAME_H), 0, -1,
                           "BT.601 studio swing, default chroma upsampling of RPP");
        std::cout << "  RGB frame RPP vs FKL: max " << worst << " levels, mean " << sum / (3.0 * nv12::FRAME_W * nv12::FRAME_H) << std::endl;
    }
    const std::vector<float> ref = nv12::referenceTensor<N>(nv12::hostRaw(rgbFkl), crops);
    agreementRows(out, N, "fkl_fused vs scalar reference", fkl.fusedOut(), ref.data(), N, true, "");
    agreementRows(out, N, "fkl_perop_batch vs fkl_fused", fkl.batchOut(), fkl.fusedOut(), N, true, "");
    agreementRows(out, N, "rpp_fused_api vs fkl_fused", rppOut.data(), fkl.fusedOut(), N, false, "different RGB frames and sampling conventions");
    const std::vector<float> refHalf = nv12::referenceTensor<N>(nv12::hostRaw(rgbFkl), crops, fklb::ResizeSampling::HALF_PIXEL);
    agreementRows(out, N, "fkl_fused_halfpixel vs scalar reference HALF_PIXEL", fklHalf.fusedOut(), refHalf.data(), N, true, "");
    agreementRows(out, N, "rpp_fused_api vs fkl_fused_halfpixel", rppOut.data(), fklHalf.fusedOut(), N, false,
                  "both HALF_PIXEL; the RGB frames differ by at most 1 level; RPP clamps the source coordinate to size-2: the last 2 columns and rows differ");

    // Timing
    timeVariant(out, N, "rpp_convert", 1, [&] { convertRpp(); RPPB_CHECK_HIP(hipStreamSynchronize(stream.getHIPStream())); });
    timeVariant(out, N, "fkl_convert", 1, [&] { fkl.convert(); fkl.sync(); });
    timeVariant(out, N, "fkl_chain_fused", 1, [&] { fkl.fused(); fkl.sync(); });
    timeVariant(out, N, "fkl_chain_fused_halfpixel", 1, [&] { fklHalf.fused(); fklHalf.sync(); });
    timeVariant(out, N, "fkl_chain_perop_batch", 5, [&] { fkl.perOpBatch(); fkl.sync(); });
    timeVariant(out, N, "rpp_chain_fused_api", 1, [&] { rpp.run(); RPPB_CHECK_HIP(hipStreamSynchronize(stream.getHIPStream())); });
    timeVariant(out, N, "fkl_pipeline_fused", 2, [&] { fkl.convert(); fkl.fused(); fkl.sync(); });
    timeVariant(out, N, "fkl_pipeline_fused_halfpixel", 2, [&] { fklHalf.convert(); fklHalf.fused(); fklHalf.sync(); });
    timeVariant(out, N, "fkl_pipeline_perop_batch", 6, [&] { fkl.convert(); fkl.perOpBatch(); fkl.sync(); });
    timeVariant(out, N, "rpp_pipeline_fused_api", 2, [&] { convertRpp(); rpp.run(); RPPB_CHECK_HIP(hipStreamSynchronize(stream.getHIPStream())); });
    timeVariant(out, N, "mixed_fkl_convert_rpp_chain", 2, [&] { fkl.convert(); fkl.sync(); rpp.run(); RPPB_CHECK_HIP(hipStreamSynchronize(stream.getHIPStream())); });
    out.timing->flush();
    out.agreement->flush();
}

} // namespace rppbench

int launch() {
    using namespace rppbench;
    const char* imagePath = std::getenv("E5_IMAGE");
    const char* imageInfo = std::getenv("E5_IMAGE_INFO");
    const int warmup = envInt("E5_WARMUP", 3), iters = envInt("E5_ITERS", 30);
    try {
        fk::Stream_<fk::ParArch::GPU_AMD> stream;
        std::string sourceText;

        // The image of pipeline (a), in device memory
        fk::Ptr2D<U3> source(SRC_W, SRC_H);
        {
            std::vector<uchar> ppm;
            int w = 0, h = 0;
            const bool natural = imagePath != nullptr && nv12::readPpm(imagePath, ppm, w, h);
            uint32_t state = 12345u;
            for (int y = 0; y < SRC_H; ++y) {
                for (int x = 0; x < SRC_W; ++x) {
                    if (natural) {
                        const uchar* p = &ppm[(static_cast<size_t>(y % h) * w + (x % w)) * 3];
                        source.at(x, y) = fk::make_<U3>(p[0], p[1], p[2]);
                    } else {
                        state = state * 1664525u + 1013904223u;
                        source.at(x, y) = fk::make_<U3>(static_cast<uchar>(state >> 24), static_cast<uchar>(state >> 16), static_cast<uchar>(state >> 8));
                    }
                }
            }
            sourceText = natural ? std::string("natural image ") + imagePath + " (" + std::to_string(w) + "x" + std::to_string(h) + ") tiled over the frame" +
                                       (imageInfo ? std::string("; ") + imageInfo : "")
                                 : "seeded noise (LCG, seed 12345)";
            source.upload(stream);
        }

        const std::string rppText = "RPP " + std::to_string(RPP_VERSION_MAJOR) + "." + std::to_string(RPP_VERSION_MINOR) + "." + std::to_string(RPP_VERSION_PATCH) +
                                    " (ROCm package amdrocm-rpp), rppt tensor API, HIP back-end; rppt_resize_mirror_normalize: BILINEAR, ROI XYWH, source U8 NHWC "
                                    "(batch stride 0, one ROI per crop), destination F32 NCHW, mirror off; rppt_channel_permute F32 NCHW; rppt_yuv_to_rgb: "
                                    "RpptColorStandard_BT601, RpptColorRange_STUDIO, default chroma upsampling; output widths are multiples of 8";
        const std::string notes = "FKL HIP vs RPP; source: " + sourceText + "; " + rppText +
                                  "; wall clock including the stream synchronisation, 3 warm-up runs and at least 0.5 s of warm-up";

        fklb::CsvWriter timingA("benchmark_rpp_preprocessing_a_GPU_AMD.csv", "benchmark_rpp_preprocessing_a",
                                {"batch", "variant", "calls", "median_ms", "min_ms", "max_ms", "mean_ms", "iters"}, notes);
        fklb::CsvWriter agreeA("benchmark_rpp_preprocessing_a_agreement_GPU_AMD.csv", "benchmark_rpp_preprocessing_a",
                               {"batch", "comparison", "region", "max_abs_diff", "mean_abs_diff", "fraction_over_1e-4", "pass", "note"},
                               notes + "; pass -1 = informational; tolerance 1e-4 absolute");
        Output outA{&timingA, &agreeA, true, warmup, iters};
        runAll(stream, source, outA, std::make_integer_sequence<int, 30>{});

        // Pipeline (b)
        nv12::Frame frame(nv12::FRAME_W, nv12::FRAME_H);
        if (imagePath != nullptr) {
            std::vector<uchar> ppm;
            int w = 0, h = 0;
            if (nv12::readPpm(imagePath, ppm, w, h)) {
                nv12::fillFromRgb(frame, ppm, w, h);
            } else {
                nv12::fillNoise(frame);
            }
        } else {
            nv12::fillNoise(frame);
        }
        frame.getData().upload(stream);
        stream.sync();
        fk::Ptr2D<U3> rgbFkl(nv12::FRAME_W, nv12::FRAME_H), rgbRpp(nv12::FRAME_W, nv12::FRAME_H);
        rppHandle_t convertHandle{};
        RPPB_CHECK_RPP(rppCreate(&convertHandle, 1, 0, stream.getHIPStream(), RppBackend::RPP_HIP_BACKEND));
        fklb::CsvWriter timingB("benchmark_rpp_preprocessing_b_GPU_AMD.csv", "benchmark_rpp_preprocessing_b",
                                {"crops", "variant", "calls", "median_ms", "min_ms", "max_ms", "mean_ms", "iters"}, notes);
        fklb::CsvWriter agreeB("benchmark_rpp_preprocessing_b_agreement_GPU_AMD.csv", "benchmark_rpp_preprocessing_b",
                               {"crops", "comparison", "region", "max_abs_diff", "mean_abs_diff", "fraction_over_1e-4", "pass", "note"},
                               notes + "; pass -1 = informational; tolerance 1e-4 absolute");
        Output outB{&timingB, &agreeB, true, warmup, iters};
        runB<1>(stream, frame, rgbFkl, rgbRpp, convertHandle, outB);
        runB<8>(stream, frame, rgbFkl, rgbRpp, convertHandle, outB);
        runB<16>(stream, frame, rgbFkl, rgbRpp, convertHandle, outB);
        runB<32>(stream, frame, rgbFkl, rgbRpp, convertHandle, outB);
        runB<64>(stream, frame, rgbFkl, rgbRpp, convertHandle, outB);
        (void)rppDestroy(convertHandle, RppBackend::RPP_HIP_BACKEND);
        std::cout << ((outA.allPass && outB.allPass) ? "all required numerical checks passed" : "NUMERICAL CHECK FAILED") << std::endl;
        return (outA.allPass && outB.allPass) ? 0 : -1;
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}
