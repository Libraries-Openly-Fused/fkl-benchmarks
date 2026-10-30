// E5, pipeline (a): batched DNN pre-processing on the CPU. The chain of cvGPUSpeedup-FastOpenCV LTS-C++17 (6d26c65),
// tests/batchresize/test_batchresize_x_split3D.cu, 8UC3 -> 32FC3: BATCH crops of 60x120 at (i, i) of a 3840x2160 image,
// bilinear resize to 64x128, multiply by 0.3, RGB -> BGR, subtract {1, 4, 3.2}, divide by {3.2, 0.6, 11.8}, split into
// an NCHW float tensor. Compared on the FKL CPU back-end (single threaded ParArch::CPU and ParArch::CPU_MULTITHREAD)
// and OpenCV on the CPU:
//   OpenCV   opencv_*            the cv::Mat mirror of the "OCVBatch" loop of the test: convertTo and resize per crop,
//                                multiply, cvtColor, subtract and divide on the whole batch, split per crop;
//                                opencv_percrop_* does every operation per crop
//   FKL      fkl_fused_*         one executeOperations for the whole chain and batch (batched read + 5 operations)
//            fkl_perop_crop_*    one executeOperations per operation and per crop, through host Ptr2D intermediates
//            fkl_perop_batch_*   one executeOperations per operation for the whole batch, through a host Tensor
// Wall clock (std::chrono::steady_clock): warm-up runs, then timed runs; median, min, max and mean per point.
// Before timing, every implementation runs once and its output is compared (FKL against OpenCV within 1e-4, the
// tolerance of the test; single against multithreaded FKL byte by byte). Environment variables: E5_ITERS (timed runs,
// default 30), E5_WARMUP (3), E5_IMAGE (a binary PPM instead of the seeded noise), E5_IMAGE_INFO (text for the header), E5_OUT_DIR (CSV directory, default "."), E5_CHECK_ONLY=1 (check, do not time).
#include "e5_common.h"

#include <common/main.h>
#include <common/resize_reference.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace {

constexpr int SRC_W = 3840, SRC_H = 2160, CROP_W = 60, CROP_H = 120, UP_W = 64, UP_H = 128;
constexpr int PLANE = UP_W * UP_H;
constexpr double TOLERANCE = 1e-4;
constexpr int BATCHES[] = {E5_BATCHES};

using F3 = fk::float3;
using U3 = fk::uchar3;
using Rgb2Bgr = fk::ColorConversion<fk::ColorConversionCodes::COLOR_RGB2BGR, F3, F3>;
using StreamST = fk::Stream_<fk::ParArch::CPU>;
using StreamMT = fk::Stream_<fk::ParArch::CPU_MULTITHREAD>;
using DppST = fk::TransformDPP<fk::ParArch::CPU>;
using DppMT = fk::TransformDPP<fk::ParArch::CPU_MULTITHREAD>;

const cv::Scalar ALPHA_S(0.3, 0.3, 0.3), SUB_S(1.f, 4.f, 3.2f), DIV_S(3.2f, 0.6f, 11.8f);

int envInt(const char* name, const int fallback) {
    const char* value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

// ------------------------------------------------------------------------------------------------------------------
// FKL operations of the chain
inline fk::RawPtr<fk::ND::_2D, U3> roiOf(const fk::RawPtr<fk::ND::_2D, U3>& full, const int x, const int y) {
    fk::RawPtr<fk::ND::_2D, U3> roi = full;
    roi.data = reinterpret_cast<U3*>(reinterpret_cast<char*>(full.data) + static_cast<size_t>(y) * full.dims.pitch) + x;
    roi.dims.width = CROP_W;
    roi.dims.height = CROP_H;
    return roi;
}

inline fk::RawPtr<fk::ND::_3D, float> tensorSlice(fk::RawPtr<fk::ND::_3D, float> tensor, const int i) {
    tensor.data = reinterpret_cast<float*>(reinterpret_cast<char*>(tensor.data) +
                                           static_cast<size_t>(i) * tensor.dims.plane_pitch * tensor.dims.color_planes);
    tensor.dims.planes = 1;
    return tensor;
}

inline auto resizeOf(const fk::RawPtr<fk::ND::_2D, U3>& roi) {
    return fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::PerThreadRead<fk::ND::_2D, U3>::build(roi),
                                                                  fk::Size(UP_W, UP_H));
}
inline auto read2D(const fk::RawPtr<fk::ND::_2D, F3>& p) { return fk::PerThreadRead<fk::ND::_2D, F3>::build(p); }
inline auto write2D(const fk::RawPtr<fk::ND::_2D, F3>& p) { return fk::PerThreadWrite<fk::ND::_2D, F3>::build(p); }
inline auto read3D(const fk::RawPtr<fk::ND::_3D, F3>& p) { return fk::PerThreadRead<fk::ND::_3D, F3>::build(p); }
inline auto write3D(const fk::RawPtr<fk::ND::_3D, F3>& p) { return fk::PerThreadWrite<fk::ND::_3D, F3>::build(p); }
inline auto split3D(const fk::RawPtr<fk::ND::_3D, float>& p) { return fk::TensorSplit<F3>::build(p); }
inline auto mulOp() { return fk::Mul<F3>::build(F3{0.3f, 0.3f, 0.3f}); }
inline auto subOp() { return fk::Sub<F3>::build(F3{1.f, 4.f, 3.2f}); }
inline auto divOp() { return fk::Div<F3>::build(F3{3.2f, 0.6f, 11.8f}); }
inline auto colorOp() { return Rgb2Bgr::build(); }

using ResizeOp = decltype(resizeOf(std::declval<fk::RawPtr<fk::ND::_2D, U3>>()));
using Read2DOp = decltype(read2D(std::declval<fk::RawPtr<fk::ND::_2D, F3>>()));
using Write2DOp = decltype(write2D(std::declval<fk::RawPtr<fk::ND::_2D, F3>>()));
using Read3DOp = decltype(read3D(std::declval<fk::RawPtr<fk::ND::_3D, F3>>()));
using Write3DOp = decltype(write3D(std::declval<fk::RawPtr<fk::ND::_3D, F3>>()));
using SplitOp = decltype(split3D(std::declval<fk::RawPtr<fk::ND::_3D, float>>()));
using MulOp = decltype(mulOp());
using SubOp = decltype(subOp());
using DivOp = decltype(divOp());
using ColorOp = decltype(colorOp());

// ------------------------------------------------------------------------------------------------------------------
// The source image, shared by every implementation: the same bytes and the same pitch
struct Source {
    fk::Ptr2D<U3> image{SRC_W, SRC_H, 0, fk::MemType::Host};
    cv::Mat mat;
    std::string description;
    // Seeded noise, or the binary PPM (P6, 8 bit) at ppmPath tiled over the frame
    explicit Source(const char* ppmPath = nullptr) {
        const fk::RawPtr<fk::ND::_2D, U3> raw = image.ptr();
        if (ppmPath == nullptr) {
            uint32_t state = 12345u;
            for (int y = 0; y < SRC_H; ++y) {
                U3* row = reinterpret_cast<U3*>(reinterpret_cast<char*>(raw.data) + static_cast<size_t>(y) * raw.dims.pitch);
                for (int x = 0; x < SRC_W; ++x) {
                    state = state * 1664525u + 1013904223u;
                    row[x] = fk::make_<U3>(static_cast<uchar>(state >> 24), static_cast<uchar>(state >> 16),
                                           static_cast<uchar>(state >> 8));
                }
            }
            description = "deterministic noise (LCG, seed 12345, x = x * 1664525 + 1013904223, bytes 31..24, 23..16, 15..8)";
        } else {
            std::ifstream in(ppmPath, std::ios::binary);
            std::string magic;
            int w = 0, h = 0, maxval = 0;
            in >> magic >> w >> h >> maxval;
            in.get();
            if (!in || magic != "P6" || maxval != 255 || w <= 0 || h <= 0) {
                throw std::runtime_error(std::string("Cannot read the binary PPM ") + ppmPath);
            }
            std::vector<uchar> pixels(static_cast<size_t>(w) * h * 3);
            in.read(reinterpret_cast<char*>(pixels.data()), static_cast<std::streamsize>(pixels.size()));
            if (!in) {
                throw std::runtime_error(std::string("Truncated PPM ") + ppmPath);
            }
            for (int y = 0; y < SRC_H; ++y) {
                U3* row = reinterpret_cast<U3*>(reinterpret_cast<char*>(raw.data) + static_cast<size_t>(y) * raw.dims.pitch);
                for (int x = 0; x < SRC_W; ++x) {
                    const uchar* p = &pixels[(static_cast<size_t>(y % h) * w + (x % w)) * 3];
                    row[x] = fk::make_<U3>(p[0], p[1], p[2]);
                }
            }
            description = std::string("natural image ") + ppmPath + " (" + std::to_string(w) + "x" + std::to_string(h) +
                          ") tiled over the frame";
        }
        mat = cv::Mat(SRC_H, SRC_W, CV_8UC3, raw.data, raw.dims.pitch);
    }
};

// ------------------------------------------------------------------------------------------------------------------
// OpenCV: the cv::Mat mirror of the "OCVBatch" baseline of the test
class OpenCvBatch {
public:
    OpenCvBatch(const cv::Mat& src, const int batch)
        : batch_(batch), crop32F_(CROP_H, CROP_W, CV_32FC3), resized_(batch, PLANE, CV_32FC3),
          temp_(batch, PLANE, CV_32FC3), temp2_(batch, PLANE, CV_32FC3), tensor_(batch, PLANE * 3, CV_32F) {
        for (int i = 0; i < batch; ++i) {
            rois_.push_back(src(cv::Rect(i, i, CROP_W, CROP_H)));
            resizedRows_.emplace_back(UP_H, UP_W, CV_32FC3, resized_.ptr(i));
            float* const base = tensor_.ptr<float>(i);
            planes_.push_back({cv::Mat(UP_H, UP_W, CV_32F, base), cv::Mat(UP_H, UP_W, CV_32F, base + PLANE),
                               cv::Mat(UP_H, UP_W, CV_32F, base + 2 * PLANE)});
        }
    }
    void run() {
        for (int i = 0; i < batch_; ++i) {
            rois_[i].convertTo(crop32F_, CV_32FC3);
            cv::resize(crop32F_, resizedRows_[i], cv::Size(UP_W, UP_H), 0., 0., cv::INTER_LINEAR);
        }
        cv::multiply(resized_, ALPHA_S, temp_);
        cv::cvtColor(temp_, temp_, cv::COLOR_RGB2BGR);
        cv::subtract(temp_, SUB_S, temp2_);
        cv::divide(temp2_, DIV_S, temp_);
        for (int i = 0; i < batch_; ++i) {
            cv::split(temp_.row(i).reshape(3, UP_H), planes_[i]);
        }
    }
    const float* tensor() const { return tensor_.ptr<float>(0); }

private:
    int batch_;
    cv::Mat crop32F_, resized_, temp_, temp2_, tensor_;
    std::vector<cv::Mat> rois_, resizedRows_;
    std::vector<std::vector<cv::Mat>> planes_;
};

// OpenCV, every operation on each crop (the non batched baseline of the test)
class OpenCvPerCrop {
public:
    OpenCvPerCrop(const cv::Mat& src, const int batch)
        : batch_(batch), crop32F_(CROP_H, CROP_W, CV_32FC3), resized_(UP_H, UP_W, CV_32FC3),
          temp_(UP_H, UP_W, CV_32FC3), temp2_(UP_H, UP_W, CV_32FC3), tensor_(batch, PLANE * 3, CV_32F) {
        for (int i = 0; i < batch; ++i) {
            rois_.push_back(src(cv::Rect(i, i, CROP_W, CROP_H)));
            float* const base = tensor_.ptr<float>(i);
            planes_.push_back({cv::Mat(UP_H, UP_W, CV_32F, base), cv::Mat(UP_H, UP_W, CV_32F, base + PLANE),
                               cv::Mat(UP_H, UP_W, CV_32F, base + 2 * PLANE)});
        }
    }
    void run() {
        for (int i = 0; i < batch_; ++i) {
            rois_[i].convertTo(crop32F_, CV_32FC3);
            cv::resize(crop32F_, resized_, cv::Size(UP_W, UP_H), 0., 0., cv::INTER_LINEAR);
            cv::multiply(resized_, ALPHA_S, temp_);
            cv::cvtColor(temp_, temp_, cv::COLOR_RGB2BGR);
            cv::subtract(temp_, SUB_S, temp2_);
            cv::divide(temp2_, DIV_S, temp_);
            cv::split(temp_, planes_[i]);
        }
    }
    const float* tensor() const { return tensor_.ptr<float>(0); }

private:
    int batch_;
    cv::Mat crop32F_, resized_, temp_, temp2_, tensor_;
    std::vector<cv::Mat> rois_;
    std::vector<std::vector<cv::Mat>> planes_;
};

// OpenCV with the crops processed in parallel (cv::parallel_for_ over the crops, one set of temporaries per worker, every
// operation of a crop on one thread): what a user does to use all the cores, since OpenCV itself does not split the
// operations of crops this small between threads
class OpenCvCropsParallel {
public:
    OpenCvCropsParallel(const cv::Mat& src, const int batch) : batch_(batch), tensor_(batch, PLANE * 3, CV_32F) {
        for (int i = 0; i < batch; ++i) {
            rois_.push_back(src(cv::Rect(i, i, CROP_W, CROP_H)));
            float* const base = tensor_.ptr<float>(i);
            planes_.push_back({cv::Mat(UP_H, UP_W, CV_32F, base), cv::Mat(UP_H, UP_W, CV_32F, base + PLANE),
                               cv::Mat(UP_H, UP_W, CV_32F, base + 2 * PLANE)});
        }
    }
    void run() {
        cv::parallel_for_(cv::Range(0, batch_), [this](const cv::Range& range) {
            // The temporaries of each worker thread (cv::getThreadNum() is not defined for the pthreads backend)
            thread_local Scratch s;
            for (int i = range.start; i < range.end; ++i) {
                rois_[i].convertTo(s.crop32F, CV_32FC3);
                cv::resize(s.crop32F, s.resized, cv::Size(UP_W, UP_H), 0., 0., cv::INTER_LINEAR);
                cv::multiply(s.resized, ALPHA_S, s.temp);
                cv::cvtColor(s.temp, s.temp, cv::COLOR_RGB2BGR);
                cv::subtract(s.temp, SUB_S, s.temp2);
                cv::divide(s.temp2, DIV_S, s.temp);
                cv::split(s.temp, planes_[i]);
            }
        });
    }
    const float* tensor() const { return tensor_.ptr<float>(0); }

private:
    struct Scratch {
        cv::Mat crop32F, resized, temp, temp2;
    };
    int batch_;
    cv::Mat tensor_;
    std::vector<cv::Mat> rois_;
    std::vector<std::vector<cv::Mat>> planes_;
};

// ------------------------------------------------------------------------------------------------------------------
// FKL, the whole chain in one executeOperations
template <typename Dpp, typename StreamT, int BATCH>
class FklFused {
public:
    FklFused(const Source& source, fk::Tensor<float>& tensor, StreamT& stream)
        : stream_(stream), ops_(build(source, tensor)) {}
    void run() {
        std::apply([this](const auto&... iOps) { fk::executeOperations<Dpp>(stream_, iOps...); }, ops_);
        stream_.sync();
    }

private:
    static auto build(const Source& source, fk::Tensor<float>& tensor) {
        std::array<fk::RawPtr<fk::ND::_2D, U3>, BATCH> rois;
        for (int i = 0; i < BATCH; ++i) {
            rois[i] = roiOf(source.image.ptr(), i, i);
        }
        const auto readBatch = fk::PerThreadRead<fk::ND::_2D, U3>::build_batch(rois);
        const auto sizes = fk::make_set_std_array<BATCH>(fk::Size(UP_W, UP_H));
        return std::make_tuple(fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(readBatch, sizes), colorOp(),
                               mulOp(), subOp(), divOp(), split3D(tensor.ptr()));
    }
    StreamT& stream_;
    decltype(build(std::declval<const Source&>(), std::declval<fk::Tensor<float>&>())) ops_;
};

// FKL, one executeOperations per operation and per crop, through two host Ptr2D intermediates
template <typename Dpp, typename StreamT>
class FklPerOpCrop {
public:
    FklPerOpCrop(const Source& source, fk::Tensor<float>& tensor, const int batch, StreamT& stream)
        : stream_(stream), a_(UP_W, UP_H, 0, fk::MemType::Host), b_(UP_W, UP_H, 0, fk::MemType::Host),
          readA_(read2D(a_.ptr())), readB_(read2D(b_.ptr())), writeA_(write2D(a_.ptr())), writeB_(write2D(b_.ptr())),
          mul_(mulOp()), sub_(subOp()), div_(divOp()), color_(colorOp()) {
        for (int i = 0; i < batch; ++i) {
            resize_.push_back(resizeOf(roiOf(source.image.ptr(), i, i)));
            split_.push_back(split3D(tensorSlice(tensor.ptr(), i)));
        }
    }
    void run() {
        for (size_t i = 0; i < resize_.size(); ++i) {
            fk::executeOperations<Dpp>(stream_, resize_[i], writeA_);
            fk::executeOperations<Dpp>(stream_, readA_, mul_, writeB_);
            fk::executeOperations<Dpp>(stream_, readB_, color_, writeA_);
            fk::executeOperations<Dpp>(stream_, readA_, sub_, writeB_);
            fk::executeOperations<Dpp>(stream_, readB_, div_, writeA_);
            fk::executeOperations<Dpp>(stream_, readA_, split_[i]);
        }
        stream_.sync();
    }

private:
    StreamT& stream_;
    fk::Ptr2D<F3> a_, b_;
    Read2DOp readA_, readB_;
    Write2DOp writeA_, writeB_;
    MulOp mul_;
    SubOp sub_;
    DivOp div_;
    ColorOp color_;
    std::vector<ResizeOp> resize_;
    std::vector<SplitOp> split_;
};

// FKL, one executeOperations per operation for the whole batch, through two host Tensor intermediates
template <typename Dpp, typename StreamT, int BATCH>
class FklPerOpBatch {
public:
    FklPerOpBatch(const Source& source, fk::Tensor<float>& tensor, StreamT& stream)
        : stream_(stream), a_(UP_W, UP_H, BATCH, 1, fk::MemType::Host), b_(UP_W, UP_H, BATCH, 1, fk::MemType::Host),
          readA_(read3D(a_.ptr())), readB_(read3D(b_.ptr())), writeA_(write3D(a_.ptr())), writeB_(write3D(b_.ptr())),
          mul_(mulOp()), sub_(subOp()), div_(divOp()), color_(colorOp()), split_(split3D(tensor.ptr())),
          resize_(buildResize(source)) {}
    void run() {
        fk::executeOperations<Dpp>(stream_, resize_, writeA_);
        fk::executeOperations<Dpp>(stream_, readA_, mul_, writeB_);
        fk::executeOperations<Dpp>(stream_, readB_, color_, writeA_);
        fk::executeOperations<Dpp>(stream_, readA_, sub_, writeB_);
        fk::executeOperations<Dpp>(stream_, readB_, div_, writeA_);
        fk::executeOperations<Dpp>(stream_, readA_, split_);
        stream_.sync();
    }

private:
    static auto buildResize(const Source& source) {
        std::array<fk::RawPtr<fk::ND::_2D, U3>, BATCH> rois;
        for (int i = 0; i < BATCH; ++i) {
            rois[i] = roiOf(source.image.ptr(), i, i);
        }
        const auto readBatch = fk::PerThreadRead<fk::ND::_2D, U3>::build_batch(rois);
        return fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(
            readBatch, fk::make_set_std_array<BATCH>(fk::Size(UP_W, UP_H)));
    }
    StreamT& stream_;
    fk::Tensor<F3> a_, b_;
    Read3DOp readA_, readB_;
    Write3DOp writeA_, writeB_;
    MulOp mul_;
    SubOp sub_;
    DivOp div_;
    ColorOp color_;
    SplitOp split_;
    decltype(buildResize(std::declval<const Source&>())) resize_;
};

// ------------------------------------------------------------------------------------------------------------------
// A plain scalar implementation of the chain in double precision (common/resize_reference.h for the resize), for the two
// sampling conventions of a linear resize (ResizeSampling): ASYMMETRIC, as FKL and cv::cuda::resize, maps a destination pixel x
// to the source position x * (src / dst); HALF_PIXEL, as cv::resize and NPP, to (x + 0.5) * (src / dst) - 0.5, so that the centres
// of the pixels are aligned. The two give different images unless the crop is constant (the test of cvGPUSpeedup resizes constant
// images, so it cannot see it).
void scalarReference(const Source& source, const int batch, const bool pixelCentres, std::vector<float>& out) {
    out.assign(static_cast<size_t>(batch) * 3 * PLANE, 0.f);
    const fk::RawPtr<fk::ND::_2D, U3> raw = source.image.ptr();
    const double alpha = static_cast<double>(0.3f), sub[3] = {1.0, 4.0, static_cast<double>(3.2f)},
                 div[3] = {static_cast<double>(3.2f), static_cast<double>(0.6f), static_cast<double>(11.8f)};
    const fklb::ResizeSampling sampling = pixelCentres ? fklb::ResizeSampling::HALF_PIXEL : fklb::ResizeSampling::ASYMMETRIC;
    std::vector<double> resized(static_cast<size_t>(PLANE) * 3);
    for (int i = 0; i < batch; ++i) {
        const fk::RawPtr<fk::ND::_2D, U3> roi = roiOf(raw, i, i);
        fklb::resizeBilinear(reinterpret_cast<const uchar*>(roi.data), roi.dims.pitch, CROP_W, CROP_H, 3, UP_W, UP_H,
                             sampling, resized.data());
        for (int p = 0; p < PLANE; ++p) {
            for (int c = 0; c < 3; ++c) {
                const int outChannel = 2 - c; // RGB -> BGR
                out[(static_cast<size_t>(i) * 3 + outChannel) * PLANE + p] =
                    static_cast<float>((resized[static_cast<size_t>(p) * 3 + c] * alpha - sub[outChannel]) / div[outChannel]);
            }
        }
    }
}

// ------------------------------------------------------------------------------------------------------------------
struct Row {
    int batch;
    std::string variant;
    int threads;
    e5::Stats stats;
};

struct Agreement {
    int batch;
    std::string pair;
    double maxAbsDiff;
    double meanAbsDiff;
    double fractionOverTolerance; // fraction of the values that differ by more than TOLERANCE
    bool pass;
    bool required; // false: informational, not part of the pass/fail result
};

double meanAbsDiff(const float* a, const float* b, const size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return sum / static_cast<double>(n);
}

double fractionOver(const float* a, const float* b, const size_t n) {
    size_t over = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!(std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])) <= TOLERANCE)) {
            ++over;
        }
    }
    return static_cast<double>(over) / static_cast<double>(n);
}

double maxAbsDiff(const float* a, const float* b, const size_t n) {
    double worst = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (!(d <= worst)) { // also catches NaN
            worst = d;
        }
    }
    return worst;
}

const float* tensorData(const fk::Tensor<float>& tensor, const int batch) {
    const fk::RawPtr<fk::ND::_3D, float> raw = tensor.ptr();
    const size_t expected = static_cast<size_t>(batch) * 3 * PLANE;
    if (raw.dims.pitch != UP_W * sizeof(float) || raw.dims.plane_pitch * raw.dims.color_planes * raw.dims.planes != expected * sizeof(float)) {
        throw std::runtime_error("The FKL tensor is not dense");
    }
    return raw.data;
}

template <int BATCH>
void runPoint(const Source& source, const e5::Context& ctx, const int warmup, const int iters, const bool checkOnly,
              std::vector<Row>& rows, std::vector<Agreement>& agreements) {
    constexpr size_t N = static_cast<size_t>(BATCH) * 3 * PLANE;
    std::cout << "batch " << BATCH << std::endl;

    OpenCvBatch ocv(source.mat, BATCH);
    OpenCvPerCrop ocvPc(source.mat, BATCH);
    OpenCvCropsParallel ocvPar(source.mat, BATCH);

    fk::Tensor<float> tFusedST(UP_W, UP_H, BATCH, 3, fk::MemType::Host), tFusedMT(UP_W, UP_H, BATCH, 3, fk::MemType::Host);
    fk::Tensor<float> tCropST(UP_W, UP_H, BATCH, 3, fk::MemType::Host), tCropMT(UP_W, UP_H, BATCH, 3, fk::MemType::Host);
    fk::Tensor<float> tBatchST(UP_W, UP_H, BATCH, 3, fk::MemType::Host), tBatchMT(UP_W, UP_H, BATCH, 3, fk::MemType::Host);
    StreamST streamST;
    StreamMT streamMT;
    FklFused<DppST, StreamST, BATCH> fusedST(source, tFusedST, streamST);
    FklFused<DppMT, StreamMT, BATCH> fusedMT(source, tFusedMT, streamMT);
    FklPerOpCrop<DppST, StreamST> cropST(source, tCropST, BATCH, streamST);
    FklPerOpCrop<DppMT, StreamMT> cropMT(source, tCropMT, BATCH, streamMT);
    FklPerOpBatch<DppST, StreamST, BATCH> batchST(source, tBatchST, streamST);
    FklPerOpBatch<DppMT, StreamMT, BATCH> batchMT(source, tBatchMT, streamMT);

    // Numerical agreement: one run of everything, then compare
    const int defaultThreads = ctx.cvThreadsDefault;
    cv::setNumThreads(defaultThreads);
    ocv.run();
    ocvPc.run();
    ocvPar.run();
    const std::vector<float> ocvDefault(ocv.tensor(), ocv.tensor() + N);
    cv::setNumThreads(1);
    ocv.run();
    const bool ocvThreadsSame = std::memcmp(ocvDefault.data(), ocv.tensor(), N * sizeof(float)) == 0;
    fusedST.run();
    fusedMT.run();
    cropST.run();
    cropMT.run();
    batchST.run();
    batchMT.run();
    const float* fused = tensorData(tFusedST, BATCH);
    std::vector<float> refFkl, refCv;
    scalarReference(source, BATCH, false, refFkl);
    scalarReference(source, BATCH, true, refCv);
    const auto compare = [&](const std::string& pair, const float* a, const float* b, const bool exact, const bool required = true) {
        const double d = maxAbsDiff(a, b, N);
        const bool pass = exact ? std::memcmp(a, b, N * sizeof(float)) == 0 : d <= TOLERANCE;
        agreements.push_back({BATCH, pair, d, meanAbsDiff(a, b, N), fractionOver(a, b, N), pass, required});
        std::cout << "  " << pair << ": max abs diff " << d << (required ? (pass ? " ok" : " FAIL") : " (informational)")
                  << std::endl;
    };
    compare("opencv_default vs opencv_1thread (identical)", ocvDefault.data(), ocv.tensor(), true);
    compare("opencv_percrop vs opencv_default (<= 1e-4)", ocvPc.tensor(), ocvDefault.data(), false);
    compare("opencv_percrop_parallel vs opencv_percrop (identical)", ocvPar.tensor(), ocvPc.tensor(), true);
    compare("opencv_default vs scalar reference, HALF_PIXEL sampling (<= 1e-4)", ocvDefault.data(), refCv.data(), false);
    compare("fkl_fused_cpu vs scalar reference, ASYMMETRIC sampling x*(src/dst) (<= 1e-4)", fused, refFkl.data(), false);
    compare("fkl_fused_cpu_mt vs fkl_fused_cpu (identical)", tensorData(tFusedMT, BATCH), fused, true);
    compare("fkl_perop_crop_cpu vs fkl_fused_cpu (<= 1e-4)", tensorData(tCropST, BATCH), fused, false);
    compare("fkl_perop_crop_cpu_mt vs fkl_perop_crop_cpu (identical)", tensorData(tCropMT, BATCH), tensorData(tCropST, BATCH), true);
    compare("fkl_perop_batch_cpu vs fkl_fused_cpu (<= 1e-4)", tensorData(tBatchST, BATCH), fused, false);
    compare("fkl_perop_batch_cpu_mt vs fkl_perop_batch_cpu (identical)", tensorData(tBatchMT, BATCH), tensorData(tBatchST, BATCH), true);
    compare("fkl_fused_cpu vs opencv_default (the two sampling conventions)", fused, ocvDefault.data(), false, false);
    compare("scalar reference ASYMMETRIC sampling vs HALF_PIXEL sampling", refFkl.data(), refCv.data(), false, false);
    if (checkOnly) {
        return;
    }

    const auto time = [&](const std::string& variant, const int threads, const std::function<void()>& fn) {
        const e5::Stats s = e5::measure(fn, warmup, iters);
        rows.push_back({BATCH, variant, threads, s});
        std::cout << "  " << variant << ": median " << s.median << " ms (min " << s.min << ", max " << s.max << ")"
                  << std::endl;
    };
    cv::setNumThreads(defaultThreads);
    time("opencv_default", defaultThreads, [&] { ocv.run(); });
    time("opencv_percrop_default", defaultThreads, [&] { ocvPc.run(); });
    time("opencv_percrop_parallel", defaultThreads, [&] { ocvPar.run(); });
    cv::setNumThreads(1);
    time("opencv_1thread", 1, [&] { ocv.run(); });
    time("opencv_percrop_1thread", 1, [&] { ocvPc.run(); });
    cv::setNumThreads(defaultThreads);
    const int pool = static_cast<int>(ctx.fklPoolSize);
    time("fkl_perop_crop_cpu", 1, [&] { cropST.run(); });
    time("fkl_perop_batch_cpu", 1, [&] { batchST.run(); });
    time("fkl_fused_cpu", 1, [&] { fusedST.run(); });
    time("fkl_perop_crop_cpu_mt", pool, [&] { cropMT.run(); });
    time("fkl_perop_batch_cpu_mt", pool, [&] { batchMT.run(); });
    time("fkl_fused_cpu_mt", pool, [&] { fusedMT.run(); });
}

template <size_t... Is>
void runSweep(const Source& source, const e5::Context& ctx, const int warmup, const int iters, const bool checkOnly,
              std::vector<Row>& rows, std::vector<Agreement>& agreements, std::index_sequence<Is...>) {
    (runPoint<BATCHES[Is]>(source, ctx, warmup, iters, checkOnly, rows, agreements), ...);
}

} // namespace

int launch() {
    const int iters = envInt("E5_ITERS", 30);
    const int warmup = envInt("E5_WARMUP", 3);
    const bool checkOnly = envInt("E5_CHECK_ONLY", 0) != 0;
    const char* outDirEnv = std::getenv("E5_OUT_DIR");
    const std::string outDir = outDirEnv ? std::string(outDirEnv) + "/" : "";

    e5::Context ctx;
    ctx.experiment = "e5_preprocessing_a_batched_dnn";
    ctx.notes = "FKL CPU back-end vs OpenCV on the CPU, batched DNN pre-processing (cvGPUSpeedup test_batchresize_x_split3D chain)";
    ctx.cvThreadsDefault = cv::getNumThreads();
    ctx.fklPoolSize = fk::cpuThreadPool().size();

    const char* imagePath = std::getenv("E5_IMAGE");
    const char* imageInfo = std::getenv("E5_IMAGE_INFO");
    const Source source(imagePath);
    std::vector<Row> rows;
    std::vector<Agreement> agreements;
    runSweep(source, ctx, warmup, iters, checkOnly, rows, agreements,
             std::make_index_sequence<sizeof(BATCHES) / sizeof(BATCHES[0])>{});

    bool allPass = true;
    for (const auto& a : agreements) {
        allPass &= a.pass || !a.required;
    }
    {
        std::ofstream os(outDir + "e5_preprocessing_a_agreement_CPU.csv");
        e5::writeSystemInfoHeader(os, ctx);
        os << "# source image: " << source.description << (imageInfo ? std::string("; ") + imageInfo : "") << "\n";
        os << "# tolerance: " << TOLERANCE << " absolute (the tolerance of test_batchresize_x_split3D); identical = byte for byte\n";
        os << "batch, comparison, max_abs_diff, mean_abs_diff, fraction_over_1e-4, pass, required\n";
        for (const auto& a : agreements) {
            os << a.batch << ", " << a.pair << ", " << a.maxAbsDiff << ", " << a.meanAbsDiff << ", " << a.fractionOverTolerance << ", " << (a.pass ? 1 : 0) << ", " << (a.required ? 1 : 0)
               << "\n";
        }
    }
    if (!checkOnly) {
        std::ofstream os(outDir + "e5_preprocessing_a_CPU.csv");
        e5::writeSystemInfoHeader(os, ctx);
        os << "# timing: std::chrono::steady_clock wall clock of one full pipeline run (all crops of the batch), "
           << warmup << " warm-up runs then " << iters << " timed runs per point; times in ms\n";
        os << "# source: " << SRC_W << "x" << SRC_H << " 8UC3 (" << source.description << (imageInfo ? std::string("; ") + imageInfo : "") << "), crops " << CROP_W << "x" << CROP_H
           << " at (i, i), output " << UP_W << "x" << UP_H << " 32FC3, NCHW tensor\n";
        os << "batch, variant, threads, median_ms, min_ms, max_ms, mean_ms, iters\n";
        for (const auto& r : rows) {
            os << r.batch << ", " << r.variant << ", " << r.threads << ", " << r.stats.median << ", " << r.stats.min << ", "
               << r.stats.max << ", " << r.stats.mean << ", " << r.stats.iters << "\n";
        }
    }
    std::cout << (allPass ? "all numerical checks passed" : "NUMERICAL CHECK FAILED") << std::endl;
    return allPass ? 0 : -1;
}
