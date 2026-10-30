// E5, resolution sweep (c): the 7-operation chain of pipeline (a) (resize, multiply by 0.3, RGB to BGR, subtract {1, 4, 3.2}, divide
// by {3.2, 0.6, 11.8}, split to an NCHW float tensor) on whole frames, one image per run (batch 1), at 480p, 1080p and 4K; the
// output is 64/60 x 128/120 times the size of the input (the ratio of pipeline (a)). A frame is a working set that no longer fits in the caches (4K: 25 MB in, 113 MB out), the regime that the 60x120 crops of pipeline (a) never
// reach. FKL (ParArch::CPU and ParArch::CPU_MULTITHREAD, fused and one execution per operation) against OpenCV (default threads,
// one thread). Wall clock, warm-up runs then timed runs, median/min/max/mean; the same checks as pipeline (a).
// Environment variables: E5_ITERS (default 30), E5_WARMUP (3), E5_OUT_DIR (default "."), E5_CHECK_ONLY=1.
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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

using F3 = fk::float3;
using U3 = fk::uchar3;
using Rgb2Bgr = fk::ColorConversion<fk::ColorConversionCodes::COLOR_RGB2BGR, F3, F3>;
using StreamST = fk::Stream_<fk::ParArch::CPU>;
using StreamMT = fk::Stream_<fk::ParArch::CPU_MULTITHREAD>;
using DppST = fk::TransformDPP<fk::ParArch::CPU>;
using DppMT = fk::TransformDPP<fk::ParArch::CPU_MULTITHREAD>;

constexpr double TOLERANCE = 1e-4;
struct Resolution {
    const char* name;
    int w, h;
};
constexpr Resolution RESOLUTIONS[] = {{"480p", 720, 480}, {"1080p", 1920, 1080}, {"4K", 3840, 2160}};

const cv::Scalar ALPHA_S(0.3, 0.3, 0.3), SUB_S(1.f, 4.f, 3.2f), DIV_S(3.2f, 0.6f, 11.8f);

int envInt(const char* name, const int fallback) {
    const char* value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

inline auto mulOp() { return fk::Mul<F3>::build(F3{0.3f, 0.3f, 0.3f}); }
inline auto subOp() { return fk::Sub<F3>::build(F3{1.f, 4.f, 3.2f}); }
inline auto divOp() { return fk::Div<F3>::build(F3{3.2f, 0.6f, 11.8f}); }
inline auto colorOp() { return Rgb2Bgr::build(); }

// One resolution: the source frame, the output size and every implementation
struct Case {
    int w, h, ow, oh;
    fk::Ptr2D<U3> image;
    cv::Mat mat;
    fk::Tensor<float> tFusedST, tFusedMT, tOpST, tOpMT;
    fk::Ptr2D<F3> a, b;
    cv::Mat tensorOcv, resized, temp, temp2;
    std::vector<cv::Mat> planes;
    StreamST streamST;
    StreamMT streamMT;

    Case(const Resolution& r)
        : w(r.w), h(r.h), ow(static_cast<int>(std::lround(r.w * 64.0 / 60.0))), oh(static_cast<int>(std::lround(r.h * 128.0 / 120.0))),
          image(r.w, r.h, 0, fk::MemType::Host), tFusedST(ow, oh, 1, 3, fk::MemType::Host), tFusedMT(ow, oh, 1, 3, fk::MemType::Host),
          tOpST(ow, oh, 1, 3, fk::MemType::Host), tOpMT(ow, oh, 1, 3, fk::MemType::Host), a(ow, oh, 0, fk::MemType::Host),
          b(ow, oh, 0, fk::MemType::Host), tensorOcv(1, ow * oh * 3, CV_32F), resized(oh, ow, CV_32FC3), temp(oh, ow, CV_32FC3),
          temp2(oh, ow, CV_32FC3) {
        const fk::RawPtr<fk::ND::_2D, U3> raw = image.ptr();
        uint32_t state = 12345u;
        for (int y = 0; y < h; ++y) {
            U3* row = reinterpret_cast<U3*>(reinterpret_cast<char*>(raw.data) + static_cast<size_t>(y) * raw.dims.pitch);
            for (int x = 0; x < w; ++x) {
                state = state * 1664525u + 1013904223u;
                row[x] = fk::make_<U3>(static_cast<uchar>(state >> 24), static_cast<uchar>(state >> 16), static_cast<uchar>(state >> 8));
            }
        }
        mat = cv::Mat(h, w, CV_8UC3, raw.data, raw.dims.pitch);
        float* base = tensorOcv.ptr<float>(0);
        const size_t plane = static_cast<size_t>(ow) * oh;
        planes = {cv::Mat(oh, ow, CV_32F, base), cv::Mat(oh, ow, CV_32F, base + plane), cv::Mat(oh, ow, CV_32F, base + 2 * plane)};
    }

    void opencv() {
        cv::Mat crop32F;
        mat.convertTo(crop32F, CV_32FC3);
        cv::resize(crop32F, resized, cv::Size(ow, oh), 0., 0., cv::INTER_LINEAR);
        cv::multiply(resized, ALPHA_S, temp);
        cv::cvtColor(temp, temp, cv::COLOR_RGB2BGR);
        cv::subtract(temp, SUB_S, temp2);
        cv::divide(temp2, DIV_S, temp);
        cv::split(temp, planes);
    }
    // The chain in one executeOperations
    template <typename Dpp, typename StreamT>
    void fused(StreamT& stream, fk::Tensor<float>& out) {
        fk::executeOperations<Dpp>(stream, fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::PerThreadRead<fk::ND::_2D, U3>::build(image.ptr()), fk::Size(ow, oh)),
                                   colorOp(), mulOp(), subOp(), divOp(), fk::TensorSplit<F3>::build(out.ptr()));
        stream.sync();
    }
    // One execution per operation, through two host Ptr2D
    template <typename Dpp, typename StreamT>
    void perOp(StreamT& stream, fk::Tensor<float>& out) {
        fk::executeOperations<Dpp>(stream, fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::PerThreadRead<fk::ND::_2D, U3>::build(image.ptr()), fk::Size(ow, oh)),
                                   fk::PerThreadWrite<fk::ND::_2D, F3>::build(a.ptr()));
        fk::executeOperations<Dpp>(stream, fk::PerThreadRead<fk::ND::_2D, F3>::build(a.ptr()), mulOp(), fk::PerThreadWrite<fk::ND::_2D, F3>::build(b.ptr()));
        fk::executeOperations<Dpp>(stream, fk::PerThreadRead<fk::ND::_2D, F3>::build(b.ptr()), colorOp(), fk::PerThreadWrite<fk::ND::_2D, F3>::build(a.ptr()));
        fk::executeOperations<Dpp>(stream, fk::PerThreadRead<fk::ND::_2D, F3>::build(a.ptr()), subOp(), fk::PerThreadWrite<fk::ND::_2D, F3>::build(b.ptr()));
        fk::executeOperations<Dpp>(stream, fk::PerThreadRead<fk::ND::_2D, F3>::build(b.ptr()), divOp(), fk::PerThreadWrite<fk::ND::_2D, F3>::build(a.ptr()));
        fk::executeOperations<Dpp>(stream, fk::PerThreadRead<fk::ND::_2D, F3>::build(a.ptr()), fk::TensorSplit<F3>::build(out.ptr()));
        stream.sync();
    }
    std::vector<float> reference(const bool halfPixel) const {
        const size_t plane = static_cast<size_t>(ow) * oh;
        std::vector<float> out(plane * 3);
        std::vector<double> r(plane * 3);
        const fk::RawPtr<fk::ND::_2D, U3> raw = image.ptr();
        fklb::resizeBilinear(reinterpret_cast<const uchar*>(raw.data), raw.dims.pitch, w, h, 3, ow, oh,
                             halfPixel ? fklb::ResizeSampling::HALF_PIXEL : fklb::ResizeSampling::ASYMMETRIC, r.data());
        const double alpha = static_cast<double>(0.3f), sub[3] = {1.0, 4.0, static_cast<double>(3.2f)},
                     div[3] = {static_cast<double>(3.2f), static_cast<double>(0.6f), static_cast<double>(11.8f)};
        for (size_t p = 0; p < plane; ++p) {
            for (int c = 0; c < 3; ++c) {
                const int oc = 2 - c;
                out[static_cast<size_t>(oc) * plane + p] = static_cast<float>((r[p * 3 + c] * alpha - sub[oc]) / div[oc]);
            }
        }
        return out;
    }
    size_t values() const { return static_cast<size_t>(ow) * oh * 3; }
};

const float* hostPtr(const fk::Tensor<float>& t) { return t.ptr().data; }

double maxAbsDiff(const float* a, const float* b, const size_t n) {
    double worst = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (!(d <= worst)) {
            worst = d;
        }
    }
    return worst;
}

struct Row {
    std::string resolution, variant;
    int threads;
    e5::Stats stats;
};
struct Agreement {
    std::string resolution, pair;
    double maxAbsDiff;
    bool pass, required;
};

} // namespace

int launch() {
    const int iters = envInt("E5_ITERS", 30);
    const int warmup = envInt("E5_WARMUP", 3);
    const bool checkOnly = envInt("E5_CHECK_ONLY", 0) != 0;
    const char* outDirEnv = std::getenv("E5_OUT_DIR");
    const std::string outDir = outDirEnv ? std::string(outDirEnv) + "/" : "";
    e5::Context ctx;
    ctx.experiment = "e5_preprocessing_c_resolution";
    ctx.notes = "FKL CPU back-end vs OpenCV on the CPU, the chain of pipeline (a) on whole frames, batch 1, output 64/60 x 128/120 of the input; seeded noise";
    ctx.cvThreadsDefault = cv::getNumThreads();
    ctx.fklPoolSize = fk::cpuThreadPool().size();
    std::vector<Row> rows;
    std::vector<Agreement> agreements;
    bool allPass = true;

    for (const Resolution& res : RESOLUTIONS) {
        std::cout << res.name << " " << res.w << "x" << res.h << std::endl;
        Case c(res);
        const auto compare = [&](const std::string& pair, const float* a, const float* b, const bool exact, const bool required = true) {
            const double d = maxAbsDiff(a, b, c.values());
            const bool pass = exact ? std::memcmp(a, b, c.values() * sizeof(float)) == 0 : d <= TOLERANCE;
            agreements.push_back({res.name, pair, d, pass, required});
            allPass &= pass || !required;
            std::cout << "  " << pair << ": max abs diff " << d << (required ? (pass ? " ok" : " FAIL") : " (informational)") << std::endl;
        };
        cv::setNumThreads(ctx.cvThreadsDefault);
        c.opencv();
        const std::vector<float> ocvDefault(c.tensorOcv.ptr<float>(0), c.tensorOcv.ptr<float>(0) + c.values());
        cv::setNumThreads(1);
        c.opencv();
        const bool same = std::memcmp(ocvDefault.data(), c.tensorOcv.ptr<float>(0), c.values() * sizeof(float)) == 0;
        cv::setNumThreads(ctx.cvThreadsDefault);
        c.fused<DppST>(c.streamST, c.tFusedST);
        c.fused<DppMT>(c.streamMT, c.tFusedMT);
        c.perOp<DppST>(c.streamST, c.tOpST);
        c.perOp<DppMT>(c.streamMT, c.tOpMT);
        const std::vector<float> refAsym = c.reference(false), refHalf = c.reference(true);
        agreements.push_back({res.name, "opencv_default vs opencv_1thread (identical)", same ? 0.0 : 1.0, same, true});
        allPass &= same;
        compare("opencv vs scalar reference, HALF_PIXEL sampling (<= 1e-4)", ocvDefault.data(), refHalf.data(), false);
        compare("fkl_fused_cpu vs scalar reference, ASYMMETRIC sampling (<= 1e-4)", hostPtr(c.tFusedST), refAsym.data(), false);
        compare("fkl_fused_cpu_mt vs fkl_fused_cpu (identical)", hostPtr(c.tFusedMT), hostPtr(c.tFusedST), true);
        compare("fkl_perop_cpu vs fkl_fused_cpu (<= 1e-4)", hostPtr(c.tOpST), hostPtr(c.tFusedST), false);
        compare("fkl_perop_cpu_mt vs fkl_perop_cpu (identical)", hostPtr(c.tOpMT), hostPtr(c.tOpST), true);
        compare("fkl_fused_cpu vs opencv (the sampling conventions)", hostPtr(c.tFusedST), ocvDefault.data(), false, false);
        if (checkOnly) {
            continue;
        }
        const auto time = [&](const std::string& variant, const int threads, const std::function<void()>& fn) {
            const e5::Stats s = e5::measure(fn, warmup, iters);
            rows.push_back({res.name, variant, threads, s});
            std::cout << "  " << variant << ": median " << s.median << " ms (min " << s.min << ", max " << s.max << ")" << std::endl;
        };
        const int pool = static_cast<int>(ctx.fklPoolSize);
        cv::setNumThreads(ctx.cvThreadsDefault);
        time("opencv_default", ctx.cvThreadsDefault, [&] { c.opencv(); });
        cv::setNumThreads(1);
        time("opencv_1thread", 1, [&] { c.opencv(); });
        cv::setNumThreads(ctx.cvThreadsDefault);
        time("fkl_perop_cpu", 1, [&] { c.perOp<DppST>(c.streamST, c.tOpST); });
        time("fkl_fused_cpu", 1, [&] { c.fused<DppST>(c.streamST, c.tFusedST); });
        time("fkl_perop_cpu_mt", pool, [&] { c.perOp<DppMT>(c.streamMT, c.tOpMT); });
        time("fkl_fused_cpu_mt", pool, [&] { c.fused<DppMT>(c.streamMT, c.tFusedMT); });
    }

    {
        std::ofstream os(outDir + "e5_preprocessing_c_agreement_CPU.csv");
        e5::writeSystemInfoHeader(os, ctx);
        os << "# tolerance: " << TOLERANCE << " absolute; identical = byte for byte\n";
        os << "resolution, comparison, max_abs_diff, pass, required\n";
        for (const auto& a : agreements) {
            os << a.resolution << ", " << a.pair << ", " << a.maxAbsDiff << ", " << (a.pass ? 1 : 0) << ", " << (a.required ? 1 : 0) << "\n";
        }
    }
    if (!checkOnly) {
        std::ofstream os(outDir + "e5_preprocessing_c_CPU.csv");
        e5::writeSystemInfoHeader(os, ctx);
        os << "# timing: std::chrono::steady_clock wall clock of one run of the chain on one frame, " << warmup << " warm-up runs then " << iters
           << " timed runs per resolution and variant; times in ms\n";
        os << "# frames: 720x480, 1920x1080, 3840x2160 8UC3 (seeded noise), output 64/60 x 128/120 of the input (768x512, 2048x1152, 4096x2304; the scale factor 15/16 is exact in float, any other factor adds the float rounding of the source positions of FKL) 32FC3 NCHW\n";
        os << "resolution, variant, threads, median_ms, min_ms, max_ms, mean_ms, iters\n";
        for (const auto& r : rows) {
            os << r.resolution << ", " << r.variant << ", " << r.threads << ", " << r.stats.median << ", " << r.stats.min << ", "
               << r.stats.max << ", " << r.stats.mean << ", " << r.stats.iters << "\n";
        }
    }
    std::cout << (allPass ? "all numerical checks passed" : "NUMERICAL CHECK FAILED") << std::endl;
    return allPass ? 0 : -1;
}
