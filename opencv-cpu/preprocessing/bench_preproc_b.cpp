// E5, pipeline (b): the NV12 video pre-processing pipeline on the CPU, FKL (ParArch::CPU and ParArch::CPU_MULTITHREAD) against
// OpenCV. A 3840x2160 NV12 frame, N detection crops (N = 1, 8, 16, 32, 64; fkl-self/pipelines/nv12_pipeline.h):
//   FKL      kernel 1 NV12 -> RGB (ReadYUV + ConvertYUVToRGB), kernel 2 the crops resized to 64x128, normalised and split to an
//            NCHW float tensor: fused into one executeOperations, one per operation for all the crops, or one per operation and
//            per crop (the same code as the CUDA and HIP target of fkl-self/pipelines)
//   OpenCV   cv::cvtColor(COLOR_YUV2RGB_NV12), then per crop: convertTo, resize, multiply, subtract, divide, split into the tensor
//            (opencv_percrop_*: default threads and one thread; opencv_percrop_parallel: the crops in parallel with
//            cv::parallel_for_)
// Wall clock (std::chrono::steady_clock): warm-up runs, then timed runs; median, min, max and mean per point. The checks before
// timing: the FKL chain against a scalar double precision reference (FKL sampling) computed from the FKL RGB frame, the
// OpenCV tensor against one with pixel-centre sampling computed from the OpenCV RGB frame (both 1e-4), single against
// multithreaded FKL byte for byte; the RGB frames of FKL and OpenCV (levels) and the two tensors (the sampling conventions
// differ, see common/resize_reference.h) are reported as informational.
// Environment variables: E5_ITERS (default 30), E5_WARMUP (3), E5_OUT_DIR (default "."), E5_CHECK_ONLY=1, E5_IMAGE (a binary PPM
// tiled over the frame and converted to NV12 instead of seeded noise), E5_IMAGE_INFO (text for the header).
#include "e5_common.h"

#include <common/main.h>
#include <fkl-self/pipelines/nv12_pipeline.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double TOLERANCE = 1e-4;
constexpr int BATCHES[] = {1, 8, 16, 32, 64};
using StreamST = fk::Stream_<fk::ParArch::CPU>;
using StreamMT = fk::Stream_<fk::ParArch::CPU_MULTITHREAD>;

int envInt(const char* name, const int fallback) {
    const char* value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

struct Row {
    int crops;
    std::string variant;
    int threads;
    nv12::Stats stats;
};

struct Agreement {
    int crops;
    std::string pair;
    double maxAbsDiff, meanAbsDiff, fractionOver;
    bool pass, required;
};

// OpenCV: cv::cvtColor on the NV12 frame, then the crops one by one (or in parallel)
template <int N>
class OpenCvNv12 {
public:
    OpenCvNv12(const nv12::Frame& frame, const std::array<fk::Rect, N>& crops)
        : crops_(crops), rgb_(nv12::FRAME_H, nv12::FRAME_W, CV_8UC3), tensor_(N, nv12::PLANE * 3, CV_32F) {
        const auto raw = frame.getData().ptr();
        yuv_ = cv::Mat(nv12::FRAME_H * 3 / 2, nv12::FRAME_W, CV_8UC1, raw.data, raw.dims.pitch);
        for (int i = 0; i < N; ++i) {
            float* const base = tensor_.ptr<float>(i);
            planes_.push_back({cv::Mat(nv12::OUT_H, nv12::OUT_W, CV_32F, base),
                               cv::Mat(nv12::OUT_H, nv12::OUT_W, CV_32F, base + nv12::PLANE),
                               cv::Mat(nv12::OUT_H, nv12::OUT_W, CV_32F, base + 2 * nv12::PLANE)});
        }
    }
    void convert() { cv::cvtColor(yuv_, rgb_, cv::COLOR_YUV2RGB_NV12); }
    void chain() {
        Scratch& s = scratchMain_;
        for (int i = 0; i < N; ++i) {
            chainCrop(i, s);
        }
    }
    void chainParallel() {
        cv::parallel_for_(cv::Range(0, N), [this](const cv::Range& range) {
            thread_local Scratch s;
            for (int i = range.start; i < range.end; ++i) {
                chainCrop(i, s);
            }
        });
    }
    const float* tensor() const { return tensor_.ptr<float>(0); }
    const cv::Mat& rgb() const { return rgb_; }

private:
    struct Scratch {
        cv::Mat crop32F, resized, norm;
    };
    void chainCrop(const int i, Scratch& s) {
        const fk::Rect& r = crops_[i];
        rgb_(cv::Rect(static_cast<int>(r.x), static_cast<int>(r.y), r.width, r.height)).convertTo(s.crop32F, CV_32FC3);
        cv::resize(s.crop32F, s.resized, cv::Size(nv12::OUT_W, nv12::OUT_H), 0., 0., cv::INTER_LINEAR);
        cv::multiply(s.resized, cv::Scalar(1.f / 255.f, 1.f / 255.f, 1.f / 255.f), s.norm);
        cv::subtract(s.norm, cv::Scalar(nv12::MEAN[0], nv12::MEAN[1], nv12::MEAN[2]), s.resized);
        cv::divide(s.resized, cv::Scalar(nv12::STD[0], nv12::STD[1], nv12::STD[2]), s.norm);
        cv::split(s.norm, planes_[i]);
    }
    std::array<fk::Rect, N> crops_;
    cv::Mat yuv_, rgb_, tensor_;
    std::vector<std::vector<cv::Mat>> planes_;
    Scratch scratchMain_;
};

double meanAbsDiff(const float* a, const float* b, const size_t n) {
    double sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    }
    return sum / static_cast<double>(n);
}

double fractionOver(const float* a, const float* b, const size_t n, const double tol) {
    size_t over = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!(std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])) <= tol)) {
            ++over;
        }
    }
    return static_cast<double>(over) / static_cast<double>(n);
}

// The scalar reference of kernel 2 with pixel-centre sampling, from an OpenCV RGB frame
template <int N>
std::vector<float> referenceOpenCv(const cv::Mat& rgb, const std::array<fk::Rect, N>& crops) {
    std::vector<float> out(static_cast<size_t>(N) * 3 * nv12::PLANE);
    std::vector<double> resized(static_cast<size_t>(nv12::PLANE) * 3);
    for (int i = 0; i < N; ++i) {
        const fk::Rect& r = crops[i];
        fklb::resizeBilinear(rgb.ptr<uchar>(static_cast<int>(r.y)) + static_cast<size_t>(r.x) * 3, rgb.step[0], r.width, r.height, 3,
                             nv12::OUT_W, nv12::OUT_H, fklb::ResizeSampling::PIXEL_CENTRE, resized.data());
        for (int p = 0; p < nv12::PLANE; ++p) {
            for (int c = 0; c < 3; ++c) {
                out[(static_cast<size_t>(i) * 3 + c) * nv12::PLANE + p] = static_cast<float>(
                    (resized[static_cast<size_t>(p) * 3 + c] * static_cast<double>(1.f / 255.f) - static_cast<double>(nv12::MEAN[c])) /
                    static_cast<double>(nv12::STD[c]));
            }
        }
    }
    return out;
}

template <int N>
void runPoint(nv12::Frame& frame, fk::Ptr2D<nv12::U3>& rgb, const e5::Context& ctx, const int warmup, const int iters,
              const bool checkOnly, std::vector<Row>& rows, std::vector<Agreement>& agreements) {
    constexpr size_t NV = static_cast<size_t>(N) * 3 * nv12::PLANE;
    std::cout << "crops " << N << std::endl;
    const auto crops = nv12::makeCrops<N>();
    StreamST streamST;
    StreamMT streamMT;
    nv12::Pipeline<fk::ParArch::CPU, N> st(streamST, frame, rgb, crops);
    nv12::Pipeline<fk::ParArch::CPU_MULTITHREAD, N> mt(streamMT, frame, rgb, crops);
    OpenCvNv12<N> ocv(frame, crops);

    // Agreement
    const int defaultThreads = ctx.cvThreadsDefault;
    cv::setNumThreads(defaultThreads);
    ocv.convert();
    ocv.chain();
    const std::vector<float> ocvTensor(ocv.tensor(), ocv.tensor() + NV);
    ocv.chainParallel();
    const bool parallelSame = std::memcmp(ocvTensor.data(), ocv.tensor(), NV * sizeof(float)) == 0;
    cv::Mat ocvRgb = ocv.rgb().clone();
    st.convert();
    st.fused();
    st.perOpBatch();
    st.perOpCrop();
    st.download();
    st.sync();
    const std::vector<float> fusedST(st.fusedOut(), st.fusedOut() + NV);
    const std::vector<float> batchST(st.batchOut(), st.batchOut() + NV);
    const std::vector<float> cropST(st.cropOut(), st.cropOut() + NV);
    mt.convert();
    mt.fused();
    mt.perOpBatch();
    mt.perOpCrop();
    mt.download();
    mt.sync();
    const std::vector<float> refFkl = nv12::referenceTensor<N>(nv12::hostRaw(rgb), crops);
    const std::vector<float> refOcv = referenceOpenCv<N>(ocvRgb, crops);
    const auto compare = [&](const std::string& pair, const float* a, const float* b, const bool exact, const bool required = true) {
        const double d = nv12::maxAbsDiff(a, b, NV);
        const bool pass = exact ? std::memcmp(a, b, NV * sizeof(float)) == 0 : d <= TOLERANCE;
        agreements.push_back({N, pair, d, meanAbsDiff(a, b, NV), fractionOver(a, b, NV, TOLERANCE), pass, required});
        std::cout << "  " << pair << ": max abs diff " << d << (required ? (pass ? " ok" : " FAIL") : " (informational)") << std::endl;
    };
    // RGB frames, in levels
    {
        const fk::RawPtr<fk::ND::_2D, nv12::U3> f = nv12::hostRaw(rgb);
        double worst = 0, sum = 0;
        size_t over = 0;
        for (int y = 0; y < nv12::FRAME_H; ++y) {
            const nv12::U3* row = reinterpret_cast<const nv12::U3*>(reinterpret_cast<const char*>(f.data) + static_cast<size_t>(y) * f.dims.pitch);
            const uchar* orow = ocvRgb.ptr<uchar>(y);
            for (int x = 0; x < nv12::FRAME_W; ++x) {
                const int d0 = std::abs(int(row[x].x) - int(orow[3 * x])), d1 = std::abs(int(row[x].y) - int(orow[3 * x + 1])),
                          d2 = std::abs(int(row[x].z) - int(orow[3 * x + 2]));
                const int d = std::max(d0, std::max(d1, d2));
                worst = std::max(worst, static_cast<double>(d));
                sum += d0 + d1 + d2;
                over += d > 0;
            }
        }
        const double pixels = static_cast<double>(nv12::FRAME_W) * nv12::FRAME_H;
        agreements.push_back({N, "RGB frame: FKL ConvertYUVToRGB vs cv::cvtColor YUV2RGB_NV12 (levels, informational)", worst,
                              sum / (3 * pixels), static_cast<double>(over) / pixels, true, false});
        std::cout << "  RGB frame FKL vs OpenCV: max " << worst << " levels, mean " << sum / (3 * pixels) << ", pixels that differ "
                  << static_cast<double>(over) / pixels << " (informational)" << std::endl;
    }
    compare("opencv_percrop_parallel vs opencv_percrop (identical)", ocv.tensor(), ocvTensor.data(), true);
    (void)parallelSame;
    compare("opencv_percrop vs scalar reference, pixel-centre sampling (<= 1e-4)", ocvTensor.data(), refOcv.data(), false);
    compare("fkl_chain_fused_cpu vs scalar reference, FKL sampling (<= 1e-4)", fusedST.data(), refFkl.data(), false);
    compare("fkl_chain_perop_batch_cpu vs fkl_chain_fused_cpu (<= 1e-4)", batchST.data(), fusedST.data(), false);
    compare("fkl_chain_perop_crop_cpu vs fkl_chain_fused_cpu (<= 1e-4)", cropST.data(), fusedST.data(), false);
    compare("fkl_chain_fused_cpu_mt vs fkl_chain_fused_cpu (identical)", mt.fusedOut(), fusedST.data(), true);
    compare("fkl_chain_perop_batch_cpu_mt vs fkl_chain_perop_batch_cpu (identical)", mt.batchOut(), batchST.data(), true);
    compare("fkl_chain_perop_crop_cpu_mt vs fkl_chain_perop_crop_cpu (identical)", mt.cropOut(), cropST.data(), true);
    compare("fkl_chain_fused_cpu vs opencv_percrop (RGB frame and sampling convention differ)", fusedST.data(), ocvTensor.data(), false, false);
    if (checkOnly) {
        return;
    }

    const auto time = [&](const std::string& variant, const int threads, const std::function<void()>& fn) {
        const nv12::Stats s = nv12::measure(fn, warmup, iters);
        rows.push_back({N, variant, threads, s});
        std::cout << "  " << variant << ": median " << s.median << " ms (min " << s.min << ", max " << s.max << ")" << std::endl;
    };
    const int pool = static_cast<int>(ctx.fklPoolSize);
    cv::setNumThreads(defaultThreads);
    time("opencv_convert_default", defaultThreads, [&] { ocv.convert(); });
    time("opencv_pipeline_percrop_default", defaultThreads, [&] { ocv.convert(); ocv.chain(); });
    time("opencv_pipeline_percrop_parallel", defaultThreads, [&] { ocv.convert(); ocv.chainParallel(); });
    cv::setNumThreads(1);
    time("opencv_convert_1thread", 1, [&] { ocv.convert(); });
    time("opencv_pipeline_percrop_1thread", 1, [&] { ocv.convert(); ocv.chain(); });
    cv::setNumThreads(defaultThreads);
    time("fkl_convert_cpu", 1, [&] { st.convert(); st.sync(); });
    time("fkl_chain_fused_cpu", 1, [&] { st.fused(); st.sync(); });
    time("fkl_pipeline_fused_cpu", 1, [&] { st.convert(); st.fused(); st.sync(); });
    time("fkl_pipeline_perop_batch_cpu", 1, [&] { st.convert(); st.perOpBatch(); st.sync(); });
    time("fkl_pipeline_perop_crop_cpu", 1, [&] { st.convert(); st.perOpCrop(); st.sync(); });
    time("fkl_convert_cpu_mt", pool, [&] { mt.convert(); mt.sync(); });
    time("fkl_chain_fused_cpu_mt", pool, [&] { mt.fused(); mt.sync(); });
    time("fkl_pipeline_fused_cpu_mt", pool, [&] { mt.convert(); mt.fused(); mt.sync(); });
    time("fkl_pipeline_perop_batch_cpu_mt", pool, [&] { mt.convert(); mt.perOpBatch(); mt.sync(); });
    time("fkl_pipeline_perop_crop_cpu_mt", pool, [&] { mt.convert(); mt.perOpCrop(); mt.sync(); });
}

template <size_t... Is>
void runSweep(nv12::Frame& frame, fk::Ptr2D<nv12::U3>& rgb, const e5::Context& ctx, const int warmup, const int iters,
              const bool checkOnly, std::vector<Row>& rows, std::vector<Agreement>& agreements, std::index_sequence<Is...>) {
    // The compile-time batch sizes, as a fold over a parameter pack of constants
    (void)std::initializer_list<int>{(runPoint<BATCHES[Is]>(frame, rgb, ctx, warmup, iters, checkOnly, rows, agreements), 0)...};
}

} // namespace

int launch() {
    const int iters = envInt("E5_ITERS", 30);
    const int warmup = envInt("E5_WARMUP", 3);
    const bool checkOnly = envInt("E5_CHECK_ONLY", 0) != 0;
    const char* outDirEnv = std::getenv("E5_OUT_DIR");
    const std::string outDir = outDirEnv ? std::string(outDirEnv) + "/" : "";
    const char* imagePath = std::getenv("E5_IMAGE");
    const char* imageInfo = std::getenv("E5_IMAGE_INFO");

    nv12::Frame frame(nv12::FRAME_W, nv12::FRAME_H, fk::MemType::Host);
    std::string source;
    if (imagePath != nullptr) {
        std::vector<uchar> image;
        int w = 0, h = 0;
        if (!nv12::readPpm(imagePath, image, w, h)) {
            std::cerr << "Cannot read the binary PPM " << imagePath << std::endl;
            return -1;
        }
        nv12::fillFromRgb(frame, image, w, h);
        source = std::string("natural image ") + imagePath + " (" + std::to_string(w) + "x" + std::to_string(h) +
                 ") tiled over the frame, converted to NV12 (BT.601 studio swing, chroma averaged over 2x2)" +
                 (imageInfo ? std::string("; ") + imageInfo : "");
    } else {
        nv12::fillNoise(frame);
        source = "seeded noise (LCG, seed 424242), luma 16..235, chroma 16..240";
    }
    fk::Ptr2D<nv12::U3> rgb(nv12::FRAME_W, nv12::FRAME_H, 0, fk::MemType::Host);

    e5::Context ctx;
    ctx.experiment = "e5_preprocessing_b_nv12";
    ctx.notes = "FKL CPU back-end vs OpenCV on the CPU, NV12 video pre-processing (E4b pipeline); source: " + source;
    ctx.cvThreadsDefault = cv::getNumThreads();
    ctx.fklPoolSize = fk::cpuThreadPool().size();

    std::vector<Row> rows;
    std::vector<Agreement> agreements;
    runSweep(frame, rgb, ctx, warmup, iters, checkOnly, rows, agreements, std::make_index_sequence<sizeof(BATCHES) / sizeof(BATCHES[0])>{});

    bool allPass = true;
    for (const auto& a : agreements) {
        allPass &= a.pass || !a.required;
    }
    {
        std::ofstream os(outDir + "e5_preprocessing_b_agreement_CPU.csv");
        e5::writeSystemInfoHeader(os, ctx);
        os << "# tolerance: " << TOLERANCE << " absolute; identical = byte for byte; RGB frame rows in levels (0..255)\n";
        os << "crops, comparison, max_abs_diff, mean_abs_diff, fraction_over_1e-4, pass, required\n";
        for (const auto& a : agreements) {
            os << a.crops << ", " << a.pair << ", " << a.maxAbsDiff << ", " << a.meanAbsDiff << ", " << a.fractionOver << ", "
               << (a.pass ? 1 : 0) << ", " << (a.required ? 1 : 0) << "\n";
        }
    }
    if (!checkOnly) {
        std::ofstream os(outDir + "e5_preprocessing_b_CPU.csv");
        e5::writeSystemInfoHeader(os, ctx);
        os << "# timing: std::chrono::steady_clock wall clock of one run including the synchronisation of the stream, " << warmup
           << " warm-up runs then " << iters << " timed runs per point; times in ms; 'convert' = NV12 to RGB frame, 'chain' = crops, resize,"
              " normalise, split (kernel 2), 'pipeline' = both\n";
        os << "# frame: " << nv12::FRAME_W << "x" << nv12::FRAME_H << " NV12 (the frame is in host memory), crops of 48..176 x 96..352 pixels "
              "(seeded), output " << nv12::OUT_W << "x" << nv12::OUT_H << " 32FC3 NCHW\n";
        os << "crops, variant, threads, median_ms, min_ms, max_ms, mean_ms, iters\n";
        for (const auto& r : rows) {
            os << r.crops << ", " << r.variant << ", " << r.threads << ", " << r.stats.median << ", " << r.stats.min << ", "
               << r.stats.max << ", " << r.stats.mean << ", " << r.stats.iters << "\n";
        }
    }
    std::cout << (allPass ? "all numerical checks passed" : "NUMERICAL CHECK FAILED") << std::endl;
    return allPass ? 0 : -1;
}
