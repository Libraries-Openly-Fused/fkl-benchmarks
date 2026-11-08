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

// benchmark_vfhf_max_speedup: the maximum speedup of vertical and horizontal fusion together on one GPU, FKL against itself (the RX 9060 as a
// system of the GPU-size figure). It follows the 2025 program benchmark_vertical_fusionCompareMulAdd_cu (opencv-cuda/verticalfusion/,
// vertical_fusion_kernel_instances/mul_add), which is not run, with FKL in place of OpenCV-CUDA on the side that launches one kernel per
// operation: there is no OpenCV-CUDA on an AMD GPU.
//
// Data (vfhf_common.h): 50 images of 60x120 8 bit pixels, all 30, each one a fk::Ptr2D of its own; each side has its own 50 images.
// Sides, for NUM_OPS operations:
//   fused     vfhf::fused<NUM_OPS> (vfhf_fused.h): one executeOperations, thread fusion off: the batch read of the 50 images, the conversion to float
//             (SaturateCast, then Mul by alpha = 1), the chain of NUM_OPS operations (fk::StaticLoop of the pair Mul(0.54), Add(0.54), nested as in
//             the 2025 program) and the write into a float tensor of 50 planes of 60x120. 1 call per repetition.
//   baseline  one executeOperations per operation and per image, as the OpenCV-CUDA side of the 2025 program (convertTo, then NUM_OPS / 2 times
//             multiply and add, in place, image after image): for each image the conversion into its own float image, then NUM_OPS calls that
//             read that image and write it in place, Mul(0.54) and Add(0.54) in turn. 50 (1 + NUM_OPS) calls per repetition. Every operation is
//             built in its call.
// Points: the numbers of operations of the build (FKLB_VFHF_POINTS in fkl-self/vfhf/CMakeLists.txt, each 2 + 100 (k - 1)); the fused side of
// each point is compiled in a generated translation unit of its own.
// Timing, per point and side, on the side's own stream: an event recorded before the first call of a repetition and one after its last call,
// then a synchronisation of the second event; the time between the two events, in ms. The stream is idle when the first event is recorded, so
// the time includes the host's building and launching of the calls.
//   Warm-up: at least VFHF_WARMUP_SECONDS (2.0) s and 2 repetitions, then until the time stops changing: while a repetition takes less than
//   0.25 s, until the mean times of two consecutive windows of 0.25 s differ by at most 2 percent; when it takes longer, until two consecutive
//   repetitions differ by at most 2 percent; at most 10 s (warmup_steady = 0 when the 10 s ended it).
//   Repetitions: fused 100; baseline min(100, max(10, as many as fit in VFHF_BUDGET_SECONDS (30) s)), from the time of its last warm-up
//   repetition (rep_estimate_ms). The column reps_rule says which applied: fixed (fused), budget, max (100) or min (10).
//   Per point and side: mean, median, min and max, and every timed repetition (the repetitions CSV). Three speedups per point: the ratio of the
//   means (speedup_ratio_of_means), the ratio of the medians (speedup_ratio_of_medians), and the mean of the per-repetition ratios
//   baseline_i / fused_i over the first min(reps) repetitions of the two blocks (speedup_block_paired_mean_of_ratios). The last is the estimator
//   of the 2025 harness ("Mean Speedup", opencv-cuda/testsCommon.cuh), which timed the two sides back to back in each iteration; here they are
//   paired by index within two separately timed blocks.
//   Clocks and links (common/hip_wallclock.h), read outside the timed intervals: at the start of the warm-up, at the start of the timed
//   repetitions and after them, -1 where they cannot be read (an NVIDIA GPU). sclk_* and mclk_*: the graphics and the memory clock; card_link_*:
//   the GPU function's own PCIe link (on the RX 9060 the link inside the card, to its PCIe switch); host_link_*: the link of the root port above
//   the GPU, to the host.
// Order: the fused side of a point first, then its baseline; VFHF_BASELINE_FIRST=1 times the baseline first.
// Checks, once per point before the timing, tolerance 1e-4 absolute: fused against the double precision reference of the chain (required), fused
// against the baseline (required), the baseline against the reference (informational). The outputs of both sides are filled with NaN before the
// checked repetition, so that an output a side does not write fails. The chain converges to its fixed point 0.54 / (1 - 0.54) = 1.1739 within the
// tolerance after about 42 operations, so these rows cannot see its length beyond NUM_OPS = 2: the length is verified in the device code instead,
// NUM_OPS / 2 FMA operations in each fused kernel (gfx1200 and sm_89).
// Environment (checked: common/env_args.h): VFHF_BASELINE_FIRST (0 or 1), VFHF_WARMUP_SECONDS (2.0; 0 to 10), VFHF_BUDGET_SECONDS (30; 1 to 600),
// VFHF_MAX_OPS (0: every point; otherwise the points with more operations are left out, for a short check).
// CSVs in the working directory (<arch>: GPU_AMD or GPU_NVIDIA): benchmark_vfhf_max_speedup_<arch>.csv (a row per point and side),
// benchmark_vfhf_max_speedup_speedup_<arch>.csv (a row per point), benchmark_vfhf_max_speedup_reps_<arch>.csv (a row per timed repetition,
// written after the timing of its point: the sides in the order they ran, each side's repetitions in the order they ran) and
// benchmark_vfhf_max_speedup_agreement_<arch>.csv.
#include <common/csv_writer.h>
#include <common/env_args.h>
#include <common/hip_wallclock.h>
#include <common/main.h>
#include <common/status_check.h>

#include "vfhf_points.h"   // generated by fkl-self/vfhf/CMakeLists.txt: vfhf::POINTS and the declarations of their fused sides

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/execution_model/executors.h>

#if defined(FK_HIP_BACKEND)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace vfhf {

#if defined(FK_HIP_BACKEND)
constexpr const char* ARCH_NAME = "GPU_AMD";
#else
constexpr const char* ARCH_NAME = "GPU_NVIDIA";
#endif
constexpr double TOLERANCE = 1e-4;
constexpr size_t FUSED_REPS = 100, MIN_REPS = 10, MAX_REPS = 100;
constexpr double WINDOW_SECONDS = 0.25, STEADY = 0.02, MAX_WARMUP_SECONDS = 10.0;

// The GPU time of one repetition of a side: an event recorded on its stream before the first call and one after the last call (timing
// infrastructure: it calls the HIP or CUDA event API and no FKL function)
class EventTimer {
public:
#if defined(FK_HIP_BACKEND)
    explicit EventTimer(const hipStream_t stream) : m_stream(stream) {
        FKLB_HIP_CHECK(hipEventCreate(&m_start));
        FKLB_HIP_CHECK(hipEventCreate(&m_stop));
    }
    ~EventTimer() {
        (void)hipEventDestroy(m_start);
        (void)hipEventDestroy(m_stop);
    }
    void start() { FKLB_HIP_CHECK(hipEventRecord(m_start, m_stream)); }
    double stopMs() {
        FKLB_HIP_CHECK(hipEventRecord(m_stop, m_stream));
        FKLB_HIP_CHECK(hipEventSynchronize(m_stop));
        float ms = 0;
        FKLB_HIP_CHECK(hipEventElapsedTime(&ms, m_start, m_stop));
        return ms;
    }
#else
    explicit EventTimer(const cudaStream_t stream) : m_stream(stream) {
        FKLB_CUDA_CHECK(cudaEventCreate(&m_start));
        FKLB_CUDA_CHECK(cudaEventCreate(&m_stop));
    }
    ~EventTimer() {
        (void)cudaEventDestroy(m_start);
        (void)cudaEventDestroy(m_stop);
    }
    void start() { FKLB_CUDA_CHECK(cudaEventRecord(m_start, m_stream)); }
    double stopMs() {
        FKLB_CUDA_CHECK(cudaEventRecord(m_stop, m_stream));
        FKLB_CUDA_CHECK(cudaEventSynchronize(m_stop));
        float ms = 0;
        FKLB_CUDA_CHECK(cudaEventElapsedTime(&ms, m_start, m_stop));
        return ms;
    }
#endif
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;

private:
#if defined(FK_HIP_BACKEND)
    hipStream_t m_stream;
    hipEvent_t m_start{}, m_stop{};
#else
    cudaStream_t m_stream;
    cudaEvent_t m_start{}, m_stop{};
#endif
};

struct Settings {
    bool baselineFirst{false};
    double minWarmupSeconds{2.0};
    double budgetSeconds{30.0};
    size_t maxOps{0};   // 0: every point
};

// The clocks and PCIe links of an AMD GPU (common/hip_wallclock.h), -1 where they cannot be read
struct Readings {
    double sclkMHz{-1}, mclkMHz{-1}, cardLinkGTs{-1}, cardLinkWidth{-1}, hostLinkGTs{-1}, hostLinkWidth{-1};
};

inline Readings readClocksAndLinks() {
    return {fklb::amdGpuClockMHz(), fklb::amdGpuMemClockMHz(), fklb::amdCardLinkGTs(), fklb::amdCardLinkWidth(), fklb::amdHostLinkGTs(),
            fklb::amdHostLinkWidth()};
}

struct SideTimes {
    size_t reps{0};
    std::string repsRule;
    double repEstimateMs{std::numeric_limits<double>::quiet_NaN()};   // the last warm-up repetition, from which the baseline's count is chosen
    double mean{0}, median{0}, min{0}, max{0};
    double warmupSeconds{0};
    int warmupRuns{0};
    bool warmupSteady{false};
    Readings start, timed, end;   // at the start of the warm-up, at the start of the timed repetitions and after them
    std::vector<double> times;    // the timed repetitions in the order they ran, ms
};

struct Outputs {
    fklb::CsvWriter* timing;
    fklb::CsvWriter* speedup;
    fklb::CsvWriter* reps;
    fklb::CsvWriter* agreement;
    bool allPass{true};
};

// The warm-up and the timed repetitions of one side (see the header). run() makes the calls of one repetition.
template <typename Run>
SideTimes timeSide(EventTimer& timer, const Run& run, const bool fixedReps, const Settings& settings) {
    SideTimes s;
    const auto once = [&] {
        timer.start();
        run();
        return timer.stopMs();
    };
    s.start = readClocksAndLinks();
    const auto begin = fklb::WallClock::now();
    const auto elapsed = [&] { return std::chrono::duration<double>(fklb::WallClock::now() - begin).count(); };
    double windowStart = 0, windowSum = 0, previousWindowMean = -1, previousMs = -1;
    int inWindow = 0;
    while (true) {
        const double ms = once();
        ++s.warmupRuns;
        if (ms < WINDOW_SECONDS * 1000.0) {
            windowSum += ms;
            ++inWindow;
            if (elapsed() - windowStart >= WINDOW_SECONDS) {
                const double mean = windowSum / inWindow;
                s.warmupSteady = previousWindowMean > 0 && std::abs(mean - previousWindowMean) <= STEADY * previousWindowMean;
                previousWindowMean = mean;
                windowSum = 0;
                inWindow = 0;
                windowStart = elapsed();
            }
        } else {
            s.warmupSteady = previousMs > 0 && std::abs(ms - previousMs) <= STEADY * previousMs;
        }
        previousMs = ms;
        const double t = elapsed();
        if (s.warmupRuns >= 2 && t >= settings.minWarmupSeconds && (s.warmupSteady || t >= MAX_WARMUP_SECONDS)) {
            s.warmupSeconds = t;
            break;
        }
    }
    if (fixedReps) {
        s.reps = FUSED_REPS;
        s.repsRule = "fixed";
    } else {
        s.repEstimateMs = previousMs;
        const double fit = std::floor(settings.budgetSeconds * 1000.0 / previousMs);
        if (fit >= static_cast<double>(MAX_REPS)) {
            s.reps = MAX_REPS;
            s.repsRule = "max";
        } else if (fit <= static_cast<double>(MIN_REPS)) {
            s.reps = MIN_REPS;
            s.repsRule = "min";
        } else {
            s.reps = static_cast<size_t>(fit);
            s.repsRule = "budget";
        }
    }
    s.timed = readClocksAndLinks();
    s.times.resize(s.reps);
    for (double& t : s.times) {
        t = once();
    }
    s.end = readClocksAndLinks();
    double sum = 0;
    for (const double t : s.times) {
        sum += t;
    }
    s.mean = sum / static_cast<double>(s.times.size());
    std::vector<double> sorted = s.times;
    std::sort(sorted.begin(), sorted.end());
    s.min = sorted.front();
    s.max = sorted.back();
    s.median = sorted.size() % 2 ? sorted[sorted.size() / 2] : 0.5 * (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]);
    return s;
}

// max |a - b| over the values seen, NaN as soon as one of them is NaN
inline void trackMaxAbs(double& maxAbs, const double a, const double b) {
    const double d = std::fabs(a - b);
    maxAbs = (std::isnan(d) || std::isnan(maxAbs)) ? std::numeric_limits<double>::quiet_NaN() : std::max(maxAbs, d);
}

template <size_t NUM_OPS>
bool runPoint(fk::Stream_<fk::defaultParArch>& fusedStream, EventTimer& fusedTimer, const std::array<fk::Ptr2D<uchar>, BATCH>& fusedImages,
              fk::Tensor<float>& fusedOutput, fk::Stream_<fk::defaultParArch>& baselineStream, EventTimer& baselineTimer,
              const std::array<fk::Ptr2D<uchar>, BATCH>& baselineImages, std::array<fk::Ptr2D<float>, BATCH>& baselineFloat,
              const Settings& settings, Outputs& out) {
    std::cout << "NUM_OPS " << NUM_OPS << std::endl;
    const auto fusedRun = [&] { fused<NUM_OPS>(fusedStream, fusedImages, fusedOutput); };
    const auto baselineRun = [&] {
        for (int i = 0; i < BATCH; ++i) {
            fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(baselineStream,
                fk::PerThreadRead<fk::ND::_2D, uchar>::build(baselineImages[i].ptr()),
                fk::SaturateCast<uchar, float>::build().then(fk::Mul<float>::build(ALPHA)),
                fk::PerThreadWrite<fk::ND::_2D, float>::build(baselineFloat[i].ptr()));
            for (size_t op = 0; op < NUM_OPS; op += OPS_PER_ITER) {
                fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(baselineStream,
                    fk::PerThreadRead<fk::ND::_2D, float>::build(baselineFloat[i].ptr()), fk::Mul<float>::build(VALUE),
                    fk::PerThreadWrite<fk::ND::_2D, float>::build(baselineFloat[i].ptr()));
                fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(baselineStream,
                    fk::PerThreadRead<fk::ND::_2D, float>::build(baselineFloat[i].ptr()), fk::Add<float>::build(VALUE),
                    fk::PerThreadWrite<fk::ND::_2D, float>::build(baselineFloat[i].ptr()));
            }
        }
    };

    // ---- Checks: one repetition of each side, the outputs on the host. Both sides' outputs are filled with NaN first: from the third point on
    // they would otherwise still hold the previous point's correct values (the fixed point of the chain), and an output that a side did not
    // write would pass
    constexpr float NOT_WRITTEN = std::numeric_limits<float>::quiet_NaN();
    fk::setTo(NOT_WRITTEN, fusedOutput, fusedStream);
    for (fk::Ptr2D<float>& image : baselineFloat) {
        fk::setTo(NOT_WRITTEN, image, baselineStream);
    }
    fusedRun();
    fusedOutput.download(fusedStream);
    baselineRun();
    for (fk::Ptr2D<float>& image : baselineFloat) {
        image.download(baselineStream);
    }
    fusedStream.sync();
    baselineStream.sync();
    double reference = static_cast<double>(static_cast<float>(INPUT_VALUE)) * static_cast<double>(ALPHA);
    for (size_t op = 0; op < NUM_OPS; op += OPS_PER_ITER) {
        reference = reference * static_cast<double>(VALUE) + static_cast<double>(VALUE);
    }
    double fusedVsReference = 0, fusedVsBaseline = 0, baselineVsReference = 0;
    for (int z = 0; z < BATCH; ++z) {
        for (int y = 0; y < HEIGHT; ++y) {
            for (int x = 0; x < WIDTH; ++x) {
                const double f = fusedOutput.at(x, y, z);
                const double b = baselineFloat[z].at(x, y);
                trackMaxAbs(fusedVsReference, f, reference);
                trackMaxAbs(fusedVsBaseline, f, b);
                trackMaxAbs(baselineVsReference, b, reference);
            }
        }
    }
    const auto agreementRow = [&](const char* what, const double maxAbs, const bool required) {
        const bool pass = maxAbs <= TOLERANCE;
        out.agreement->row(NUM_OPS, what, maxAbs, reference, required ? (pass ? 1 : 0) : -1);
        out.allPass &= pass || !required;
        std::cout << "  " << what << ": max " << maxAbs << (required ? (pass ? " ok" : " FAIL") : " (informational)") << std::endl;
    };
    agreementRow("fused vs reference", fusedVsReference, true);
    agreementRow("fused vs baseline", fusedVsBaseline, true);
    agreementRow("baseline vs reference", baselineVsReference, false);

    // ---- Timing
    SideTimes f, b;
    if (settings.baselineFirst) {
        b = timeSide(baselineTimer, baselineRun, false, settings);
        f = timeSide(fusedTimer, fusedRun, true, settings);
    } else {
        f = timeSide(fusedTimer, fusedRun, true, settings);
        b = timeSide(baselineTimer, baselineRun, false, settings);
    }
    const auto timingRow = [&](const char* side, const size_t calls, const int position, const SideTimes& s) {
        out.timing->row(NUM_OPS, side, calls, position, s.reps, s.repsRule, s.repEstimateMs, s.mean, s.median, s.min, s.max, s.warmupSeconds,
                        s.warmupRuns, s.warmupSteady ? 1 : 0, s.start.sclkMHz, s.timed.sclkMHz, s.end.sclkMHz, s.start.mclkMHz, s.timed.mclkMHz,
                        s.end.mclkMHz, s.start.cardLinkGTs, s.timed.cardLinkGTs, s.end.cardLinkGTs, s.start.cardLinkWidth, s.timed.cardLinkWidth,
                        s.end.cardLinkWidth, s.start.hostLinkGTs, s.timed.hostLinkGTs, s.end.hostLinkGTs, s.start.hostLinkWidth,
                        s.timed.hostLinkWidth, s.end.hostLinkWidth);
        std::cout << "  " << side << ": mean " << s.mean << " ms, median " << s.median << " (min " << s.min << ", max " << s.max << "; " << s.reps
                  << " repetitions, " << s.repsRule << "; warm-up " << s.warmupSeconds << " s, " << s.warmupRuns << " runs, steady "
                  << s.warmupSteady << ")" << std::endl;
    };
    timingRow("fused", 1, settings.baselineFirst ? 2 : 1, f);
    timingRow("baseline", static_cast<size_t>(BATCH) * (1 + NUM_OPS), settings.baselineFirst ? 1 : 2, b);
    // The mean of the per-repetition ratios baseline_i / fused_i over the first min(reps) repetitions of the two blocks (see the header)
    const size_t pairs = std::min(b.times.size(), f.times.size());
    double ratioSum = 0;
    for (size_t i = 0; i < pairs; ++i) {
        ratioSum += b.times[i] / f.times[i];
    }
    const double blockPaired = ratioSum / static_cast<double>(pairs);
    out.speedup->row(NUM_OPS, b.mean / f.mean, b.median / f.median, blockPaired, f.mean, b.mean, f.median, b.median);
    std::cout << "  speedup: " << b.mean / f.mean << " (ratio of the means), " << b.median / f.median << " (ratio of the medians), "
              << blockPaired << " (block-paired mean of " << pairs << " ratios)" << std::endl;
    const auto repRows = [&](const char* side, const int position, const SideTimes& s) {
        for (size_t i = 0; i < s.times.size(); ++i) {
            out.reps->row(NUM_OPS, side, position, i, s.times[i]);
        }
    };
    if (settings.baselineFirst) {   // the sides in the order they ran (the timeSide calls above)
        repRows("baseline", 1, b);
        repRows("fused", 2, f);
    } else {
        repRows("fused", 1, f);
        repRows("baseline", 2, b);
    }
    out.timing->flush();
    out.speedup->flush();
    out.reps->flush();
    out.agreement->flush();
    return true;
}

template <size_t... I>
void runAll(std::index_sequence<I...>, fk::Stream_<fk::defaultParArch>& fusedStream, EventTimer& fusedTimer,
            const std::array<fk::Ptr2D<uchar>, BATCH>& fusedImages, fk::Tensor<float>& fusedOutput,
            fk::Stream_<fk::defaultParArch>& baselineStream, EventTimer& baselineTimer, const std::array<fk::Ptr2D<uchar>, BATCH>& baselineImages,
            std::array<fk::Ptr2D<float>, BATCH>& baselineFloat, const Settings& settings, Outputs& out) {
    ((settings.maxOps == 0 || POINTS[I] <= settings.maxOps
          ? runPoint<POINTS[I]>(fusedStream, fusedTimer, fusedImages, fusedOutput, baselineStream, baselineTimer, baselineImages, baselineFloat,
                                settings, out)
          : true),
     ...);
}

}  // namespace vfhf

int launch() {
    using namespace vfhf;
    Settings settings;
    settings.baselineFirst = fklb::envInt("VFHF_BASELINE_FIRST", 0, 0, 1) == 1;
    settings.minWarmupSeconds = fklb::envDouble("VFHF_WARMUP_SECONDS", 2.0, 0.0, 10.0);
    settings.budgetSeconds = fklb::envDouble("VFHF_BUDGET_SECONDS", 30.0, 1.0, 600.0);
    settings.maxOps = static_cast<size_t>(fklb::envInt("VFHF_MAX_OPS", 0, 0, 1000000));

    // Each side: its own stream, its own 50 images (set to 30 on the device, as cv::cuda::GpuMat(cropSize, CV_8UC1, 30) does), its own output
    fk::Stream_<fk::defaultParArch> fusedStream, baselineStream;
    std::array<fk::Ptr2D<uchar>, BATCH> fusedImages, baselineImages;
    std::array<fk::Ptr2D<float>, BATCH> baselineFloat;
    for (int i = 0; i < BATCH; ++i) {
        fusedImages[i] = fk::Ptr2D<uchar>(WIDTH, HEIGHT);
        baselineImages[i] = fk::Ptr2D<uchar>(WIDTH, HEIGHT);
        baselineFloat[i] = fk::Ptr2D<float>(WIDTH, HEIGHT);
        fk::setTo(INPUT_VALUE, fusedImages[i], fusedStream);
        fk::setTo(INPUT_VALUE, baselineImages[i], baselineStream);
    }
    fk::Tensor<float> fusedOutput(WIDTH, HEIGHT, BATCH, 1);
    fusedStream.sync();
    baselineStream.sync();
    EventTimer fusedTimer(fusedStream), baselineTimer(baselineStream);

    std::string pointsText;
    for (const size_t p : POINTS) {
        pointsText += (pointsText.empty() ? "" : " ") + std::to_string(p);
    }
    const auto oneDecimal = [](const double v) { const std::string s = std::to_string(v); return s.substr(0, s.find('.') + 2); };
    const std::string notes =
        std::string("FKL against itself, vertical and horizontal fusion of the 2025 mul_add chain: 50 images of 60x120 uchar, all 30; fused = one "
                    "executeOperations (batch read of the 50 images, SaturateCast<uchar, float> then Mul(1), the NUM_OPS operations as fk::StaticLoop "
                    "of Mul(0.54) then Add(0.54), PerThreadWrite into a 60x120x50 float tensor), thread fusion off; baseline = one executeOperations "
                    "per operation and per image (the conversion into a float image, then NUM_OPS in place), 50 (1 + NUM_OPS) calls, every "
                    "operation built in its call; GPU events on the side's own stream around all calls of one repetition; warm-up at least ") +
        oneDecimal(settings.minWarmupSeconds) +
        " s and 2 repetitions, until the mean of two consecutive windows of 0.25 s differs by at most 2 percent (repetitions shorter than 0.25 s) "
        "or two consecutive repetitions differ by at most 2 percent (longer ones), at most 10 s; repetitions: fused 100, baseline "
        "min(100, max(10, as many as fit in " + oneDecimal(settings.budgetSeconds) + " s from its last warm-up repetition)); order: " +
        (settings.baselineFirst ? "baseline first" : "fused first") + "; points (NUM_OPS): " + pointsText +
        (settings.maxOps ? "; only the points up to " + std::to_string(settings.maxOps) + " operations (VFHF_MAX_OPS)" : std::string()) +
        "; speedup_ratio_of_means = mean baseline / mean fused; speedup_ratio_of_medians = median baseline / median fused; "
        "speedup_block_paired_mean_of_ratios = the mean over i < min(reps) of baseline_i / fused_i, the i-th timed repetitions of the two blocks "
        "(the estimator of the 2025 harness's Mean Speedup, which timed both sides back to back in each iteration; here the sides are timed in "
        "separate blocks and paired by index); sclk_* and mclk_*: the graphics and the memory clock of an "
        "AMD GPU (MHz); card_link_gts_* and card_link_width_*: the speed (GT/s) and width (lanes) of the GPU function's own PCIe link, which on "
        "a card with its own PCIe switch is the link inside the card; host_link_gts_* and host_link_width_*: the same of the root port above the "
        "GPU, the link to the host; all read outside the timed intervals, at the start of the warm-up, at the start of the timed repetitions and "
        "after them (-1: not readable)";
    fklb::CsvWriter timing(std::string("benchmark_vfhf_max_speedup_") + ARCH_NAME + ".csv", "benchmark_vfhf_max_speedup",
                           {"num_ops", "side", "calls_per_rep", "position", "reps", "reps_rule", "rep_estimate_ms", "mean_ms", "median_ms", "min_ms",
                            "max_ms", "warmup_s", "warmup_runs", "warmup_steady", "sclk_start_mhz", "sclk_timed_start_mhz", "sclk_end_mhz",
                            "mclk_start_mhz", "mclk_timed_start_mhz", "mclk_end_mhz", "card_link_gts_start", "card_link_gts_timed_start",
                            "card_link_gts_end", "card_link_width_start", "card_link_width_timed_start", "card_link_width_end",
                            "host_link_gts_start", "host_link_gts_timed_start", "host_link_gts_end", "host_link_width_start",
                            "host_link_width_timed_start", "host_link_width_end"},
                           notes);
    fklb::CsvWriter speedup(std::string("benchmark_vfhf_max_speedup_speedup_") + ARCH_NAME + ".csv", "benchmark_vfhf_max_speedup",
                            {"num_ops", "speedup_ratio_of_means", "speedup_ratio_of_medians", "speedup_block_paired_mean_of_ratios",
                             "fused_mean_ms", "baseline_mean_ms", "fused_median_ms", "baseline_median_ms"},
                            notes);
    fklb::CsvWriter repetitions(std::string("benchmark_vfhf_max_speedup_reps_") + ARCH_NAME + ".csv", "benchmark_vfhf_max_speedup",
                                {"num_ops", "side", "position", "rep", "ms"},
                                notes + "; a row per timed repetition, written after the timing of its point: the sides in the order they ran, "
                                        "each side's repetitions in the order they ran (rep 0 first)");
    fklb::CsvWriter agreement(std::string("benchmark_vfhf_max_speedup_agreement_") + ARCH_NAME + ".csv", "benchmark_vfhf_max_speedup",
                              {"num_ops", "comparison", "max_abs_diff", "reference", "pass"},
                              notes + "; checks once per point before the timing, both sides' outputs filled with NaN before the checked repetition; "
                                      "tolerance 1e-4 absolute; pass -1 = informational");
    Outputs out{&timing, &speedup, &repetitions, &agreement, true};
    runAll(std::make_index_sequence<sizeof(POINTS) / sizeof(POINTS[0])>{}, fusedStream, fusedTimer, fusedImages, fusedOutput, baselineStream,
           baselineTimer, baselineImages, baselineFloat, settings, out);
    std::cout << (out.allPass ? "all required numerical checks passed" : "NUMERICAL CHECK FAILED") << std::endl;
    return out.allPass ? 0 : -1;
}
