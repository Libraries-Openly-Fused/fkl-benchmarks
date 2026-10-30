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

// E4b: the NV12 video pre-processing pipeline (fkl-self/pipelines/nv12_pipeline.h) on the back-end this target is built
// for (CUDA, HIP or the CPU; the CPU target also runs ParArch::CPU_MULTITHREAD), for N = 1, 8, 16, 32, 64 detection crops of
// a 4K NV12 frame. Per point and variant: wall clock of the whole run including the synchronisation of the stream (the frame is
// already in device memory; the upload is not timed), 3 warm-up runs and E5_ITERS (30) timed runs, median, min, max and mean.
// Variants: convert (kernel 1 alone), chain_fused (kernel 2 alone), chain_perop_batch (kernel 2 as one execution per
// operation for all the crops), pipeline_fused (kernels 1 and 2: two launches), pipeline_perop_batch, and on the CPU
// back-ends pipeline_perop_crop (one execution per operation and per crop).
// Every output is checked before timing against a scalar double precision reference of kernel 2 (tolerance 1e-4); on the CPU
// the multithreaded outputs must be the same bytes as the single threaded ones.
// Environment: E5_ITERS, E5_WARMUP, E5_IMAGE (a binary PPM tiled over the frame instead of seeded noise), E5_IMAGE_INFO.
// CSV files in the working directory: benchmark_nv12_pipeline_<ParArch>.csv and benchmark_nv12_pipeline_agreement_<ParArch>.csv.
#include <common/csv_writer.h>
#include <common/main.h>
#include <fkl-self/pipelines/nv12_pipeline.h>

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace nv12bench {

constexpr double TOLERANCE = 1e-4;
constexpr int BATCHES[] = {1, 8, 16, 32, 64};

inline int envInt(const char* name, const int fallback) {
    const char* value = std::getenv(name);
    return value ? std::atoi(value) : fallback;
}

constexpr bool isCpu(const fk::ParArch pa) { return pa == fk::ParArch::CPU || pa == fk::ParArch::CPU_MULTITHREAD; }

struct Output {
    fklb::CsvWriter timing;
    fklb::CsvWriter agreement;
    bool allPass{true};
    int warmup, iters;
};

// One back-end PA and N crops. `cpuFused` keeps the fused output of the single threaded CPU run for the byte comparison
template <fk::ParArch PA, int N>
void runPoint(nv12::Frame& frame, fk::Ptr2D<nv12::U3>& rgb, Output& out, std::vector<float>& cpuFused) {
    const std::string backEnd(fk::toStrView(PA));
    std::cout << backEnd << " N=" << N << std::endl;
    const auto crops = nv12::makeCrops<N>();
    fk::Stream_<PA> stream;
    nv12::Pipeline<PA, N> pipe(stream, frame, rgb, crops);

    // The frame goes to the device once
    frame.getData().upload(stream);
    stream.sync();

    // Agreement
    pipe.convert();
    pipe.fused();
    pipe.perOpBatch();
    constexpr bool cpu = isCpu(PA);
    if constexpr (cpu) {
        pipe.perOpCrop();
    }
    pipe.download();
    pipe.sync();
    const std::vector<float> reference = nv12::referenceTensor<N>(nv12::hostRaw(rgb), crops);
    const auto check = [&](const std::string& what, const float* a, const float* b, const bool exact) {
        const size_t n = nv12::Pipeline<PA, N>::outValues();
        const double d = nv12::maxAbsDiff(a, b, n);
        const bool pass = exact ? std::memcmp(a, b, n * sizeof(float)) == 0 : d <= TOLERANCE;
        out.agreement.row(N, backEnd, what, d, pass ? 1 : 0);
        out.allPass &= pass;
        std::cout << "  " << what << ": max abs diff " << d << (pass ? " ok" : " FAIL") << std::endl;
    };
    check("chain_fused vs scalar reference (ASYMMETRIC sampling)", pipe.fusedOut(), reference.data(), false);
    check("chain_perop_batch vs chain_fused", pipe.batchOut(), pipe.fusedOut(), false);
    if constexpr (cpu) {
        check("chain_perop_crop vs chain_fused", pipe.cropOut(), pipe.fusedOut(), false);
    }
    if constexpr (PA == fk::ParArch::CPU) {
        cpuFused.assign(pipe.fusedOut(), pipe.fusedOut() + nv12::Pipeline<PA, N>::outValues());
    } else if constexpr (PA == fk::ParArch::CPU_MULTITHREAD) {
        check("chain_fused CPU_MULTITHREAD vs CPU (identical)", pipe.fusedOut(), cpuFused.data(), true);
    }

    // Timing
    int threads = 0;
#ifdef FK_CPU_MULTITHREAD_ENABLED
    if constexpr (PA == fk::ParArch::CPU_MULTITHREAD) {
        threads = static_cast<int>(fk::cpuThreadPool().size());
    }
#endif
    if constexpr (PA == fk::ParArch::CPU) {
        threads = 1;
    }
    const auto time = [&](const std::string& variant, const int kernels, const std::function<void()>& fn) {
        const nv12::Stats s = nv12::measure(fn, out.warmup, out.iters);
        out.timing.row(N, backEnd, variant, kernels, threads, s.median, s.min, s.max, s.mean, s.iters);
        std::cout << "  " << variant << ": median " << s.median << " ms (min " << s.min << ", max " << s.max << ")" << std::endl;
    };
    time("convert", 1, [&] { pipe.convert(); pipe.sync(); });
    time("chain_fused", 1, [&] { pipe.fused(); pipe.sync(); });
    time("chain_perop_batch", 5, [&] { pipe.perOpBatch(); pipe.sync(); });
    time("pipeline_fused", 2, [&] { pipe.convert(); pipe.fused(); pipe.sync(); });
    time("pipeline_perop_batch", 6, [&] { pipe.convert(); pipe.perOpBatch(); pipe.sync(); });
    if constexpr (cpu) {
        time("pipeline_perop_crop", 1 + 5 * N, [&] { pipe.convert(); pipe.perOpCrop(); pipe.sync(); });
    }
    out.timing.flush();
    out.agreement.flush();
}

template <int N>
void runBatch(nv12::Frame& frame, fk::Ptr2D<nv12::U3>& rgb, Output& out) {
    std::vector<float> cpuFused;
    runPoint<fk::defaultParArch, N>(frame, rgb, out, cpuFused);
#ifdef FK_CPU_MULTITHREAD_ENABLED
    if constexpr (fk::defaultParArch == fk::ParArch::CPU) {
        runPoint<fk::ParArch::CPU_MULTITHREAD, N>(frame, rgb, out, cpuFused);
    }
#endif
}

} // namespace nv12bench

int launch() {
    const char* imagePath = std::getenv("E5_IMAGE");
    const char* imageInfo = std::getenv("E5_IMAGE_INFO");
    nv12::Frame frame(nv12::FRAME_W, nv12::FRAME_H);
    std::string source;
    if (imagePath != nullptr) {
        std::vector<uchar> rgbImage;
        int w = 0, h = 0;
        if (!nv12::readPpm(imagePath, rgbImage, w, h)) {
            std::cerr << "Cannot read the binary PPM " << imagePath << std::endl;
            return -1;
        }
        nv12::fillFromRgb(frame, rgbImage, w, h);
        source = std::string("natural image ") + imagePath + " (" + std::to_string(w) + "x" + std::to_string(h) +
                 ") tiled over the frame, converted to NV12 (BT.601 limited, chroma averaged over 2x2)" +
                 (imageInfo ? std::string("; ") + imageInfo : "");
    } else {
        nv12::fillNoise(frame);
        source = "seeded noise (LCG, seed 424242), luma 16..235, chroma 16..240";
    }
    fk::Ptr2D<nv12::U3> rgb(nv12::FRAME_W, nv12::FRAME_H);

    const std::string arch(fk::toStrView(fk::defaultParArch));
    const std::string notes = "E4b NV12 pipeline: NV12 4K frame -> RGB kernel -> fused crops, resize 64x128, normalise, NCHW split; source: " +
                              source + "; the frame is already in device memory, wall clock includes the stream synchronisation";
    nv12bench::Output out{
        fklb::CsvWriter("benchmark_nv12_pipeline_" + arch + ".csv", "benchmark_nv12_pipeline",
                        {"crops", "back_end", "variant", "kernels", "threads", "median_ms", "min_ms", "max_ms", "mean_ms", "iters"}, notes),
        fklb::CsvWriter("benchmark_nv12_pipeline_agreement_" + arch + ".csv", "benchmark_nv12_pipeline",
                        {"crops", "back_end", "comparison", "max_abs_diff", "pass"}, notes + "; tolerance 1e-4 absolute, identical = byte for byte"),
        true, nv12bench::envInt("E5_WARMUP", 3), nv12bench::envInt("E5_ITERS", 30)};

    nv12bench::runBatch<1>(frame, rgb, out);
    nv12bench::runBatch<8>(frame, rgb, out);
    nv12bench::runBatch<16>(frame, rgb, out);
    nv12bench::runBatch<32>(frame, rgb, out);
    nv12bench::runBatch<64>(frame, rgb, out);
    std::cout << (out.allPass ? "all numerical checks passed" : "NUMERICAL CHECK FAILED") << std::endl;
    return out.allPass ? 0 : -1;
}
