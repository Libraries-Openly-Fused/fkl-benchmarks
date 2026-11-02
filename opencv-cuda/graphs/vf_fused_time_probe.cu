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

// E0, the time of the fused kernel (cvGPUSpeedup) of the VF + HF experiment (mul_add: 50 crops of 60 x 120, 8-bit to float, Mul then Add, NumOps operations) in the loops of the experiment and on its own,
// the counterpart of fkl-self/graphs/hf_fused_time_probe.cu. The loop of the experiment (STOP_OCV_START_CVGS_BENCHMARK, opencv-cuda/testsCommon.cuh) is, per iteration:
//     record(start); OPENCV; record(stop); synchronize(stop); first = elapsed(start, stop); record(start); CVGS; record(stop); synchronize(stop); second = elapsed(start, stop)
// and the speedup of an iteration is first / second. Modes (mode names as in the HF probe):
//   loop_plain / loop_child / loop_flat_serial     the loop with the OpenCV-CUDA work as plain calls on the stream (pass 1 of the experiment), as the graph of child nodes of the experiment (pass 2), and as one
//                                                  stream capture of the calls of all the crops (the structure of mul_add1 and mul1)
//   alone_synced                                   the loop with an empty baseline: the stream is idle, and the fused kernel is launched the way the loop of the experiment launches it
//   alone_synced_pollute2MB / 64MB                 a host memory sweep of that size between the synchronisation and the launch
//   alone_pipelined10 / 50                         10 or 50 launches in a row between two events, the time divided by the number
//   alone_queued_behind_spin                       the launch enqueued while a kernel of about 1 ms is running, the time between the events around the fused kernel only
// The fused launch is VerticalFusion<CV_8UC1, CV_32FC1, 2, NumOps, FuncType>::execute of benchmarks/verticalfusion/vertical_fusion_static_loop.cuh, the call of the generated launch<N> of the experiment.
// NumOps: FKLB_VF_OPS (comma separated, a subset of 502,2502,10002,17702; default all). Iterations: FKLB_VF_LOOP_ITERS (default: 200 for 502 operations, 100 for 2502, 20 for 10002, 12 for 17702),
// FKLB_VF_LOOP_WARMUP (3), FKLB_VF_ALONE_ITERS (500), FKLB_VF_ALONE_WARMUP (100); FKLB_VF_MODES selects modes by prefix. Output: vf_fused_time_probe.csv and vf_fused_time_probe_raw.csv, as the HF probe.
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/graph_structures.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>

#ifndef CPP_NUM_EXPERIMENTS
#define CPP_NUM_EXPERIMENTS 200
#endif
#include <opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul_add/realBatch.h>
#include <opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul_add/opType.cuh>
#include <benchmarks/verticalfusion/vertical_fusion_static_loop.cuh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using fklb::graphs::Clock;
using fklb::graphs::microsecondsBetween;
constexpr int CROP_W = 60, CROP_H = 120;
constexpr long long SPIN_CYCLES = 2700000;   // about 1 ms at 2.7 GHz

__global__ void spinKernel(const long long cycles) {
    const long long begin = clock64();
    while (clock64() - begin < cycles) {
    }
}

volatile unsigned long long g_sink = 0;

struct HostSweep {
    std::vector<unsigned char> buffer;
    explicit HostSweep(const size_t bytes) : buffer(bytes, 1) {}
    void run() {
        unsigned long long sum = 0;
        for (size_t i = 0; i < buffer.size(); i += 64) {
            buffer[i] = static_cast<unsigned char>(buffer[i] + 1);
            sum += buffer[i];
        }
        g_sink = sum;
    }
};

struct Samples {
    std::vector<float> firstMs, secondMs;
    std::vector<double> baselineUs, fusedUs;
};

double percentile(std::vector<double> v, const double p) {
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(p * static_cast<double>(v.size() - 1))];
}

struct Options {
    int loopIters{0};   // 0: by the number of operations
    int loopWarmup{3};
    int aloneIters{500};
    int aloneWarmup{100};
    std::string modes;
    bool wants(const std::string& name) const {
        if (modes.empty()) { return true; }
        std::stringstream ss(modes);
        std::string prefix;
        while (std::getline(ss, prefix, ',')) {
            if (name.rfind(prefix, 0) == 0) { return true; }
        }
        return false;
    }
    int loopItersFor(const size_t numOps) const {
        if (loopIters > 0) { return loopIters; }
        return numOps <= 600 ? 200 : (numOps <= 3000 ? 100 : (numOps <= 12000 ? 20 : 12));
    }
};

template <size_t NUM_OPS>
class Probe {
    cv::cuda::Stream m_cvStream;
    cudaStream_t m_raw;
    std::array<cv::cuda::GpuMat, REAL_BATCH> m_in, m_out;
    cv::cuda::GpuMat m_outFused;
    const cv::Size m_cropSize{CROP_W, CROP_H};
    const cv::Scalar m_value{0.54f};
    const float m_alpha{1.f};
    const float m_val{0.54f};
    const FuncType m_dFunc;
    cudaEvent_t m_start{nullptr}, m_stop{nullptr}, m_mid{nullptr};

public:
    Probe()
        : m_raw(cv::cuda::StreamAccessor::getStream(m_cvStream)), m_outFused(REAL_BATCH, CROP_W * CROP_H, CV_32FC1),
          m_dFunc(fk::Mul<float>::build(m_val).then(fk::Add<float>::build(m_val))) {
        for (int i = 0; i < REAL_BATCH; ++i) {
            m_in[i] = cv::cuda::GpuMat(m_cropSize, CV_8UC1, cv::Scalar(30u));
            m_out[i].create(m_cropSize, CV_32FC1);
        }
        m_outFused.step = static_cast<size_t>(CROP_W) * CROP_H * sizeof(float);   // as in the experiment
        FKLB_CUDA(cudaEventCreate(&m_start));
        FKLB_CUDA(cudaEventCreate(&m_stop));
        FKLB_CUDA(cudaEventCreate(&m_mid));
    }
    ~Probe() {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
        cudaEventDestroy(m_mid);
    }

    // the OpenCV-CUDA calls of one crop, those of the experiment (alpha = 1.0); on the stream of the loop they use its cv::cuda::Stream, as the experiment does
    void calls(const int crop, cv::cuda::Stream& s) {
        m_in[crop].convertTo(m_out[crop], CV_32FC1, 1.0, s);
        for (size_t op = 0; op < NUM_OPS; op += 2) {
            cv::cuda::multiply(m_out[crop], m_value, m_out[crop], 1.0, -1, s);
            cv::cuda::add(m_out[crop], m_value, m_out[crop], cv::noArray(), -1, s);
        }
    }
    void enqueue(const int crop, cudaStream_t stream) {
        if (stream == m_raw) {
            calls(crop, m_cvStream);
        } else {
            cv::cuda::Stream s = cv::cuda::StreamAccessor::wrapStream(stream);
            calls(crop, s);
        }
    }
    void plainBaseline() {
        for (int crop = 0; crop < REAL_BATCH; ++crop) { enqueue(crop, m_raw); }
    }
    // the fused launch of the experiment
    void fused() { VerticalFusion<CV_8UC1, CV_32FC1, 2, NUM_OPS, FuncType>::execute(m_in, m_cvStream, m_alpha, m_outFused, m_cropSize, m_dFunc); }

    std::vector<float> download(const cv::cuda::GpuMat& mat, const int rows) {
        cv::Mat h;
        mat.download(h);
        const float* p = reinterpret_cast<const float*>(h.data);
        return std::vector<float>(p, p + static_cast<size_t>(rows) * CROP_W * CROP_H);
    }
    std::vector<float> downloadCrops(const std::vector<int>& picks) {
        std::vector<float> all;
        for (const int k : picks) {
            const std::vector<float> one = download(m_out[k], 1);
            all.insert(all.end(), one.begin(), one.end());
        }
        return all;
    }
    // the fused kernel against OpenCV-CUDA for the first, a middle and the last crop: the largest relative difference
    double checkFused() {
        for (auto& o : m_out) { o.setTo(cv::Scalar(0.f)); }
        FKLB_CUDA(cudaDeviceSynchronize());
        plainBaseline();
        FKLB_CUDA(cudaStreamSynchronize(m_raw));
        const std::vector<int> picks{0, REAL_BATCH / 2, REAL_BATCH - 1};
        const std::vector<float> reference = downloadCrops(picks);
        fused();
        FKLB_CUDA(cudaStreamSynchronize(m_raw));
        const std::vector<float> got = download(m_outFused, REAL_BATCH);
        double worst = 0.0;
        for (size_t k = 0; k < picks.size(); ++k) {
            for (int i = 0; i < CROP_W * CROP_H; ++i) {
                const double a = reference[k * CROP_W * CROP_H + i], b = got[static_cast<size_t>(picks[k]) * CROP_W * CROP_H + i];
                worst = std::max(worst, std::fabs(a - b) / std::max(1e-30, std::max(std::fabs(a), std::fabs(b))));
            }
        }
        return worst;
    }

    template <class Baseline, class Between>
    Samples loop(const int warmup, const int iters, Baseline&& baseline, Between&& between) {
        Samples samples;
        for (int it = 0; it < warmup + iters; ++it) {
            FKLB_CUDA(cudaEventRecord(m_start, m_raw));
            const auto t0 = Clock::now();
            baseline();
            const auto t1 = Clock::now();
            FKLB_CUDA(cudaEventRecord(m_stop, m_raw));
            FKLB_CUDA(cudaEventSynchronize(m_stop));
            float first = 0.f;
            FKLB_CUDA(cudaEventElapsedTime(&first, m_start, m_stop));
            between();
            FKLB_CUDA(cudaEventRecord(m_start, m_raw));
            const auto t2 = Clock::now();
            fused();
            const auto t3 = Clock::now();
            FKLB_CUDA(cudaEventRecord(m_stop, m_raw));
            FKLB_CUDA(cudaEventSynchronize(m_stop));
            float second = 0.f;
            FKLB_CUDA(cudaEventElapsedTime(&second, m_start, m_stop));
            if (it >= warmup) {
                samples.firstMs.push_back(first);
                samples.secondMs.push_back(second);
                samples.baselineUs.push_back(microsecondsBetween(t0, t1));
                samples.fusedUs.push_back(microsecondsBetween(t2, t3));
            }
        }
        return samples;
    }

    Samples pipelined(const Options& options, const int launches) {
        Samples samples;
        for (int it = 0; it < options.aloneWarmup + options.aloneIters; ++it) {
            FKLB_CUDA(cudaEventRecord(m_start, m_raw));
            const auto t0 = Clock::now();
            for (int k = 0; k < launches; ++k) { fused(); }
            const auto t1 = Clock::now();
            FKLB_CUDA(cudaEventRecord(m_stop, m_raw));
            FKLB_CUDA(cudaEventSynchronize(m_stop));
            float total = 0.f;
            FKLB_CUDA(cudaEventElapsedTime(&total, m_start, m_stop));
            if (it >= options.aloneWarmup) {
                samples.firstMs.push_back(0.f);
                samples.secondMs.push_back(total / static_cast<float>(launches));
                samples.baselineUs.push_back(0.0);
                samples.fusedUs.push_back(microsecondsBetween(t0, t1) / launches);
            }
        }
        return samples;
    }

    Samples queuedBehindSpin(const Options& options) {
        Samples samples;
        for (int it = 0; it < options.aloneWarmup + options.aloneIters; ++it) {
            FKLB_CUDA(cudaEventRecord(m_start, m_raw));
            spinKernel<<<1, 1, 0, m_raw>>>(SPIN_CYCLES);
            FKLB_CUDA(cudaEventRecord(m_mid, m_raw));
            const auto t2 = Clock::now();
            fused();
            const auto t3 = Clock::now();
            FKLB_CUDA(cudaEventRecord(m_stop, m_raw));
            FKLB_CUDA(cudaEventSynchronize(m_stop));
            float first = 0.f, second = 0.f;
            FKLB_CUDA(cudaEventElapsedTime(&first, m_start, m_mid));
            FKLB_CUDA(cudaEventElapsedTime(&second, m_mid, m_stop));
            if (it >= options.aloneWarmup) {
                samples.firstMs.push_back(first);
                samples.secondMs.push_back(second);
                samples.baselineUs.push_back(0.0);
                samples.fusedUs.push_back(microsecondsBetween(t2, t3));
            }
        }
        return samples;
    }

    int run(fklb::CsvWriter& csv, std::ofstream& raw, const Options& options) {
        int failures = 0;
        const double difference = checkFused();
        std::cout << "ops " << NUM_OPS << ": the fused kernel against OpenCV-CUDA, largest relative difference over 3 crops: " << difference << std::endl;
        const int loopIters = options.loopItersFor(NUM_OPS);
        const auto report = [&](const std::string& mode, const std::string& kind, const int iters, const Samples& s) {
            std::vector<double> second(s.secondMs.begin(), s.secondMs.end());
            std::vector<double> first(s.firstMs.begin(), s.firstMs.end());
            const double secondMean = fklb::graphs::mean(second);
            csv.row(NUM_OPS, mode, kind, iters, fklb::graphs::mean(first), secondMean, percentile(second, 0.5), percentile(second, 0.0), percentile(second, 1.0), percentile(second, 0.1),
                    percentile(second, 0.9), percentile(s.fusedUs, 0.5), percentile(s.fusedUs, 0.9), percentile(s.baselineUs, 0.5));
            csv.flush();
            for (size_t i = 0; i < s.secondMs.size(); ++i) {
                raw << NUM_OPS << ',' << mode << ',' << i << ',' << s.firstMs[i] << ',' << s.secondMs[i] << ',' << s.fusedUs[i] << '\n';
            }
            std::cout << "ops " << NUM_OPS << " " << mode << ": second mean " << secondMean * 1000.0 << " us, median " << percentile(second, 0.5) * 1000.0 << " us (p10 " << percentile(second, 0.1) * 1000.0
                      << ", p90 " << percentile(second, 0.9) * 1000.0 << "); fused launch call " << percentile(s.fusedUs, 0.5) << " us; first mean " << fklb::graphs::mean(first) << " ms; " << iters << " iterations"
                      << std::endl;
        };
        const auto timed = [&](const std::string& mode, const std::string& kind, const int iters, auto&& body) {
            if (!options.wants(mode)) { return; }
            try {
                const std::string range = mode + "@" + std::to_string(NUM_OPS);
                nvtxRangePushA(range.c_str());
                const Samples s = body();
                nvtxRangePop();
                report(mode, kind, iters, s);
            } catch (const std::exception& e) {
                ++failures;
                std::cerr << "ops " << NUM_OPS << " " << mode << " failed: " << e.what() << std::endl;
                cudaGetLastError();
            }
        };
        const auto none = [] {};
        HostSweep sweep2(2u << 20), sweep64(64u << 20);
        const auto enqueueFn = [this](const int crop, cudaStream_t stream) { this->enqueue(crop, stream); };

        fklb::graphs::Graph child, flatSerial;
        if (options.modes.empty() || options.modes.find("loop") != std::string::npos) {
            fklb::graphs::buildChild(child, REAL_BATCH, enqueueFn, false);
            fklb::graphs::buildFlatSerial(flatSerial, REAL_BATCH, enqueueFn);
        }
        const auto graphLaunch = [this](fklb::graphs::Graph& g) { return [this, &g] { FKLB_CUDA(cudaGraphLaunch(g.exec, m_raw)); }; };
        const auto plain = [this] { plainBaseline(); };
        const auto pollute2 = [&] { sweep2.run(); };
        const auto pollute64 = [&] { sweep64.run(); };

        timed("loop_plain", "loop", loopIters, [&] { return loop(options.loopWarmup, loopIters, plain, none); });
        timed("loop_child", "loop", loopIters, [&] { return loop(options.loopWarmup, loopIters, graphLaunch(child), none); });
        timed("loop_flat_serial", "loop", loopIters, [&] { return loop(options.loopWarmup, loopIters, graphLaunch(flatSerial), none); });
        timed("alone_synced", "alone", options.aloneIters, [&] { return loop(options.aloneWarmup, options.aloneIters, none, none); });
        timed("alone_synced_pollute2MB", "alone", options.aloneIters, [&] { return loop(options.aloneWarmup, options.aloneIters, none, pollute2); });
        timed("alone_synced_pollute64MB", "alone", options.aloneIters, [&] { return loop(options.aloneWarmup, options.aloneIters, none, pollute64); });
        timed("alone_pipelined10", "alone", options.aloneIters, [&] { return pipelined(options, 10); });
        timed("alone_pipelined50", "alone", options.aloneIters, [&] { return pipelined(options, 50); });
        timed("alone_queued_behind_spin", "alone", options.aloneIters, [&] { return queuedBehindSpin(options); });
        return failures;
    }
};

template <size_t... Ops>
int dispatch(const std::vector<size_t>& opsList, fklb::CsvWriter& csv, std::ofstream& raw, const Options& options, std::index_sequence<Ops...>) {
    int failures = 0;
    for (const size_t ops : opsList) {
        bool found = false;
        const auto one = [&](auto opsConstant) {
            constexpr size_t numOps = decltype(opsConstant)::value;
            if (ops == numOps) {
                found = true;
                Probe<numOps> probe;
                failures += probe.run(csv, raw, options);
            }
        };
        (one(std::integral_constant<size_t, Ops>{}), ...);
        if (!found) {
            std::cerr << "vf_fused_time_probe: " << ops << " operations is not one of the compiled sizes" << std::endl;
            ++failures;
        }
    }
    return failures;
}
}  // namespace

int launch() {
    std::vector<size_t> opsList;
    {
        const char* env = std::getenv("FKLB_VF_OPS");
        std::stringstream ss(env != nullptr ? env : "502,2502,10002,17702");
        std::string item;
        while (std::getline(ss, item, ',')) { opsList.push_back(static_cast<size_t>(std::stoull(item))); }
    }
    Options options;
    if (const char* env = std::getenv("FKLB_VF_LOOP_ITERS")) { options.loopIters = std::atoi(env); }
    if (const char* env = std::getenv("FKLB_VF_LOOP_WARMUP")) { options.loopWarmup = std::atoi(env); }
    if (const char* env = std::getenv("FKLB_VF_ALONE_ITERS")) { options.aloneIters = std::atoi(env); }
    if (const char* env = std::getenv("FKLB_VF_ALONE_WARMUP")) { options.aloneWarmup = std::atoi(env); }
    if (const char* env = std::getenv("FKLB_VF_MODES")) { options.modes = env; }
    int failures = 0;
    try {
        fklb::CsvWriter csv("vf_fused_time_probe.csv", "E0 the time of the fused kernel (cvGPUSpeedup) of the VF + HF experiment in the loops of the experiment and on its own (50 crops of 60 x 120)",
                            {"ops", "mode", "kind", "iters", "first_ms_mean", "second_ms_mean", "second_ms_median", "second_ms_min", "second_ms_max", "second_ms_p10", "second_ms_p90",
                             "fused_launch_call_us_median", "fused_launch_call_us_p90", "baseline_launch_us_median"},
                            "second = the interval the experiment reports for the fused kernel (events recorded on the stream; loop_* and alone_synced as the loop of the experiment; alone_pipelined*: total over N launches "
                            "divided by N; alone_queued_behind_spin: between the event after a kernel of about 1 ms and the event after the fused kernel); first = the baseline interval; "
                            "fused_launch_call_us = host time of the call that launches the fused kernel (per launch for alone_pipelined*)");
        std::ofstream raw("vf_fused_time_probe_raw.csv");
        raw << "ops,mode,iteration,first_ms,second_ms,fused_launch_call_us\n";
        raw << std::setprecision(9);
        failures += dispatch(opsList, csv, raw, options, std::index_sequence<502, 2502, 10002, 17702>{});
    } catch (const std::exception& e) {
        std::cerr << "vf_fused_time_probe: " << e.what() << std::endl;
        return -1;
    }
    return failures == 0 ? 0 : -1;
}
