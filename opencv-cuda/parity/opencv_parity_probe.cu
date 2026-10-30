// Copyright 2026 Oscar Amoros Huguet
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Diagnostic of the data-size experiment (E0): in benchmark_image_resolution_MAD_loop the OpenCV-CUDA time of the one-channel pairs is about
// 3x higher at every point whose element count N satisfies N mod 4 = 2 than at its neighbours with N mod 4 = 0. This program runs only the
// OpenCV-CUDA side of that benchmark, exactly as it is written there (a 1 x N GpuMat, convertTo, then 100 x (multiply by a scalar, add a scalar)),
// for the element counts and type pairs given in the environment, and reports the time of the whole sequence and of each of the three calls on
// their own, and (for the one-channel pairs) the cvGS fused kernel of the same benchmark. With FKLB_PARITY_ONCE=1 it launches convertTo, one multiply,
// one add and the fused kernel once per case and nothing else, for Nsight Compute to name the kernels and to measure them.
//
//   FKLB_PARITY_TYPES  "8UC1:32FC1,32FC1:32FC1,8UC3:32FC3"   input:output types (8UC1 8UC3 16UC4 32SC4 32FC1 32FC3 32FC4 64FC4)
//   FKLB_PARITY_SIZES  "282370,564640,..."                    element counts (columns of the 1 x N matrix)
//   FKLB_PARITY_REPS   100                                    timed repetitions of the whole sequence and of each call
//   FKLB_PARITY_ONCE   0                                      1: the three launches only (for ncu)
//   FKLB_PARITY_OUT    opencv_parity_probe.csv
#include <common/main.h>
#include <common/csv_writer.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>

#include <cvGPUSpeedup.cuh>
#include <fused_kernel/algorithms/basic_ops/static_loop.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(call)                                                                                              \
    {                                                                                                            \
        const cudaError_t err_ = (call);                                                                         \
        if (err_ != cudaSuccess) {                                                                               \
            std::cerr << #call << " failed: " << cudaGetErrorString(err_) << " (" << __FILE__ << ":" << __LINE__ \
                      << ")" << std::endl;                                                                       \
            return -1;                                                                                           \
        }                                                                                                        \
    }

namespace {
const std::map<std::string, int>& typeTable() {
    static const std::map<std::string, int> t{{"8UC1", CV_8UC1},   {"8UC3", CV_8UC3},   {"16UC4", CV_16UC4}, {"32SC4", CV_32SC4},
                                              {"32FC1", CV_32FC1}, {"32FC3", CV_32FC3}, {"32FC4", CV_32FC4}, {"64FC4", CV_64FC4}};
    return t;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, sep)) {
        if (!item.empty()) out.push_back(item);
    }
    return out;
}

std::string envOr(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return v ? v : fallback;
}

// the cvGS side of benchmark_image_resolution_MAD_loop (200 operations: 100 x (Mul, Add) in a StaticLoop), copied from there
template <int CV_TYPE_I, int CV_TYPE_O>
void fusedMAD(const cv::cuda::GpuMat& cvInput, cv::cuda::Stream& cv_stream, const cv::Scalar& val_mul, const cv::Scalar& val_add, const cv::cuda::GpuMat& d_output) {
    using InputType = CUDA_T(CV_TYPE_I);
    using OutputType = CUDA_T(CV_TYPE_O);
    using Loop = fk::Binary<fk::StaticLoop<fk::FusedOperation<fk::Binary<fk::Mul<OutputType>>, fk::Binary<fk::Add<OutputType>>>, 200 / 2>>;
    Loop loop;
    fk::get_opt<0>(loop.params).params = cvGS::cvScalar2CUDAV<CV_TYPE_O>::get(val_mul);
    fk::get_opt<1>(loop.params).params = cvGS::cvScalar2CUDAV<CV_TYPE_O>::get(val_add);
    const uint inputWidth = static_cast<uint>(cvInput.cols);
    const uint outputWidth = static_cast<uint>(d_output.cols);
    const fk::RawPtr<fk::ND::_1D, InputType> fkInput{reinterpret_cast<InputType*>(cvInput.data), {inputWidth, static_cast<uint>(inputWidth * sizeof(InputType))}};
    const auto readOp = fk::PerThreadRead<fk::ND::_1D, InputType>::build(fkInput);
    const fk::RawPtr<fk::ND::_1D, OutputType> fkOutput{reinterpret_cast<OutputType*>(d_output.data), {outputWidth, static_cast<uint>(outputWidth * sizeof(OutputType))}};
    const auto writeOp = fk::PerThreadWrite<fk::ND::_1D, OutputType>::build(fkOutput);
    const auto tDetails = fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>::build_details(readOp, cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>(), loop, writeOp);
    const dim3 block(256);
    const dim3 grid(ceil(inputWidth / static_cast<float>(block.x)));
    const cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cv_stream);
    fk::launchTransformDPP_Kernel<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED, true><<<grid, block, 0, stream>>>(tDetails, readOp, cvGS::convertTo<CV_TYPE_I, CV_TYPE_O>(), loop, writeOp);
}

// false when the pair has no fused instance here
bool launchFused(int tIn, int tOut, const cv::cuda::GpuMat& in, cv::cuda::Stream& stream, const cv::Scalar& mul, const cv::Scalar& add, const cv::cuda::GpuMat& out) {
    if (tIn == CV_8UC1 && tOut == CV_32FC1) { fusedMAD<CV_8UC1, CV_32FC1>(in, stream, mul, add, out); return true; }
    if (tIn == CV_32FC1 && tOut == CV_32FC1) { fusedMAD<CV_32FC1, CV_32FC1>(in, stream, mul, add, out); return true; }
    return false;
}

// the scalars of benchmark_image_resolution_MAD_loop, per number of channels
struct Scalars {
    cv::Scalar init, mul, add;
};
Scalars scalarsFor(int channels) {
    const double init[4] = {1., 2., 3., 4.}, mul[4] = {1., 4., 2., 0.5}, add[4] = {-3.2f, -0.6f, -3.8f, -33.};
    Scalars s;
    for (int c = 0; c < 4; ++c) {
        s.init[c] = c < channels ? init[c] : 0.;
        s.mul[c] = c < channels ? mul[c] : 0.;
        s.add[c] = c < channels ? add[c] : 0.;
    }
    return s;
}

template <typename F>
double timeMs(cudaStream_t stream, F&& f) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, stream);
    f();
    cudaEventRecord(b, stream);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms;
}
}  // namespace

int launch() {
    const bool once = envOr("FKLB_PARITY_ONCE", "0") == "1";
    const int reps = std::stoi(envOr("FKLB_PARITY_REPS", "100"));
    const auto types = split(envOr("FKLB_PARITY_TYPES", "8UC1:32FC1,32FC1:32FC1,8UC3:32FC3"), ',');
    const auto sizes = split(envOr("FKLB_PARITY_SIZES",
                                   "100,282370,564640,846910,1129180,1000000,1000001,1000002,1000003,1000004,1000005,1000006,1000007,2000000,2000002"), ',');
    constexpr int PAIRS = 100;      // as in the benchmark: numOp from 0 to 198 step 2
    constexpr int CALLS = 20;       // back-to-back calls of one kind for the per-call time

    cv::cuda::Stream cv_stream;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cv_stream);
    fklb::CsvWriter csv(envOr("FKLB_PARITY_OUT", "opencv_parity_probe.csv"), "OpenCV-CUDA side of the data-size benchmark, by element count",
                        {"type_in", "type_out", "n", "n_mod_4", "n_mod_8", "sequence_mean_ms", "sequence_min_ms", "convert_call_us", "multiply_call_us", "add_call_us", "fused_mean_ms"});

    for (const std::string& pair : types) {
        const auto parts = split(pair, ':');
        if (parts.size() != 2 || !typeTable().count(parts[0]) || !typeTable().count(parts[1])) {
            std::cerr << "bad type pair '" << pair << "'" << std::endl;
            return -1;
        }
        const int tIn = typeTable().at(parts[0]), tOut = typeTable().at(parts[1]);
        const Scalars sc = scalarsFor(CV_MAT_CN(tOut));
        for (const std::string& sizeText : sizes) {
            const int n = std::stoi(sizeText);
            const cv::Size dataSize(n, 1);
            cv::cuda::GpuMat cvInput(dataSize, tIn, sc.init), cvOutput(dataSize, tOut, cv::Scalar(0)), cvTemp(dataSize, tOut, cv::Scalar(0)), cvFused(dataSize, tOut, cv::Scalar(0));

            auto sequence = [&] {
                cvInput.convertTo(cvOutput, tOut, 1, cv_stream);
                for (int p = 0; p < PAIRS; ++p) {
                    cv::cuda::multiply(cvOutput, sc.mul, cvTemp, 1.0, -1, cv_stream);
                    cv::cuda::add(cvTemp, sc.add, cvOutput, cv::noArray(), -1, cv_stream);
                }
            };

            if (once) {
                cvInput.convertTo(cvOutput, tOut, 1, cv_stream);
                cv::cuda::multiply(cvOutput, sc.mul, cvTemp, 1.0, -1, cv_stream);
                cv::cuda::add(cvTemp, sc.add, cvOutput, cv::noArray(), -1, cv_stream);
                const bool fused = launchFused(tIn, tOut, cvInput, cv_stream, sc.mul, sc.add, cvFused);
                CHECK(cudaStreamSynchronize(stream));
                std::cout << parts[0] << "->" << parts[1] << " n=" << n << " (mod 4 = " << n % 4 << "): " << (fused ? "4" : "3") << " launches" << std::endl;
                continue;
            }

            for (int w = 0; w < 3; ++w) sequence();
            CHECK(cudaStreamSynchronize(stream));
            std::vector<double> seq;
            for (int r = 0; r < reps; ++r) seq.push_back(timeMs(stream, sequence));

            auto perCallUs = [&](auto&& call) -> double {
                for (int w = 0; w < 3; ++w) call();
                cudaStreamSynchronize(stream);
                double total = 0;
                for (int r = 0; r < reps; ++r) {
                    total += timeMs(stream, [&] { for (int c = 0; c < CALLS; ++c) call(); });
                }
                return total / (static_cast<double>(reps) * CALLS) * 1000.0;
            };
            const double convUs = perCallUs([&] { cvInput.convertTo(cvOutput, tOut, 1, cv_stream); });
            const double mulUs = perCallUs([&] { cv::cuda::multiply(cvOutput, sc.mul, cvTemp, 1.0, -1, cv_stream); });
            const double addUs = perCallUs([&] { cv::cuda::add(cvTemp, sc.add, cvOutput, cv::noArray(), -1, cv_stream); });

            double fusedMs = -1.0;
            if (launchFused(tIn, tOut, cvInput, cv_stream, sc.mul, sc.add, cvFused)) {   // one fused launch, mean of reps
                cudaStreamSynchronize(stream);
                double total = 0;
                for (int r = 0; r < reps; ++r) total += timeMs(stream, [&] { launchFused(tIn, tOut, cvInput, cv_stream, sc.mul, sc.add, cvFused); });
                fusedMs = total / reps;
            }
            const double mean = std::accumulate(seq.begin(), seq.end(), 0.0) / seq.size();
            const double mn = *std::min_element(seq.begin(), seq.end());
            csv.row(parts[0], parts[1], n, n % 4, n % 8, mean, mn, convUs, mulUs, addUs, fusedMs);
            csv.flush();
            std::cout << parts[0] << "->" << parts[1] << " n=" << n << " (mod 4 = " << n % 4 << ", mod 8 = " << n % 8 << "): sequence " << mean << " ms, per call: convertTo "
                      << convUs << " us, multiply " << mulUs << " us, add " << addUs << " us, fused " << fusedMs << " ms" << std::endl;
        }
    }
    return 0;
}
