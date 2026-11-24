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

// FKL's filter DPPs on the GPU (ParArch::GPU_AMD) before and after task B (the batched filters). The same source is built against FKL 24478f0
// (before) and against the task B head (after) by build.sh. The only difference between the two is the details, which lose width and height
// after task B (makeDetails below picks the form of the FKL tree); executeConvQuad returns void before and bool after, and the result is not used.
//
// 1. The outputs. The GPU calls of the existing tests whose checks have a tolerance (tests/image_processing at 24478f0), with the same
//    instantiations and inputs, each checked against its host reference with the test's tolerance, and every output appended to
//    filters_outputs.bin, so that the outputs of the two builds can be compared bit for bit:
//    - test_linear_filter_dpp: box3 and box5 (executeBoxFilter), anisotropic-fused, mutate-mul and mutate-add (executeLinearFilter), with
//      LinearFilterDPPDetails<float, 16, 8, 7, 7>; tolerance 2e-5;
//    - test_median_filter_dpp: median3, median5, median7x3-fused and minimum-selection (executeMedianFilter), with
//      MedianFilterDPPDetails<float, 16, 8, 7, 7>; tolerance 1e-6;
//    - test_convolution_fast_dpp: 3x3-odd, 5x5, 7x7-ragged, static-details and runtime-3x5 (executeConvQuad, ConvQuadDPP<GPU_AMD, float, EX,
//      EY, KW, KH>); tolerance 2e-5.
//    The Quads of uchar (MorphQuad, MedianQuad, BoxFilterQuad) are not here: their tests compare every output with == to the reference.
// 2. The speed of the scalar median filter (Oscar's request): the four median cases, with E19's protocol (fkl-benchmarks
//    rpp/e19_batched_box.h): one gpuSpin of 2 s (FILTERS_CLOCK_WARMUP_MS) to raise the clock, then per case
//      whole_call   fklb::timeIt (common/hip_timing.h): events around one call, 100 calls after one warm-up call;
//      kernel_only  fklb::timePreparedLaunch with a gpuSpin of 5 ms (FILTERS_SPIN_US) before each call: the call's GPU interval alone;
//    with the graphics clock (hwmon sclk) read before and after each interval, into filters_speed_GPU_AMD.csv. Only when every check passed.
// Every call is a direct fk:: execute function on the stream; nothing is wrapped around it. The program prints the FKL commit it was built with
// (FILTERS_FKL_COMMIT, set by build.sh). FILTERS_NO_TIMING=1: the outputs only.

#include <common/hip_timing.h>
#include <common/hip_wallclock.h>

#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/convolution_fast.h>
#include <fused_kernel/algorithms/image_processing/linear_filter.h>
#include <fused_kernel/algorithms/image_processing/median_filter.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#ifndef FILTERS_FKL_COMMIT
#error "build.sh sets FILTERS_FKL_COMMIT to the FKL commit"
#endif

namespace {

using GpuStream = fk::Stream_<fk::ParArch::GPU_AMD>;

// The details of the FKL tree: with width and height (24478f0) or without them (task B)
template <typename D, typename = void>
struct HasImageSize : std::false_type {};
template <typename D>
struct HasImageSize<D, std::void_t<decltype(std::declval<D&>().width)>> : std::true_type {};

template <typename D>
D makeDetails(const int width, const int height, const int windowWidth, const int windowHeight, const int anchorX, const int anchorY) {
    if constexpr (HasImageSize<D>::value) {
        return D{width, height, windowWidth, windowHeight, anchorX, anchorY};
    } else {
        (void)width;
        (void)height;
        return D{windowWidth, windowHeight, anchorX, anchorY};
    }
}

fk::Ptr2D<float> image(const int width, const int height, const std::function<float(int, int)>& value, GpuStream& stream) {
    fk::Ptr2D<float> p(static_cast<uint>(width), static_cast<uint>(height));
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) p.at(fk::Point{x, y, 0}) = value(x, y);
    p.upload(stream);
    return p;
}

// The result of one case's check: its output, its host reference and the tolerance
struct Checked {
    std::string name;
    bool launched;
    std::vector<float> output, expected;
    float tolerance;
};

std::vector<float> download(fk::Ptr2D<float>& p, GpuStream& stream) {
    p.download(stream);
    stream.sync();
    const int width = static_cast<int>(p.ptr().dims.width), height = static_cast<int>(p.ptr().dims.height);
    std::vector<float> out(static_cast<size_t>(width * height));
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) out[static_cast<size_t>(y * width + x)] = p.at(fk::Point{x, y, 0});
    return out;
}

// ---- test_linear_filter_dpp
namespace linear {
using Details = fk::LinearFilterDPPDetails<float, 16, 8, 7, 7>;
enum class Arithmetic { NORMAL, MUTATE_MUL_TO_ADD, MUTATE_ADD_TO_SUB };
float inputValue(const int x, const int y) { return static_cast<float>((x * 5 + y * 7) % 19 - 9) * 0.125f; }
float coefficientValue(const int x, const int y) { return static_cast<float>((x * 3 + y * 2) % 11 - 5) * 0.0625f; }

std::vector<float> reference(const int width, const int height, const int kw, const int kh, const int ax, const int ay, const bool box,
                             const bool fused, const bool fusedKernel, const Arithmetic arithmetic) {
    std::vector<float> output(static_cast<size_t>(width * height));
    for (int oy = 0; oy < height; ++oy)
        for (int ox = 0; ox < width; ++ox) {
            float accumulator = 0.f;
            for (int ky = 0; ky < kh; ++ky)
                for (int kx = 0; kx < kw; ++kx) {
                    const int sx = std::max(0, std::min(width - 1, ox + kx - ax));
                    const int sy = std::max(0, std::min(height - 1, oy + ky - ay));
                    float value = inputValue(sx, sy);
                    float coefficient = box ? 1.f / static_cast<float>(kw * kh) : coefficientValue(kx, ky);
                    if (fused) value = value * 2.f + 1.f;
                    if (fusedKernel) coefficient *= 0.5f;
                    const float product = arithmetic == Arithmetic::MUTATE_MUL_TO_ADD ? value + coefficient : value * coefficient;
                    accumulator = arithmetic == Arithmetic::MUTATE_ADD_TO_SUB ? accumulator - product : accumulator + product;
                }
            output[static_cast<size_t>(oy * width + ox)] = fused ? accumulator * 0.25f : accumulator;
        }
    return output;
}

template <typename Multiply, typename Accumulate>
Checked runCase(const char* name, const int width, const int height, const int kw, const int kh, const int ax, const int ay, const bool box,
                const bool fused, const bool fusedKernel, const Arithmetic arithmetic, const Multiply& multiply, const Accumulate& accumulate,
                GpuStream& stream) {
    const Details details = makeDetails<Details>(width, height, kw, kh, ax, ay);
    auto input = image(width, height, inputValue, stream);
    auto kernel = image(kw, kh, [&](int x, int y) { return box ? 1.f / static_cast<float>(kw * kh) : coefficientValue(x, y); }, stream);
    auto output = image(width, height, [](int, int) { return -999.f; }, stream);
    const auto imageBase = fk::PerThreadRead<fk::ND::_2D, float>::build(input);
    const auto kernelBase = fk::PerThreadRead<fk::ND::_2D, float>::build(kernel);
    const auto writeBase = fk::PerThreadWrite<fk::ND::_2D, float>::build(output);
    bool launched = false;
    if (box && !fused && !fusedKernel && arithmetic == Arithmetic::NORMAL) {
        launched = fk::executeBoxFilter(details, imageBase, writeBase, 1.f, stream);
    } else if (fused) {
        const auto imageRead = imageBase.then(fk::Mul<float>::build(2.f)).then(fk::Add<float>::build(1.f));
        const auto kernelRead = kernelBase.then(fk::Mul<float>::build(0.5f));
        const auto write = fk::Mul<float>::build(0.25f).then(writeBase);
        launched = fk::executeLinearFilter(details, fk::make_tuple(imageRead, kernelRead), multiply, accumulate, write, stream);
    } else if (fusedKernel) {
        const auto kernelRead = kernelBase.then(fk::Mul<float>::build(0.5f));
        launched = fk::executeLinearFilter(details, fk::make_tuple(imageBase, kernelRead), multiply, accumulate, writeBase, stream);
    } else {
        launched = fk::executeLinearFilter(details, fk::make_tuple(imageBase, kernelBase), multiply, accumulate, writeBase, stream);
    }
    return {std::string("linear ") + name, launched, download(output, stream),
            reference(width, height, kw, kh, ax, ay, box, fused, fusedKernel, arithmetic), 2e-5f};
}

void run(std::vector<Checked>& checked, GpuStream& stream) {
    const auto mul = fk::Mul<float, float, float, fk::UnaryType>::build();
    const auto add = fk::Add<float, float, float, fk::UnaryType>::build();
    const auto sub = fk::Sub<float, float, float, fk::UnaryType>::build();
    checked.push_back(runCase("box3", 37, 19, 3, 3, 1, 1, true, false, false, Arithmetic::NORMAL, mul, add, stream));
    checked.push_back(runCase("box5", 35, 17, 5, 5, 2, 2, true, false, false, Arithmetic::NORMAL, mul, add, stream));
    checked.push_back(runCase("anisotropic-fused", 33, 21, 7, 3, 5, 1, false, true, true, Arithmetic::NORMAL, mul, add, stream));
    checked.push_back(runCase("mutate-mul", 19, 13, 3, 3, 0, 2, false, false, false, Arithmetic::MUTATE_MUL_TO_ADD, add, add, stream));
    checked.push_back(runCase("mutate-add", 21, 15, 3, 5, 1, 3, false, false, false, Arithmetic::MUTATE_ADD_TO_SUB, mul, sub, stream));
}
}  // namespace linear

// ---- test_median_filter_dpp, also the cases of the speed
namespace median {
using Details = fk::MedianFilterDPPDetails<float, 16, 8, 7, 7>;
constexpr int CAPACITY = Details::WINDOW_CAPACITY;
float inputValue(const int x, const int y) { return static_cast<float>((x * 11 + y * 7) % 29 - 14) * 0.125f; }

struct Case {
    std::string name;
    int width, height, windowWidth, windowHeight, anchorX, anchorY;
    bool fused, minimum;
};
const std::vector<Case> CASES = {{"median3", 37, 19, 3, 3, 1, 1, false, false},
                                 {"median5", 35, 17, 5, 5, 2, 2, false, false},
                                 {"median7x3-fused", 33, 21, 7, 3, 5, 1, true, false},
                                 {"minimum-selection", 31, 15, 3, 5, 0, 3, false, true}};

std::vector<float> reference(const Case& c) {
    std::vector<float> output(static_cast<size_t>(c.width * c.height));
    for (int oy = 0; oy < c.height; ++oy)
        for (int ox = 0; ox < c.width; ++ox) {
            std::vector<float> values;
            for (int wy = 0; wy < c.windowHeight; ++wy)
                for (int wx = 0; wx < c.windowWidth; ++wx) {
                    const int sx = std::max(0, std::min(c.width - 1, ox + wx - c.anchorX));
                    const int sy = std::max(0, std::min(c.height - 1, oy + wy - c.anchorY));
                    float value = inputValue(sx, sy);
                    if (c.fused) value = value * 2.f + 1.f;
                    values.push_back(value);
                }
            std::sort(values.begin(), values.end());
            const float selected = c.minimum ? values.front() : values[values.size() / 2];
            output[static_cast<size_t>(oy * c.width + ox)] = c.fused ? selected * 0.5f : selected;
        }
    return output;
}

// One case: its buffers and its call (one fk::executeMedianFilter on the stream)
struct Prepared {
    Case c;
    fk::Ptr2D<float> input, output;
    std::function<bool()> call;
};

std::vector<Prepared> prepare(GpuStream& stream) {
    std::vector<Prepared> prepared;
    for (const Case& c : CASES) {
        Prepared p{c, image(c.width, c.height, inputValue, stream), image(c.width, c.height, [](int, int) { return -999.f; }, stream), {}};
        const Details details = makeDetails<Details>(c.width, c.height, c.windowWidth, c.windowHeight, c.anchorX, c.anchorY);
        const auto read = fk::PerThreadRead<fk::ND::_2D, float>::build(p.input);
        const auto write = fk::PerThreadWrite<fk::ND::_2D, float>::build(p.output);
        if (c.fused) {
            const auto fusedRead = read.then(fk::Mul<float>::build(2.f)).then(fk::Add<float>::build(1.f));
            const auto fusedWrite = fk::Mul<float>::build(0.5f).then(write);
            const auto median = fk::MedianWindowSelect<float, CAPACITY>::build();
            p.call = [details, fusedRead, fusedWrite, median, &stream] {
                return fk::executeMedianFilter(details, fusedRead, fusedWrite, median, stream);
            };
        } else if (c.minimum) {
            const auto minimum = fk::MinimumWindowSelect<float, CAPACITY>::build();
            p.call = [details, read, write, minimum, &stream] { return fk::executeMedianFilter(details, read, write, minimum, stream); };
        } else {
            const auto median = fk::MedianWindowSelect<float, CAPACITY>::build();
            p.call = [details, read, write, median, &stream] { return fk::executeMedianFilter(details, read, write, median, stream); };
        }
        prepared.push_back(std::move(p));
    }
    return prepared;
}

void run(std::vector<Checked>& checked, std::vector<Prepared>& prepared, GpuStream& stream) {
    for (Prepared& p : prepared) {
        const bool launched = p.call();
        checked.push_back({"median " + p.c.name, launched, download(p.output, stream), reference(p.c), 1e-6f});
    }
}
}  // namespace median

// ---- test_convolution_fast_dpp
namespace conv {
constexpr float READ_BIAS = 0.25f;
constexpr float WRITE_BIAS = -0.5f;
float sourceValue(const int x, const int y) { return static_cast<float>(((x * 19 + y * 23) % 97) - 48) / 17.f; }
std::vector<float> coefficientsOf(const int width, const int height) {
    std::vector<float> values(static_cast<size_t>(width * height));
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            values[static_cast<size_t>(y * width + x)] =
                static_cast<float>(((x + 1) * 7 + (y + 2) * 11) % 29 - 14) / static_cast<float>(width * height * 5);
    return values;
}

// The details of the FKL tree, as makeDetails, with the coefficients left to the caller
template <typename D>
D convDetails(const int width, const int height, const int kw, const int kh, const int ax, const int ay) {
    if constexpr (HasImageSize<D>::value) {
        return D{width, height, kw, kh, ax, ay, {}};
    } else {
        (void)width;
        (void)height;
        return D{kw, kh, ax, ay, {}};
    }
}

template <int EX, int EY, int KW, int KH>
Checked runCase(const char* name, const int width, const int height, const int runtimeKW, const int runtimeKH, const int ax, const int ay,
                GpuStream& stream) {
    const int kw = KW > 0 ? KW : runtimeKW, kh = KH > 0 ? KH : runtimeKH;
    const std::vector<float> coefficients = coefficientsOf(kw, kh);
    auto details = convDetails<fk::ConvQuadDetails>(width, height, runtimeKW, runtimeKH, ax, ay);
    std::copy(coefficients.begin(), coefficients.end(), details.coefficients);
    auto input = image(width, height, sourceValue, stream);
    auto output = image(width, height, [](int, int) { return 1234.f; }, stream);
    const auto read = fk::PerThreadRead<fk::ND::_2D, float>::build(input).then(fk::Add<float>::build(READ_BIAS));
    const auto compute = fk::make_tuple(fk::Mul<float, float, float, fk::UnaryType>::build(), fk::Add<float, float, float, fk::UnaryType>::build());
    const auto write = fk::Add<float>::build(WRITE_BIAS).then(fk::PerThreadWrite<fk::ND::_2D, float>::build(output));
    using DPP = fk::ConvQuadDPP<fk::ParArch::GPU_AMD, float, EX, EY, KW, KH>;
    fk::executeConvQuad<DPP>(stream, details, read, compute, write);
    std::vector<float> expected(static_cast<size_t>(width * height));
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) {
            float result = 0.f;
            for (int ky = 0; ky < kh; ++ky)
                for (int kx = 0; kx < kw; ++kx) {
                    const int sx = std::max(0, std::min(width - 1, x + kx - ax));
                    const int sy = std::max(0, std::min(height - 1, y + ky - ay));
                    result += coefficients[static_cast<size_t>(ky * kw + kx)] * (sourceValue(sx, sy) + READ_BIAS);
                }
            expected[static_cast<size_t>(y * width + x)] = result + WRITE_BIAS;
        }
    return {std::string("conv ") + name, true, download(output, stream), expected, 2e-5f};
}

void run(std::vector<Checked>& checked, GpuStream& stream) {
    checked.push_back(runCase<4, 4, 3, 3>("3x3-odd", 257, 129, 3, 3, 1, 1, stream));
    checked.push_back(runCase<4, 4, 5, 5>("5x5", 192, 108, 5, 5, 2, 2, stream));
    checked.push_back(runCase<2, 3, 7, 7>("7x7-ragged", 65, 37, 7, 7, 3, 3, stream));
    checked.push_back(runCase<4, 4, 7, 7>("static-details", 39, 27, 1, 1, 3, 3, stream));
    checked.push_back(runCase<4, 4, 0, 0>("runtime-3x5", 73, 41, 3, 5, 0, 3, stream));
}
}  // namespace conv

__global__ void gpuSpin(const unsigned long long ticks) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {
    }
}

unsigned long long envOr(const char* name, const unsigned long long fallback) {
    const char* e = std::getenv(name);
    return e != nullptr ? std::strtoull(e, nullptr, 10) : fallback;
}

long long realtimeNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

}  // namespace

int launch() {
    std::cout << "filters_before_after: FKL " << FILTERS_FKL_COMMIT << std::endl;
    int wallClockKHz = 0;
    FKLB_HIP_CHECK(hipDeviceGetAttribute(&wallClockKHz, hipDeviceAttributeWallClockRate, 0));
    GpuStream stream;
    const hipStream_t hipStream = stream.getHIPStream();

    // ---- 1. The outputs
    std::vector<Checked> checked;
    linear::run(checked, stream);
    std::vector<median::Prepared> medians = median::prepare(stream);
    median::run(checked, medians, stream);
    conv::run(checked, stream);
    bool allOk = true;
    std::ofstream outputs("filters_outputs.bin", std::ios::binary);
    for (const Checked& c : checked) {
        size_t beyond = 0;
        for (size_t k = 0; k < c.output.size(); ++k)
            if (!(std::fabs(c.output[k] - c.expected[k]) <= c.tolerance)) ++beyond;
        outputs.write(reinterpret_cast<const char*>(c.output.data()), static_cast<std::streamsize>(c.output.size() * sizeof(float)));
        const bool ok = c.launched && beyond == 0;
        std::cout << "check " << c.name << ": " << (c.launched ? "launched" : "NOT LAUNCHED") << ", " << c.output.size() << " values, " << beyond
                  << " beyond " << c.tolerance << " of the host reference: " << (ok ? "ok" : "FAILED") << std::endl;
        allOk = allOk && ok;
    }
    outputs.close();
    std::cout << (allOk ? "filters_before_after checks passed" : "filters_before_after CHECK FAILED") << std::endl;
    if (!allOk) return 1;
    if (std::getenv("FILTERS_NO_TIMING") != nullptr) return 0;

    // ---- 2. The speed of the median cases: E19's protocol
    const unsigned long long warmupMs = envOr("FILTERS_CLOCK_WARMUP_MS", 2000);
    const unsigned long long spinUs = envOr("FILTERS_SPIN_US", 5000);
    const unsigned long long spinTicks = spinUs * static_cast<unsigned long long>(wallClockKHz) / 1000ULL;
    std::ofstream csv("filters_speed_GPU_AMD.csv");
    csv << "# filters_before_after (FKL " << FILTERS_FKL_COMMIT << "): the median cases' whole-call and kernel-only times; clock warm-up: one gpuSpin of "
        << warmupMs << " ms before the timing\n";
    csv << "case,interval,mean_ms,min_ms,host_prepare_mean_ms,host_prepare_min_ms,iterations,spin_us,sclk_before_mhz,sclk_after_mhz\n";
    const double sclkBeforeWarmup = fklb::amdGpuClockMHz();
    const long long warmupStartNs = realtimeNs();
    gpuSpin<<<1, 1, 0, hipStream>>>(warmupMs * static_cast<unsigned long long>(wallClockKHz));
    FKLB_HIP_CHECK(hipGetLastError());
    FKLB_HIP_CHECK(hipDeviceSynchronize());
    const double sclkAfterWarmup = fklb::amdGpuClockMHz();
    const long long timingStartNs = realtimeNs();
    for (const median::Prepared& p : medians) {
        fklb::Timing whole, gpu, host;
        FKLB_HIP_CHECK(hipDeviceSynchronize());
        const double s0 = fklb::amdGpuClockMHz();
        fklb::timeIt(hipStream, [&] { p.call(); }, whole);
        FKLB_HIP_CHECK(hipDeviceSynchronize());
        const double s1 = fklb::amdGpuClockMHz();
        const double s2 = fklb::amdGpuClockMHz();
        fklb::timePreparedLaunch(
            hipStream, [] { return 0; }, [&] { gpuSpin<<<1, 1, 0, hipStream>>>(spinTicks); }, [&](int) { p.call(); }, gpu, host);
        const double s3 = fklb::amdGpuClockMHz();
        csv << p.c.name << ",whole_call," << whole.mean << "," << whole.min << ",,," << fklb::ITERATIONS << ",," << s0 << "," << s1 << "\n";
        csv << p.c.name << ",kernel_only," << gpu.mean << "," << gpu.min << "," << host.mean << "," << host.min << "," << fklb::ITERATIONS << ","
            << spinUs << "," << s2 << "," << s3 << "\n";
        std::cout << p.c.name << ": whole call mean " << whole.mean << " ms (min " << whole.min << "); kernel only mean " << gpu.mean << " ms (min "
                  << gpu.min << "); sclk " << s0 << " / " << s1 << " / " << s2 << " / " << s3 << " MHz" << std::endl;
    }
    const long long timingEndNs = realtimeNs();
    csv.close();
    std::cout << "clock warm-up: one gpuSpin of " << warmupMs << " ms; sclk " << sclkBeforeWarmup << " MHz before it, " << sclkAfterWarmup
              << " MHz after it" << std::endl;
    std::cout << "filters_before_after timing (realtime ns): warm-up start " << warmupStartNs << ", timing start " << timingStartNs
              << ", timing end " << timingEndNs << std::endl;
    FKLB_HIP_CHECK(hipDeviceSynchronize());
    return 0;
}

int main() {
    try {
        return launch();
    } catch (const std::exception& e) {
        std::cerr << "filters_before_after: " << e.what() << std::endl;
        return 2;
    }
}
