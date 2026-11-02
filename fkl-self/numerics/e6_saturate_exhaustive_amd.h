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

// E6 (numerics), AMD part: fk::SaturateCast<float, T> over ALL 2^32 float bit patterns, for T in uchar, schar, short, ushort, int, on the device (gfx1200) and on the host.
// The CUDA program (saturate_exhaustive.cu) compares against cv::cuda::device::saturate_cast and the PTX conversion cvt.rni.sat, which exist only for NVIDIA; this one compares
//   fkl_dev   fk::SaturateCast<float, T>::exec on the device
//   fkl_host  the same function on the host (cxp::saturate_cast, the branch free path of FKL #416)
//   hw        the conversion of HIP on the device: __float2int_rn (round to nearest even; HIP's header defines it as (int)__builtin_elementwise_rint(x), which the language leaves unspecified out of range: on this toolchain it saturates to the range of int and a NaN gives 0) and then the clamp to the range of T
//   ref       an independent reference on the host: a NaN gives 0, otherwise std::nearbyint of the value as a double, clamped to the range of T
// in four comparisons, each over all 2^32 patterns: fkl_dev vs hw, fkl_dev vs ref, fkl_host vs fkl_dev, fkl_host vs ref. The disagreeing set is reported per class of float (NaN, +Inf,
// -Inf, denormal, zero, normal) with the first examples (bits, both values), in the columns of e6_saturate_exhaustive.csv of the CUDA program.
// The patterns are swept in chunks of 2^26: the device computes fkl_dev and hw for the chunk (and counts its comparison), the host gets fkl_dev and evaluates fkl_host and ref for the
// same patterns on all its threads. E6_HOST=0 skips the host half (E6_HOST and E6_EXPECT_DISAGREEMENTS are the integers 0 or 1, an empty variable counts as unset: any other value stops the program with exit status 2, before the CSV is opened).
// Exit status: 0 when no pattern disagrees in any comparison; 1 when some do (a regression of fk::SaturateCast fails the run and ctest); 255 when a HIP call failed.
// E6_EXPECT_DISAGREEMENTS=1 is for the negative control (the program built against headers that are known to disagree): the exit status is 0 only if some patterns DO disagree.
#include <common/csv_writer.h>
#include <common/env_args.h>
#include <common/main.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace e6amd {
constexpr int NCLASS = 6, NCMP = 4, NEX = 4;
const char* CLASS_NAMES[NCLASS] = {"NaN", "+Inf", "-Inf", "denormal", "zero", "normal"};
const char* CMP_NAMES[NCMP] = {"fkl_dev_vs_hw", "fkl_dev_vs_ref", "fkl_host_vs_fkl_dev", "fkl_host_vs_ref"};
constexpr uint64_t CLASS_COUNT[NCLASS] = {2ull * ((1ull << 23) - 1), 1, 1, 2ull * ((1ull << 23) - 1), 2, 2ull * 254 * (1ull << 23)};
constexpr uint64_t CHUNK = 1ull << 26;
static_assert((1ull << 32) % CHUNK == 0, "the sweep takes whole chunks: no tail is tested");

struct Example { uint32_t bits; long long a, b; };
struct Counters {
    unsigned long long mismatch[NCMP][NCLASS];
    unsigned int nex[NCMP][NCLASS];
    Example ex[NCMP][NCLASS][NEX];
};

__host__ __device__ inline int classOf(const uint32_t b) {
    const uint32_t e = (b >> 23) & 0xff, m = b & 0x7fffff;
    if (e == 0xff) return m ? 0 : 1 + static_cast<int>(b >> 31);
    if (e == 0) return m ? 3 : 4;
    return 5;
}

// The conversion of HIP and the clamp to the range of T
template <typename T>
__device__ inline long long hwConvert(const float v) {
    const long long r = __float2int_rn(v);
    constexpr long long hi = std::numeric_limits<T>::max(), lo = std::numeric_limits<T>::lowest();
    return r > hi ? hi : (r < lo ? lo : r);
}

// The independent reference, on the host
template <typename T>
inline long long refConvert(const float v) {
    constexpr long long hi = std::numeric_limits<T>::max(), lo = std::numeric_limits<T>::lowest();
    if (v != v) {
        return 0;
    }
    const double r = std::nearbyint(static_cast<double>(v));
    return r >= static_cast<double>(hi) ? hi : (r <= static_cast<double>(lo) ? lo : static_cast<long long>(r));
}

__device__ inline void record(Counters* c, const int cmp, const int cls, const uint32_t bits, const long long a, const long long b) {
    atomicAdd(&c->mismatch[cmp][cls], 1ull);
    const unsigned int k = atomicAdd(&c->nex[cmp][cls], 1u);
    if (k < NEX) c->ex[cmp][cls][k] = Example{bits, a, b};
}

template <typename T>
__global__ void sweepDevice(const uint64_t first, const uint64_t count, int* fklOut, Counters* counters) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t bits = static_cast<uint32_t>(first + i);
    const float v = __uint_as_float(bits);
    const long long f = fk::SaturateCast<float, T>::exec(v);
    const long long h = hwConvert<T>(v);
    fklOut[i] = static_cast<int>(f);
    if (f != h) record(counters, 0, classOf(bits), bits, f, h);
}

inline void hostRecord(Counters& c, const int cmp, const int cls, const uint32_t bits, const long long a, const long long b) {
    ++c.mismatch[cmp][cls];
    if (c.nex[cmp][cls] < NEX) c.ex[cmp][cls][c.nex[cmp][cls]] = Example{bits, a, b};
    ++c.nex[cmp][cls];
}

// The host half of a chunk on `threads` threads; the examples of the merge are the ones with the smallest bit patterns
template <typename T>
void hostHalf(const uint64_t first, const std::vector<int>& dev, Counters& host, const unsigned threads) {
    std::vector<Counters> parts(threads);
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) {
        pool.emplace_back([&, t] {
            Counters& c = parts[t];
            std::memset(&c, 0, sizeof(c));
            for (uint64_t i = t; i < CHUNK; i += threads) {
                const uint32_t bits = static_cast<uint32_t>(first + i);
                float v;
                std::memcpy(&v, &bits, sizeof(v));
                const long long fh = fk::SaturateCast<float, T>::exec(v);
                const long long rf = refConvert<T>(v);
                const int cls = classOf(bits);
                if (fh != dev[i]) hostRecord(c, 2, cls, bits, fh, dev[i]);
                if (fh != rf) hostRecord(c, 3, cls, bits, fh, rf);
                if (static_cast<long long>(dev[i]) != rf) hostRecord(c, 1, cls, bits, dev[i], rf);
            }
        });
    }
    for (auto& th : pool) th.join();
    for (int cmp = 1; cmp < NCMP; ++cmp) {
        for (int cls = 0; cls < NCLASS; ++cls) {
            std::vector<Example> all;
            for (const auto& p : parts) {
                host.mismatch[cmp][cls] += p.mismatch[cmp][cls];
                for (unsigned k = 0; k < std::min<unsigned>(p.nex[cmp][cls], NEX); ++k) all.push_back(p.ex[cmp][cls][k]);
            }
            for (unsigned k = 0; k < host.nex[cmp][cls] && k < NEX; ++k) all.push_back(host.ex[cmp][cls][k]);
            std::sort(all.begin(), all.end(), [](const Example& x, const Example& y) { return x.bits < y.bits; });
            host.nex[cmp][cls] = static_cast<unsigned>(std::min<size_t>(all.size(), NEX));
            for (unsigned k = 0; k < host.nex[cmp][cls]; ++k) host.ex[cmp][cls][k] = all[k];
        }
    }
}

template <typename T>
bool sweepType(const char* typeName, const bool withHost, fklb::CsvWriter& csv, unsigned long long& disagreeing) {
    int* dF = nullptr;
    Counters* dCounters = nullptr;
    if (hipMalloc(&dF, CHUNK * sizeof(int)) != hipSuccess || hipMalloc(&dCounters, sizeof(Counters)) != hipSuccess) return false;
    if (hipMemset(dCounters, 0, sizeof(Counters)) != hipSuccess) return false;
    Counters host{};
    std::vector<int> hF(CHUNK);
    const unsigned threads = std::max(1u, std::thread::hardware_concurrency());
    for (uint64_t first = 0; first < (1ull << 32); first += CHUNK) {
        sweepDevice<T><<<static_cast<unsigned int>((CHUNK + 255) / 256), 256>>>(first, CHUNK, dF, dCounters);
        if (hipGetLastError() != hipSuccess) return false;
        if (withHost) {
            if (hipMemcpy(hF.data(), dF, CHUNK * sizeof(int), hipMemcpyDeviceToHost) != hipSuccess) return false;
            hostHalf<T>(first, hF, host, threads);
        }
    }
    if (hipDeviceSynchronize() != hipSuccess) return false;
    Counters dev{};
    if (hipMemcpy(&dev, dCounters, sizeof(Counters), hipMemcpyDeviceToHost) != hipSuccess) return false;
    for (int cmp = 0; cmp < NCMP; ++cmp) {
        const bool isHost = cmp >= 1 && cmp != 0;
        if (isHost && !withHost) continue;
        const Counters& src = cmp == 0 ? dev : host;
        for (int cls = 0; cls < NCLASS; ++cls) {
            std::ostringstream examples;
            for (unsigned int k = 0; k < std::min<unsigned int>(src.nex[cmp][cls], NEX); ++k) {
                const Example& e = src.ex[cmp][cls][k];
                examples << (k ? "; " : "") << "0x" << std::hex << e.bits << std::dec << ": " << e.a << " vs " << e.b;
            }
            csv.row(typeName, CMP_NAMES[cmp], CLASS_NAMES[cls], CLASS_COUNT[cls], src.mismatch[cmp][cls], examples.str());
        }
    }
    csv.flush();
    std::cout << typeName << ": disagreeing patterns of 2^32:";
    for (int cmp = 0; cmp < NCMP; ++cmp) {
        if (cmp >= 1 && !withHost) continue;
        const Counters& src = cmp == 0 ? dev : host;
        unsigned long long total = 0;
        for (int cls = 0; cls < NCLASS; ++cls) total += src.mismatch[cmp][cls];
        disagreeing += total;
        std::cout << "  " << CMP_NAMES[cmp] << " " << total;
    }
    std::cout << std::endl;
    (void)hipFree(dF);
    (void)hipFree(dCounters);
    return true;
}
} // namespace e6amd

int launch() {
    const bool withHost = fklb::envInt("E6_HOST", 1, 0, 1) != 0;
    const bool expectDisagreements = fklb::envInt("E6_EXPECT_DISAGREEMENTS", 0, 0, 1) != 0;
    fklb::CsvWriter csv("e6_saturate_exhaustive_amd.csv", "E6 fk::SaturateCast<float, T> over all 2^32 float bit patterns, gfx1200",
                        {"type", "comparison", "class", "patterns_in_class", "disagreeing", "examples"},
                        std::string("comparisons: fkl_dev = fk::SaturateCast on the device; hw = __float2int_rn of HIP (round to nearest even; unspecified out of range by the language, on this toolchain saturating with NaN giving 0) clamped to the range of the type; ") +
                            "fkl_host = fk::SaturateCast on the host (cxp::saturate_cast); ref = a NaN gives 0, otherwise std::nearbyint of the value as a double clamped to the range of the type, on the host; " +
                            "classes by IEEE-754 binary32 bit pattern; examples: the first 4 disagreeing patterns (the smallest bit patterns for the host comparisons), as bits: value of the first against the second; host half " +
                            (withHost ? "included" : "skipped (E6_HOST=0)"));
    bool ok = true;
    unsigned long long disagreeing = 0;
    ok = e6amd::sweepType<unsigned char>("uchar", withHost, csv, disagreeing) && ok;
    ok = e6amd::sweepType<signed char>("schar", withHost, csv, disagreeing) && ok;
    ok = e6amd::sweepType<short>("short", withHost, csv, disagreeing) && ok;
    ok = e6amd::sweepType<unsigned short>("ushort", withHost, csv, disagreeing) && ok;
    ok = e6amd::sweepType<int>("int", withHost, csv, disagreeing) && ok;
    if (!ok) {
        std::cout << "E6 FAILED: a HIP call failed" << std::endl;
        return -1;
    }
    // disagreeing is a sum over (pattern, comparison) pairs: a pattern that two comparisons get wrong counts twice
    const std::string scope = withHost ? "" : " (the host half was skipped: E6_HOST=0)";
    if (expectDisagreements) {
        if (disagreeing == 0) {
            std::cout << "E6 FAILED: E6_EXPECT_DISAGREEMENTS=1 (negative control) and nothing disagrees: the program did not detect what it should" << scope << std::endl;
            return 1;
        }
        std::cout << "E6 negative control: " << disagreeing << " disagreements (pattern x comparison), as expected" << scope << std::endl;
        return 0;
    }
    if (disagreeing != 0) {
        std::cout << "E6 FAILED: " << disagreeing << " disagreements (pattern x comparison; the CSV says which)" << scope << std::endl;
        return 1;
    }
    std::cout << "E6: 0 disagreements (pattern x comparison)" << scope << std::endl;
    return 0;
}
