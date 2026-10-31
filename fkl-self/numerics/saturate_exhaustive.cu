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

// E6 (numerics): fk::SaturateCast<float, T> over ALL 2^32 float bit patterns, for T in uchar, schar, short, ushort, int, against
//   opencv_dev   cv::cuda::device::saturate_cast<T>(float)   (opencv2/core/cuda/saturate_cast.hpp, what cv::cuda::* kernels use)
//   hw           the PTX conversion cvt.rni.sat (round to nearest even with saturation: what the hardware does; NaN gives 0)
// on the device, and on the host
//   fkl_host     fk::SaturateCast<float, T>::exec on the host (cxp::saturate_cast, constexpr)
//   opencv_host  cv::saturate_cast<T>(float) (cvRound: lrintf)
// Six comparisons, each over all 2^32 patterns: fkl_dev vs opencv_dev, fkl_dev vs hw, opencv_dev vs hw, fkl_host vs fkl_dev, opencv_host vs opencv_dev, fkl_host vs
// opencv_host. The disagreeing set is reported per class of float (NaN, +Inf, -Inf, denormal, zero, normal) with the first examples (bits, both values).
// The patterns are swept in chunks of 2^26: the device computes fkl_dev and opencv_dev for the chunk (and counts its three comparisons), the host gets them and
// evaluates the host functions for the same patterns. Output: e6_saturate_exhaustive.csv, one row per (type, comparison, class).
//   E6_HOST=0 skips the host half (the device comparisons take seconds, the host half minutes on one core).
#include <common/main.h>
#include <common/csv_writer.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>

#include <opencv2/core/cuda/saturate_cast.hpp>
#include <opencv2/core/saturate.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace {
constexpr int NCLASS = 6, NCMP = 6, NEX = 4;
const char* CLASS_NAMES[NCLASS] = {"NaN", "+Inf", "-Inf", "denormal", "zero", "normal"};
const char* CMP_NAMES[NCMP] = {"fkl_dev_vs_opencv_dev", "fkl_dev_vs_hw", "opencv_dev_vs_hw", "fkl_host_vs_fkl_dev", "opencv_host_vs_opencv_dev", "fkl_host_vs_opencv_host"};
constexpr uint64_t CLASS_COUNT[NCLASS] = {2ull * ((1ull << 23) - 1), 1, 1, 2ull * ((1ull << 23) - 1), 2, 2ull * 254 * (1ull << 23)};
constexpr uint64_t CHUNK = 1ull << 26;

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

template <typename T> __device__ inline long long hwConvert(const float v);
template <> __device__ inline long long hwConvert<unsigned char>(const float v) { unsigned int r; asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(r) : "f"(v)); return r; }
template <> __device__ inline long long hwConvert<signed char>(const float v) { int r; asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(r) : "f"(v)); return static_cast<signed char>(r); }
template <> __device__ inline long long hwConvert<short>(const float v) { int r; asm("cvt.rni.sat.s16.f32 %0, %1;" : "=r"(r) : "f"(v)); return static_cast<short>(r); }
template <> __device__ inline long long hwConvert<unsigned short>(const float v) { unsigned int r; asm("cvt.rni.sat.u16.f32 %0, %1;" : "=r"(r) : "f"(v)); return r; }
template <> __device__ inline long long hwConvert<int>(const float v) { int r; asm("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(r) : "f"(v)); return r; }

__device__ inline void record(Counters* c, const int cmp, const int cls, const uint32_t bits, const long long a, const long long b) {
    atomicAdd(&c->mismatch[cmp][cls], 1ull);
    const unsigned int k = atomicAdd(&c->nex[cmp][cls], 1u);
    if (k < NEX) c->ex[cmp][cls][k] = Example{bits, a, b};
}

template <typename T>
__global__ void sweepDevice(const uint64_t first, const uint64_t count, int* fklOut, int* ocvOut, Counters* counters) {
    const uint64_t i = static_cast<uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t bits = static_cast<uint32_t>(first + i);
    const float v = __uint_as_float(bits);
    const long long f = fk::SaturateCast<float, T>::exec(v);
    const long long o = cv::cuda::device::saturate_cast<T>(v);
    const long long h = hwConvert<T>(v);
    fklOut[i] = static_cast<int>(f);
    ocvOut[i] = static_cast<int>(o);
    const int cls = classOf(bits);
    if (f != o) record(counters, 0, cls, bits, f, o);
    if (f != h) record(counters, 1, cls, bits, f, h);
    if (o != h) record(counters, 2, cls, bits, o, h);
}

inline void hostRecord(Counters& c, const int cmp, const int cls, const uint32_t bits, const long long a, const long long b) {
    ++c.mismatch[cmp][cls];
    if (c.nex[cmp][cls] < NEX) c.ex[cmp][cls][c.nex[cmp][cls]] = Example{bits, a, b};
    ++c.nex[cmp][cls];
}

template <typename T>
bool sweepType(const char* typeName, const bool withHost, fklb::CsvWriter& csv) {
    int *dF = nullptr, *dO = nullptr;
    Counters* dCounters = nullptr;
    if (cudaMalloc(&dF, CHUNK * sizeof(int)) != cudaSuccess || cudaMalloc(&dO, CHUNK * sizeof(int)) != cudaSuccess || cudaMalloc(&dCounters, sizeof(Counters)) != cudaSuccess) return false;
    cudaMemset(dCounters, 0, sizeof(Counters));
    Counters host{};
    std::vector<int> hF(CHUNK), hO(CHUNK);
    for (uint64_t first = 0; first < (1ull << 32); first += CHUNK) {
        sweepDevice<T><<<static_cast<unsigned int>((CHUNK + 255) / 256), 256>>>(first, CHUNK, dF, dO, dCounters);
        if (cudaGetLastError() != cudaSuccess) return false;
        if (withHost) {
            if (cudaMemcpy(hF.data(), dF, CHUNK * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess || cudaMemcpy(hO.data(), dO, CHUNK * sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
            for (uint64_t i = 0; i < CHUNK; ++i) {
                const uint32_t bits = static_cast<uint32_t>(first + i);
                float v;
                std::memcpy(&v, &bits, sizeof(v));
                const long long fh = fk::SaturateCast<float, T>::exec(v);
                const long long oh = cv::saturate_cast<T>(v);
                const int cls = classOf(bits);
                if (fh != hF[i]) hostRecord(host, 3, cls, bits, fh, hF[i]);
                if (oh != hO[i]) hostRecord(host, 4, cls, bits, oh, hO[i]);
                if (fh != oh) hostRecord(host, 5, cls, bits, fh, oh);
            }
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) return false;
    Counters dev{};
    if (cudaMemcpy(&dev, dCounters, sizeof(Counters), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    for (int cmp = 0; cmp < NCMP; ++cmp) {
        const bool isHost = cmp >= 3;
        if (isHost && !withHost) continue;
        const Counters& src = isHost ? host : dev;
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
    unsigned long long total[NCMP] = {};
    for (int cmp = 0; cmp < NCMP; ++cmp)
        for (int cls = 0; cls < NCLASS; ++cls) total[cmp] += (cmp >= 3 ? host : dev).mismatch[cmp][cls];
    std::cout << typeName << ": disagreeing patterns of 2^32:";
    for (int cmp = 0; cmp < NCMP; ++cmp) {
        if (cmp >= 3 && !withHost) continue;
        std::cout << "  " << CMP_NAMES[cmp] << " " << total[cmp];
    }
    std::cout << std::endl;
    cudaFree(dF);
    cudaFree(dO);
    cudaFree(dCounters);
    return true;
}
}  // namespace

int launch() {
    const bool withHost = std::getenv("E6_HOST") == nullptr || std::getenv("E6_HOST")[0] != '0';
    fklb::CsvWriter csv("e6_saturate_exhaustive.csv", "E6 fk::SaturateCast<float, T> over all 2^32 float bit patterns",
                        {"type", "comparison", "class", "patterns_in_class", "disagreeing", "examples"},
                        std::string("comparisons: fkl_dev = fk::SaturateCast on the device; opencv_dev = cv::cuda::device::saturate_cast; hw = PTX cvt.rni.sat (round to nearest even, saturating); ") +
                            "fkl_host = fk::SaturateCast on the host (cxp::saturate_cast); opencv_host = cv::saturate_cast (cvRound); classes by IEEE-754 binary32 bit pattern; " +
                            "examples: the first 4 disagreeing patterns, as bits: value of the first against the second; host half " + (withHost ? "included" : "skipped (E6_HOST=0)"));
    bool ok = true;
    ok = sweepType<unsigned char>("uchar", withHost, csv) && ok;
    ok = sweepType<signed char>("schar", withHost, csv) && ok;
    ok = sweepType<short>("short", withHost, csv) && ok;
    ok = sweepType<unsigned short>("ushort", withHost, csv) && ok;
    ok = sweepType<int>("int", withHost, csv) && ok;
    return ok ? 0 : -1;
}
