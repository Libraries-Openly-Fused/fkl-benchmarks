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

// E6 (numerics): what does fk::SaturateCast give for NaN on the device? Four NaN bit patterns (quiet, signalling, and the negative of
// each) of float and of double are converted to uchar, schar, short, ushort and int in a kernel with FKL's SaturateCast (the code of
// cxp::saturate_cast), and for reference with the CUDA conversion that does the same job (__float2int_rn / __double2int_rn followed by
// the clamp, and the saturating PTX cvt for the 8 and 16 bit types). The values are printed and written to nan_saturate.csv.
// Written to define the host behaviour of cxp::saturate_cast from the device's.
#include <common/main.h>
#include <common/csv_writer.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {
struct Pattern { const char* name; uint64_t bits64; uint32_t bits32; };
constexpr Pattern PATTERNS[] = {
    {"+qNaN", 0x7ff8000000000000ull, 0x7fc00000u},
    {"+sNaN", 0x7ff4000000000000ull, 0x7fa00000u},
    {"-qNaN", 0xfff8000000000000ull, 0xffc00000u},
    {"-sNaN", 0xfff4000000000000ull, 0xffa00000u},
};
constexpr int NP = 4;
constexpr int NT = 5;   // uchar, schar, short, ushort, int
const char* TYPES[NT] = {"uchar", "schar", "short", "ushort", "int"};

template <typename F>
__device__ void fklRow(const F v, long long* out) {
    out[0] = fk::SaturateCast<F, unsigned char>::exec(v);
    out[1] = fk::SaturateCast<F, signed char>::exec(v);
    out[2] = fk::SaturateCast<F, short>::exec(v);
    out[3] = fk::SaturateCast<F, unsigned short>::exec(v);
    out[4] = fk::SaturateCast<F, int>::exec(v);
}

// What the hardware conversions give: cvt.rni with saturation, the reference of "what the GPU does" (not FKL code)
__device__ void hwRowF(const float v, long long* out) {
    unsigned int u8; int s8, s16; unsigned int u16; int s32;
    asm("cvt.rni.sat.u8.f32 %0, %1;" : "=r"(u8) : "f"(v));
    asm("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(s8) : "f"(v));
    asm("cvt.rni.sat.s16.f32 %0, %1;" : "=r"(s16) : "f"(v));
    asm("cvt.rni.sat.u16.f32 %0, %1;" : "=r"(u16) : "f"(v));
    asm("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(s32) : "f"(v));
    out[0] = u8; out[1] = static_cast<signed char>(s8); out[2] = static_cast<short>(s16); out[3] = u16; out[4] = s32;
}

__global__ void kernel(const uint32_t* bits32, const uint64_t* bits64, long long* fklFloat, long long* fklDouble, long long* hwFloat) {
    const int i = threadIdx.x;
    if (i >= NP) return;
    fklRow<float>(__uint_as_float(bits32[i]), fklFloat + i * NT);
    fklRow<double>(__longlong_as_double(static_cast<long long>(bits64[i])), fklDouble + i * NT);
    hwRowF(__uint_as_float(bits32[i]), hwFloat + i * NT);
}
}  // namespace

int launch() {
    uint32_t h32[NP]; uint64_t h64[NP];
    for (int i = 0; i < NP; ++i) { h32[i] = PATTERNS[i].bits32; h64[i] = PATTERNS[i].bits64; }
    uint32_t* d32; uint64_t* d64; long long *dFF, *dFD, *dHF;
    cudaMalloc(&d32, sizeof(h32)); cudaMalloc(&d64, sizeof(h64));
    cudaMalloc(&dFF, NP * NT * sizeof(long long)); cudaMalloc(&dFD, NP * NT * sizeof(long long)); cudaMalloc(&dHF, NP * NT * sizeof(long long));
    cudaMemcpy(d32, h32, sizeof(h32), cudaMemcpyHostToDevice); cudaMemcpy(d64, h64, sizeof(h64), cudaMemcpyHostToDevice);
    kernel<<<1, 32>>>(d32, d64, dFF, dFD, dHF);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::cerr << "kernel failed: " << cudaGetErrorString(cudaGetLastError()) << std::endl; return -1; }
    std::vector<long long> fF(NP * NT), fD(NP * NT), hF(NP * NT);
    cudaMemcpy(fF.data(), dFF, fF.size() * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(fD.data(), dFD, fD.size() * 8, cudaMemcpyDeviceToHost);
    cudaMemcpy(hF.data(), dHF, hF.size() * 8, cudaMemcpyDeviceToHost);

    fklb::CsvWriter csv("nan_saturate.csv", "E6 NaN through fk::SaturateCast on the device",
                        {"input", "bits", "source_type", "target_type", "fkl_saturate_cast", "hardware_cvt_rni_sat"},
                        "quiet NaN 0x7fc00000 / 0x7ff8..., signalling NaN 0x7fa00000 / 0x7ff4... (the signalling bit patterns are loaded as bits, not computed); "
                        "hardware_cvt_rni_sat is PTX cvt.rni.sat for float only; values of the 8-bit types are printed as signed or unsigned values of the type");
    std::cout << "fk::SaturateCast on the device (hardware cvt.rni.sat for float in brackets)\n";
    char buf[64];
    for (int p = 0; p < NP; ++p) {
        for (int pass = 0; pass < 2; ++pass) {
            const bool isFloat = pass == 0;
            std::snprintf(buf, sizeof(buf), "0x%016llx", isFloat ? (unsigned long long)PATTERNS[p].bits32 : (unsigned long long)PATTERNS[p].bits64);
            std::cout << "  " << PATTERNS[p].name << (isFloat ? " float  " : " double ");
            for (int t = 0; t < NT; ++t) {
                const long long v = (isFloat ? fF : fD)[p * NT + t];
                std::cout << TYPES[t] << "=" << v;
                if (isFloat) std::cout << "(" << hF[p * NT + t] << ")";
                std::cout << "  ";
                csv.row(PATTERNS[p].name, buf, isFloat ? "float" : "double", TYPES[t], v, isFloat ? std::to_string(hF[p * NT + t]) : std::string("n/a"));
            }
            std::cout << "\n";
        }
    }
    return 0;
}
