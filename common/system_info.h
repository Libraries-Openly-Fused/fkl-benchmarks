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

// The header of every CSV of this repository: comment lines starting with "# " with what was built (submodule commits,
// nvcc, flags) and where it ran (GPU, driver, runtime, clocks). Read the CSVs with comment='#' (pandas).

#pragma once

#include <fklb_build_info.h>

#include <cstdio>
#include <fstream>
#include <ostream>
#include <sstream>
#include <string>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#elif defined(__HIPCC__)
#include <hip/hip_runtime.h>
#endif

namespace fklb {

// Output of a shell command, first line only, "unavailable" if it fails.
inline std::string firstLineOf(const std::string& command) {
    std::string out;
    if (FILE* pipe = popen((command + " 2>/dev/null").c_str(), "r")) {
        char buffer[512];
        if (fgets(buffer, sizeof(buffer), pipe)) {
            out = buffer;
        }
        pclose(pipe);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r')) out.pop_back();
    return out.empty() ? "unavailable" : out;
}

inline std::string cpuModel() {
    return firstLineOf("grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//'");
}

inline void writeSystemInfoHeader(std::ostream& os, const std::string& experiment) {
    os << "# experiment: " << experiment << "\n";
    os << "# repo: fkl-benchmarks " << FKLB_COMMIT_REPO << "\n";
    os << "# fkl: " << FKLB_COMMIT_FKL << "\n";
    os << "# cvGPUSpeedup: " << FKLB_COMMIT_CVGS << "\n";
    os << "# FastNPP: " << FKLB_COMMIT_FASTNPP << "\n";
    os << "# nvcc: " << FKLB_NVCC_VERSION << " (" << FKLB_NVCC_PATH << "), architectures " << FKLB_CUDA_ARCH
       << ", -fmad " << FKLB_FMAD << ", -lineinfo " << FKLB_LINEINFO << "\n";
    os << "# host compiler: " << FKLB_CXX_COMPILER << ", " << FKLB_BUILD_TYPE << ", -march=" << FKLB_CPU_ARCH
       << ", -ftemplate-depth=" << FKLB_TEMPLATE_DEPTH << "\n";
    os << "# opencv: " << FKLB_OPENCV_VERSION << "\n";
    os << "# cpu: " << cpuModel() << "\n";
#if defined(__CUDACC__)
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        int driver = 0, runtime = 0;
        cudaDriverGetVersion(&driver);
        cudaRuntimeGetVersion(&runtime);
        char busId[32] = "";
        cudaDeviceGetPCIBusId(busId, sizeof(busId), device);
        os << "# gpu: " << prop.name << ", cc " << prop.major << "." << prop.minor << ", " << prop.multiProcessorCount
           << " SMs, " << (prop.totalGlobalMem >> 20) << " MiB, PCI " << busId << "\n";
        os << "# cuda driver api: " << driver / 1000 << "." << (driver % 1000) / 10 << ", cuda runtime: " << runtime / 1000
           << "." << (runtime % 1000) / 10 << "\n";
        const std::string smi = "nvidia-smi -i " + std::string(busId) + " --format=csv,noheader --query-gpu=";
        os << "# nvidia-smi driver: " << firstLineOf(smi + "driver_version") << "\n";
        os << "# clocks (sm, mem, max sm, max mem, MHz): "
           << firstLineOf(smi + "clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem") << "\n";
        os << "# power limit, temperature at start: " << firstLineOf(smi + "power.limit,temperature.gpu") << "\n";
    }
#elif defined(__HIPCC__)
    int device = 0;
    (void)hipGetDevice(&device);
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, device) == hipSuccess) {
        int driver = 0, runtime = 0;
        (void)hipDriverGetVersion(&driver);
        (void)hipRuntimeGetVersion(&runtime);
        os << "# gpu: " << prop.name << ", " << prop.gcnArchName << ", " << prop.multiProcessorCount << " CUs, "
           << (prop.totalGlobalMem >> 20) << " MiB\n";
        os << "# hip driver: " << driver << ", hip runtime: " << runtime << "\n";
        os << "# clocks: " << firstLineOf("rocm-smi --showclocks | head -20 | tr '\\n' ' '") << "\n";
    }
#else
    os << "# gpu: none (CPU back-end)\n";
#endif
}

}  // namespace fklb
