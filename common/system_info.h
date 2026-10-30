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

// hipDriverGetVersion / hipRuntimeGetVersion return major * 10^7 + minor * 10^5 + patch
inline std::string hipVersionString(const int version) {
    return std::to_string(version / 10000000) + "." + std::to_string((version / 100000) % 100) + "." +
           std::to_string(version % 100000);
}

// Current and maximum clocks of the GPU with the given PCI bus id, from amd-smi (ROCm 7 and later have no rocm-smi):
// "gfx, mem, max gfx, max mem" in MHz, like the nvidia-smi line of the CUDA header
inline std::string amdSmiClocks(const std::string& busId) {
    return firstLineOf("(amd-smi metric -g " + busId + " --clock | awk '/^ +[A-Z]+_[0-9]+:$/ "
                       "{s = ($1 == \"GFX_0:\") ? 1 : (($1 == \"MEM_0:\") ? 2 : 0)} "
                       "$1 == \"CLK:\" {if (s == 1) gc = $2; if (s == 2) mc = $2} "
                       "$1 == \"MAX_CLK:\" {if (s == 1) gm = $2; if (s == 2) mm = $2} "
                       "END {if (gc == \"\") exit 1; print gc \", \" mc \", \" gm \", \" mm}')");
}

// notes: free text about the protocol of this run (what was warmed up, which fix applies...), one "# notes:" line
inline void writeSystemInfoHeader(std::ostream& os, const std::string& experiment, const std::string& notes = "") {
    os << "# experiment: " << experiment << "\n";
    if (!notes.empty()) {
        os << "# notes: " << notes << "\n";
    }
    os << "# repo: fkl-benchmarks " << FKLB_COMMIT_REPO << "\n";
    os << "# fkl: " << FKLB_COMMIT_FKL << "\n";
    os << "# cvGPUSpeedup: " << FKLB_COMMIT_CVGS << "\n";
    os << "# FastNPP: " << FKLB_COMMIT_FASTNPP << "\n";
    os << "# nvcc: " << FKLB_NVCC_VERSION << " (" << FKLB_NVCC_PATH << "), architectures " << FKLB_CUDA_ARCH
       << ", -fmad " << FKLB_FMAD << ", -lineinfo " << FKLB_LINEINFO << "\n";
    os << "# hip: " << FKLB_HIP_VERSION << " (" << FKLB_HIP_PATH << "), architectures " << FKLB_HIP_ARCH << "\n";
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
        char busId[32] = "";
        (void)hipDeviceGetPCIBusId(busId, sizeof(busId), device);
        // On gfx10 and later multiProcessorCount counts workgroup processors (WGPs), which have two CUs each
        const std::string arch = prop.gcnArchName;
        const bool wgps = arch.rfind("gfx10", 0) == 0 || arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0;
        os << "# gpu: " << prop.name << ", " << arch << ", " << prop.multiProcessorCount;
        if (wgps) {
            os << " WGPs (" << 2 * prop.multiProcessorCount << " CUs)";
        } else {
            os << " CUs";
        }
        os << ", wave" << prop.warpSize << ", " << (prop.totalGlobalMem >> 20) << " MiB, PCI " << busId << "\n";
        os << "# hip driver: " << hipVersionString(driver) << ", hip runtime: " << hipVersionString(runtime) << "\n";
        os << "# clocks (gfx, mem, max gfx, max mem, MHz): " << amdSmiClocks(busId) << "\n";
        os << "# amd-smi: " << firstLineOf("amd-smi version | head -1") << "\n";
    }
#else
    os << "# gpu: none (CPU back-end)\n";
#endif
}

}  // namespace fklb
