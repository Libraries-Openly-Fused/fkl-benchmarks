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

// Prints "key = value" for the device attributes that matter for fusion and for the architecture-features paragraph of the
// paper (compute capability, SM count, shared memory, registers, cluster launch...). Built by scripts/system_info.sh for the
// architectures of this repository and run once per GPU (CUDA_VISIBLE_DEVICES selects it).
#include <cuda_runtime.h>

#include <cstdio>

#define ATTR(name, attr)                                                  \
    {                                                                     \
        int value = -1;                                                   \
        const cudaError_t err = cudaDeviceGetAttribute(&value, attr, dev); \
        if (err == cudaSuccess) std::printf("%s = %d\n", name, value);    \
        else std::printf("%s = unavailable (%s)\n", name, cudaGetErrorString(err)); \
    }

int main() {
    int dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, dev);
    int driver = 0, runtime = 0;
    cudaDriverGetVersion(&driver);
    cudaRuntimeGetVersion(&runtime);
    std::printf("name = %s\n", p.name);
    std::printf("uuid/pci = %04x:%02x:%02x\n", p.pciDomainID, p.pciBusID, p.pciDeviceID);
    std::printf("cuda driver api version = %d.%d\n", driver / 1000, (driver % 1000) / 10);
    std::printf("cuda runtime version = %d.%d\n", runtime / 1000, (runtime % 1000) / 10);
    std::printf("global memory MiB = %zu\n", p.totalGlobalMem >> 20);
    ATTR("compute capability major", cudaDevAttrComputeCapabilityMajor)
    ATTR("compute capability minor", cudaDevAttrComputeCapabilityMinor)
    ATTR("multiprocessor count", cudaDevAttrMultiProcessorCount)
    ATTR("warp size", cudaDevAttrWarpSize)
    ATTR("max threads per multiprocessor", cudaDevAttrMaxThreadsPerMultiProcessor)
    ATTR("max threads per block", cudaDevAttrMaxThreadsPerBlock)
    ATTR("max blocks per multiprocessor", cudaDevAttrMaxBlocksPerMultiprocessor)
    ATTR("registers per block", cudaDevAttrMaxRegistersPerBlock)
    ATTR("registers per multiprocessor", cudaDevAttrMaxRegistersPerMultiprocessor)
    ATTR("shared memory per block (default) bytes", cudaDevAttrMaxSharedMemoryPerBlock)
    ATTR("shared memory per block (opt-in) bytes", cudaDevAttrMaxSharedMemoryPerBlockOptin)
    ATTR("shared memory per multiprocessor bytes", cudaDevAttrMaxSharedMemoryPerMultiprocessor)
    ATTR("reserved shared memory per block bytes", cudaDevAttrReservedSharedMemoryPerBlock)
    ATTR("L2 cache bytes", cudaDevAttrL2CacheSize)
    ATTR("max persisting L2 cache bytes", cudaDevAttrMaxPersistingL2CacheSize)
    ATTR("global memory bus width bits", cudaDevAttrGlobalMemoryBusWidth)
    ATTR("cluster launch (thread block clusters)", cudaDevAttrClusterLaunch)
    ATTR("cooperative launch", cudaDevAttrCooperativeLaunch)
    ATTR("async engine count", cudaDevAttrAsyncEngineCount)
    ATTR("concurrent kernels", cudaDevAttrConcurrentKernels)
    ATTR("memory pools supported", cudaDevAttrMemoryPoolsSupported)
    ATTR("ECC enabled", cudaDevAttrEccEnabled)
    ATTR("integrated GPU", cudaDevAttrIntegrated)
    ATTR("single to double precision perf ratio", cudaDevAttrSingleToDoublePrecisionPerfRatio)
    return 0;
}
