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

// Theoretical occupancy of kernels of a cubin, from cudaOccupancyMaxActiveBlocksPerMultiprocessor.
//
//   occupancy_probe <file.cubin>      reads lines "<mangled kernel name> <threads per block> <dynamic shared memory bytes>" from stdin
//
// and prints, tab separated: name, max active blocks per SM, warps per block, max warps per SM, theoretical occupancy in percent.
// The cubin is the one cuobjdump extracts from the benchmark binary (scripts/e1_table.py does it), loaded with the runtime library
// API, so the numbers are those of the code that runs, for the launch configuration the profiler saw (threads per block as given).
#include <cuda_runtime.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(call)                                                                                                         \
    {                                                                                                                       \
        const cudaError_t e_ = (call);                                                                                      \
        if (e_ != cudaSuccess) {                                                                                            \
            std::fprintf(stderr, "%s failed: %s (line %d)\n", #call, cudaGetErrorString(e_), __LINE__);                      \
            return 1;                                                                                                       \
        }                                                                                                                   \
    }

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s file.cubin < lines of 'kernel threads dynamic_smem'\n", argv[0]);
        return 2;
    }
    std::ifstream in(argv[1], std::ios::binary);
    const std::vector<char> data((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (data.empty()) {
        std::fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    int device = 0, maxThreadsPerSM = 0;
    CHECK(cudaGetDevice(&device));
    CHECK(cudaDeviceGetAttribute(&maxThreadsPerSM, cudaDevAttrMaxThreadsPerMultiProcessor, device));
    cudaLibrary_t lib;
    CHECK(cudaLibraryLoadData(&lib, data.data(), nullptr, nullptr, 0, nullptr, nullptr, 0));

    std::string line;
    while (std::getline(std::cin, line)) {
        std::istringstream ls(line);
        std::string name;
        int threads = 0;
        size_t dynamicSmem = 0;
        if (!(ls >> name >> threads >> dynamicSmem)) continue;
        cudaKernel_t kernel;
        const cudaError_t found = cudaLibraryGetKernel(&kernel, lib, name.c_str());
        if (found != cudaSuccess) {
            std::fprintf(stderr, "%s: %s\n", name.c_str(), cudaGetErrorString(found));
            cudaGetLastError();
            continue;
        }
        int blocks = 0;
        CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, reinterpret_cast<const void*>(kernel), threads, dynamicSmem));
        const int warpsPerBlock = (threads + 31) / 32, maxWarps = maxThreadsPerSM / 32;
        std::printf("%s\t%d\t%d\t%d\t%.4f\n", name.c_str(), blocks, warpsPerBlock, maxWarps, 100.0 * blocks * warpsPerBlock / maxWarps);
    }
    return 0;
}
