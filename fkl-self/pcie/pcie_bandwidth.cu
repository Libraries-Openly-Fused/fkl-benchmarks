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

// Link check: the bandwidth of host to device and device to host copies of E16_PCIE_MB (default 256) MiB, from pinned and from pageable host memory, each direction for at least PCIE_SECONDS seconds (default 3,
// pageable 1.5 s) so that a sampler of the link state (nvidia-smi pcie.link.gen.current, pcie.link.width.current, every 100 ms) sees the link busy. One CSV row per direction and memory kind: copies, seconds, GB/s (median, smallest,
// largest of the single copies, each timed with CUDA events). A PCIe 4.0 x16 link gives about 25 GB/s each way from pinned memory, PCIe 3.0 x1 about 0.9 GB/s.
#include <common/main.h>
#include <common/csv_writer.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define PCIE_CUDA(call)                                                                                                                                              \
    do {                                                                                                                                                             \
        const cudaError_t pcieErr = (call);                                                                                                                          \
        if (pcieErr != cudaSuccess) { throw std::runtime_error(std::string(#call) + " failed: " + cudaGetErrorString(pcieErr)); }                                       \
    } while (0)

namespace {
double envDouble(const char* name, const double fallback) {
    const char* v = std::getenv(name);
    return v ? std::atof(v) : fallback;
}

struct Result {
    int copies{0};
    double seconds{0}, median{0}, min{0}, max{0};
};

Result run(void* dst, const void* src, const size_t bytes, const cudaMemcpyKind kind, const double seconds) {
    cudaEvent_t a, b;
    PCIE_CUDA(cudaEventCreate(&a));
    PCIE_CUDA(cudaEventCreate(&b));
    std::vector<double> gbps;
    const auto start = std::chrono::steady_clock::now();
    do {
        PCIE_CUDA(cudaEventRecord(a));
        PCIE_CUDA(cudaMemcpyAsync(dst, src, bytes, kind));
        PCIE_CUDA(cudaEventRecord(b));
        PCIE_CUDA(cudaEventSynchronize(b));
        float ms = 0.f;
        PCIE_CUDA(cudaEventElapsedTime(&ms, a, b));
        gbps.push_back(static_cast<double>(bytes) / (ms * 1e-3) / 1e9);
    } while (std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count() < seconds);
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(gbps.begin(), gbps.end());
    return {static_cast<int>(gbps.size()), elapsed, gbps[gbps.size() / 2], gbps.front(), gbps.back()};
}
}  // namespace

int launch() {
    try {
        const size_t bytes = static_cast<size_t>(envDouble("E16_PCIE_MB", 256.0)) * 1024 * 1024;
        const double seconds = envDouble("PCIE_SECONDS", 3.0);
        void *dev = nullptr, *pinned = nullptr;
        PCIE_CUDA(cudaMalloc(&dev, bytes));
        PCIE_CUDA(cudaMallocHost(&pinned, bytes));
        std::vector<char> pageableStorage(bytes, 1);
        void* pageable = pageableStorage.data();
        PCIE_CUDA(cudaMemset(dev, 0, bytes));
        std::fill_n(static_cast<char*>(pinned), bytes, 1);
        fklb::CsvWriter csv("pcie_bandwidth.csv", "link check: host to device and device to host copy bandwidth", {"direction", "memory", "bytes", "copies", "seconds", "gbps_median", "gbps_min", "gbps_max"},
                            "copies of the same buffer, each timed with CUDA events around the copy; the link state is sampled by nvidia-smi at the same time (see the sampler file of the window)");
        struct Case { const char* direction; const char* memory; void* dst; const void* src; cudaMemcpyKind kind; double secs; };
        const Case cases[] = {{"H2D", "pinned", dev, pinned, cudaMemcpyHostToDevice, seconds}, {"D2H", "pinned", pinned, dev, cudaMemcpyDeviceToHost, seconds},
                              {"H2D", "pageable", dev, pageable, cudaMemcpyHostToDevice, seconds / 2}, {"D2H", "pageable", pageable, dev, cudaMemcpyDeviceToHost, seconds / 2}};
        for (const Case& c : cases) {
            run(c.dst, c.src, bytes, c.kind, 0.2);   // warm-up
            const Result r = run(c.dst, c.src, bytes, c.kind, c.secs);
            csv.row(c.direction, c.memory, bytes, r.copies, r.seconds, r.median, r.min, r.max);
            csv.flush();
            std::cout << c.direction << " " << c.memory << ": " << r.copies << " copies of " << bytes / (1024 * 1024) << " MiB in " << r.seconds << " s, median " << r.median << " GB/s (smallest " << r.min << ", largest " << r.max << ")" << std::endl;
        }
        cudaFreeHost(pinned);
        cudaFree(dev);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "pcie_bandwidth: " << e.what() << std::endl;
        return -1;
    }
}
