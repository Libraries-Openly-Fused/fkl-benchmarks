/* Task C-MAIN-NV: the before/after output harness of FKL PR #436 (task C on FKL main; CUDA, RTX 4090), adapted from task C-NV's
   harness of FKL PR #435 (scratch/pr435/harness/pr435_*). Common helpers of pr436_image.cu and pr436_attention.cu. Each program is
   compiled twice, once against FKL main ebb3891 (the base) and once against #436's head 1df2738 (with -DPR436_HEAD); only the calls
   into FKL differ between the two builds (#ifdef PR436_HEAD). Every case runs once on fixed inputs (an integer hash of the coordinates: no random generator) and
   writes the bytes of its output to <out dir>/<case>.bin; compare_harness.py compares the two folders. Every output buffer
   is filled with 0xff bytes before the call, so an element the call does not write keeps that pattern. Nothing is timed. */
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace h436 {

inline std::string g_out;
inline int g_files = 0;

#ifdef PR436_HEAD
constexpr const char* SIDE = "head (FKL #436 1df2738 API: executeOperations<DPP>)";
#else
constexpr const char* SIDE = "base (FKL main ebb3891 API: the per-DPP entry functions)";
#endif

inline uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
inline uint32_t hashOf(const int x, const int y, const int salt) {
    return hash32(static_cast<uint32_t>(x) * 73856093u ^ hash32(static_cast<uint32_t>(y) * 19349663u ^
                  hash32(static_cast<uint32_t>(salt) * 83492791u + 0x9e3779b9u)));
}
// a float in [lo, hi) with 24 random bits
inline float fval(const int x, const int y, const int salt, const float lo = -4.f, const float hi = 4.f) {
    return lo + static_cast<float>(hashOf(x, y, salt) >> 8) * ((hi - lo) / 16777216.f);
}
inline unsigned char uval(const int x, const int y, const int salt) {
    return static_cast<unsigned char>(hashOf(x, y, salt) >> 24);
}

inline void fail(const std::string& what) {
    std::fprintf(stderr, "HARNESS FAILURE: %s\n", what.c_str());
    std::fflush(stderr);
    std::exit(2);
}
inline void require(const bool ok, const std::string& what) {
    if (!ok) fail(what);
}
inline void cudaCheck(const cudaError_t e, const char* what) {
    if (e != cudaSuccess) fail(std::string(what) + ": " + cudaGetErrorString(e));
}

inline void writeBin(const std::string& name, const void* data, const size_t bytes) {
    const std::string path = g_out + "/" + name + ".bin";
    FILE* f = std::fopen(path.c_str(), "wbx");  // x: never overwrite
    if (f == nullptr) fail("cannot create " + path);
    const size_t n = std::fwrite(data, 1, bytes, f);
    if (std::fclose(f) != 0 || n != bytes) fail("cannot write " + path);
    std::printf("OUT %-48s %10zu bytes\n", name.c_str(), bytes);
    ++g_files;
}

// device buffer of n elements filled with 0xff bytes
template <typename T>
T* deviceSentinel(const size_t n) {
    T* p = nullptr;
    cudaCheck(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    cudaCheck(cudaMemset(p, 0xff, n * sizeof(T)), "cudaMemset");
    return p;
}
template <typename T>
T* deviceCopy(const std::vector<T>& h) {
    T* p = nullptr;
    cudaCheck(cudaMalloc(&p, h.size() * sizeof(T)), "cudaMalloc");
    cudaCheck(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return p;
}
template <typename T>
void dumpDevice(const std::string& name, const T* p, const size_t n) {
    std::vector<T> h(n);
    cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    cudaCheck(cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    writeBin(name, h.data(), n * sizeof(T));
}

inline int start(const int argc, char** argv, const char* program) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <existing empty output folder>\n", program);
        std::exit(1);
    }
    g_out = argv[1];
    int dev = 0;
    cudaDeviceProp prop{};
    cudaCheck(cudaGetDevice(&dev), "cudaGetDevice");
    cudaCheck(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");
    std::printf("%s, %s: device %d %s (sm_%d%d, %d SMs), output folder %s\n", program, SIDE, dev, prop.name,
                prop.major, prop.minor, prop.multiProcessorCount, g_out.c_str());
    return 0;
}
inline int finish(const char* program) {
    cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize at the end");
    cudaCheck(cudaGetLastError(), "cudaGetLastError at the end");
    std::printf("%s: %d output files written; PASS (every call ran without a CUDA error or a rejection)\n", program,
                g_files);
    return 0;
}

} // namespace h436
