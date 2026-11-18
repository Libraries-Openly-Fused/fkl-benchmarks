// Diagnostic (not a benchmark): does a 2 s single-thread gpuSpin (wall_clock64) bring the RX 9060's sclk from idle to its full clock?
// sclk is read from hwmon (fklb::amdGpuClockMHz) every 100 ms during the spin and for 0.5 s after it. Nothing is timed.
#include <common/hip_wallclock.h>
#include <hip/hip_runtime.h>
#include <chrono>
#include <cstdio>
#include <thread>
__global__ void gpuSpin(const unsigned long long ticks) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {
    }
}
int main() {
    int device = 0, khz = 0;
    hipGetDevice(&device);
    hipDeviceGetAttribute(&khz, hipDeviceAttributeWallClockRate, device);
    std::printf("before: sclk %.0f MHz\n", fklb::amdGpuClockMHz());
    const auto t0 = std::chrono::steady_clock::now();
    gpuSpin<<<1, 1>>>(2000ull * static_cast<unsigned long long>(khz));
    for (int i = 1; i <= 25; ++i) {
        std::this_thread::sleep_until(t0 + std::chrono::milliseconds(100 * i));
        std::printf("t=%4d ms sclk %.0f MHz%s\n", 100 * i, fklb::amdGpuClockMHz(), hipStreamQuery(nullptr) == hipSuccess ? " (spin done)" : "");
    }
    hipDeviceSynchronize();
    std::printf("hip error: %s\n", hipGetErrorString(hipGetLastError()));
}
