// hostsplit: where the time of FKL's fused chain of pipeline (a) goes at a given pin of FKL
//   host   = time until fk::executeOperations returns (the launch, no synchronisation)
//   total  = launch + stream synchronisation (what the benchmark times, wall clock)
//   device = HIP events around the launch
// batches 100, 200, 300; the ASYMMETRIC (default) and the HALF_PIXEL resize
#define launch rppbench_original_launch
#include <rpp/benchmark_rpp_preprocessing.h>
#undef launch

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <mutex>
#include <thread>
#include <type_traits>

using namespace rppbench;
using Clock = std::chrono::steady_clock;


// the graphics clock of the GPU (hwmon freq1_input, Hz), sampled by a thread at about 200 Hz with the time of the sample
struct ClockSampler {
    std::string path;
    std::atomic<bool> stop{false};
    std::mutex m;
    std::vector<std::pair<double, double>> samples; // (seconds since the sampler started, MHz)
    std::thread t;
    Clock::time_point t0 = Clock::now();
    ClockSampler() {
        for (int i = 0; i < 16; ++i) {
            const std::string c = "/sys/class/drm/card1/device/hwmon/hwmon" + std::to_string(i) + "/freq1_input";
            std::ifstream f(c);
            if (f.good()) { path = c; break; }
        }
        t = std::thread([this] {
            while (!stop.load()) {
                std::ifstream f(path);
                double hz = 0;
                f >> hz;
                {
                    std::lock_guard<std::mutex> g(m);
                    samples.emplace_back(std::chrono::duration<double>(Clock::now() - t0).count(), hz / 1e6);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        });
    }
    ~ClockSampler() { stop = true; t.join(); }
    double now() const { return std::chrono::duration<double>(Clock::now() - t0).count(); }
    void summary(const double a, const double b, double& mean, double& lo, double& hi) {
        std::lock_guard<std::mutex> g(m);
        double sum = 0; int n = 0; lo = 1e9; hi = 0;
        for (const auto& [ts, mhz] : samples) if (ts >= a && ts <= b) { sum += mhz; ++n; lo = std::min(lo, mhz); hi = std::max(hi, mhz); }
        mean = n ? sum / n : 0; if (!n) { lo = hi = 0; }
    }
};
static ClockSampler* g_clock = nullptr;

static double med(std::vector<double>& v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

template <int N, typename ResizeT>
void measure(const char* label, fk::Stream_<fk::ParArch::GPU_AMD>& stream, FklA<N, ResizeT>& fkl, const double warmupSeconds = 0.5, const bool trace = false) {
    const hipStream_t hs = stream.getHIPStream();
    hipEvent_t e0, e1;
    RPPB_CHECK_HIP(hipEventCreate(&e0));
    RPPB_CHECK_HIP(hipEventCreate(&e1));
    const auto t0 = Clock::now();
    std::vector<std::pair<double, double>> warm; // (seconds since the start of the warm-up, ms of one iteration)
    while (std::chrono::duration<double>(Clock::now() - t0).count() < warmupSeconds) {
        const auto a = Clock::now();
        fkl.fused();
        fkl.sync();
        warm.emplace_back(std::chrono::duration<double>(a - t0).count(), std::chrono::duration<double, std::milli>(Clock::now() - a).count());
    }
    const double c0 = g_clock->now();
    if (trace) {
        const double w0 = c0 - warmupSeconds;
        std::printf("   warm-up trace (every 0.25 s: mean ms per iteration | mean sclk MHz):");
        for (double a = 0; a < warmupSeconds - 1e-9; a += 0.25) {
            double sum = 0; int n = 0;
            for (const auto& [ts, ms] : warm) if (ts >= a && ts < a + 0.25) { sum += ms; ++n; }
            double cm, cl, ch;
            g_clock->summary(w0 + a, w0 + a + 0.25, cm, cl, ch);
            std::printf(" [%.2f %.0f]", n ? sum / n : 0.0, cm);
        }
        std::printf("\n");
    }
    std::vector<double> host, total, device, hostSync;
    for (int i = 0; i < 300; ++i) {
        const auto a = Clock::now();
        RPPB_CHECK_HIP(hipEventRecord(e0, hs));
        fkl.fused();
        RPPB_CHECK_HIP(hipEventRecord(e1, hs));
        const auto b = Clock::now();
        fkl.sync();
        const auto c = Clock::now();
        float ms = 0;
        RPPB_CHECK_HIP(hipEventElapsedTime(&ms, e0, e1));
        host.push_back(std::chrono::duration<double, std::milli>(b - a).count());
        total.push_back(std::chrono::duration<double, std::milli>(c - a).count());
        device.push_back(ms);
    }
    // the launch alone, back to back, to see the cost of the host side without waiting for the device
    std::vector<double> burst;
    for (int r = 0; r < 20; ++r) {
        const auto a = Clock::now();
        for (int i = 0; i < 20; ++i) fkl.fused();
        const auto b = Clock::now();
        fkl.sync();
        burst.push_back(std::chrono::duration<double, std::milli>(b - a).count() / 20.0);
    }
    const double c1 = g_clock->now();
    double cm, cl, ch;
    g_clock->summary(c0, c1, cm, cl, ch);
    std::printf("%-12s N=%3d  host(launch) %.4f ms  total(launch+sync) %.4f ms  device(events) %.4f ms  burst launch %.4f ms  | sclk mean %.0f min %.0f max %.0f MHz\n", label, N, med(host), med(total),
                med(device), med(burst), cm, cl, ch);
    (void)hostSync;
    RPPB_CHECK_HIP(hipEventDestroy(e0));
    RPPB_CHECK_HIP(hipEventDestroy(e1));
}

int launch() {
    try {
        fk::Stream_<fk::ParArch::GPU_AMD> stream;
        fk::Ptr2D<U3> source(SRC_W, SRC_H);
        uint32_t state = 12345u;
        for (int y = 0; y < SRC_H; ++y) {
            for (int x = 0; x < SRC_W; ++x) {
                state = state * 1664525u + 1013904223u;
                source.at(x, y) = fk::make_<U3>(static_cast<uchar>(state >> 24), static_cast<uchar>(state >> 16), static_cast<uchar>(state >> 8));
            }
        }
        source.upload(stream);
        stream.sync();
        using HalfPixel = fk::ResizeHalfPixel<fk::InterpolationType::INTER_LINEAR>;
        ClockSampler sampler;
        g_clock = &sampler;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // the order of the benchmark: the default (ASYMMETRIC) object is allocated first, the HALF_PIXEL one second
        const auto each = [&](auto n_c) {
            constexpr int N = decltype(n_c)::value;
            FklA<N> asym(stream, source);
            FklA<N, HalfPixel> half(stream, source);
            std::printf("-- allocation order: asymmetric first, half-pixel second\n");
            for (int rep = 0; rep < 2; ++rep) {
                measure<N>("asymmetric", stream, asym);
                measure<N>("half-pixel", stream, half);
            }
        };
        each(std::integral_constant<int, 100>{});
        each(std::integral_constant<int, 200>{});
        each(std::integral_constant<int, 300>{});
        // after an idle gap (the host works, as in the benchmark between two batch sizes), with a warm-up of 0.5 s, 2 s and 5 s
        {
            constexpr int N = 300;
            FklA<N> asym(stream, source);
            FklA<N, HalfPixel> half(stream, source);
            for (const double w : {0.5, 2.0, 5.0, 0.5}) {
                std::this_thread::sleep_for(std::chrono::seconds(3));
                std::printf("-- after 3 s without GPU work, asymmetric with a warm-up of %.1f s, then half-pixel with 0.5 s\n", w);
                measure<N>("asymmetric", stream, asym, w, true);
                measure<N>("half-pixel", stream, half, 0.5, false);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
    return 0;
}
