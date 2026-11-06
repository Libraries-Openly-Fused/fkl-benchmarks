// TEST B (2026-11-06): the host time of BUILDING the descriptors of the timed call of benchmark_CPU_OpenCV_vs_cvGS (cvGS) and benchmark_CPU_NPP_vs_FastNPP (FastNPP), at the CURRENT pins
// (cvGS 6d26c65, FastNPP a1b5e245, FKL c8fcbe8), as a function of the host optimisation flags and of the number of crops B. HOST ONLY: no CUDA call, no kernel launch (run with CUDA_VISIBLE_DEVICES empty:
// a CUDA call would fail). What is built per call (the same expressions as the timed calls of the two harnesses, without the launch):
//   cvGS:     cvGS::resize<CV_8UC3, INTER_LINEAR, B>(crops, up, B), cvtColor, multiply, subtract, divide, split(tensor)
//   FastNPP:  ResizeBatch_8u32f_C3R_Advanced_Ctx<LINEAR, B>(...) (body copied from include/fast_npp.h, its one cudaGetDevice call replaced by a constant: it is a constant per call, not per crop),
//             SwapChannels, MulC, SubC, DivC, CopyBatch_32f_C3P3R_Ctx<B> (the per-crop write descriptor)
#include <cvGPUSpeedup.cuh>
#include <fast_npp.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

static volatile unsigned long long sink;
template <typename T> static inline void keep(const T& v) { asm volatile("" : : "r"(&v) : "memory"); }

template <int BATCH>
static auto fastnppResize(const int nMaxWidth, const int nMaxHeight, const NppiImageDescriptor* const h_pBatchSrc) {   // body of fastNPP::ResizeBatch_8u32f_C3R_Advanced_Ctx at a1b5e245 (fast_npp.h:31-49), cudaGetDevice -> constant
    const int currentDevice{ 0 };
    std::array<fk::Ptr2D<uchar3>, BATCH> srcBatch;
    for (int i = 0; i < BATCH; ++i) {
        srcBatch[i] = fk::Ptr2D<uchar3>(reinterpret_cast<uchar3*>(h_pBatchSrc[i].pData), h_pBatchSrc[i].oSize.width, h_pBatchSrc[i].oSize.height, h_pBatchSrc[i].nStep, fk::MemType::Device, currentDevice);
    }
    const fk::Size dstSize(nMaxWidth, nMaxHeight);
    return fk::PerThreadRead<fk::ND::_2D, uchar3>::build(srcBatch).then(fk::ResizeHalfPixel<fk::InterpolationType::INTER_LINEAR>::build(dstSize));
}

template <typename F>
static double timePerCall(F&& f, const int rounds = 7) {      // median over rounds of the mean of a block of calls; the block is sized to about 30 ms
    for (int i = 0; i < 50; ++i) f();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 50; ++i) f();
    const double once = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 50;
    const int block = std::max(20, static_cast<int>(0.03 / std::max(once, 1e-7)));
    std::vector<double> t;
    for (int r = 0; r < rounds; ++r) {
        auto a = std::chrono::steady_clock::now();
        for (int i = 0; i < block; ++i) f();
        t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count() / block);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2] * 1e6;      // microseconds
}

template <int BATCH>
static void one(const char* flagset) {
    std::array<cv::cuda::GpuMat, BATCH> crops;
    for (int i = 0; i < BATCH; ++i) crops[i] = cv::cuda::GpuMat(120, 60, CV_8UC3, reinterpret_cast<void*>(0x100000 + 64 * i), 3840 * 3);   // headers only: no allocation, no CUDA call
    cv::cuda::GpuMat out(BATCH, 64 * 128 * 3, CV_32F, reinterpret_cast<void*>(0x10000), 64 * 128 * 3 * sizeof(float));
    const cv::Size up(64, 128);
    const cv::Scalar a(0.3, 0.3, 0.3), s(1, 4, 3.2), d(3.2, 0.6, 11.8);
    const double tCvgsRead = timePerCall([&] { auto r = cvGS::resize<CV_8UC3, cv::INTER_LINEAR, BATCH>(crops, up, BATCH); keep(r); sink = reinterpret_cast<unsigned long long>(&r); });
    const double tCvgs = timePerCall([&] {
        auto r = cvGS::resize<CV_8UC3, cv::INTER_LINEAR, BATCH>(crops, up, BATCH);
        auto c = cvGS::cvtColor<cv::COLOR_RGB2BGR, CV_32FC3>(); auto m = cvGS::multiply<CV_32FC3>(a); auto su = cvGS::subtract<CV_32FC3>(s); auto dv = cvGS::divide<CV_32FC3>(d); auto w = cvGS::split<CV_32FC3>(out, up);
        keep(r); keep(c); keep(m); keep(su); keep(dv); keep(w); });
    std::vector<NppiImageDescriptor> desc(BATCH);
    for (int i = 0; i < BATCH; ++i) { desc[i].pData = reinterpret_cast<void*>(0x100000 + 64 * i); desc[i].nStep = 3840 * 3; desc[i].oSize = NppiSize{60, 120}; }
    std::array<Npp32f*, BATCH> dst[3];
    for (int c = 0; c < 3; ++c) for (int i = 0; i < BATCH; ++i) dst[c][i] = reinterpret_cast<Npp32f*>(0x200000 + 64 * (c * BATCH + i));
    const int order[3] = {2, 1, 0};
    const float mul[3] = {0.3f, 0.3f, 0.3f}, sub[3] = {1.f, 4.f, 3.2f}, dv[3] = {3.2f, 0.6f, 11.8f};
    const double tNppRead = timePerCall([&] { auto r = fastnppResize<BATCH>(64, 128, desc.data()); keep(r); });
    const double tNpp = timePerCall([&] {
        auto r = fastnppResize<BATCH>(64, 128, desc.data());
        auto sw = fastNPP::SwapChannels_32f_C3R_Ctx(order); auto m = fastNPP::MulC_32f_C3R_Ctx(mul); auto su = fastNPP::SubC_32f_C3R_Ctx(sub); auto d2 = fastNPP::DivC_32f_C3R_Ctx(dv);
        auto w = fastNPP::CopyBatch_32f_C3P3R_Ctx<BATCH>(dst, 64 * 4, NppiSize{64, 128});
        keep(r); keep(sw); keep(m); keep(su); keep(d2); keep(w); });
    std::printf("%s,%d,%.4f,%.4f,%.4f,%.4f\n", flagset, BATCH, tCvgsRead, tCvgs, tNppRead, tNpp);
}

int main(int argc, char** argv) {
    const char* flagset = argc > 1 ? argv[1] : "unnamed";
    one<2>(flagset); one<12>(flagset); one<42>(flagset); one<142>(flagset);
    return 0;
}
