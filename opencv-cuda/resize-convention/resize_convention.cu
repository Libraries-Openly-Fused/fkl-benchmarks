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

// Which pixel-sampling convention does each bilinear resize follow? Written for the numerics section (E6) and to decide what the
// resize-based benchmarks may claim to check.
//
// Two scalar references in double precision, on a seeded noise image (float, 3 channels, values in [0, 255), so there is no
// rounding in the way):
//   "FKL"           src = x * (srcSize / dstSize)                    (fk::Resize, resize.h exec_resize)
//                   x1 = floor(src), x2 = x1 + 1, the second read is clamped to the last column/row, weights (x2 - src), (src - x1)
//   "pixel centre"  src = (x + 0.5) * (srcSize / dstSize) - 0.5      (cv::resize)
//                   x1 = floor(src), weights (1 - f), f with f = src - x1, both reads clamped to [0, size - 1]
// The float cases are followed by the same cases with an 8UC3 source (the type of the benchmarks), where cv::cuda::resize and the 8-bit NPP
// call round their output.
// Candidates: FKL through cvGS::resize, cv::cuda::resize (INTER_LINEAR), nppiResize_32f_C3R_Ctx (NPPI_INTER_LINEAR) and
// nppiResizeBatch_32f_C3R_Advanced_Ctx with a batch of two (the call the NPP baseline of the paper uses; the batch API rejects a batch of one). For each candidate and
// each reference: maximum and mean absolute difference and the fraction of values that differ by more than 1e-3. A candidate
// follows a convention when its difference to that reference is at the level of float rounding (about 1e-5).
//
// The seed is fixed and written to the CSV header. Writes resize_convention.csv.
#include <common/main.h>
#include <common/csv_writer.h>

#include <cvGPUSpeedup.cuh>
#include <npp.h>
#include <nppi_geometry_transforms.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudawarping.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                                                            \
    {                                                                                                               \
        const cudaError_t err_ = (call);                                                                            \
        if (err_ != cudaSuccess) {                                                                                  \
            std::cerr << #call << ": " << cudaGetErrorString(err_) << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            return -1;                                                                                              \
        }                                                                                                           \
    }
#define CHECK_NPP(call)                                                                                          \
    {                                                                                                            \
        const NppStatus st_ = (call);                                                                            \
        if (st_ != NPP_SUCCESS) {                                                                                \
            std::cerr << #call << ": NPP status " << st_ << " (" << __FILE__ << ":" << __LINE__ << ")\n";        \
            return -1;                                                                                           \
        }                                                                                                        \
    }

namespace {
constexpr unsigned long long SEED = 20261030ull;

using Image = std::vector<float>;  // height x width x 3, row major

Image makeNoise(const int w, const int h) {
    std::mt19937_64 gen(SEED + 1000003ull * w + 7919ull * h);
    std::uniform_real_distribution<float> dist(0.f, 255.f);
    Image img(static_cast<size_t>(w) * h * 3);
    for (auto& v : img) v = dist(gen);
    return img;
}

float at(const Image& s, const int w, const int x, const int y, const int c) {
    return s[(static_cast<size_t>(y) * w + x) * 3 + c];
}

// FKL: src = x * scale, the second read is clamped, the weights use the unclamped x2 = x1 + 1
Image referenceFKL(const Image& s, const int sw, const int sh, const int dw, const int dh) {
    Image d(static_cast<size_t>(dw) * dh * 3);
    const double fx = static_cast<double>(sw) / dw, fy = static_cast<double>(sh) / dh;
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            const double sx = x * fx, sy = y * fy;
            const int x1 = static_cast<int>(std::floor(sx)), y1 = static_cast<int>(std::floor(sy));
            const int x2 = x1 + 1, y2 = y1 + 1;
            const int x2r = std::min(x2, sw - 1), y2r = std::min(y2, sh - 1);
            for (int c = 0; c < 3; ++c) {
                const double v = at(s, sw, x1, y1, c) * ((x2 - sx) * (y2 - sy)) + at(s, sw, x2r, y1, c) * ((sx - x1) * (y2 - sy)) +
                                 at(s, sw, x1, y2r, c) * ((x2 - sx) * (sy - y1)) + at(s, sw, x2r, y2r, c) * ((sx - x1) * (sy - y1));
                d[(static_cast<size_t>(y) * dw + x) * 3 + c] = static_cast<float>(v);
            }
        }
    }
    return d;
}

// pixel centre (cv::resize): src = (x + 0.5) * scale - 0.5, both reads clamped to the image
Image referencePixelCentre(const Image& s, const int sw, const int sh, const int dw, const int dh) {
    Image d(static_cast<size_t>(dw) * dh * 3);
    const double fx = static_cast<double>(sw) / dw, fy = static_cast<double>(sh) / dh;
    for (int y = 0; y < dh; ++y) {
        for (int x = 0; x < dw; ++x) {
            const double sx = (x + 0.5) * fx - 0.5, sy = (y + 0.5) * fy - 0.5;
            const int x1 = static_cast<int>(std::floor(sx)), y1 = static_cast<int>(std::floor(sy));
            const double ax = sx - x1, ay = sy - y1;
            const int xa = std::clamp(x1, 0, sw - 1), xb = std::clamp(x1 + 1, 0, sw - 1);
            const int ya = std::clamp(y1, 0, sh - 1), yb = std::clamp(y1 + 1, 0, sh - 1);
            for (int c = 0; c < 3; ++c) {
                const double v = at(s, sw, xa, ya, c) * ((1 - ax) * (1 - ay)) + at(s, sw, xb, ya, c) * (ax * (1 - ay)) +
                                 at(s, sw, xa, yb, c) * ((1 - ax) * ay) + at(s, sw, xb, yb, c) * (ax * ay);
                d[(static_cast<size_t>(y) * dw + x) * 3 + c] = static_cast<float>(v);
            }
        }
    }
    return d;
}

struct Diff {
    double maxAbs, meanAbs, fracGt;
};

Diff compare(const Image& a, const Image& b) {
    double maxAbs = 0, sum = 0;
    size_t gt = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - b[i]);
        maxAbs = std::max(maxAbs, d);
        sum += d;
        gt += d > 1e-3;
    }
    return {maxAbs, sum / a.size(), static_cast<double>(gt) / a.size()};
}

NppStreamContext makeNppContext(cudaStream_t stream) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    NppStreamContext ctx{};
    ctx.hStream = stream;
    ctx.nCudaDeviceId = device;
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = prop.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = prop.minor;
    unsigned int flags = 0;
    cudaStreamGetFlags(stream, &flags);
    ctx.nStreamFlags = flags;
    return ctx;
}

struct Case {
    int sw, sh, dw, dh;
    const char* what;
};
}  // namespace

int launch() {
    const std::vector<Case> cases = {
        {60, 120, 64, 128, "benchmark crop 60x120 -> 64x128 (up, x1.07)"},
        {60, 120, 30, 60, "60x120 -> 30x60 (down, x2)"},
        {33, 17, 67, 31, "33x17 -> 67x31 (odd sizes, up)"},
        {3840, 2160, 640, 360, "3840x2160 -> 640x360 (down, x6)"},
    };

    cv::cuda::Stream cvStream;
    cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cvStream);
    const NppStreamContext nppCtx = makeNppContext(stream);

    fklb::CsvWriter csv("resize_convention.csv", "resize sampling convention: candidates vs two scalar double references",
                        {"case", "candidate", "reference", "max_abs_diff", "mean_abs_diff", "fraction_gt_1e-3"},
                        "seeded float noise in [0,255), seed " + std::to_string(SEED) + " + 1000003*width + 7919*height; "
                        "reference FKL: src=x*scale; reference pixel centre: src=(x+0.5)*scale-0.5 with clamped reads; "
                        "float images, so no rounding is involved");

    for (const Case& cs : cases) {
        const Image src = makeNoise(cs.sw, cs.sh);
        const Image refFKL = referenceFKL(src, cs.sw, cs.sh, cs.dw, cs.dh);
        const Image refPC = referencePixelCentre(src, cs.sw, cs.sh, cs.dw, cs.dh);

        cv::Mat hSrc(cs.sh, cs.sw, CV_32FC3, const_cast<float*>(src.data()));
        cv::cuda::GpuMat dSrc;
        dSrc.upload(hSrc, cvStream);
        const cv::Size dsize(cs.dw, cs.dh);
        std::vector<std::pair<std::string, Image>> candidates;

        // 1. FKL through cvGS::resize
        {
            cv::cuda::GpuMat dOut(dsize, CV_32FC3);
            cvGS::executeOperations(cvStream, cvGS::resize<CV_32FC3, cv::INTER_LINEAR>(dSrc, dsize, 0., 0.),
                                    cvGS::write<CV_32FC3>(dOut));
            cv::Mat hOut;
            dOut.download(hOut, cvStream);
            cvStream.waitForCompletion();
            candidates.emplace_back("FKL (cvGS::resize, INTER_LINEAR)", Image((float*)hOut.data, (float*)hOut.data + hOut.total() * 3));
        }
        // 2. cv::cuda::resize
        {
            cv::cuda::GpuMat dOut;
            cv::cuda::resize(dSrc, dOut, dsize, 0., 0., cv::INTER_LINEAR, cvStream);
            cv::Mat hOut;
            dOut.download(hOut, cvStream);
            cvStream.waitForCompletion();
            candidates.emplace_back("cv::cuda::resize (INTER_LINEAR)", Image((float*)hOut.data, (float*)hOut.data + hOut.total() * 3));
        }
        // 3. nppiResize_32f_C3R_Ctx and 4. nppiResizeBatch_32f_C3R_Advanced_Ctx (the call of the NPP baseline), on the same data
        {
            Npp32f *dNppSrc = nullptr, *dNppDst = nullptr;
            size_t srcPitch = 0, dstPitch = 0;
            CHECK_CUDA(cudaMallocPitch(reinterpret_cast<void**>(&dNppSrc), &srcPitch, cs.sw * 3 * sizeof(float), cs.sh));
            CHECK_CUDA(cudaMallocPitch(reinterpret_cast<void**>(&dNppDst), &dstPitch, cs.dw * 3 * sizeof(float), cs.dh));
            CHECK_CUDA(cudaMemcpy2DAsync(dNppSrc, srcPitch, src.data(), cs.sw * 3 * sizeof(float), cs.sw * 3 * sizeof(float), cs.sh,
                                         cudaMemcpyHostToDevice, stream));
            const NppiSize srcSize{cs.sw, cs.sh}, dstSize{cs.dw, cs.dh};
            const NppiRect srcROI{0, 0, cs.sw, cs.sh}, dstROI{0, 0, cs.dw, cs.dh};
            CHECK_NPP(nppiResize_32f_C3R_Ctx(dNppSrc, static_cast<int>(srcPitch), srcSize, srcROI, dNppDst, static_cast<int>(dstPitch),
                                             dstSize, dstROI, NPPI_INTER_LINEAR, nppCtx));
            Image out(static_cast<size_t>(cs.dw) * cs.dh * 3);
            CHECK_CUDA(cudaMemcpy2DAsync(out.data(), cs.dw * 3 * sizeof(float), dNppDst, dstPitch, cs.dw * 3 * sizeof(float), cs.dh,
                                         cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            candidates.emplace_back("nppiResize_32f_C3R_Ctx (NPPI_INTER_LINEAR)", out);

            // batch of two (NPP's batch API rejects a batch of one: NPP_BAD_ARGUMENT_ERROR): the same source twice, two destinations,
            // descriptors on the device, as the baseline of the paper calls it; the first destination is compared
            Npp32f* dNppDst2 = nullptr;
            size_t dstPitch2 = 0;
            CHECK_CUDA(cudaMallocPitch(reinterpret_cast<void**>(&dNppDst2), &dstPitch2, cs.dw * 3 * sizeof(float), cs.dh));
            NppiImageDescriptor hSrcDesc[2]{}, hDstDesc[2]{};
            NppiResizeBatchROI_Advanced hROI[2]{};
            for (int b = 0; b < 2; ++b) {
                hSrcDesc[b].pData = dNppSrc; hSrcDesc[b].nStep = static_cast<int>(srcPitch); hSrcDesc[b].oSize = srcSize;
                hDstDesc[b].pData = b == 0 ? dNppDst : dNppDst2;
                hDstDesc[b].nStep = static_cast<int>(b == 0 ? dstPitch : dstPitch2);
                hDstDesc[b].oSize = dstSize;
                hROI[b].oSrcRectROI = srcROI;
                hROI[b].oDstRectROI = dstROI;
            }
            NppiImageDescriptor *dSrcDesc = nullptr, *dDstDesc = nullptr;
            NppiResizeBatchROI_Advanced* dROI = nullptr;
            CHECK_CUDA(cudaMalloc(&dSrcDesc, sizeof(hSrcDesc)));
            CHECK_CUDA(cudaMalloc(&dDstDesc, sizeof(hDstDesc)));
            CHECK_CUDA(cudaMalloc(&dROI, sizeof(hROI)));
            CHECK_CUDA(cudaMemcpyAsync(dSrcDesc, hSrcDesc, sizeof(hSrcDesc), cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(dDstDesc, hDstDesc, sizeof(hDstDesc), cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaMemcpyAsync(dROI, hROI, sizeof(hROI), cudaMemcpyHostToDevice, stream));
            CHECK_CUDA(cudaMemset2DAsync(dNppDst, dstPitch, 0, cs.dw * 3 * sizeof(float), cs.dh, stream));
            CHECK_NPP(nppiResizeBatch_32f_C3R_Advanced_Ctx(cs.dw, cs.dh, dSrcDesc, dDstDesc, dROI, 2, NPPI_INTER_LINEAR, nppCtx));
            Image outB(static_cast<size_t>(cs.dw) * cs.dh * 3);
            CHECK_CUDA(cudaMemcpy2DAsync(outB.data(), cs.dw * 3 * sizeof(float), dNppDst, dstPitch, cs.dw * 3 * sizeof(float), cs.dh,
                                         cudaMemcpyDeviceToHost, stream));
            CHECK_CUDA(cudaStreamSynchronize(stream));
            candidates.emplace_back("nppiResizeBatch_32f_C3R_Advanced_Ctx (batch of 2, NPPI_INTER_LINEAR)", outB);
            cudaFree(dNppDst2);
            cudaFree(dSrcDesc); cudaFree(dDstDesc); cudaFree(dROI); cudaFree(dNppSrc); cudaFree(dNppDst);
        }

        // the references against each other, then every candidate against both references and against the others
        {
            const Diff d = compare(refFKL, refPC);
            csv.row(cs.what, "scalar reference FKL", "scalar reference pixel centre", d.maxAbs, d.meanAbs, d.fracGt);
            std::cout << cs.what << "\n  reference FKL vs reference pixel centre: max " << d.maxAbs << ", fraction > 1e-3 " << d.fracGt << "\n";
        }
        for (const auto& [name, img] : candidates) {
            const Diff dF = compare(img, refFKL), dP = compare(img, refPC);
            csv.row(cs.what, name, "scalar reference FKL", dF.maxAbs, dF.meanAbs, dF.fracGt);
            csv.row(cs.what, name, "scalar reference pixel centre", dP.maxAbs, dP.meanAbs, dP.fracGt);
            std::cout << "  " << name << "\n      vs FKL convention:          max " << dF.maxAbs << ", mean " << dF.meanAbs << ", fraction>1e-3 " << dF.fracGt
                      << "\n      vs pixel-centre convention: max " << dP.maxAbs << ", mean " << dP.meanAbs << ", fraction>1e-3 " << dP.fracGt << "\n";
        }
        for (size_t i = 0; i < candidates.size(); ++i) {
            for (size_t j = i + 1; j < candidates.size(); ++j) {
                const Diff d = compare(candidates[i].second, candidates[j].second);
                csv.row(cs.what, candidates[i].first, candidates[j].first, d.maxAbs, d.meanAbs, d.fracGt);
            }
        }
        csv.flush();

        // The same cases with an 8-bit 3-channel source (the type of the benchmarks): the integer image holds the rounded noise values.
        // cv::cuda::resize and nppiResize_8u_C3R round their output to 8 bits, so a candidate that follows a convention is within
        // about 0.5 of that reference (never exactly 0 or 1e-5); the FKL path (cvGS::resize) keeps float output.
        {
            Image src8f(src.size());
            std::vector<unsigned char> src8(src.size());
            for (size_t i = 0; i < src.size(); ++i) { src8[i] = static_cast<unsigned char>(std::floor(src[i])); src8f[i] = src8[i]; }
            const Image ref8FKL = referenceFKL(src8f, cs.sw, cs.sh, cs.dw, cs.dh);
            const Image ref8PC = referencePixelCentre(src8f, cs.sw, cs.sh, cs.dw, cs.dh);
            cv::Mat hSrc8(cs.sh, cs.sw, CV_8UC3, src8.data());
            cv::cuda::GpuMat dSrc8;
            dSrc8.upload(hSrc8, cvStream);
            std::vector<std::pair<std::string, Image>> cand8;
            {
                cv::cuda::GpuMat dOut(dsize, CV_32FC3);
                cvGS::executeOperations(cvStream, cvGS::resize<CV_8UC3, cv::INTER_LINEAR>(dSrc8, dsize, 0., 0.), cvGS::write<CV_32FC3>(dOut));
                cv::Mat hOut;
                dOut.download(hOut, cvStream);
                cvStream.waitForCompletion();
                cand8.emplace_back("FKL (cvGS::resize, 8UC3 input, float output)", Image((float*)hOut.data, (float*)hOut.data + hOut.total() * 3));
            }
            {
                cv::cuda::GpuMat dOut;
                cv::cuda::resize(dSrc8, dOut, dsize, 0., 0., cv::INTER_LINEAR, cvStream);
                cv::Mat hOut;
                dOut.download(hOut, cvStream);
                cvStream.waitForCompletion();
                Image out(hOut.total() * 3);
                for (size_t i = 0; i < out.size(); ++i) out[i] = hOut.data[i];
                cand8.emplace_back("cv::cuda::resize (INTER_LINEAR, 8UC3 input and output)", out);
            }
            {
                Npp8u *dNppSrc = nullptr, *dNppDst = nullptr;
                size_t srcPitch = 0, dstPitch = 0;
                CHECK_CUDA(cudaMallocPitch(reinterpret_cast<void**>(&dNppSrc), &srcPitch, cs.sw * 3, cs.sh));
                CHECK_CUDA(cudaMallocPitch(reinterpret_cast<void**>(&dNppDst), &dstPitch, cs.dw * 3, cs.dh));
                CHECK_CUDA(cudaMemcpy2DAsync(dNppSrc, srcPitch, src8.data(), cs.sw * 3, cs.sw * 3, cs.sh, cudaMemcpyHostToDevice, stream));
                const NppiSize srcSize{cs.sw, cs.sh}, dstSize{cs.dw, cs.dh};
                const NppiRect srcROI{0, 0, cs.sw, cs.sh}, dstROI{0, 0, cs.dw, cs.dh};
                CHECK_NPP(nppiResize_8u_C3R_Ctx(dNppSrc, static_cast<int>(srcPitch), srcSize, srcROI, dNppDst, static_cast<int>(dstPitch), dstSize,
                                                dstROI, NPPI_INTER_LINEAR, nppCtx));
                std::vector<unsigned char> out8(static_cast<size_t>(cs.dw) * cs.dh * 3);
                CHECK_CUDA(cudaMemcpy2DAsync(out8.data(), cs.dw * 3, dNppDst, dstPitch, cs.dw * 3, cs.dh, cudaMemcpyDeviceToHost, stream));
                CHECK_CUDA(cudaStreamSynchronize(stream));
                Image out(out8.size());
                for (size_t i = 0; i < out.size(); ++i) out[i] = out8[i];
                cand8.emplace_back("nppiResize_8u_C3R_Ctx (NPPI_INTER_LINEAR, 8UC3 input and output)", out);
                cudaFree(dNppSrc); cudaFree(dNppDst);
            }
            for (const auto& [name, img] : cand8) {
                const Diff dF = compare(img, ref8FKL), dP = compare(img, ref8PC);
                csv.row(cs.what, name, "scalar reference FKL (8-bit source)", dF.maxAbs, dF.meanAbs, dF.fracGt);
                csv.row(cs.what, name, "scalar reference pixel centre (8-bit source)", dP.maxAbs, dP.meanAbs, dP.fracGt);
                std::cout << "  [8UC3] " << name << "\n      vs FKL convention:          max " << dF.maxAbs << ", mean " << dF.meanAbs
                          << "\n      vs pixel-centre convention: max " << dP.maxAbs << ", mean " << dP.meanAbs << "\n";
            }
        }
        csv.flush();
    }
    return 0;
}
