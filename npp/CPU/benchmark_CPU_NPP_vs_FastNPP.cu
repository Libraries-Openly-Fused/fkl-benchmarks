/* Copyright 2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// CPU time of NPP against FastNPP: the host time each side spends to enqueue the work of one frame (std::chrono::steady_clock around the
// calls, no synchronisation inside), 100 iterations per batch size after a warm-up sweep (npp/nppbenchmark.h). The pipeline is that of
// npp/batchresize/benchmark_npp_batchresize_x_split3D.cu: BATCH crops of 60 x 120 pixels (8 bit, 3 channels), crop i at (i, i) of a
// 3840 x 2160 frame, converted to float, resized to 64 x 128 (bilinear), channels swapped (RGB -> BGR), multiplied, subtracted, divided,
// and written as the three planes of a float tensor.
// Everything a side does for a frame is timed, building its batch included: the crop locations may change on every image, so NPP computes
// the source pointer of every crop in the call, and FastNPP fills its descriptor and ROI arrays and builds its operations in the call.
// Each side owns its buffers: its own copy of the frame (cudaMallocPitch, uploaded from the same host bytes) and its own output. NPP's
// intermediate images are cudaMallocPitch images, one per crop; FastNPP writes one dense tensor (cudaMalloc). The NPP stream context is filled
// field by field, as NPP documents it, and FastNPP takes the same context.

#include <common/main.h>
#include <common/sequence.h>
#include <common/status_check.h>

#include <npp.h>

#include <npp/nppbenchmark.h>
#include <npp/resize_agreement.h>
#include <fast_npp.h>

#include <array>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#ifdef ENABLE_BENCHMARK
constexpr char VARIABLE_DIMENSION[]{ "Batch size" };

#ifndef CUDART_MAJOR_VERSION
#error CUDART_MAJOR_VERSION Undefined!
#elif (CUDART_MAJOR_VERSION == 11)
constexpr size_t NUM_EXPERIMENTS = 8;
#elif (CUDART_MAJOR_VERSION >= 12)
constexpr size_t NUM_EXPERIMENTS = 16;
#endif // CUDART_MAJOR_VERSION

constexpr size_t FIRST_VALUE = 2;
constexpr size_t INCREMENT = 10;
constexpr std::array<size_t, NUM_EXPERIMENTS> batchValues = fklb::arrayIndexSequence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

constexpr int CROP_W = 60;
constexpr int CROP_H = 120;
constexpr int UP_W = 64;
constexpr int UP_H = 128;
constexpr size_t PLANE = static_cast<size_t>(UP_W) * UP_H;   // floats per plane of the output
constexpr float ALPHA = 0.3f;

template <int BATCH>
bool test_NPP_cpu_batchresize_x_split3D(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, cudaStream_t stream) {
    // Same input on both sides: seeded noise, so that the agreement check can tell the sampling conventions apart (npp/resize_agreement.h)
    const std::vector<unsigned char>& sourceNoise = fklb_npp::seededNoise(static_cast<int>(NUM_ELEMS_X), static_cast<int>(NUM_ELEMS_Y));
    constexpr NppiSize crop_size{ CROP_W, CROP_H };
    constexpr NppiSize up_size{ UP_W, UP_H };
    // the constants of the pipeline: set up once, they do not depend on the crops
    constexpr Npp32f mulValue[3] = { ALPHA, ALPHA, ALPHA };
    constexpr Npp32f subValue[3] = { 1.f, 4.f, 3.2f };
    constexpr Npp32f divValue[3] = { 1.f, 4.f, 3.2f };
    const int aDstOrder[3] = { 2, 1, 0 };   // output channel c is resized channel aDstOrder[c]: RGB -> BGR

    // The NPP stream context, filled field by field from the device properties and the stream's flags (CUDA 13 has no nppGetStreamContext)
    int device = 0;
    FKLB_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    FKLB_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    NppStreamContext nppContext{};
    nppContext.hStream = stream;
    nppContext.nCudaDeviceId = device;
    nppContext.nMultiProcessorCount = props.multiProcessorCount;
    nppContext.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
    nppContext.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
    nppContext.nSharedMemPerBlock = props.sharedMemPerBlock;
    nppContext.nCudaDevAttrComputeCapabilityMajor = props.major;
    nppContext.nCudaDevAttrComputeCapabilityMinor = props.minor;
    FKLB_CUDA_CHECK(cudaStreamGetFlags(stream, &nppContext.nStreamFlags));

    // NPP: its own copy of the frame, uploaded from the same host bytes as FastNPP's
    Npp8u* nppSrc = nullptr;
    size_t nppSrcStep = 0;
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppSrc), &nppSrcStep, NUM_ELEMS_X * 3, NUM_ELEMS_Y));
    FKLB_CUDA_CHECK(cudaMemcpy2DAsync(nppSrc, nppSrcStep, sourceNoise.data(), NUM_ELEMS_X * 3, NUM_ELEMS_X * 3, NUM_ELEMS_Y,
                                      cudaMemcpyHostToDevice, stream));
    // NPP's images, one cudaMallocPitch per image: per crop, the float crop, the resized, swapped, multiplied, subtracted and divided images,
    // and the three output planes. cudaMallocPitch gives every image of one size the same step (checked).
    std::array<Npp32f*, BATCH> nppCrop, nppResized, nppSwap, nppMul, nppSub, nppDiv;
    std::array<Npp32f*, BATCH> nppPlane[3];
    size_t cropStep = 0, upStep = 0, planeStep = 0;
    for (int i = 0; i < BATCH; ++i) {
        size_t s[9];
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppCrop[i]), &s[0], CROP_W * 3 * sizeof(Npp32f), CROP_H));
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppResized[i]), &s[1], UP_W * 3 * sizeof(Npp32f), UP_H));
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppSwap[i]), &s[2], UP_W * 3 * sizeof(Npp32f), UP_H));
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppMul[i]), &s[3], UP_W * 3 * sizeof(Npp32f), UP_H));
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppSub[i]), &s[4], UP_W * 3 * sizeof(Npp32f), UP_H));
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppDiv[i]), &s[5], UP_W * 3 * sizeof(Npp32f), UP_H));
        for (int c = 0; c < 3; ++c) {
            FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppPlane[c][i]), &s[6 + c], UP_W * sizeof(Npp32f), UP_H));
        }
        if (i == 0) {
            cropStep = s[0];
            upStep = s[1];
            planeStep = s[6];
        }
        if (s[0] != cropStep || s[1] != upStep || s[2] != upStep || s[3] != upStep || s[4] != upStep || s[5] != upStep ||
            s[6] != planeStep || s[7] != planeStep || s[8] != planeStep) {
            throw std::runtime_error("cudaMallocPitch returned different steps for images of the same size");
        }
    }
    // NPP's batched-resize descriptors are set up once: they point only at NPP's own float crops and resized images, with full-image ROIs,
    // so they do not depend on where the crops are in the frame.
    NppiImageDescriptor *hResizeSrc = nullptr, *hResizeDst = nullptr, *dResizeSrc = nullptr, *dResizeDst = nullptr;
    NppiResizeBatchROI_Advanced *hResizeROI = nullptr, *dResizeROI = nullptr;
    FKLB_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hResizeSrc), sizeof(NppiImageDescriptor) * BATCH));
    FKLB_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hResizeDst), sizeof(NppiImageDescriptor) * BATCH));
    FKLB_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hResizeROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));
    FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dResizeSrc), sizeof(NppiImageDescriptor) * BATCH));
    FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dResizeDst), sizeof(NppiImageDescriptor) * BATCH));
    FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dResizeROI), sizeof(NppiResizeBatchROI_Advanced) * BATCH));
    for (int i = 0; i < BATCH; ++i) {
        hResizeSrc[i].pData = nppCrop[i];
        hResizeSrc[i].nStep = static_cast<int>(cropStep);
        hResizeSrc[i].oSize = crop_size;
        hResizeDst[i].pData = nppResized[i];
        hResizeDst[i].nStep = static_cast<int>(upStep);
        hResizeDst[i].oSize = up_size;
        hResizeROI[i].oSrcRectROI = NppiRect{ 0, 0, CROP_W, CROP_H };
        hResizeROI[i].oDstRectROI = NppiRect{ 0, 0, UP_W, UP_H };
    }
    FKLB_CUDA_CHECK(cudaMemcpyAsync(dResizeSrc, hResizeSrc, sizeof(NppiImageDescriptor) * BATCH, cudaMemcpyHostToDevice, stream));
    FKLB_CUDA_CHECK(cudaMemcpyAsync(dResizeDst, hResizeDst, sizeof(NppiImageDescriptor) * BATCH, cudaMemcpyHostToDevice, stream));
    FKLB_CUDA_CHECK(cudaMemcpyAsync(dResizeROI, hResizeROI, sizeof(NppiResizeBatchROI_Advanced) * BATCH, cudaMemcpyHostToDevice, stream));

    // FastNPP: its own copy of the frame (the same host bytes) and its own output, one dense tensor of BATCH images of 3 planes
    Npp8u* fastSrc = nullptr;
    size_t fastSrcStep = 0;
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&fastSrc), &fastSrcStep, NUM_ELEMS_X * 3, NUM_ELEMS_Y));
    FKLB_CUDA_CHECK(cudaMemcpy2DAsync(fastSrc, fastSrcStep, sourceNoise.data(), NUM_ELEMS_X * 3, NUM_ELEMS_X * 3, NUM_ELEMS_Y,
                                      cudaMemcpyHostToDevice, stream));
    Npp32f* fastOut = nullptr;
    FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&fastOut), BATCH * 3 * PLANE * sizeof(Npp32f)));
    // The plane pointers into FastNPP's output are set up once: they depend only on the layout of the output tensor, not on the crops.
    std::array<Npp32f*, BATCH> fastPlanes[3];
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < BATCH; ++i) {
            fastPlanes[c][i] = fastOut + (static_cast<size_t>(i) * 3 + c) * PLANE;
        }
    }
    // FastNPP's descriptor and ROI arrays, in host memory: FastNPP reads the descriptors (pointer, step and size of every crop) on the host
    // when it builds its operations. It does not read the ROI array (its resize assumes full-image ROIs); the array is filled anyway, in the
    // timed interval, as NPP's interface asks for it.
    std::array<NppiImageDescriptor, BATCH> fastSrcDesc{};
    std::array<NppiResizeBatchROI_Advanced, BATCH> fastROI{};

    if (BATCH == batchValues[0] && !warmup) {
        std::cout << "steps (bytes): NPP frame " << nppSrcStep << ", FastNPP frame " << fastSrcStep << ", NPP float crop " << cropStep
                  << ", NPP 64 x 128 float image " << upStep << ", NPP plane " << planeStep << "; FastNPP output dense, "
                  << UP_W * sizeof(Npp32f) << " bytes per row" << std::endl;
    }

    FKLB_CUDA_CHECK(cudaStreamSynchronize(stream));   // the uploads of the setup are done before the first timed interval of either side
    auto benchmarkTemp = initCPUBenchmark<ITERS, BATCH, FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>(__func__, VARIABLE_DIMENSION);
    for (int it = 0; it < ITERS; it++) {
        startNPP_CPU(benchmarkTemp);
        // NPP, timed. The crop locations may change on every image, so everything derived from them, the source pointer of each crop, is
        // computed here, in the call.
        for (int j = 0; j < BATCH; ++j) {
            FKLB_NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(nppSrc + j * nppSrcStep + j * 3, static_cast<int>(nppSrcStep), nppCrop[j],
                                                     static_cast<int>(cropStep), crop_size, nppContext));
        }
        FKLB_NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(UP_W, UP_H, dResizeSrc, dResizeDst, dResizeROI, BATCH, NPPI_INTER_LINEAR, nppContext));
        for (int j = 0; j < BATCH; ++j) {
            FKLB_NPP_CHECK(nppiSwapChannels_32f_C3R_Ctx(nppResized[j], static_cast<int>(upStep), nppSwap[j], static_cast<int>(upStep), up_size,
                                                        aDstOrder, nppContext));
            FKLB_NPP_CHECK(nppiMulC_32f_C3R_Ctx(nppSwap[j], static_cast<int>(upStep), mulValue, nppMul[j], static_cast<int>(upStep), up_size,
                                                nppContext));
            FKLB_NPP_CHECK(nppiSubC_32f_C3R_Ctx(nppMul[j], static_cast<int>(upStep), subValue, nppSub[j], static_cast<int>(upStep), up_size,
                                                nppContext));
            FKLB_NPP_CHECK(nppiDivC_32f_C3R_Ctx(nppSub[j], static_cast<int>(upStep), divValue, nppDiv[j], static_cast<int>(upStep), up_size,
                                                nppContext));
            Npp32f* const planes[3] = { nppPlane[0][j], nppPlane[1][j], nppPlane[2][j] };
            FKLB_NPP_CHECK(nppiCopy_32f_C3P3R_Ctx(nppDiv[j], static_cast<int>(upStep), planes, static_cast<int>(planeStep), up_size, nppContext));
        }
        stopNPP_startFastNPP_CPU(benchmarkTemp, it);

        // FastNPP, timed. The crop locations may change on every image, so its descriptor and ROI arrays are filled here, in the call, and
        // its operations are built on them; they are released before the interval ends.
        {
            for (int j = 0; j < BATCH; ++j) {
                fastSrcDesc[j].pData = fastSrc + j * fastSrcStep + j * 3;
                fastSrcDesc[j].nStep = static_cast<int>(fastSrcStep);
                fastSrcDesc[j].oSize = crop_size;
                fastROI[j].oSrcRectROI = NppiRect{ 0, 0, CROP_W, CROP_H };
                fastROI[j].oDstRectROI = NppiRect{ 0, 0, UP_W, UP_H };
            }
            const auto resizeIOp =
                fastNPP::ResizeBatch_8u32f_C3R_Advanced_Ctx<NPPI_INTER_LINEAR, BATCH>(UP_W, UP_H, fastSrcDesc.data(), fastROI.data());
            fastNPP::executeOperations(nppContext, resizeIOp, fastNPP::SwapChannels_32f_C3R_Ctx(aDstOrder),
                                       fastNPP::MulC_32f_C3R_Ctx(mulValue), fastNPP::SubC_32f_C3R_Ctx(subValue),
                                       fastNPP::DivC_32f_C3R_Ctx(divValue),
                                       fastNPP::CopyBatch_32f_C3P3R_Ctx(fastPlanes, static_cast<int>(UP_W * sizeof(Npp32f)), up_size));
        }
        stopFastNPP_CPU(benchmarkTemp, it);
        if (warmup) break;
    }
    processExecution<BATCH, ITERS, batchValues.size(), batchValues>(benchmarkTemp.resF, __func__, benchmarkTemp.NPPelapsedTime,
                                                                    benchmarkTemp.FastNPPelapsedTime, VARIABLE_DIMENSION);

    // results to the host (pinned memory), then the agreement check: each implementation must match ONE of the two sampling conventions
    // (npp/resize_agreement.h); the difference between FastNPP and NPP is recorded, not required to be small
    Npp32f *hFast = nullptr, *hNpp = nullptr;
    FKLB_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hFast), BATCH * 3 * PLANE * sizeof(Npp32f)));
    FKLB_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hNpp), BATCH * 3 * PLANE * sizeof(Npp32f)));
    FKLB_CUDA_CHECK(cudaMemcpyAsync(hFast, fastOut, BATCH * 3 * PLANE * sizeof(Npp32f), cudaMemcpyDeviceToHost, stream));
    for (int i = 0; i < BATCH; ++i) {
        for (int c = 0; c < 3; ++c) {
            FKLB_CUDA_CHECK(cudaMemcpy2DAsync(hNpp + (static_cast<size_t>(i) * 3 + c) * PLANE, UP_W * sizeof(Npp32f), nppPlane[c][i], planeStep,
                                              UP_W * sizeof(Npp32f), UP_H, cudaMemcpyDeviceToHost, stream));
        }
    }
    FKLB_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto fastAt = [&](int j, int c, int k) { return hFast[(static_cast<size_t>(j) * 3 + c) * PLANE + k]; };
    const auto nppAt = [&](int j, int c, int k) { return hNpp[(static_cast<size_t>(j) * 3 + c) * PLANE + k]; };
    const bool passed = fklb_npp::checkBatch(std::string(__func__), BATCH, static_cast<int>(NUM_ELEMS_X), CROP_W, CROP_H, UP_W, UP_H, aDstOrder,
                                             mulValue, subValue, divValue, fastAt, nppAt, sourceNoise);

    for (int i = 0; i < BATCH; ++i) {
        FKLB_CUDA_CHECK(cudaFree(nppCrop[i]));
        FKLB_CUDA_CHECK(cudaFree(nppResized[i]));
        FKLB_CUDA_CHECK(cudaFree(nppSwap[i]));
        FKLB_CUDA_CHECK(cudaFree(nppMul[i]));
        FKLB_CUDA_CHECK(cudaFree(nppSub[i]));
        FKLB_CUDA_CHECK(cudaFree(nppDiv[i]));
        for (int c = 0; c < 3; ++c) {
            FKLB_CUDA_CHECK(cudaFree(nppPlane[c][i]));
        }
    }
    FKLB_CUDA_CHECK(cudaFree(nppSrc));
    FKLB_CUDA_CHECK(cudaFree(fastSrc));
    FKLB_CUDA_CHECK(cudaFree(fastOut));
    FKLB_CUDA_CHECK(cudaFree(dResizeSrc));
    FKLB_CUDA_CHECK(cudaFree(dResizeDst));
    FKLB_CUDA_CHECK(cudaFree(dResizeROI));
    FKLB_CUDA_CHECK(cudaFreeHost(hResizeSrc));
    FKLB_CUDA_CHECK(cudaFreeHost(hResizeDst));
    FKLB_CUDA_CHECK(cudaFreeHost(hResizeROI));
    FKLB_CUDA_CHECK(cudaFreeHost(hFast));
    FKLB_CUDA_CHECK(cudaFreeHost(hNpp));
    return passed;
}

template <size_t... Is>
bool test_cpu_batchresize_x_split3D(const size_t NUM_ELEMS_X, const size_t NUM_ELEMS_Y, std::index_sequence<Is...>, cudaStream_t stream) {
    bool passed = true;
    ((passed &= test_NPP_cpu_batchresize_x_split3D<batchValues[Is]>(NUM_ELEMS_X, NUM_ELEMS_Y, stream)), ...);
    return passed;
}

#endif // ENABLE_BENCHMARK

int launch() {
#ifdef ENABLE_BENCHMARK
    constexpr size_t NUM_ELEMS_X = 3840;
    constexpr size_t NUM_ELEMS_Y = 2160;

    // one plain CUDA stream for both sides
    cudaStream_t stream;
    FKLB_CUDA_CHECK(cudaStreamCreate(&stream));
    unsigned int streamFlags = 0;
    FKLB_CUDA_CHECK(cudaStreamGetFlags(stream, &streamFlags));
    std::cout << "stream: cudaStreamCreate, flags " << streamFlags << std::endl;

    std::unordered_map<std::string, bool> results;
    const auto iSeq = std::make_index_sequence<batchValues.size()>{};
    try {
        // warm-up sweep: one iteration per point, nothing written to the timing CSV
        warmup = true;
        test_cpu_batchresize_x_split3D(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, stream);
        warmup = false;

        benchmarkNotes = "Host time to enqueue the work of one frame (steady_clock, no synchronisation inside). Each side owns its frame copy "
                         "and its output. Everything a side does for a frame is timed, building its batch included, because the crop "
                         "locations may change on every image: NPP computes the source pointer of every crop in the call; FastNPP fills "
                         "its descriptor and ROI arrays and builds its operations in the call. NPP's batched-resize descriptors point at "
                         "NPP's own images with full-image ROIs and are set up once.";
        results["test_NPP_cpu_batchresize_x_split3D"] = test_cpu_batchresize_x_split3D(NUM_ELEMS_X, NUM_ELEMS_Y, iSeq, stream);
    } catch (...) {
        // a failed run must not leave files that look complete
        markOpenFilesIncomplete();
        fklb_npp::markAgreementFilesIncomplete();
        throw;
    }
    stopCPUBenchmark();
    FKLB_CUDA_CHECK(cudaStreamDestroy(stream));

    int returnValue = 0;
    for (const auto& [key, passed] : results) {
        if (passed) {
            std::cout << key << " passed!!" << std::endl;
        } else {
            std::cout << key << " failed!!" << std::endl;
            returnValue = -1;
        }
    }
    return returnValue;
#else
    return 0;
#endif // ENABLE_BENCHMARK
}
