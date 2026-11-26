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

// FKL's DPPs before and after FKL #435 (task C: every DPP through executeOperations<DPP> and launchDPP_Kernel), on every back-end of this
// machine. The same source is built by build.sh against FKL b10fc66 (HARNESS_EXECUTORS=0: the entry functions of that tree) and against the
// #435 head (HARNESS_EXECUTORS=1: executeOperations<DPP>); the calls of the two forms are in the adaptors below, nothing else differs. Every case
// runs once per back-end on fixed inputs, its output buffer (pre-filled with a sentinel, all of it, so a value written outside the image also
// shows) is written to <back-end>_<case>.bin in the working directory, and outputs.tsv gets one line per case: back-end, case, bytes, sha256,
// sentinel values left, values. The two builds' files are compared byte for byte afterwards (compare.sh); nothing here compares or times anything.
// The cases:
// - LinearFilterDPP: float with fused read, coefficients and write (5x3, anchor 3,1); uchar3 to float3 box 3x3 (the coefficient read of the
//   removed executeBoxFilter); a BatchRead of 3 crops of different sizes (per-image sizes) into a TensorWrite; a 3D read into a 3D write.
// - MedianFilterDPP: float median 5x5; float minimum 3x5 with a fused read and write; a BatchRead of 3 uchar3 crops into a TensorWrite.
// - MorphologyDPP: float erosion 5x3; uchar opening 3x5; float box blur of 2 passes; E19's fkl_batch form (Crop of 3 rectangles, Resize, box
//   blur of 1 pass, TensorWrite); float3 box blur of 1 pass.
// - The Quad DPPs, each a 2D scalar case with static kernel sizes and a BatchRead of 3 vector crops with run-time sizes into a TensorWrite:
//   BoxFilterQuadDPP (float 3x3; float3 5x3), ConvQuadDPP (float 5x5; uchar3 to float3 3x3), MedianQuadDPP (uchar 3x3; float3 3x3),
//   MorphQuadDPP (uchar 3x3; float4 5x5).
// - TransformDPP: uchar3 to float3 (no thread fusion); thread fusion with a width it divides, with one it does not, and with a misaligned
//   read (the non thread fused fallback): the three kernels of its run-time choice.
// - DivergentBatchTransformDPP (GPU only, as on LTS): two sequences into two planes.
// The back-ends: ParArch::CPU and CPU_MULTITHREAD (FK_CPU_MULTITHREAD_ENABLED) in every build, and ParArch::GPU_AMD in a HIP build when
// HARNESS_GPU=1 (so a HIP binary can also run on the CPU only). The program prints the FKL commit it was built with (HARNESS_FKL_COMMIT).

#include <common/sha256.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/box_filter_fast.h>
#include <fused_kernel/algorithms/image_processing/convolution_fast.h>
#include <fused_kernel/algorithms/image_processing/crop.h>
#include <fused_kernel/algorithms/image_processing/linear_filter.h>
#include <fused_kernel/algorithms/image_processing/median_fast.h>
#include <fused_kernel/algorithms/image_processing/median_filter.h>
#include <fused_kernel/algorithms/image_processing/morphology.h>
#include <fused_kernel/algorithms/image_processing/morphology_fast.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/rect.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#ifndef HARNESS_FKL_COMMIT
#error "build.sh sets HARNESS_FKL_COMMIT to the FKL commit"
#endif
#ifndef HARNESS_EXECUTORS
#error "build.sh sets HARNESS_EXECUTORS: 1 for an FKL tree with the executors of every DPP (#435), 0 for one with the entry functions"
#endif

namespace {

using namespace fk;
// FKL's vector types (in a HIP build the HIP headers declare global ones with the same names)
using uchar3 = fk::uchar3;
using float3 = fk::float3;
using float4 = fk::float4;

template <ParArch PA>
constexpr bool IS_GPU = PA == ParArch::GPU_NVIDIA || PA == ParArch::GPU_AMD;
template <ParArch PA>
constexpr MemType MEMORY = IS_GPU<PA> ? MemType::DeviceAndPinned : MemType::Host;

template <ParArch PA>
const char* nameOf() {
    return PA == ParArch::CPU ? "CPU" : (PA == ParArch::CPU_MULTITHREAD ? "CPU_MULTITHREAD" : "GPU_AMD");
}

// One back-end: its stream, and the copies to and from the GPU (nothing on the CPU back-ends, whose buffers are host memory)
template <ParArch PA>
struct Backend {
    Stream_<PA> stream;
    template <typename P>
    void upload(P& p) {
        if constexpr (IS_GPU<PA>) p.upload(stream);
        (void)p;
    }
    template <typename P>
    void download(P& p) {
        if constexpr (IS_GPU<PA>) p.download(stream);
        (void)p;
    }
    void sync() {
        if constexpr (PA != ParArch::CPU) stream.sync();
    }
};

void require(const bool ran, const char* what) {
    if (!ran) throw std::runtime_error(std::string(what) + " returned false for valid arguments");
}

// ---- the calls of the two FKL forms
template <ParArch PA, typename D, typename Reads, typename M, typename A, typename W>
void linearFilter(Backend<PA>& b, const D& d, const Reads& reads, const M& m, const A& a, const W& w) {
#if HARNESS_EXECUTORS
    executeOperations<LinearFilterDPP<PA, D>>(b.stream, d, reads, m, a, w);
#else
    if constexpr (PA == ParArch::CPU) LinearFilterDPP<PA, D>::exec(d, reads, m, a, w);
    else require(executeLinearFilter(d, reads, m, a, w, b.stream), "executeLinearFilter");
#endif
}
template <ParArch PA, typename D, typename R, typename W, typename S>
void medianFilter(Backend<PA>& b, const D& d, const R& r, const W& w, const S& sel) {
#if HARNESS_EXECUTORS
    executeOperations<MedianFilterDPP<PA, D>>(b.stream, d, r, w, sel);
#else
    if constexpr (PA == ParArch::CPU) MedianFilterDPP<PA, D>::exec(d, r, w, sel);
    else require(executeMedianFilter(d, r, w, sel, b.stream), "executeMedianFilter");
#endif
}
template <ParArch PA, typename D, typename R, typename W, typename Red>
void morphology(Backend<PA>& b, const D& d, const R& r, const W& w, const Red& red) {
#if HARNESS_EXECUTORS
    executeOperations<MorphologyDPP<PA, D>>(b.stream, d, r, w, red);
#else
    if constexpr (PA == ParArch::CPU) MorphologyDPP<PA, D>::exec(d, r, w, red);
    else require(executeMorphology(d, r, w, red, b.stream), "executeMorphology");
#endif
}
template <typename T, int PASSES, typename D>
auto boxBlur(const D& d) {
#if HARNESS_EXECUTORS
    return boxBlurReducers<T, PASSES>(d);
#else
    return morphology_detail::boxBlurReducers<T, PASSES>(d);
#endif
}
template <typename DPP, ParArch PA, typename... IOps>
void boxQuad(Backend<PA>& b, const BoxFilterQuadDetails& d, const IOps&... i) {
#if HARNESS_EXECUTORS
    executeOperations<DPP>(b.stream, d, i...);
#else
    require(executeBoxFilterQuad<DPP>(b.stream, d, i...), "executeBoxFilterQuad");
#endif
}
template <typename DPP, ParArch PA, typename... IOps>
void convQuad(Backend<PA>& b, const ConvQuadDetails& d, const IOps&... i) {
#if HARNESS_EXECUTORS
    executeOperations<DPP>(b.stream, d, i...);
#else
    require(executeConvQuad<DPP>(b.stream, d, i...), "executeConvQuad");
#endif
}
template <typename DPP, ParArch PA, typename... IOps>
void medianQuad(Backend<PA>& b, const MedianQuadDetails& d, const IOps&... i) {
#if HARNESS_EXECUTORS
    executeOperations<DPP>(b.stream, d, i...);
#else
    require(executeMedianQuad<DPP>(b.stream, d, i...), "executeMedianQuad");
#endif
}
template <typename DPP, ParArch PA, typename... IOps>
void morphQuad(Backend<PA>& b, const MorphQuadDetails& d, const IOps&... i) {
#if HARNESS_EXECUTORS
    executeOperations<DPP>(b.stream, d, i...);
#else
    require(executeMorphQuad<DPP>(b.stream, d, i...), "executeMorphQuad");
#endif
}

// ---- the inputs
template <typename T>
struct Values;
template <>
struct Values<float> {
    static float at(const int x, const int y, const int salt) { return static_cast<float>((x * 5 + y * 7 + salt * 3) % 19 - 9) * 0.125f; }
};
template <>
struct Values<uchar> {
    static uchar at(const int x, const int y, const int salt) { return static_cast<uchar>((x * 31 + y * 17 + salt * 7) % 251); }
};
template <>
struct Values<uchar3> {
    static uchar3 at(const int x, const int y, const int salt) {
        return make_<uchar3>(Values<uchar>::at(x, y, salt), Values<uchar>::at(x + 3, y, salt + 1), Values<uchar>::at(x, y + 5, salt + 2));
    }
};
template <>
struct Values<float3> {
    static float3 at(const int x, const int y, const int salt) {
        return make_<float3>(Values<float>::at(x, y, salt), Values<float>::at(x + 3, y, salt + 1), Values<float>::at(x, y + 5, salt + 2));
    }
};
template <>
struct Values<float4> {
    static float4 at(const int x, const int y, const int salt) {
        return make_<float4>(Values<float>::at(x, y, salt), Values<float>::at(x + 3, y, salt + 1), Values<float>::at(x, y + 5, salt + 2),
                             Values<float>::at(x + 1, y + 1, salt + 3));
    }
};

template <typename T, ParArch PA>
Ptr2D<T> makeImage(const int width, const int height, const int salt) {
    Ptr2D<T> image(static_cast<uint>(width), static_cast<uint>(height), 0, MEMORY<PA>);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) image.at(Point{x, y, 0}) = Values<T>::at(x, y, salt);
    return image;
}
template <typename T, ParArch PA>
Ptr2D<T> makeFilled(const int width, const int height, const T value) {
    Ptr2D<T> image(static_cast<uint>(width), static_cast<uint>(height), 0, MEMORY<PA>);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x) image.at(Point{x, y, 0}) = value;
    return image;
}
template <typename T, ParArch PA>
Tensor<T> makeTensor(const int width, const int height, const int planes, const int salt, const bool filled, const T value) {
    Tensor<T> tensor(static_cast<uint>(width), static_cast<uint>(height), static_cast<uint>(planes), 1, MEMORY<PA>);
    for (int z = 0; z < planes; ++z)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) tensor.at(Point{x, y, z}) = filled ? value : Values<T>::at(x, y, salt + z);
    return tensor;
}

template <typename T>
T sentinel() {
    if constexpr (std::is_same_v<VBase<T>, uchar>) return make_set<T>(static_cast<uchar>(7));
    else return make_set<T>(-999.f);
}

// The frame and its crops of different sizes (two of them at the frame's edges), as FKL's tests/image_processing/test_*_batch.h
constexpr int FRAME_W = 53;
constexpr int FRAME_H = 37;
constexpr int CROPS = 3;
struct CropRect {
    int x, y, w, h;
};
constexpr std::array<CropRect, CROPS> CROP_RECTS{{{0, 0, 21, 13}, {30, 25, 23, 12}, {10, 8, 17, 19}}};
constexpr int MAX_W = 23;
constexpr int MAX_H = 19;

template <typename T, ParArch PA>
std::array<Ptr2D<T>, CROPS> cropsOf(Ptr2D<T>& frame) {
    std::array<Ptr2D<T>, CROPS> crops;
    for (int i = 0; i < CROPS; ++i) {
        const CropRect& r = CROP_RECTS[static_cast<size_t>(i)];
        crops[static_cast<size_t>(i)] =
            frame.crop2D(Point{r.x, r.y, 0}, PtrDims<ND::_2D>(static_cast<uint>(r.w), static_cast<uint>(r.h)));
    }
    return crops;
}

// ---- the outputs: every element of the buffer, row by row, plane by plane
struct Recorder {
    std::ofstream tsv{"outputs.tsv", std::ios::app};
    int cases = 0;

    template <typename T>
    void write(const std::string& backend, const std::string& name, const std::vector<T>& values) {
        const std::string file = backend + "_" + name + ".bin";
        {
            std::ofstream out(file, std::ios::binary);
            out.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(T)));
        }
        const T s = sentinel<T>();
        size_t left = 0;
        for (const T& v : values) left += std::memcmp(&v, &s, sizeof(T)) == 0 ? 1 : 0;
        tsv << backend << "\t" << name << "\t" << values.size() * sizeof(T) << "\t" << fklb::sha256File(file) << "\t" << left << "\t"
            << values.size() << "\n";
        ++cases;
        std::cout << backend << " " << name << ": " << values.size() << " values, " << left << " sentinel" << std::endl;
    }
    template <typename T>
    void record(const std::string& backend, const std::string& name, Ptr2D<T>& p) {
        std::vector<T> v;
        for (int y = 0; y < static_cast<int>(p.dims().height); ++y)
            for (int x = 0; x < static_cast<int>(p.dims().width); ++x) v.push_back(p.at(Point{x, y, 0}));
        write(backend, name, v);
    }
    template <typename T>
    void record(const std::string& backend, const std::string& name, Tensor<T>& t, const int width, const int height, const int planes) {
        std::vector<T> v;
        for (int z = 0; z < planes; ++z)
            for (int y = 0; y < height; ++y)
                for (int x = 0; x < width; ++x) v.push_back(t.at(Point{x, y, z}));
        write(backend, name, v);
    }
};

template <typename T>
using UMul = Mul<T, T, T, UnaryType>;
template <typename T>
using UAdd = Add<T, T, T, UnaryType>;

// ---- LinearFilterDPP
template <ParArch PA>
void linearCases(Recorder& rec) {
    const std::string be = nameOf<PA>();
    {   // float, fused read, coefficients and write
        using D = LinearFilterDPPDetails<float>;
        Backend<PA> b;
        auto in = makeImage<float, PA>(37, 23, 1);
        auto k = makeImage<float, PA>(5, 3, 2);
        auto out = makeFilled<float, PA>(37, 23, sentinel<float>());
        b.upload(in); b.upload(k); b.upload(out);
        const auto read = PerThreadRead<ND::_2D, float>::build(in).then(Mul<float>::build(2.f)).then(Add<float>::build(1.f));
        const auto coefficients = PerThreadRead<ND::_2D, float>::build(k).then(Mul<float>::build(0.5f));
        const auto write = Mul<float>::build(0.25f).then(PerThreadWrite<ND::_2D, float>::build(out));
        linearFilter(b, D{5, 3, 3, 1}, make_tuple(read, coefficients), UMul<float>::build(), UAdd<float>::build(), write);
        b.download(out); b.sync();
        rec.record(be, "linear-float-5x3-fused", out);
    }
    {   // uchar3 to float3, the box filter's constant coefficient read
        using D = LinearFilterDPPDetails<uchar3>;
        Backend<PA> b;
        auto in = makeImage<uchar3, PA>(41, 29, 3);
        auto out = makeFilled<float3, PA>(41, 29, sentinel<float3>());
        b.upload(in); b.upload(out);
        const auto coefficient = ConstantFilterRead<float3>::build(make_set<float3>(1.f / static_cast<float>(3 * 3)));
        linearFilter(b, D{3, 3, 1, 1}, make_tuple(PerThreadRead<ND::_2D, uchar3>::build(in), coefficient),
                     Mul<uchar3, float3, float3, UnaryType>::build(), UAdd<float3>::build(), PerThreadWrite<ND::_2D, float3>::build(out));
        b.download(out); b.sync();
        rec.record(be, "linear-uchar3-box3", out);
    }
    {   // a BatchRead of 3 crops of different sizes into a TensorWrite
        using D = LinearFilterDPPDetails<float>;
        Backend<PA> b;
        auto frame = makeImage<float, PA>(FRAME_W, FRAME_H, 4);
        auto k = makeImage<float, PA>(3, 3, 5);
        auto out = makeTensor<float, PA>(MAX_W, MAX_H, CROPS, 0, true, sentinel<float>());
        b.upload(frame); b.upload(k); b.upload(out);
        auto crops = cropsOf<float, PA>(frame);
        linearFilter(b, D{3, 3, 1, 1}, make_tuple(PerThreadRead<ND::_2D, float>::build(crops), PerThreadRead<ND::_2D, float>::build(k)),
                     UMul<float>::build(), UAdd<float>::build(), TensorWrite<float>::build(out.ptr()));
        b.download(out); b.sync();
        rec.record(be, "linear-batch-crops", out, MAX_W, MAX_H, CROPS);
    }
    {   // a 3D read into a 3D write
        using D = LinearFilterDPPDetails<float>;
        Backend<PA> b;
        auto in = makeTensor<float, PA>(29, 17, 2, 6, false, 0.f);
        auto k = makeImage<float, PA>(3, 5, 7);
        auto out = makeTensor<float, PA>(29, 17, 2, 0, true, sentinel<float>());
        b.upload(in); b.upload(k); b.upload(out);
        linearFilter(b, D{3, 5, 1, 3}, make_tuple(TensorRead<float>::build(in.ptr()), PerThreadRead<ND::_2D, float>::build(k)),
                     UMul<float>::build(), UAdd<float>::build(), TensorWrite<float>::build(out.ptr()));
        b.download(out); b.sync();
        rec.record(be, "linear-3d", out, 29, 17, 2);
    }
}

// ---- MedianFilterDPP
template <ParArch PA>
void medianCases(Recorder& rec) {
    const std::string be = nameOf<PA>();
    {
        using D = MedianFilterDPPDetails<float>;
        Backend<PA> b;
        auto in = makeImage<float, PA>(37, 19, 8);
        auto out = makeFilled<float, PA>(37, 19, sentinel<float>());
        b.upload(in); b.upload(out);
        medianFilter(b, D{5, 5, 2, 2}, PerThreadRead<ND::_2D, float>::build(in), PerThreadWrite<ND::_2D, float>::build(out),
                     MedianWindowSelect<float, D::WINDOW_CAPACITY>::build());
        b.download(out); b.sync();
        rec.record(be, "median-float-5x5", out);
    }
    {
        using D = MedianFilterDPPDetails<float>;
        Backend<PA> b;
        auto in = makeImage<float, PA>(31, 15, 9);
        auto out = makeFilled<float, PA>(31, 15, sentinel<float>());
        b.upload(in); b.upload(out);
        medianFilter(b, D{3, 5, 0, 3}, PerThreadRead<ND::_2D, float>::build(in).then(Mul<float>::build(2.f)).then(Add<float>::build(1.f)),
                     Mul<float>::build(0.5f).then(PerThreadWrite<ND::_2D, float>::build(out)),
                     MinimumWindowSelect<float, D::WINDOW_CAPACITY>::build());
        b.download(out); b.sync();
        rec.record(be, "median-float-min-3x5-fused", out);
    }
    {
        using D = MedianFilterDPPDetails<uchar3>;
        Backend<PA> b;
        auto frame = makeImage<uchar3, PA>(FRAME_W, FRAME_H, 10);
        auto out = makeTensor<uchar3, PA>(MAX_W, MAX_H, CROPS, 0, true, sentinel<uchar3>());
        b.upload(frame); b.upload(out);
        auto crops = cropsOf<uchar3, PA>(frame);
        medianFilter(b, D{3, 3, 1, 1}, PerThreadRead<ND::_2D, uchar3>::build(crops), TensorWrite<uchar3>::build(out.ptr()),
                     MedianWindowSelect<uchar3, D::WINDOW_CAPACITY>::build());
        b.download(out); b.sync();
        rec.record(be, "median-batch-crops-uchar3", out, MAX_W, MAX_H, CROPS);
    }
}

// ---- MorphologyDPP
template <ParArch PA>
void morphologyCases(Recorder& rec) {
    const std::string be = nameOf<PA>();
    {
        using D = MorphologyDPPDetails<float>;
        Backend<PA> b;
        auto in = makeImage<float, PA>(37, 23, 11);
        auto out = makeFilled<float, PA>(37, 23, sentinel<float>());
        b.upload(in); b.upload(out);
        morphology(b, D{37, 23, 5, 3, 2, 1}, PerThreadRead<ND::_2D, float>::build(in), PerThreadWrite<ND::_2D, float>::build(out),
                   make_tuple(Min<float, float, float, UnaryType>::build()));
        b.download(out); b.sync();
        rec.record(be, "morph-erode-float-5x3", out);
    }
    {
        using D = MorphologyDPPDetails<uchar>;
        Backend<PA> b;
        auto in = makeImage<uchar, PA>(41, 29, 12);
        auto out = makeFilled<uchar, PA>(41, 29, sentinel<uchar>());
        b.upload(in); b.upload(out);
        morphology(b, D{41, 29, 3, 5, 1, 2}, PerThreadRead<ND::_2D, uchar>::build(in), PerThreadWrite<ND::_2D, uchar>::build(out),
                   make_tuple(Min<uchar, uchar, uchar, UnaryType>::build(), Max<uchar, uchar, uchar, UnaryType>::build()));
        b.download(out); b.sync();
        rec.record(be, "morph-open-uchar-3x5", out);
    }
    {
        using D = MorphologyDPPDetails<float>;
        Backend<PA> b;
        auto in = makeImage<float, PA>(37, 23, 13);
        auto out = makeFilled<float, PA>(37, 23, sentinel<float>());
        b.upload(in); b.upload(out);
        const D d{37, 23, 3, 3, 1, 1};
        morphology(b, d, PerThreadRead<ND::_2D, float>::build(in), PerThreadWrite<ND::_2D, float>::build(out), boxBlur<float, 2>(d));
        b.download(out); b.sync();
        rec.record(be, "morph-boxblur2-float", out);
    }
    {   // E19's fkl_batch form: Crop of 3 rectangles, Resize to one size, a box blur of one pass, into a TensorWrite
        using D = MorphologyDPPDetails<float>;
        constexpr int W = 24, H = 16;
        Backend<PA> b;
        auto source = makeImage<float, PA>(97, 61, 14);
        auto out = makeTensor<float, PA>(W, H, 3, 0, true, sentinel<float>());
        b.upload(source); b.upload(out);
        constexpr std::array<Rect, 3> rects{Rect(3, 5, 20, 10), Rect(40, 20, 32, 16), Rect(60, 30, 37, 31)};
        const auto read = PerThreadRead<ND::_2D, float>::build(source)
                              .then(Crop<>::build(rects))
                              .then(Resize<InterpolationType::INTER_LINEAR, AspectRatio::IGNORE_AR>::build(Size(W, H)));
        const D d{W, H, 3, 3, 1, 1};
        morphology(b, d, read, TensorWrite<float>::build(out.ptr()), boxBlur<float, 1>(d));
        b.download(out); b.sync();
        rec.record(be, "morph-batch-crop-resize-boxblur1", out, W, H, 3);
    }
    {
        using D = MorphologyDPPDetails<float3>;
        Backend<PA> b;
        auto in = makeImage<float3, PA>(23, 13, 15);
        auto out = makeFilled<float3, PA>(23, 13, sentinel<float3>());
        b.upload(in); b.upload(out);
        const D d{23, 13, 5, 5, 2, 2};
        morphology(b, d, PerThreadRead<ND::_2D, float3>::build(in), PerThreadWrite<ND::_2D, float3>::build(out), boxBlur<float3, 1>(d));
        b.download(out); b.sync();
        rec.record(be, "morph-boxblur1-float3", out);
    }
}

// ---- the Quad DPPs
template <ParArch PA>
void quadCases(Recorder& rec) {
    const std::string be = nameOf<PA>();
    {
        Backend<PA> b;
        auto in = makeImage<float, PA>(61, 35, 16);
        auto out = makeFilled<float, PA>(61, 35, sentinel<float>());
        b.upload(in); b.upload(out);
        boxQuad<BoxFilterQuadDPP<PA, float, 4, 4, 3, 3>>(b, BoxFilterQuadDetails{3, 3, 1, 1}, PerThreadRead<ND::_2D, float>::build(in),
            make_tuple(UAdd<float>::build(), Sub<float, float, float, UnaryType>::build(), Div<float>::build(9.f)),
            PerThreadWrite<ND::_2D, float>::build(out));
        b.download(out); b.sync();
        rec.record(be, "boxquad-float-3x3", out);
    }
    {
        Backend<PA> b;
        auto frame = makeImage<float3, PA>(FRAME_W, FRAME_H, 17);
        auto out = makeTensor<float3, PA>(MAX_W, MAX_H, CROPS, 0, true, sentinel<float3>());
        b.upload(frame); b.upload(out);
        auto crops = cropsOf<float3, PA>(frame);
        boxQuad<BoxFilterQuadDPP<PA, float3>>(b, BoxFilterQuadDetails{5, 3, 2, 1}, PerThreadRead<ND::_2D, float3>::build(crops),
            make_tuple(UAdd<float3>::build(), Sub<float3, float3, float3, UnaryType>::build(), Div<float3>::build(make_set<float3>(15.f))),
            TensorWrite<float3>::build(out.ptr()));
        b.download(out); b.sync();
        rec.record(be, "boxquad-batch-float3-5x3", out, MAX_W, MAX_H, CROPS);
    }
    {
        Backend<PA> b;
        auto in = makeImage<float, PA>(65, 37, 18);
        auto out = makeFilled<float, PA>(65, 37, sentinel<float>());
        b.upload(in); b.upload(out);
        ConvQuadDetails d{5, 5, 2, 2, {}};
        for (int i = 0; i < 25; ++i) d.coefficients[i] = static_cast<float>((i * 7) % 11 - 5) * 0.0625f;
        convQuad<ConvQuadDPP<PA, float, 4, 4, 5, 5>>(b, d, PerThreadRead<ND::_2D, float>::build(in).then(Add<float>::build(0.25f)),
            make_tuple(UMul<float>::build(), UAdd<float>::build()), Add<float>::build(-0.5f).then(PerThreadWrite<ND::_2D, float>::build(out)));
        b.download(out); b.sync();
        rec.record(be, "convquad-float-5x5", out);
    }
    {
        Backend<PA> b;
        auto frame = makeImage<uchar3, PA>(FRAME_W, FRAME_H, 19);
        auto out = makeTensor<float3, PA>(MAX_W, MAX_H, CROPS, 0, true, sentinel<float3>());
        b.upload(frame); b.upload(out);
        auto crops = cropsOf<uchar3, PA>(frame);
        ConvQuadDetails d{3, 3, 1, 1, {}};
        for (int i = 0; i < 9; ++i) d.coefficients[i] = static_cast<float>((i * 5) % 7 - 3) * 0.125f;
        convQuad<ConvQuadDPP<PA, uchar3>>(b, d, PerThreadRead<ND::_2D, uchar3>::build(crops),
            make_tuple(Mul<float, uchar3, float3, UnaryType>::build(), UAdd<float3>::build()), TensorWrite<float3>::build(out.ptr()));
        b.download(out); b.sync();
        rec.record(be, "convquad-batch-uchar3-3x3", out, MAX_W, MAX_H, CROPS);
    }
    {
        Backend<PA> b;
        auto in = makeImage<uchar, PA>(57, 33, 20);
        auto out = makeFilled<uchar, PA>(57, 33, sentinel<uchar>());
        b.upload(in); b.upload(out);
        medianQuad<MedianQuadDPP<PA, uchar, 4, 4, 3, 3>>(b, MedianQuadDetails{3, 3, 1, 1}, PerThreadRead<ND::_2D, uchar>::build(in),
            make_tuple(Min<uchar, uchar, uchar, UnaryType>::build(), Max<uchar, uchar, uchar, UnaryType>::build()),
            PerThreadWrite<ND::_2D, uchar>::build(out));
        b.download(out); b.sync();
        rec.record(be, "medianquad-uchar-3x3", out);
    }
    {
        Backend<PA> b;
        auto frame = makeImage<float3, PA>(FRAME_W, FRAME_H, 21);
        auto out = makeTensor<float3, PA>(MAX_W, MAX_H, CROPS, 0, true, sentinel<float3>());
        b.upload(frame); b.upload(out);
        auto crops = cropsOf<float3, PA>(frame);
        medianQuad<MedianQuadDPP<PA, float3>>(b, MedianQuadDetails{3, 3, 1, 1}, PerThreadRead<ND::_2D, float3>::build(crops),
            make_tuple(Min<float3, float3, float3, UnaryType>::build(), Max<float3, float3, float3, UnaryType>::build()),
            TensorWrite<float3>::build(out.ptr()));
        b.download(out); b.sync();
        rec.record(be, "medianquad-batch-float3-3x3", out, MAX_W, MAX_H, CROPS);
    }
    {
        Backend<PA> b;
        auto in = makeImage<uchar, PA>(59, 31, 22);
        auto out = makeFilled<uchar, PA>(59, 31, sentinel<uchar>());
        b.upload(in); b.upload(out);
        morphQuad<MorphQuadDPP<PA, uchar, 4, 4, 3, 3>>(b, MorphQuadDetails{3, 3, 1, 1}, PerThreadRead<ND::_2D, uchar>::build(in),
            Min<uchar, uchar, uchar, UnaryType>::build(), PerThreadWrite<ND::_2D, uchar>::build(out));
        b.download(out); b.sync();
        rec.record(be, "morphquad-uchar-3x3", out);
    }
    {
        Backend<PA> b;
        auto frame = makeImage<float4, PA>(FRAME_W, FRAME_H, 23);
        auto out = makeTensor<float4, PA>(MAX_W, MAX_H, CROPS, 0, true, sentinel<float4>());
        b.upload(frame); b.upload(out);
        auto crops = cropsOf<float4, PA>(frame);
        morphQuad<MorphQuadDPP<PA, float4>>(b, MorphQuadDetails{5, 5, 2, 2}, PerThreadRead<ND::_2D, float4>::build(crops),
            Max<float4, float4, float4, UnaryType>::build(), TensorWrite<float4>::build(out.ptr()));
        b.download(out); b.sync();
        rec.record(be, "morphquad-batch-float4-5x5", out, MAX_W, MAX_H, CROPS);
    }
}

// ---- TransformDPP (the same call in both FKL forms; its kernel is renamed by #435)
template <ParArch PA>
void transformCases(Recorder& rec) {
    const std::string be = nameOf<PA>();
    {
        Backend<PA> b;
        auto in = makeImage<uchar3, PA>(45, 27, 24);
        auto out = makeFilled<float3, PA>(45, 27, sentinel<float3>());
        b.upload(in); b.upload(out);
        executeOperations<TransformDPP<PA>>(b.stream, PerThreadRead<ND::_2D, uchar3>::build(in), Cast<uchar3, float3>::build(),
                                            Mul<float3>::build(make_set<float3>(0.5f)), PerThreadWrite<ND::_2D, float3>::build(out));
        b.download(out); b.sync();
        rec.record(be, "transform-uchar3-float3", out);
    }
    // thread fusion: a width it divides, one it does not divide, and a misaligned read (the non thread fused fallback)
    for (int which = 0; which < 3; ++which) {
        Backend<PA> b;
        const int width = which == 0 ? 128 : 67;
        auto in = makeImage<float, PA>(which == 2 ? 70 : width, 16, 25 + which);
        auto out = makeFilled<float, PA>(width, 16, sentinel<float>());
        b.upload(in); b.upload(out);
        Ptr2D<float> source = which == 2 ? in.crop2D(Point{1, 0, 0}, PtrDims<ND::_2D>(static_cast<uint>(width), 16u)) : in;
        executeOperations<TransformDPP<PA, TF::ENABLED>>(b.stream, PerThreadRead<ND::_2D, float>::build(source), Mul<float>::build(3.f),
                                                        PerThreadWrite<ND::_2D, float>::build(out));
        b.download(out); b.sync();
        rec.record(be, which == 0 ? "transform-tf-divisible" : (which == 1 ? "transform-tf-not-divisible" : "transform-tf-misaligned"), out);
    }
}

struct TwoPlanes {
    FK_HOST_DEVICE_FUSE uint at(const uint& index) { return index == 0 ? 0u : 1u; }
};

#if defined(FK_HIP_BACKEND)
// ---- DivergentBatchTransformDPP (GPU only: LTS has no CPU executor for it)
void divergentCase(Recorder& rec) {
    constexpr ParArch PA = ParArch::GPU_AMD;
    Backend<PA> b;
    auto a = makeImage<float, PA>(29, 11, 30);
    auto c = makeImage<float, PA>(29, 11, 31);
    auto out = makeTensor<float, PA>(29, 11, 2, 0, true, sentinel<float>());
    b.upload(a); b.upload(c); b.upload(out);
    const auto seq1 = buildOperationSequence(PerThreadRead<ND::_2D, float>::build(a), Mul<float>::build(2.f), TensorWrite<float>::build(out));
    const auto seq2 = buildOperationSequence(PerThreadRead<ND::_2D, float>::build(c), Add<float>::build(100.f), TensorWrite<float>::build(out));
    Executor<DivergentBatchTransformDPP<PA, TwoPlanes>>::executeOperations(b.stream, seq1, seq2);
    b.download(out); b.sync();
    rec.record(nameOf<PA>(), "divergent-two-sequences", out, 29, 11, 2);
}
#endif

template <ParArch PA>
void allCases(Recorder& rec) {
    linearCases<PA>(rec);
    medianCases<PA>(rec);
    morphologyCases<PA>(rec);
    quadCases<PA>(rec);
    transformCases<PA>(rec);
}

}  // namespace

int main() {
    std::cout << "dpps_before_after: FKL " << HARNESS_FKL_COMMIT << ", " << (HARNESS_EXECUTORS ? "executeOperations<DPP>" : "entry functions")
              << std::endl;
    try {
        Recorder rec;
        allCases<ParArch::CPU>(rec);
#ifdef FK_CPU_MULTITHREAD_ENABLED
        allCases<ParArch::CPU_MULTITHREAD>(rec);
#endif
#if defined(FK_HIP_BACKEND)
        const char* gpu = std::getenv("HARNESS_GPU");
        if (gpu != nullptr && std::string(gpu) == "1") {
            allCases<ParArch::GPU_AMD>(rec);
            divergentCase(rec);
        }
#endif
        std::cout << "dpps_before_after: " << rec.cases << " outputs written" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "dpps_before_after: " << e.what() << std::endl;
        return 2;
    }
    return 0;
}
