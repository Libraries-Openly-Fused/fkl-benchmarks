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

// FKL main's DPPs before and after FKL #436 (task C ported to main: every DPP through executeOperations<DPP> and launchDPP_Kernel), on every
// back-end of this machine. It is fkl-benchmarks' results/rx9060/fkl-pr435-check/harness/dpps_before_after.cpp (the LTS-C++17 version),
// adapted to main: main's filter details carry the image size (main has no batching of LinearFilterDPP, MedianFilterDPP and the Quad DPPs, so
// their batched and 3D cases are 2D cases here), main's Quad entry functions return void, and main has CPU executors for
// DivergentBatchTransformDPP. The same source is built by build.sh against FKL main ebb3891 (HARNESS_EXECUTORS=0: the entry functions of that
// tree) and against the #436 head (HARNESS_EXECUTORS=1: executeOperations<DPP>); the calls of the two forms are in the adaptors below, nothing
// else differs. Every case
// runs once per back-end on fixed inputs, its output buffer (pre-filled with a sentinel, all of it, so a value written outside the image also
// shows) is written to <back-end>_<case>.bin in the working directory, and outputs.tsv gets one line per case: back-end, case, bytes, sha256,
// sentinel values left, values. The two builds' files are compared byte for byte afterwards (compare.sh); nothing here compares or times anything.
// The cases:
// - LinearFilterDPP: float with fused read, coefficients and write (5x3, anchor 3,1); float3 box 3x3 (the coefficient read of the removed
//   executeBoxFilter, with scale 1); float 3x5 with a plain coefficient read (anchor 1,3).
// - MedianFilterDPP: float median 5x5; float minimum 3x5 with a fused read and write; uchar3 median 3x3.
// - MorphologyDPP: float erosion 5x3; uchar opening 3x5; float box blur of 2 passes; E19's fkl_batch form (Crop of 3 rectangles, Resize, box
//   blur of 1 pass, TensorWrite); float3 box blur of 1 pass.
// - The Quad DPPs, each a 2D case with static kernel sizes and one with run-time sizes: BoxFilterQuadDPP (float 3x3; float 5x3),
//   ConvQuadDPP (float 5x5; float 3x3), MedianQuadDPP (uchar 3x3; float 3x3), MorphQuadDPP (uchar 3x3; float 5x5).
// - TransformDPP: uchar3 to float3 (no thread fusion); thread fusion with a width it divides, with one it does not, and with a misaligned
//   read (the non thread fused fallback): the three kernels of its run-time choice.
// - DivergentBatchTransformDPP: two sequences into two planes, on every back-end (main has its CPU and CPU_MULTITHREAD executors).
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
#error "build.sh sets HARNESS_EXECUTORS: 1 for an FKL tree with the executors of every DPP (#436), 0 for one with the entry functions"
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
    executeBoxFilterQuad<DPP>(b.stream, d, i...);   // main's Quad entry functions return void
#endif
}
template <typename DPP, ParArch PA, typename... IOps>
void convQuad(Backend<PA>& b, const ConvQuadDetails& d, const IOps&... i) {
#if HARNESS_EXECUTORS
    executeOperations<DPP>(b.stream, d, i...);
#else
    executeConvQuad<DPP>(b.stream, d, i...);   // main's Quad entry functions return void
#endif
}
template <typename DPP, ParArch PA, typename... IOps>
void medianQuad(Backend<PA>& b, const MedianQuadDetails& d, const IOps&... i) {
#if HARNESS_EXECUTORS
    executeOperations<DPP>(b.stream, d, i...);
#else
    executeMedianQuad<DPP>(b.stream, d, i...);   // main's Quad entry functions return void
#endif
}
template <typename DPP, ParArch PA, typename... IOps>
void morphQuad(Backend<PA>& b, const MorphQuadDetails& d, const IOps&... i) {
#if HARNESS_EXECUTORS
    executeOperations<DPP>(b.stream, d, i...);
#else
    executeMorphQuad<DPP>(b.stream, d, i...);   // main's Quad entry functions return void
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

// ---- LinearFilterDPP (main's details: width, height, kernel size, anchor)
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
        linearFilter(b, D{37, 23, 5, 3, 3, 1}, make_tuple(read, coefficients), UMul<float>::build(), UAdd<float>::build(), write);
        b.download(out); b.sync();
        rec.record(be, "linear-float-5x3-fused", out);
    }
    {   // float3, the box filter's constant coefficient read (what the removed executeBoxFilter built, with scale 1)
        using D = LinearFilterDPPDetails<float3>;
        Backend<PA> b;
        auto in = makeImage<float3, PA>(41, 29, 3);
        auto out = makeFilled<float3, PA>(41, 29, sentinel<float3>());
        b.upload(in); b.upload(out);
        const auto coefficient = ConstantFilterRead<float3>::build(make_set<float3>(1.f / static_cast<float>(3 * 3)));
        linearFilter(b, D{41, 29, 3, 3, 1, 1}, make_tuple(PerThreadRead<ND::_2D, float3>::build(in), coefficient),
                     UMul<float3>::build(), UAdd<float3>::build(), PerThreadWrite<ND::_2D, float3>::build(out));
        b.download(out); b.sync();
        rec.record(be, "linear-float3-box3", out);
    }
    {   // float 3x5, a plain coefficient read
        using D = LinearFilterDPPDetails<float>;
        Backend<PA> b;
        auto in = makeImage<float, PA>(29, 17, 6);
        auto k = makeImage<float, PA>(3, 5, 7);
        auto out = makeFilled<float, PA>(29, 17, sentinel<float>());
        b.upload(in); b.upload(k); b.upload(out);
        linearFilter(b, D{29, 17, 3, 5, 1, 3}, make_tuple(PerThreadRead<ND::_2D, float>::build(in), PerThreadRead<ND::_2D, float>::build(k)),
                     UMul<float>::build(), UAdd<float>::build(), PerThreadWrite<ND::_2D, float>::build(out));
        b.download(out); b.sync();
        rec.record(be, "linear-float-3x5", out);
    }
}

// ---- MedianFilterDPP (main's details: width, height, window size, anchor)
template <ParArch PA>
void medianCases(Recorder& rec) {
    const std::string be = nameOf<PA>();
    {
        using D = MedianFilterDPPDetails<float>;
        Backend<PA> b;
        auto in = makeImage<float, PA>(37, 19, 8);
        auto out = makeFilled<float, PA>(37, 19, sentinel<float>());
        b.upload(in); b.upload(out);
        medianFilter(b, D{37, 19, 5, 5, 2, 2}, PerThreadRead<ND::_2D, float>::build(in), PerThreadWrite<ND::_2D, float>::build(out),
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
        medianFilter(b, D{31, 15, 3, 5, 0, 3}, PerThreadRead<ND::_2D, float>::build(in).then(Mul<float>::build(2.f)).then(Add<float>::build(1.f)),
                     Mul<float>::build(0.5f).then(PerThreadWrite<ND::_2D, float>::build(out)),
                     MinimumWindowSelect<float, D::WINDOW_CAPACITY>::build());
        b.download(out); b.sync();
        rec.record(be, "median-float-min-3x5-fused", out);
    }
    {
        using D = MedianFilterDPPDetails<uchar3>;
        Backend<PA> b;
        auto in = makeImage<uchar3, PA>(33, 21, 10);
        auto out = makeFilled<uchar3, PA>(33, 21, sentinel<uchar3>());
        b.upload(in); b.upload(out);
        medianFilter(b, D{33, 21, 3, 3, 1, 1}, PerThreadRead<ND::_2D, uchar3>::build(in), PerThreadWrite<ND::_2D, uchar3>::build(out),
                     MedianWindowSelect<uchar3, D::WINDOW_CAPACITY>::build());
        b.download(out); b.sync();
        rec.record(be, "median-uchar3-3x3", out);
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

// ---- the Quad DPPs (main's details: width, height, kernel size, anchor; scalar types only), each with static and with run-time kernel sizes
template <ParArch PA>
void quadCases(Recorder& rec) {
    const std::string be = nameOf<PA>();
    {
        Backend<PA> b;
        auto in = makeImage<float, PA>(61, 35, 16);
        auto out = makeFilled<float, PA>(61, 35, sentinel<float>());
        b.upload(in); b.upload(out);
        boxQuad<BoxFilterQuadDPP<PA, float, 4, 4, 3, 3>>(b, BoxFilterQuadDetails{61, 35, 3, 3, 1, 1}, PerThreadRead<ND::_2D, float>::build(in),
            make_tuple(UAdd<float>::build(), Sub<float, float, float, UnaryType>::build(), Div<float>::build(9.f)),
            PerThreadWrite<ND::_2D, float>::build(out));
        b.download(out); b.sync();
        rec.record(be, "boxquad-float-3x3", out);
    }
    {
        Backend<PA> b;
        auto in = makeImage<float, PA>(53, 37, 17);
        auto out = makeFilled<float, PA>(53, 37, sentinel<float>());
        b.upload(in); b.upload(out);
        boxQuad<BoxFilterQuadDPP<PA, float>>(b, BoxFilterQuadDetails{53, 37, 5, 3, 2, 1}, PerThreadRead<ND::_2D, float>::build(in),
            make_tuple(UAdd<float>::build(), Sub<float, float, float, UnaryType>::build(), Div<float>::build(15.f)),
            PerThreadWrite<ND::_2D, float>::build(out));
        b.download(out); b.sync();
        rec.record(be, "boxquad-float-runtime-5x3", out);
    }
    {
        Backend<PA> b;
        auto in = makeImage<float, PA>(65, 37, 18);
        auto out = makeFilled<float, PA>(65, 37, sentinel<float>());
        b.upload(in); b.upload(out);
        ConvQuadDetails d{65, 37, 5, 5, 2, 2, {}};
        for (int i = 0; i < 25; ++i) d.coefficients[i] = static_cast<float>((i * 7) % 11 - 5) * 0.0625f;
        convQuad<ConvQuadDPP<PA, float, 4, 4, 5, 5>>(b, d, PerThreadRead<ND::_2D, float>::build(in).then(Add<float>::build(0.25f)),
            make_tuple(UMul<float>::build(), UAdd<float>::build()), Add<float>::build(-0.5f).then(PerThreadWrite<ND::_2D, float>::build(out)));
        b.download(out); b.sync();
        rec.record(be, "convquad-float-5x5", out);
    }
    {
        Backend<PA> b;
        auto in = makeImage<float, PA>(53, 37, 19);
        auto out = makeFilled<float, PA>(53, 37, sentinel<float>());
        b.upload(in); b.upload(out);
        ConvQuadDetails d{53, 37, 3, 3, 1, 1, {}};
        for (int i = 0; i < 9; ++i) d.coefficients[i] = static_cast<float>((i * 5) % 7 - 3) * 0.125f;
        convQuad<ConvQuadDPP<PA, float>>(b, d, PerThreadRead<ND::_2D, float>::build(in),
            make_tuple(UMul<float>::build(), UAdd<float>::build()), PerThreadWrite<ND::_2D, float>::build(out));
        b.download(out); b.sync();
        rec.record(be, "convquad-float-runtime-3x3", out);
    }
    {
        Backend<PA> b;
        auto in = makeImage<uchar, PA>(57, 33, 20);
        auto out = makeFilled<uchar, PA>(57, 33, sentinel<uchar>());
        b.upload(in); b.upload(out);
        medianQuad<MedianQuadDPP<PA, uchar, 4, 4, 3, 3>>(b, MedianQuadDetails{57, 33, 3, 3, 1, 1}, PerThreadRead<ND::_2D, uchar>::build(in),
            make_tuple(Min<uchar, uchar, uchar, UnaryType>::build(), Max<uchar, uchar, uchar, UnaryType>::build()),
            PerThreadWrite<ND::_2D, uchar>::build(out));
        b.download(out); b.sync();
        rec.record(be, "medianquad-uchar-3x3", out);
    }
    {
        Backend<PA> b;
        auto in = makeImage<float, PA>(53, 37, 21);
        auto out = makeFilled<float, PA>(53, 37, sentinel<float>());
        b.upload(in); b.upload(out);
        medianQuad<MedianQuadDPP<PA, float>>(b, MedianQuadDetails{53, 37, 3, 3, 1, 1}, PerThreadRead<ND::_2D, float>::build(in),
            make_tuple(Min<float, float, float, UnaryType>::build(), Max<float, float, float, UnaryType>::build()),
            PerThreadWrite<ND::_2D, float>::build(out));
        b.download(out); b.sync();
        rec.record(be, "medianquad-float-runtime-3x3", out);
    }
    {
        Backend<PA> b;
        auto in = makeImage<uchar, PA>(59, 31, 22);
        auto out = makeFilled<uchar, PA>(59, 31, sentinel<uchar>());
        b.upload(in); b.upload(out);
        morphQuad<MorphQuadDPP<PA, uchar, 4, 4, 3, 3>>(b, MorphQuadDetails{59, 31, 3, 3, 1, 1}, PerThreadRead<ND::_2D, uchar>::build(in),
            Min<uchar, uchar, uchar, UnaryType>::build(), PerThreadWrite<ND::_2D, uchar>::build(out));
        b.download(out); b.sync();
        rec.record(be, "morphquad-uchar-3x3", out);
    }
    {
        Backend<PA> b;
        auto in = makeImage<float, PA>(53, 37, 23);
        auto out = makeFilled<float, PA>(53, 37, sentinel<float>());
        b.upload(in); b.upload(out);
        morphQuad<MorphQuadDPP<PA, float>>(b, MorphQuadDetails{53, 37, 5, 5, 2, 2}, PerThreadRead<ND::_2D, float>::build(in),
            Max<float, float, float, UnaryType>::build(), PerThreadWrite<ND::_2D, float>::build(out));
        b.download(out); b.sync();
        rec.record(be, "morphquad-float-runtime-5x5", out);
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

// ---- DivergentBatchTransformDPP (main has its executors on every back-end)
template <ParArch PA>
void divergentCase(Recorder& rec) {
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

template <ParArch PA>
void allCases(Recorder& rec) {
    linearCases<PA>(rec);
    medianCases<PA>(rec);
    morphologyCases<PA>(rec);
    quadCases<PA>(rec);
    transformCases<PA>(rec);
    divergentCase<PA>(rec);
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
#if defined(__HIPCC__)
        const char* gpu = std::getenv("HARNESS_GPU");
        if (gpu != nullptr && std::string(gpu) == "1") {
            allCases<ParArch::GPU_AMD>(rec);
        }
#endif
        std::cout << "dpps_before_after: " << rec.cases << " outputs written" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "dpps_before_after: " << e.what() << std::endl;
        return 2;
    }
    return 0;
}
