/* Task C-NV harness, image DPPs: LinearFilterDPP, MedianFilterDPP, MorphologyDPP (single images and batches, the linear
   and median batches with per-image sizes), the four Quad DPPs (BoxFilterQuad, ConvQuad, MedianQuad, MorphQuad: batches of
   crops of different sizes and single vector images), TransformDPP and DivergentBatchTransformDPP, all on ParArch::GPU_NVIDIA.
   The base build calls the entry functions of b10fc66 (executeLinearFilter, executeBoxFilter, executeMedianFilter,
   executeErode/Dilate/Open/Close, executeBoxBlur, executeMorphology, execute{BoxFilter,Conv,Median,Morph}Quad<DPP>); the
   head build calls executeOperations<DPP>(stream, details, iOps...) of e322cad, with the arguments the PR says the callers
   build for the removed convenience functions (ConstantFilterRead of 1 / (kw * kh) with Mul and Add for executeBoxFilter;
   make_tuple(Min / Max ...) for erode, dilate, open, close; fk::boxBlurReducers<T, PASSES>(details) for executeBoxBlur).
   TransformDPP and DivergentBatchTransformDPP have the same call at both commits. See pr435_common.h. */
#include "pr435_common.h"

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/box_filter_fast.h>
#include <fused_kernel/algorithms/image_processing/convolution_fast.h>
#include <fused_kernel/algorithms/image_processing/linear_filter.h>
#include <fused_kernel/algorithms/image_processing/median_fast.h>
#include <fused_kernel/algorithms/image_processing/median_filter.h>
#include <fused_kernel/algorithms/image_processing/morphology.h>
#include <fused_kernel/algorithms/image_processing/morphology_fast.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <array>
#include <type_traits>

using namespace fk;
using namespace h435;

namespace {

constexpr ParArch GPU = ParArch::GPU_NVIDIA;
constexpr MemType MEM = MemType::DeviceAndPinned;
using GpuStream = Stream_<ParArch::GPU_NVIDIA>;

constexpr int FW = 61, FH = 43;
struct Rc { int x, y, w, h; };
// three crops of different sizes, two of them on the frame's edges (the linear and median batches, the Quad batches)
constexpr std::array<Rc, 3> VAR_CROPS{{{0, 0, 21, 13}, {38, 29, 23, 14}, {12, 9, 17, 19}}};
constexpr int VAR_MAX_W = 23, VAR_MAX_H = 19;
// three crops of one size (the morphology batches: MorphologyDPPDetails has one width and height)
constexpr std::array<Rc, 3> EQ_CROPS{{{0, 0, 20, 14}, {41, 29, 20, 14}, {12, 9, 20, 14}}};

template <typename T>
T pixel(const int x, const int y, const int salt) {
    using S = VBase<T>;
    const auto ch = [&](const int c) -> S {
        if constexpr (std::is_floating_point_v<S>) {
            return static_cast<S>(fval(x, y, salt * 8 + c));
        } else {
            return static_cast<S>(uval(x, y, salt * 8 + c));
        }
    };
    if constexpr (cn<T> == 1) {
        return ch(0);
    } else if constexpr (cn<T> == 3) {
        return make_<T>(ch(0), ch(1), ch(2));
    } else {
        return make_<T>(ch(0), ch(1), ch(2), ch(3));
    }
}

template <typename T>
Ptr2D<T> frame(GpuStream& s, const int salt, const int w = FW, const int h = FH) {
    Ptr2D<T> p(static_cast<uint>(w), static_cast<uint>(h), 0, MEM);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) p.at(Point{x, y, 0}) = pixel<T>(x, y, salt);
    p.upload(s);
    return p;
}
template <typename T>
Ptr2D<T> filled(GpuStream& s, const int w, const int h) {
    Ptr2D<T> p(static_cast<uint>(w), static_cast<uint>(h), 0, MEM);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) std::memset(&p.at(Point{x, y, 0}), 0xff, sizeof(T));
    p.upload(s);
    return p;
}
template <typename T>
Tensor<T> tensorFilled(GpuStream& s, const int w, const int h, const int planes) {
    Tensor<T> t(static_cast<uint>(w), static_cast<uint>(h), static_cast<uint>(planes), 1, MEM);
    for (int z = 0; z < planes; ++z)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) std::memset(&t.at(Point{x, y, z}), 0xff, sizeof(T));
    t.upload(s);
    return t;
}
template <typename T>
std::array<Ptr2D<T>, 3> cropsOf(const Ptr2D<T>& f, const std::array<Rc, 3>& r) {
    std::array<Ptr2D<T>, 3> c;
    for (size_t i = 0; i < 3; ++i)
        c[i] = f.crop2D(Point{r[i].x, r[i].y, 0}, PtrDims<ND::_2D>(static_cast<uint>(r[i].w), static_cast<uint>(r[i].h)));
    return c;
}
template <typename T>
std::array<Ptr2D<T>, 3> filledOutputs(GpuStream& s, const std::array<Rc, 3>& r) {
    std::array<Ptr2D<T>, 3> o;
    for (size_t i = 0; i < 3; ++i) o[i] = filled<T>(s, r[i].w, r[i].h);
    return o;
}
// downloads and writes every element in row order (no pitch padding)
template <typename T>
void dump(GpuStream& s, const std::string& name, Ptr2D<T>& p) {
    p.download(s);
    s.sync();
    const int w = static_cast<int>(p.dims().width), h = static_cast<int>(p.dims().height);
    std::vector<T> v(static_cast<size_t>(w) * h);
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) v[static_cast<size_t>(y) * w + x] = p.at(Point{x, y, 0});
    writeBin(name, v.data(), v.size() * sizeof(T));
}
template <typename T>
void dump(GpuStream& s, const std::string& name, Tensor<T>& t, const int w, const int h, const int planes) {
    t.download(s);
    s.sync();
    std::vector<T> v(static_cast<size_t>(w) * h * planes);
    for (int z = 0; z < planes; ++z)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) v[(static_cast<size_t>(z) * h + y) * w + x] = t.at(Point{x, y, z});
    writeBin(name, v.data(), v.size() * sizeof(T));
}
template <typename T>
void dump(GpuStream& s, const std::string& name, std::array<Ptr2D<T>, 3>& outs) {
    for (size_t i = 0; i < 3; ++i) dump(s, name + "_image" + std::to_string(i), outs[i]);
}
template <typename T>
Ptr2D<T> coefficients(GpuStream& s, const int kw, const int kh, const int salt) {
    Ptr2D<T> k(static_cast<uint>(kw), static_cast<uint>(kh), 0, MEM);
    for (int y = 0; y < kh; ++y)
        for (int x = 0; x < kw; ++x) {
            if constexpr (std::is_same_v<T, float>) {
                k.at(Point{x, y, 0}) = fval(x, y, salt, -0.25f, 0.25f);
            } else {
                k.at(Point{x, y, 0}) = make_<T>(fval(x, y, salt, -0.25f, 0.25f), fval(x, y, salt + 1, -0.25f, 0.25f),
                                                fval(x, y, salt + 2, -0.25f, 0.25f));
            }
        }
    k.upload(s);
    return k;
}

// ---- the calls: the only lines that differ between the base and the head build ----
template <typename D, typename Reads, typename M, typename A, typename W>
void runLinear(GpuStream& s, const D& d, const Reads& r, const M& m, const A& a, const W& w) {
#ifdef PR435_HEAD
    executeOperations<LinearFilterDPP<GPU, D>>(s, d, r, m, a, w);
#else
    require(executeLinearFilter(d, r, m, a, w, s), "executeLinearFilter returned false");
#endif
}
// executeBoxFilter(details, image, write, 1.f, stream) at the base; the PR's replacement at the head
template <typename D, typename R, typename W>
void runBoxFilter(GpuStream& s, const D& d, const R& image, const W& w) {
#ifdef PR435_HEAD
    using T = typename D::ValueType;
    using Acc = typename D::AccumulatorType;
    const auto coefficient = ConstantFilterRead<Acc>::build(
        make_set<Acc>(1.f / static_cast<float>(d.kernelWidth * d.kernelHeight)));
    executeOperations<LinearFilterDPP<GPU, D>>(s, d, make_tuple(image, coefficient),
                                               Mul<T, Acc, Acc, UnaryType>::build(),
                                               Add<Acc, Acc, Acc, UnaryType>::build(), w);
#else
    require(executeBoxFilter(d, image, w, 1.f, s), "executeBoxFilter returned false");
#endif
}
template <typename D, typename R, typename W, typename Sel>
void runMedian(GpuStream& s, const D& d, const R& r, const W& w, const Sel& sel) {
#ifdef PR435_HEAD
    executeOperations<MedianFilterDPP<GPU, D>>(s, d, r, w, sel);
#else
    require(executeMedianFilter(d, r, w, sel, s), "executeMedianFilter returned false");
#endif
}
template <typename D, typename R, typename W, typename Red>
void runMorphology(GpuStream& s, const D& d, const R& r, const W& w, const Red& red) {
#ifdef PR435_HEAD
    executeOperations<MorphologyDPP<GPU, D>>(s, d, r, w, red);
#else
    require(executeMorphology(d, r, w, red, s), "executeMorphology returned false");
#endif
}
enum class Morph { Erode, Dilate, Open, Close };
template <Morph M, typename D, typename R, typename W>
void runMorphNamed(GpuStream& s, const D& d, const R& r, const W& w) {
    using T = typename D::ValueType;
#ifdef PR435_HEAD
    const auto mn = Min<T, T, T, UnaryType>::build();
    const auto mx = Max<T, T, T, UnaryType>::build();
    if constexpr (M == Morph::Erode) executeOperations<MorphologyDPP<GPU, D>>(s, d, r, w, make_tuple(mn));
    else if constexpr (M == Morph::Dilate) executeOperations<MorphologyDPP<GPU, D>>(s, d, r, w, make_tuple(mx));
    else if constexpr (M == Morph::Open) executeOperations<MorphologyDPP<GPU, D>>(s, d, r, w, make_tuple(mn, mx));
    else executeOperations<MorphologyDPP<GPU, D>>(s, d, r, w, make_tuple(mx, mn));
#else
    bool ok;
    if constexpr (M == Morph::Erode) ok = executeErode(d, r, w, s);
    else if constexpr (M == Morph::Dilate) ok = executeDilate(d, r, w, s);
    else if constexpr (M == Morph::Open) ok = executeOpen(d, r, w, s);
    else ok = executeClose(d, r, w, s);
    require(ok, "executeErode/Dilate/Open/Close returned false");
#endif
}
template <int PASSES, typename D, typename R, typename W>
void runBoxBlur(GpuStream& s, const D& d, const R& r, const W& w) {
#ifdef PR435_HEAD
    using T = typename D::ValueType;
    executeOperations<MorphologyDPP<GPU, D>>(s, d, r, w, boxBlurReducers<T, PASSES>(d));
#else
    require(executeBoxBlur<PASSES>(d, r, w, s), "executeBoxBlur returned false");
#endif
}
template <typename DPP, typename Details, typename... IOps>
void runQuad(GpuStream& s, const Details& d, const IOps&... iOps) {
#ifdef PR435_HEAD
    executeOperations<DPP>(s, d, iOps...);
#else
    bool ok;
    if constexpr (std::is_same_v<Details, BoxFilterQuadDetails>) ok = executeBoxFilterQuad<DPP>(s, d, iOps...);
    else if constexpr (std::is_same_v<Details, ConvQuadDetails>) ok = executeConvQuad<DPP>(s, d, iOps...);
    else if constexpr (std::is_same_v<Details, MedianQuadDetails>) ok = executeMedianQuad<DPP>(s, d, iOps...);
    else ok = executeMorphQuad<DPP>(s, d, iOps...);
    require(ok, "execute*Quad returned false");
#endif
}

// ---- LinearFilterDPP ----
void linearCases(GpuStream& s) {
    {   // float, a 5x3 kernel read from a buffer, anchor (2, 1)
        using D = LinearFilterDPPDetails<float>;
        const D d{5, 3, 2, 1};
        auto img = frame<float>(s, 1);
        auto ker = coefficients<float>(s, 5, 3, 101);
        auto out = filled<float>(s, FW, FH);
        runLinear(s, d, make_tuple(PerThreadRead<ND::_2D, float>::build(img), PerThreadRead<ND::_2D, float>::build(ker)),
                  Mul<float, float, float, UnaryType>::build(), Add<float, float, float, UnaryType>::build(),
                  PerThreadWrite<ND::_2D, float>::build(out));
        dump(s, "linear_f32_5x3", out);
    }
    {   // float, a 7x7 kernel, fused reads and write: image * 2 + 1, kernel * 0.5, output * 0.25
        using D = LinearFilterDPPDetails<float>;
        const D d{7, 7, 3, 3};
        auto img = frame<float>(s, 2);
        auto ker = coefficients<float>(s, 7, 7, 102);
        auto out = filled<float>(s, FW, FH);
        const auto imageRead = PerThreadRead<ND::_2D, float>::build(img).then(Mul<float>::build(2.f)).then(Add<float>::build(1.f));
        const auto kernelRead = PerThreadRead<ND::_2D, float>::build(ker).then(Mul<float>::build(0.5f));
        runLinear(s, d, make_tuple(imageRead, kernelRead), Mul<float, float, float, UnaryType>::build(),
                  Add<float, float, float, UnaryType>::build(),
                  Mul<float>::build(0.25f).then(PerThreadWrite<ND::_2D, float>::build(out)));
        dump(s, "linear_f32_7x7_fused", out);
    }
    {   // the box filter: executeBoxFilter at the base, its replacement at the head; float 5x5 and uchar3 3x3 (float3 out)
        using D = LinearFilterDPPDetails<float>;
        auto img = frame<float>(s, 3);
        auto out = filled<float>(s, FW, FH);
        runBoxFilter(s, D{5, 5, 2, 2}, PerThreadRead<ND::_2D, float>::build(img), PerThreadWrite<ND::_2D, float>::build(out));
        dump(s, "linear_f32_box5x5", out);
        using D3 = LinearFilterDPPDetails<uchar3>;
        auto img3 = frame<uchar3>(s, 4);
        auto out3 = filled<float3>(s, FW, FH);
        runBoxFilter(s, D3{3, 3, 1, 1}, PerThreadRead<ND::_2D, uchar3>::build(img3),
                     PerThreadWrite<ND::_2D, float3>::build(out3));
        dump(s, "linear_u8c3_box3x3", out3);
    }
    {   // a batch of 3 crops of different sizes (per-image sizes), 3x5 kernel, written by a TensorWrite and by a BatchWrite
        using D = LinearFilterDPPDetails<float>;
        const D d{3, 5, 1, 2};
        auto f = frame<float>(s, 5);
        auto ker = coefficients<float>(s, 3, 5, 105);
        const auto crops = cropsOf(f, VAR_CROPS);
        auto tensor = tensorFilled<float>(s, VAR_MAX_W, VAR_MAX_H, 3);
        auto outs = filledOutputs<float>(s, VAR_CROPS);
        const auto reads = make_tuple(PerThreadRead<ND::_2D, float>::build(crops), PerThreadRead<ND::_2D, float>::build(ker));
        runLinear(s, d, reads, Mul<float, float, float, UnaryType>::build(), Add<float, float, float, UnaryType>::build(),
                  TensorWrite<float>::build(tensor.ptr()));
        runLinear(s, d, reads, Mul<float, float, float, UnaryType>::build(), Add<float, float, float, UnaryType>::build(),
                  PerThreadWrite<ND::_2D, float>::build(outs));
        dump(s, "linear_f32_batch3x5_tensor", tensor, VAR_MAX_W, VAR_MAX_H, 3);
        dump(s, "linear_f32_batch3x5_batchwrite", outs);
    }
    {   // a batch of 3 uchar3 crops of different sizes, float3 coefficients from a buffer, float3 out
        using D = LinearFilterDPPDetails<uchar3>;
        const D d{3, 3, 1, 1};
        auto f = frame<uchar3>(s, 6);
        auto ker = coefficients<float3>(s, 3, 3, 106);
        auto tensor = tensorFilled<float3>(s, VAR_MAX_W, VAR_MAX_H, 3);
        runLinear(s, d, make_tuple(PerThreadRead<ND::_2D, uchar3>::build(cropsOf(f, VAR_CROPS)),
                                   PerThreadRead<ND::_2D, float3>::build(ker)),
                  Mul<uchar3, float3, float3, UnaryType>::build(), Add<float3, float3, float3, UnaryType>::build(),
                  TensorWrite<float3>::build(tensor.ptr()));
        dump(s, "linear_u8c3_batch3x3_tensor", tensor, VAR_MAX_W, VAR_MAX_H, 3);
    }
}

// ---- MedianFilterDPP ----
void medianCases(GpuStream& s) {
    {
        using D = MedianFilterDPPDetails<uchar>;
        auto img = frame<uchar>(s, 11);
        auto out = filled<uchar>(s, FW, FH);
        runMedian(s, D{3, 3, 1, 1}, PerThreadRead<ND::_2D, uchar>::build(img), PerThreadWrite<ND::_2D, uchar>::build(out),
                  MedianWindowSelect<uchar, D::WINDOW_CAPACITY>::build());
        dump(s, "median_u8_3x3", out);
    }
    {
        using D = MedianFilterDPPDetails<uchar3>;
        auto img = frame<uchar3>(s, 12);
        auto out = filled<uchar3>(s, FW, FH);
        runMedian(s, D{5, 5, 2, 2}, PerThreadRead<ND::_2D, uchar3>::build(img), PerThreadWrite<ND::_2D, uchar3>::build(out),
                  MedianWindowSelect<uchar3, D::WINDOW_CAPACITY>::build());
        dump(s, "median_u8c3_5x5", out);
    }
    {   // a batch of 3 float crops of different sizes, read * 2 + 1, written * 0.5; the median 3x3 and the minimum 5x3
        using D = MedianFilterDPPDetails<float, 16, 8, 7, 7>;
        auto f = frame<float>(s, 13);
        const auto read = PerThreadRead<ND::_2D, float>::build(cropsOf(f, VAR_CROPS))
                              .then(Mul<float>::build(2.f)).then(Add<float>::build(1.f));
        auto tensor = tensorFilled<float>(s, VAR_MAX_W, VAR_MAX_H, 3);
        auto outs = filledOutputs<float>(s, VAR_CROPS);
        runMedian(s, D{3, 3, 1, 1}, read, Mul<float>::build(0.5f).then(TensorWrite<float>::build(tensor.ptr())),
                  MedianWindowSelect<float, D::WINDOW_CAPACITY>::build());
        runMedian(s, D{5, 3, 2, 1}, read, Mul<float>::build(0.5f).then(PerThreadWrite<ND::_2D, float>::build(outs)),
                  MinimumWindowSelect<float, D::WINDOW_CAPACITY>::build());
        dump(s, "median_f32_batch3x3_tensor", tensor, VAR_MAX_W, VAR_MAX_H, 3);
        dump(s, "median_f32_batch5x3_minimum_batchwrite", outs);
    }
}

// ---- MorphologyDPP ----
void morphologyCases(GpuStream& s) {
    {
        using D = MorphologyDPPDetails<uchar>;
        auto img = frame<uchar>(s, 21);
        auto out = filled<uchar>(s, FW, FH);
        runMorphNamed<Morph::Erode>(s, D{FW, FH, 3, 3, 1, 1}, PerThreadRead<ND::_2D, uchar>::build(img),
                                    PerThreadWrite<ND::_2D, uchar>::build(out));
        dump(s, "morph_u8_erode3x3", out);
        auto out2 = filled<uchar>(s, FW, FH);
        runMorphNamed<Morph::Dilate>(s, D{FW, FH, 5, 3, 2, 1}, PerThreadRead<ND::_2D, uchar>::build(img),
                                     PerThreadWrite<ND::_2D, uchar>::build(out2));
        dump(s, "morph_u8_dilate5x3", out2);
    }
    {
        using D = MorphologyDPPDetails<float>;
        auto img = frame<float>(s, 22);
        auto out = filled<float>(s, FW, FH);
        runMorphNamed<Morph::Open>(s, D{FW, FH, 3, 3, 1, 1}, PerThreadRead<ND::_2D, float>::build(img),
                                   PerThreadWrite<ND::_2D, float>::build(out));
        dump(s, "morph_f32_open3x3", out);
        auto out2 = filled<float>(s, FW, FH);
        runMorphNamed<Morph::Close>(s, D{FW, FH, 5, 5, 2, 2}, PerThreadRead<ND::_2D, float>::build(img),
                                    PerThreadWrite<ND::_2D, float>::build(out2));
        dump(s, "morph_f32_close5x5", out2);
    }
    {   // the box blur: executeBoxBlur at the base, boxBlurReducers at the head
        // a batch of 3 float crops of one size, 1 pass 3x3, TensorWrite (as E19's fkl_batch, smaller)
        using D = MorphologyDPPDetails<float>;
        auto f = frame<float>(s, 23);
        auto tensor = tensorFilled<float>(s, 20, 14, 3);
        runBoxBlur<1>(s, D{20, 14, 3, 3, 1, 1}, PerThreadRead<ND::_2D, float>::build(cropsOf(f, EQ_CROPS)),
                      TensorWrite<float>::build(tensor.ptr()));
        dump(s, "morph_f32_boxblur1_batch3x3_tensor", tensor, 20, 14, 3);
        // float3, 2 passes 5x5 on one image
        using D3 = MorphologyDPPDetails<float3>;
        auto img3 = frame<float3>(s, 24);
        auto out3 = filled<float3>(s, FW, FH);
        runBoxBlur<2>(s, D3{FW, FH, 5, 5, 2, 2}, PerThreadRead<ND::_2D, float3>::build(img3),
                      PerThreadWrite<ND::_2D, float3>::build(out3));
        dump(s, "morph_f32c3_boxblur2_5x5", out3);
    }
    {   // explicit reducers, 3 passes (Max, Min, Max), a batch of 3 uchar crops of one size, BatchWrite
        using D = MorphologyDPPDetails<uchar>;
        auto f = frame<uchar>(s, 25);
        auto outs = filledOutputs<uchar>(s, EQ_CROPS);
        runMorphology(s, D{20, 14, 3, 5, 1, 2}, PerThreadRead<ND::_2D, uchar>::build(cropsOf(f, EQ_CROPS)),
                      PerThreadWrite<ND::_2D, uchar>::build(outs),
                      make_tuple(Max<uchar, uchar, uchar, UnaryType>::build(), Min<uchar, uchar, uchar, UnaryType>::build(),
                                 Max<uchar, uchar, uchar, UnaryType>::build()));
        dump(s, "morph_u8_3pass_batch3x5_batchwrite", outs);
    }
}

// ---- the Quad DPPs ----
template <typename T>
auto boxCompute(const int kw, const int kh) {
    return make_tuple(Add<T, T, T, UnaryType>::build(), Sub<T, T, T, UnaryType>::build(),
                      Div<T>::build(make_set<T>(static_cast<float>(kw * kh))));
}
ConvQuadDetails convDetails(const int kw, const int kh) {
    ConvQuadDetails d{kw, kh, kw / 2, kh / 2, {}};
    for (int i = 0; i < kw * kh; ++i) d.coefficients[i] = fval(i, 0, 301, -0.5f, 0.5f);
    return d;
}
void quadCases(GpuStream& s) {
    {   // batches of 3 crops of different sizes, TensorWrite
        auto fu = frame<uchar>(s, 31);
        auto ff = frame<float>(s, 32);
        const auto cu = cropsOf(fu, VAR_CROPS);
        const auto cf = cropsOf(ff, VAR_CROPS);
        auto tMorph = tensorFilled<uchar>(s, VAR_MAX_W, VAR_MAX_H, 3);
        runQuad<MorphQuadDPP<GPU, uchar>>(s, MorphQuadDetails{3, 3, 1, 1}, PerThreadRead<ND::_2D, uchar>::build(cu),
                                          Min<uchar, uchar, uchar, UnaryType>::build(), TensorWrite<uchar>::build(tMorph.ptr()));
        dump(s, "quad_morph_u8_batch3x3_tensor", tMorph, VAR_MAX_W, VAR_MAX_H, 3);
        auto tMedian = tensorFilled<uchar>(s, VAR_MAX_W, VAR_MAX_H, 3);
        runQuad<MedianQuadDPP<GPU, uchar>>(s, MedianQuadDetails{5, 3, 2, 1}, PerThreadRead<ND::_2D, uchar>::build(cu),
                                           make_tuple(Min<uchar, uchar, uchar, UnaryType>::build(),
                                                      Max<uchar, uchar, uchar, UnaryType>::build()),
                                           TensorWrite<uchar>::build(tMedian.ptr()));
        dump(s, "quad_median_u8_batch5x3_tensor", tMedian, VAR_MAX_W, VAR_MAX_H, 3);
        auto tBox = tensorFilled<float>(s, VAR_MAX_W, VAR_MAX_H, 3);
        runQuad<BoxFilterQuadDPP<GPU, float>>(s, BoxFilterQuadDetails{5, 3, 2, 1}, PerThreadRead<ND::_2D, float>::build(cf),
                                              boxCompute<float>(5, 3), TensorWrite<float>::build(tBox.ptr()));
        dump(s, "quad_box_f32_batch5x3_tensor", tBox, VAR_MAX_W, VAR_MAX_H, 3);
        auto tConv = tensorFilled<float>(s, VAR_MAX_W, VAR_MAX_H, 3);
        runQuad<ConvQuadDPP<GPU, float>>(s, convDetails(3, 5), PerThreadRead<ND::_2D, float>::build(cf),
                                         make_tuple(Mul<float, float, float, UnaryType>::build(),
                                                    Add<float, float, float, UnaryType>::build()),
                                         TensorWrite<float>::build(tConv.ptr()));
        dump(s, "quad_conv_f32_batch3x5_tensor", tConv, VAR_MAX_W, VAR_MAX_H, 3);
    }
    {   // single vector images
        auto fu3 = frame<uchar3>(s, 33);
        auto ff4 = frame<float4>(s, 34);
        auto oMorph = filled<uchar3>(s, FW, FH);
        runQuad<MorphQuadDPP<GPU, uchar3>>(s, MorphQuadDetails{3, 3, 1, 1}, PerThreadRead<ND::_2D, uchar3>::build(fu3),
                                           Max<uchar3, uchar3, uchar3, UnaryType>::build(),
                                           PerThreadWrite<ND::_2D, uchar3>::build(oMorph));
        dump(s, "quad_morph_u8c3_3x3", oMorph);
        auto oMedian = filled<float4>(s, FW, FH);
        runQuad<MedianQuadDPP<GPU, float4>>(s, MedianQuadDetails{5, 3, 2, 1}, PerThreadRead<ND::_2D, float4>::build(ff4),
                                            make_tuple(Min<float4, float4, float4, UnaryType>::build(),
                                                       Max<float4, float4, float4, UnaryType>::build()),
                                            PerThreadWrite<ND::_2D, float4>::build(oMedian));
        dump(s, "quad_median_f32c4_5x3", oMedian);
        auto oBox = filled<float4>(s, FW, FH);
        runQuad<BoxFilterQuadDPP<GPU, float4>>(s, BoxFilterQuadDetails{5, 3, 2, 1}, PerThreadRead<ND::_2D, float4>::build(ff4),
                                               boxCompute<float4>(5, 3), PerThreadWrite<ND::_2D, float4>::build(oBox));
        dump(s, "quad_box_f32c4_5x3", oBox);
        auto oConv = filled<float3>(s, FW, FH);
        runQuad<ConvQuadDPP<GPU, uchar3>>(s, convDetails(3, 3), PerThreadRead<ND::_2D, uchar3>::build(fu3),
                                          make_tuple(Mul<float, uchar3, float3, UnaryType>::build(),
                                                     Add<float3, float3, float3, UnaryType>::build()),
                                          PerThreadWrite<ND::_2D, float3>::build(oConv));
        dump(s, "quad_conv_u8c3_3x3", oConv);
    }
}

// ---- TransformDPP and DivergentBatchTransformDPP (the same call at both commits; only their kernel is renamed) ----
struct TwoPlanes {
    FK_HOST_DEVICE_FUSE uint at(const uint& index) { return index == 0 ? 0u : 1u; }
};
void transformCases(GpuStream& s) {
    {   // no thread fusion, 61 x 43
        auto img = frame<float>(s, 41);
        auto out = filled<float>(s, FW, FH);
        executeOperations<TransformDPP<>>(s, PerThreadRead<ND::_2D, float>::build(img), Mul<float>::build(1.5f),
                                          Add<float>::build(0.25f), PerThreadWrite<ND::_2D, float>::build(out));
        dump(s, "transform_f32", out);
    }
    {   // thread fusion enabled, uchar: 64 x 32 (thread divisible) and 61 x 43 (not)
        auto a = frame<uchar>(s, 42, 64, 32);
        auto oa = filled<uchar>(s, 64, 32);
        executeOperations<TransformDPP<GPU, TF::ENABLED>>(s, PerThreadRead<ND::_2D, uchar>::build(a), Add<uchar>::build(7),
                                                         PerThreadWrite<ND::_2D, uchar>::build(oa));
        dump(s, "transform_u8_tf_64x32", oa);
        auto b = frame<uchar>(s, 43);
        auto ob = filled<uchar>(s, FW, FH);
        executeOperations<TransformDPP<GPU, TF::ENABLED>>(s, PerThreadRead<ND::_2D, uchar>::build(b), Add<uchar>::build(7),
                                                         PerThreadWrite<ND::_2D, uchar>::build(ob));
        dump(s, "transform_u8_tf_61x43", ob);
    }
    {   // DivergentBatchTransformDPP: plane 0 = image A * 2, plane 1 = image B + 100
        auto a = frame<float>(s, 44, 32, 16);
        auto b = frame<float>(s, 45, 32, 16);
        auto t = tensorFilled<float>(s, 32, 16, 2);
        const auto seq1 = buildOperationSequence(PerThreadRead<ND::_2D, float>::build(a), Mul<float>::build(2.f),
                                                 TensorWrite<float>::build(t));
        const auto seq2 = buildOperationSequence(PerThreadRead<ND::_2D, float>::build(b), Add<float>::build(100.f),
                                                 TensorWrite<float>::build(t));
        Executor<DivergentBatchTransformDPP<GPU, TwoPlanes>>::executeOperations(s, seq1, seq2);
        dump(s, "divergent_batch_f32_2planes", t, 32, 16, 2);
    }
}

} // namespace

int main(int argc, char** argv) {
    start(argc, argv, "pr435_image");
    try {
        GpuStream s;
        linearCases(s);
        medianCases(s);
        morphologyCases(s);
        quadCases(s);
        transformCases(s);
        s.sync();
    } catch (const std::exception& e) {
        fail(std::string("exception: ") + e.what());
    }
    return finish("pr435_image");
}
