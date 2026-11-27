/* Task C-MAIN-NV harness (FKL #436, task C on FKL main), image DPPs on ParArch::GPU_NVIDIA: LinearFilterDPP (with the box filter),
   MedianFilterDPP, MorphologyDPP (erode, dilate, open, close, the box blur and explicit reducers), the four Quad DPPs
   (BoxFilterQuad, ConvQuad, MedianQuad, MorphQuad), TransformDPP and DivergentBatchTransformDPP. Adapted from task C-NV's
   harness of FKL #435 (scratch/pr435/harness/pr435_image.cu) to FKL main's image DPPs, which differ from LTS-C++17's (#433 is
   not on main): the details carry the image's width and height, each call filters one image (no batches), and the Quad DPPs take
   scalar pixels; so every case here is one image of a scalar type (float or uchar).
   The base build calls the entry functions of main ebb3891 (executeLinearFilter, executeBoxFilter, executeMedianFilter,
   executeErode/Dilate/Open/Close, executeBoxBlur, executeMorphology, which return false for invalid details, and
   execute{BoxFilter,Conv,Median,Morph}Quad<DPP>); the head build calls executeOperations<DPP>(stream, details, iOps...) of
   1df2738, with the arguments the removed convenience functions built (executeBoxFilter: ConstantFilterRead of
   1 / (kw * kh) with Mul and Add, as main's executeBoxFilter does at ebb3891; make_tuple(Min / Max ...) for erode, dilate,
   open, close; fk::boxBlurReducers<T, PASSES>(details) for executeBoxBlur). TransformDPP and DivergentBatchTransformDPP have
   the same call at both commits. See pr436_common.h. */
#include "pr436_common.h"

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

using namespace h436;

// A named namespace, not an unnamed one: on FKL main, fk has its own vector types (fk::uint3, ...), and the host stub code that nvcc
// adds at global scope names CUDA's uint3. A using-directive in an unnamed namespace would reach the global scope too; FKL main's
// tests also keep theirs in a narrower scope.
namespace h436run {
using namespace fk;

constexpr ParArch GPU = ParArch::GPU_NVIDIA;
constexpr MemType MEM = MemType::DeviceAndPinned;
using GpuStream = Stream_<ParArch::GPU_NVIDIA>;

constexpr int FW = 61, FH = 43;

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
#ifdef PR436_HEAD
    executeOperations<LinearFilterDPP<GPU, D>>(s, d, r, m, a, w);
#else
    require(executeLinearFilter(d, r, m, a, w, s), "executeLinearFilter returned false");
#endif
}
// executeBoxFilter(details, image, write, 1, stream) at the base; at the head, what it built (main ebb3891 linear_filter.h:333-339):
// the coefficient scale / (kw * kh) in the image type, a ConstantFilterRead of it, Mul and Add of that type
template <typename D, typename R, typename W>
void runBoxFilter(GpuStream& s, const D& d, const R& image, const W& w) {
    using T = typename D::ValueType;
#ifdef PR436_HEAD
    const T coefficient = static_cast<T>(1) / static_cast<T>(d.kernelWidth * d.kernelHeight);
    executeOperations<LinearFilterDPP<GPU, D>>(s, d, make_tuple(image, ConstantFilterRead<T>::build(coefficient)),
                                               Mul<T, T, T, UnaryType>::build(), Add<T, T, T, UnaryType>::build(), w);
#else
    require(executeBoxFilter(d, image, w, static_cast<T>(1), s), "executeBoxFilter returned false");
#endif
}
template <typename D, typename R, typename W, typename Sel>
void runMedian(GpuStream& s, const D& d, const R& r, const W& w, const Sel& sel) {
#ifdef PR436_HEAD
    executeOperations<MedianFilterDPP<GPU, D>>(s, d, r, w, sel);
#else
    require(executeMedianFilter(d, r, w, sel, s), "executeMedianFilter returned false");
#endif
}
template <typename D, typename R, typename W, typename Red>
void runMorphology(GpuStream& s, const D& d, const R& r, const W& w, const Red& red) {
#ifdef PR436_HEAD
    executeOperations<MorphologyDPP<GPU, D>>(s, d, r, w, red);
#else
    require(executeMorphology(d, r, w, red, s), "executeMorphology returned false");
#endif
}
enum class Morph { Erode, Dilate, Open, Close };
template <Morph M, typename D, typename R, typename W>
void runMorphNamed(GpuStream& s, const D& d, const R& r, const W& w) {
    using T = typename D::ValueType;
#ifdef PR436_HEAD
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
#ifdef PR436_HEAD
    using T = typename D::ValueType;
    executeOperations<MorphologyDPP<GPU, D>>(s, d, r, w, boxBlurReducers<T, PASSES>(d));
#else
    require(executeBoxBlur<PASSES>(d, r, w, s), "executeBoxBlur returned false");
#endif
}
// main's execute*Quad<DPP> return nothing (they throw for invalid details)
template <typename DPP, typename Details, typename... IOps>
void runQuad(GpuStream& s, const Details& d, const IOps&... iOps) {
#ifdef PR436_HEAD
    executeOperations<DPP>(s, d, iOps...);
#else
    if constexpr (std::is_same_v<Details, BoxFilterQuadDetails>) executeBoxFilterQuad<DPP>(s, d, iOps...);
    else if constexpr (std::is_same_v<Details, ConvQuadDetails>) executeConvQuad<DPP>(s, d, iOps...);
    else if constexpr (std::is_same_v<Details, MedianQuadDetails>) executeMedianQuad<DPP>(s, d, iOps...);
    else executeMorphQuad<DPP>(s, d, iOps...);
#endif
}

// ---- LinearFilterDPP (details: width, height, kernelWidth, kernelHeight, anchorX, anchorY) ----
void linearCases(GpuStream& s) {
    {   // float, a 5x3 kernel read from a buffer, anchor (2, 1)
        using D = LinearFilterDPPDetails<float>;
        const D d{FW, FH, 5, 3, 2, 1};
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
        const D d{FW, FH, 7, 7, 3, 3};
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
    {   // the box filter: executeBoxFilter at the base, what it built at the head; float 5x5 and 3x7 (anchor (1, 3))
        using D = LinearFilterDPPDetails<float>;
        auto img = frame<float>(s, 3);
        auto out = filled<float>(s, FW, FH);
        runBoxFilter(s, D{FW, FH, 5, 5, 2, 2}, PerThreadRead<ND::_2D, float>::build(img), PerThreadWrite<ND::_2D, float>::build(out));
        dump(s, "linear_f32_box5x5", out);
        auto out2 = filled<float>(s, FW, FH);
        runBoxFilter(s, D{FW, FH, 3, 7, 1, 3}, PerThreadRead<ND::_2D, float>::build(img), PerThreadWrite<ND::_2D, float>::build(out2));
        dump(s, "linear_f32_box3x7", out2);
    }
}

// ---- MedianFilterDPP (details: width, height, windowWidth, windowHeight, anchorX, anchorY) ----
void medianCases(GpuStream& s) {
    {
        using D = MedianFilterDPPDetails<uchar>;
        auto img = frame<uchar>(s, 11);
        auto out = filled<uchar>(s, FW, FH);
        runMedian(s, D{FW, FH, 3, 3, 1, 1}, PerThreadRead<ND::_2D, uchar>::build(img), PerThreadWrite<ND::_2D, uchar>::build(out),
                  MedianWindowSelect<uchar, D::WINDOW_CAPACITY>::build());
        dump(s, "median_u8_3x3", out);
    }
    {   // float, read * 2 + 1, written * 0.5; the median 3x3 and the minimum 5x3
        using D = MedianFilterDPPDetails<float, 16, 8, 7, 7>;
        auto img = frame<float>(s, 13);
        const auto read = PerThreadRead<ND::_2D, float>::build(img).then(Mul<float>::build(2.f)).then(Add<float>::build(1.f));
        auto out = filled<float>(s, FW, FH);
        auto out2 = filled<float>(s, FW, FH);
        runMedian(s, D{FW, FH, 3, 3, 1, 1}, read, Mul<float>::build(0.5f).then(PerThreadWrite<ND::_2D, float>::build(out)),
                  MedianWindowSelect<float, D::WINDOW_CAPACITY>::build());
        runMedian(s, D{FW, FH, 5, 3, 2, 1}, read, Mul<float>::build(0.5f).then(PerThreadWrite<ND::_2D, float>::build(out2)),
                  MinimumWindowSelect<float, D::WINDOW_CAPACITY>::build());
        dump(s, "median_f32_fused3x3", out);
        dump(s, "median_f32_fused5x3_minimum", out2);
    }
}

// ---- MorphologyDPP (details: width, height, maskW, maskH, anchorX, anchorY) ----
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
    {   // the box blur: executeBoxBlur at the base, boxBlurReducers at the head; float, 1 pass 3x3 and 2 passes 5x5
        using D = MorphologyDPPDetails<float>;
        auto img = frame<float>(s, 23);
        auto out = filled<float>(s, FW, FH);
        runBoxBlur<1>(s, D{FW, FH, 3, 3, 1, 1}, PerThreadRead<ND::_2D, float>::build(img), PerThreadWrite<ND::_2D, float>::build(out));
        dump(s, "morph_f32_boxblur1_3x3", out);
        auto out2 = filled<float>(s, FW, FH);
        runBoxBlur<2>(s, D{FW, FH, 5, 5, 2, 2}, PerThreadRead<ND::_2D, float>::build(img), PerThreadWrite<ND::_2D, float>::build(out2));
        dump(s, "morph_f32_boxblur2_5x5", out2);
    }
    {   // explicit reducers, 3 passes (Max, Min, Max), uchar 3x5
        using D = MorphologyDPPDetails<uchar>;
        auto img = frame<uchar>(s, 25);
        auto out = filled<uchar>(s, FW, FH);
        runMorphology(s, D{FW, FH, 3, 5, 1, 2}, PerThreadRead<ND::_2D, uchar>::build(img), PerThreadWrite<ND::_2D, uchar>::build(out),
                      make_tuple(Max<uchar, uchar, uchar, UnaryType>::build(), Min<uchar, uchar, uchar, UnaryType>::build(),
                                 Max<uchar, uchar, uchar, UnaryType>::build()));
        dump(s, "morph_u8_3pass3x5", out);
    }
}

// ---- the Quad DPPs (details: width, height, kernelWidth, kernelHeight, anchorX, anchorY; scalar pixels on main) ----
template <typename T>
auto boxCompute(const int kw, const int kh) {
    return make_tuple(Add<T, T, T, UnaryType>::build(), Sub<T, T, T, UnaryType>::build(),
                      Div<T>::build(static_cast<T>(kw * kh)));
}
ConvQuadDetails convDetails(const int kw, const int kh) {
    ConvQuadDetails d{FW, FH, kw, kh, kw / 2, kh / 2, {}};
    for (int i = 0; i < kw * kh; ++i) d.coefficients[i] = fval(i, 0, 301, -0.5f, 0.5f);
    return d;
}
void quadCases(GpuStream& s) {
    auto fu = frame<uchar>(s, 31);
    auto ff = frame<float>(s, 32);
    auto oMorph = filled<uchar>(s, FW, FH);
    runQuad<MorphQuadDPP<GPU, uchar>>(s, MorphQuadDetails{FW, FH, 3, 3, 1, 1}, PerThreadRead<ND::_2D, uchar>::build(fu),
                                      Min<uchar, uchar, uchar, UnaryType>::build(), PerThreadWrite<ND::_2D, uchar>::build(oMorph));
    dump(s, "quad_morph_u8_3x3", oMorph);
    auto oMedian = filled<uchar>(s, FW, FH);
    runQuad<MedianQuadDPP<GPU, uchar>>(s, MedianQuadDetails{FW, FH, 5, 3, 2, 1}, PerThreadRead<ND::_2D, uchar>::build(fu),
                                       make_tuple(Min<uchar, uchar, uchar, UnaryType>::build(), Max<uchar, uchar, uchar, UnaryType>::build()),
                                       PerThreadWrite<ND::_2D, uchar>::build(oMedian));
    dump(s, "quad_median_u8_5x3", oMedian);
    auto oBox = filled<float>(s, FW, FH);
    runQuad<BoxFilterQuadDPP<GPU, float>>(s, BoxFilterQuadDetails{FW, FH, 5, 3, 2, 1}, PerThreadRead<ND::_2D, float>::build(ff),
                                          boxCompute<float>(5, 3), PerThreadWrite<ND::_2D, float>::build(oBox));
    dump(s, "quad_box_f32_5x3", oBox);
    auto oConv = filled<float>(s, FW, FH);
    runQuad<ConvQuadDPP<GPU, float>>(s, convDetails(3, 5), PerThreadRead<ND::_2D, float>::build(ff),
                                     make_tuple(Mul<float, float, float, UnaryType>::build(), Add<float, float, float, UnaryType>::build()),
                                     PerThreadWrite<ND::_2D, float>::build(oConv));
    dump(s, "quad_conv_f32_3x5", oConv);
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

} // namespace h436run

int main(int argc, char** argv) {
    using namespace h436run;
    start(argc, argv, "pr436_image");
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
    return finish("pr436_image");
}
