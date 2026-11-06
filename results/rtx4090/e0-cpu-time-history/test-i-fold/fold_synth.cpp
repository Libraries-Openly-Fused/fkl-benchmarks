// SYNTHETIC TEST (i), 2026-11-06: ONE FUNCTION OF FKL 1f33224 COPIED AS TEXT INTO A TEST PROGRAM. NOT A BUILD OF THE 2025 LIBRARY.
// What is copied: cxp::max with its helper `internal::max_helper` exactly as in include/fused_kernel/core/constexpr_libs/constexpr_cmath.cuh at 1f33224 (lines 49-76; the macro FK_HOST_DEVICE_CNST is defined as
// `constexpr` and the static_assert uses std::is_same instead of fk::all_types_are_same: compile-time only). What is a stand-in: the per-argument call Operation::num_elems_x(Point(0u, 0u, 0u), instantiableOperations[Idx]),
// a function returning the destination width of a 48-byte operation data (resize.cuh:84-90 at 1f33224; the instantiable operation DERIVES from its operation data, batch_operations.cuh / instantiable_operations.cuh:64, so the
// argument binds without a copy). The fold is the one of BatchRead<..., CONDITIONAL_WITH_DEFAULT>::build_helper (batch_operations.cuh:255-256 at 1f33224): max_width and max_height, each `cxp::max(num_elems(...)...)` over BATCH arguments.
// For comparison, the LOOP form of the current library (batch_operations.h:437-441 at c8fcbe8): max_width = cxp::max::f(max_width, num_elems_x(...)) per crop, with a 2-argument stand-in for cxp::max::f.
// Host only: g++, no CUDA. Time per call in microseconds for B = 2, 12, 42, 62, 102, 142, 152, as a function of the host flags.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>
#include <vector>

#define FK_HOST_DEVICE_CNST constexpr
using uint = unsigned int;

// ---- text of 1f33224 constexpr_cmath.cuh lines 49-76 (only the macro and the static_assert's trait are replaced) ----
namespace cxp {
    namespace internal {
        template <typename Type>
        FK_HOST_DEVICE_CNST auto max_helper(const Type& value) {
            return value;
        }
        template <typename FirstType, typename... Types>
        FK_HOST_DEVICE_CNST auto max_helper(const FirstType& firstValue,
                                            const Types&... values) {
            const auto previousMax = max_helper(values...);
            return firstValue >= previousMax ? firstValue : previousMax;
        }
    } // namespace internal

    template <typename FirstType, typename... Types>
    FK_HOST_DEVICE_CNST auto max(const FirstType& value, const Types&... values) {
        static_assert((std::is_same_v<FirstType, Types> && ...), "All types must be the same");
        return internal::max_helper(value, values...);
    }
    // stand-in of the current cxp::max::f(a, b) (constexpr_cmath.h:323-337 at c8fcbe8): two arguments
    struct max2 { static constexpr uint f(const uint& a, const uint& b) { return a >= b ? a : b; } };
}

// ---- stand-ins: a 48-byte operation data, an instantiable operation that derives from it, the Point, the num_elems functions ----
struct Point { uint x, y, z; constexpr Point(uint x_, uint y_, uint z_) : x(x_), y(y_), z(z_) {} };
struct OperationData { int dstW, dstH; float fx, fy; int srcW, srcH; void* data; unsigned width, height, pitch, pad; };   // 8 + 8 + 8 + 24 = 48 bytes
static_assert(sizeof(OperationData) == 48, "48 bytes per crop, as measured and read");
struct Instantiable final : public OperationData {};
struct Op {
    static constexpr uint num_elems_x(const Point&, const OperationData& d) { return static_cast<uint>(d.dstW); }
    static constexpr uint num_elems_y(const Point&, const OperationData& d) { return static_cast<uint>(d.dstH); }
};

template <size_t... Idx>
static uint paperFold(const std::array<Instantiable, sizeof...(Idx)>& instantiableOperations, const std::index_sequence<Idx...>&) {
    const uint max_width = cxp::max(Op::num_elems_x(Point(0u, 0u, 0u), instantiableOperations[Idx])...);
    const uint max_height = cxp::max(Op::num_elems_y(Point(0u, 0u, 0u), instantiableOperations[Idx])...);
    return max_width * 1000003u + max_height;
}
template <size_t B>
static uint loopNow(const std::array<Instantiable, B>& iOps) {
    uint max_width{0};
    uint max_height{0};
    for (int i = 0; i < static_cast<int>(B); ++i) {
        max_width = cxp::max2::f(max_width, Op::num_elems_x(Point{0, 0, 0}, iOps[i]));
        max_height = cxp::max2::f(max_height, Op::num_elems_y(Point{0, 0, 0}, iOps[i]));
    }
    return max_width * 1000003u + max_height;
}

static volatile uint sink;
template <typename F>
static double timePerCall(F&& f) {      // median over 7 blocks of the mean of a block of calls (the block is sized to about 30 ms)
    for (int i = 0; i < 20; ++i) f();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 20; ++i) f();
    const double once = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / 20;
    const int block = std::max(10, static_cast<int>(0.03 / std::max(once, 1e-8)));
    std::vector<double> t;
    for (int r = 0; r < 7; ++r) {
        auto a = std::chrono::steady_clock::now();
        for (int i = 0; i < block; ++i) f();
        t.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - a).count() / block);
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2] * 1e6;
}

template <size_t B>
static void one(const char* flagset, const unsigned seed) {
    std::array<Instantiable, B> arr{};
    for (size_t i = 0; i < B; ++i) { arr[i].dstW = 64 + static_cast<int>((i * seed) % 5); arr[i].dstH = 128 + static_cast<int>((i * (seed + 1)) % 7); }
    const double tPaper = timePerCall([&] { sink = paperFold(arr, std::make_index_sequence<B>{}); });
    const double tLoop = timePerCall([&] { sink = loopNow(arr); });
    std::printf("%s,%zu,%.4f,%.4f\n", flagset, B, tPaper, tLoop);
}

int main(int argc, char** argv) {
    const char* flagset = argc > 1 ? argv[1] : "unnamed";
    const unsigned seed = argc > 2 ? static_cast<unsigned>(std::atoi(argv[2])) : 3u;
    one<2>(flagset, seed); one<12>(flagset, seed); one<42>(flagset, seed); one<62>(flagset, seed); one<102>(flagset, seed); one<142>(flagset, seed); one<152>(flagset, seed);
    return 0;
}
