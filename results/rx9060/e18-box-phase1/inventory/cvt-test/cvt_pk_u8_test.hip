// cvt_pk_u8_test.hip: how the RX 9060 (gfx1200) rounds in V_CVT_PK_U8_F32, the instruction behind RPP's U8 stores, and what RPP 3.2.0's
// rppt_box_filter 3x3 U8 really returns. Not a benchmark: nothing is timed.
//
// Part 1: __builtin_amdgcn_cvt_pk_u8_f32(x, 0, 0) (as RPP's rpp_hip_pack calls it, rpp_hip_load_store.hpp:434-440) on chosen inputs: ties x.5
//   for even and odd x, the float just below and just above each, integers, negatives, values above 255, NaN, +/-inf, a denormal and
//   254.99997f = 2295 * 0.1111111f. Each result is compared with four host rules: round to nearest, ties to even, saturating (RNE); truncation
//   toward zero, saturating (TRUNC); round half up, saturating (RHU); round half away from zero, saturating (RHA). NaN is shown as is.
//   Also: rpp_hip_pack's nested form on groups of 4 (must equal the single calls), and the byte selector (other bytes of the third operand kept).
// Part 2: one 64 x 32 U8 NHWC 3-channel image (tiles with a constant background and one changed pixel, so that window sums are 9v, 9v +/- 4,
//   9v +/- 5, ..., and tiles of LCG noise), rppt_box_filter kernelSize 3, REPLICATE, the whole image as the ROI, through the single-image path
//   (n = 1) and the batched path (n = 2, two copies). Every output value is compared with: round(sum / 9) (exact rational, no ties since 9 is
//   odd), floor(sum / 9), and the prediction "RPP's formula": f = (float)sum * 0.1111111f on the host, converted by the builtin on the GPU.
//   The sums are exact integer 3 x 3 sums with the border replicated, computed on the host.
// Build: amdclang++ -x hip --offload-arch=gfx1200 -O2 -std=c++17 cvt_pk_u8_test.hip -I/opt/rocm/core-10.1/include/rpp -L/opt/rocm/core-10.1/lib -lrpp
#include <hip/hip_runtime.h>
#include <rpp.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#define CHECK_HIP(x)                                                                                   \
    do {                                                                                               \
        const hipError_t e_ = (x);                                                                     \
        if (e_ != hipSuccess) {                                                                        \
            std::fprintf(stderr, "%s failed: %s (%s:%d)\n", #x, hipGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                              \
        }                                                                                              \
    } while (0)
#define CHECK_RPP(x)                                                                                   \
    do {                                                                                               \
        const int s_ = static_cast<int>(x);                                                            \
        if (s_ != 0) {                                                                                 \
            std::fprintf(stderr, "%s failed: RppStatus %d (%s:%d)\n", #x, s_, __FILE__, __LINE__);     \
            std::exit(2);                                                                              \
        }                                                                                              \
    } while (0)

__global__ void cvtSingle(const float* in, unsigned* out, const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __builtin_amdgcn_cvt_pk_u8_f32(in[i], 0, 0);
}

// rpp_hip_pack's nested form (rpp_hip_load_store.hpp:434-440) on groups of 4
__global__ void cvtPack4(const float* in, unsigned* out, const int groups) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g < groups) {
        const float4 src = make_float4(in[4 * g], in[4 * g + 1], in[4 * g + 2], in[4 * g + 3]);
        out[g] = __builtin_amdgcn_cvt_pk_u8_f32(
            src.w, 3, __builtin_amdgcn_cvt_pk_u8_f32(src.z, 2, __builtin_amdgcn_cvt_pk_u8_f32(src.y, 1, __builtin_amdgcn_cvt_pk_u8_f32(src.x, 0, 0))));
    }
}

// the byte selector: x into byte `sel` of `base`, the other bytes kept
__global__ void cvtSelect(const float x, const unsigned base, unsigned* out) {
    out[0] = __builtin_amdgcn_cvt_pk_u8_f32(x, 0, base);
    out[1] = __builtin_amdgcn_cvt_pk_u8_f32(x, 1, base);
    out[2] = __builtin_amdgcn_cvt_pk_u8_f32(x, 2, base);
    out[3] = __builtin_amdgcn_cvt_pk_u8_f32(x, 3, base);
}

static std::vector<unsigned> runBuiltin(const std::vector<float>& in) {
    float* dIn = nullptr;
    unsigned* dOut = nullptr;
    const int n = static_cast<int>(in.size());
    CHECK_HIP(hipMalloc(&dIn, sizeof(float) * n));
    CHECK_HIP(hipMalloc(&dOut, sizeof(unsigned) * n));
    CHECK_HIP(hipMemcpy(dIn, in.data(), sizeof(float) * n, hipMemcpyHostToDevice));
    cvtSingle<<<(n + 255) / 256, 256>>>(dIn, dOut, n);
    CHECK_HIP(hipGetLastError());
    std::vector<unsigned> out(n);
    CHECK_HIP(hipMemcpy(out.data(), dOut, sizeof(unsigned) * n, hipMemcpyDeviceToHost));
    CHECK_HIP(hipFree(dIn));
    CHECK_HIP(hipFree(dOut));
    return out;
}

static int sat(const double v) { return v <= 0. ? 0 : (v >= 255. ? 255 : static_cast<int>(v)); }
static int ruleRNE(const float x) { return std::isnan(x) ? -1 : sat(std::nearbyint(static_cast<double>(x))); }
static int ruleTRUNC(const float x) { return std::isnan(x) ? -1 : sat(std::trunc(static_cast<double>(x))); }
static int ruleRHU(const float x) { return std::isnan(x) ? -1 : sat(std::floor(static_cast<double>(x) + 0.5)); }
static int ruleRHA(const float x) { return std::isnan(x) ? -1 : sat(std::round(static_cast<double>(x))); }

static unsigned bitsOf(const float x) {
    unsigned u;
    std::memcpy(&u, &x, 4);
    return u;
}

int main() {
    hipDeviceProp_t prop;
    CHECK_HIP(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s (%s)\n", prop.name, prop.gcnArchName);

    // ---- Part 1
    std::vector<float> in;
    std::vector<std::string> what;
    const auto add = [&](const float x, const std::string& w) { in.push_back(x); what.push_back(w); };
    for (const float t : {0.5f, 1.5f, 2.5f, 3.5f, 127.5f, 128.5f, 253.5f, 254.5f}) {
        add(std::nextafter(t, 0.f), "just below tie");
        add(t, "tie");
        add(std::nextafter(t, 1000.f), "just above tie");
    }
    for (const float x : {0.f, 1.f, 2.f, 0.25f, 0.75f, 1.25f, 1.75f, 100.3f, 100.7f, 254.f, 255.f}) add(x, "plain");
    add(-0.f, "negative zero");
    for (const float x : {-0.25f, -0.5f, -0.75f, -1.f, -1.5f, -2.5f, -100.f}) add(x, "negative");
    add(std::nextafter(255.f, 1000.f), "just above 255");
    for (const float x : {255.25f, 255.5f, 255.75f, 256.f, 300.f, 1e10f, std::numeric_limits<float>::max()}) add(x, "above 255");
    add(std::numeric_limits<float>::denorm_min(), "denormal");
    add(std::numeric_limits<float>::quiet_NaN(), "NaN");
    add(-std::numeric_limits<float>::quiet_NaN(), "-NaN");
    add(std::numeric_limits<float>::infinity(), "+inf");
    add(-std::numeric_limits<float>::infinity(), "-inf");
    const volatile float inv9 = 0.1111111f, s2295 = 2295.f, inv49 = 0.02040816f, s12495 = 12495.f;
    add(s2295 * inv9, "255*9 * 0.1111111f (box 3x3 on 255)");
    add(s12495 * inv49, "255*49 * 0.02040816f (box 7x7 on 255)");
    add(900.f * inv9, "100*9 * 0.1111111f");
    while (in.size() % 4 != 0) add(1.5f, "padding");
    const std::vector<unsigned> got = runBuiltin(in);
    int agree[4] = {0, 0, 0, 0}, nonNan = 0;
    std::printf("\nPart 1: __builtin_amdgcn_cvt_pk_u8_f32(x, 0, 0)\n");
    std::printf("%-40s %-16s %-12s %6s %5s %5s %5s %5s\n", "input", "value", "bits", "GPU", "RNE", "TRUNC", "RHU", "RHA");
    FILE* csv = std::fopen("part1_builtin.csv", "w");
    std::fprintf(csv, "what,value,bits,gpu_word,gpu_byte0,rne,trunc,rhu,rha\n");
    for (size_t i = 0; i < in.size(); ++i) {
        const float x = in[i];
        const int g = static_cast<int>(got[i] & 0xffu);
        const int r[4] = {ruleRNE(x), ruleTRUNC(x), ruleRHU(x), ruleRHA(x)};
        if (!std::isnan(x)) {
            ++nonNan;
            for (int k = 0; k < 4; ++k) agree[k] += (r[k] == g) ? 1 : 0;
        }
        std::printf("%-40s %-16.9g 0x%08x %6d %5d %5d %5d %5d%s\n", what[i].c_str(), static_cast<double>(x), bitsOf(x), g, r[0], r[1], r[2], r[3],
                    (got[i] >> 8) != 0 ? "  (upper bytes not 0!)" : "");
        std::fprintf(csv, "%s,%.9g,0x%08x,0x%08x,%d,%d,%d,%d,%d\n", what[i].c_str(), static_cast<double>(x), bitsOf(x), got[i], g, r[0], r[1], r[2], r[3]);
    }
    std::fclose(csv);
    std::printf("agreement over %d non-NaN inputs: RNE %d, TRUNC %d, RHU %d, RHA %d\n", nonNan, agree[0], agree[1], agree[2], agree[3]);
    {
        // the nested form of rpp_hip_pack must give the same bytes
        float* dIn = nullptr;
        unsigned* dOut = nullptr;
        const int groups = static_cast<int>(in.size() / 4);
        CHECK_HIP(hipMalloc(&dIn, sizeof(float) * in.size()));
        CHECK_HIP(hipMalloc(&dOut, sizeof(unsigned) * groups));
        CHECK_HIP(hipMemcpy(dIn, in.data(), sizeof(float) * in.size(), hipMemcpyHostToDevice));
        cvtPack4<<<(groups + 255) / 256, 256>>>(dIn, dOut, groups);
        CHECK_HIP(hipGetLastError());
        std::vector<unsigned> packed(groups);
        CHECK_HIP(hipMemcpy(packed.data(), dOut, sizeof(unsigned) * groups, hipMemcpyDeviceToHost));
        int mismatches = 0;
        for (int gidx = 0; gidx < groups; ++gidx) {
            for (int k = 0; k < 4; ++k) mismatches += ((packed[gidx] >> (8 * k)) & 0xffu) != (got[4 * gidx + k] & 0xffu) ? 1 : 0;
        }
        std::printf("rpp_hip_pack nested form vs single calls: %d of %zu bytes differ\n", mismatches, in.size());
        unsigned* dSel = nullptr;
        CHECK_HIP(hipMalloc(&dSel, sizeof(unsigned) * 4));
        cvtSelect<<<1, 1>>>(2.5f, 0xAABBCCDDu, dSel);
        CHECK_HIP(hipGetLastError());
        unsigned sel[4];
        CHECK_HIP(hipMemcpy(sel, dSel, sizeof(sel), hipMemcpyDeviceToHost));
        std::printf("byte selector, x = 2.5 into 0xAABBCCDD: sel0 0x%08x, sel1 0x%08x, sel2 0x%08x, sel3 0x%08x\n", sel[0], sel[1], sel[2], sel[3]);
        CHECK_HIP(hipFree(dIn));
        CHECK_HIP(hipFree(dOut));
        CHECK_HIP(hipFree(dSel));
    }

    // ---- Part 2: rppt_box_filter 3x3 U8 NHWC 3 channels
    constexpr int W = 64, H = 32, C = 3, TILE = 8;
    std::vector<unsigned char> img(static_cast<size_t>(W) * H * C);
    {
        // tiles 0..15: background v and one pixel at (4, 4) of the tile changed by d[c]; tiles 16..31: LCG noise
        const int vs[16] = {0, 1, 2, 5, 9, 10, 100, 127, 128, 200, 245, 250, 251, 253, 254, 255};
        const int ds[16][3] = {{0, 0, 0}, {4, 5, 0},  {4, 5, 0},   {4, 5, -4},  {4, 5, -5}, {-4, -5, 4}, {4, 5, -4}, {-5, 4, 5},
                               {5, -4, -5}, {4, -5, 1}, {4, 5, -5}, {4, 5, -4}, {4, -4, -5}, {-4, -5, 0}, {0, -4, -5}, {0, 0, 0}};
        uint32_t state = 12345u;
        for (int ty = 0; ty < H / TILE; ++ty) {
            for (int tx = 0; tx < W / TILE; ++tx) {
                const int t = ty * (W / TILE) + tx;
                for (int y = 0; y < TILE; ++y) {
                    for (int x = 0; x < TILE; ++x) {
                        for (int c = 0; c < C; ++c) {
                            int v;
                            if (t < 16) {
                                v = vs[t] + ((x == 4 && y == 4) ? ds[t][c] : 0);
                                v = v < 0 ? 0 : (v > 255 ? 255 : v);
                            } else {
                                state = state * 1664525u + 1013904223u;
                                v = static_cast<int>(state >> 24);
                            }
                            img[(static_cast<size_t>(ty * TILE + y) * W + tx * TILE + x) * C + c] = static_cast<unsigned char>(v);
                        }
                    }
                }
            }
        }
    }
    // exact 3 x 3 sums with the border replicated
    std::vector<int> sums(img.size());
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            for (int c = 0; c < C; ++c) {
                int s = 0;
                for (int dy = -1; dy <= 1; ++dy) {
                    for (int dx = -1; dx <= 1; ++dx) {
                        const int sx = std::min(std::max(x + dx, 0), W - 1), sy = std::min(std::max(y + dy, 0), H - 1);
                        s += img[(static_cast<size_t>(sy) * W + sx) * C + c];
                    }
                }
                sums[(static_cast<size_t>(y) * W + x) * C + c] = s;
            }
        }
    }
    // the prediction from RPP's formula: (float)sum * 0.1111111f on the host, converted by the builtin on the GPU
    std::vector<float> products(sums.size());
    for (size_t i = 0; i < sums.size(); ++i) products[i] = static_cast<float>(sums[i]) * inv9;
    const std::vector<unsigned> predicted = runBuiltin(products);

    hipStream_t stream;
    CHECK_HIP(hipStreamCreate(&stream));
    constexpr unsigned OFFSET = 64;   // rppt_box_filter's HIP path requires srcDesc offsetInBytes >= 12 * (kernelSize / 2)
    FILE* csv2 = std::fopen("part2_box3x3.csv", "w");
    std::fprintf(csv2, "path,x,y,c,sum,sum_mod_9,rpp,round_sum_div_9,floor_sum_div_9,float_product,predicted_builtin\n");
    std::vector<unsigned char> outFirst;
    for (const int n : {1, 2}) {
        const size_t imageBytes = img.size();
        unsigned char *dSrc = nullptr, *dDst = nullptr;
        CHECK_HIP(hipMalloc(&dSrc, OFFSET + imageBytes * n));
        CHECK_HIP(hipMalloc(&dDst, imageBytes * n));
        CHECK_HIP(hipMemset(dSrc, 0, OFFSET + imageBytes * n));
        CHECK_HIP(hipMemset(dDst, 0xEE, imageBytes * n));
        for (int k = 0; k < n; ++k) CHECK_HIP(hipMemcpy(dSrc + OFFSET + k * imageBytes, img.data(), imageBytes, hipMemcpyHostToDevice));
        RpptDesc srcDesc(n, C, H, W, RpptDataType::U8, RpptLayout::NHWC);
        srcDesc.offsetInBytes = OFFSET;
        RpptDesc dstDesc(n, C, H, W, RpptDataType::U8, RpptLayout::NHWC);
        RpptROI* dRoi = nullptr;
        CHECK_HIP(hipMalloc(&dRoi, sizeof(RpptROI) * 256));   // the ROI conversion kernels write up to 256 entries
        std::vector<RpptROI> rois(256);
        std::memset(rois.data(), 0, sizeof(RpptROI) * 256);
        for (int k = 0; k < n; ++k) {
            rois[k].xywhROI.xy.x = 0;
            rois[k].xywhROI.xy.y = 0;
            rois[k].xywhROI.roiWidth = W;
            rois[k].xywhROI.roiHeight = H;
        }
        CHECK_HIP(hipMemcpy(dRoi, rois.data(), sizeof(RpptROI) * 256, hipMemcpyHostToDevice));
        rppHandle_t handle;
        CHECK_RPP(rppCreate(&handle, n, 0, stream, RppBackend::RPP_HIP_BACKEND));
        CHECK_RPP(rppt_box_filter(dSrc, &srcDesc, dDst, &dstDesc, 3, RpptImageBorderType::REPLICATE, dRoi, RpptRoiType::XYWH, handle,
                                  RppBackend::RPP_HIP_BACKEND));
        CHECK_HIP(hipStreamSynchronize(stream));
        CHECK_HIP(hipGetLastError());
        std::vector<unsigned char> out(imageBytes * n);
        CHECK_HIP(hipMemcpy(out.data(), dDst, imageBytes * n, hipMemcpyDeviceToHost));
        CHECK_RPP(rppDestroy(handle, RppBackend::RPP_HIP_BACKEND));
        const char* path = n == 1 ? "single-image (n=1)" : "batched (n=2)";
        // per residue class of the sum: how many values equal round(sum/9), floor(sum/9), the prediction
        long cnt[9] = {}, eqRound[9] = {}, eqFloor[9] = {}, eqPred[9] = {}, below[9] = {}, imagesEqual = 1;
        for (int k = 0; k < n; ++k) {
            for (size_t i = 0; i < imageBytes; ++i) {
                const int s = sums[i], r = s % 9, rpp = out[k * imageBytes + i];
                const int rnd = (2 * s + 9) / 18;   // nearest integer to s / 9 (no ties: 9 is odd)
                const int flo = s / 9;
                const int pred = static_cast<int>(predicted[i] & 0xffu);
                ++cnt[r];
                eqRound[r] += rpp == rnd;
                eqFloor[r] += rpp == flo;
                eqPred[r] += rpp == pred;
                below[r] += rpp < rnd;
                if (k == 0) {
                    const size_t p = i / C;
                    std::fprintf(csv2, "%s,%zu,%zu,%zu,%d,%d,%d,%d,%d,%.9g,%d\n", n == 1 ? "n1" : "n2", p % W, p / W, i % C, s, r, rpp, rnd, flo,
                                 static_cast<double>(products[i]), pred);
                }
                if (k > 0 && out[k * imageBytes + i] != out[i]) imagesEqual = 0;
            }
        }
        std::printf("\nPart 2: rppt_box_filter 3x3 U8 NHWC, %s, %d x %d x 3, %ld values per image\n", path, W, H, static_cast<long>(imageBytes));
        std::printf("%-10s %8s %14s %14s %16s %12s\n", "sum mod 9", "values", "== round(s/9)", "== floor(s/9)", "== RPP formula", "< round");
        long total = 0, tRound = 0, tFloor = 0, tPred = 0;
        for (int r = 0; r < 9; ++r) {
            std::printf("%-10d %8ld %14ld %14ld %16ld %12ld\n", r, cnt[r], eqRound[r], eqFloor[r], eqPred[r], below[r]);
            total += cnt[r];
            tRound += eqRound[r];
            tFloor += eqFloor[r];
            tPred += eqPred[r];
        }
        std::printf("%-10s %8ld %14ld %14ld %16ld\n", "all", total, tRound, tFloor, tPred);
        if (n == 2) std::printf("the two images of the batch are %s\n", imagesEqual ? "identical" : "DIFFERENT");
        if (n == 1) outFirst.assign(out.begin(), out.begin() + imageBytes);
        if (n == 2) std::printf("batched image 0 vs single-image output: %s\n", std::memcmp(outFirst.data(), out.data(), imageBytes) == 0 ? "identical" : "DIFFERENT");
        // the constant tiles: background v, output at the tile's (1, 1), whose window is inside the tile
        if (n == 1) {
            std::printf("constant 3 x 3 windows (tile (1,1) pixel, channel 0): ");
            for (int t = 0; t < 16; ++t) {
                const int ty = t / (W / TILE), tx = t % (W / TILE);
                const size_t i = (static_cast<size_t>(ty * TILE + 1) * W + tx * TILE + 1) * C;
                std::printf("%d->%d%s", img[i], out[i], t < 15 ? ", " : "\n");
            }
        }
        CHECK_HIP(hipFree(dSrc));
        CHECK_HIP(hipFree(dDst));
        CHECK_HIP(hipFree(dRoi));
    }
    std::fclose(csv2);
    CHECK_HIP(hipStreamDestroy(stream));
    return 0;
}
