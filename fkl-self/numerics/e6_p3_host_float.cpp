// CPU-only check of E6 P3: host float32 implementation of FKL's bilinear resize (same formulas as external/fkl interpolation.h / resize.h) against the double evaluation and the device outputs.
// Inputs: the P3 noise images of seeds 1100..1104, regenerated from the same std::mt19937 stream as e6_fused_vs_unfused.cu (P2's noise and positions are drawn first).
#include <common/resize_reference.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>
using namespace std;
static const float ALPHA = 0.3f, SUBV[3] = {1.f, 4.f, 3.2f}, DIVV[3] = {3.2f, 0.6f, 11.8f};
static int64_t mono(float f) { uint32_t u; memcpy(&u, &f, 4); return (u & 0x80000000u) ? -(int64_t)(u & 0x7fffffffu) : (int64_t)u; }
static vector<float> readBin(const string& p) { ifstream f(p, ios::binary | ios::ate); size_t n = f.tellg() / 4; vector<float> v(n); f.seekg(0); f.read((char*)v.data(), n * 4); return v; }
// FKL: factor = float(1.0 / (dst / src)) computed in double; ASYM src = x * factor; HALF src = max((x + 0.5f) * factor - 0.5f, 0); x1 = floor, x2 = x1 + 1, x2r = min(x2, w - 1);
// result = ((p00 * ((x2 - sx) * (y2 - sy)) + p10 * ((sx - x1) * (y2 - sy))) + p01 * ((x2 - sx) * (sy - y1))) + p11 * ((sx - x1) * (sy - y1)), all float; then reorder, Mul, Sub, Div in float.
static vector<float> hostFloat(const vector<unsigned char>& img, int W, int H, int OW, int OH, bool half) {
    const float fx = (float)(1.0 / ((double)OW / (double)W)), fy = (float)(1.0 / ((double)OH / (double)H));
    vector<float> out((size_t)3 * OW * OH);
    for (int y = 0; y < OH; ++y) for (int x = 0; x < OW; ++x) {
        volatile float sx = half ? max((x + 0.5f) * fx - 0.5f, 0.f) : x * fx, sy = half ? max((y + 0.5f) * fy - 0.5f, 0.f) : y * fy;
        const int x1 = (int)floorf(sx), y1 = (int)floorf(sy), x2 = x1 + 1, y2 = y1 + 1, x2r = min(x2, W - 1), y2r = min(y2, H - 1);
        for (int c = 0; c < 3; ++c) {
            auto P = [&](int xx, int yy) { return (float)img[((size_t)yy * W + xx) * 3 + c]; };
            volatile float w00 = (x2 - sx) * (y2 - sy), w10 = (sx - x1) * (y2 - sy), w01 = (x2 - sx) * (sy - y1), w11 = (sx - x1) * (sy - y1);
            volatile float t0 = P(x1, y1) * w00, t1 = P(x2r, y1) * w10, t2 = P(x1, y2r) * w01, t3 = P(x2r, y2r) * w11;
            volatile float s = t0 + t1; s = s + t2; s = s + t3;
            const int k = 2 - c;   // output channel k takes source channel 2 - k
            volatile float m = s * ALPHA; m = m - SUBV[k]; m = m / DIVV[k];
            out[((size_t)k * OH + y) * OW + x] = m;
        }
    }
    return out;
}
static vector<float> hostCoordDouble(const vector<unsigned char>& img, int W, int H, int OW, int OH, bool half) {
    const double fx = (double)W / OW, fy = (double)H / OH;
    vector<float> out((size_t)3 * OW * OH);
    for (int y = 0; y < OH; ++y) for (int x = 0; x < OW; ++x) {
        double sx = half ? max((x + 0.5) * fx - 0.5, 0.0) : x * fx, sy = half ? max((y + 0.5) * fy - 0.5, 0.0) : y * fy;
        const int x1 = (int)floor(sx), y1 = (int)floor(sy), x2 = x1 + 1, y2 = y1 + 1, x2r = min(x2, W - 1), y2r = min(y2, H - 1);
        for (int c = 0; c < 3; ++c) {
            auto P = [&](int xx, int yy) { return (double)img[((size_t)yy * W + xx) * 3 + c]; };
            const double v = P(x1, y1) * ((x2 - sx) * (y2 - sy)) + P(x2r, y1) * ((sx - x1) * (y2 - sy)) + P(x1, y2r) * ((x2 - sx) * (sy - y1)) + P(x2r, y2r) * ((sx - x1) * (sy - y1));
            const int k = 2 - c;
            volatile float s = (float)v; volatile float m = s * ALPHA; m = m - SUBV[k]; m = m / DIVV[k];
            out[((size_t)k * OH + y) * OW + x] = m;
        }
    }
    return out;
}
int main() {
    const int SW = 3840, SH = 2160, W = 1920, H = 1080, OW = 224, OH = 224;
    printf("seed conv | host float32 vs double: max abs, max abs / max|out|, max ULP | host float32 vs device (fmad_on / fmad_off): identical elements, max abs, max ULP | device vs double: max abs\n");
    for (int seed = 0; seed < 5; ++seed) {
        mt19937 rng(1000 + 100 + seed);
        vector<unsigned char> n2((size_t)SW * SH * 3);
        for (auto& v : n2) v = (unsigned char)rng();
        { const int mx = 3840 - 60, my = 2160 - 120; (void)(rng() % (mx - 2)); (void)(rng() % (my - 2)); uniform_int_distribution<int> dx(1, mx - 1), dy(1, my - 1); for (int i = 0; i < 44; ++i) { (void)dx(rng); (void)dy(rng); } }
        vector<unsigned char> img((size_t)W * H * 3);
        for (auto& v : img) v = (unsigned char)rng();
        for (int half = 0; half < 2; ++half) {
            const vector<float> hf = hostFloat(img, W, H, OW, OH, half);
            vector<double> res((size_t)OW * OH * 3);
            fklb::resizeBilinear<unsigned char>(img.data(), (size_t)W * 3, W, H, 3, OW, OH, half ? fklb::ResizeSampling::HALF_PIXEL : fklb::ResizeSampling::ASYMMETRIC, res.data());
            vector<double> ref(hf.size());
            double maxOut = 0;
            for (int y = 0; y < OH; ++y) for (int x = 0; x < OW; ++x) for (int k = 0; k < 3; ++k) {
                const double v = (res[((size_t)y * OW + x) * 3 + (2 - k)] * (double)ALPHA - (double)SUBV[k]) / (double)DIVV[k];
                ref[((size_t)k * OH + y) * OW + x] = v; maxOut = max(maxOut, fabs(v));
            }
            double aHD = 0; int64_t uHD = 0;
            for (size_t i = 0; i < hf.size(); ++i) { aHD = max(aHD, fabs((double)hf[i] - ref[i])); uHD = max(uHD, (int64_t)llabs(mono(hf[i]) - mono((float)ref[i]))); }
            { const vector<float> hd = hostCoordDouble(img, W, H, OW, OH, half); double a = 0; for (size_t i = 0; i < hd.size(); ++i) a = max(a, fabs((double)hd[i] - ref[i])); printf("seed %d %s coordinates+weights in double, chain in float: max abs vs double %.3g (%.2e of max|out|)\n", seed, half ? "HALF" : "ASYM", a, a / maxOut); }
            string line = "seed " + to_string(seed) + (half ? " HALF " : " ASYM ");
            char buf[400];
            snprintf(buf, sizeof buf, "| %.3g, %.2e, %lld ", aHD, aHD / maxOut, (long long)uHD); line += buf;
            for (const char* tag : {"fmad_on", "fmad_off"}) {
                const vector<float> dev = readBin(string("/home/oscar/tfkl-bench/results/cuda13.4/e6-raw/") + tag + "/P3_" + (half ? "HALF_PIXEL" : "ASYMMETRIC") + "_seed" + to_string(seed) + "_fused.bin");
                size_t same = 0; double aHV = 0, aDV = 0; int64_t uHV = 0;
                for (size_t i = 0; i < hf.size(); ++i) { same += memcmp(&hf[i], &dev[i], 4) == 0; aHV = max(aHV, fabs((double)hf[i] - dev[i])); uHV = max(uHV, (int64_t)llabs(mono(hf[i]) - mono(dev[i]))); aDV = max(aDV, fabs((double)dev[i] - ref[i])); }
                snprintf(buf, sizeof buf, "| %s: %zu of %zu identical, max abs %.3g, max ULP %lld, dev-vs-double %.3g ", tag, same, hf.size(), aHV, (long long)uHV, aDV); line += buf;
            }
            puts(line.c_str());
        }
    }
}
