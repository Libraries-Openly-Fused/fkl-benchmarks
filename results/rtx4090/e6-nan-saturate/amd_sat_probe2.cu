// Probe for the AMD agent (FKL PR 416): cxp::saturate_cast on the device for float values at the limits of the output types and for rounding ties.
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/constexpr_libs/constexpr_saturate.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#define CHECK(x) do { auto e = (x); if (e != cudaSuccess) { printf("error %d line %d\n", (int)e, __LINE__); return 1; } } while (0)
__global__ void k(const float* in, const double* ind, int* out, int* outd, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i * 5 + 0] = cxp::saturate_cast<uchar>::f(in[i]);
        out[i * 5 + 1] = cxp::saturate_cast<signed char>::f(in[i]);
        out[i * 5 + 2] = cxp::saturate_cast<short>::f(in[i]);
        out[i * 5 + 3] = cxp::saturate_cast<ushort>::f(in[i]);
        out[i * 5 + 4] = cxp::saturate_cast<int>::f(in[i]);
        outd[i * 2 + 0] = cxp::saturate_cast<int>::f(ind[i]);
        outd[i * 2 + 1] = static_cast<int>(cxp::saturate_cast<unsigned int>::f(ind[i]) >> 1);  // unsigned int from double, halved to fit the printout
    }
}
int main() {
    const uint32_t bits[] = {0x4f000000u, 0x4effffffu, 0xcf000000u, 0xcf000001u, 0x437f8000u /*255.5*/, 0x437f0000u /*255*/, 0x437e8000u /*254.5*/, 0x3f000000u /*0.5*/,
                             0x3fc00000u /*1.5*/, 0x40200000u /*2.5*/, 0xbf000000u /*-0.5*/, 0xbfc00000u /*-1.5*/, 0x46fffe00u /*32767.0*/, 0x47000000u /*32768.0*/, 0x477fff00u /*65535.0*/, 0x47800000u /*65536.0*/};
    const char* names[] = {"2^31 (0x4f000000)", "2147483520 (0x4effffff)", "-2^31 (0xcf000000)", "-2147483904 (0xcf000001)", "255.5", "255", "254.5", "0.5", "1.5", "2.5", "-0.5", "-1.5", "32767", "32768", "65535", "65536"};
    constexpr int N = 16;
    float h[N]; double hd[N];
    for (int i = 0; i < N; ++i) { std::memcpy(&h[i], &bits[i], 4); hd[i] = static_cast<double>(h[i]); }
    float* d; double* dd; int* o; int* od; CHECK(cudaMalloc((void**)&d, N * 4)); CHECK(cudaMalloc((void**)&dd, N * 8)); CHECK(cudaMalloc((void**)&o, N * 5 * 4)); CHECK(cudaMalloc((void**)&od, N * 2 * 4));
    CHECK(cudaMemcpy(d, h, N * 4, cudaMemcpyHostToDevice)); CHECK(cudaMemcpy(dd, hd, N * 8, cudaMemcpyHostToDevice));
    k<<<1, 32>>>(d, dd, o, od, N); CHECK(cudaDeviceSynchronize());
    int r[N * 5], rd[N * 2]; CHECK(cudaMemcpy(r, o, N * 5 * 4, cudaMemcpyDeviceToHost)); CHECK(cudaMemcpy(rd, od, N * 2 * 4, cudaMemcpyDeviceToHost));
    printf("%-28s %8s %8s %8s %8s %12s | double->int  double->uint(>>1)\n", "float input", "uchar", "schar", "short", "ushort", "int");
    for (int i = 0; i < N; ++i) printf("%-28s %8d %8d %8d %8d %12d | %12d %12d\n", names[i], r[i*5], r[i*5+1], r[i*5+2], r[i*5+3], r[i*5+4], rd[i*2], rd[i*2+1]);
    return 0;
}
