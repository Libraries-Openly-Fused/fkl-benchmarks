#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/constexpr_libs/constexpr_saturate.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#define CHECK(x) do { auto e = (x); if (e != cudaSuccess) { printf("error %d line %d\n", (int)e, __LINE__); return 1; } } while (0)
__global__ void k(const float* in, int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i * 5 + 0] = cxp::saturate_cast<uchar>::f(in[i]);
        out[i * 5 + 1] = cxp::saturate_cast<signed char>::f(in[i]);
        out[i * 5 + 2] = cxp::saturate_cast<short>::f(in[i]);
        out[i * 5 + 3] = cxp::saturate_cast<ushort>::f(in[i]);
        out[i * 5 + 4] = cxp::saturate_cast<int>::f(in[i]);
    }
}
int main() {
    const uint32_t bits[] = {0x7fc00000u, 0xffc00000u, 0x7fa00000u, 0xffa00000u, 0x7f800001u, 0x7fffffffu, 0x7f800000u, 0xff800000u};
    const char* names[] = {"quiet NaN +", "quiet NaN -", "signaling NaN +", "signaling NaN -", "signaling NaN + (payload 1)", "NaN +, all payload bits", "+inf", "-inf"};
    constexpr int N = 8;
    float h[N]; for (int i = 0; i < N; ++i) std::memcpy(&h[i], &bits[i], 4);
    float* d; int* o; CHECK(cudaMalloc((void**)&d, N * 4)); CHECK(cudaMalloc((void**)&o, N * 5 * 4));
    CHECK(cudaMemcpy(d, h, N * 4, cudaMemcpyHostToDevice)); k<<<1, 32>>>(d, o, N); CHECK(cudaDeviceSynchronize());
    int r[N * 5]; CHECK(cudaMemcpy(r, o, N * 5 * 4, cudaMemcpyDeviceToHost));
    printf("%-30s %8s %8s %8s %8s %12s\n", "input", "uchar", "schar", "short", "ushort", "int");
    for (int i = 0; i < N; ++i) printf("%-30s %8d %8d %8d %8d %12d\n", names[i], r[i*5], r[i*5+1], r[i*5+2], r[i*5+3], r[i*5+4]);
    return 0;
}
