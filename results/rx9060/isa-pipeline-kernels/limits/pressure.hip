// Synthetic kernels that keep K floats live at one point, to read gfx1200's occupancy limits from the compiler's resource-usage remarks
#include <hip/hip_runtime.h>
template <int K>
__global__ void __launch_bounds__(256) pressure(const float* __restrict__ in, float* __restrict__ out) {
    float v[K];
#pragma unroll
    for (int i = 0; i < K; ++i) v[i] = in[threadIdx.x * K + i];
#pragma unroll
    for (int i = 0; i < K; ++i) asm volatile("" : "+v"(v[i]));
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < K; ++i) acc += v[i];
    out[threadIdx.x] = acc;
}
template __global__ void pressure<16>(const float*, float*);
template __global__ void pressure<40>(const float*, float*);
template __global__ void pressure<64>(const float*, float*);
template __global__ void pressure<90>(const float*, float*);
template __global__ void pressure<96>(const float*, float*);
template __global__ void pressure<100>(const float*, float*);
template __global__ void pressure<120>(const float*, float*);
template __global__ void pressure<128>(const float*, float*);
template __global__ void pressure<160>(const float*, float*);
template __global__ void pressure<192>(const float*, float*);
template __global__ void pressure<250>(const float*, float*);
