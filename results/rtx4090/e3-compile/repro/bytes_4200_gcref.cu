#include <cuda_runtime.h>
struct Blob { float v[1050]; };
__device__ __forceinline__ float readAll(const Blob& X) { return X.v[0] + X.v[26] + X.v[52] + X.v[78] + X.v[104] + X.v[130] + X.v[156] + X.v[182] + X.v[208] + X.v[234] + X.v[260] + X.v[286] + X.v[312] + X.v[338] + X.v[364] + X.v[390] + X.v[416] + X.v[442] + X.v[468] + X.v[494] + X.v[520] + X.v[546] + X.v[572] + X.v[598] + X.v[624] + X.v[650] + X.v[676] + X.v[702] + X.v[728] + X.v[754] + X.v[780] + X.v[806] + X.v[832] + X.v[858] + X.v[884] + X.v[910] + X.v[936] + X.v[962] + X.v[988] + X.v[1014] + X.v[1040] + X.v[1049]; }
__global__ void k_gcref(const __grid_constant__ Blob b, float* out) { out[threadIdx.x] = readAll(b); }
struct Half { float v[525]; };
