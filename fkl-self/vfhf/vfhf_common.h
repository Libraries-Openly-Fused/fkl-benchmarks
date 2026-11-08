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

// The data of benchmark_vfhf_max_speedup (benchmark_vfhf_max_speedup.h) and the declaration of its fused side, vfhf::fused<NUM_OPS>. The definition
// is in vfhf_fused.h, and each number of operations is instantiated in a translation unit of its own (generated from templates/vfhf_point.in),
// as the 2025 program opencv-cuda/verticalfusion/ does with one kernelN.cu per experiment: no translation unit compiles more than one of the
// long kernels.
#pragma once

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/parallel_architectures.h>
#include <fused_kernel/core/execution_model/stream.h>

#include <array>
#include <cstddef>

namespace vfhf {

// The data of the 2025 program (opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul_add: launcher.cu, realBatch.h): REAL_BATCH = 50
// images of 60x120 8 bit pixels, all 30 (val_init); the conversion to float with alpha = 1 (convertTo); the chain alternates a multiplication
// and an addition by 0.54 (val_mul), two operations per iteration (OPS_PER_ITERATION); its points are NUM_OPS = FIRST_VALUE + INCREMENT (k - 1)
// = 2 + 100 (k - 1); vertical_fusion_static_loop.cuh loops over INCREMENT operations at a time
constexpr int BATCH = 50, WIDTH = 60, HEIGHT = 120;
constexpr unsigned char INPUT_VALUE = 30;
constexpr float ALPHA = 1.0f, VALUE = 0.54f;
constexpr size_t OPS_PER_ITER = 2, FIRST_VALUE = 2, INCREMENT = 100;

// The fused side: one executeOperations for the whole batch (vfhf_fused.h)
template <size_t NUM_OPS>
void fused(fk::Stream_<fk::defaultParArch>& stream, const std::array<fk::Ptr2D<uchar>, BATCH>& images, const fk::Tensor<float>& output);

}  // namespace vfhf
