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

// The fused side of benchmark_vfhf_max_speedup: the definition of vfhf::fused<NUM_OPS>, included only by the generated translation unit of each
// number of operations (templates/vfhf_point.in).
//
// It is the cvGPUSpeedup side of the 2025 program written with FKL: opencv-cuda/verticalfusion/vertical_fusion_static_loop.cuh builds the chain
// as one fk::StaticLoop of the fused pair Mul(0.54) then Add(0.54) when NUM_OPS < INCREMENT, and otherwise as a StaticLoop of a StaticLoop of
// INCREMENT / 2 pairs, repeated NUM_OPS / INCREMENT times, followed by a StaticLoop of the (NUM_OPS % INCREMENT) / 2 remaining pairs when
// NUM_OPS % INCREMENT is not 0. It then calls cvGS::executeOperations<false>(crops, stream, cvGS::convertTo<CV_8UC1, CV_32FC1>(alpha), loop,
// [remainder,] cvGS::write<CV_32FC1>(output, cropSize)), which is fk::executeOperations with thread fusion off, the batch read of the 50 images
// (one read for all of them), the conversion (fk::SaturateCast<uchar, float> then fk::Mul<float> by alpha, the two operations of convertTo with
// alpha) and fk::PerThreadWrite<ND::_3D, float> into a tensor of 50 planes of 60x120. The same operations here, written out in FKL.
#pragma once

#include "vfhf_common.h"

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/static_loop.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/core/execution_model/executors.h>

namespace vfhf {

template <size_t NUM_OPS>
void fused(fk::Stream_<fk::defaultParArch>& stream, const std::array<fk::Ptr2D<uchar>, BATCH>& images, const fk::Tensor<float>& output) {
    static_assert(NUM_OPS >= OPS_PER_ITER && NUM_OPS % OPS_PER_ITER == 0, "the chain is made of pairs of a multiplication and an addition");
    const auto read = fk::PerThreadRead<fk::ND::_2D, uchar>::build(images);
    const auto convert = fk::SaturateCast<uchar, float>::build().then(fk::Mul<float>::build(ALPHA));
    const auto mulAdd = fk::Mul<float>::build(VALUE).then(fk::Add<float>::build(VALUE));
    const auto write = fk::PerThreadWrite<fk::ND::_3D, float>::build(output.ptr());
    if constexpr (NUM_OPS < INCREMENT) {
        const auto loop = fk::StaticLoop<typename decltype(mulAdd)::Operation, NUM_OPS / OPS_PER_ITER>::build(mulAdd.params);
        fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(stream, read, convert, loop, write);
    } else {
        const auto loop = fk::StaticLoop<fk::StaticLoop<typename decltype(mulAdd)::Operation, INCREMENT / OPS_PER_ITER>, NUM_OPS / INCREMENT>::build(mulAdd.params);
        if constexpr (NUM_OPS % INCREMENT == 0) {
            fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(stream, read, convert, loop, write);
        } else {
            const auto remainder = fk::StaticLoop<typename decltype(mulAdd)::Operation, (NUM_OPS % INCREMENT) / OPS_PER_ITER>::build(mulAdd.params);
            fk::executeOperations<fk::TransformDPP<fk::defaultParArch, fk::TF::DISABLED>>(stream, read, convert, loop, remainder, write);
        }
    }
}

}  // namespace vfhf
