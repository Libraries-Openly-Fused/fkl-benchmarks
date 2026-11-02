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

// E11, mistake 1: the output type of an operation is not the input type of the next one.
// The chain reads a float3 image and multiplies it by a float3 (Mul<float3>: float3 -> float3), then adds a float (Add<float>: float -> float): the Add receives a float3.
// Compile only (never run); built with -DE11_WRONG for the wrong chain, without it for the correct one (all float3), which must compile with every compiler.
// One source for every back-end: fk::Stream and fk::TransformDPP<> take the parallel architecture of the compiler (nvcc: CUDA, amdclang++ -x hip: HIP, g++/clang++: CPU).
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/execution_model/executors.h>

int main() {
    constexpr int W = 64, H = 64;
    fk::Stream stream;
    fk::Ptr2D<fk::float3> in(W, H);
#ifdef E11_WRONG
    fk::Ptr2D<float> out(W, H);
#else
    fk::Ptr2D<fk::float3> out(W, H);
#endif
    fk::executeOperations<fk::TransformDPP<>>(stream,
        fk::PerThreadRead<fk::ND::_2D, fk::float3>::build(in),
        fk::Mul<fk::float3>::build(fk::make_set<fk::float3>(2.f)),
#ifdef E11_WRONG
        fk::Add<float>::build(1.f),                                    // the previous operation gives a float3
        fk::PerThreadWrite<fk::ND::_2D, float>::build(out));
#else
        fk::Add<fk::float3>::build(fk::make_set<fk::float3>(1.f)),
        fk::PerThreadWrite<fk::ND::_2D, fk::float3>::build(out));
#endif
    stream.sync();
    return 0;
}
