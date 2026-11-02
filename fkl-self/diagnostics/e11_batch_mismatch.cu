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

// E11, mistake 3: a batch of 50 images read and a batch of 60 images written.
// BatchRead<50> (PerThreadRead of std::array<Ptr2D, 50>), Mul, BatchWrite<60> (PerThreadWrite of std::array<Ptr2D, 60>); the correct chain writes 50.
// Compile only (never run); -DE11_WRONG for the wrong chain.
#include <array>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/execution_model/executors.h>

int main() {
    fk::Stream stream;
    std::array<fk::Ptr2D<float>, 50> ins;
#ifdef E11_WRONG
    std::array<fk::Ptr2D<float>, 60> outs;                             // 60 outputs for 50 inputs
#else
    std::array<fk::Ptr2D<float>, 50> outs;
#endif
    fk::executeOperations<fk::TransformDPP<>>(stream,
        fk::PerThreadRead<fk::ND::_2D, float>::build(ins),
        fk::Mul<float>::build(2.f),
        fk::PerThreadWrite<fk::ND::_2D, float>::build(outs));
    stream.sync();
    return 0;
}
