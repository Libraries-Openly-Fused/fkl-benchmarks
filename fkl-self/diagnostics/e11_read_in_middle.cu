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

// E11, mistake 2: a complete Read operation in the middle of the chain.
// Read, Mul, Read (the second image), Add, Write: only the first operation of a chain can be a complete Read (a ReadBack operation, which takes the previous
// value as input, can follow). Compile only (never run); -DE11_WRONG adds the second Read, without it the chain is correct and must compile with every compiler.
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/execution_model/executors.h>

int main() {
    constexpr int W = 64, H = 64;
    fk::Stream stream;
    fk::Ptr2D<float> in(W, H), in2(W, H), out(W, H);
    fk::executeOperations<fk::TransformDPP<>>(stream,
        fk::PerThreadRead<fk::ND::_2D, float>::build(in),
        fk::Mul<float>::build(2.f),
#ifdef E11_WRONG
        fk::PerThreadRead<fk::ND::_2D, float>::build(in2),             // a complete Read in the middle of the chain
#endif
        fk::Add<float>::build(1.f),
        fk::PerThreadWrite<fk::ND::_2D, float>::build(out));
    stream.sync();
    return 0;
}
