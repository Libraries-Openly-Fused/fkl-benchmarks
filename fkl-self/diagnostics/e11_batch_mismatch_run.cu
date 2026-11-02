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

// E11, mistake 3, RUN: what the chain of e11_batch_mismatch.cu (BatchRead<50>, Mul, BatchWrite<60>) does when it is executed.
// 50 input images of 8x4 floats filled with 1, 60 output images filled with -1; the chain multiplies by 2. Each output image is classified by all of its
// pixels: processed (all 2), untouched (all -1) or mixed. Only the wrong chain with MORE OUTPUTS than inputs is run: with more inputs than outputs the
// last inputs would index past the end of the outputs (undefined behaviour), and that is not executed.
// The back-end is the one of the compiler: g++ / clang++ (-x c++): CPU; amdclang++ -x hip: HIP; nvcc: CUDA. -DE11_RUN_CPU_MT (with
// -DFK_CPU_MULTITHREAD_ENABLED, the thread-pool include directory and -pthread) selects the multithreaded CPU back-end.
#include <array>
#include <cstdio>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/execution_model/executors.h>

#if defined(E11_RUN_CPU_MT)
constexpr fk::ParArch ARCH = fk::ParArch::CPU_MULTITHREAD;
#else
constexpr fk::ParArch ARCH = fk::defaultParArch;
#endif

int main() {
    constexpr int W = 8, H = 4, IN = 50, OUT = 60;
    fk::Stream_<ARCH> stream;
    std::array<fk::Ptr2D<float>, IN> ins;
    std::array<fk::Ptr2D<float>, OUT> outs;
    for (auto& p : ins) {
        p = fk::Ptr2D<float>(W, H);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) p.at(x, y) = 1.f;
        p.upload(stream);
    }
    for (auto& p : outs) {
        p = fk::Ptr2D<float>(W, H);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) p.at(x, y) = -1.f;
        p.upload(stream);
    }
    stream.sync();
    fk::executeOperations<fk::TransformDPP<ARCH>>(stream,
        fk::PerThreadRead<fk::ND::_2D, float>::build(ins),
        fk::Mul<float>::build(2.f),
        fk::PerThreadWrite<fk::ND::_2D, float>::build(outs));
    for (auto& p : outs) p.download(stream);
    stream.sync();
    int processed = 0, untouched = 0, mixed = 0, firstUntouched = -1;
    for (int i = 0; i < OUT; ++i) {
        int twos = 0, minus = 0;
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            const float v = outs[i].at(x, y);
            if (v == 2.f) ++twos; else if (v == -1.f) ++minus;
        }
        if (twos == W * H) ++processed;
        else if (minus == W * H) { ++untouched; if (firstUntouched < 0) firstUntouched = i; }
        else ++mixed;
    }
    std::printf("back-end %s: %d inputs read, %d outputs: %d processed (all %d pixels = 2), %d untouched (all -1; the first is output %d), %d mixed\n",
                fk::toStrView(ARCH).data(), IN, OUT, processed, W * H, untouched, firstUntouched, mixed);
    return 0;
}
