/* Copyright 2024-2025 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <common/main.h>
#include <fkl-self/fkBenchmarksCommon.h>
#include <fkl-self/oneExecutionBenchmark.h>
 
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/executors.h>
#include <fused_kernel/core/execution_model/stream.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/static_loop.h>
#include <fused_kernel/core/data/ptr_utils.h>

constexpr char VARIABLE_DIMENSION_NAME[]{ "Number of Operations" };

constexpr size_t NUM_EXPERIMENTS = 60; // 1, 21, ... 1181 operations (the paper showed 30 points, up to 581)
constexpr size_t FIRST_VALUE = 1;
constexpr size_t INCREMENT = 20;

constexpr std::array<size_t, NUM_EXPERIMENTS> variableDimensionValues = arrayIndexSecuence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;

constexpr int NUM_ELEMENTS = 3840 * 2160 * 8;

template <typename InputType, typename OutputType, size_t NumOps, typename IOp>
struct VerticalFusion {
    static inline void execute(const fk::Ptr1D<InputType>& input, fk::Stream& stream,
                               const fk::Ptr1D<OutputType>& output, const IOp& dFunc) {
        const fk::ActiveThreads activeThreads{ output.ptr().dims.width };
        fk::Read<fk::PerThreadRead<fk::ND::_1D, InputType>> readDF{ {input.ptr()} };
        const fk::Write<fk::PerThreadWrite<fk::ND::_1D, OutputType>> writeDF{ {output.ptr()} };

        // StaticLoop<X, 0> executes X once (not zero times), so NumOps operations are built as
        // NumOps / INCREMENT repetitions of a loop of INCREMENT operations plus a loop for the NumOps % INCREMENT left
        // over. The first version of this benchmark ran INCREMENT operations for the point labelled 1 and
        // INCREMENT * (NumOps / INCREMENT) for the others.
        if constexpr (NumOps < INCREMENT) {
            using Loop = fk::Binary<fk::StaticLoop<typename IOp::Operation, NumOps>>;
            Loop loop;
            loop.params = dFunc.params;

            fk::executeOperations<fk::TransformDPP<>>(stream, readDF, loop, writeDF);
        } else {
            using Loop = fk::Binary<fk::StaticLoop<fk::StaticLoop<typename IOp::Operation, INCREMENT>, NumOps/INCREMENT>>;
            Loop loop;
            loop.params = dFunc.params;

            if constexpr (NumOps % INCREMENT == 0) {
                fk::executeOperations<fk::TransformDPP<>>(stream, readDF, loop, writeDF);
            } else {
                using RemainderLoop = fk::Binary<fk::StaticLoop<typename IOp::Operation, NumOps % INCREMENT>>;
                RemainderLoop remainderLoop;
                remainderLoop.params = dFunc.params;

                fk::executeOperations<fk::TransformDPP<>>(stream, readDF, loop, remainderLoop, writeDF);
            }
        }
    }
};

template <int VARIABLE_DIMENSION>
inline int testLatencyHiding(fk::Stream& stream) {
    fk::Ptr1D<float> input(NUM_ELEMENTS);
    fk::Ptr1D<float> output(NUM_ELEMENTS);

    constexpr float init_val{ 1 };

    fk::setTo(init_val, input, stream);

    using IOp = fk::Binary<fk::Mul<float>>;
    IOp df{ fk::make_set<float>(2) };

    // Warmup
    VerticalFusion<float, float, VARIABLE_DIMENSION, IOp>::execute(input, stream, output, df);

    START_FK_BENCHMARK

    VerticalFusion<float, float, VARIABLE_DIMENSION, IOp>::execute(input, stream, output, df);

    STOP_FK_BENCHMARK

    return 0;
}

template <int... Idx>
inline int testLatencyHidingHelper(fk::Stream& stream, const std::integer_sequence<int, Idx...>& seq) {
    const bool result = ((testLatencyHiding<variableDimensionValues[Idx]>(stream) == 0) && ...);
    if (result) {
        return 0;
    } else {
        return -1;
    }
}

int launch() {
    fk::Stream stream;

    const int result = testLatencyHidingHelper(stream, std::make_integer_sequence<int, variableDimensionValues.size()>{});

    CLOSE_BENCHMARK

    return result;

    return 0;
}
