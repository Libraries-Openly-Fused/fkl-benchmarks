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

// The values of a sweep, START, START + STEP, ..., START + (N - 1) * STEP, as a constexpr std::array<size_t, N>:
//   constexpr std::array<size_t, NUM_EXPERIMENTS> batchValues = fklb::arrayIndexSequence<FIRST_VALUE, INCREMENT, NUM_EXPERIMENTS>;
// It is arrayIndexSecuence of FastNPP's tests/testsNppCommon.h (the same values), moved here with its spelling corrected, so that the NPP
// programs no longer include FastNPP's test header.
#pragma once

#include <array>
#include <cstddef>
#include <utility>

namespace fklb {

namespace detail {
template <size_t START, size_t STEP, size_t... Is>
constexpr std::array<size_t, sizeof...(Is)> arithmeticSequence(std::index_sequence<Is...>) {
    return {{(START + STEP * Is)...}};
}
}  // namespace detail

template <size_t START, size_t STEP, size_t N>
constexpr std::array<size_t, N> arrayIndexSequence = detail::arithmeticSequence<START, STEP>(std::make_index_sequence<N>{});

}  // namespace fklb
