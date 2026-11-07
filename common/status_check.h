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

// Status checks around library calls: one macro per library, used the same way on every compared side (the clean-up rules of 2026-11-06:
// status checks are infrastructure, with one shared macro per library in common/).
// A failed call throws std::runtime_error("<call> failed: <status> (<file>:<line>)"). The program's main catches it, prints the message and
// returns 1, and an fklb::CsvWriter that is destroyed while the exception propagates marks its file INCOMPLETE (common/csv_writer.h).
// Every status other than success is a failure; for NPP that includes the warnings, as the checks before these macros did.
// The header includes no library header: each program includes the headers of the libraries it calls, so a HIP build never sees CUDA and the
// other way round. The macros are variadic, so that a call with a top-level comma (a template argument list) can be checked as it is.
//
//   FKLB_CUDA_CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
//   FKLB_CUDA_CHECK(cudaGetLastError());                                    // after a kernel launch
//   FKLB_NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(src, srcStep, dst, dstStep, size, nppContext));
//   FKLB_HIP_CHECK(hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice));
//   FKLB_RPP_CHECK(rppt_channel_permute_gpu(src, srcDesc, dst, dstDesc, order, handle));
#pragma once

#include <stdexcept>
#include <string>

namespace fklb {
// the text of a failed check
inline std::string statusMessage(const char* call, const std::string& status, const char* file, const int line) {
    return std::string(call) + " failed: " + status + " (" + file + ":" + std::to_string(line) + ")";
}
}  // namespace fklb

// CUDA runtime: cudaError_t, success is cudaSuccess
#define FKLB_CUDA_CHECK(...)                                                                                                       \
    do {                                                                                                                           \
        const cudaError_t fklbStatus_ = (__VA_ARGS__);                                                                             \
        if (fklbStatus_ != cudaSuccess) {                                                                                          \
            throw std::runtime_error(fklb::statusMessage(#__VA_ARGS__,                                                             \
                                                         std::string(cudaGetErrorName(fklbStatus_)) + " (" +                       \
                                                             cudaGetErrorString(fklbStatus_) + ")",                                \
                                                         __FILE__, __LINE__));                                                     \
        }                                                                                                                          \
    } while (0)

// NPP: NppStatus, success is NPP_SUCCESS (0); errors are negative and warnings positive, and both count as a failure
#define FKLB_NPP_CHECK(...)                                                                                                        \
    do {                                                                                                                           \
        const NppStatus fklbStatus_ = (__VA_ARGS__);                                                                               \
        if (fklbStatus_ != NPP_SUCCESS) {                                                                                          \
            throw std::runtime_error(fklb::statusMessage(#__VA_ARGS__, "NppStatus " + std::to_string(static_cast<int>(fklbStatus_)), \
                                                         __FILE__, __LINE__));                                                     \
        }                                                                                                                          \
    } while (0)

// HIP runtime: hipError_t, success is hipSuccess
#define FKLB_HIP_CHECK(...)                                                                                                        \
    do {                                                                                                                           \
        const hipError_t fklbStatus_ = (__VA_ARGS__);                                                                              \
        if (fklbStatus_ != hipSuccess) {                                                                                           \
            throw std::runtime_error(fklb::statusMessage(#__VA_ARGS__,                                                             \
                                                         std::string(hipGetErrorName(fklbStatus_)) + " (" +                        \
                                                             hipGetErrorString(fklbStatus_) + ")",                                 \
                                                         __FILE__, __LINE__));                                                     \
        }                                                                                                                          \
    } while (0)

// RPP: RppStatus, success is RPP_SUCCESS (0); compared as an int, as RPPB_CHECK_RPP did, so that no RPP name is needed here
#define FKLB_RPP_CHECK(...)                                                                                                        \
    do {                                                                                                                           \
        const int fklbStatus_ = static_cast<int>(__VA_ARGS__);                                                                     \
        if (fklbStatus_ != 0) {                                                                                                    \
            throw std::runtime_error(fklb::statusMessage(#__VA_ARGS__, "RppStatus " + std::to_string(fklbStatus_), __FILE__, __LINE__)); \
        }                                                                                                                          \
    } while (0)
