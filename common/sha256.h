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

// fklb::sha256File(path): the SHA-256 (FIPS 180-4) of a file, as lowercase hex, the value sha256sum prints. For the provenance of input files that
// are deleted after a run: the CSV notes record the hash of every file a program read. A file that cannot be read throws.
#pragma once

#include <array>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fklb {

inline std::string sha256File(const std::string& path) {
    static constexpr uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
    std::array<uint32_t, 8> h = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    const auto rotr = [](const uint32_t x, const int n) { return (x >> n) | (x << (32 - n)); };
    const auto block = [&](const unsigned char* p) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) | (uint32_t(p[4 * i + 2]) << 8) | uint32_t(p[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
        for (int i = 0; i < 64; ++i) {
            const uint32_t t1 = hh + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K[i] + w[i];
            const uint32_t t2 = (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c));
            hh = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h[0] += a; h[1] += b; h[2] += c; h[3] += d; h[4] += e; h[5] += f; h[6] += g; h[7] += hh;
    };
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot read " + path + " for its SHA-256");
    std::vector<unsigned char> buf(1 << 20);
    uint64_t bytes = 0;
    size_t pending = 0;   // bytes of an incomplete block at the start of buf
    while (true) {
        f.read(reinterpret_cast<char*>(buf.data()) + pending, static_cast<std::streamsize>(buf.size() - pending));
        const size_t got = static_cast<size_t>(f.gcount());
        bytes += got;
        const size_t have = pending + got;
        size_t at = 0;
        for (; at + 64 <= have; at += 64) block(buf.data() + at);
        pending = have - at;
        for (size_t i = 0; i < pending; ++i) buf[i] = buf[at + i];
        if (!f) break;
    }
    if (f.bad()) throw std::runtime_error("cannot read " + path + " for its SHA-256");
    unsigned char tail[128] = {};
    for (size_t i = 0; i < pending; ++i) tail[i] = buf[i];
    tail[pending] = 0x80;
    const size_t tailBlocks = pending + 1 + 8 <= 64 ? 1 : 2;
    const uint64_t bits = bytes * 8;
    for (int i = 0; i < 8; ++i) tail[tailBlocks * 64 - 1 - i] = static_cast<unsigned char>(bits >> (8 * i));
    for (size_t i = 0; i < tailBlocks; ++i) block(tail + 64 * i);
    std::string hex;
    char two[3];
    for (const uint32_t word : h) {
        for (int i = 3; i >= 0; --i) {
            std::snprintf(two, sizeof(two), "%02x", static_cast<unsigned>((word >> (8 * i)) & 0xff));
            hex += two;
        }
    }
    return hex;
}

}  // namespace fklb
