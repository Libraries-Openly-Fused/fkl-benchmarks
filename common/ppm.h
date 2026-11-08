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

// fklb::readPpm(path, rgb, w, h): a binary PPM (P6, maximum value 255) into rgb, w * h * 3 bytes, row after row, R G B. Returns false when the file
// is not such a PPM or cannot be read. Input infrastructure: the programs that take a natural image use it to read the file, then give the same
// host bytes to every compared side.
#pragma once

#include <fstream>
#include <ios>
#include <string>
#include <vector>

namespace fklb {

inline bool readPpm(const char* path, std::vector<unsigned char>& rgb, int& w, int& h) {
    std::ifstream in(path, std::ios::binary);
    std::string magic;
    int maxval = 0;
    in >> magic >> w >> h >> maxval;
    in.get();
    if (!in || magic != "P6" || maxval != 255 || w <= 0 || h <= 0) {
        return false;
    }
    rgb.resize(static_cast<size_t>(w) * h * 3);
    in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return static_cast<bool>(in);
}

}  // namespace fklb
