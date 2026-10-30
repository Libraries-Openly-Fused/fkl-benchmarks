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

// One CSV file, one block: the system info as comment lines ("# key: value", common/system_info.h), one header row, then
// one row per measurement. Used by the benchmarks written in this repository (the moved ones keep the harness they came
// with, fixed to write a single block). Read with pandas.read_csv(path, comment='#').
//
//   fklb::CsvWriter csv("handwritten_madchain.csv", "E1 hand-written MAD chain", {"ops", "mean_ms", "min_ms", "max_ms"});
//   csv.row(ops, mean, min, max);

#pragma once

#include <common/system_info.h>

#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <stdexcept>
#include <utility>
#include <string>

namespace fklb {

class CsvWriter {
    std::ofstream m_file;
    size_t m_columns;

    template <typename T>
    void put(const T& value, bool first) {
        if (!first) {
            m_file << ',';
        }
        m_file << value;
    }

public:
    CsvWriter(const std::string& path, const std::string& experiment, std::initializer_list<const char*> columns)
        : m_file(path), m_columns(columns.size()) {
        if (!m_file) {
            throw std::runtime_error("cannot open " + path);
        }
        writeSystemInfoHeader(m_file, experiment);
        bool first = true;
        for (const char* column : columns) {
            put(column, first);
            first = false;
        }
        m_file << '\n';
        m_file << std::setprecision(9);
    }

    template <typename... Values>
    void row(const Values&... values) {
        if (sizeof...(Values) != m_columns) {
            throw std::runtime_error("CsvWriter::row: the number of values is not the number of columns");
        }
        bool first = true;
        (put(values, std::exchange(first, false)), ...);
        m_file << '\n';
    }

    void flush() { m_file.flush(); }
};

}  // namespace fklb
