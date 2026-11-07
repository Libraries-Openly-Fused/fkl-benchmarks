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
//
// A run that stops with an error must not leave a file that looks complete (the coordinator, 2026-11-07): when the writer is destroyed while
// an exception propagates (a failed FKLB_*_CHECK of common/status_check.h, caught in main), its last line is
//   # INCOMPLETE: the program stopped with an error after <n> rows; do not use this file
// (a comment line, which pandas.read_csv(path, comment='#') skips). A normal run's file is unchanged.

#pragma once

#include <common/system_info.h>

#include <exception>
#include <fstream>
#include <initializer_list>
#include <iomanip>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace fklb {

class CsvWriter {
    std::ofstream m_file;
    size_t m_columns;
    int m_uncaughtAtStart;   // std::uncaught_exceptions() when the writer was made: more of them at its destruction = the run failed
    size_t m_rows = 0;

    // A text value with a comma, a quote or a line break is written quoted (RFC 4180), so that every row has as many fields as the header
    void putText(const std::string_view text) {
        if (text.find_first_of(",\"\r\n") == std::string_view::npos) {
            m_file << text;
            return;
        }
        m_file << '"';
        for (const char c : text) {
            if (c == '"') {
                m_file << '"';
            }
            m_file << c;
        }
        m_file << '"';
    }

    template <typename T>
    void put(const T& value, bool first) {
        if (!first) {
            m_file << ',';
        }
        if constexpr (std::is_convertible_v<const T&, std::string_view>) {
            putText(std::string_view(value));
        } else {
            m_file << value;
        }
    }

public:
    CsvWriter(const std::string& path, const std::string& experiment, std::initializer_list<const char*> columns,
              const std::string& notes = "")
        : m_file(path), m_columns(columns.size()), m_uncaughtAtStart(std::uncaught_exceptions()) {
        if (!m_file) {
            throw std::runtime_error("cannot open " + path);
        }
        writeSystemInfoHeader(m_file, experiment, notes);
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
        ++m_rows;
    }

    void flush() { m_file.flush(); }

    ~CsvWriter() {
        if (std::uncaught_exceptions() > m_uncaughtAtStart) {
            m_file << "# INCOMPLETE: the program stopped with an error after " << m_rows << " rows; do not use this file\n";
        }
    }
};

}  // namespace fklb
