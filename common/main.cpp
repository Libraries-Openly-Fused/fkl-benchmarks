/* Copyright 2023 Oscar Amoros Huguet
   Copyright 2023 Albert Andaluz Gonzalez

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

#include <exception>
#include <iostream>

// A failed status check (common/status_check.h) throws, and the program stops there. Catching the exception here unwinds the stack first, so
// every fklb::CsvWriter that is still open marks its file INCOMPLETE (common/csv_writer.h); then the message is printed and the program
// returns 1. Without this catch an exception that leaves launch() calls std::terminate, which need not unwind.
int main(int argc, char **argv) {
    try {
        return launch();
    } catch (const std::exception& e) {
        std::cerr << "stopped: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "stopped: an exception that is not a std::exception" << std::endl;
        return 1;
    }
}

