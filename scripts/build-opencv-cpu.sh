#!/bin/bash
# Builds the OpenCV of the CPU benchmarks (opencv-cpu/): OpenCV 4.14.0, CPU only, modules core and imgproc, shared
# libraries, Release, the default CPU dispatch (SSE4.1 to AVX512_SKX, picked at run time), pthreads as the parallel
# framework, and Intel IPP (ICV, downloaded by OpenCV's own build). This is the configuration of the paper's results; the
# CSV header records cv::getBuildInformation().
#
#   scripts/build-opencv-cpu.sh [install prefix]        (default: $HOME/opencv-4.14.0-cpu-g++-13)
#   CXX=g++-13 CC=gcc-13 JOBS=16 WORK_DIR=/tmp/opencv-build scripts/build-opencv-cpu.sh
#
# Then configure fkl-benchmarks with -DFKLB_BUILD_OPENCV_CPU=ON -DOPENCV_DIR=<install prefix>.
set -euo pipefail

VERSION=${OPENCV_VERSION:-4.14.0}
SHA256=${OPENCV_SHA256:-ee8fb9b30eb60850431b4656447080e3737b56e45719c92b67f245950609f86e}  # of the 4.14.0 source archive
CXX=${CXX:-g++-13}
CC=${CC:-${CXX/g++/gcc}}
JOBS=${JOBS:-$(nproc)}
PREFIX=${1:-$HOME/opencv-${VERSION}-cpu-${CXX}}
WORK_DIR=${WORK_DIR:-$(mktemp -d)}

if [ "$VERSION" != "4.14.0" ] && [ -z "${OPENCV_SHA256:-}" ]; then
    echo "Set OPENCV_SHA256 to the SHA-256 of the source archive of OpenCV $VERSION" >&2
    exit 1
fi

mkdir -p "$WORK_DIR"
curl -fL "https://github.com/opencv/opencv/archive/refs/tags/${VERSION}.tar.gz" -o "$WORK_DIR/opencv.tar.gz"
echo "${SHA256}  $WORK_DIR/opencv.tar.gz" | sha256sum -c -
tar -xzf "$WORK_DIR/opencv.tar.gz" -C "$WORK_DIR"

cmake -G Ninja -S "$WORK_DIR/opencv-${VERSION}" -B "$WORK_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release "-DCMAKE_INSTALL_PREFIX=$PREFIX" \
    "-DCMAKE_C_COMPILER=$CC" "-DCMAKE_CXX_COMPILER=$CXX" \
    -DBUILD_LIST=core,imgproc -DBUILD_SHARED_LIBS=ON -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
    -DBUILD_DOCS=OFF -DBUILD_opencv_apps=OFF -DBUILD_JAVA=OFF -DBUILD_opencv_python3=OFF \
    -DOPENCV_GENERATE_SETUPVARS=OFF -DWITH_FFMPEG=OFF -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_QT=OFF \
    -DWITH_V4L=OFF -DWITH_MSMF=OFF -DWITH_DSHOW=OFF -DWITH_CUDA=OFF
cmake --build "$WORK_DIR/build" --target install --parallel "$JOBS"
echo "OpenCV ${VERSION} installed in $PREFIX"
