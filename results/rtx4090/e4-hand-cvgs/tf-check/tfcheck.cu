// Thread-fusion check of cvgs_fused (opencv-cuda/batched-preprocessing/e4_preprocessing.cu), compiled only: the cvGPUSpeedup call of cvgs_fused, with the
// default cvGS::executeOperations (thread fusion enabled, as cvgs_fused calls it) or, with -DTF_CHECK_EXECUTE_FALSE=1, with cvGS::executeOperations<false>,
// at batch 10, 150 and 300. Built with the nvcc command line of e4_preprocessing_cu; the SASS of the kernels of the two builds is compared. It is not a program
// and is never run.
#include <cvGPUSpeedup.cuh>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <array>

#ifndef TF_CHECK_EXECUTE_FALSE
#define TF_CHECK_EXECUTE_FALSE 0
#endif

template <int BATCH>
void cvgsCall(const cv::cuda::GpuMat& src, const cv::cuda::GpuMat& out, const cv::cuda::Stream& cvs, const cv::Scalar& alphaS, const cv::Scalar& subS,
              const cv::Scalar& divS, const cv::Size& outSize) {
    std::array<cv::cuda::GpuMat, BATCH> crops;
    for (int i = 0; i < BATCH; ++i) crops[i] = src(cv::Rect(i, i, 60, 120));
#if TF_CHECK_EXECUTE_FALSE
    cvGS::executeOperations<false>(cvs, cvGS::resize<CV_8UC3, cv::INTER_LINEAR, BATCH>(crops, outSize, BATCH), cvGS::cvtColor<cv::COLOR_RGB2BGR, CV_32FC3>(),
                                   cvGS::multiply<CV_32FC3>(alphaS), cvGS::subtract<CV_32FC3>(subS), cvGS::divide<CV_32FC3>(divS),
                                   cvGS::split<CV_32FC3>(out, outSize));
#else
    cvGS::executeOperations(cvs, cvGS::resize<CV_8UC3, cv::INTER_LINEAR, BATCH>(crops, outSize, BATCH), cvGS::cvtColor<cv::COLOR_RGB2BGR, CV_32FC3>(),
                            cvGS::multiply<CV_32FC3>(alphaS), cvGS::subtract<CV_32FC3>(subS), cvGS::divide<CV_32FC3>(divS),
                            cvGS::split<CV_32FC3>(out, outSize));
#endif
}

template void cvgsCall<10>(const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, const cv::cuda::Stream&, const cv::Scalar&, const cv::Scalar&, const cv::Scalar&,
                           const cv::Size&);
template void cvgsCall<150>(const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, const cv::cuda::Stream&, const cv::Scalar&, const cv::Scalar&, const cv::Scalar&,
                            const cv::Size&);
template void cvgsCall<300>(const cv::cuda::GpuMat&, const cv::cuda::GpuMat&, const cv::cuda::Stream&, const cv::Scalar&, const cv::Scalar&, const cv::Scalar&,
                            const cv::Size&);
