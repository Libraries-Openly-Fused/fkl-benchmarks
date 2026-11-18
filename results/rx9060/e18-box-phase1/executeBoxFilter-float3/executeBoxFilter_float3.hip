// compile-only: fk::executeBoxFilter with LinearFilterDPPDetails<fk::float3> on GPU_AMD (FKL 24478f0)
#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/image_processing/linear_filter.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/core/data/ptr_utils.h>
void f(fk::Ptr2D<fk::uchar3>& in, fk::Ptr2D<fk::uchar3>& out, fk::Stream_<fk::ParArch::GPU_AMD>& s) {
    const auto image = fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(in).then(fk::Cast<fk::uchar3, fk::float3>::build());
    const auto write = fk::SaturateCast<fk::float3, fk::uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, fk::uchar3>::build(out));
    const fk::LinearFilterDPPDetails<fk::float3> d{64, 32, 3, 3, 1, 1};
    fk::executeBoxFilter(d, image, write, fk::float3{1.f, 1.f, 1.f}, s);
}
