// Accumulator precision / overflow probe for TensorOps matmul2d.
//
// Question under test: when the destination is half, does the hardware
// accumulate in half (so long dot products lose precision or overflow at
// 65504), or in something wider that is only rounded to half at the end?
// Each kernel computes one 64x128 tile; the host reads C[0] of a dot product
// with a known exact integer value.

#include <metal_stdlib>
#include <metal_packed_numeric>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

template <typename AT, typename BT, typename CT>
inline void accum_probe_tile(typename tensor<device AT, dextents<int32_t, 2>, tensor_inline>::data_handle_type a,
                             typename tensor<device BT, dextents<int32_t, 2>, tensor_inline>::data_handle_type b,
                             typename tensor<device CT, dextents<int32_t, 2>, tensor_inline>::data_handle_type c,
                             constant uint &m,
                             constant uint &n,
                             constant uint &k,
                             uint2 tgid)
{
    constexpr auto desc = matmul2d_descriptor(64, 128, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<4>> op;

    using ATensor = tensor<device AT, dextents<int32_t, 2>, tensor_inline>;
    using BTensor = tensor<device BT, dextents<int32_t, 2>, tensor_inline>;
    using CTensor = tensor<device CT, dextents<int32_t, 2>, tensor_inline>;
    auto tA = ATensor(a, dextents<int32_t, 2>{int32_t(k), int32_t(m)}, array<int32_t, 2>{1, int32_t(k)});
    auto tB = BTensor(b, dextents<int32_t, 2>{int32_t(n), int32_t(k)}, array<int32_t, 2>{int32_t(k), 1});
    auto tC = CTensor(c, dextents<int32_t, 2>{int32_t(n), int32_t(m)}, array<int32_t, 2>{1, int32_t(n)});

    auto mA = tA.slice(0, tgid.y * 64);
    auto mB = tB.slice(tgid.x * 128, 0);
    auto mC = tC.slice(tgid.x * 128, tgid.y * 64);
    op.run(mA, mB, mC);
}

kernel void accum_h_h_h(device half *a [[buffer(0)]],
                        device half *b [[buffer(1)]],
                        device half *c [[buffer(2)]],
                        constant uint &m [[buffer(3)]],
                        constant uint &n [[buffer(4)]],
                        constant uint &k [[buffer(5)]],
                        uint2 tgid [[threadgroup_position_in_grid]])
{
    accum_probe_tile<half, half, half>(a, b, c, m, n, k, tgid);
}

kernel void accum_h_h_f(device half *a [[buffer(0)]],
                        device half *b [[buffer(1)]],
                        device float *c [[buffer(2)]],
                        constant uint &m [[buffer(3)]],
                        constant uint &n [[buffer(4)]],
                        constant uint &k [[buffer(5)]],
                        uint2 tgid [[threadgroup_position_in_grid]])
{
    accum_probe_tile<half, half, float>(a, b, c, m, n, k, tgid);
}

kernel void accum_h_fp4_h(device half *a [[buffer(0)]],
                          device uchar *b [[buffer(1)]],
                          device half *c [[buffer(2)]],
                          constant uint &m [[buffer(3)]],
                          constant uint &n [[buffer(4)]],
                          constant uint &k [[buffer(5)]],
                          uint2 tgid [[threadgroup_position_in_grid]])
{
    accum_probe_tile<half, metal_fp4_e2m1_format, half>(a, b, c, m, n, k, tgid);
}

kernel void accum_h_fp4_f(device half *a [[buffer(0)]],
                          device uchar *b [[buffer(1)]],
                          device float *c [[buffer(2)]],
                          constant uint &m [[buffer(3)]],
                          constant uint &n [[buffer(4)]],
                          constant uint &k [[buffer(5)]],
                          uint2 tgid [[threadgroup_position_in_grid]])
{
    accum_probe_tile<half, metal_fp4_e2m1_format, float>(a, b, c, m, n, k, tgid);
}

kernel void accum_fp4_fp4_h(device uchar *a [[buffer(0)]],
                            device uchar *b [[buffer(1)]],
                            device half *c [[buffer(2)]],
                            constant uint &m [[buffer(3)]],
                            constant uint &n [[buffer(4)]],
                            constant uint &k [[buffer(5)]],
                            uint2 tgid [[threadgroup_position_in_grid]])
{
    accum_probe_tile<metal_fp4_e2m1_format, metal_fp4_e2m1_format, half>(a, b, c, m, n, k, tgid);
}

kernel void accum_fp4_fp4_f(device uchar *a [[buffer(0)]],
                            device uchar *b [[buffer(1)]],
                            device float *c [[buffer(2)]],
                            constant uint &m [[buffer(3)]],
                            constant uint &n [[buffer(4)]],
                            constant uint &k [[buffer(5)]],
                            uint2 tgid [[threadgroup_position_in_grid]])
{
    accum_probe_tile<metal_fp4_e2m1_format, metal_fp4_e2m1_format, float>(a, b, c, m, n, k, tgid);
}

kernel void accum_fp8_fp8_h(device uchar *a [[buffer(0)]],
                            device uchar *b [[buffer(1)]],
                            device half *c [[buffer(2)]],
                            constant uint &m [[buffer(3)]],
                            constant uint &n [[buffer(4)]],
                            constant uint &k [[buffer(5)]],
                            uint2 tgid [[threadgroup_position_in_grid]])
{
    accum_probe_tile<metal_fp8_e4m3_format, metal_fp8_e4m3_format, half>(a, b, c, m, n, k, tgid);
}

kernel void accum_fp8_fp8_f(device uchar *a [[buffer(0)]],
                            device uchar *b [[buffer(1)]],
                            device float *c [[buffer(2)]],
                            constant uint &m [[buffer(3)]],
                            constant uint &n [[buffer(4)]],
                            constant uint &k [[buffer(5)]],
                            uint2 tgid [[threadgroup_position_in_grid]])
{
    accum_probe_tile<metal_fp8_e4m3_format, metal_fp8_e4m3_format, float>(a, b, c, m, n, k, tgid);
}
