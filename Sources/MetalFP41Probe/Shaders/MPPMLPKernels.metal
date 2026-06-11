#include <metal_stdlib>
#include <metal_packed_numeric>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

struct MLPConfig {
    uint tokens;
    uint hidden;
    uint intermediate;
};

template <typename AT, typename BT, typename CT, int NT>
inline void mpp_matmul_tile(typename tensor<device AT, dextents<int32_t, 2>, tensor_inline>::data_handle_type a,
                            typename tensor<device BT, dextents<int32_t, 2>, tensor_inline>::data_handle_type b,
                            typename tensor<device CT, dextents<int32_t, 2>, tensor_inline>::data_handle_type c,
                            constant uint &m,
                            constant uint &n,
                            constant uint &k,
                            uint2 tgid)
{
    constexpr auto desc = matmul2d_descriptor(64, NT, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<4>> op;

    using ATensor = tensor<device AT, dextents<int32_t, 2>, tensor_inline>;
    using BTensor = tensor<device BT, dextents<int32_t, 2>, tensor_inline>;
    using CTensor = tensor<device CT, dextents<int32_t, 2>, tensor_inline>;
    auto tA = ATensor(a, dextents<int32_t, 2>{int32_t(k), int32_t(m)}, array<int32_t, 2>{1, int32_t(k)});
    auto tB = BTensor(b, dextents<int32_t, 2>{int32_t(n), int32_t(k)}, array<int32_t, 2>{int32_t(k), 1});
    auto tC = CTensor(c, dextents<int32_t, 2>{int32_t(n), int32_t(m)}, array<int32_t, 2>{1, int32_t(n)});

    auto mA = tA.slice(0, tgid.y * 64);
    auto mB = tB.slice(tgid.x * NT, 0);
    auto mC = tC.slice(tgid.x * NT, tgid.y * 64);
    op.run(mA, mB, mC);
}

kernel void mpp_mlp_h_h_f_n128(device half *a [[buffer(0)]],
                               device half *b [[buffer(1)]],
                               device float *c [[buffer(2)]],
                               constant uint &m [[buffer(3)]],
                               constant uint &n [[buffer(4)]],
                               constant uint &k [[buffer(5)]],
                               uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp_matmul_tile<half, half, float, 128>(a, b, c, m, n, k, tgid);
}

kernel void mpp_mlp_h_i8_f_n128(device half *a [[buffer(0)]],
                                device int8_t *b [[buffer(1)]],
                                device float *c [[buffer(2)]],
                                constant uint &m [[buffer(3)]],
                                constant uint &n [[buffer(4)]],
                                constant uint &k [[buffer(5)]],
                                uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp_matmul_tile<half, int8_t, float, 128>(a, b, c, m, n, k, tgid);
}

kernel void mpp_mlp_i8_i8_i32_n128(device int8_t *a [[buffer(0)]],
                                   device int8_t *b [[buffer(1)]],
                                   device int32_t *c [[buffer(2)]],
                                   constant uint &m [[buffer(3)]],
                                   constant uint &n [[buffer(4)]],
                                   constant uint &k [[buffer(5)]],
                                   uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp_matmul_tile<int8_t, int8_t, int32_t, 128>(a, b, c, m, n, k, tgid);
}

kernel void mpp_mlp_h_i4_f_n128(device half *a [[buffer(0)]],
                                device uchar *b [[buffer(1)]],
                                device float *c [[buffer(2)]],
                                constant uint &m [[buffer(3)]],
                                constant uint &n [[buffer(4)]],
                                constant uint &k [[buffer(5)]],
                                uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp_matmul_tile<half, int4b_format, float, 128>(a, b, c, m, n, k, tgid);
}

kernel void mpp_mlp_h_fp4_f_n128(device half *a [[buffer(0)]],
                                 device uchar *b [[buffer(1)]],
                                 device float *c [[buffer(2)]],
                                 constant uint &m [[buffer(3)]],
                                 constant uint &n [[buffer(4)]],
                                 constant uint &k [[buffer(5)]],
                                 uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp_matmul_tile<half, metal_fp4_e2m1_format, float, 128>(a, b, c, m, n, k, tgid);
}

kernel void mpp_mlp_h_fp8e4m3_f_n128(device half *a [[buffer(0)]],
                                     device uchar *b [[buffer(1)]],
                                     device float *c [[buffer(2)]],
                                     constant uint &m [[buffer(3)]],
                                     constant uint &n [[buffer(4)]],
                                     constant uint &k [[buffer(5)]],
                                     uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp_matmul_tile<half, metal_fp8_e4m3_format, float, 128>(a, b, c, m, n, k, tgid);
}

kernel void mpp_mlp_h_fp8e4m3_h_n128(device half *a [[buffer(0)]],
                                     device uchar *b [[buffer(1)]],
                                     device half *c [[buffer(2)]],
                                     constant uint &m [[buffer(3)]],
                                     constant uint &n [[buffer(4)]],
                                     constant uint &k [[buffer(5)]],
                                     uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp_matmul_tile<half, metal_fp8_e4m3_format, half, 128>(a, b, c, m, n, k, tgid);
}

kernel void mpp_mlp_h_fp8e5m2_f_n128(device half *a [[buffer(0)]],
                                     device uchar *b [[buffer(1)]],
                                     device float *c [[buffer(2)]],
                                     constant uint &m [[buffer(3)]],
                                     constant uint &n [[buffer(4)]],
                                     constant uint &k [[buffer(5)]],
                                     uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp_matmul_tile<half, metal_fp8_e5m2_format, float, 128>(a, b, c, m, n, k, tgid);
}

#if defined(__HAVE_INT2B_FORMAT_TYPE__)
kernel void mpp_mlp_h_i2_f_n128(device half *a [[buffer(0)]],
                                device uchar *b [[buffer(1)]],
                                device float *c [[buffer(2)]],
                                constant uint &m [[buffer(3)]],
                                constant uint &n [[buffer(4)]],
                                constant uint &k [[buffer(5)]],
                                uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp_matmul_tile<half, int2b_format, float, 128>(a, b, c, m, n, k, tgid);
}
#endif

kernel void mpp_mlp_i8_i4_i32_n128(device int8_t *a [[buffer(0)]],
                                   device uchar *b [[buffer(1)]],
                                   device int32_t *c [[buffer(2)]],
                                   constant uint &m [[buffer(3)]],
                                   constant uint &n [[buffer(4)]],
                                   constant uint &k [[buffer(5)]],
                                   uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp_matmul_tile<int8_t, int4b_format, int32_t, 128>(a, b, c, m, n, k, tgid);
}

#if defined(__HAVE_INT2B_FORMAT_TYPE__)
kernel void mpp_mlp_i8_i2_i32_n128(device int8_t *a [[buffer(0)]],
                                   device uchar *b [[buffer(1)]],
                                   device int32_t *c [[buffer(2)]],
                                   constant uint &m [[buffer(3)]],
                                   constant uint &n [[buffer(4)]],
                                   constant uint &k [[buffer(5)]],
                                   uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp_matmul_tile<int8_t, int2b_format, int32_t, 128>(a, b, c, m, n, k, tgid);
}
#endif

kernel void mpp_mlp_f_h_f_n128(device float *a [[buffer(0)]],
                               device half *b [[buffer(1)]],
                               device float *c [[buffer(2)]],
                               constant uint &m [[buffer(3)]],
                               constant uint &n [[buffer(4)]],
                               constant uint &k [[buffer(5)]],
                               uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp_matmul_tile<float, half, float, 128>(a, b, c, m, n, k, tgid);
}

kernel void mpp_mlp_f_i8_f_n128(device float *a [[buffer(0)]],
                                device int8_t *b [[buffer(1)]],
                                device float *c [[buffer(2)]],
                                constant uint &m [[buffer(3)]],
                                constant uint &n [[buffer(4)]],
                                constant uint &k [[buffer(5)]],
                                uint2 tgid [[threadgroup_position_in_grid]])
{
    mpp_matmul_tile<float, int8_t, float, 128>(a, b, c, m, n, k, tgid);
}

kernel void mpp_mlp_swiglu_float(device const float *gate [[buffer(0)]],
                                 device const float *up [[buffer(1)]],
                                 device float *mid [[buffer(2)]],
                                 constant MLPConfig &config [[buffer(3)]],
                                 uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.intermediate) {
        return;
    }

    uint offset = token * config.intermediate + channel;
    float g = gate[offset];
    float u = up[offset];
    mid[offset] = (g / (1.0f + exp(-g))) * u;
}

kernel void mpp_mlp_swiglu_half(device const float *gate [[buffer(0)]],
                                device const float *up [[buffer(1)]],
                                device half *mid [[buffer(2)]],
                                constant MLPConfig &config [[buffer(3)]],
                                uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.intermediate) {
        return;
    }

    uint offset = token * config.intermediate + channel;
    float g = clamp(gate[offset] * 0.0625f, -20.0f, 20.0f);
    float u = up[offset] * 0.0625f;
    mid[offset] = half((g / (1.0f + exp(-g))) * u);
}

kernel void mpp_mlp_swiglu_half_from_half(device const half *gate [[buffer(0)]],
                                          device const half *up [[buffer(1)]],
                                          device half *mid [[buffer(2)]],
                                          constant MLPConfig &config [[buffer(3)]],
                                          uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.intermediate) {
        return;
    }

    uint offset = token * config.intermediate + channel;
    float g = clamp(float(gate[offset]) * 0.0625f, -20.0f, 20.0f);
    float u = float(up[offset]) * 0.0625f;
    mid[offset] = half((g / (1.0f + exp(-g))) * u);
}

kernel void mpp_mlp_swiglu_i8(device const int32_t *gate [[buffer(0)]],
                              device const int32_t *up [[buffer(1)]],
                              device int8_t *mid [[buffer(2)]],
                              constant MLPConfig &config [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.intermediate) {
        return;
    }

    uint offset = token * config.intermediate + channel;
    float g = clamp(float(gate[offset]) * 0.000244140625f, -20.0f, 20.0f);
    float u = float(up[offset]) * 0.000244140625f;
    float value = (g / (1.0f + exp(-g))) * u * 32.0f;
    mid[offset] = int8_t(int(rint(clamp(value, -128.0f, 127.0f))));
}
