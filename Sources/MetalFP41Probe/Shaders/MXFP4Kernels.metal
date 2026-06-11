// DeepSeek-style MXFP4 expert (MLP) kernels.
//
// MXFP4 layout: FP4 E2M1 weight data + one E8M0 scale byte per 32 weights
// along the reduction (K) dimension. scales[outputRow * (k / 32) + kBlock].
//
// Strategies compared:
//  - MPP 4.1 native scale-plane: matmul2d consumes FP4 data + E8M0 scale
//    plane together via the tensor_blockwise<tensor_plane_scales, ...> tag
//    (requires transpose_right for a scaled right operand). Fastest.
//  - MPP 4.1 manual scaling: native FP4 matmul per 32-wide k-block, E8M0
//    scales applied to partial products in cooperative destination tensors.
//  - Coop-input fused (MPP 4.0-style): LUT decode + scale while filling a
//    right-input cooperative tensor; no FP4 TensorOps type involved.
//  - Materialized dequant (4.1 native unpack or 4.0 LUT decode) to half
//    weights in device memory, then a plain half x half matmul2d.

#include <metal_stdlib>
#include <metal_packed_numeric>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

constant constexpr int kMXFP4BlockSize = 32;

static inline float e8m0_to_float(uchar bits)
{
    return as_type<float>(uint(bits) << 23);
}

// FP4 E2M1 nibble values; bit-identical to the hardware decode (validated by
// matching checksums between the LUT and native-unpack dequant rows).
constant half kFP4E2M1Table[16] = {
    0.0h, 0.5h, 1.0h, 1.5h, 2.0h, 3.0h, 4.0h, 6.0h,
    -0.0h, -0.5h, -1.0h, -1.5h, -2.0h, -3.0h, -4.0h, -6.0h
};

// MPP 4.1 fused path: weights stay packed FP4 in device memory, matmul2d
// consumes them natively per 32-wide k-block, and each partial product tile
// is scaled by its E8M0 block scale in registers before accumulation. No half
// weights are ever materialized.
template <typename AT, int NT>
inline void mxfp4_matmul_fused_tile(typename tensor<device AT, dextents<int32_t, 2>, tensor_inline>::data_handle_type a,
                                    typename tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline>::data_handle_type b,
                                    typename tensor<device float, dextents<int32_t, 2>, tensor_inline>::data_handle_type c,
                                    device const uchar *scales,
                                    constant uint &m,
                                    constant uint &n,
                                    constant uint &k,
                                    uint2 tgid)
{
    constexpr auto desc = matmul2d_descriptor(64, NT, kMXFP4BlockSize);
    matmul2d<desc, execution_simdgroups<4>> op;

    using ATensor = tensor<device AT, dextents<int32_t, 2>, tensor_inline>;
    using BTensor = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline>;
    using CTensor = tensor<device float, dextents<int32_t, 2>, tensor_inline>;
    auto tA = ATensor(a, dextents<int32_t, 2>{int32_t(k), int32_t(m)}, array<int32_t, 2>{1, int32_t(k)});
    auto tB = BTensor(b, dextents<int32_t, 2>{int32_t(n), int32_t(k)}, array<int32_t, 2>{int32_t(k), 1});
    auto tC = CTensor(c, dextents<int32_t, 2>{int32_t(n), int32_t(m)}, array<int32_t, 2>{1, int32_t(n)});

    auto mA0 = tA.slice(0, tgid.y * 64);
    auto mB0 = tB.slice(tgid.x * NT, 0);
    auto mC = tC.slice(tgid.x * NT, tgid.y * 64);

    using LeftT = remove_addrspace_t<decltype(mA0)>;
    using RightT = remove_addrspace_t<decltype(mB0)>;
    auto accT = op.template get_destination_cooperative_tensor<LeftT, RightT, float>();
    auto partT = op.template get_destination_cooperative_tensor<LeftT, RightT, float>();

#pragma unroll
    for (uint16_t i = 0; i < accT.get_capacity(); ++i) {
        if (accT.is_valid_element(i)) {
            accT[i] = 0.0f;
        }
    }

    const uint kBlocks = k / kMXFP4BlockSize;
    for (uint kb = 0; kb < kBlocks; ++kb) {
        auto mA = tA.slice(kb * kMXFP4BlockSize, tgid.y * 64);
        auto mB = tB.slice(tgid.x * NT, kb * kMXFP4BlockSize);

#pragma unroll
        for (uint16_t i = 0; i < partT.get_capacity(); ++i) {
            if (partT.is_valid_element(i)) {
                partT[i] = 0.0f;
            }
        }

        op.run(mA, mB, partT);

#pragma unroll
        for (uint16_t i = 0; i < partT.get_capacity(); ++i) {
            if (partT.is_valid_element(i)) {
                auto idx = partT.get_multidimensional_index(i);
                uint column = tgid.x * NT + uint(idx[0]);
                if (column < n) {
                    float scale = e8m0_to_float(scales[column * kBlocks + kb]);
                    accT[i] = fma(partT[i], scale, accT[i]);
                }
            }
        }
    }

    accT.store(mC);
}

kernel void mxfp4_fused_h_f_n128(device half *a [[buffer(0)]],
                                 device uchar *b [[buffer(1)]],
                                 device float *c [[buffer(2)]],
                                 constant uint &m [[buffer(3)]],
                                 constant uint &n [[buffer(4)]],
                                 constant uint &k [[buffer(5)]],
                                 device const uchar *scales [[buffer(6)]],
                                 uint2 tgid [[threadgroup_position_in_grid]])
{
    mxfp4_matmul_fused_tile<half, 128>(a, b, c, scales, m, n, k, tgid);
}

kernel void mxfp4_fused_h_f_n64(device half *a [[buffer(0)]],
                                device uchar *b [[buffer(1)]],
                                device float *c [[buffer(2)]],
                                constant uint &m [[buffer(3)]],
                                constant uint &n [[buffer(4)]],
                                constant uint &k [[buffer(5)]],
                                device const uchar *scales [[buffer(6)]],
                                uint2 tgid [[threadgroup_position_in_grid]])
{
    mxfp4_matmul_fused_tile<half, 64>(a, b, c, scales, m, n, k, tgid);
}

// Native scale-plane path (MPP 4.1): the FP4 weight tensor carries its E8M0
// scale plane via the tensor_blockwise<tensor_plane_scales, ...> tag and
// matmul2d consumes both directly. Scaled operands require the right tensor
// in transposed form (extents (k, n), contraction along dim 0), which matches
// the row-major [n][k] packed weight layout. The scale plane uses the same
// scales[j * (k/32) + kb] layout as every other kernel in this file.
template <int NT>
inline void mxfp4_matmul_native_scaleplane_tile(device half *a,
                                                device uchar *b,
                                                device float *c,
                                                device const uchar *scales,
                                                uint m,
                                                uint n,
                                                uint k,
                                                uint2 tgid)
{
    constexpr auto desc = matmul2d_descriptor(64, NT, static_cast<int>(dynamic_extent),
                                              false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;

    using ATensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using ScalesTag = tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;
    using BTensor = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline, ScalesTag>;
    using CTensor = tensor<device float, dextents<int32_t, 2>, tensor_inline>;

    auto tA = ATensor((ATensor::data_handle_type)a,
                      dextents<int32_t, 2>{int32_t(k), int32_t(m)},
                      array<int32_t, 2>{1, int32_t(k)});
    ScalesTag plane(reinterpret_cast<ScalesTag::data_handle_type>(const_cast<device uchar *>(scales)));
    auto tB = BTensor((BTensor::data_handle_type)b,
                      dextents<int32_t, 2>{int32_t(k), int32_t(n)},
                      array<int32_t, 2>{1, int32_t(k)},
                      plane);
    auto tC = CTensor((CTensor::data_handle_type)c,
                      dextents<int32_t, 2>{int32_t(n), int32_t(m)},
                      array<int32_t, 2>{1, int32_t(n)});

    auto mA = tA.slice(0, tgid.y * 64);
    auto mB = tB.slice(0, tgid.x * NT);
    auto mC = tC.slice(tgid.x * NT, tgid.y * 64);
    op.run(mA, mB, mC);
}

// Host-created multi-plane tensor variant: the weight tensor (FP4 data plane
// + E8M0 scale plane) is built on the host with MTLTensorDescriptor +
// MTLTensorBufferAttachments and bound bindlessly (gpuResourceID in an
// argument struct + useResource). The kernel-side type carries the same
// tensor_blockwise tag; extents and the scale plane travel with the handle.
struct MXFP4ScalePlaneArgs {
    tensor<device metal_fp4_e2m1_format,
           dextents<int32_t, 2>,
           tensor_handle,
           tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>> weights;
};

template <int NT>
inline void mxfp4_matmul_native_sp_handle_tile(device half *a,
                                               constant MXFP4ScalePlaneArgs &args,
                                               device float *c,
                                               uint m,
                                               uint n,
                                               uint k,
                                               uint2 tgid)
{
    constexpr auto desc = matmul2d_descriptor(64, NT, static_cast<int>(dynamic_extent),
                                              false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;

    using ATensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using CTensor = tensor<device float, dextents<int32_t, 2>, tensor_inline>;

    auto tA = ATensor((ATensor::data_handle_type)a,
                      dextents<int32_t, 2>{int32_t(k), int32_t(m)},
                      array<int32_t, 2>{1, int32_t(k)});
    auto tC = CTensor((CTensor::data_handle_type)c,
                      dextents<int32_t, 2>{int32_t(n), int32_t(m)},
                      array<int32_t, 2>{1, int32_t(n)});

    auto tB = args.weights;
    auto mA = tA.slice(0, tgid.y * 64);
    auto mB = tB.slice(0, tgid.x * NT);
    auto mC = tC.slice(tgid.x * NT, tgid.y * 64);
    op.run(mA, mB, mC);
}

kernel void mxfp4_native_sph_h_f_n64(device half *a [[buffer(0)]],
                                     constant MXFP4ScalePlaneArgs &args [[buffer(1)]],
                                     device float *c [[buffer(2)]],
                                     constant uint &m [[buffer(3)]],
                                     constant uint &n [[buffer(4)]],
                                     constant uint &k [[buffer(5)]],
                                     uint2 tgid [[threadgroup_position_in_grid]])
{
    mxfp4_matmul_native_sp_handle_tile<64>(a, args, c, m, n, k, tgid);
}

// Control for the scale-plane row: identical transposed-B FP4 matmul with no
// scale plane (numerically unscaled), isolating the cost of the scale plane
// itself from the cost of the transpose_right operand form.
kernel void mxfp4_native_nosp_h_f_n128(device half *a [[buffer(0)]],
                                       device uchar *b [[buffer(1)]],
                                       device float *c [[buffer(2)]],
                                       constant uint &m [[buffer(3)]],
                                       constant uint &n [[buffer(4)]],
                                       constant uint &k [[buffer(5)]],
                                       device const uchar *scales [[buffer(6)]],
                                       uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(64, 128, static_cast<int>(dynamic_extent),
                                              false, true, false);
    matmul2d<desc, execution_simdgroups<4>> op;

    using ATensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using BTensor = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline>;
    using CTensor = tensor<device float, dextents<int32_t, 2>, tensor_inline>;

    auto tA = ATensor((ATensor::data_handle_type)a,
                      dextents<int32_t, 2>{int32_t(k), int32_t(m)},
                      array<int32_t, 2>{1, int32_t(k)});
    auto tB = BTensor((BTensor::data_handle_type)b,
                      dextents<int32_t, 2>{int32_t(k), int32_t(n)},
                      array<int32_t, 2>{1, int32_t(k)});
    auto tC = CTensor((CTensor::data_handle_type)c,
                      dextents<int32_t, 2>{int32_t(n), int32_t(m)},
                      array<int32_t, 2>{1, int32_t(n)});

    auto mA = tA.slice(0, tgid.y * 64);
    auto mB = tB.slice(0, tgid.x * 128);
    auto mC = tC.slice(tgid.x * 128, tgid.y * 64);
    op.run(mA, mB, mC);
}

kernel void mxfp4_native_sp_h_f_n128(device half *a [[buffer(0)]],
                                     device uchar *b [[buffer(1)]],
                                     device float *c [[buffer(2)]],
                                     constant uint &m [[buffer(3)]],
                                     constant uint &n [[buffer(4)]],
                                     constant uint &k [[buffer(5)]],
                                     device const uchar *scales [[buffer(6)]],
                                     uint2 tgid [[threadgroup_position_in_grid]])
{
    mxfp4_matmul_native_scaleplane_tile<128>(a, b, c, scales, m, n, k, tgid);
}

kernel void mxfp4_native_sp_h_f_n64(device half *a [[buffer(0)]],
                                    device uchar *b [[buffer(1)]],
                                    device float *c [[buffer(2)]],
                                    constant uint &m [[buffer(3)]],
                                    constant uint &n [[buffer(4)]],
                                    constant uint &k [[buffer(5)]],
                                    device const uchar *scales [[buffer(6)]],
                                    uint2 tgid [[threadgroup_position_in_grid]])
{
    mxfp4_matmul_native_scaleplane_tile<64>(a, b, c, scales, m, n, k, tgid);
}

// Cooperative-input fused path: FP4 weights are decoded (LUT) and E8M0 block
// scales applied while filling a right-input cooperative tensor, which feeds
// matmul2d directly from registers. The k-chunk is decoupled from the 32-wide
// scale block, and no half weights ever touch device or threadgroup memory.
// Note: this kernel never uses the metal_fp4_e2m1_format TensorOps type, so
// it is MPP 4.0-style and does not need M5 FP4 hardware (it does require the
// cooperative input tensor API). Input cooperative tensors require
// execution_simdgroup scope, so each SIMD group owns one MT x NT output tile.
// run() with a cooperative destination overwrites in mode::multiply; chunked
// accumulation requires mode::multiply_accumulate.
template <int MT, int NT, int KCHUNK>
inline void mxfp4_matmul_coopinput_tile(typename tensor<device half, dextents<int32_t, 2>, tensor_inline>::data_handle_type a,
                                        device const uchar *packed,
                                        typename tensor<device float, dextents<int32_t, 2>, tensor_inline>::data_handle_type c,
                                        device const uchar *scales,
                                        constant uint &m,
                                        constant uint &n,
                                        constant uint &k,
                                        uint2 tgid)
{
    constexpr auto desc = matmul2d_descriptor(MT, NT, KCHUNK, false, false, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;

    using ATensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using CTensor = tensor<device float, dextents<int32_t, 2>, tensor_inline>;
    auto tA = ATensor(a, dextents<int32_t, 2>{int32_t(k), int32_t(m)}, array<int32_t, 2>{1, int32_t(k)});
    auto tC = CTensor(c, dextents<int32_t, 2>{int32_t(n), int32_t(m)}, array<int32_t, 2>{1, int32_t(n)});

    auto mA0 = tA.slice(0, tgid.y * MT);
    auto mC = tC.slice(tgid.x * NT, tgid.y * MT);

    auto ctB = op.template get_right_input_cooperative_tensor<half, half, float>();
    using LeftT = remove_addrspace_t<decltype(mA0)>;
    using RightT = remove_addrspace_t<decltype(ctB)>;
    auto ctO = op.template get_destination_cooperative_tensor<LeftT, RightT, float>();

#pragma unroll
    for (uint16_t i = 0; i < ctO.get_capacity(); ++i) {
        if (ctO.is_valid_element(i)) {
            ctO[i] = 0.0f;
        }
    }

    const uint kBlocks = k / kMXFP4BlockSize;
    const uint bytesPerRow = k / 2;
    for (uint kc = 0; kc * KCHUNK < k; ++kc) {
        const uint kBase = kc * KCHUNK;

#pragma unroll
        for (uint16_t i = 0; i < ctB.get_capacity(); ++i) {
            if (ctB.is_valid_element(i)) {
                auto idx = ctB.get_multidimensional_index(i);
                uint column = tgid.x * NT + uint(idx[0]);
                uint kg = kBase + uint(idx[1]);
                if (column < n && kg < k) {
                    uchar bits = packed[column * bytesPerRow + kg / 2];
                    uchar nibble = (kg & 1) ? (bits >> 4) : (bits & 0x0f);
                    float scale = e8m0_to_float(scales[column * kBlocks + kg / kMXFP4BlockSize]);
                    ctB[i] = half(float(kFP4E2M1Table[nibble]) * scale);
                } else {
                    ctB[i] = 0.0h;
                }
            }
        }

        auto mA = tA.slice(kBase, tgid.y * MT);
        op.run(mA, ctB, ctO);
    }

    ctO.store(mC);
}

kernel void mxfp4_fused_ci_h_f_sg_k64(device half *a [[buffer(0)]],
                                      device uchar *b [[buffer(1)]],
                                      device float *c [[buffer(2)]],
                                      constant uint &m [[buffer(3)]],
                                      constant uint &n [[buffer(4)]],
                                      constant uint &k [[buffer(5)]],
                                      device const uchar *scales [[buffer(6)]],
                                      uint2 tgid [[threadgroup_position_in_grid]])
{
    mxfp4_matmul_coopinput_tile<32, 32, 64>(a, b, c, scales, m, n, k, tgid);
}

kernel void mxfp4_fused_ci_h_f_sg_k32(device half *a [[buffer(0)]],
                                      device uchar *b [[buffer(1)]],
                                      device float *c [[buffer(2)]],
                                      constant uint &m [[buffer(3)]],
                                      constant uint &n [[buffer(4)]],
                                      constant uint &k [[buffer(5)]],
                                      device const uchar *scales [[buffer(6)]],
                                      uint2 tgid [[threadgroup_position_in_grid]])
{
    mxfp4_matmul_coopinput_tile<32, 32, 32>(a, b, c, scales, m, n, k, tgid);
}

// Decomposition probes for the fused path (numerically WRONG on purpose,
// perf-only): they peel the fused kernel apart to attribute its cost.
//
// coopstore: one full-K matmul into a cooperative tensor, one elementwise
// scale pass (block-0 scale only), store. Measures cooperative-tensor
// materialization + store overhead on top of the plain h_fp4_f row, without
// the k-block split.
kernel void mxfp4_probe_coopstore_h_f_n64(device half *a [[buffer(0)]],
                                          device uchar *b [[buffer(1)]],
                                          device float *c [[buffer(2)]],
                                          constant uint &m [[buffer(3)]],
                                          constant uint &n [[buffer(4)]],
                                          constant uint &k [[buffer(5)]],
                                          device const uchar *scales [[buffer(6)]],
                                          uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(64, 64, static_cast<int>(dynamic_extent));
    matmul2d<desc, execution_simdgroups<4>> op;

    using ATensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using BTensor = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline>;
    using CTensor = tensor<device float, dextents<int32_t, 2>, tensor_inline>;
    auto tA = ATensor(a, dextents<int32_t, 2>{int32_t(k), int32_t(m)}, array<int32_t, 2>{1, int32_t(k)});
    auto tB = BTensor(b, dextents<int32_t, 2>{int32_t(n), int32_t(k)}, array<int32_t, 2>{int32_t(k), 1});
    auto tC = CTensor(c, dextents<int32_t, 2>{int32_t(n), int32_t(m)}, array<int32_t, 2>{1, int32_t(n)});

    auto mA = tA.slice(0, tgid.y * 64);
    auto mB = tB.slice(tgid.x * 64, 0);
    auto mC = tC.slice(tgid.x * 64, tgid.y * 64);

    using LeftT = remove_addrspace_t<decltype(mA)>;
    using RightT = remove_addrspace_t<decltype(mB)>;
    auto cT = op.template get_destination_cooperative_tensor<LeftT, RightT, float>();

#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            cT[i] = 0.0f;
        }
    }

    op.run(mA, mB, cT);

    const uint kBlocks = k / kMXFP4BlockSize;
#pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (cT.is_valid_element(i)) {
            auto idx = cT.get_multidimensional_index(i);
            uint column = tgid.x * 64 + uint(idx[0]);
            if (column < n) {
                cT[i] *= e8m0_to_float(scales[column * kBlocks]);
            }
        }
    }

    cT.store(mC);
}

// noscale: identical k-block split and cooperative accumulate as the fused
// kernel, but no scale load/multiply. Isolates the cost of the 32-wide
// accumulation granularity itself.
kernel void mxfp4_probe_noscale_h_f_n64(device half *a [[buffer(0)]],
                                        device uchar *b [[buffer(1)]],
                                        device float *c [[buffer(2)]],
                                        constant uint &m [[buffer(3)]],
                                        constant uint &n [[buffer(4)]],
                                        constant uint &k [[buffer(5)]],
                                        device const uchar *scales [[buffer(6)]],
                                        uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = matmul2d_descriptor(64, 64, kMXFP4BlockSize);
    matmul2d<desc, execution_simdgroups<4>> op;

    using ATensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using BTensor = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline>;
    using CTensor = tensor<device float, dextents<int32_t, 2>, tensor_inline>;
    auto tA = ATensor(a, dextents<int32_t, 2>{int32_t(k), int32_t(m)}, array<int32_t, 2>{1, int32_t(k)});
    auto tB = BTensor(b, dextents<int32_t, 2>{int32_t(n), int32_t(k)}, array<int32_t, 2>{int32_t(k), 1});
    auto tC = CTensor(c, dextents<int32_t, 2>{int32_t(n), int32_t(m)}, array<int32_t, 2>{1, int32_t(n)});

    auto mA0 = tA.slice(0, tgid.y * 64);
    auto mB0 = tB.slice(tgid.x * 64, 0);
    auto mC = tC.slice(tgid.x * 64, tgid.y * 64);

    using LeftT = remove_addrspace_t<decltype(mA0)>;
    using RightT = remove_addrspace_t<decltype(mB0)>;
    auto accT = op.template get_destination_cooperative_tensor<LeftT, RightT, float>();
    auto partT = op.template get_destination_cooperative_tensor<LeftT, RightT, float>();

#pragma unroll
    for (uint16_t i = 0; i < accT.get_capacity(); ++i) {
        if (accT.is_valid_element(i)) {
            accT[i] = 0.0f;
        }
    }

    const uint kBlocks = k / kMXFP4BlockSize;
    for (uint kb = 0; kb < kBlocks; ++kb) {
        auto mA = tA.slice(kb * kMXFP4BlockSize, tgid.y * 64);
        auto mB = tB.slice(tgid.x * 64, kb * kMXFP4BlockSize);

#pragma unroll
        for (uint16_t i = 0; i < partT.get_capacity(); ++i) {
            if (partT.is_valid_element(i)) {
                partT[i] = 0.0f;
            }
        }

        op.run(mA, mB, partT);

#pragma unroll
        for (uint16_t i = 0; i < partT.get_capacity(); ++i) {
            if (partT.is_valid_element(i)) {
                accT[i] += partT[i];
            }
        }
    }

    accT.store(mC);
}

// MPP 4.1 decode, materialized: native packed-numeric unpack of 16 FP4 values
// per thread, scaled by the E8M0 block scale, written out as half weights for
// a plain half x half matmul2d.
kernel void mxfp4_dequant_native_half(device const uchar *packed [[buffer(0)]],
                                      device const uchar *scales [[buffer(1)]],
                                      device half *out [[buffer(2)]],
                                      constant uint &rows [[buffer(3)]],
                                      constant uint &depth [[buffer(4)]],
                                      uint2 gid [[thread_position_in_grid]])
{
    uint chunk = gid.x;
    uint row = gid.y;
    uint chunksPerRow = depth / 16;
    if (row >= rows || chunk >= chunksPerRow) {
        return;
    }

    uint kBase = chunk * 16;
    uint byteOffset = row * (depth / 2) + chunk * 8;
    packed_numeric_type<metal_fp4_e2m1_format, 16>::storage_type storage;
    for (ushort i = 0; i < 8; ++i) {
        storage[i] = packed[byteOffset + i];
    }
    auto packedValues = packed_numeric_type<metal_fp4_e2m1_format, 16>(storage);
    vec<float, 16> values = unpack<float>(packedValues);

    // A 16-value chunk never straddles a 32-value scale block.
    uint kBlocks = depth / kMXFP4BlockSize;
    float scale = e8m0_to_float(scales[row * kBlocks + kBase / kMXFP4BlockSize]);
    for (ushort i = 0; i < 16; ++i) {
        out[row * depth + kBase + i] = half(values[i] * scale);
    }
}

// MPP 4.0 emulation: nibble lookup table instead of native FP4 instructions,
// same E8M0 block scale, same materialized half output.

kernel void mxfp4_dequant_lut_half(device const uchar *packed [[buffer(0)]],
                                   device const uchar *scales [[buffer(1)]],
                                   device half *out [[buffer(2)]],
                                   constant uint &rows [[buffer(3)]],
                                   constant uint &depth [[buffer(4)]],
                                   uint2 gid [[thread_position_in_grid]])
{
    uint byteIndex = gid.x;
    uint row = gid.y;
    uint bytesPerRow = depth / 2;
    if (row >= rows || byteIndex >= bytesPerRow) {
        return;
    }

    uint k = byteIndex * 2;
    uint kBlocks = depth / kMXFP4BlockSize;
    float scale = e8m0_to_float(scales[row * kBlocks + k / kMXFP4BlockSize]);

    uchar bits = packed[row * bytesPerRow + byteIndex];
    out[row * depth + k] = half(float(kFP4E2M1Table[bits & 0x0f]) * scale);
    out[row * depth + k + 1] = half(float(kFP4E2M1Table[bits >> 4]) * scale);
}
