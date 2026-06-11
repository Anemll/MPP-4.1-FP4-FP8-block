# MXFP4 MLP on Apple GPUs with MPP 4.1 — Porting Guide

Scaffolding notes for a system currently on macOS 26 (MPP 4.0, no FP4
TensorOps headers) that wants to be ready for the macOS 27 / MSL 4.1 native
FP4 + scale-plane path. Everything here was measured and validated on an
Apple M5 with macOS 27.0 in the MetalFP41Probe project.

## TL;DR

MXFP4 = FP4 E2M1 weight data + one E8M0 scale byte per 32 weights along K.
On macOS 27 + M5, `matmul2d` consumes the FP4 data and the E8M0 scale plane
**together, natively** — and the scale plane is essentially free. On
macOS 26 the best strategy is LUT-dequant to half + a plain `half x half`
`matmul2d`; it is ~2.6x slower than the native path but uses no 4.1 API, so
it is the correct fallback arm of the same code.

Measured MLP cost on M5 (tokens=128, hidden=1024, intermediate=4096, gated
MLP `down(silu(gate(x)) * up(x))`, dequant inside the timed loop):

| Strategy | Needs | ms/MLP | TOPS |
|---|---|---:|---:|
| Native scale-plane `matmul2d` (n64 tile) | macOS 27 + M5 | 0.34–0.40 | ~9 |
| Host multi-plane `MTLTensor`, bindless handle | macOS 27 + M5 | 0.35 | 9.2 |
| LUT dequant -> half, then `h_h_f` matmul | macOS 26+, any Metal-4 GPU | ~0.88 | 3.7 |
| In-register manual scaling (any variant) | — | 1.5–1.6 | ~2.1 (don't bother) |
| Reference: raw FP4, no scales | macOS 27 + M5 | 0.286 | 11.3 |
| Reference: half-weights MLP | macOS 26+ | ~0.62 | 5.2 |
| Reference: W4A8 `i8 x int4b -> i32` | macOS 26+ | 0.18 | 17.8 |

Notes that follow from the probe:
- Native scale-plane MXFP4 reaches ~85% of unscaled-FP4 throughput; the
  remaining gap is the `transpose_right` operand form, not the scaling.
- Do not build manual-scaling fused kernels (per-32-block matmuls or
  cooperative-input operand fill): every variant measured slower than plain
  dequant-to-half.
- W4A8 (INT4 weights, INT8 activations) is still ~2x faster than native
  MXFP4 because 8-bit activations double the MAC rate; FP4 weight paths top
  out at the same ~11 TOPS class as INT4/FP8 with half activations.
- `matmul2d` accumulates in full width (FP32-like) even with half
  destinations; half destinations overflow to `inf` only at the final
  rounding (no saturation, no precision loss along K).

## Data layout (shared by all paths)

```text
weights: row-major [n][k], two FP4 E2M1 values per byte, low nibble = even k
         row stride = k/2 bytes; k must be a multiple of 32
scales:  one E8M0 byte per (output row j, 32-wide k-block kb)
         scales[j * (k/32) + kb]; value = 2^(byte - 127)
```

This is exactly what the macOS 27 host API expects as a multi-plane
`MTLTensor` with dims `{k, n}`, strides `{1, k}`, scale plane blockFactors
`{32, 1}` — so data prepared on macOS 26 needs **no repacking** when the
native path lights up.

FP4 E2M1 nibble values (bit-identical to the M5 hardware decode):

```text
nibble:  0    1    2    3    4    5    6    7    8     9    ...  15
value:   0.0  0.5  1.0  1.5  2.0  3.0  4.0  6.0  -0.0  -0.5 ... -6.0
```

## Feature detection

Shader side (compile-time): the macOS 27 toolchain defines
`__HAVE_TENSOR_MULTIPLANE__` when the `tensor_blockwise` scale-plane
machinery exists. The include below keys everything off it, so one source
file serves both OS versions.

Host side (runtime):

```swift
var hasNativeMXFP4 = false
if #available(macOS 27.0, *) {
    // Definitive probe: try creating a scale-plane tensor.
    hasNativeMXFP4 = (try? makeMXFP4WeightTensor(device: device, ...)) != nil
}
```

## MSL include (drop-in, compiles on macOS 26 and 27)

Verified to compile under `-std=metal3.2` (fallback path only) and
`-std=metal4.1` (all paths) with the macOS 27 toolchain.

```metal
// MXFP4ScaledMatmul.h
//
// Portable Metal include for DeepSeek-style MXFP4 (FP4 E2M1 weights + E8M0
// block-32 scales) MLP matmuls on Apple GPUs.
//
//   MXFP4_HAS_NATIVE_SCALE_PLANE   matmul2d consumes FP4 + E8M0 scale plane
//                                  directly (macOS 27 toolchain, M5 GPU)
//
// When unavailable, use the always-present mxfp4_dequant_to_half kernel and
// run a plain half x half matmul2d.

#pragma once

#include <metal_stdlib>
#include <metal_packed_numeric>
#if __has_include(<MetalPerformancePrimitives/MetalPerformancePrimitives.h>)
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#define MXFP4_HAS_MPP 1
#endif

#if defined(__HAVE_TENSOR_MULTIPLANE__) && defined(MXFP4_HAS_MPP)
#define MXFP4_HAS_NATIVE_SCALE_PLANE 1
#endif

namespace mxfp4 {

using namespace metal;

constant constexpr int kBlockSize = 32;

// E8M0 decode: value = 2^(bits - 127). Exact bit manipulation, no exp2 call.
inline float e8m0_to_float(uchar bits)
{
    return as_type<float>(uint(bits) << 23);
}

// FP4 E2M1 values by nibble; bit-identical to the M5 hardware decode.
constant half kFP4E2M1Table[16] = {
    0.0h, 0.5h, 1.0h, 1.5h, 2.0h, 3.0h, 4.0h, 6.0h,
    -0.0h, -0.5h, -1.0h, -1.5h, -2.0h, -3.0h, -4.0h, -6.0h
};

inline half decode_fp4(uchar nibble)
{
    return kFP4E2M1Table[nibble & 0x0f];
}

} // namespace mxfp4

// ---------------------------------------------------------------------------
// Fallback path (macOS 26 and 27): LUT dequant to half, then run a plain
// half x half matmul2d (or any FP16 GEMM). One thread per packed byte.
// Dispatch: grid (k/2, n), threadsPerThreadgroup e.g. (16, 16).
// ---------------------------------------------------------------------------
kernel void mxfp4_dequant_to_half(device const uchar *packed [[buffer(0)]],
                                  device const uchar *scales [[buffer(1)]],
                                  device half *out [[buffer(2)]],
                                  constant uint &rows [[buffer(3)]],   // n
                                  constant uint &depth [[buffer(4)]],  // k
                                  uint2 gid [[thread_position_in_grid]])
{
    uint byteIndex = gid.x;
    uint row = gid.y;
    uint bytesPerRow = depth / 2;
    if (row >= rows || byteIndex >= bytesPerRow) {
        return;
    }

    uint k = byteIndex * 2;
    uint kBlocks = depth / mxfp4::kBlockSize;
    float scale = mxfp4::e8m0_to_float(scales[row * kBlocks + k / mxfp4::kBlockSize]);

    uchar bits = packed[row * bytesPerRow + byteIndex];
    out[row * depth + k] = half(float(mxfp4::decode_fp4(bits)) * scale);
    out[row * depth + k + 1] = half(float(mxfp4::decode_fp4(bits >> 4)) * scale);
}

// ---------------------------------------------------------------------------
// Native scale-plane path (macOS 27 / MPP 4.1 / M5 only).
//
// Constraints enforced by the MPP 4.1 headers:
//   - scale element type must be metal_fp8_ue8m0_format
//   - block sizes must be (32, 1)
//   - a scaled RIGHT operand requires transpose_right == true, i.e. the
//     weight tensor is (k, n) with k contiguous: exactly the row-major
//     [n][k] packed layout above
//   - a scaled LEFT operand requires transpose_left == false
//   - the destination must not carry scales
//
// One threadgroup computes a 64 x NT output tile with 4 SIMD groups.
// Dispatch: threadgroups ((n + NT-1)/NT, (m + 63)/64),
//           threadsPerThreadgroup (threadExecutionWidth * 4, 1, 1).
// NT = 64 measured fastest on M5 for MLP shapes (NT = 128 was ~2x slower).
// ---------------------------------------------------------------------------
#if defined(MXFP4_HAS_NATIVE_SCALE_PLANE)

namespace mxfp4 {

using namespace mpp::tensor_ops;

using scales_tag = tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;

// Inline-constructed variant: weight bytes and scale bytes arrive as raw
// device pointers (classic setBuffer binding).
template <int NT>
inline void scaled_matmul_tile(device half *a,            // activations [m][k], k contiguous
                               device uchar *weights,     // FP4 [n][k] packed
                               device const uchar *scales,
                               device float *c,           // out [m][n], n contiguous
                               uint m,
                               uint n,
                               uint k,
                               uint2 tgid)
{
    constexpr auto desc = matmul2d_descriptor(64, NT, static_cast<int>(dynamic_extent),
                                              /*transpose_left*/ false,
                                              /*transpose_right*/ true,
                                              /*relaxed_precision*/ false);
    matmul2d<desc, execution_simdgroups<4>> op;

    using ATensor = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    using BTensor = tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline, scales_tag>;
    using CTensor = tensor<device float, dextents<int32_t, 2>, tensor_inline>;

    auto tA = ATensor((ATensor::data_handle_type)a,
                      dextents<int32_t, 2>{int32_t(k), int32_t(m)},
                      array<int32_t, 2>{1, int32_t(k)});
    scales_tag plane(reinterpret_cast<scales_tag::data_handle_type>(const_cast<device uchar *>(scales)));
    auto tB = BTensor((BTensor::data_handle_type)weights,
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

// Handle variant: the weight tensor is a host-created multi-plane MTLTensor
// (FP4 data plane + E8M0 scales plane) bound bindlessly. Put this struct in
// an argument buffer slot, write tensor.gpuResourceID into it from the host
// (setBytes works), and call useResource(tensor, .read) on the encoder.
// Extents and the scale plane travel inside the handle.
struct scaled_weights_args {
    tensor<device metal_fp4_e2m1_format,
           dextents<int32_t, 2>,
           tensor_handle,
           scales_tag> weights;
};

template <int NT>
inline void scaled_matmul_tile(device half *a,
                               constant scaled_weights_args &args,
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

} // namespace mxfp4

// Ready-made kernels. An MLP is three of these plus an elementwise SwiGLU:
//   gate = x @ Wg^T ; up = x @ Wu^T ; mid = silu(gate) * up ; out = mid @ Wd^T
kernel void mxfp4_scaled_matmul_n64(device half *a [[buffer(0)]],
                                    device uchar *weights [[buffer(1)]],
                                    device float *c [[buffer(2)]],
                                    constant uint &m [[buffer(3)]],
                                    constant uint &n [[buffer(4)]],
                                    constant uint &k [[buffer(5)]],
                                    device const uchar *scales [[buffer(6)]],
                                    uint2 tgid [[threadgroup_position_in_grid]])
{
    mxfp4::scaled_matmul_tile<64>(a, weights, scales, c, m, n, k, tgid);
}

kernel void mxfp4_scaled_matmul_handle_n64(device half *a [[buffer(0)]],
                                           constant mxfp4::scaled_weights_args &args [[buffer(1)]],
                                           device float *c [[buffer(2)]],
                                           constant uint &m [[buffer(3)]],
                                           constant uint &n [[buffer(4)]],
                                           constant uint &k [[buffer(5)]],
                                           uint2 tgid [[threadgroup_position_in_grid]])
{
    mxfp4::scaled_matmul_tile<64>(a, args, c, m, n, k, tgid);
}

#endif // MXFP4_HAS_NATIVE_SCALE_PLANE
```

## Host side (Swift)

### Creating the multi-plane weight tensor (macOS 27 only)

Wraps existing `MTLBuffer`s — no data copies, same bytes the fallback path
uses.

```swift
@available(macOS 27.0, *)
func makeMXFP4WeightTensor(
    device: MTLDevice,
    dataBuffer: MTLBuffer,    // n * k/2 bytes, packed FP4 [n][k]
    scaleBuffer: MTLBuffer,   // n * k/32 bytes, E8M0
    k: Int,
    n: Int
) throws -> MTLTensor {
    let dims = [NSInteger(k), NSInteger(n)]
    let strides = [NSInteger(1), NSInteger(k)]
    let blocks = [NSInteger(32), NSInteger(1)]
    // MTLTensorExtents(rank:values:) wrappers elided for brevity.

    let scalePlane = MTLTensorAuxiliaryPlaneDescriptor()
    scalePlane.dataType = .float8ue8m0
    scalePlane.blockFactors = extents(blocks)
    let planes = MTLTensorAuxiliaryPlaneDescriptorMap()
    planes.setDescriptor(scalePlane, for: .scales)

    let desc = MTLTensorDescriptor()
    desc.dimensions = extents(dims)
    desc.strides = extents(strides)      // REQUIRED for buffer-backed tensors
    desc.dataType = .float4e2m1
    desc.usage = .compute
    desc.resourceOptions = .storageModeShared
    desc.auxiliaryPlanes = planes

    let attachments = MTLTensorBufferAttachments()
    attachments.setBuffer(dataBuffer, offset: 0, for: .data)
    attachments.setBuffer(scaleBuffer, offset: 0, for: .scales)

    return try device.makeTensor(descriptor: desc, attachments: attachments)
}
```

### Binding and dispatch

There is no `setTensor` on the classic compute encoder — bind bindlessly:

```swift
encoder.setComputePipelineState(pipeline)        // mxfp4_scaled_matmul_handle_n64
encoder.setBuffer(activations, offset: 0, index: 0)
var rid = weightTensor.gpuResourceID             // 8 bytes, fills scaled_weights_args
encoder.setBytes(&rid, length: MemoryLayout<MTLResourceID>.stride, index: 1)
encoder.setBuffer(output, offset: 0, index: 2)
// m, n, k at indices 3, 4, 5 ...
encoder.useResource(weightTensor, usage: .read)

let nTile = 64
let grid = MTLSize(width: (n + nTile - 1) / nTile, height: (m + 63) / 64, depth: 1)
let threads = MTLSize(width: pipeline.threadExecutionWidth * 4, height: 1, depth: 1)
encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
```

The raw-pointer kernel (`mxfp4_scaled_matmul_n64`) skips the tensor object
entirely: bind the weight and scale buffers with plain `setBuffer` at
indices 1 and 6. Identical performance; use whichever fits your binding
model.

### macOS 26 fallback arm

```swift
// once per weight use (per expert activation in MoE):
dispatch mxfp4_dequant_to_half  (grid (k/2, n))      // -> half weights
// then your existing FP16 matmul2d / GEMM path
```

## Gotchas (each one cost real debugging time)

1. `desc.strides` must be non-nil for buffer-backed tensor creation; the
   validator rejects nil at rank >= 2. Strides for format data types must be
   128-byte aligned at dim >= 1 (k/2 bytes per row: any k >= 256 works; 1024
   is fine).
2. A scaled right operand **must** use `transpose_right = true`. The
   un-transposed form that works for unscaled FP4 fails the static_assert.
3. `run()` into a *cooperative destination* tensor **overwrites** in default
   `multiply` mode; chunked accumulation needs
   `matmul2d_descriptor::mode::multiply_accumulate`. (Only relevant if you
   write manual-scaling kernels — which, per the numbers above, you should
   not.)
4. Input cooperative tensors require `execution_simdgroup` (single SIMD
   group) scope — `execution_simdgroups<4>` fails a static_assert.
5. The cooperative-tensor element predicate is `is_valid_element(i)`; the
   `get_mask(i)` shown in some header comments does not exist.
6. dim0 of a format-type tensor must be a multiple of 32 elements; with
   scale planes every dimension must divide by its block factor.
7. Half destinations overflow to `inf` past 65504 (true value >= 65520).
   Accumulation itself is full-width, so prefer float destinations for long
   reductions; on M5 the cost difference is ~3%.
8. The E8M0 decode is one shift: `as_type<float>(uint(byte) << 23)`. Byte
   value 127 = 1.0; byte 0 decodes to 0.
