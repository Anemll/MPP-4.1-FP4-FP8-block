# Metal FP4/FP8 4.1 Probe

SwiftPM command-line project for testing Metal 4.1 / MSL 4.1 low-precision formats, tensor scale planes, and TensorOps/MPP MLP performance on Apple Silicon.

## Objectives

This project is meant to answer four practical questions:

- Do the new Metal 4.1 packed floating-point formats compile and round-trip correctly?
- Can `MTLTensor` create FP4/FP8 tensors and attach block scale planes?
- How fast are FP4, FP8, INT4, INT2, INT8, and half paths for large matmul and MLP-shaped workloads?
- Which paths look native through TensorOps/MPP, and which paths would need custom emulation such as NVFP4?

The main workload target is a transformer MLP:

```text
down(silu(gate(x)) * up(x))
```

All MLP quantized-weight rows use the same quantized type for gate, up, and down weights.

## Requirements

- macOS 27.0 or later
- Xcode 27.0 or later
- Apple Silicon GPU with Metal family 4

The project runtime-compiles Metal shaders with `MTLLanguageVersion.version4_1`.

## Files

- `Sources/MetalFP41Probe/main.swift`: Swift host code, tensor creation probes, benchmarks, timing, checksums.
- `Sources/MetalFP41Probe/Shaders/PackedNumericKernels.metal`: scalar pack/unpack, scalar matmul, scalar MLP kernels.
- `Sources/MetalFP41Probe/Shaders/MPPMLPKernels.metal`: TensorOps/MPP/NAX MLP kernels.
- `Sources/MetalFP41Probe/Shaders/MXFP4Kernels.metal`: DeepSeek-style MXFP4 expert kernels (fused native FP4 + E8M0, and dequant-to-half paths).
- `Sources/MetalFP41Probe/Shaders/AccumProbeKernels.metal`: matmul2d accumulator precision/overflow probe kernels.
- `docs/MXFP4-MPP41-GUIDE.md`: self-contained porting guide for MXFP4 MLPs on MPP 4.1, with a drop-in MSL include (compiles on macOS 26 with fallback, macOS 27 native) and host-side Swift recipes.
- `Package.swift`: SwiftPM package.

## Default Probe

Run:

```sh
swift run MetalFP41Probe
```

This tests:

- `metal_fp4_e2m1_format`
- `metal_fp8_e4m3_format`
- `metal_fp8_e5m2_format`

For each format it:

- packs and unpacks known values on the GPU
- checks the unpacked values against expected clipped/rounded values
- prints packed bytes
- creates rank-2 `MTLTensor` objects with `usage = .compute`

Expected final line:

```text
PASS
```

To skip tensor creation and run only shader pack/unpack checks:

```sh
swift run MetalFP41Probe -- --skip-tensor
```

## Tensor Scale-Plane Probe

The default run also creates multi-plane `MTLTensor` objects with:

- data plane: `float4e2m1`, `float8e4m3`, or `float8e5m2`
- scale plane: `float8ue8m0`
- block factors: `{32, 1}`

This checks the Metal 4.1 / macOS 27 host API for MX-style block scaling. The public `MTLTensor` scale plane is E8M0 with a 32-element first-dimension block factor, so this should not be described as native NVFP4 support.

Current expected output includes:

```text
MTLTensor scale-plane creation:
  [ok] float4e2m1 + e8m0 scales: dims=64x2 scaleType=145 blockFactors=32x1 rank=2
  [ok] float8e4m3 + e8m0 scales: dims=64x2 scaleType=145 blockFactors=32x1 rank=2
  [ok] float8e5m2 + e8m0 scales: dims=64x2 scaleType=145 blockFactors=32x1 rank=2
```

## Big Matmul Benchmark

Run:

```sh
swift run MetalFP41Probe -- --benchmark
```

Default shape:

```text
A: 2048 x 2048
B: 2048 x 2048
C: 2048 x 2048
```

Rows tested:

- FP4 E2M1 x FP4 E2M1 -> FP32 accumulate
- half x FP4 E2M1 -> half accumulate
- half x INT4 -> half accumulate
- INT8 x half -> half accumulate
- INT4 x INT4 -> INT32 accumulate

Optional size overrides:

```sh
swift run MetalFP41Probe -- --benchmark --matmul-size=1024 --iterations=5
swift run MetalFP41Probe -- --benchmark --m=1024 --n=2048 --k=4096
```

FP4 unpack group sweep:

```sh
swift run MetalFP41Probe -- --benchmark --fp4-group-sweep
```

This scalar benchmark is useful for validating format logic and rough memory behavior. It is not the main TensorOps performance result.

## MLP Benchmark

Run:

```sh
swift run MetalFP41Probe -- --mlp-benchmark
```

Default shape:

```text
tokens = 128
hidden = 1024
intermediate = 4096
```

Optional overrides:

```sh
swift run MetalFP41Probe -- --mlp-benchmark --mlp-tokens=64 --mlp-hidden=1024 --mlp-intermediate=4096 --mlp-iterations=5
```

Add FP4 group-size sweep:

```sh
swift run MetalFP41Probe -- --mlp-benchmark --fp4-group-sweep
```

Scalar MLP rows:

- half weights: `half x half -> half`
- INT8 weights: `half x int8_t -> half`
- INT4 weights: `half x int4b_format -> half`
- INT2 weights: `half x int2b_format -> half`
- FP4 E2M1 weights: `half x FP4 -> half`
- optional FP4 group-size rows: g16 and g8

TensorOps/MPP/NAX MLP rows:

- half weights: `h_h_f + f_h_f`
- INT8 weights: `h_i8_f + f_i8_f`
- INT4 weights: `h_i4_f + h_i4_f`
- INT2 weights: `h_i2_f + h_i2_f` when exposed by the runtime compiler
- FP4 E2M1 weights: `h_fp4_f + h_fp4_f`
- FP8 E4M3 weights: `h_fp8_f + h_fp8_f`
- FP8 E4M3 weights with half outputs: `h_fp8_h + h_fp8_h`
- FP8 E5M2 weights: `h_fp8_f + h_fp8_f`
- W8A8 DS4-style path: `i8 x int8_t -> i32`
- W4A8 DS4-style path: `i8 x int4b -> i32`
- W2A8 DS4-style path: `i8 x int2b -> i32` when exposed by the runtime compiler

DeepSeek-style MXFP4 expert rows (FP4 E2M1 data + E8M0 scale per 32 weights along K):

- MXFP4 fused MPP 4.1: native FP4 `matmul2d` per 32-wide k-block, E8M0 scale applied in registers via cooperative destination tensors, n128 and n64 tiles
- MXFP4 dequant MPP 4.1: native FP4 unpack + scale to materialized half weights, then `h_h_f + f_h_f`
- MXFP4 dequant MPP 4.0: LUT nibble decode + scale to materialized half weights (pre-M5 emulation, no FP4 instructions), then `h_h_f + f_h_f`

Dequantization of all three weight matrices runs inside the timed loop, since MoE expert weights cannot all be kept materialized.

MPP/NAX rows run only when `tokens` is a multiple of 64.

## Representative M5 Result

Last measured command:

```sh
swift run MetalFP41Probe -- --mlp-benchmark --fp4-group-sweep --mlp-iterations=10
```

Shape:

```text
tokens=128, hidden=1024, intermediate=4096
```

Best row is used as the baseline, and other rows report slowdown relative to it.

| Row | ms/MLP | TOPS | Slowdown |
|---|---:|---:|---:|
| MPP/NAX W8A8: `i8 x int8_t -> i32` | 0.168 | 19.201 | 1.00x |
| MPP/NAX W2A8: `i8 x int2b -> i32` | 0.172 | 18.756 | 1.02x |
| MPP/NAX W4A8: `i8 x int4b -> i32` | 0.183 | 17.565 | 1.09x |
| MPP/NAX FP8 E5M2: `h_fp8_f + h_fp8_f` | 0.279 | 11.536 | 1.66x |
| MPP/NAX FP8 E4M3: `h_fp8_f + h_fp8_f` | 0.280 | 11.495 | 1.67x |
| MPP/NAX INT4 weights: `h_i4_f + h_i4_f` | 0.284 | 11.353 | 1.69x |
| MPP/NAX FP4 E2M1: `h_fp4_f + h_fp4_f` | 0.286 | 11.263 | 1.70x |
| MPP/NAX FP8 E4M3: `h_fp8_h + h_fp8_h` | 0.288 | 11.172 | 1.71x |
| MPP/NAX INT2 weights: `h_i2_f + h_i2_f` | 0.289 | 11.158 | 1.72x |
| MPP/NAX INT8 weights: `h_i8_f + f_i8_f` | 0.617 | 5.222 | 3.67x |
| MPP/NAX half weights: `h_h_f + f_h_f` | 0.626 | 5.147 | 3.73x |
| MXFP4 dequant MPP 4.1 (native FP4 unpack) + `h_h_f` | 0.981 | 3.282 | 5.84x |
| MXFP4 dequant MPP 4.0 (LUT, no FP4 hardware) + `h_h_f` | 0.992 | 3.249 | 5.90x |
| MXFP4 fused MPP 4.1 (native FP4 + E8M0 in-register, n64) | 1.461 | 2.205 | 8.70x |
| MXFP4 fused MPP 4.1 (native FP4 + E8M0 in-register, n128) | 2.272 | 1.418 | 13.52x |

The scalar rows are much slower and are mainly kept for format validation and fallback comparison.

## DeepSeek-Style MXFP4 Experts: MPP 4.0 vs MPP 4.1

The MXFP4 rows answer the missing comparison: does the new M5 FP4 TensorOps
support help a DeepSeek-style block-scaled FP4 expert MLP, given that the new
`MTLTensor` scale plane cannot be consumed from MSL TensorOps?

MXFP4 always carries block scales, so only scaled paths are compared; the
unscaled `h_fp4_f` row in the main table is a format probe, not a deployable
quantization. Measured on M5, tokens=128, hidden=1024, intermediate=4096,
per-expert dequantization inside the timed loop:

- The practical MXFP4 path today is dequantize-to-half then `h_h_f`:
  0.981 ms with the MPP 4.1 native FP4 unpack, 0.992 ms with the MPP 4.0
  LUT decode. Both produce identical checksums, confirming the LUT matches
  the hardware decode bit-for-bit.
- The fused MPP 4.1 path (one native FP4 `matmul2d` per 32-wide k-block,
  E8M0 scales applied in registers through cooperative destination tensors)
  is slower than materialized dequant: 1.461 ms with the n64 tile, 2.272 ms
  with n128. The per-block matmul granularity destroys NAX efficiency, and
  the doubled cooperative-tensor register pressure makes the n128 tile worse
  than n64. Checksums for n64 and n128 match exactly.

A third fused variant uses Apple's cooperative-input pattern
(`get_right_input_cooperative_tensor`, `mode::multiply_accumulate`,
`execution_simdgroup` scope): FP4 weights are LUT-decoded and scaled while
filling the matmul's right-input cooperative tensor, so the k-chunk is
decoupled from the 32-wide scale block and nothing is materialized. This
kernel never touches the FP4 TensorOps type, so it is the MPP 4.0-style
fused path and needs no M5 FP4 hardware. It lands at 1.57 ms (k32 chunks) /
1.90 ms (k64 chunks; the bigger operand tensor costs more registers than the
longer reduction saves). k32 and k64 produce identical checksums. It still
loses to materialized dequant because the O(N x K) per-element operand fill
is repeated for every 32-row m-tile, while materialized dequant decodes each
weight exactly once and shares it across all token tiles.

The winner is the native scale-plane path, which an earlier draft of this
README incorrectly called "not expressible". MSL does expose it (macOS 27 /
MSL 4.1): tag the weight tensor with
`tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>`
and pass the E8M0 scale pointer as the plane when constructing the inline
tensor; `matmul2d` then consumes FP4 data and scales together. Constraints
enforced by the MPP headers: scale type must be E8M0, block sizes must be
(32, 1), a scaled right operand must use `transpose_right = true` (extents
(k, n), the natural row-major [n][k] packed weight layout), a scaled left
operand must not be transposed, and the destination may not carry scales.

The full strategy matrix, all with correct per-32 E8M0 scaling (tokens=128):

| | Materialized dequant | Fused, in-register |
|---|---:|---:|
| MPP 4.1 native scale-plane `matmul2d` | — | **0.37 ms (n64), 0.67 ms (n128)** |
| MPP 4.1 (native FP4, manual scaling) | 0.880 ms | 1.46 ms (FP4 matmul + scaled partials) |
| MPP 4.0 (LUT, no FP4 hw) | 0.887 ms | 1.57 ms (coop-input) |

The host-side route also works end to end: create the multi-plane tensor
with `MTLTensorDescriptor` (FP4 data plane + E8M0 scales plane,
`newTensorWithDescriptor:attachments:` wrapping existing `MTLBuffer`s —
explicit `strides` are required for buffer-backed creation), bind it
bindlessly (write `gpuResourceID` into an argument struct and call
`useResource`), and declare the kernel argument as
`tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_handle,
tensor_blockwise<...>>`. It benchmarks at 0.35 ms (n64) — identical to the
inline-constructed version.

All three native scale-plane rows (inline n64/n128 and host-tensor handle)
produce the same checksum (403573) as the coop-input implementation,
cross-validating per-32 scaling against an independent software path. A perf-only control (`--mxfp4-decompose`) running
the identical transposed-B FP4 matmul without the scale plane lands at
0.380 ms, the same as the scaled n64 row: the scale plane itself is
essentially free, and the remaining ~1.3x gap to the raw FP4 row (0.286 ms)
is the cost of the transposed operand form.

`--mxfp4-decompose` adds two perf-only probe rows (intentionally wrong math)
that attribute the fused path's cost:

- full-K native FP4 matmul into a cooperative tensor + one elementwise scale
  pass + store: 0.276 ms, same as the raw `h_fp4_f` row (0.285 ms). The
  cooperative-tensor path, scale loads, and scale multiplies are free.
- the same k-block split as the fused kernel with the scaling removed:
  1.456 ms, identical to the full fused kernel (1.461 ms).

So 100% of the fused-path penalty is the 32-wide accumulation granularity
(one `matmul2d` per scale block), 0% is the scaling math. A
`matmul2d` variant that accepted a per-k-block scale vector, or wider scale
blocks, would close the gap.

Conclusions:

- MPP 4.1 gives a real MXFP4 advantage, but only through the native
  scale-plane path: ~0.37 ms (8.7 TOPS) vs ~0.88 ms for the best
  MPP 4.0-feasible strategy (dequant + `h_h_f`) — about 2.4x. Every manual
  scaling strategy (dequant, destination-scaled, coop-input) is 4.0-class
  performance regardless of which decode instructions it uses.
- Native scale-plane MXFP4 beats the half-weights MLP (0.59-0.63 ms) by
  ~1.6x while using 4x smaller weights, and runs within ~1.3x of unscaled
  raw FP4 — the residue being the transposed operand form, not the scaling.
- DS4-style W4A8 (0.18 ms) is still about 2x faster than native scale-plane
  MXFP4, because 8-bit activations double the MAC rate (FP4 paths top out at
  the same ~11 TOPS rate class as INT4/FP8 with half activations).

## Accumulator Precision and Overflow

Run:

```sh
swift run MetalFP41Probe -- --accum-probe --skip-tensor
```

This probes whether `matmul2d` with a half destination accumulates in half
(precision loss along K, overflow at 65504) or wider. All inputs are constant
fills so every output element has the same exact integer value.

Measured on M5:

- Sum-of-ones dot products are exact through K=131072 with a float
  destination and exact up to 65504 with a half destination, for
  `half x half`, `half x FP4`, `FP4 x FP4`, and `FP8E4M3 x FP8E4M3`. A serial
  half accumulator would have stalled at 2048, so the accumulator is
  effectively full-width (FP32-like); there is no "14-bit" precision loss
  along K.
- Half-destination results deviate from exact only by one final
  float-to-half rounding (e.g. exact 65448 -> 65440), and overflow to `inf`
  exactly when the true value crosses the half rounding boundary 65520
  (e.g. FP8 x FP8 with product 256 at K=256: half dest `inf`, float dest
  exact 65536). No saturation to 65504 and no wrapping was observed.
- Practical takeaway: `fp8 x fp8 -> half` (and `-> half` paths generally)
  can overflow to `inf` for large-magnitude dot products, but switching the
  destination to float costs almost nothing in this benchmark
  (`h_fp8_f` 0.280 ms vs `h_fp8_h` 0.288 ms) and is exact.

## Interpreting MXFP4 and NVFP4

The Metal tensor scale-plane probe maps naturally to MX-style scaling:

- FP4 E2M1 data
- E8M0 scale
- block size 32

This scaled FP4 comparison matrix is now measured (see the MXFP4 section
above):

- native MXFP4-style TensorOps consuming the scale plane directly in
  `matmul2d`: implemented via the MSL `tensor_blockwise<tensor_plane_scales,
  device metal_fp8_ue8m0_format, 32, 1>` tag; the fastest scaled path at
  ~0.37 ms and within ~1.3x of raw FP4
- fused dequant without materialization: implemented two ways (per-k-block
  destination scaling, coop-input operand fill); both about 4x slower than
  the native scale-plane path
- LUT/materialized dequant: implemented; the fastest option that works
  without M5 FP4 hardware

The key result: scale-plane-aware TensorOps is real and nearly free — the
scale plane adds no measurable cost over the transposed FP4 matmul it
requires. Every software scaling strategy loses to it by 2.4x or more.

NVFP4 is different:

- FP4 E2M1 data
- FP8 E4M3 scale per 16 values
- extra higher-level FP32 scale

Because the public `MTLTensor` auxiliary scale plane is E8M0 with block factor 32, this project treats NVFP4 as a custom/emulated layout on current public APIs. NVFP4 has modest storage overhead versus MXFP4, but the runtime cost depends on whether dequantization can stay inside a fused TensorOps kernel/register path or requires materializing half values.

## Notes

- Checksums marked `raw16 sum` are sums of raw `UInt16` half bit patterns. They are quick output-change checks, not numeric error metrics.
- The current benchmark generates deterministic synthetic data, not model-accurate quantized weights.
- The benchmark is intended for relative path comparison on the same machine, build, and OS. Do not compare raw numbers across different systems without rerunning all rows.
