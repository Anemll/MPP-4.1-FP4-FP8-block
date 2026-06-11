#include <metal_stdlib>
#include <metal_packed_numeric>

using namespace metal;

struct MatmulConfig {
    uint rows;
    uint columns;
    uint depth;
    uint packedDepthBytes;
};

struct MLPConfig {
    uint tokens;
    uint hidden;
    uint intermediate;
};

static inline half silu_half(half value)
{
    float x = float(value);
    return half(x / (1.0f + exp(-x)));
}

static inline half signed_int2_to_half(uchar bits)
{
    return half((int(bits & 0x03) ^ 2) - 2);
}

static inline half signed_int4_to_half(uchar bits)
{
    return half((int(bits & 0x0f) ^ 8) - 8);
}

kernel void fp4_e2m1_roundtrip(device const float *input [[buffer(0)]],
                               device uchar *packedBytes [[buffer(1)]],
                               device float *output [[buffer(2)]])
{
    vec<float, 16> values;
    for (ushort i = 0; i < 16; ++i) {
        values[i] = input[i];
    }

    auto packed = pack<metal_fp4_e2m1_format>(values);
    auto storage = packed.as_storage_type();
    for (ushort i = 0; i < 8; ++i) {
        packedBytes[i] = storage[i];
    }

    vec<float, 16> unpacked = unpack<float>(packed);
    for (ushort i = 0; i < 16; ++i) {
        output[i] = unpacked[i];
    }
}

kernel void fp8_e4m3_roundtrip(device const float *input [[buffer(0)]],
                               device uchar *packedBytes [[buffer(1)]],
                               device float *output [[buffer(2)]])
{
    vec<float, 8> values;
    for (ushort i = 0; i < 8; ++i) {
        values[i] = input[i];
    }

    auto packed = pack<metal_fp8_e4m3_format>(values);
    auto storage = packed.as_storage_type();
    for (ushort i = 0; i < 8; ++i) {
        packedBytes[i] = storage[i];
    }

    vec<float, 8> unpacked = unpack<float>(packed);
    for (ushort i = 0; i < 8; ++i) {
        output[i] = unpacked[i];
    }
}

kernel void fp8_e5m2_roundtrip(device const float *input [[buffer(0)]],
                               device uchar *packedBytes [[buffer(1)]],
                               device float *output [[buffer(2)]])
{
    vec<float, 8> values;
    for (ushort i = 0; i < 8; ++i) {
        values[i] = input[i];
    }

    auto packed = pack<metal_fp8_e5m2_format>(values);
    auto storage = packed.as_storage_type();
    for (ushort i = 0; i < 8; ++i) {
        packedBytes[i] = storage[i];
    }

    vec<float, 8> unpacked = unpack<float>(packed);
    for (ushort i = 0; i < 8; ++i) {
        output[i] = unpacked[i];
    }
}

kernel void fp4_e2m1_matmul_benchmark(device const uchar *aPacked [[buffer(0)]],
                                      device const uchar *bTransposedPacked [[buffer(1)]],
                                      device float *c [[buffer(2)]],
                                      constant MatmulConfig &config [[buffer(3)]],
                                      uint2 gid [[thread_position_in_grid]])
{
    uint column = gid.x;
    uint row = gid.y;
    if (row >= config.rows || column >= config.columns) {
        return;
    }

    uint aRowOffset = row * config.packedDepthBytes;
    uint bRowOffset = column * config.packedDepthBytes;
    float acc = 0.0f;

    for (uint packedOffset = 0; packedOffset < config.packedDepthBytes; packedOffset += 8) {
        packed_numeric_type<metal_fp4_e2m1_format, 16>::storage_type aStorage;
        packed_numeric_type<metal_fp4_e2m1_format, 16>::storage_type bStorage;
        for (ushort i = 0; i < 8; ++i) {
            aStorage[i] = aPacked[aRowOffset + packedOffset + i];
            bStorage[i] = bTransposedPacked[bRowOffset + packedOffset + i];
        }

        auto aPackedValues = packed_numeric_type<metal_fp4_e2m1_format, 16>(aStorage);
        auto bPackedValues = packed_numeric_type<metal_fp4_e2m1_format, 16>(bStorage);
        vec<float, 16> a = unpack<float>(aPackedValues);
        vec<float, 16> b = unpack<float>(bPackedValues);

        for (ushort i = 0; i < 16; ++i) {
            acc = fma(a[i], b[i], acc);
        }
    }

    c[row * config.columns + column] = acc;
}

kernel void fp4_e2m1_matmul_g8_benchmark(device const uchar *aPacked [[buffer(0)]],
                                         device const uchar *bTransposedPacked [[buffer(1)]],
                                         device float *c [[buffer(2)]],
                                         constant MatmulConfig &config [[buffer(3)]],
                                         uint2 gid [[thread_position_in_grid]])
{
    uint column = gid.x;
    uint row = gid.y;
    if (row >= config.rows || column >= config.columns) {
        return;
    }

    uint aRowOffset = row * config.packedDepthBytes;
    uint bRowOffset = column * config.packedDepthBytes;
    float acc = 0.0f;

    for (uint packedOffset = 0; packedOffset < config.packedDepthBytes; packedOffset += 4) {
        packed_numeric_type<metal_fp4_e2m1_format, 8>::storage_type aStorage;
        packed_numeric_type<metal_fp4_e2m1_format, 8>::storage_type bStorage;
        for (ushort i = 0; i < 4; ++i) {
            aStorage[i] = aPacked[aRowOffset + packedOffset + i];
            bStorage[i] = bTransposedPacked[bRowOffset + packedOffset + i];
        }

        auto aPackedValues = packed_numeric_type<metal_fp4_e2m1_format, 8>(aStorage);
        auto bPackedValues = packed_numeric_type<metal_fp4_e2m1_format, 8>(bStorage);
        vec<float, 8> a = unpack<float>(aPackedValues);
        vec<float, 8> b = unpack<float>(bPackedValues);

        for (ushort i = 0; i < 8; ++i) {
            acc = fma(a[i], b[i], acc);
        }
    }

    c[row * config.columns + column] = acc;
}

kernel void fp4_e2m1_matmul_half_benchmark(device const uchar *aPacked [[buffer(0)]],
                                           device const uchar *bTransposedPacked [[buffer(1)]],
                                           device half *c [[buffer(2)]],
                                           constant MatmulConfig &config [[buffer(3)]],
                                           uint2 gid [[thread_position_in_grid]])
{
    uint column = gid.x;
    uint row = gid.y;
    if (row >= config.rows || column >= config.columns) {
        return;
    }

    uint aRowOffset = row * config.packedDepthBytes;
    uint bRowOffset = column * config.packedDepthBytes;
    half acc = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.packedDepthBytes; packedOffset += 8) {
        packed_numeric_type<metal_fp4_e2m1_format, 16>::storage_type aStorage;
        packed_numeric_type<metal_fp4_e2m1_format, 16>::storage_type bStorage;
        for (ushort i = 0; i < 8; ++i) {
            aStorage[i] = aPacked[aRowOffset + packedOffset + i];
            bStorage[i] = bTransposedPacked[bRowOffset + packedOffset + i];
        }

        auto aPackedValues = packed_numeric_type<metal_fp4_e2m1_format, 16>(aStorage);
        auto bPackedValues = packed_numeric_type<metal_fp4_e2m1_format, 16>(bStorage);
        vec<half, 16> a = unpack<half>(aPackedValues);
        vec<half, 16> b = unpack<half>(bPackedValues);

        for (ushort i = 0; i < 16; ++i) {
            acc = fma(a[i], b[i], acc);
        }
    }

    c[row * config.columns + column] = acc;
}

kernel void fp4_e2m1_matmul_half_g8_benchmark(device const uchar *aPacked [[buffer(0)]],
                                              device const uchar *bTransposedPacked [[buffer(1)]],
                                              device half *c [[buffer(2)]],
                                              constant MatmulConfig &config [[buffer(3)]],
                                              uint2 gid [[thread_position_in_grid]])
{
    uint column = gid.x;
    uint row = gid.y;
    if (row >= config.rows || column >= config.columns) {
        return;
    }

    uint aRowOffset = row * config.packedDepthBytes;
    uint bRowOffset = column * config.packedDepthBytes;
    half acc = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.packedDepthBytes; packedOffset += 4) {
        packed_numeric_type<metal_fp4_e2m1_format, 8>::storage_type aStorage;
        packed_numeric_type<metal_fp4_e2m1_format, 8>::storage_type bStorage;
        for (ushort i = 0; i < 4; ++i) {
            aStorage[i] = aPacked[aRowOffset + packedOffset + i];
            bStorage[i] = bTransposedPacked[bRowOffset + packedOffset + i];
        }

        auto aPackedValues = packed_numeric_type<metal_fp4_e2m1_format, 8>(aStorage);
        auto bPackedValues = packed_numeric_type<metal_fp4_e2m1_format, 8>(bStorage);
        vec<half, 8> a = unpack<half>(aPackedValues);
        vec<half, 8> b = unpack<half>(bPackedValues);

        for (ushort i = 0; i < 8; ++i) {
            acc = fma(a[i], b[i], acc);
        }
    }

    c[row * config.columns + column] = acc;
}

kernel void half_fp4_matmul_half_benchmark(device const half *a [[buffer(0)]],
                                           device const uchar *bTransposedPacked [[buffer(1)]],
                                           device half *c [[buffer(2)]],
                                           constant MatmulConfig &config [[buffer(3)]],
                                           uint2 gid [[thread_position_in_grid]])
{
    uint column = gid.x;
    uint row = gid.y;
    if (row >= config.rows || column >= config.columns) {
        return;
    }

    uint aRowOffset = row * config.depth;
    uint bRowOffset = column * config.packedDepthBytes;
    half acc = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.packedDepthBytes; packedOffset += 8) {
        packed_numeric_type<metal_fp4_e2m1_format, 16>::storage_type bStorage;
        for (ushort i = 0; i < 8; ++i) {
            bStorage[i] = bTransposedPacked[bRowOffset + packedOffset + i];
        }

        auto bPackedValues = packed_numeric_type<metal_fp4_e2m1_format, 16>(bStorage);
        vec<half, 16> b = unpack<half>(bPackedValues);
        uint k = packedOffset * 2;
        for (ushort i = 0; i < 16; ++i) {
            acc = fma(a[aRowOffset + k + i], b[i], acc);
        }
    }

    c[row * config.columns + column] = acc;
}

kernel void half_fp4_matmul_half_g8_benchmark(device const half *a [[buffer(0)]],
                                              device const uchar *bTransposedPacked [[buffer(1)]],
                                              device half *c [[buffer(2)]],
                                              constant MatmulConfig &config [[buffer(3)]],
                                              uint2 gid [[thread_position_in_grid]])
{
    uint column = gid.x;
    uint row = gid.y;
    if (row >= config.rows || column >= config.columns) {
        return;
    }

    uint aRowOffset = row * config.depth;
    uint bRowOffset = column * config.packedDepthBytes;
    half acc = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.packedDepthBytes; packedOffset += 4) {
        packed_numeric_type<metal_fp4_e2m1_format, 8>::storage_type bStorage;
        for (ushort i = 0; i < 4; ++i) {
            bStorage[i] = bTransposedPacked[bRowOffset + packedOffset + i];
        }

        auto bPackedValues = packed_numeric_type<metal_fp4_e2m1_format, 8>(bStorage);
        vec<half, 8> b = unpack<half>(bPackedValues);
        uint k = packedOffset * 2;
        for (ushort i = 0; i < 8; ++i) {
            acc = fma(a[aRowOffset + k + i], b[i], acc);
        }
    }

    c[row * config.columns + column] = acc;
}

kernel void half_int4_matmul_half_benchmark(device const half *a [[buffer(0)]],
                                            device const uchar *bTransposedPacked [[buffer(1)]],
                                            device half *c [[buffer(2)]],
                                            constant MatmulConfig &config [[buffer(3)]],
                                            uint2 gid [[thread_position_in_grid]])
{
    uint column = gid.x;
    uint row = gid.y;
    if (row >= config.rows || column >= config.columns) {
        return;
    }

    uint aRowOffset = row * config.depth;
    uint bRowOffset = column * config.packedDepthBytes;
    half acc = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.packedDepthBytes; ++packedOffset) {
        uchar bByte = bTransposedPacked[bRowOffset + packedOffset];
        char b0 = char((int(bByte & 0x0f) ^ 8) - 8);
        char b1 = char((int((bByte >> 4) & 0x0f) ^ 8) - 8);
        uint k = packedOffset * 2;

        acc = fma(a[aRowOffset + k], half(int(b0)), acc);
        acc = fma(a[aRowOffset + k + 1], half(int(b1)), acc);
    }

    c[row * config.columns + column] = acc;
}

kernel void half_int2_matmul_half_benchmark(device const half *a [[buffer(0)]],
                                            device const uchar *bTransposedPacked [[buffer(1)]],
                                            device half *c [[buffer(2)]],
                                            constant MatmulConfig &config [[buffer(3)]],
                                            uint2 gid [[thread_position_in_grid]])
{
    uint column = gid.x;
    uint row = gid.y;
    if (row >= config.rows || column >= config.columns) {
        return;
    }

    uint aRowOffset = row * config.depth;
    uint bRowOffset = column * (config.depth / 4);
    half acc = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.depth / 4; ++packedOffset) {
        uchar bByte = bTransposedPacked[bRowOffset + packedOffset];
        uint k = packedOffset * 4;

        acc = fma(a[aRowOffset + k], signed_int2_to_half(bByte), acc);
        acc = fma(a[aRowOffset + k + 1], signed_int2_to_half(bByte >> 2), acc);
        acc = fma(a[aRowOffset + k + 2], signed_int2_to_half(bByte >> 4), acc);
        acc = fma(a[aRowOffset + k + 3], signed_int2_to_half(bByte >> 6), acc);
    }

    c[row * config.columns + column] = acc;
}

kernel void int8_half_matmul_half_benchmark(device const char *a [[buffer(0)]],
                                            device const half *bTransposed [[buffer(1)]],
                                            device half *c [[buffer(2)]],
                                            constant MatmulConfig &config [[buffer(3)]],
                                            uint2 gid [[thread_position_in_grid]])
{
    uint column = gid.x;
    uint row = gid.y;
    if (row >= config.rows || column >= config.columns) {
        return;
    }

    uint aRowOffset = row * config.depth;
    uint bRowOffset = column * config.depth;
    half acc = half(0.0);

    for (uint k = 0; k < config.depth; ++k) {
        acc = fma(half(int(a[aRowOffset + k])), bTransposed[bRowOffset + k], acc);
    }

    c[row * config.columns + column] = acc;
}

kernel void int4_int8_matmul_benchmark(device const uchar *aPacked [[buffer(0)]],
                                       device const uchar *bTransposedPacked [[buffer(1)]],
                                       device int *c [[buffer(2)]],
                                       constant MatmulConfig &config [[buffer(3)]],
                                       uint2 gid [[thread_position_in_grid]])
{
    uint column = gid.x;
    uint row = gid.y;
    if (row >= config.rows || column >= config.columns) {
        return;
    }

    uint aRowOffset = row * config.packedDepthBytes;
    uint bRowOffset = column * config.packedDepthBytes;
    int acc = 0;

    for (uint packedOffset = 0; packedOffset < config.packedDepthBytes; ++packedOffset) {
        uchar aByte = aPacked[aRowOffset + packedOffset];
        uchar bByte = bTransposedPacked[bRowOffset + packedOffset];

        char a0 = char((int(aByte & 0x0f) ^ 8) - 8);
        char a1 = char((int((aByte >> 4) & 0x0f) ^ 8) - 8);
        char b0 = char((int(bByte & 0x0f) ^ 8) - 8);
        char b1 = char((int((bByte >> 4) & 0x0f) ^ 8) - 8);

        acc += int(a0) * int(b0);
        acc += int(a1) * int(b1);
    }

    c[row * config.columns + column] = acc;
}

kernel void half_mlp_gate_up_benchmark(device const half *x [[buffer(0)]],
                                       device const half *gateWeightTransposed [[buffer(1)]],
                                       device const half *upWeightTransposed [[buffer(2)]],
                                       device half *intermediate [[buffer(3)]],
                                       constant MLPConfig &config [[buffer(4)]],
                                       uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.intermediate) {
        return;
    }

    uint xOffset = token * config.hidden;
    uint weightOffset = channel * config.hidden;
    half gate = half(0.0);
    half up = half(0.0);

    for (uint i = 0; i < config.hidden; ++i) {
        half xv = x[xOffset + i];
        gate = fma(xv, gateWeightTransposed[weightOffset + i], gate);
        up = fma(xv, upWeightTransposed[weightOffset + i], up);
    }

    intermediate[token * config.intermediate + channel] = silu_half(gate) * up;
}

kernel void int8_half_mlp_gate_up_benchmark(device const char *x [[buffer(0)]],
                                            device const half *gateWeightTransposed [[buffer(1)]],
                                            device const half *upWeightTransposed [[buffer(2)]],
                                            device half *intermediate [[buffer(3)]],
                                            constant MLPConfig &config [[buffer(4)]],
                                            uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.intermediate) {
        return;
    }

    uint xOffset = token * config.hidden;
    uint weightOffset = channel * config.hidden;
    half gate = half(0.0);
    half up = half(0.0);

    for (uint i = 0; i < config.hidden; ++i) {
        half xv = half(int(x[xOffset + i]));
        gate = fma(xv, gateWeightTransposed[weightOffset + i], gate);
        up = fma(xv, upWeightTransposed[weightOffset + i], up);
    }

    intermediate[token * config.intermediate + channel] = silu_half(gate) * up;
}

kernel void half_int8_mlp_gate_up_benchmark(device const half *x [[buffer(0)]],
                                            device const char *gateWeightTransposed [[buffer(1)]],
                                            device const char *upWeightTransposed [[buffer(2)]],
                                            device half *intermediate [[buffer(3)]],
                                            constant MLPConfig &config [[buffer(4)]],
                                            uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.intermediate) {
        return;
    }

    uint xOffset = token * config.hidden;
    uint weightOffset = channel * config.hidden;
    half gate = half(0.0);
    half up = half(0.0);

    for (uint i = 0; i < config.hidden; ++i) {
        half xv = x[xOffset + i];
        gate = fma(xv, half(int(gateWeightTransposed[weightOffset + i])), gate);
        up = fma(xv, half(int(upWeightTransposed[weightOffset + i])), up);
    }

    intermediate[token * config.intermediate + channel] = silu_half(gate) * up;
}

kernel void half_int4_mlp_gate_up_benchmark(device const half *x [[buffer(0)]],
                                            device const uchar *gateWeightTransposedPacked [[buffer(1)]],
                                            device const uchar *upWeightTransposedPacked [[buffer(2)]],
                                            device half *intermediate [[buffer(3)]],
                                            constant MLPConfig &config [[buffer(4)]],
                                            uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.intermediate) {
        return;
    }

    uint xOffset = token * config.hidden;
    uint weightByteOffset = channel * (config.hidden / 2);
    half gate = half(0.0);
    half up = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.hidden / 2; ++packedOffset) {
        uchar gateByte = gateWeightTransposedPacked[weightByteOffset + packedOffset];
        uchar upByte = upWeightTransposedPacked[weightByteOffset + packedOffset];
        uint k = packedOffset * 2;

        half xv0 = x[xOffset + k];
        half xv1 = x[xOffset + k + 1];

        gate = fma(xv0, signed_int4_to_half(gateByte), gate);
        gate = fma(xv1, signed_int4_to_half(gateByte >> 4), gate);

        up = fma(xv0, signed_int4_to_half(upByte), up);
        up = fma(xv1, signed_int4_to_half(upByte >> 4), up);
    }

    intermediate[token * config.intermediate + channel] = silu_half(gate) * up;
}

kernel void half_fp4_mlp_gate_up_benchmark(device const half *x [[buffer(0)]],
                                           device const uchar *gateWeightTransposedPacked [[buffer(1)]],
                                           device const uchar *upWeightTransposedPacked [[buffer(2)]],
                                           device half *intermediate [[buffer(3)]],
                                           constant MLPConfig &config [[buffer(4)]],
                                           uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.intermediate) {
        return;
    }

    uint xOffset = token * config.hidden;
    uint weightByteOffset = channel * (config.hidden / 2);
    half gate = half(0.0);
    half up = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.hidden / 2; packedOffset += 8) {
        packed_numeric_type<metal_fp4_e2m1_format, 16>::storage_type gateStorage;
        packed_numeric_type<metal_fp4_e2m1_format, 16>::storage_type upStorage;
        for (ushort i = 0; i < 8; ++i) {
            gateStorage[i] = gateWeightTransposedPacked[weightByteOffset + packedOffset + i];
            upStorage[i] = upWeightTransposedPacked[weightByteOffset + packedOffset + i];
        }

        auto gatePacked = packed_numeric_type<metal_fp4_e2m1_format, 16>(gateStorage);
        auto upPacked = packed_numeric_type<metal_fp4_e2m1_format, 16>(upStorage);
        vec<half, 16> gateW = unpack<half>(gatePacked);
        vec<half, 16> upW = unpack<half>(upPacked);
        uint hiddenOffset = packedOffset * 2;
        for (ushort i = 0; i < 16; ++i) {
            half xv = x[xOffset + hiddenOffset + i];
            gate = fma(xv, gateW[i], gate);
            up = fma(xv, upW[i], up);
        }
    }

    intermediate[token * config.intermediate + channel] = silu_half(gate) * up;
}

kernel void half_fp4_mlp_gate_up_g8_benchmark(device const half *x [[buffer(0)]],
                                              device const uchar *gateWeightTransposedPacked [[buffer(1)]],
                                              device const uchar *upWeightTransposedPacked [[buffer(2)]],
                                              device half *intermediate [[buffer(3)]],
                                              constant MLPConfig &config [[buffer(4)]],
                                              uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.intermediate) {
        return;
    }

    uint xOffset = token * config.hidden;
    uint weightByteOffset = channel * (config.hidden / 2);
    half gate = half(0.0);
    half up = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.hidden / 2; packedOffset += 4) {
        packed_numeric_type<metal_fp4_e2m1_format, 8>::storage_type gateStorage;
        packed_numeric_type<metal_fp4_e2m1_format, 8>::storage_type upStorage;
        for (ushort i = 0; i < 4; ++i) {
            gateStorage[i] = gateWeightTransposedPacked[weightByteOffset + packedOffset + i];
            upStorage[i] = upWeightTransposedPacked[weightByteOffset + packedOffset + i];
        }

        auto gatePacked = packed_numeric_type<metal_fp4_e2m1_format, 8>(gateStorage);
        auto upPacked = packed_numeric_type<metal_fp4_e2m1_format, 8>(upStorage);
        vec<half, 8> gateW = unpack<half>(gatePacked);
        vec<half, 8> upW = unpack<half>(upPacked);
        uint hiddenOffset = packedOffset * 2;
        for (ushort i = 0; i < 8; ++i) {
            half xv = x[xOffset + hiddenOffset + i];
            gate = fma(xv, gateW[i], gate);
            up = fma(xv, upW[i], up);
        }
    }

    intermediate[token * config.intermediate + channel] = silu_half(gate) * up;
}

kernel void half_int2_mlp_gate_up_benchmark(device const half *x [[buffer(0)]],
                                            device const uchar *gateWeightTransposedPacked [[buffer(1)]],
                                            device const uchar *upWeightTransposedPacked [[buffer(2)]],
                                            device half *intermediate [[buffer(3)]],
                                            constant MLPConfig &config [[buffer(4)]],
                                            uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.intermediate) {
        return;
    }

    uint xOffset = token * config.hidden;
    uint weightByteOffset = channel * (config.hidden / 4);
    half gate = half(0.0);
    half up = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.hidden / 4; ++packedOffset) {
        uchar gateByte = gateWeightTransposedPacked[weightByteOffset + packedOffset];
        uchar upByte = upWeightTransposedPacked[weightByteOffset + packedOffset];
        uint k = packedOffset * 4;

        half xv0 = x[xOffset + k];
        half xv1 = x[xOffset + k + 1];
        half xv2 = x[xOffset + k + 2];
        half xv3 = x[xOffset + k + 3];

        gate = fma(xv0, signed_int2_to_half(gateByte), gate);
        gate = fma(xv1, signed_int2_to_half(gateByte >> 2), gate);
        gate = fma(xv2, signed_int2_to_half(gateByte >> 4), gate);
        gate = fma(xv3, signed_int2_to_half(gateByte >> 6), gate);

        up = fma(xv0, signed_int2_to_half(upByte), up);
        up = fma(xv1, signed_int2_to_half(upByte >> 2), up);
        up = fma(xv2, signed_int2_to_half(upByte >> 4), up);
        up = fma(xv3, signed_int2_to_half(upByte >> 6), up);
    }

    intermediate[token * config.intermediate + channel] = silu_half(gate) * up;
}

kernel void half_mlp_down_benchmark(device const half *intermediate [[buffer(0)]],
                                    device const half *downWeightTransposed [[buffer(1)]],
                                    device half *output [[buffer(2)]],
                                    constant MLPConfig &config [[buffer(3)]],
                                    uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.hidden) {
        return;
    }

    uint intermediateOffset = token * config.intermediate;
    uint weightOffset = channel * config.intermediate;
    half acc = half(0.0);

    for (uint i = 0; i < config.intermediate; ++i) {
        acc = fma(intermediate[intermediateOffset + i], downWeightTransposed[weightOffset + i], acc);
    }

    output[token * config.hidden + channel] = acc;
}

kernel void half_int8_mlp_down_benchmark(device const half *intermediate [[buffer(0)]],
                                         device const char *downWeightTransposed [[buffer(1)]],
                                         device half *output [[buffer(2)]],
                                         constant MLPConfig &config [[buffer(3)]],
                                         uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.hidden) {
        return;
    }

    uint intermediateOffset = token * config.intermediate;
    uint weightOffset = channel * config.intermediate;
    half acc = half(0.0);

    for (uint i = 0; i < config.intermediate; ++i) {
        acc = fma(intermediate[intermediateOffset + i], half(int(downWeightTransposed[weightOffset + i])), acc);
    }

    output[token * config.hidden + channel] = acc;
}

kernel void half_int4_mlp_down_benchmark(device const half *intermediate [[buffer(0)]],
                                         device const uchar *downWeightTransposedPacked [[buffer(1)]],
                                         device half *output [[buffer(2)]],
                                         constant MLPConfig &config [[buffer(3)]],
                                         uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.hidden) {
        return;
    }

    uint intermediateOffset = token * config.intermediate;
    uint weightByteOffset = channel * (config.intermediate / 2);
    half acc = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.intermediate / 2; ++packedOffset) {
        uchar downByte = downWeightTransposedPacked[weightByteOffset + packedOffset];
        uint k = packedOffset * 2;

        acc = fma(intermediate[intermediateOffset + k], signed_int4_to_half(downByte), acc);
        acc = fma(intermediate[intermediateOffset + k + 1], signed_int4_to_half(downByte >> 4), acc);
    }

    output[token * config.hidden + channel] = acc;
}

kernel void half_int2_mlp_down_benchmark(device const half *intermediate [[buffer(0)]],
                                         device const uchar *downWeightTransposedPacked [[buffer(1)]],
                                         device half *output [[buffer(2)]],
                                         constant MLPConfig &config [[buffer(3)]],
                                         uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.hidden) {
        return;
    }

    uint intermediateOffset = token * config.intermediate;
    uint weightByteOffset = channel * (config.intermediate / 4);
    half acc = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.intermediate / 4; ++packedOffset) {
        uchar downByte = downWeightTransposedPacked[weightByteOffset + packedOffset];
        uint k = packedOffset * 4;

        acc = fma(intermediate[intermediateOffset + k], signed_int2_to_half(downByte), acc);
        acc = fma(intermediate[intermediateOffset + k + 1], signed_int2_to_half(downByte >> 2), acc);
        acc = fma(intermediate[intermediateOffset + k + 2], signed_int2_to_half(downByte >> 4), acc);
        acc = fma(intermediate[intermediateOffset + k + 3], signed_int2_to_half(downByte >> 6), acc);
    }

    output[token * config.hidden + channel] = acc;
}

kernel void half_fp4_mlp_down_g8_benchmark(device const half *intermediate [[buffer(0)]],
                                           device const uchar *downWeightTransposedPacked [[buffer(1)]],
                                           device half *output [[buffer(2)]],
                                           constant MLPConfig &config [[buffer(3)]],
                                           uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.hidden) {
        return;
    }

    uint intermediateOffset = token * config.intermediate;
    uint weightByteOffset = channel * (config.intermediate / 2);
    half acc = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.intermediate / 2; packedOffset += 4) {
        packed_numeric_type<metal_fp4_e2m1_format, 8>::storage_type downStorage;
        for (ushort i = 0; i < 4; ++i) {
            downStorage[i] = downWeightTransposedPacked[weightByteOffset + packedOffset + i];
        }

        auto downPacked = packed_numeric_type<metal_fp4_e2m1_format, 8>(downStorage);
        vec<half, 8> downW = unpack<half>(downPacked);
        uint k = packedOffset * 2;
        for (ushort i = 0; i < 8; ++i) {
            acc = fma(intermediate[intermediateOffset + k + i], downW[i], acc);
        }
    }

    output[token * config.hidden + channel] = acc;
}

kernel void half_fp4_mlp_down_benchmark(device const half *intermediate [[buffer(0)]],
                                        device const uchar *downWeightTransposedPacked [[buffer(1)]],
                                        device half *output [[buffer(2)]],
                                        constant MLPConfig &config [[buffer(3)]],
                                        uint2 gid [[thread_position_in_grid]])
{
    uint channel = gid.x;
    uint token = gid.y;
    if (token >= config.tokens || channel >= config.hidden) {
        return;
    }

    uint intermediateOffset = token * config.intermediate;
    uint weightByteOffset = channel * (config.intermediate / 2);
    half acc = half(0.0);

    for (uint packedOffset = 0; packedOffset < config.intermediate / 2; packedOffset += 8) {
        packed_numeric_type<metal_fp4_e2m1_format, 16>::storage_type downStorage;
        for (ushort i = 0; i < 8; ++i) {
            downStorage[i] = downWeightTransposedPacked[weightByteOffset + packedOffset + i];
        }

        auto downPacked = packed_numeric_type<metal_fp4_e2m1_format, 16>(downStorage);
        vec<half, 16> downW = unpack<half>(downPacked);
        uint k = packedOffset * 2;
        for (ushort i = 0; i < 16; ++i) {
            acc = fma(intermediate[intermediateOffset + k + i], downW[i], acc);
        }
    }

    output[token * config.hidden + channel] = acc;
}
