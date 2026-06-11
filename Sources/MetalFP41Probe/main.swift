import Darwin
import Foundation
import Metal

enum ProbeError: Error, CustomStringConvertible {
    case invalidArgument(String)
    case missingDevice
    case missingShaderResource
    case missingFunction(String)
    case missingCommandQueue
    case missingCommandBuffer
    case missingComputeEncoder
    case missingBuffer(String)
    case gpuExecutionFailed(String)

    var description: String {
        switch self {
        case .invalidArgument(let message):
            return message
        case .missingDevice:
            return "No Metal device is available."
        case .missingShaderResource:
            return "Could not find PackedNumericKernels.metal in the package resources."
        case .missingFunction(let name):
            return "The Metal library does not contain kernel \(name)."
        case .missingCommandQueue:
            return "Could not create a Metal command queue."
        case .missingCommandBuffer:
            return "Could not create a Metal command buffer."
        case .missingComputeEncoder:
            return "Could not create a Metal compute encoder."
        case .missingBuffer(let label):
            return "Could not create Metal buffer \(label)."
        case .gpuExecutionFailed(let message):
            return message
        }
    }
}

struct PackedNumericCase {
    let name: String
    let kernelName: String
    let inputs: [Float]
    let expected: [Float]
    let packedByteCount: Int
}

struct PackedNumericResult {
    let testCase: PackedNumericCase
    let packedBytes: [UInt8]
    let outputs: [Float]
    let passed: Bool
}

struct MatmulConfig {
    let rows: UInt32
    let columns: UInt32
    let depth: UInt32
    let packedDepthBytes: UInt32
}

struct MLPConfig {
    let tokens: UInt32
    let hidden: UInt32
    let intermediate: UInt32
}

struct MatmulBenchmarkResult {
    let name: String
    let seconds: Double
    let iterations: Int
    let rows: Int
    let columns: Int
    let depth: Int
    let checksum: String
}

typealias BenchmarkResult = MatmulBenchmarkResult

let cliArguments = Array(CommandLine.arguments.dropFirst())
let arguments = Set(cliArguments)

do {
    try runPackedNumericProbe(skipTensorProbe: arguments.contains("--skip-tensor"))
} catch {
    fputs("error: \(error)\n", stderr)
    exit(1)
}

func runPackedNumericProbe(skipTensorProbe: Bool) throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
        throw ProbeError.missingDevice
    }

    print("Device: \(device.name)")
    print("Metal family 4: \(device.supportsFamily(.metal4) ? "yes" : "no")")
    print("MSL language version: 4.1")
    printTensorDataTypeSummary()

    let shaderSource = try loadShaderSource()
    let options = MTLCompileOptions()
    options.languageVersion = .version4_1
    options.mathMode = .safe

    let library = try device.makeLibrary(source: shaderSource, options: options)
    let queue = try makeQueue(device: device)

    let cases = [
        PackedNumericCase(
            name: "FP4 E2M1",
            kernelName: "fp4_e2m1_roundtrip",
            inputs: [-8.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
                     0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0],
            expected: [-6.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5,
                       0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            packedByteCount: 8
        ),
        PackedNumericCase(
            name: "FP8 E4M3",
            kernelName: "fp8_e4m3_roundtrip",
            inputs: [-512.0, -448.0, -16.0, -0.5, 0.0, 0.125, 16.0, 512.0],
            expected: [-448.0, -448.0, -16.0, -0.5, 0.0, 0.125, 16.0, 448.0],
            packedByteCount: 8
        ),
        PackedNumericCase(
            name: "FP8 E5M2",
            kernelName: "fp8_e5m2_roundtrip",
            inputs: [-57_344.0, -1024.0, -2.0, -0.5, 0.0, 0.25, 1024.0, 57_344.0],
            expected: [-57_344.0, -1024.0, -2.0, -0.5, 0.0, 0.25, 1024.0, 57_344.0],
            packedByteCount: 8
        )
    ]

    var allPassed = true
    for testCase in cases {
        let result = try run(testCase: testCase, device: device, library: library, queue: queue)
        allPassed = allPassed && result.passed
        printResult(result)
    }

    if skipTensorProbe {
        print("MTLTensor creation probe: skipped by --skip-tensor.")
    } else {
        try runTensorCreationProbe(device: device)
    }

    if arguments.contains("--benchmark") {
        try runBigMatmulBenchmark(device: device, library: library, queue: queue)
    }
    if arguments.contains("--mlp-benchmark") {
        try runMLPBenchmark(device: device, library: library, queue: queue)
    }
    if arguments.contains("--accum-probe") {
        try runAccumulatorProbe(device: device, queue: queue)
    }

    if allPassed {
        print("PASS")
    } else {
        throw ProbeError.gpuExecutionFailed("One or more packed numeric checks failed.")
    }
}

func loadShaderSource() throws -> String {
    try loadShaderSource(named: "PackedNumericKernels")
}

func loadShaderSource(named resourceName: String) throws -> String {
    guard let url = Bundle.module.url(
        forResource: resourceName,
        withExtension: "metal",
        subdirectory: "Shaders"
    ) else {
        throw ProbeError.missingShaderResource
    }
    return try String(contentsOf: url, encoding: .utf8)
}

func makeMPPMLPLibrary(device: MTLDevice) throws -> MTLLibrary {
    let source = try loadShaderSource(named: "MPPMLPKernels")
    let options = MTLCompileOptions()
    options.languageVersion = .version4_1
    options.mathMode = .safe
    return try device.makeLibrary(source: source, options: options)
}

func makeMXFP4Library(device: MTLDevice) throws -> MTLLibrary {
    let source = try loadShaderSource(named: "MXFP4Kernels")
    let options = MTLCompileOptions()
    options.languageVersion = .version4_1
    options.mathMode = .safe
    return try device.makeLibrary(source: source, options: options)
}

func makeQueue(device: MTLDevice) throws -> MTLCommandQueue {
    guard let queue = device.makeCommandQueue() else {
        throw ProbeError.missingCommandQueue
    }
    return queue
}

func run(
    testCase: PackedNumericCase,
    device: MTLDevice,
    library: MTLLibrary,
    queue: MTLCommandQueue
) throws -> PackedNumericResult {
    guard let function = library.makeFunction(name: testCase.kernelName) else {
        throw ProbeError.missingFunction(testCase.kernelName)
    }

    let pipeline = try device.makeComputePipelineState(function: function)
    let inputByteCount = testCase.inputs.count * MemoryLayout<Float>.stride
    let outputByteCount = testCase.expected.count * MemoryLayout<Float>.stride

    guard let inputBuffer = device.makeBuffer(
        bytes: testCase.inputs,
        length: inputByteCount,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("input")
    }
    guard let packedBuffer = device.makeBuffer(length: testCase.packedByteCount, options: .storageModeShared) else {
        throw ProbeError.missingBuffer("packed")
    }
    guard let outputBuffer = device.makeBuffer(length: outputByteCount, options: .storageModeShared) else {
        throw ProbeError.missingBuffer("output")
    }
    memset(outputBuffer.contents(), 0, outputByteCount)
    memset(packedBuffer.contents(), 0, testCase.packedByteCount)

    guard let commandBuffer = queue.makeCommandBuffer() else {
        throw ProbeError.missingCommandBuffer
    }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw ProbeError.missingComputeEncoder
    }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(inputBuffer, offset: 0, index: 0)
    encoder.setBuffer(packedBuffer, offset: 0, index: 1)
    encoder.setBuffer(outputBuffer, offset: 0, index: 2)
    encoder.dispatchThreads(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1)
    )
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    if let error = commandBuffer.error {
        throw ProbeError.gpuExecutionFailed("\(testCase.name) command buffer failed: \(error)")
    }
    guard commandBuffer.status == .completed else {
        throw ProbeError.gpuExecutionFailed("\(testCase.name) command buffer finished with status \(commandBuffer.status.rawValue).")
    }

    let outputs = readFloats(from: outputBuffer, count: testCase.expected.count)
    let packedBytes = readBytes(from: packedBuffer, count: testCase.packedByteCount)
    let passed = zip(outputs, testCase.expected).allSatisfy { nearlyEqual($0, $1) }
    return PackedNumericResult(testCase: testCase, packedBytes: packedBytes, outputs: outputs, passed: passed)
}

func readFloats(from buffer: MTLBuffer, count: Int) -> [Float] {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}

func readBytes(from buffer: MTLBuffer, count: Int) -> [UInt8] {
    let pointer = buffer.contents().bindMemory(to: UInt8.self, capacity: count)
    return Array(UnsafeBufferPointer(start: pointer, count: count))
}

func nearlyEqual(_ lhs: Float, _ rhs: Float) -> Bool {
    if lhs.isNaN || rhs.isNaN {
        return lhs.isNaN && rhs.isNaN
    }
    if lhs.isInfinite || rhs.isInfinite {
        return lhs == rhs
    }
    return abs(lhs - rhs) <= 0.000_01
}

func printResult(_ result: PackedNumericResult) {
    let status = result.passed ? "ok" : "FAIL"
    print("")
    print("[\(status)] \(result.testCase.name)")
    print("  input:    \(format(result.testCase.inputs))")
    print("  unpacked: \(format(result.outputs))")
    print("  expected: \(format(result.testCase.expected))")
    print("  bytes:    \(formatHex(result.packedBytes))")
}

func printTensorDataTypeSummary() {
    print("MTLTensorDataType raw values:")
    print("  float4e2m1=\(MTLTensorDataType.float4e2m1.rawValue)")
    print("  float8e4m3=\(MTLTensorDataType.float8e4m3.rawValue)")
    print("  float8e5m2=\(MTLTensorDataType.float8e5m2.rawValue)")
    print("  float8ue8m0=\(MTLTensorDataType.float8ue8m0.rawValue)")
}

func format(_ values: [Float]) -> String {
    values.map { value in
        if value.isInfinite {
            return value.sign == .minus ? "-inf" : "inf"
        }
        if value.rounded() == value {
            return String(format: "%.0f", value)
        }
        return String(format: "%.6g", value)
    }.joined(separator: ", ")
}

func formatHex(_ bytes: [UInt8]) -> String {
    bytes.map { String(format: "%02X", $0) }.joined(separator: " ")
}

func runTensorCreationProbe(device: MTLDevice) throws {
    let tensorTypes: [(String, MTLTensorDataType)] = [
        ("float4e2m1", .float4e2m1),
        ("float8e4m3", .float8e4m3),
        ("float8e5m2", .float8e5m2)
    ]

    print("")
    print("MTLTensor creation:")
    for (name, dataType) in tensorTypes {
        let dimensions = [NSInteger(32), NSInteger(2)]
        let extents = dimensions.withUnsafeBufferPointer { pointer in
            MTLTensorExtents(__rank: dimensions.count, values: pointer.baseAddress!)!
        }
        let descriptor = MTLTensorDescriptor()
        descriptor.dimensions = extents
        descriptor.dataType = dataType
        descriptor.usage = .compute

        let tensor = try device.makeTensor(descriptor: descriptor)
        print("  [ok] \(name): rank=\(tensor.dimensions.rank) dims=32x2 dataType=\(tensor.dataType.rawValue)")
    }

    let scaledTensorTypes: [(String, MTLTensorDataType)] = [
        ("float4e2m1 + e8m0 scales", .float4e2m1),
        ("float8e4m3 + e8m0 scales", .float8e4m3),
        ("float8e5m2 + e8m0 scales", .float8e5m2)
    ]

    print("")
    print("MTLTensor scale-plane creation:")
    for (name, dataType) in scaledTensorTypes {
        let dimensions = [NSInteger(64), NSInteger(2)]
        let blockFactors = [NSInteger(32), NSInteger(1)]
        let tensorExtents = dimensions.withUnsafeBufferPointer { pointer in
            MTLTensorExtents(__rank: dimensions.count, values: pointer.baseAddress!)!
        }
        let scaleBlockExtents = blockFactors.withUnsafeBufferPointer { pointer in
            MTLTensorExtents(__rank: blockFactors.count, values: pointer.baseAddress!)!
        }

        let scalePlane = MTLTensorAuxiliaryPlaneDescriptor()
        scalePlane.dataType = .float8ue8m0
        scalePlane.blockFactors = scaleBlockExtents

        let auxiliaryPlanes = MTLTensorAuxiliaryPlaneDescriptorMap()
        auxiliaryPlanes.setDescriptor(scalePlane, for: .scales)

        let descriptor = MTLTensorDescriptor()
        descriptor.dimensions = tensorExtents
        descriptor.dataType = dataType
        descriptor.usage = .compute
        descriptor.auxiliaryPlanes = auxiliaryPlanes

        let tensor = try device.makeTensor(descriptor: descriptor)
        let plane = tensor.auxiliaryPlanes.first { $0.planeType == .scales }
        let scaleType = plane?.dataType.rawValue ?? -1
        let scaleRank = plane?.blockFactors.rank ?? 0
        print("  [ok] \(name): dims=64x2 scaleType=\(scaleType) blockFactors=32x1 rank=\(scaleRank)")
    }
}

func runBigMatmulBenchmark(device: MTLDevice, library: MTLLibrary, queue: MTLCommandQueue) throws {
    let defaultSize = intArgument("--matmul-size", defaultValue: 2048)
    let rows = intArgument("--m", defaultValue: defaultSize)
    let columns = intArgument("--n", defaultValue: defaultSize)
    let depth = intArgument("--k", defaultValue: defaultSize)
    let iterations = intArgument("--iterations", defaultValue: 3)

    guard rows > 0, columns > 0, depth > 0, iterations > 0 else {
        throw ProbeError.invalidArgument("Matmul sizes and iterations must be positive.")
    }
    guard depth % 16 == 0 else {
        throw ProbeError.invalidArgument("Matmul K/depth must be a multiple of 16 for packed FP4 groups.")
    }

    let packedDepthBytes = depth / 2
    let aByteCount = rows * packedDepthBytes
    let bByteCount = columns * packedDepthBytes
    let outputElementCount = rows * columns

    print("")
    print("Big packed matmul benchmark:")
    print("  shape: A \(rows)x\(depth), B \(depth)x\(columns), C \(rows)x\(columns)")
    print("  packed A bytes: \(formatBytes(aByteCount)), packed B^T bytes: \(formatBytes(bByteCount))")
    print("  half A bytes: \(formatBytes(rows * depth * MemoryLayout<Float16>.stride))")
    print("  iterations: \(iterations)")
    if arguments.contains("--fp4-group-sweep") {
        print("  FP4 unpack group sweep: 8 and 16 elements")
    }

    let aBytes = makePackedMatrixBytes(count: aByteCount, seed: 0x1234_5678)
    let bTransposedBytes = makePackedMatrixBytes(count: bByteCount, seed: 0x8765_4321)
    let aHalfValues = makeHalfMatrixValues(count: rows * depth, seed: 0x2468_ace0)
    let aInt8Values = makeInt8MatrixValues(count: rows * depth, seed: 0x1357_9bdf)
    let bHalfValues = makeHalfMatrixValues(count: columns * depth, seed: 0xfdb9_7531, scale: 0.03125)

    guard let aBuffer = device.makeBuffer(bytes: aBytes, length: aBytes.count, options: .storageModeShared) else {
        throw ProbeError.missingBuffer("matmul A")
    }
    guard let aHalfBuffer = device.makeBuffer(
        bytes: aHalfValues,
        length: aHalfValues.count * MemoryLayout<Float16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("matmul half A")
    }
    guard let aInt8Buffer = device.makeBuffer(bytes: aInt8Values, length: aInt8Values.count, options: .storageModeShared) else {
        throw ProbeError.missingBuffer("matmul int8 A")
    }
    guard let bHalfBuffer = device.makeBuffer(
        bytes: bHalfValues,
        length: bHalfValues.count * MemoryLayout<Float16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("matmul half B^T")
    }
    guard let bBuffer = device.makeBuffer(bytes: bTransposedBytes, length: bTransposedBytes.count, options: .storageModeShared) else {
        throw ProbeError.missingBuffer("matmul B^T")
    }
    guard let floatOutputBuffer = device.makeBuffer(
        length: outputElementCount * MemoryLayout<Float>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("matmul FP4 output")
    }
    guard let halfOutputBuffer = device.makeBuffer(
        length: outputElementCount * MemoryLayout<UInt16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("matmul FP4 half output")
    }
    guard let intOutputBuffer = device.makeBuffer(
        length: outputElementCount * MemoryLayout<Int32>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("matmul INT4 output")
    }

    let config = MatmulConfig(
        rows: UInt32(rows),
        columns: UInt32(columns),
        depth: UInt32(depth),
        packedDepthBytes: UInt32(packedDepthBytes)
    )

    let fp4Result = try runMatmulBenchmarkKernel(
        name: "FP4 E2M1 g16 -> FP32 accumulate",
        kernelName: "fp4_e2m1_matmul_benchmark",
        device: device,
        library: library,
        queue: queue,
        aBuffer: aBuffer,
        bBuffer: bBuffer,
        outputBuffer: floatOutputBuffer,
        outputElementCount: outputElementCount,
        outputByteCount: outputElementCount * MemoryLayout<Float>.stride,
        config: config,
        iterations: iterations,
        checksum: { sampleFloatChecksum(buffer: floatOutputBuffer, count: outputElementCount) }
    )

    var fp4GroupSweepResults: [BenchmarkResult] = []
    if arguments.contains("--fp4-group-sweep") {
        fp4GroupSweepResults.append(try runMatmulBenchmarkKernel(
            name: "FP4 E2M1 g8 -> FP32 accumulate",
            kernelName: "fp4_e2m1_matmul_g8_benchmark",
            device: device,
            library: library,
            queue: queue,
            aBuffer: aBuffer,
            bBuffer: bBuffer,
            outputBuffer: floatOutputBuffer,
            outputElementCount: outputElementCount,
            outputByteCount: outputElementCount * MemoryLayout<Float>.stride,
            config: config,
            iterations: iterations,
            checksum: { sampleFloatChecksum(buffer: floatOutputBuffer, count: outputElementCount) }
        ))
    }

    let halfFP4Result = try runMatmulBenchmarkKernel(
        name: "half x FP4 E2M1 g16 -> half accumulate",
        kernelName: "half_fp4_matmul_half_benchmark",
        device: device,
        library: library,
        queue: queue,
        aBuffer: aHalfBuffer,
        bBuffer: bBuffer,
        outputBuffer: halfOutputBuffer,
        outputElementCount: outputElementCount,
        outputByteCount: outputElementCount * MemoryLayout<UInt16>.stride,
        config: config,
        iterations: iterations,
        checksum: { sampleHalfRawChecksum(buffer: halfOutputBuffer, count: outputElementCount) }
    )

    if arguments.contains("--fp4-group-sweep") {
        fp4GroupSweepResults.append(try runMatmulBenchmarkKernel(
            name: "half x FP4 E2M1 g8 -> half accumulate",
            kernelName: "half_fp4_matmul_half_g8_benchmark",
            device: device,
            library: library,
            queue: queue,
            aBuffer: aHalfBuffer,
            bBuffer: bBuffer,
            outputBuffer: halfOutputBuffer,
            outputElementCount: outputElementCount,
            outputByteCount: outputElementCount * MemoryLayout<UInt16>.stride,
            config: config,
            iterations: iterations,
            checksum: { sampleHalfRawChecksum(buffer: halfOutputBuffer, count: outputElementCount) }
        ))
    }

    let halfInt4Result = try runMatmulBenchmarkKernel(
        name: "half x INT4 -> half accumulate",
        kernelName: "half_int4_matmul_half_benchmark",
        device: device,
        library: library,
        queue: queue,
        aBuffer: aHalfBuffer,
        bBuffer: bBuffer,
        outputBuffer: halfOutputBuffer,
        outputElementCount: outputElementCount,
        outputByteCount: outputElementCount * MemoryLayout<UInt16>.stride,
        config: config,
        iterations: iterations,
        checksum: { sampleHalfRawChecksum(buffer: halfOutputBuffer, count: outputElementCount) }
    )

    let int8HalfResult = try runMatmulBenchmarkKernel(
        name: "INT8 x half -> half accumulate",
        kernelName: "int8_half_matmul_half_benchmark",
        device: device,
        library: library,
        queue: queue,
        aBuffer: aInt8Buffer,
        bBuffer: bHalfBuffer,
        outputBuffer: halfOutputBuffer,
        outputElementCount: outputElementCount,
        outputByteCount: outputElementCount * MemoryLayout<UInt16>.stride,
        config: config,
        iterations: iterations,
        checksum: { sampleHalfRawChecksum(buffer: halfOutputBuffer, count: outputElementCount) }
    )

    let int4Result = try runMatmulBenchmarkKernel(
        name: "INT4 x INT4 -> INT32 accumulate",
        kernelName: "int4_int8_matmul_benchmark",
        device: device,
        library: library,
        queue: queue,
        aBuffer: aBuffer,
        bBuffer: bBuffer,
        outputBuffer: intOutputBuffer,
        outputElementCount: outputElementCount,
        outputByteCount: outputElementCount * MemoryLayout<Int32>.stride,
        config: config,
        iterations: iterations,
        checksum: { sampleIntChecksum(buffer: intOutputBuffer, count: outputElementCount) }
    )

    let results = [fp4Result] + fp4GroupSweepResults + [halfFP4Result, halfInt4Result, int8HalfResult, int4Result]
    results.forEach(printMatmulBenchmarkResult)
    printRelativeSlowdowns(results)
}

func runMLPBenchmark(device: MTLDevice, library: MTLLibrary, queue: MTLCommandQueue) throws {
    let tokens = intArgument("--mlp-tokens", defaultValue: 128)
    let hidden = intArgument("--mlp-hidden", defaultValue: 1024)
    let intermediate = intArgument("--mlp-intermediate", defaultValue: hidden * 4)
    let iterations = intArgument("--mlp-iterations", defaultValue: 3)

    guard tokens > 0, hidden > 0, intermediate > 0, iterations > 0 else {
        throw ProbeError.invalidArgument("MLP sizes and iterations must be positive.")
    }
    guard hidden % 16 == 0, intermediate % 16 == 0 else {
        throw ProbeError.invalidArgument("MLP hidden and intermediate sizes must be multiples of 16 for packed FP4 weights.")
    }

    let inputElementCount = tokens * hidden
    let intermediateElementCount = tokens * intermediate
    let outputElementCount = tokens * hidden
    let gateWeightElementCount = intermediate * hidden
    let downWeightElementCount = hidden * intermediate

    print("")
    print("Gated MLP benchmark:")
    print("  op: down(silu(gate(x)) * up(x))")
    print("  shape: tokens=\(tokens), hidden=\(hidden), intermediate=\(intermediate)")
    print("  iterations: \(iterations)")

    let xHalf = makeHalfMatrixValues(count: inputElementCount, seed: 0x1111_2222, scale: 0.125)
    let xInt8 = makeInt8MatrixValues(count: inputElementCount, seed: 0x2222_3333)
    let gateWeight = makeHalfMatrixValues(count: gateWeightElementCount, seed: 0x5555_6666, scale: 0.015625)
    let upWeight = makeHalfMatrixValues(count: gateWeightElementCount, seed: 0x7777_8888, scale: 0.015625)
    let downWeight = makeHalfMatrixValues(count: downWeightElementCount, seed: 0x9999_aaaa, scale: 0.00390625)
    let gateWeightInt8 = makeInt8MatrixValues(count: gateWeightElementCount, seed: 0x3333_4444)
    let upWeightInt8 = makeInt8MatrixValues(count: gateWeightElementCount, seed: 0x4444_5555)
    let downWeightInt8 = makeInt8MatrixValues(count: downWeightElementCount, seed: 0x5555_7777)
    let gateWeightInt4Bytes = makePackedMatrixBytes(count: gateWeightElementCount / 2, seed: 0x1357_2468)
    let upWeightInt4Bytes = makePackedMatrixBytes(count: gateWeightElementCount / 2, seed: 0xace0_1357)
    let downWeightInt4Bytes = makePackedMatrixBytes(count: downWeightElementCount / 2, seed: 0xc001_d00d)
    let gateWeightInt2Bytes = makePackedMatrixBytes(count: gateWeightElementCount / 4, seed: 0x2468_1357)
    let upWeightInt2Bytes = makePackedMatrixBytes(count: gateWeightElementCount / 4, seed: 0xbad0_cafe)
    let downWeightInt2Bytes = makePackedMatrixBytes(count: downWeightElementCount / 4, seed: 0xd00d_c001)
    let gateWeightFP4Bytes = makePackedMatrixBytes(count: gateWeightElementCount / 2, seed: 0xabcd_1357)
    let upWeightFP4Bytes = makePackedMatrixBytes(count: gateWeightElementCount / 2, seed: 0x2468_bdf0)
    let downWeightFP4Bytes = makePackedMatrixBytes(count: downWeightElementCount / 2, seed: 0x0bad_cafe)
    let gateWeightFP8E4M3Bytes = makeFiniteFP8E4M3Bytes(count: gateWeightElementCount, seed: 0x1111_7777)
    let upWeightFP8E4M3Bytes = makeFiniteFP8E4M3Bytes(count: gateWeightElementCount, seed: 0x2222_8888)
    let downWeightFP8E4M3Bytes = makeFiniteFP8E4M3Bytes(count: downWeightElementCount, seed: 0x3333_9999)
    let gateWeightFP8E5M2Bytes = makeFiniteFP8E5M2Bytes(count: gateWeightElementCount, seed: 0x4444_aaaa)
    let upWeightFP8E5M2Bytes = makeFiniteFP8E5M2Bytes(count: gateWeightElementCount, seed: 0x5555_bbbb)
    let downWeightFP8E5M2Bytes = makeFiniteFP8E5M2Bytes(count: downWeightElementCount, seed: 0x6666_cccc)
    // MXFP4 E8M0 block scales (one byte per 32 weights along K), 1/8x..8x.
    let mxfp4ScalePattern: [UInt8] = [124, 125, 126, 127, 127, 128, 128, 129, 130]
    let gateScaleE8M0Bytes = makeBytePatternValues(count: intermediate * (hidden / 32), seed: 0x7777_dddd, pattern: mxfp4ScalePattern)
    let upScaleE8M0Bytes = makeBytePatternValues(count: intermediate * (hidden / 32), seed: 0x8888_eeee, pattern: mxfp4ScalePattern)
    let downScaleE8M0Bytes = makeBytePatternValues(count: hidden * (intermediate / 32), seed: 0x9999_ffff, pattern: mxfp4ScalePattern)

    guard let xHalfBuffer = device.makeBuffer(
        bytes: xHalf,
        length: xHalf.count * MemoryLayout<Float16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP half input")
    }
    guard let xInt8Buffer = device.makeBuffer(
        bytes: xInt8,
        length: xInt8.count * MemoryLayout<Int8>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP INT8 input")
    }
    guard let gateWeightBuffer = device.makeBuffer(
        bytes: gateWeight,
        length: gateWeight.count * MemoryLayout<Float16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP gate weight")
    }
    guard let upWeightBuffer = device.makeBuffer(
        bytes: upWeight,
        length: upWeight.count * MemoryLayout<Float16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP up weight")
    }
    guard let downWeightBuffer = device.makeBuffer(
        bytes: downWeight,
        length: downWeight.count * MemoryLayout<Float16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP down weight")
    }
    guard let gateWeightInt8Buffer = device.makeBuffer(
        bytes: gateWeightInt8,
        length: gateWeightInt8.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP INT8 gate weight")
    }
    guard let upWeightInt8Buffer = device.makeBuffer(
        bytes: upWeightInt8,
        length: upWeightInt8.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP INT8 up weight")
    }
    guard let downWeightInt8Buffer = device.makeBuffer(
        bytes: downWeightInt8,
        length: downWeightInt8.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP INT8 down weight")
    }
    guard let gateWeightInt4Buffer = device.makeBuffer(
        bytes: gateWeightInt4Bytes,
        length: gateWeightInt4Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP INT4 gate weight")
    }
    guard let upWeightInt4Buffer = device.makeBuffer(
        bytes: upWeightInt4Bytes,
        length: upWeightInt4Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP INT4 up weight")
    }
    guard let downWeightInt4Buffer = device.makeBuffer(
        bytes: downWeightInt4Bytes,
        length: downWeightInt4Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP INT4 down weight")
    }
    guard let gateWeightInt2Buffer = device.makeBuffer(
        bytes: gateWeightInt2Bytes,
        length: gateWeightInt2Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP INT2 gate weight")
    }
    guard let upWeightInt2Buffer = device.makeBuffer(
        bytes: upWeightInt2Bytes,
        length: upWeightInt2Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP INT2 up weight")
    }
    guard let downWeightInt2Buffer = device.makeBuffer(
        bytes: downWeightInt2Bytes,
        length: downWeightInt2Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP INT2 down weight")
    }
    guard let gateWeightFP4Buffer = device.makeBuffer(
        bytes: gateWeightFP4Bytes,
        length: gateWeightFP4Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP FP4 gate weight")
    }
    guard let upWeightFP4Buffer = device.makeBuffer(
        bytes: upWeightFP4Bytes,
        length: upWeightFP4Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP FP4 up weight")
    }
    guard let downWeightFP4Buffer = device.makeBuffer(
        bytes: downWeightFP4Bytes,
        length: downWeightFP4Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP FP4 down weight")
    }
    guard let gateWeightFP8E4M3Buffer = device.makeBuffer(
        bytes: gateWeightFP8E4M3Bytes,
        length: gateWeightFP8E4M3Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP FP8 E4M3 gate weight")
    }
    guard let upWeightFP8E4M3Buffer = device.makeBuffer(
        bytes: upWeightFP8E4M3Bytes,
        length: upWeightFP8E4M3Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP FP8 E4M3 up weight")
    }
    guard let downWeightFP8E4M3Buffer = device.makeBuffer(
        bytes: downWeightFP8E4M3Bytes,
        length: downWeightFP8E4M3Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP FP8 E4M3 down weight")
    }
    guard let gateWeightFP8E5M2Buffer = device.makeBuffer(
        bytes: gateWeightFP8E5M2Bytes,
        length: gateWeightFP8E5M2Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP FP8 E5M2 gate weight")
    }
    guard let upWeightFP8E5M2Buffer = device.makeBuffer(
        bytes: upWeightFP8E5M2Bytes,
        length: upWeightFP8E5M2Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP FP8 E5M2 up weight")
    }
    guard let downWeightFP8E5M2Buffer = device.makeBuffer(
        bytes: downWeightFP8E5M2Bytes,
        length: downWeightFP8E5M2Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP FP8 E5M2 down weight")
    }
    guard let gateScaleBuffer = device.makeBuffer(
        bytes: gateScaleE8M0Bytes,
        length: gateScaleE8M0Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MXFP4 gate scales")
    }
    guard let upScaleBuffer = device.makeBuffer(
        bytes: upScaleE8M0Bytes,
        length: upScaleE8M0Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MXFP4 up scales")
    }
    guard let downScaleBuffer = device.makeBuffer(
        bytes: downScaleE8M0Bytes,
        length: downScaleE8M0Bytes.count,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MXFP4 down scales")
    }
    guard let gateWeightDequantBuffer = device.makeBuffer(
        length: gateWeightElementCount * MemoryLayout<Float16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MXFP4 dequantized gate weight")
    }
    guard let upWeightDequantBuffer = device.makeBuffer(
        length: gateWeightElementCount * MemoryLayout<Float16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MXFP4 dequantized up weight")
    }
    guard let downWeightDequantBuffer = device.makeBuffer(
        length: downWeightElementCount * MemoryLayout<Float16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MXFP4 dequantized down weight")
    }
    guard let intermediateBuffer = device.makeBuffer(
        length: intermediateElementCount * MemoryLayout<UInt16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP intermediate")
    }
    guard let outputBuffer = device.makeBuffer(
        length: outputElementCount * MemoryLayout<UInt16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP output")
    }
    guard let gateFloatBuffer = device.makeBuffer(
        length: intermediateElementCount * MemoryLayout<Float>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP MPP gate output")
    }
    guard let upFloatBuffer = device.makeBuffer(
        length: intermediateElementCount * MemoryLayout<Float>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP MPP up output")
    }
    guard let midFloatBuffer = device.makeBuffer(
        length: intermediateElementCount * MemoryLayout<Float>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP MPP intermediate")
    }
    guard let gateHalfBuffer = device.makeBuffer(
        length: intermediateElementCount * MemoryLayout<UInt16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP MPP half gate output")
    }
    guard let upHalfBuffer = device.makeBuffer(
        length: intermediateElementCount * MemoryLayout<UInt16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP MPP half up output")
    }
    guard let midHalfBuffer = device.makeBuffer(
        length: intermediateElementCount * MemoryLayout<UInt16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP MPP half intermediate")
    }
    guard let outputFloatBuffer = device.makeBuffer(
        length: outputElementCount * MemoryLayout<Float>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP MPP output")
    }
    guard let gateIntBuffer = device.makeBuffer(
        length: intermediateElementCount * MemoryLayout<Int32>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP MPP INT gate output")
    }
    guard let upIntBuffer = device.makeBuffer(
        length: intermediateElementCount * MemoryLayout<Int32>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP MPP INT up output")
    }
    guard let midInt8Buffer = device.makeBuffer(
        length: intermediateElementCount * MemoryLayout<Int8>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP MPP INT8 intermediate")
    }
    guard let outputIntBuffer = device.makeBuffer(
        length: outputElementCount * MemoryLayout<Int32>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("MLP MPP INT output")
    }

    let config = MLPConfig(tokens: UInt32(tokens), hidden: UInt32(hidden), intermediate: UInt32(intermediate))
    let mppIntermediateFloatByteCount = intermediateElementCount * MemoryLayout<Float>.stride
    let mppIntermediateHalfByteCount = intermediateElementCount * MemoryLayout<UInt16>.stride
    let mppIntermediateIntByteCount = intermediateElementCount * MemoryLayout<Int32>.stride
    let mppIntermediateInt8ByteCount = intermediateElementCount * MemoryLayout<Int8>.stride
    let mppOutputFloatByteCount = outputElementCount * MemoryLayout<Float>.stride
    let mppOutputIntByteCount = outputElementCount * MemoryLayout<Int32>.stride

    let halfResult = try runMLPBenchmarkVariant(
        name: "half MLP: half x half -> half",
        gateKernelName: "half_mlp_gate_up_benchmark",
        downKernelName: "half_mlp_down_benchmark",
        device: device,
        library: library,
        queue: queue,
        inputBuffer: xHalfBuffer,
        gateWeightBuffer: gateWeightBuffer,
        upWeightBuffer: upWeightBuffer,
        intermediateBuffer: intermediateBuffer,
        downWeightBuffer: downWeightBuffer,
        outputBuffer: outputBuffer,
        config: config,
        iterations: iterations,
        checksum: { sampleHalfRawChecksum(buffer: outputBuffer, count: outputElementCount) }
    )

    var mppResults: [BenchmarkResult] = []
    if tokens % 64 == 0 {
        let mppLibrary = try makeMPPMLPLibrary(device: device)
        let mxfp4Library = try makeMXFP4Library(device: device)
        let hasMPPTensorOpsInt2 = mppLibrary.makeFunction(name: "mpp_mlp_h_i2_f_n128") != nil
            && mppLibrary.makeFunction(name: "mpp_mlp_i8_i2_i32_n128") != nil
        mppResults.append(try runMPPMLPBenchmarkVariant(
            name: "MPP/NAX MLP half weights: h_h_f + f_h_f",
            gateKernelName: "mpp_mlp_h_h_f_n128",
            downKernelName: "mpp_mlp_f_h_f_n128",
            device: device,
            library: mppLibrary,
            queue: queue,
            inputBuffer: xHalfBuffer,
            gateWeightBuffer: gateWeightBuffer,
            upWeightBuffer: upWeightBuffer,
            gateBuffer: gateFloatBuffer,
            upBuffer: upFloatBuffer,
            midBuffer: midFloatBuffer,
            downWeightBuffer: downWeightBuffer,
            outputBuffer: outputFloatBuffer,
            gateByteCount: mppIntermediateFloatByteCount,
            midByteCount: mppIntermediateFloatByteCount,
            outputByteCount: mppOutputFloatByteCount,
            swigluKernelName: "mpp_mlp_swiglu_float",
            config: config,
            iterations: iterations,
            checksum: { sampleFloatChecksum(buffer: outputFloatBuffer, count: outputElementCount) }
        ))
        mppResults.append(try runMPPMLPBenchmarkVariant(
            name: "MPP/NAX MLP INT8 weights: h_i8_f + f_i8_f",
            gateKernelName: "mpp_mlp_h_i8_f_n128",
            downKernelName: "mpp_mlp_f_i8_f_n128",
            device: device,
            library: mppLibrary,
            queue: queue,
            inputBuffer: xHalfBuffer,
            gateWeightBuffer: gateWeightInt8Buffer,
            upWeightBuffer: upWeightInt8Buffer,
            gateBuffer: gateFloatBuffer,
            upBuffer: upFloatBuffer,
            midBuffer: midFloatBuffer,
            downWeightBuffer: downWeightInt8Buffer,
            outputBuffer: outputFloatBuffer,
            gateByteCount: mppIntermediateFloatByteCount,
            midByteCount: mppIntermediateFloatByteCount,
            outputByteCount: mppOutputFloatByteCount,
            swigluKernelName: "mpp_mlp_swiglu_float",
            config: config,
            iterations: iterations,
            checksum: { sampleFloatChecksum(buffer: outputFloatBuffer, count: outputElementCount) }
        ))
        mppResults.append(try runMPPMLPBenchmarkVariant(
            name: "MPP/NAX MLP INT4 weights: h_i4_f + h_i4_f",
            gateKernelName: "mpp_mlp_h_i4_f_n128",
            downKernelName: "mpp_mlp_h_i4_f_n128",
            device: device,
            library: mppLibrary,
            queue: queue,
            inputBuffer: xHalfBuffer,
            gateWeightBuffer: gateWeightInt4Buffer,
            upWeightBuffer: upWeightInt4Buffer,
            gateBuffer: gateFloatBuffer,
            upBuffer: upFloatBuffer,
            midBuffer: midHalfBuffer,
            downWeightBuffer: downWeightInt4Buffer,
            outputBuffer: outputFloatBuffer,
            gateByteCount: mppIntermediateFloatByteCount,
            midByteCount: mppIntermediateHalfByteCount,
            outputByteCount: mppOutputFloatByteCount,
            swigluKernelName: "mpp_mlp_swiglu_half",
            config: config,
            iterations: iterations,
            checksum: { sampleFloatChecksum(buffer: outputFloatBuffer, count: outputElementCount) }
        ))
        mppResults.append(try runMPPMLPBenchmarkVariant(
            name: "MPP/NAX MLP FP4 E2M1 weights: h_fp4_f + h_fp4_f",
            gateKernelName: "mpp_mlp_h_fp4_f_n128",
            downKernelName: "mpp_mlp_h_fp4_f_n128",
            device: device,
            library: mppLibrary,
            queue: queue,
            inputBuffer: xHalfBuffer,
            gateWeightBuffer: gateWeightFP4Buffer,
            upWeightBuffer: upWeightFP4Buffer,
            gateBuffer: gateFloatBuffer,
            upBuffer: upFloatBuffer,
            midBuffer: midHalfBuffer,
            downWeightBuffer: downWeightFP4Buffer,
            outputBuffer: outputFloatBuffer,
            gateByteCount: mppIntermediateFloatByteCount,
            midByteCount: mppIntermediateHalfByteCount,
            outputByteCount: mppOutputFloatByteCount,
            swigluKernelName: "mpp_mlp_swiglu_half",
            config: config,
            iterations: iterations,
            checksum: { sampleFloatChecksum(buffer: outputFloatBuffer, count: outputElementCount) }
        ))
        if hidden % 32 == 0, intermediate % 32 == 0 {
            for (tileName, tileWidth) in [("n128", 128), ("n64", 64)] {
                mppResults.append(try runMXFP4FusedMLPBenchmarkVariant(
                    name: "MXFP4 fused MPP 4.1 (native FP4 + E8M0 in-register, \(tileName)): h_mxfp4_f",
                    matmulKernelName: "mxfp4_fused_h_f_\(tileName)",
                    nTile: tileWidth,
                    device: device,
                    mxfp4Library: mxfp4Library,
                    mppLibrary: mppLibrary,
                    queue: queue,
                    inputBuffer: xHalfBuffer,
                    gateWeightBuffer: gateWeightFP4Buffer,
                    upWeightBuffer: upWeightFP4Buffer,
                    downWeightBuffer: downWeightFP4Buffer,
                    gateScaleBuffer: gateScaleBuffer,
                    upScaleBuffer: upScaleBuffer,
                    downScaleBuffer: downScaleBuffer,
                    gateBuffer: gateFloatBuffer,
                    upBuffer: upFloatBuffer,
                    midBuffer: midHalfBuffer,
                    outputBuffer: outputFloatBuffer,
                    gateByteCount: mppIntermediateFloatByteCount,
                    midByteCount: mppIntermediateHalfByteCount,
                    outputByteCount: mppOutputFloatByteCount,
                    config: config,
                    iterations: iterations,
                    checksum: { sampleFloatChecksum(buffer: outputFloatBuffer, count: outputElementCount) }
                ))
            }
            for (tileName, tileWidth) in [("n128", 128), ("n64", 64)] {
                mppResults.append(try runMXFP4FusedMLPBenchmarkVariant(
                    name: "MXFP4 native scale-plane MPP 4.1 (FP4 + E8M0 consumed by matmul2d, \(tileName)): h_mxfp4sp_f",
                    matmulKernelName: "mxfp4_native_sp_h_f_\(tileName)",
                    nTile: tileWidth,
                    device: device,
                    mxfp4Library: mxfp4Library,
                    mppLibrary: mppLibrary,
                    queue: queue,
                    inputBuffer: xHalfBuffer,
                    gateWeightBuffer: gateWeightFP4Buffer,
                    upWeightBuffer: upWeightFP4Buffer,
                    downWeightBuffer: downWeightFP4Buffer,
                    gateScaleBuffer: gateScaleBuffer,
                    upScaleBuffer: upScaleBuffer,
                    downScaleBuffer: downScaleBuffer,
                    gateBuffer: gateFloatBuffer,
                    upBuffer: upFloatBuffer,
                    midBuffer: midHalfBuffer,
                    outputBuffer: outputFloatBuffer,
                    gateByteCount: mppIntermediateFloatByteCount,
                    midByteCount: mppIntermediateHalfByteCount,
                    outputByteCount: mppOutputFloatByteCount,
                    config: config,
                    iterations: iterations,
                    checksum: { sampleFloatChecksum(buffer: outputFloatBuffer, count: outputElementCount) }
                ))
            }
            let gateWeightTensor = try makeMXFP4ScalePlaneTensor(
                device: device,
                dataBuffer: gateWeightFP4Buffer,
                scaleBuffer: gateScaleBuffer,
                k: hidden,
                n: intermediate
            )
            let upWeightTensor = try makeMXFP4ScalePlaneTensor(
                device: device,
                dataBuffer: upWeightFP4Buffer,
                scaleBuffer: upScaleBuffer,
                k: hidden,
                n: intermediate
            )
            let downWeightTensor = try makeMXFP4ScalePlaneTensor(
                device: device,
                dataBuffer: downWeightFP4Buffer,
                scaleBuffer: downScaleBuffer,
                k: intermediate,
                n: hidden
            )
            mppResults.append(try runMXFP4HandleMLPBenchmarkVariant(
                name: "MXFP4 host MTLTensor + scale plane (bindless handle, n64): h_mxfp4sph_f",
                matmulKernelName: "mxfp4_native_sph_h_f_n64",
                nTile: 64,
                device: device,
                mxfp4Library: mxfp4Library,
                mppLibrary: mppLibrary,
                queue: queue,
                inputBuffer: xHalfBuffer,
                gateWeightTensor: gateWeightTensor,
                upWeightTensor: upWeightTensor,
                downWeightTensor: downWeightTensor,
                gateBuffer: gateFloatBuffer,
                upBuffer: upFloatBuffer,
                midBuffer: midHalfBuffer,
                outputBuffer: outputFloatBuffer,
                gateByteCount: mppIntermediateFloatByteCount,
                midByteCount: mppIntermediateHalfByteCount,
                outputByteCount: mppOutputFloatByteCount,
                config: config,
                iterations: iterations,
                checksum: { sampleFloatChecksum(buffer: outputFloatBuffer, count: outputElementCount) }
            ))
            for (label, kernel) in [
                ("MXFP4 fused coop-input (LUT decode+scale into registers, MPP 4.0-style, k64): h_mxfp4ci_f", "mxfp4_fused_ci_h_f_sg_k64"),
                ("MXFP4 fused coop-input (LUT decode+scale into registers, MPP 4.0-style, k32): h_mxfp4ci_f", "mxfp4_fused_ci_h_f_sg_k32")
            ] {
                mppResults.append(try runMXFP4FusedMLPBenchmarkVariant(
                    name: label,
                    matmulKernelName: kernel,
                    nTile: 32,
                    mTile: 32,
                    simdgroupsPerThreadgroup: 1,
                    device: device,
                    mxfp4Library: mxfp4Library,
                    mppLibrary: mppLibrary,
                    queue: queue,
                    inputBuffer: xHalfBuffer,
                    gateWeightBuffer: gateWeightFP4Buffer,
                    upWeightBuffer: upWeightFP4Buffer,
                    downWeightBuffer: downWeightFP4Buffer,
                    gateScaleBuffer: gateScaleBuffer,
                    upScaleBuffer: upScaleBuffer,
                    downScaleBuffer: downScaleBuffer,
                    gateBuffer: gateFloatBuffer,
                    upBuffer: upFloatBuffer,
                    midBuffer: midHalfBuffer,
                    outputBuffer: outputFloatBuffer,
                    gateByteCount: mppIntermediateFloatByteCount,
                    midByteCount: mppIntermediateHalfByteCount,
                    outputByteCount: mppOutputFloatByteCount,
                    config: config,
                    iterations: iterations,
                    checksum: { sampleFloatChecksum(buffer: outputFloatBuffer, count: outputElementCount) }
                ))
            }
            if arguments.contains("--mxfp4-decompose") {
                for (label, kernel, probeTile) in [
                    ("MXFP4 probe coop-store only (PERF ONLY, wrong math, n64)", "mxfp4_probe_coopstore_h_f_n64", 64),
                    ("MXFP4 probe k-block split, no scaling (PERF ONLY, wrong math, n64)", "mxfp4_probe_noscale_h_f_n64", 64),
                    ("MXFP4 probe transposed-B FP4, no scale plane (PERF ONLY, unscaled, n128)", "mxfp4_native_nosp_h_f_n128", 128)
                ] {
                    mppResults.append(try runMXFP4FusedMLPBenchmarkVariant(
                        name: label,
                        matmulKernelName: kernel,
                        nTile: probeTile,
                        device: device,
                        mxfp4Library: mxfp4Library,
                        mppLibrary: mppLibrary,
                        queue: queue,
                        inputBuffer: xHalfBuffer,
                        gateWeightBuffer: gateWeightFP4Buffer,
                        upWeightBuffer: upWeightFP4Buffer,
                        downWeightBuffer: downWeightFP4Buffer,
                        gateScaleBuffer: gateScaleBuffer,
                        upScaleBuffer: upScaleBuffer,
                        downScaleBuffer: downScaleBuffer,
                        gateBuffer: gateFloatBuffer,
                        upBuffer: upFloatBuffer,
                        midBuffer: midHalfBuffer,
                        outputBuffer: outputFloatBuffer,
                        gateByteCount: mppIntermediateFloatByteCount,
                        midByteCount: mppIntermediateHalfByteCount,
                        outputByteCount: mppOutputFloatByteCount,
                        config: config,
                        iterations: iterations,
                        checksum: { sampleFloatChecksum(buffer: outputFloatBuffer, count: outputElementCount) }
                    ))
                }
            }
            for (label, dequantKernel) in [
                ("MXFP4 dequant MPP 4.1 (native FP4 unpack -> half) + h_h_f", "mxfp4_dequant_native_half"),
                ("MXFP4 dequant MPP 4.0 (LUT, no FP4 hardware -> half) + h_h_f", "mxfp4_dequant_lut_half")
            ] {
                mppResults.append(try runMXFP4DequantMLPBenchmarkVariant(
                    name: label,
                    dequantKernelName: dequantKernel,
                    device: device,
                    mxfp4Library: mxfp4Library,
                    mppLibrary: mppLibrary,
                    queue: queue,
                    inputBuffer: xHalfBuffer,
                    gateWeightBuffer: gateWeightFP4Buffer,
                    upWeightBuffer: upWeightFP4Buffer,
                    downWeightBuffer: downWeightFP4Buffer,
                    gateScaleBuffer: gateScaleBuffer,
                    upScaleBuffer: upScaleBuffer,
                    downScaleBuffer: downScaleBuffer,
                    gateWeightDequantBuffer: gateWeightDequantBuffer,
                    upWeightDequantBuffer: upWeightDequantBuffer,
                    downWeightDequantBuffer: downWeightDequantBuffer,
                    gateBuffer: gateFloatBuffer,
                    upBuffer: upFloatBuffer,
                    midBuffer: midFloatBuffer,
                    outputBuffer: outputFloatBuffer,
                    gateByteCount: mppIntermediateFloatByteCount,
                    midByteCount: mppIntermediateFloatByteCount,
                    outputByteCount: mppOutputFloatByteCount,
                    config: config,
                    iterations: iterations,
                    checksum: { sampleFloatChecksum(buffer: outputFloatBuffer, count: outputElementCount) }
                ))
            }
        } else {
            print("  MXFP4 rows: skipped; hidden and intermediate must be multiples of 32.")
        }
        mppResults.append(try runMPPMLPBenchmarkVariant(
            name: "MPP/NAX MLP FP8 E4M3 weights: h_fp8_f + h_fp8_f",
            gateKernelName: "mpp_mlp_h_fp8e4m3_f_n128",
            downKernelName: "mpp_mlp_h_fp8e4m3_f_n128",
            device: device,
            library: mppLibrary,
            queue: queue,
            inputBuffer: xHalfBuffer,
            gateWeightBuffer: gateWeightFP8E4M3Buffer,
            upWeightBuffer: upWeightFP8E4M3Buffer,
            gateBuffer: gateFloatBuffer,
            upBuffer: upFloatBuffer,
            midBuffer: midHalfBuffer,
            downWeightBuffer: downWeightFP8E4M3Buffer,
            outputBuffer: outputFloatBuffer,
            gateByteCount: mppIntermediateFloatByteCount,
            midByteCount: mppIntermediateHalfByteCount,
            outputByteCount: mppOutputFloatByteCount,
            swigluKernelName: "mpp_mlp_swiglu_half",
            config: config,
            iterations: iterations,
            checksum: { sampleFloatChecksum(buffer: outputFloatBuffer, count: outputElementCount) }
        ))
        mppResults.append(try runMPPMLPBenchmarkVariant(
            name: "MPP/NAX MLP FP8 E4M3 weights: h_fp8_h + h_fp8_h",
            gateKernelName: "mpp_mlp_h_fp8e4m3_h_n128",
            downKernelName: "mpp_mlp_h_fp8e4m3_h_n128",
            device: device,
            library: mppLibrary,
            queue: queue,
            inputBuffer: xHalfBuffer,
            gateWeightBuffer: gateWeightFP8E4M3Buffer,
            upWeightBuffer: upWeightFP8E4M3Buffer,
            gateBuffer: gateHalfBuffer,
            upBuffer: upHalfBuffer,
            midBuffer: midHalfBuffer,
            downWeightBuffer: downWeightFP8E4M3Buffer,
            outputBuffer: outputBuffer,
            gateByteCount: mppIntermediateHalfByteCount,
            midByteCount: mppIntermediateHalfByteCount,
            outputByteCount: outputElementCount * MemoryLayout<UInt16>.stride,
            swigluKernelName: "mpp_mlp_swiglu_half_from_half",
            config: config,
            iterations: iterations,
            checksum: { sampleHalfRawChecksum(buffer: outputBuffer, count: outputElementCount) }
        ))
        mppResults.append(try runMPPMLPBenchmarkVariant(
            name: "MPP/NAX MLP FP8 E5M2 weights: h_fp8_f + h_fp8_f",
            gateKernelName: "mpp_mlp_h_fp8e5m2_f_n128",
            downKernelName: "mpp_mlp_h_fp8e5m2_f_n128",
            device: device,
            library: mppLibrary,
            queue: queue,
            inputBuffer: xHalfBuffer,
            gateWeightBuffer: gateWeightFP8E5M2Buffer,
            upWeightBuffer: upWeightFP8E5M2Buffer,
            gateBuffer: gateFloatBuffer,
            upBuffer: upFloatBuffer,
            midBuffer: midHalfBuffer,
            downWeightBuffer: downWeightFP8E5M2Buffer,
            outputBuffer: outputFloatBuffer,
            gateByteCount: mppIntermediateFloatByteCount,
            midByteCount: mppIntermediateHalfByteCount,
            outputByteCount: mppOutputFloatByteCount,
            swigluKernelName: "mpp_mlp_swiglu_half",
            config: config,
            iterations: iterations,
            checksum: { sampleFloatChecksum(buffer: outputFloatBuffer, count: outputElementCount) }
        ))
        if hasMPPTensorOpsInt2 {
            mppResults.append(try runMPPMLPBenchmarkVariant(
                name: "MPP/NAX MLP INT2 weights: h_i2_f + h_i2_f",
                gateKernelName: "mpp_mlp_h_i2_f_n128",
                downKernelName: "mpp_mlp_h_i2_f_n128",
                device: device,
                library: mppLibrary,
                queue: queue,
                inputBuffer: xHalfBuffer,
                gateWeightBuffer: gateWeightInt2Buffer,
                upWeightBuffer: upWeightInt2Buffer,
                gateBuffer: gateFloatBuffer,
                upBuffer: upFloatBuffer,
                midBuffer: midHalfBuffer,
                downWeightBuffer: downWeightInt2Buffer,
                outputBuffer: outputFloatBuffer,
                gateByteCount: mppIntermediateFloatByteCount,
                midByteCount: mppIntermediateHalfByteCount,
                outputByteCount: mppOutputFloatByteCount,
                swigluKernelName: "mpp_mlp_swiglu_half",
                config: config,
                iterations: iterations,
                checksum: { sampleFloatChecksum(buffer: outputFloatBuffer, count: outputElementCount) }
            ))
        } else {
            print("  MPP/NAX INT2 TensorOps rows: skipped; this runtime compiler does not expose int2b_format.")
        }
        mppResults.append(try runMPPMLPBenchmarkVariant(
            name: "MPP/NAX W8A8 MLP: i8 x int8_t -> i32",
            gateKernelName: "mpp_mlp_i8_i8_i32_n128",
            downKernelName: "mpp_mlp_i8_i8_i32_n128",
            device: device,
            library: mppLibrary,
            queue: queue,
            inputBuffer: xInt8Buffer,
            gateWeightBuffer: gateWeightInt8Buffer,
            upWeightBuffer: upWeightInt8Buffer,
            gateBuffer: gateIntBuffer,
            upBuffer: upIntBuffer,
            midBuffer: midInt8Buffer,
            downWeightBuffer: downWeightInt8Buffer,
            outputBuffer: outputIntBuffer,
            gateByteCount: mppIntermediateIntByteCount,
            midByteCount: mppIntermediateInt8ByteCount,
            outputByteCount: mppOutputIntByteCount,
            swigluKernelName: "mpp_mlp_swiglu_i8",
            config: config,
            iterations: iterations,
            checksum: { sampleIntChecksum(buffer: outputIntBuffer, count: outputElementCount) }
        ))
        mppResults.append(try runMPPMLPBenchmarkVariant(
            name: "MPP/NAX W4A8 MLP: i8 x int4b -> i32",
            gateKernelName: "mpp_mlp_i8_i4_i32_n128",
            downKernelName: "mpp_mlp_i8_i4_i32_n128",
            device: device,
            library: mppLibrary,
            queue: queue,
            inputBuffer: xInt8Buffer,
            gateWeightBuffer: gateWeightInt4Buffer,
            upWeightBuffer: upWeightInt4Buffer,
            gateBuffer: gateIntBuffer,
            upBuffer: upIntBuffer,
            midBuffer: midInt8Buffer,
            downWeightBuffer: downWeightInt4Buffer,
            outputBuffer: outputIntBuffer,
            gateByteCount: mppIntermediateIntByteCount,
            midByteCount: mppIntermediateInt8ByteCount,
            outputByteCount: mppOutputIntByteCount,
            swigluKernelName: "mpp_mlp_swiglu_i8",
            config: config,
            iterations: iterations,
            checksum: { sampleIntChecksum(buffer: outputIntBuffer, count: outputElementCount) }
        ))
        if hasMPPTensorOpsInt2 {
            mppResults.append(try runMPPMLPBenchmarkVariant(
                name: "MPP/NAX W2A8 MLP: i8 x int2b -> i32",
                gateKernelName: "mpp_mlp_i8_i2_i32_n128",
                downKernelName: "mpp_mlp_i8_i2_i32_n128",
                device: device,
                library: mppLibrary,
                queue: queue,
                inputBuffer: xInt8Buffer,
                gateWeightBuffer: gateWeightInt2Buffer,
                upWeightBuffer: upWeightInt2Buffer,
                gateBuffer: gateIntBuffer,
                upBuffer: upIntBuffer,
                midBuffer: midInt8Buffer,
                downWeightBuffer: downWeightInt2Buffer,
                outputBuffer: outputIntBuffer,
                gateByteCount: mppIntermediateIntByteCount,
                midByteCount: mppIntermediateInt8ByteCount,
                outputByteCount: mppOutputIntByteCount,
                swigluKernelName: "mpp_mlp_swiglu_i8",
                config: config,
                iterations: iterations,
                checksum: { sampleIntChecksum(buffer: outputIntBuffer, count: outputElementCount) }
            ))
        }
    } else {
        print("  MPP/NAX MLP rows: skipped because tokens must be a multiple of 64.")
    }

    let halfInt8WeightResult = try runMLPBenchmarkVariant(
        name: "half MLP with INT8 weights: half x int8_t -> half",
        gateKernelName: "half_int8_mlp_gate_up_benchmark",
        downKernelName: "half_int8_mlp_down_benchmark",
        device: device,
        library: library,
        queue: queue,
        inputBuffer: xHalfBuffer,
        gateWeightBuffer: gateWeightInt8Buffer,
        upWeightBuffer: upWeightInt8Buffer,
        intermediateBuffer: intermediateBuffer,
        downWeightBuffer: downWeightInt8Buffer,
        outputBuffer: outputBuffer,
        config: config,
        iterations: iterations,
        checksum: { sampleHalfRawChecksum(buffer: outputBuffer, count: outputElementCount) }
    )

    let halfInt4WeightResult = try runMLPBenchmarkVariant(
        name: "half MLP with INT4 weights: half x int4b_format -> half",
        gateKernelName: "half_int4_mlp_gate_up_benchmark",
        downKernelName: "half_int4_mlp_down_benchmark",
        device: device,
        library: library,
        queue: queue,
        inputBuffer: xHalfBuffer,
        gateWeightBuffer: gateWeightInt4Buffer,
        upWeightBuffer: upWeightInt4Buffer,
        intermediateBuffer: intermediateBuffer,
        downWeightBuffer: downWeightInt4Buffer,
        outputBuffer: outputBuffer,
        config: config,
        iterations: iterations,
        checksum: { sampleHalfRawChecksum(buffer: outputBuffer, count: outputElementCount) }
    )

    let halfInt2WeightResult = try runMLPBenchmarkVariant(
        name: "half MLP with INT2 weights: half x int2b_format -> half",
        gateKernelName: "half_int2_mlp_gate_up_benchmark",
        downKernelName: "half_int2_mlp_down_benchmark",
        device: device,
        library: library,
        queue: queue,
        inputBuffer: xHalfBuffer,
        gateWeightBuffer: gateWeightInt2Buffer,
        upWeightBuffer: upWeightInt2Buffer,
        intermediateBuffer: intermediateBuffer,
        downWeightBuffer: downWeightInt2Buffer,
        outputBuffer: outputBuffer,
        config: config,
        iterations: iterations,
        checksum: { sampleHalfRawChecksum(buffer: outputBuffer, count: outputElementCount) }
    )

    let halfFP4Result = try runMLPBenchmarkVariant(
        name: "half MLP with FP4 E2M1 g16 weights: half x FP4 -> half",
        gateKernelName: "half_fp4_mlp_gate_up_benchmark",
        downKernelName: "half_fp4_mlp_down_benchmark",
        device: device,
        library: library,
        queue: queue,
        inputBuffer: xHalfBuffer,
        gateWeightBuffer: gateWeightFP4Buffer,
        upWeightBuffer: upWeightFP4Buffer,
        intermediateBuffer: intermediateBuffer,
        downWeightBuffer: downWeightFP4Buffer,
        outputBuffer: outputBuffer,
        config: config,
        iterations: iterations,
        checksum: { sampleHalfRawChecksum(buffer: outputBuffer, count: outputElementCount) }
    )

    var fp4GroupSweepResults: [BenchmarkResult] = []
    if arguments.contains("--fp4-group-sweep") {
        fp4GroupSweepResults.append(try runMLPBenchmarkVariant(
            name: "half MLP with FP4 E2M1 g8 weights: half x FP4 -> half",
            gateKernelName: "half_fp4_mlp_gate_up_g8_benchmark",
            downKernelName: "half_fp4_mlp_down_g8_benchmark",
            device: device,
            library: library,
            queue: queue,
            inputBuffer: xHalfBuffer,
            gateWeightBuffer: gateWeightFP4Buffer,
            upWeightBuffer: upWeightFP4Buffer,
            intermediateBuffer: intermediateBuffer,
            downWeightBuffer: downWeightFP4Buffer,
            outputBuffer: outputBuffer,
            config: config,
            iterations: iterations,
            checksum: { sampleHalfRawChecksum(buffer: outputBuffer, count: outputElementCount) }
        ))
    }

    let results = [
        halfResult,
    ] + mppResults + [
        halfInt8WeightResult,
        halfInt4WeightResult,
        halfInt2WeightResult,
        halfFP4Result
    ] + fp4GroupSweepResults
    results.forEach(printMLPBenchmarkResult)
    printRelativeSlowdowns(results)
}

struct AccumProbePath {
    let name: String
    let halfKernel: String
    let floatKernel: String
    let aBytesPerElement: Double
    let bBytesPerElement: Double
    let aFill: UInt8
    let bFill: UInt8
    let productValue: Double
}

func runAccumulatorProbe(device: MTLDevice, queue: MTLCommandQueue) throws {
    let source = try loadShaderSource(named: "AccumProbeKernels")
    let options = MTLCompileOptions()
    options.languageVersion = .version4_1
    options.mathMode = .safe
    let library = try device.makeLibrary(source: source, options: options)

    let m = 64
    let n = 128
    let maxK = 131_072

    // Constant fills: 0x3c00 = half 1.0, 0x4800 = half 8.0 (per 16-bit element);
    // packed bytes 0x22 = two FP4 1.0 nibbles, 0x77 = two FP4 6.0 nibbles,
    // 0x38 = FP8 E4M3 1.0, 0x58 = FP8 E4M3 16.0.
    func makeConstantHalfBuffer(elementCount: Int, value: Float16, label: String) throws -> MTLBuffer {
        let values = [Float16](repeating: value, count: elementCount)
        guard let buffer = device.makeBuffer(
            bytes: values,
            length: elementCount * MemoryLayout<Float16>.stride,
            options: .storageModeShared
        ) else {
            throw ProbeError.missingBuffer(label)
        }
        return buffer
    }
    func makeFilledByteBuffer(byteCount: Int, fill: UInt8, label: String) throws -> MTLBuffer {
        guard let buffer = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            throw ProbeError.missingBuffer(label)
        }
        memset(buffer.contents(), Int32(fill), byteCount)
        return buffer
    }

    let aOnesHalf = try makeConstantHalfBuffer(elementCount: m * maxK, value: 1.0, label: "accum A half ones")
    let bOnesHalf = try makeConstantHalfBuffer(elementCount: n * maxK, value: 1.0, label: "accum B half ones")
    let aBigHalf = try makeConstantHalfBuffer(elementCount: m * maxK, value: 8.0, label: "accum A half 8.0")
    let bBigHalf = try makeConstantHalfBuffer(elementCount: n * maxK, value: 8.0, label: "accum B half 8.0")
    let aOnesFP4 = try makeFilledByteBuffer(byteCount: m * maxK / 2, fill: 0x22, label: "accum A fp4 ones")
    let bOnesFP4 = try makeFilledByteBuffer(byteCount: n * maxK / 2, fill: 0x22, label: "accum B fp4 ones")
    let aBigFP4 = try makeFilledByteBuffer(byteCount: m * maxK / 2, fill: 0x77, label: "accum A fp4 6.0")
    let bBigFP4 = try makeFilledByteBuffer(byteCount: n * maxK / 2, fill: 0x77, label: "accum B fp4 6.0")
    let aOnesFP8 = try makeFilledByteBuffer(byteCount: m * maxK, fill: 0x38, label: "accum A fp8 ones")
    let bOnesFP8 = try makeFilledByteBuffer(byteCount: n * maxK, fill: 0x38, label: "accum B fp8 ones")
    let aBigFP8 = try makeFilledByteBuffer(byteCount: m * maxK, fill: 0x58, label: "accum A fp8 16.0")
    let bBigFP8 = try makeFilledByteBuffer(byteCount: n * maxK, fill: 0x58, label: "accum B fp8 16.0")

    guard let cHalfBuffer = device.makeBuffer(
        length: m * n * MemoryLayout<UInt16>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("accum C half")
    }
    guard let cFloatBuffer = device.makeBuffer(
        length: m * n * MemoryLayout<Float>.stride,
        options: .storageModeShared
    ) else {
        throw ProbeError.missingBuffer("accum C float")
    }

    func runCase(kernelName: String, aBuffer: MTLBuffer, bBuffer: MTLBuffer, k: Int, halfResult: Bool) throws -> (Double, Double) {
        guard let function = library.makeFunction(name: kernelName) else {
            throw ProbeError.missingFunction(kernelName)
        }
        let pipeline = try device.makeComputePipelineState(function: function)
        let cBuffer = halfResult ? cHalfBuffer : cFloatBuffer
        memset(cBuffer.contents(), 0, cBuffer.length)

        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw ProbeError.missingCommandBuffer
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ProbeError.missingComputeEncoder
        }
        encodeMPPMatmulDispatch(
            encoder: encoder,
            pipeline: pipeline,
            aBuffer: aBuffer,
            bBuffer: bBuffer,
            cBuffer: cBuffer,
            m: UInt32(m),
            n: UInt32(n),
            k: UInt32(k)
        )
        encoder.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw ProbeError.gpuExecutionFailed("\(kernelName) k=\(k) failed: \(error)")
        }

        if halfResult {
            let pointer = cBuffer.contents().bindMemory(to: Float16.self, capacity: m * n)
            return (Double(pointer[0]), Double(pointer[m * n - 1]))
        }
        let pointer = cBuffer.contents().bindMemory(to: Float.self, capacity: m * n)
        return (Double(pointer[0]), Double(pointer[m * n - 1]))
    }

    func formatValue(_ value: Double) -> String {
        if value.isInfinite {
            return value < 0 ? "-inf" : "inf"
        }
        if value.isNaN {
            return "nan"
        }
        if value == value.rounded() && abs(value) < 1e15 {
            return String(format: "%.0f", value)
        }
        return String(format: "%.6g", value)
    }

    func runSweep(path: AccumProbePath, ks: [Int]) throws {
        print("")
        print("  \(path.name), product per element = \(formatValue(path.productValue)):")
        print("    K | exact | half dest C[0] | float dest C[0]")
        for k in ks {
            let exact = path.productValue * Double(k)
            let aBuffer: MTLBuffer
            let bBuffer: MTLBuffer
            switch (path.halfKernel, path.productValue) {
            case ("accum_h_h_h", 1.0): aBuffer = aOnesHalf; bBuffer = bOnesHalf
            case ("accum_h_h_h", _): aBuffer = aBigHalf; bBuffer = bBigHalf
            case ("accum_h_fp4_h", _): aBuffer = aOnesHalf; bBuffer = bOnesFP4
            case ("accum_fp4_fp4_h", 1.0): aBuffer = aOnesFP4; bBuffer = bOnesFP4
            case ("accum_fp4_fp4_h", _): aBuffer = aBigFP4; bBuffer = bBigFP4
            case ("accum_fp8_fp8_h", 1.0): aBuffer = aOnesFP8; bBuffer = bOnesFP8
            default: aBuffer = aBigFP8; bBuffer = bBigFP8
            }
            let (halfFirst, halfLast) = try runCase(kernelName: path.halfKernel, aBuffer: aBuffer, bBuffer: bBuffer, k: k, halfResult: true)
            let (floatFirst, floatLast) = try runCase(kernelName: path.floatKernel, aBuffer: aBuffer, bBuffer: bBuffer, k: k, halfResult: false)
            let uniformNote = (halfFirst == halfLast && floatFirst == floatLast) ? "" : "  [non-uniform: half last \(formatValue(halfLast)), float last \(formatValue(floatLast))]"
            print("    \(k) | \(formatValue(exact)) | \(formatValue(halfFirst)) | \(formatValue(floatFirst))\(uniformNote)")
        }
    }

    print("")
    print("Accumulator precision/overflow probe (matmul2d, 64x128 tile, dot length K):")
    print("  All inputs are constant fills, so every C element has the same exact value.")
    print("  A deviation in 'half dest' but not 'float dest' isolates destination/accumulator precision.")

    let onesKs = [1024, 2048, 2080, 4096, 4160, 8192, 16384, 32768, 65504, 65536, 131_072]
    try runSweep(path: AccumProbePath(
        name: "half x half -> C, ones",
        halfKernel: "accum_h_h_h", floatKernel: "accum_h_h_f",
        aBytesPerElement: 2, bBytesPerElement: 2, aFill: 0, bFill: 0, productValue: 1.0
    ), ks: onesKs)
    try runSweep(path: AccumProbePath(
        name: "half x FP4 E2M1 -> C, ones",
        halfKernel: "accum_h_fp4_h", floatKernel: "accum_h_fp4_f",
        aBytesPerElement: 2, bBytesPerElement: 0.5, aFill: 0, bFill: 0x22, productValue: 1.0
    ), ks: onesKs)
    try runSweep(path: AccumProbePath(
        name: "FP4 E2M1 x FP4 E2M1 -> C, ones",
        halfKernel: "accum_fp4_fp4_h", floatKernel: "accum_fp4_fp4_f",
        aBytesPerElement: 0.5, bBytesPerElement: 0.5, aFill: 0x22, bFill: 0x22, productValue: 1.0
    ), ks: onesKs)
    try runSweep(path: AccumProbePath(
        name: "FP8 E4M3 x FP8 E4M3 -> C, ones",
        halfKernel: "accum_fp8_fp8_h", floatKernel: "accum_fp8_fp8_f",
        aBytesPerElement: 1, bBytesPerElement: 1, aFill: 0x38, bFill: 0x38, productValue: 1.0
    ), ks: onesKs)

    print("")
    print("  Overflow cases (products sized to cross half max 65504):")
    try runSweep(path: AccumProbePath(
        name: "half x half -> C, 8.0 x 8.0 (product 64)",
        halfKernel: "accum_h_h_h", floatKernel: "accum_h_h_f",
        aBytesPerElement: 2, bBytesPerElement: 2, aFill: 0, bFill: 0, productValue: 64.0
    ), ks: [512, 1022, 1024, 2048])
    try runSweep(path: AccumProbePath(
        name: "FP4 E2M1 x FP4 E2M1 -> C, 6.0 x 6.0 (product 36)",
        halfKernel: "accum_fp4_fp4_h", floatKernel: "accum_fp4_fp4_f",
        aBytesPerElement: 0.5, bBytesPerElement: 0.5, aFill: 0x77, bFill: 0x77, productValue: 36.0
    ), ks: [1024, 1818, 1820, 2048, 4096])
    try runSweep(path: AccumProbePath(
        name: "FP8 E4M3 x FP8 E4M3 -> C, 16.0 x 16.0 (product 256)",
        halfKernel: "accum_fp8_fp8_h", floatKernel: "accum_fp8_fp8_f",
        aBytesPerElement: 1, bBytesPerElement: 1, aFill: 0x58, bFill: 0x58, productValue: 256.0
    ), ks: [128, 254, 256, 1024])
}

func runMatmulBenchmarkKernel(
    name: String,
    kernelName: String,
    device: MTLDevice,
    library: MTLLibrary,
    queue: MTLCommandQueue,
    aBuffer: MTLBuffer,
    bBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    outputElementCount: Int,
    outputByteCount: Int,
    config: MatmulConfig,
    iterations: Int,
    checksum: () -> String
) throws -> MatmulBenchmarkResult {
    guard let function = library.makeFunction(name: kernelName) else {
        throw ProbeError.missingFunction(kernelName)
    }
    let pipeline = try device.makeComputePipelineState(function: function)
    let gridSize = MTLSize(width: Int(config.columns), height: Int(config.rows), depth: 1)
    let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)

    try encodeMatmul(
        pipeline: pipeline,
        queue: queue,
        aBuffer: aBuffer,
        bBuffer: bBuffer,
        outputBuffer: outputBuffer,
        config: config,
        gridSize: gridSize,
        threadsPerThreadgroup: threadsPerThreadgroup,
        iterations: 1
    )

    memset(outputBuffer.contents(), 0, outputByteCount)
    guard let commandBuffer = queue.makeCommandBuffer() else {
        throw ProbeError.missingCommandBuffer
    }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw ProbeError.missingComputeEncoder
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    encoder.setBuffer(outputBuffer, offset: 0, index: 2)
    var mutableConfig = config
    encoder.setBytes(&mutableConfig, length: MemoryLayout<MatmulConfig>.stride, index: 3)
    for _ in 0..<iterations {
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
    }
    encoder.endEncoding()

    let wallStart = DispatchTime.now().uptimeNanoseconds
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let wallEnd = DispatchTime.now().uptimeNanoseconds

    if let error = commandBuffer.error {
        throw ProbeError.gpuExecutionFailed("\(name) command buffer failed: \(error)")
    }
    guard commandBuffer.status == .completed else {
        throw ProbeError.gpuExecutionFailed("\(name) command buffer finished with status \(commandBuffer.status.rawValue).")
    }

    let wallSeconds = Double(wallEnd - wallStart) / 1_000_000_000.0
    let gpuSeconds = commandBuffer.gpuEndTime > commandBuffer.gpuStartTime
        ? commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        : wallSeconds

    return MatmulBenchmarkResult(
        name: name,
        seconds: gpuSeconds,
        iterations: iterations,
        rows: Int(config.rows),
        columns: Int(config.columns),
        depth: Int(config.depth),
        checksum: checksum()
    )
}

func runMLPBenchmarkVariant(
    name: String,
    gateKernelName: String,
    downKernelName: String,
    device: MTLDevice,
    library: MTLLibrary,
    queue: MTLCommandQueue,
    inputBuffer: MTLBuffer,
    gateWeightBuffer: MTLBuffer,
    upWeightBuffer: MTLBuffer,
    intermediateBuffer: MTLBuffer,
    downWeightBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    config: MLPConfig,
    iterations: Int,
    checksum: () -> String
) throws -> BenchmarkResult {
    guard let gateFunction = library.makeFunction(name: gateKernelName) else {
        throw ProbeError.missingFunction(gateKernelName)
    }
    guard let downFunction = library.makeFunction(name: downKernelName) else {
        throw ProbeError.missingFunction(downKernelName)
    }

    let gatePipeline = try device.makeComputePipelineState(function: gateFunction)
    let downPipeline = try device.makeComputePipelineState(function: downFunction)

    _ = try encodeMLP(
        gatePipeline: gatePipeline,
        downPipeline: downPipeline,
        queue: queue,
        inputBuffer: inputBuffer,
        gateWeightBuffer: gateWeightBuffer,
        upWeightBuffer: upWeightBuffer,
        intermediateBuffer: intermediateBuffer,
        downWeightBuffer: downWeightBuffer,
        outputBuffer: outputBuffer,
        config: config,
        iterations: 1
    )

    let intermediateByteCount = Int(config.tokens) * Int(config.intermediate) * MemoryLayout<UInt16>.stride
    let outputByteCount = Int(config.tokens) * Int(config.hidden) * MemoryLayout<UInt16>.stride
    memset(intermediateBuffer.contents(), 0, intermediateByteCount)
    memset(outputBuffer.contents(), 0, outputByteCount)

    let seconds = try encodeMLP(
        gatePipeline: gatePipeline,
        downPipeline: downPipeline,
        queue: queue,
        inputBuffer: inputBuffer,
        gateWeightBuffer: gateWeightBuffer,
        upWeightBuffer: upWeightBuffer,
        intermediateBuffer: intermediateBuffer,
        downWeightBuffer: downWeightBuffer,
        outputBuffer: outputBuffer,
        config: config,
        iterations: iterations
    )

    return BenchmarkResult(
        name: name,
        seconds: seconds,
        iterations: iterations,
        rows: Int(config.tokens),
        columns: Int(config.hidden),
        depth: Int(config.intermediate),
        checksum: checksum()
    )
}

func runMPPMLPBenchmarkVariant(
    name: String,
    gateKernelName: String,
    downKernelName: String,
    device: MTLDevice,
    library: MTLLibrary,
    queue: MTLCommandQueue,
    inputBuffer: MTLBuffer,
    gateWeightBuffer: MTLBuffer,
    upWeightBuffer: MTLBuffer,
    gateBuffer: MTLBuffer,
    upBuffer: MTLBuffer,
    midBuffer: MTLBuffer,
    downWeightBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    gateByteCount: Int,
    midByteCount: Int,
    outputByteCount: Int,
    swigluKernelName: String,
    config: MLPConfig,
    iterations: Int,
    checksum: () -> String
) throws -> BenchmarkResult {
    guard let gateFunction = library.makeFunction(name: gateKernelName) else {
        throw ProbeError.missingFunction(gateKernelName)
    }
    guard let downFunction = library.makeFunction(name: downKernelName) else {
        throw ProbeError.missingFunction(downKernelName)
    }
    guard let swigluFunction = library.makeFunction(name: swigluKernelName) else {
        throw ProbeError.missingFunction(swigluKernelName)
    }

    let gatePipeline = try device.makeComputePipelineState(function: gateFunction)
    let downPipeline = try device.makeComputePipelineState(function: downFunction)
    let swigluPipeline = try device.makeComputePipelineState(function: swigluFunction)

    _ = try encodeMPPMLP(
        gatePipeline: gatePipeline,
        downPipeline: downPipeline,
        swigluPipeline: swigluPipeline,
        queue: queue,
        inputBuffer: inputBuffer,
        gateWeightBuffer: gateWeightBuffer,
        upWeightBuffer: upWeightBuffer,
        gateBuffer: gateBuffer,
        upBuffer: upBuffer,
        midBuffer: midBuffer,
        downWeightBuffer: downWeightBuffer,
        outputBuffer: outputBuffer,
        config: config,
        iterations: 1
    )

    memset(gateBuffer.contents(), 0, gateByteCount)
    memset(upBuffer.contents(), 0, gateByteCount)
    memset(midBuffer.contents(), 0, midByteCount)
    memset(outputBuffer.contents(), 0, outputByteCount)

    let seconds = try encodeMPPMLP(
        gatePipeline: gatePipeline,
        downPipeline: downPipeline,
        swigluPipeline: swigluPipeline,
        queue: queue,
        inputBuffer: inputBuffer,
        gateWeightBuffer: gateWeightBuffer,
        upWeightBuffer: upWeightBuffer,
        gateBuffer: gateBuffer,
        upBuffer: upBuffer,
        midBuffer: midBuffer,
        downWeightBuffer: downWeightBuffer,
        outputBuffer: outputBuffer,
        config: config,
        iterations: iterations
    )

    return BenchmarkResult(
        name: name,
        seconds: seconds,
        iterations: iterations,
        rows: Int(config.tokens),
        columns: Int(config.hidden),
        depth: Int(config.intermediate),
        checksum: checksum()
    )
}

func runMXFP4FusedMLPBenchmarkVariant(
    name: String,
    matmulKernelName: String,
    nTile: Int,
    mTile: Int = 64,
    simdgroupsPerThreadgroup: Int = 4,
    device: MTLDevice,
    mxfp4Library: MTLLibrary,
    mppLibrary: MTLLibrary,
    queue: MTLCommandQueue,
    inputBuffer: MTLBuffer,
    gateWeightBuffer: MTLBuffer,
    upWeightBuffer: MTLBuffer,
    downWeightBuffer: MTLBuffer,
    gateScaleBuffer: MTLBuffer,
    upScaleBuffer: MTLBuffer,
    downScaleBuffer: MTLBuffer,
    gateBuffer: MTLBuffer,
    upBuffer: MTLBuffer,
    midBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    gateByteCount: Int,
    midByteCount: Int,
    outputByteCount: Int,
    config: MLPConfig,
    iterations: Int,
    checksum: () -> String
) throws -> BenchmarkResult {
    guard let matmulFunction = mxfp4Library.makeFunction(name: matmulKernelName) else {
        throw ProbeError.missingFunction(matmulKernelName)
    }
    guard let swigluFunction = mppLibrary.makeFunction(name: "mpp_mlp_swiglu_half") else {
        throw ProbeError.missingFunction("mpp_mlp_swiglu_half")
    }

    let matmulPipeline = try device.makeComputePipelineState(function: matmulFunction)
    let swigluPipeline = try device.makeComputePipelineState(function: swigluFunction)

    let encode = { (iterationCount: Int) throws -> Double in
        try encodeMXFP4FusedMLP(
            matmulPipeline: matmulPipeline,
            swigluPipeline: swigluPipeline,
            nTile: nTile,
            mTile: mTile,
            simdgroupsPerThreadgroup: simdgroupsPerThreadgroup,
            queue: queue,
            inputBuffer: inputBuffer,
            gateWeightBuffer: gateWeightBuffer,
            upWeightBuffer: upWeightBuffer,
            downWeightBuffer: downWeightBuffer,
            gateScaleBuffer: gateScaleBuffer,
            upScaleBuffer: upScaleBuffer,
            downScaleBuffer: downScaleBuffer,
            gateBuffer: gateBuffer,
            upBuffer: upBuffer,
            midBuffer: midBuffer,
            outputBuffer: outputBuffer,
            config: config,
            iterations: iterationCount
        )
    }

    _ = try encode(1)
    memset(gateBuffer.contents(), 0, gateByteCount)
    memset(upBuffer.contents(), 0, gateByteCount)
    memset(midBuffer.contents(), 0, midByteCount)
    memset(outputBuffer.contents(), 0, outputByteCount)
    let seconds = try encode(iterations)

    return BenchmarkResult(
        name: name,
        seconds: seconds,
        iterations: iterations,
        rows: Int(config.tokens),
        columns: Int(config.hidden),
        depth: Int(config.intermediate),
        checksum: checksum()
    )
}

func encodeMXFP4FusedMLP(
    matmulPipeline: MTLComputePipelineState,
    swigluPipeline: MTLComputePipelineState,
    nTile: Int,
    mTile: Int = 64,
    simdgroupsPerThreadgroup: Int = 4,
    queue: MTLCommandQueue,
    inputBuffer: MTLBuffer,
    gateWeightBuffer: MTLBuffer,
    upWeightBuffer: MTLBuffer,
    downWeightBuffer: MTLBuffer,
    gateScaleBuffer: MTLBuffer,
    upScaleBuffer: MTLBuffer,
    downScaleBuffer: MTLBuffer,
    gateBuffer: MTLBuffer,
    upBuffer: MTLBuffer,
    midBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    config: MLPConfig,
    iterations: Int
) throws -> Double {
    guard let commandBuffer = queue.makeCommandBuffer() else {
        throw ProbeError.missingCommandBuffer
    }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw ProbeError.missingComputeEncoder
    }

    var mutableConfig = config
    for _ in 0..<iterations {
        encodeMXFP4MatmulDispatch(
            encoder: encoder,
            pipeline: matmulPipeline,
            aBuffer: inputBuffer,
            bBuffer: gateWeightBuffer,
            cBuffer: gateBuffer,
            scaleBuffer: gateScaleBuffer,
            m: config.tokens,
            n: config.intermediate,
            k: config.hidden,
            nTile: nTile,
            mTile: mTile,
            simdgroupsPerThreadgroup: simdgroupsPerThreadgroup
        )
        encodeMXFP4MatmulDispatch(
            encoder: encoder,
            pipeline: matmulPipeline,
            aBuffer: inputBuffer,
            bBuffer: upWeightBuffer,
            cBuffer: upBuffer,
            scaleBuffer: upScaleBuffer,
            m: config.tokens,
            n: config.intermediate,
            k: config.hidden,
            nTile: nTile,
            mTile: mTile,
            simdgroupsPerThreadgroup: simdgroupsPerThreadgroup
        )

        encoder.setComputePipelineState(swigluPipeline)
        encoder.setBuffer(gateBuffer, offset: 0, index: 0)
        encoder.setBuffer(upBuffer, offset: 0, index: 1)
        encoder.setBuffer(midBuffer, offset: 0, index: 2)
        encoder.setBytes(&mutableConfig, length: MemoryLayout<MLPConfig>.stride, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: Int(config.intermediate), height: Int(config.tokens), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
        )

        encodeMXFP4MatmulDispatch(
            encoder: encoder,
            pipeline: matmulPipeline,
            aBuffer: midBuffer,
            bBuffer: downWeightBuffer,
            cBuffer: outputBuffer,
            scaleBuffer: downScaleBuffer,
            m: config.tokens,
            n: config.hidden,
            k: config.intermediate,
            nTile: nTile,
            mTile: mTile,
            simdgroupsPerThreadgroup: simdgroupsPerThreadgroup
        )
    }
    encoder.endEncoding()

    let wallStart = DispatchTime.now().uptimeNanoseconds
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let wallEnd = DispatchTime.now().uptimeNanoseconds

    if let error = commandBuffer.error {
        throw ProbeError.gpuExecutionFailed("MXFP4 fused MLP command buffer failed: \(error)")
    }
    guard commandBuffer.status == .completed else {
        throw ProbeError.gpuExecutionFailed("MXFP4 fused MLP command buffer finished with status \(commandBuffer.status.rawValue).")
    }

    let wallSeconds = Double(wallEnd - wallStart) / 1_000_000_000.0
    return commandBuffer.gpuEndTime > commandBuffer.gpuStartTime
        ? commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        : wallSeconds
}

func encodeMXFP4MatmulDispatch(
    encoder: MTLComputeCommandEncoder,
    pipeline: MTLComputePipelineState,
    aBuffer: MTLBuffer,
    bBuffer: MTLBuffer,
    cBuffer: MTLBuffer,
    scaleBuffer: MTLBuffer,
    m: UInt32,
    n: UInt32,
    k: UInt32,
    nTile: Int,
    mTile: Int = 64,
    simdgroupsPerThreadgroup: Int = 4
) {
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    encoder.setBuffer(cBuffer, offset: 0, index: 2)
    var mutableM = m
    var mutableN = n
    var mutableK = k
    encoder.setBytes(&mutableM, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&mutableN, length: MemoryLayout<UInt32>.stride, index: 4)
    encoder.setBytes(&mutableK, length: MemoryLayout<UInt32>.stride, index: 5)
    encoder.setBuffer(scaleBuffer, offset: 0, index: 6)

    let grid = MTLSize(
        width: (Int(n) + nTile - 1) / nTile,
        height: (Int(m) + mTile - 1) / mTile,
        depth: 1
    )
    let threads = MTLSize(width: pipeline.threadExecutionWidth * simdgroupsPerThreadgroup, height: 1, depth: 1)
    encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
}

func makeMXFP4ScalePlaneTensor(
    device: MTLDevice,
    dataBuffer: MTLBuffer,
    scaleBuffer: MTLBuffer,
    k: Int,
    n: Int
) throws -> MTLTensor {
    let dimensions = [NSInteger(k), NSInteger(n)]
    let blockFactors = [NSInteger(32), NSInteger(1)]
    let tensorExtents = dimensions.withUnsafeBufferPointer { pointer in
        MTLTensorExtents(__rank: dimensions.count, values: pointer.baseAddress!)!
    }
    let blockExtents = blockFactors.withUnsafeBufferPointer { pointer in
        MTLTensorExtents(__rank: blockFactors.count, values: pointer.baseAddress!)!
    }

    let scalePlane = MTLTensorAuxiliaryPlaneDescriptor()
    scalePlane.dataType = .float8ue8m0
    scalePlane.blockFactors = blockExtents
    let auxiliaryPlanes = MTLTensorAuxiliaryPlaneDescriptorMap()
    auxiliaryPlanes.setDescriptor(scalePlane, for: .scales)

    let strides = [NSInteger(1), NSInteger(k)]
    let strideExtents = strides.withUnsafeBufferPointer { pointer in
        MTLTensorExtents(__rank: strides.count, values: pointer.baseAddress!)!
    }

    let descriptor = MTLTensorDescriptor()
    descriptor.dimensions = tensorExtents
    descriptor.strides = strideExtents
    descriptor.dataType = .float4e2m1
    descriptor.usage = .compute
    descriptor.resourceOptions = .storageModeShared
    descriptor.auxiliaryPlanes = auxiliaryPlanes

    let attachments = MTLTensorBufferAttachments()
    attachments.setBuffer(dataBuffer, offset: 0, for: .data)
    attachments.setBuffer(scaleBuffer, offset: 0, for: .scales)

    return try device.makeTensor(descriptor: descriptor, attachments: attachments)
}

func runMXFP4HandleMLPBenchmarkVariant(
    name: String,
    matmulKernelName: String,
    nTile: Int,
    device: MTLDevice,
    mxfp4Library: MTLLibrary,
    mppLibrary: MTLLibrary,
    queue: MTLCommandQueue,
    inputBuffer: MTLBuffer,
    gateWeightTensor: MTLTensor,
    upWeightTensor: MTLTensor,
    downWeightTensor: MTLTensor,
    gateBuffer: MTLBuffer,
    upBuffer: MTLBuffer,
    midBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    gateByteCount: Int,
    midByteCount: Int,
    outputByteCount: Int,
    config: MLPConfig,
    iterations: Int,
    checksum: () -> String
) throws -> BenchmarkResult {
    guard let matmulFunction = mxfp4Library.makeFunction(name: matmulKernelName) else {
        throw ProbeError.missingFunction(matmulKernelName)
    }
    guard let swigluFunction = mppLibrary.makeFunction(name: "mpp_mlp_swiglu_half") else {
        throw ProbeError.missingFunction("mpp_mlp_swiglu_half")
    }

    let matmulPipeline = try device.makeComputePipelineState(function: matmulFunction)
    let swigluPipeline = try device.makeComputePipelineState(function: swigluFunction)

    let encode = { (iterationCount: Int) throws -> Double in
        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw ProbeError.missingCommandBuffer
        }
        guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
            throw ProbeError.missingComputeEncoder
        }
        encoder.useResource(gateWeightTensor, usage: .read)
        encoder.useResource(upWeightTensor, usage: .read)
        encoder.useResource(downWeightTensor, usage: .read)

        let dispatch = { (weightTensor: MTLTensor, aBuffer: MTLBuffer, cBuffer: MTLBuffer, m: UInt32, n: UInt32, k: UInt32) in
            encoder.setComputePipelineState(matmulPipeline)
            encoder.setBuffer(aBuffer, offset: 0, index: 0)
            var resourceID = weightTensor.gpuResourceID
            encoder.setBytes(&resourceID, length: MemoryLayout<MTLResourceID>.stride, index: 1)
            encoder.setBuffer(cBuffer, offset: 0, index: 2)
            var mutableM = m
            var mutableN = n
            var mutableK = k
            encoder.setBytes(&mutableM, length: MemoryLayout<UInt32>.stride, index: 3)
            encoder.setBytes(&mutableN, length: MemoryLayout<UInt32>.stride, index: 4)
            encoder.setBytes(&mutableK, length: MemoryLayout<UInt32>.stride, index: 5)
            let grid = MTLSize(
                width: (Int(n) + nTile - 1) / nTile,
                height: (Int(m) + 63) / 64,
                depth: 1
            )
            let threads = MTLSize(width: matmulPipeline.threadExecutionWidth * 4, height: 1, depth: 1)
            encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
        }

        var mutableConfig = config
        for _ in 0..<iterationCount {
            dispatch(gateWeightTensor, inputBuffer, gateBuffer, config.tokens, config.intermediate, config.hidden)
            dispatch(upWeightTensor, inputBuffer, upBuffer, config.tokens, config.intermediate, config.hidden)

            encoder.setComputePipelineState(swigluPipeline)
            encoder.setBuffer(gateBuffer, offset: 0, index: 0)
            encoder.setBuffer(upBuffer, offset: 0, index: 1)
            encoder.setBuffer(midBuffer, offset: 0, index: 2)
            encoder.setBytes(&mutableConfig, length: MemoryLayout<MLPConfig>.stride, index: 3)
            encoder.dispatchThreads(
                MTLSize(width: Int(config.intermediate), height: Int(config.tokens), depth: 1),
                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
            )

            dispatch(downWeightTensor, midBuffer, outputBuffer, config.tokens, config.hidden, config.intermediate)
        }
        encoder.endEncoding()

        let wallStart = DispatchTime.now().uptimeNanoseconds
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        let wallEnd = DispatchTime.now().uptimeNanoseconds

        if let error = commandBuffer.error {
            throw ProbeError.gpuExecutionFailed("MXFP4 handle MLP command buffer failed: \(error)")
        }
        guard commandBuffer.status == .completed else {
            throw ProbeError.gpuExecutionFailed("MXFP4 handle MLP command buffer finished with status \(commandBuffer.status.rawValue).")
        }

        let wallSeconds = Double(wallEnd - wallStart) / 1_000_000_000.0
        return commandBuffer.gpuEndTime > commandBuffer.gpuStartTime
            ? commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
            : wallSeconds
    }

    _ = try encode(1)
    memset(gateBuffer.contents(), 0, gateByteCount)
    memset(upBuffer.contents(), 0, gateByteCount)
    memset(midBuffer.contents(), 0, midByteCount)
    memset(outputBuffer.contents(), 0, outputByteCount)
    let seconds = try encode(iterations)

    return BenchmarkResult(
        name: name,
        seconds: seconds,
        iterations: iterations,
        rows: Int(config.tokens),
        columns: Int(config.hidden),
        depth: Int(config.intermediate),
        checksum: checksum()
    )
}

func runMXFP4DequantMLPBenchmarkVariant(
    name: String,
    dequantKernelName: String,
    device: MTLDevice,
    mxfp4Library: MTLLibrary,
    mppLibrary: MTLLibrary,
    queue: MTLCommandQueue,
    inputBuffer: MTLBuffer,
    gateWeightBuffer: MTLBuffer,
    upWeightBuffer: MTLBuffer,
    downWeightBuffer: MTLBuffer,
    gateScaleBuffer: MTLBuffer,
    upScaleBuffer: MTLBuffer,
    downScaleBuffer: MTLBuffer,
    gateWeightDequantBuffer: MTLBuffer,
    upWeightDequantBuffer: MTLBuffer,
    downWeightDequantBuffer: MTLBuffer,
    gateBuffer: MTLBuffer,
    upBuffer: MTLBuffer,
    midBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    gateByteCount: Int,
    midByteCount: Int,
    outputByteCount: Int,
    config: MLPConfig,
    iterations: Int,
    checksum: () -> String
) throws -> BenchmarkResult {
    guard let dequantFunction = mxfp4Library.makeFunction(name: dequantKernelName) else {
        throw ProbeError.missingFunction(dequantKernelName)
    }
    guard let gateFunction = mppLibrary.makeFunction(name: "mpp_mlp_h_h_f_n128") else {
        throw ProbeError.missingFunction("mpp_mlp_h_h_f_n128")
    }
    guard let downFunction = mppLibrary.makeFunction(name: "mpp_mlp_f_h_f_n128") else {
        throw ProbeError.missingFunction("mpp_mlp_f_h_f_n128")
    }
    guard let swigluFunction = mppLibrary.makeFunction(name: "mpp_mlp_swiglu_float") else {
        throw ProbeError.missingFunction("mpp_mlp_swiglu_float")
    }

    let dequantPipeline = try device.makeComputePipelineState(function: dequantFunction)
    let gatePipeline = try device.makeComputePipelineState(function: gateFunction)
    let downPipeline = try device.makeComputePipelineState(function: downFunction)
    let swigluPipeline = try device.makeComputePipelineState(function: swigluFunction)
    let valuesPerThread = dequantKernelName.contains("native") ? 16 : 2

    let encode = { (iterationCount: Int) throws -> Double in
        try encodeMXFP4DequantMLP(
            dequantPipeline: dequantPipeline,
            gatePipeline: gatePipeline,
            downPipeline: downPipeline,
            swigluPipeline: swigluPipeline,
            dequantValuesPerThread: valuesPerThread,
            queue: queue,
            inputBuffer: inputBuffer,
            gateWeightBuffer: gateWeightBuffer,
            upWeightBuffer: upWeightBuffer,
            downWeightBuffer: downWeightBuffer,
            gateScaleBuffer: gateScaleBuffer,
            upScaleBuffer: upScaleBuffer,
            downScaleBuffer: downScaleBuffer,
            gateWeightDequantBuffer: gateWeightDequantBuffer,
            upWeightDequantBuffer: upWeightDequantBuffer,
            downWeightDequantBuffer: downWeightDequantBuffer,
            gateBuffer: gateBuffer,
            upBuffer: upBuffer,
            midBuffer: midBuffer,
            outputBuffer: outputBuffer,
            config: config,
            iterations: iterationCount
        )
    }

    _ = try encode(1)
    memset(gateBuffer.contents(), 0, gateByteCount)
    memset(upBuffer.contents(), 0, gateByteCount)
    memset(midBuffer.contents(), 0, midByteCount)
    memset(outputBuffer.contents(), 0, outputByteCount)
    let seconds = try encode(iterations)

    return BenchmarkResult(
        name: name,
        seconds: seconds,
        iterations: iterations,
        rows: Int(config.tokens),
        columns: Int(config.hidden),
        depth: Int(config.intermediate),
        checksum: checksum()
    )
}

func encodeMXFP4DequantMLP(
    dequantPipeline: MTLComputePipelineState,
    gatePipeline: MTLComputePipelineState,
    downPipeline: MTLComputePipelineState,
    swigluPipeline: MTLComputePipelineState,
    dequantValuesPerThread: Int,
    queue: MTLCommandQueue,
    inputBuffer: MTLBuffer,
    gateWeightBuffer: MTLBuffer,
    upWeightBuffer: MTLBuffer,
    downWeightBuffer: MTLBuffer,
    gateScaleBuffer: MTLBuffer,
    upScaleBuffer: MTLBuffer,
    downScaleBuffer: MTLBuffer,
    gateWeightDequantBuffer: MTLBuffer,
    upWeightDequantBuffer: MTLBuffer,
    downWeightDequantBuffer: MTLBuffer,
    gateBuffer: MTLBuffer,
    upBuffer: MTLBuffer,
    midBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    config: MLPConfig,
    iterations: Int
) throws -> Double {
    guard let commandBuffer = queue.makeCommandBuffer() else {
        throw ProbeError.missingCommandBuffer
    }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw ProbeError.missingComputeEncoder
    }

    let encodeDequant = { (packed: MTLBuffer, scales: MTLBuffer, out: MTLBuffer, rows: UInt32, depth: UInt32) in
        encoder.setComputePipelineState(dequantPipeline)
        encoder.setBuffer(packed, offset: 0, index: 0)
        encoder.setBuffer(scales, offset: 0, index: 1)
        encoder.setBuffer(out, offset: 0, index: 2)
        var mutableRows = rows
        var mutableDepth = depth
        encoder.setBytes(&mutableRows, length: MemoryLayout<UInt32>.stride, index: 3)
        encoder.setBytes(&mutableDepth, length: MemoryLayout<UInt32>.stride, index: 4)
        encoder.dispatchThreads(
            MTLSize(width: Int(depth) / dequantValuesPerThread, height: Int(rows), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
        )
    }

    var mutableConfig = config
    for _ in 0..<iterations {
        encodeDequant(gateWeightBuffer, gateScaleBuffer, gateWeightDequantBuffer, config.intermediate, config.hidden)
        encodeDequant(upWeightBuffer, upScaleBuffer, upWeightDequantBuffer, config.intermediate, config.hidden)
        encodeDequant(downWeightBuffer, downScaleBuffer, downWeightDequantBuffer, config.hidden, config.intermediate)

        encodeMPPMatmulDispatch(
            encoder: encoder,
            pipeline: gatePipeline,
            aBuffer: inputBuffer,
            bBuffer: gateWeightDequantBuffer,
            cBuffer: gateBuffer,
            m: config.tokens,
            n: config.intermediate,
            k: config.hidden
        )
        encodeMPPMatmulDispatch(
            encoder: encoder,
            pipeline: gatePipeline,
            aBuffer: inputBuffer,
            bBuffer: upWeightDequantBuffer,
            cBuffer: upBuffer,
            m: config.tokens,
            n: config.intermediate,
            k: config.hidden
        )

        encoder.setComputePipelineState(swigluPipeline)
        encoder.setBuffer(gateBuffer, offset: 0, index: 0)
        encoder.setBuffer(upBuffer, offset: 0, index: 1)
        encoder.setBuffer(midBuffer, offset: 0, index: 2)
        encoder.setBytes(&mutableConfig, length: MemoryLayout<MLPConfig>.stride, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: Int(config.intermediate), height: Int(config.tokens), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
        )

        encodeMPPMatmulDispatch(
            encoder: encoder,
            pipeline: downPipeline,
            aBuffer: midBuffer,
            bBuffer: downWeightDequantBuffer,
            cBuffer: outputBuffer,
            m: config.tokens,
            n: config.hidden,
            k: config.intermediate
        )
    }
    encoder.endEncoding()

    let wallStart = DispatchTime.now().uptimeNanoseconds
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let wallEnd = DispatchTime.now().uptimeNanoseconds

    if let error = commandBuffer.error {
        throw ProbeError.gpuExecutionFailed("MXFP4 dequant MLP command buffer failed: \(error)")
    }
    guard commandBuffer.status == .completed else {
        throw ProbeError.gpuExecutionFailed("MXFP4 dequant MLP command buffer finished with status \(commandBuffer.status.rawValue).")
    }

    let wallSeconds = Double(wallEnd - wallStart) / 1_000_000_000.0
    return commandBuffer.gpuEndTime > commandBuffer.gpuStartTime
        ? commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        : wallSeconds
}

func encodeMatmul(
    pipeline: MTLComputePipelineState,
    queue: MTLCommandQueue,
    aBuffer: MTLBuffer,
    bBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    config: MatmulConfig,
    gridSize: MTLSize,
    threadsPerThreadgroup: MTLSize,
    iterations: Int
) throws {
    guard let commandBuffer = queue.makeCommandBuffer() else {
        throw ProbeError.missingCommandBuffer
    }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw ProbeError.missingComputeEncoder
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    encoder.setBuffer(outputBuffer, offset: 0, index: 2)
    var mutableConfig = config
    encoder.setBytes(&mutableConfig, length: MemoryLayout<MatmulConfig>.stride, index: 3)
    for _ in 0..<iterations {
        encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
    }
    encoder.endEncoding()
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()

    if let error = commandBuffer.error {
        throw ProbeError.gpuExecutionFailed("Warm-up matmul command buffer failed: \(error)")
    }
}

func encodeMLP(
    gatePipeline: MTLComputePipelineState,
    downPipeline: MTLComputePipelineState,
    queue: MTLCommandQueue,
    inputBuffer: MTLBuffer,
    gateWeightBuffer: MTLBuffer,
    upWeightBuffer: MTLBuffer,
    intermediateBuffer: MTLBuffer,
    downWeightBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    config: MLPConfig,
    iterations: Int
) throws -> Double {
    let threadsPerThreadgroup = MTLSize(width: 16, height: 16, depth: 1)
    let gateGrid = MTLSize(width: Int(config.intermediate), height: Int(config.tokens), depth: 1)
    let downGrid = MTLSize(width: Int(config.hidden), height: Int(config.tokens), depth: 1)

    guard let commandBuffer = queue.makeCommandBuffer() else {
        throw ProbeError.missingCommandBuffer
    }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw ProbeError.missingComputeEncoder
    }

    var mutableConfig = config
    for _ in 0..<iterations {
        encoder.setComputePipelineState(gatePipeline)
        encoder.setBuffer(inputBuffer, offset: 0, index: 0)
        encoder.setBuffer(gateWeightBuffer, offset: 0, index: 1)
        encoder.setBuffer(upWeightBuffer, offset: 0, index: 2)
        encoder.setBuffer(intermediateBuffer, offset: 0, index: 3)
        encoder.setBytes(&mutableConfig, length: MemoryLayout<MLPConfig>.stride, index: 4)
        encoder.dispatchThreads(gateGrid, threadsPerThreadgroup: threadsPerThreadgroup)

        encoder.setComputePipelineState(downPipeline)
        encoder.setBuffer(intermediateBuffer, offset: 0, index: 0)
        encoder.setBuffer(downWeightBuffer, offset: 0, index: 1)
        encoder.setBuffer(outputBuffer, offset: 0, index: 2)
        encoder.setBytes(&mutableConfig, length: MemoryLayout<MLPConfig>.stride, index: 3)
        encoder.dispatchThreads(downGrid, threadsPerThreadgroup: threadsPerThreadgroup)
    }
    encoder.endEncoding()

    let wallStart = DispatchTime.now().uptimeNanoseconds
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let wallEnd = DispatchTime.now().uptimeNanoseconds

    if let error = commandBuffer.error {
        throw ProbeError.gpuExecutionFailed("MLP command buffer failed: \(error)")
    }
    guard commandBuffer.status == .completed else {
        throw ProbeError.gpuExecutionFailed("MLP command buffer finished with status \(commandBuffer.status.rawValue).")
    }

    let wallSeconds = Double(wallEnd - wallStart) / 1_000_000_000.0
    return commandBuffer.gpuEndTime > commandBuffer.gpuStartTime
        ? commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        : wallSeconds
}

func encodeMPPMLP(
    gatePipeline: MTLComputePipelineState,
    downPipeline: MTLComputePipelineState,
    swigluPipeline: MTLComputePipelineState,
    queue: MTLCommandQueue,
    inputBuffer: MTLBuffer,
    gateWeightBuffer: MTLBuffer,
    upWeightBuffer: MTLBuffer,
    gateBuffer: MTLBuffer,
    upBuffer: MTLBuffer,
    midBuffer: MTLBuffer,
    downWeightBuffer: MTLBuffer,
    outputBuffer: MTLBuffer,
    config: MLPConfig,
    iterations: Int
) throws -> Double {
    guard let commandBuffer = queue.makeCommandBuffer() else {
        throw ProbeError.missingCommandBuffer
    }
    guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw ProbeError.missingComputeEncoder
    }

    var mutableConfig = config
    for _ in 0..<iterations {
        encodeMPPMatmulDispatch(
            encoder: encoder,
            pipeline: gatePipeline,
            aBuffer: inputBuffer,
            bBuffer: gateWeightBuffer,
            cBuffer: gateBuffer,
            m: config.tokens,
            n: config.intermediate,
            k: config.hidden
        )
        encodeMPPMatmulDispatch(
            encoder: encoder,
            pipeline: gatePipeline,
            aBuffer: inputBuffer,
            bBuffer: upWeightBuffer,
            cBuffer: upBuffer,
            m: config.tokens,
            n: config.intermediate,
            k: config.hidden
        )

        encoder.setComputePipelineState(swigluPipeline)
        encoder.setBuffer(gateBuffer, offset: 0, index: 0)
        encoder.setBuffer(upBuffer, offset: 0, index: 1)
        encoder.setBuffer(midBuffer, offset: 0, index: 2)
        encoder.setBytes(&mutableConfig, length: MemoryLayout<MLPConfig>.stride, index: 3)
        encoder.dispatchThreads(
            MTLSize(width: Int(config.intermediate), height: Int(config.tokens), depth: 1),
            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1)
        )

        encodeMPPMatmulDispatch(
            encoder: encoder,
            pipeline: downPipeline,
            aBuffer: midBuffer,
            bBuffer: downWeightBuffer,
            cBuffer: outputBuffer,
            m: config.tokens,
            n: config.hidden,
            k: config.intermediate
        )
    }

    encoder.endEncoding()

    let wallStart = DispatchTime.now().uptimeNanoseconds
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    let wallEnd = DispatchTime.now().uptimeNanoseconds

    if let error = commandBuffer.error {
        throw ProbeError.gpuExecutionFailed("MPP MLP command buffer failed: \(error)")
    }
    guard commandBuffer.status == .completed else {
        throw ProbeError.gpuExecutionFailed("MPP MLP command buffer finished with status \(commandBuffer.status.rawValue).")
    }

    let wallSeconds = Double(wallEnd - wallStart) / 1_000_000_000.0
    return commandBuffer.gpuEndTime > commandBuffer.gpuStartTime
        ? commandBuffer.gpuEndTime - commandBuffer.gpuStartTime
        : wallSeconds
}

func encodeMPPMatmulDispatch(
    encoder: MTLComputeCommandEncoder,
    pipeline: MTLComputePipelineState,
    aBuffer: MTLBuffer,
    bBuffer: MTLBuffer,
    cBuffer: MTLBuffer,
    m: UInt32,
    n: UInt32,
    k: UInt32
) {
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(aBuffer, offset: 0, index: 0)
    encoder.setBuffer(bBuffer, offset: 0, index: 1)
    encoder.setBuffer(cBuffer, offset: 0, index: 2)
    var mutableM = m
    var mutableN = n
    var mutableK = k
    encoder.setBytes(&mutableM, length: MemoryLayout<UInt32>.stride, index: 3)
    encoder.setBytes(&mutableN, length: MemoryLayout<UInt32>.stride, index: 4)
    encoder.setBytes(&mutableK, length: MemoryLayout<UInt32>.stride, index: 5)

    let grid = MTLSize(
        width: (Int(n) + 127) / 128,
        height: (Int(m) + 63) / 64,
        depth: 1
    )
    let threads = MTLSize(width: pipeline.threadExecutionWidth * 4, height: 1, depth: 1)
    encoder.dispatchThreadgroups(grid, threadsPerThreadgroup: threads)
}

func printMatmulBenchmarkResult(_ result: MatmulBenchmarkResult) {
    let macs = Double(result.rows) * Double(result.columns) * Double(result.depth) * Double(result.iterations)
    let tops = (macs * 2.0) / result.seconds / 1_000_000_000_000.0
    let msPerIteration = result.seconds * 1000.0 / Double(result.iterations)
    print("  \(result.name):")
    print("    \(String(format: "%.3f", msPerIteration)) ms/matmul, \(String(format: "%.3f", tops)) effective TOPS")
    print("    checksum: \(result.checksum)")
}

func printMLPBenchmarkResult(_ result: BenchmarkResult) {
    let macs = 3.0 * Double(result.rows) * Double(result.columns) * Double(result.depth) * Double(result.iterations)
    let tops = (macs * 2.0) / result.seconds / 1_000_000_000_000.0
    let msPerIteration = result.seconds * 1000.0 / Double(result.iterations)
    print("  \(result.name):")
    print("    \(String(format: "%.3f", msPerIteration)) ms/MLP, \(String(format: "%.3f", tops)) effective TOPS")
    print("    checksum: \(result.checksum)")
}

func printRelativeSlowdowns(_ results: [BenchmarkResult]) {
    guard let best = results.min(by: { $0.seconds / Double($0.iterations) < $1.seconds / Double($1.iterations) }) else {
        return
    }

    let bestSeconds = best.seconds / Double(best.iterations)
    print("  relative slowdown, best = \(best.name):")
    for result in results {
        let seconds = result.seconds / Double(result.iterations)
        print("    \(result.name): \(String(format: "%.2fx", seconds / bestSeconds))")
    }
}

func makePackedMatrixBytes(count: Int, seed: UInt32) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: count)
    var state = seed
    for index in bytes.indices {
        state = state &* 1_664_525 &+ 1_013_904_223 &+ UInt32(truncatingIfNeeded: index)
        bytes[index] = UInt8(truncatingIfNeeded: state >> 16)
    }
    return bytes
}

func makeFiniteFP8E4M3Bytes(count: Int, seed: UInt32) -> [UInt8] {
    makeBytePatternValues(
        count: count,
        seed: seed,
        pattern: [0x00, 0x20, 0x30, 0x38, 0x40, 0xb0, 0xb8, 0xc0, 0x58, 0xd8]
    )
}

func makeFiniteFP8E5M2Bytes(count: Int, seed: UInt32) -> [UInt8] {
    makeBytePatternValues(
        count: count,
        seed: seed,
        pattern: [0x00, 0x34, 0x38, 0x3c, 0x40, 0xb4, 0xb8, 0xbc, 0xc0, 0x64]
    )
}

func makeBytePatternValues(count: Int, seed: UInt32, pattern: [UInt8]) -> [UInt8] {
    var bytes = [UInt8](repeating: 0, count: count)
    var state = seed
    for index in bytes.indices {
        state = state &* 1_664_525 &+ 1_013_904_223 &+ UInt32(truncatingIfNeeded: index)
        bytes[index] = pattern[Int(state % UInt32(pattern.count))]
    }
    return bytes
}

func makeHalfMatrixValues(count: Int, seed: UInt32, scale: Float = 0.25) -> [Float16] {
    var values = [Float16](repeating: 0, count: count)
    var state = seed
    for index in values.indices {
        state = state &* 1_664_525 &+ 1_013_904_223 &+ UInt32(truncatingIfNeeded: index)
        let nibble = Int((state >> 28) & 0x0f)
        let signed = (nibble ^ 8) - 8
        values[index] = Float16(Float(signed) * scale)
    }
    return values
}

func makeInt8MatrixValues(count: Int, seed: UInt32) -> [Int8] {
    var values = [Int8](repeating: 0, count: count)
    var state = seed
    for index in values.indices {
        state = state &* 1_664_525 &+ 1_013_904_223 &+ UInt32(truncatingIfNeeded: index)
        let nibble = Int((state >> 28) & 0x0f)
        values[index] = Int8((nibble ^ 8) - 8)
    }
    return values
}

func sampleFloatChecksum(buffer: MTLBuffer, count: Int) -> String {
    let pointer = buffer.contents().bindMemory(to: Float.self, capacity: count)
    let step = max(1, count / 32)
    var sum = 0.0
    var samples = 0
    var index = 0
    while index < count {
        sum += Double(pointer[index])
        samples += 1
        index += step
    }
    return String(format: "%.6g over %d samples", sum, samples)
}

func sampleIntChecksum(buffer: MTLBuffer, count: Int) -> String {
    let pointer = buffer.contents().bindMemory(to: Int32.self, capacity: count)
    let step = max(1, count / 32)
    var sum: Int64 = 0
    var samples = 0
    var index = 0
    while index < count {
        sum += Int64(pointer[index])
        samples += 1
        index += step
    }
    return "\(sum) over \(samples) samples"
}

func sampleHalfRawChecksum(buffer: MTLBuffer, count: Int) -> String {
    let pointer = buffer.contents().bindMemory(to: UInt16.self, capacity: count)
    let step = max(1, count / 32)
    var sum: UInt64 = 0
    var samples = 0
    var index = 0
    while index < count {
        sum += UInt64(pointer[index])
        samples += 1
        index += step
    }
    return "raw16 sum \(sum) over \(samples) samples"
}

func intArgument(_ name: String, defaultValue: Int) -> Int {
    let prefix = name + "="
    for argument in cliArguments where argument.hasPrefix(prefix) {
        return Int(argument.dropFirst(prefix.count)) ?? defaultValue
    }
    return defaultValue
}

func formatBytes(_ byteCount: Int) -> String {
    let mib = Double(byteCount) / (1024.0 * 1024.0)
    return String(format: "%.2f MiB", mib)
}
