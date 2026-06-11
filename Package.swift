// swift-tools-version: 6.4

import PackageDescription

let package = Package(
    name: "MetalFP41Probe",
    platforms: [
        .macOS(.v27)
    ],
    products: [
        .executable(name: "MetalFP41Probe", targets: ["MetalFP41Probe"])
    ],
    targets: [
        .executableTarget(
            name: "MetalFP41Probe",
            resources: [
                .copy("Shaders")
            ]
        )
    ]
)
