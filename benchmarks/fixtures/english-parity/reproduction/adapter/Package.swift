// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "EnglishParityAdapter",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "fluid-parity-adapter", targets: ["FluidAdapter"])],
    dependencies: [.package(path: "/tmp/yuwp-english-parity/FluidAudio")],
    targets: [
        .target(name: "ASRIPC"),
        .target(name: "AdapterCore", dependencies: ["ASRIPC"]),
        .executableTarget(name: "FluidAdapter", dependencies: [
            "AdapterCore", "ASRIPC", .product(name: "FluidAudio", package: "FluidAudio")
        ]),
        .testTarget(name: "AdapterCoreTests", dependencies: ["AdapterCore", "ASRIPC"])
    ]
)
