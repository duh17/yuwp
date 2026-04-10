// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Yuwp",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", branch: "main"),
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.8.0"),
    ],
    targets: [
        .executableTarget(
            name: "Yuwp",
            dependencies: [
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "Sources",
            exclude: ["NativeASR", "asr-test", "asr-bench", "asr-stream-test", "asr-server", "align-test"]
        ),
        .target(
            name: "NativeASR",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ],
            path: "Sources/NativeASR",
            resources: [
                .copy("Resources")
            ]
        ),
        .executableTarget(
            name: "asr-test",
            dependencies: ["NativeASR"],
            path: "Sources/asr-test"
        ),
        .executableTarget(
            name: "asr-bench",
            dependencies: ["NativeASR"],
            path: "Sources/asr-bench"
        ),
        .executableTarget(
            name: "asr-stream-test",
            dependencies: ["NativeASR"],
            path: "Sources/asr-stream-test"
        ),
        .executableTarget(
            name: "asr-server",
            dependencies: ["NativeASR"],
            path: "Sources/asr-server"
        ),
        .executableTarget(
            name: "align-test",
            dependencies: ["NativeASR"],
            path: "Sources/align-test"
        ),
        .testTarget(
            name: "YuwpTests",
            dependencies: ["Yuwp", "NativeASR"],
            path: "Tests",
            exclude: ["fixtures"]
        ),
    ]
)
