// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Yuwp",
    platforms: [.macOS(.v14)],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", branch: "main"),
    ],
    targets: [
        .executableTarget(
            name: "Yuwp",
            path: "Sources",
            exclude: ["NativeASR", "asr-test", "asr-bench", "asr-stream-test", "asr-server"]
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
            path: "Sources/NativeASR"
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
        .testTarget(
            name: "YuwpTests",
            dependencies: ["Yuwp", "NativeASR"],
            path: "Tests",
            exclude: ["fixtures"]
        ),
    ]
)
