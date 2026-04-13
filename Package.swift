// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Yuwp",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "yuwp-asr", targets: ["yuwp_asr"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            revision: "3b11207d4870fc2b703fc6c7931741aa196ec914"
        ),
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.8.0"),
    ],
    targets: [
        .executableTarget(
            name: "Yuwp",
            dependencies: [
                .product(name: "Sparkle", package: "Sparkle"),
            ],
            path: "Sources",
            exclude: ["NativeASR", "ASRServerSupport", "asr-test", "asr-bench", "asr-stream-test", "asr-server", "yuwp-asr", "yuwp-transcribe", "align-test", "asr-stitch-debug"]
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
        .target(
            name: "ASRServerSupport",
            dependencies: ["NativeASR"],
            path: "Sources/ASRServerSupport"
        ),
        .executableTarget(
            name: "asr-server",
            dependencies: ["NativeASR", "ASRServerSupport"],
            path: "Sources/asr-server"
        ),
        .executableTarget(
            name: "yuwp_asr",
            dependencies: ["NativeASR", "ASRServerSupport"],
            path: "Sources/yuwp-asr"
        ),
        .executableTarget(
            name: "yuwp-transcribe",
            dependencies: ["NativeASR"],
            path: "Sources/yuwp-transcribe"
        ),
        .executableTarget(
            name: "align-test",
            dependencies: ["NativeASR"],
            path: "Sources/align-test"
        ),
        .executableTarget(
            name: "asr-stitch-debug",
            dependencies: ["NativeASR"],
            path: "Sources/asr-stitch-debug"
        ),
        .testTarget(
            name: "YuwpTests",
            dependencies: ["Yuwp", "NativeASR", "ASRServerSupport"],
            path: "Tests",
            exclude: ["fixtures"]
        ),
    ]
)
