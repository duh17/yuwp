// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Yuwp",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "yuwp-asr", targets: ["yuwp_asr"]),
        .executable(name: "yuwp-tts", targets: ["yuwp_tts"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/ml-explore/mlx-swift",
            revision: "3b11207d4870fc2b703fc6c7931741aa196ec914"
        ),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", from: "3.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.1.6"),
        .package(url: "https://github.com/sparkle-project/Sparkle", from: "2.8.0"),
    ],
    targets: [
        .executableTarget(
            name: "Yuwp",
            dependencies: [
                .product(name: "Sparkle", package: "Sparkle"),
                "ASRIPC",
            ],
            path: "Sources",
            exclude: ["NativeASR", "NativeTTS", "YuwpHTTPServerSupport", "ASRServerSupport", "ASRIPC", "asr-test", "tts-test", "yuwp-tts", "asr-bench", "asr-stream-test", "swift-mlx-asr-server", "yuwp-asr", "align-test", "asr-stitch-debug"]
        ),
        .target(
            name: "YuwpHTTPServerSupport",
            path: "Sources/YuwpHTTPServerSupport"
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
        .target(
            name: "NativeTTS",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers"),
            ],
            path: "Sources/NativeTTS"
        ),
        .executableTarget(
            name: "asr-test",
            dependencies: ["NativeASR"],
            path: "Sources/asr-test"
        ),
        .executableTarget(
            name: "tts-test",
            dependencies: [
                "NativeTTS",
                .product(name: "MLX", package: "mlx-swift"),
            ],
            path: "Sources/tts-test"
        ),
        .executableTarget(
            name: "yuwp_tts",
            dependencies: [
                "NativeTTS",
                .product(name: "MLX", package: "mlx-swift"),
                "YuwpHTTPServerSupport",
            ],
            path: "Sources/yuwp-tts"
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
            name: "ASRIPC",
            path: "Sources/ASRIPC"
        ),
        .target(
            name: "ASRServerSupport",
            dependencies: ["NativeASR", "ASRIPC"],
            path: "Sources/ASRServerSupport"
        ),
        .executableTarget(
            name: "swift-mlx-asr-server",
            dependencies: ["NativeASR", "ASRServerSupport", "ASRIPC"],
            path: "Sources/swift-mlx-asr-server"
        ),
        .executableTarget(
            name: "yuwp_asr",
            dependencies: ["NativeASR", "ASRServerSupport"],
            path: "Sources/yuwp-asr"
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
            dependencies: ["Yuwp", "NativeASR", "ASRServerSupport", "ASRIPC"],
            path: "Tests",
            exclude: ["fixtures"]
        ),
    ]
)
