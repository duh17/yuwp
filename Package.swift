// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "Yuwp",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "Yuwp",
            path: "Sources",
            exclude: ["sidecar"]
        ),
        .testTarget(
            name: "YuwpTests",
            dependencies: ["Yuwp"],
            path: "Tests",
            exclude: ["fixtures"]
        ),
    ]
)
