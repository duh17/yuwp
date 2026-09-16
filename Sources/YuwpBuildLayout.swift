import Foundation

enum YuwpBuildLayout {
    static func developmentExecutableCandidates(
        named name: String,
        repositoryRoot: URL
    ) -> [URL] {
        [
            repositoryRoot.appendingPathComponent(".build/out/Products/Release/\(name)"),
            repositoryRoot.appendingPathComponent(".build/out/Products/Debug/\(name)"),
            repositoryRoot.appendingPathComponent(".build/arm64-apple-macosx/release/\(name)"),
            repositoryRoot.appendingPathComponent(".build/arm64-apple-macosx/debug/\(name)"),
        ]
    }
}
