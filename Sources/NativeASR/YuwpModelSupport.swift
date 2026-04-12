import Foundation

public enum YuwpModelSupport {
    public static let yuwpDefaultsDomain = "com.yuwp.app"
    public static let defaultModelSpec = "mlx-community/Qwen3-ASR-0.6B-4bit"
    public static let defaultAlignerSpec = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"
    public static let supportedPublicModelIDs = ["qwen3-asr-0.6b", "qwen3-asr-1.7b"]

    public static func publicModelID(for spec: String?) -> String? {
        guard let spec else { return nil }
        let normalized = spec.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !normalized.isEmpty else { return nil }
        if normalized.contains("qwen3-asr-0.6b") { return "qwen3-asr-0.6b" }
        if normalized.contains("qwen3-asr-1.7b") { return "qwen3-asr-1.7b" }
        return nil
    }

    public static func isSupportedPublicModelID(_ modelID: String) -> Bool {
        supportedPublicModelIDs.contains(modelID.trimmingCharacters(in: .whitespacesAndNewlines).lowercased())
    }

    public static func resolveConfiguredModelURL(explicitSpec: String?) -> URL? {
        if let resolved = resolveModelSpec(explicitSpec) { return resolved }
        if let resolved = resolveModelSpec(defaultYuwpModelSpec()) { return resolved }
        return resolveModelSpec(defaultModelSpec)
    }

    public static func defaultAlignerURL() -> URL? {
        resolveModelSpec(defaultAlignerSpec)
    }

    public static func defaultYuwpModelSpec() -> String? {
        guard let domain = UserDefaults.standard.persistentDomain(forName: yuwpDefaultsDomain) else { return nil }
        let candidates = [
            domain["transcriptionModel"] as? String,
            domain["streamingModel"] as? String,
            domain["batchModel"] as? String,
        ]
        return candidates
            .compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }
            .first(where: { !$0.isEmpty })
    }

    public static func resolveModelSpec(_ spec: String?) -> URL? {
        guard let spec else { return nil }
        let trimmed = spec.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if looksLikePath(trimmed) {
            let expanded = NSString(string: trimmed).expandingTildeInPath
            let url = URL(fileURLWithPath: expanded).standardizedFileURL
            return isValidModelDirectory(url) ? url : nil
        }

        if isRepoId(trimmed) {
            let managed = managedModelDirectory(for: trimmed)
            if isValidModelDirectory(managed) { return managed }
            if let cached = huggingFaceSnapshot(for: trimmed) { return cached }
        }

        return nil
    }

    private static func looksLikePath(_ spec: String) -> Bool {
        spec.hasPrefix("/") || spec.hasPrefix("~") || spec.hasPrefix(".")
    }

    private static func isRepoId(_ spec: String) -> Bool {
        let parts = spec.split(separator: "/")
        return parts.count == 2 && !parts[0].isEmpty && !parts[1].isEmpty
    }

    private static func managedModelDirectory(for repoId: String) -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/Yuwp/models", isDirectory: true)
            .appendingPathComponent(repoId.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
    }

    private static func huggingFaceSnapshot(for repoId: String) -> URL? {
        guard isRepoId(repoId) else { return nil }
        let parts = repoId.split(separator: "/", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return nil }

        let roots = [
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub", isDirectory: true),
            FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches/huggingface/hub", isDirectory: true),
        ]

        for root in roots {
            let snapshotsDir = root
                .appendingPathComponent("models--\(parts[0])--\(parts[1])", isDirectory: true)
                .appendingPathComponent("snapshots", isDirectory: true)

            guard let snapshots = try? FileManager.default.contentsOfDirectory(
                at: snapshotsDir,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            ) else { continue }

            let sorted = snapshots.sorted {
                let left = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let right = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return left > right
            }

            if let match = sorted.first(where: isValidModelDirectory(_:)) {
                return match
            }
        }

        return nil
    }

    private static func isValidModelDirectory(_ directory: URL) -> Bool {
        let requiredFiles = ["config.json", "model.safetensors", "vocab.json", "merges.txt"]
        return requiredFiles.allSatisfy { FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }
}
