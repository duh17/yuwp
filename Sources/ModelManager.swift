import Foundation

struct DownloadableASRModel: Sendable {
    let label: String
    let repoId: String

    static let supported: [DownloadableASRModel] = [
        .init(label: "0.6B 4-bit (fast)", repoId: "mlx-community/Qwen3-ASR-0.6B-4bit"),
        .init(label: "0.6B bf16", repoId: "mlx-community/Qwen3-ASR-0.6B-bf16"),
        .init(label: "1.7B 4-bit", repoId: "mlx-community/Qwen3-ASR-1.7B-4bit"),
        .init(label: "1.7B bf16 (accurate)", repoId: "mlx-community/Qwen3-ASR-1.7B-bf16"),
    ]
}

enum ModelLocator {
    static let requiredFiles = ["config.json", "model.safetensors", "vocab.json", "merges.txt"]
    static let optionalFiles = [
        "chat_template.json",
        "generation_config.json",
        "preprocessor_config.json",
        "tokenizer_config.json",
        "README.md",
    ]

    static func resolve(_ spec: String) -> URL? {
        let trimmed = spec.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let local = localDirectory(for: trimmed), isValidModelDirectory(local) {
            return local
        }
        if let managed = managedDirectoryIfExists(forRepoId: trimmed) {
            return managed
        }
        if let cached = huggingFaceCachedSnapshot(forRepoId: trimmed) {
            return cached
        }
        return nil
    }

    static func displayName(for spec: String) -> String {
        let trimmed = spec.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "Not set" }
        if let resolved = resolve(trimmed) {
            if isRepoId(trimmed) { return shortRepoName(trimmed) }
            return resolved.lastPathComponent
        }
        if isRepoId(trimmed) { return shortRepoName(trimmed) }
        let expanded = NSString(string: trimmed).expandingTildeInPath
        return URL(fileURLWithPath: expanded).lastPathComponent
    }

    static func localDirectory(for spec: String) -> URL? {
        guard looksLikePath(spec) else { return nil }
        let expanded = NSString(string: spec).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded).standardizedFileURL
        var isDir: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir), isDir.boolValue else {
            return nil
        }
        return url
    }

    static func managedRoot() -> URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Yuwp/models", isDirectory: true)
        try? FileManager.default.createDirectory(at: base, withIntermediateDirectories: true)
        return base
    }

    static func managedDirectory(forRepoId repoId: String) -> URL {
        managedRoot().appendingPathComponent(repoId.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
    }

    static func managedDirectoryIfExists(forRepoId repoId: String) -> URL? {
        guard isRepoId(repoId) else { return nil }
        let url = managedDirectory(forRepoId: repoId)
        return isValidModelDirectory(url) ? url : nil
    }

    static func huggingFaceCachedSnapshot(forRepoId repoId: String) -> URL? {
        guard isRepoId(repoId) else { return nil }
        let parts = repoId.split(separator: "/", maxSplits: 1).map(String.init)
        guard parts.count == 2 else { return nil }

        for root in huggingFaceHubRoots() {
            let snapshotsDir = root
                .appendingPathComponent("models--\(parts[0])--\(parts[1])", isDirectory: true)
                .appendingPathComponent("snapshots", isDirectory: true)
            guard let snapshots = try? FileManager.default.contentsOfDirectory(
                at: snapshotsDir,
                includingPropertiesForKeys: [.contentModificationDateKey],
                options: [.skipsHiddenFiles]
            ) else {
                continue
            }
            let sorted = snapshots.sorted {
                let l = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                let r = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
                return l > r
            }
            if let match = sorted.first(where: isValidModelDirectory(_:)) {
                return match
            }
        }
        return nil
    }

    static func missingFiles(in directory: URL) -> [String] {
        requiredFiles.filter { !FileManager.default.fileExists(atPath: directory.appendingPathComponent($0).path) }
    }

    static func isValidModelDirectory(_ directory: URL) -> Bool {
        missingFiles(in: directory).isEmpty
    }

    static func shortRepoName(_ repoId: String) -> String {
        repoId.split(separator: "/").last.map(String.init) ?? repoId
    }

    static func isRepoId(_ spec: String) -> Bool {
        let trimmed = spec.trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = trimmed.split(separator: "/")
        guard parts.count == 2 else { return false }
        return !parts[0].isEmpty && !parts[1].isEmpty && !looksLikePath(trimmed)
    }

    private static func looksLikePath(_ spec: String) -> Bool {
        spec.hasPrefix("/") || spec.hasPrefix("~") || spec.hasPrefix(".")
    }

    private static func huggingFaceHubRoots() -> [URL] {
        var roots: [URL] = []
        let env = ProcessInfo.processInfo.environment
        if let hfHome = env["HF_HOME"], !hfHome.isEmpty {
            roots.append(URL(fileURLWithPath: hfHome).appendingPathComponent("hub", isDirectory: true))
        }
        if let hubCache = env["HUGGINGFACE_HUB_CACHE"], !hubCache.isEmpty {
            roots.append(URL(fileURLWithPath: hubCache, isDirectory: true))
        }
        roots.append(URL(fileURLWithPath: NSString("~/.cache/huggingface/hub").expandingTildeInPath, isDirectory: true))
        var seen = Set<String>()
        return roots.filter { seen.insert($0.standardizedFileURL.path).inserted }
    }
}

enum ModelDownloadError: LocalizedError {
    case invalidRepoId(String)
    case apiFailed(String)
    case missingRequiredFiles([String])
    case httpStatus(Int, String)
    case filesystem(String)

    var errorDescription: String? {
        switch self {
        case .invalidRepoId(let repoId):
            return "Invalid Hugging Face repo id: \(repoId)"
        case .apiFailed(let msg):
            return msg
        case .missingRequiredFiles(let files):
            return "Model is missing required files: \(files.joined(separator: ", "))"
        case .httpStatus(let code, let path):
            return "Download failed (HTTP \(code)) for \(path)"
        case .filesystem(let msg):
            return msg
        }
    }
}

actor ModelDownloadManager {
    static let shared = ModelDownloadManager()

    private struct HubModelInfo: Decodable {
        let sha: String?
        let siblings: [HubSibling]?
    }

    private struct HubSibling: Decodable {
        let rfilename: String
    }

    func download(repoId: String, progress: @escaping @Sendable (Double, String) -> Void) async throws -> URL {
        guard ModelLocator.isRepoId(repoId) else {
            throw ModelDownloadError.invalidRepoId(repoId)
        }

        if let existing = ModelLocator.managedDirectoryIfExists(forRepoId: repoId) {
            progress(1.0, "Already downloaded")
            return existing
        }

        let info = try await fetchModelInfo(repoId: repoId)
        let files = selectFiles(from: info.siblings ?? [])
        let missingRequired = Set(ModelLocator.requiredFiles).subtracting(files)
        guard missingRequired.isEmpty else {
            throw ModelDownloadError.missingRequiredFiles(Array(missingRequired).sorted())
        }

        let revision = info.sha ?? "main"
        let fm = FileManager.default
        let destination = ModelLocator.managedDirectory(forRepoId: repoId)
        let parent = destination.deletingLastPathComponent()
        try fm.createDirectory(at: parent, withIntermediateDirectories: true)

        let temp = parent.appendingPathComponent(destination.lastPathComponent + ".partial-\(UUID().uuidString)", isDirectory: true)
        try? fm.removeItem(at: temp)
        try fm.createDirectory(at: temp, withIntermediateDirectories: true)

        do {
            for (index, relativePath) in files.enumerated() {
                progress(Double(index) / Double(max(files.count, 1)), "Downloading \(relativePath)…")
                let sourceURL = try makeResolveURL(repoId: repoId, revision: revision, relativePath: relativePath)
                let (tmpURL, response) = try await URLSession.shared.download(from: sourceURL)
                guard let http = response as? HTTPURLResponse else {
                    throw ModelDownloadError.apiFailed("No HTTP response from Hugging Face")
                }
                guard http.statusCode == 200 else {
                    throw ModelDownloadError.httpStatus(http.statusCode, relativePath)
                }

                let outURL = temp.appendingPathComponent(relativePath)
                try fm.createDirectory(at: outURL.deletingLastPathComponent(), withIntermediateDirectories: true)
                try? fm.removeItem(at: outURL)
                try fm.moveItem(at: tmpURL, to: outURL)
            }

            let metadata = [
                "repo_id": repoId,
                "revision": revision,
                "downloaded_at": ISO8601DateFormatter().string(from: Date()),
            ]
            let metadataData = try JSONSerialization.data(withJSONObject: metadata, options: [.prettyPrinted, .sortedKeys])
            try metadataData.write(to: temp.appendingPathComponent("yuwp-model.json"))

            let missingAfterDownload = ModelLocator.missingFiles(in: temp)
            guard missingAfterDownload.isEmpty else {
                throw ModelDownloadError.missingRequiredFiles(missingAfterDownload)
            }

            try? fm.removeItem(at: destination)
            try fm.moveItem(at: temp, to: destination)
            progress(1.0, "Download complete")
            return destination
        } catch {
            try? fm.removeItem(at: temp)
            throw error
        }
    }

    private func fetchModelInfo(repoId: String) async throws -> HubModelInfo {
        let encodedRepo = repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? repoId
        guard let url = URL(string: "https://huggingface.co/api/models/\(encodedRepo)") else {
            throw ModelDownloadError.apiFailed("Invalid Hugging Face API URL")
        }

        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse else {
            throw ModelDownloadError.apiFailed("No HTTP response from Hugging Face API")
        }
        guard http.statusCode == 200 else {
            throw ModelDownloadError.httpStatus(http.statusCode, repoId)
        }
        do {
            return try JSONDecoder().decode(HubModelInfo.self, from: data)
        } catch {
            throw ModelDownloadError.apiFailed("Failed to decode Hugging Face model metadata: \(error.localizedDescription)")
        }
    }

    private func selectFiles(from siblings: [HubSibling]) -> [String] {
        let wanted = Set(ModelLocator.requiredFiles + ModelLocator.optionalFiles)
        let available = Set(siblings.map(\.rfilename))
        let selected = available.intersection(wanted)
        return selected.sorted()
    }

    private func makeResolveURL(repoId: String, revision: String, relativePath: String) throws -> URL {
        let encodedRepo = repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? repoId
        let encodedRevision = revision.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? revision
        let encodedPath = relativePath
            .split(separator: "/")
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        guard let url = URL(string: "https://huggingface.co/\(encodedRepo)/resolve/\(encodedRevision)/\(encodedPath)?download=1") else {
            throw ModelDownloadError.apiFailed("Invalid download URL for \(relativePath)")
        }
        return url
    }
}
