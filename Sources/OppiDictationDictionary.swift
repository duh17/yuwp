import Foundation

/// Reads Oppi's saved dictation dictionary so Yuwp.app hotkey sessions can
/// send the same `contextual_strings` Oppi server dictation already sends.
enum OppiDictationDictionary {
    static let maxPhraseCount = 100
    static let maxPhraseUTF8ByteCount = 256
    static let maxAggregateUTF8ByteCount = 8192
    /// Always-on Yuwp hints so hotkey dictation does not depend on Oppi.
    static let builtInPhrases = ["Yuwp", "Oppi"]

    static func globalPhrases(
        fileManager: FileManager = .default,
        environment: [String: String] = ProcessInfo.processInfo.environment
    ) -> [String] {
        clipped(
            builtInPhrases + phrases(from: dictionaryURL(environment: environment), fileManager: fileManager)
        )
    }

    static func phrases(from url: URL, fileManager: FileManager = .default) -> [String] {
        guard let data = fileManager.contents(atPath: url.path) else { return [] }
        guard let record = try? JSONDecoder().decode(Record.self, from: data) else { return [] }
        return clipped(record.global.phrases)
    }

    static func dictionaryURL(environment: [String: String] = ProcessInfo.processInfo.environment) -> URL {
        let root: URL
        if let override = environment["OPPI_DATA_DIR"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !override.isEmpty {
            root = URL(fileURLWithPath: override, isDirectory: true)
        } else {
            root = fileHome(environment: environment)
                .appendingPathComponent(".config/oppi", isDirectory: true)
        }
        return root.appendingPathComponent("settings/dictation-dictionary.json")
    }

    private static func fileHome(environment: [String: String]) -> URL {
        if let home = environment["HOME"], !home.isEmpty {
            return URL(fileURLWithPath: home, isDirectory: true)
        }
        return FileManager.default.homeDirectoryForCurrentUser
    }

    private static func clipped(_ phrases: [String]) -> [String] {
        var kept: [String] = []
        var seen = Set<String>()
        var aggregate = 0
        for phrase in phrases {
            let trimmed = phrase.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty, seen.insert(trimmed).inserted else { continue }
            let bytes = trimmed.utf8.count
            guard bytes <= maxPhraseUTF8ByteCount else { continue }
            guard aggregate + bytes <= maxAggregateUTF8ByteCount else { break }
            guard kept.count < maxPhraseCount else { break }
            kept.append(trimmed)
            aggregate += bytes
        }
        return kept
    }

    private struct Record: Decodable {
        var global: List
        struct List: Decodable {
            var phrases: [String]
        }
    }
}
