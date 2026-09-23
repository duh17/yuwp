import Foundation

/// Longest-stable-prefix commit for R2T2-style streaming.
///
/// `|` is a decoded-text delimiter, not a single token id. The committed
/// string is append-only: it never shrinks and never includes `|`.
public struct StablePrefixCommitter: Sendable {
    public struct Result: Equatable, Sendable {
        public var committedText: String
        public var unstableTail: String
        public var committedDelta: String
    }

    public static func commit(
        prefixText: String,
        generatedText: String,
        unfixedTokenCount: Int,
        encode: (String) -> [Int],
        decode: ([Int]) -> String
    ) -> Result {
        let prefix = prefixText
        guard !generatedText.isEmpty else {
            return Result(committedText: prefix, unstableTail: "", committedDelta: "")
        }

        let cut = Self.pipeCut(Self.stripMeta(generatedText))
        guard let hypothesis = hypothesis(prefix: prefix, cut: cut) else {
            return Result(committedText: prefix, unstableTail: "", committedDelta: "")
        }

        let tokens = encode(hypothesis)
        let unfixed = max(0, unfixedTokenCount)
        let fixedCount = tokens.count <= unfixed ? tokens.count : tokens.count - unfixed
        let fixedTokens = Array(tokens.prefix(fixedCount))
        let unstableTokens = Array(tokens.dropFirst(fixedCount))
        let fixed = decode(fixedTokens)
        let unstable = decode(unstableTokens)

        let committed: String
        if prefix.isEmpty {
            committed = fixed
        } else if fixed.hasPrefix(prefix) {
            committed = fixed
        } else {
            // Don't shrink, and don't replace a stable prefix with a divergent hypothesis.
            committed = prefix
        }

        let unstableTail = committed == prefix && !fixed.hasPrefix(prefix) ? "" : unstable
        let delta: String
        if committed.hasPrefix(prefix) {
            delta = String(committed.dropFirst(prefix.count))
        } else {
            delta = ""
        }
        return Result(committedText: committed, unstableTail: unstableTail, committedDelta: delta)
    }

    static func hypothesis(prefix: String, cut: String) -> String? {
        if prefix.isEmpty { return cut }
        if cut.isEmpty { return prefix }
        if cut.hasPrefix(prefix) { return cut }
        if isContinuation(prefix: prefix, cut: cut) { return prefix + cut }
        return nil
    }

    static func isContinuation(prefix: String, cut: String) -> Bool {
        guard let first = cut.first, let last = prefix.last else { return false }
        if first.isWhitespace || first.isPunctuation { return true }
        if isCJK(last) && isCJK(first) { return true }
        // Mixed dictation: English then 中文 without repeating the English prefix.
        if isCJK(first) { return true }
        return false
    }

    static func isCJK(_ character: Character) -> Bool {
        character.unicodeScalars.contains { scalar in
            (0x4E00...0x9FFF).contains(scalar.value)
                || (0x3400...0x4DBF).contains(scalar.value)
                || (0x3040...0x30FF).contains(scalar.value)
        }
    }

    /// Auto-language mode may emit `language English` before `<asr_text>`.
    /// That header must not become the committed transcript.
    public static func isLanguageHeaderOnly(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.hasPrefix("language") else { return false }
        return pipeCut(stripMeta(trimmed)).isEmpty
    }

    /// Displayed transcript after `|` / header strip. Never shrinks `prefixText`
    /// once that prefix is already real content.
    public static func visibleTranscript(prefixText: String, decoded: String) -> String {
        let previous = pipeCut(stripMeta(prefixText))
        let cut = pipeCut(stripMeta(decoded))
        if previous.isEmpty { return cut }
        if cut.hasPrefix(previous) { return cut }
        if isContinuation(prefix: previous, cut: cut) { return previous + cut }
        return previous
    }

    static func stripMeta(_ text: String) -> String {
        var cleaned = text
        if let range = cleaned.range(of: "<asr_text>") {
            let before = String(cleaned[..<range.lowerBound])
            let after = String(cleaned[range.upperBound...])
            if let header = before.range(of: "language ", options: [.backwards, .caseInsensitive]) {
                cleaned = String(before[..<header.lowerBound]) + after
            } else {
                cleaned = after
            }
        }
        if let replacement = cleaned.firstIndex(of: "\u{FFFD}") {
            cleaned = String(cleaned[..<replacement])
        }
        return stripLanguageHeader(cleaned)
    }

    /// `language NoneSo what` has no `<asr_text>` marker and no space after
    /// `None`. Strip a leading `language <name>` even when glued to content.
    /// Leave ordinary sentences that start with the word "language" alone.
    static func stripLanguageHeader(_ text: String) -> String {
        if text == "language" { return "" }
        guard text.hasPrefix("language ") else { return text }
        let rest = text.dropFirst("language ".count)
        guard let name = headerLanguageNames.first(where: { rest.hasPrefix($0) }) else {
            return text
        }
        return String(rest.dropFirst(name.count))
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static let headerLanguageNames: [String] = [
        "None",
        "Cantonese", "Portuguese", "Indonesian", "Vietnamese", "Macedonian",
        "Chinese", "English", "Arabic", "German", "French", "Spanish",
        "Italian", "Korean", "Russian", "Thai", "Japanese", "Turkish",
        "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish", "Polish",
        "Czech", "Filipino", "Persian", "Greek", "Romanian", "Hungarian",
    ].sorted { $0.count > $1.count }

    static func pipeCut(_ text: String) -> String {
        guard let index = text.firstIndex(of: "|") else { return text }
        return String(text[..<index])
    }
}
