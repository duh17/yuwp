import Foundation

public struct Subtitle: Equatable, Sendable, Codable {
    public let index: Int
    public let start: Double
    public let end: Double
    public let text: String

    public init(index: Int, start: Double, end: Double, text: String) {
        self.index = index
        self.start = start
        self.end = end
        self.text = text
    }
}

public func groupSubtitles(
    _ items: [ForcedAlignItem],
    maxWordsPerLine: Int = 8,
    maxDuration: Double = 5.0,
    pauseThreshold: Double = 0.5,
    sentenceEndChars: Set<Character> = [".", "!", "?", "。", "！", "？"]
) -> [Subtitle] {
    guard !items.isEmpty else { return [] }

    var subtitles: [Subtitle] = []
    var currentWords: [ForcedAlignItem] = []
    var subtitleIndex = 1

    func flush() {
        guard !currentWords.isEmpty else { return }
        let text = AlignedTextRenderer.render(tokens: currentWords.map(\.text))
        subtitles.append(Subtitle(
            index: subtitleIndex,
            start: currentWords.first!.startTime,
            end: currentWords.last!.endTime,
            text: text
        ))
        subtitleIndex += 1
        currentWords.removeAll()
    }

    for (index, item) in items.enumerated() {
        currentWords.append(item)

        let duration = item.endTime - (currentWords.first?.startTime ?? item.startTime)
        let atWordLimit = currentWords.count >= maxWordsPerLine
        let atDurationLimit = duration >= maxDuration
        let atSentenceEnd = item.text.last.map { sentenceEndChars.contains($0) } ?? false
        let hasPause = index + 1 < items.count && (items[index + 1].startTime - item.endTime) >= pauseThreshold

        if atWordLimit || atDurationLimit || atSentenceEnd || hasPause {
            flush()
        }
    }
    flush()
    return subtitles
}

private func formatTime(_ seconds: Double, separator: String = ",") -> String {
    let totalMs = max(0, Int((seconds * 1000).rounded()))
    let ms = totalMs % 1000
    let s = (totalMs / 1000) % 60
    let m = (totalMs / 60000) % 60
    let h = totalMs / 3600000
    return String(format: "%02d:%02d:%02d%@%03d", h, m, s, separator, ms)
}

private func formatLRCTime(_ seconds: Double) -> String {
    let totalCentiseconds = max(0, Int((seconds * 100).rounded()))
    let centiseconds = totalCentiseconds % 100
    let totalSeconds = totalCentiseconds / 100
    let secondsPart = totalSeconds % 60
    let minutesPart = totalSeconds / 60
    return String(format: "%02d:%02d.%02d", minutesPart, secondsPart, centiseconds)
}

public func formatSRT(_ subtitles: [Subtitle]) -> String {
    subtitles.map { sub in
        "\(sub.index)\n\(formatTime(sub.start)) --> \(formatTime(sub.end))\n\(sub.text)"
    }.joined(separator: "\n\n")
}

public func formatVTT(_ subtitles: [Subtitle]) -> String {
    "WEBVTT\n\n" + subtitles.map { sub in
        "\(formatTime(sub.start, separator: ".")) --> \(formatTime(sub.end, separator: "."))\n\(sub.text)"
    }.joined(separator: "\n\n")
}

public func formatLRC(_ subtitles: [Subtitle]) -> String {
    subtitles.map { sub in
        "[\(formatLRCTime(sub.start))]\(sub.text)"
    }.joined(separator: "\n")
}

public func normalizeLanguageCode(_ language: String?) -> String? {
    guard let language else { return nil }
    let trimmed = language.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }

    let normalized = trimmed
        .replacingOccurrences(of: "_", with: "-")
        .lowercased()

    let aliases: [String: String] = [
        "auto": "auto",
        "english": "en",
        "chinese": "zh",
        "mandarin": "zh",
        "cantonese": "yue",
        "japanese": "ja",
        "korean": "ko",
        "french": "fr",
        "german": "de",
        "spanish": "es",
        "portuguese": "pt",
        "russian": "ru",
        "arabic": "ar",
        "hindi": "hi",
        "thai": "th",
        "vietnamese": "vi",
        "indonesian": "id",
        "malay": "ms",
        "turkish": "tr",
        "italian": "it",
        "dutch": "nl",
        "polish": "pl",
        "ukrainian": "uk",
    ]
    if let alias = aliases[normalized] { return alias }

    let parts = normalized.split(separator: "-", omittingEmptySubsequences: true)
    guard let first = parts.first,
          (2...3).contains(first.count),
          first.allSatisfy(\.isLetter)
    else {
        return normalized
    }

    var output = [String(first)]
    for part in parts.dropFirst() {
        let piece = String(part)
        if piece.count == 2, piece.allSatisfy(\.isLetter) {
            output.append(piece.uppercased())
        } else {
            output.append(piece.lowercased())
        }
    }
    return output.joined(separator: "-")
}

public func formatSubtitleJSON(
    transcript: String,
    language: String,
    duration: Double,
    subtitles: [Subtitle]
) -> Data {
    let segments: [[String: Any]] = subtitles.map { sub in
        ["start": sub.start, "end": sub.end, "text": sub.text]
    }
    let payload: [String: Any] = [
        "text": transcript,
        "language": normalizeLanguageCode(language) ?? language,
        "duration": duration,
        "segments": segments,
    ]
    return (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
}
