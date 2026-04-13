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

public enum SubtitleFormattingDefaults {
    public static let maxWordsPerLine = 8
    public static let maxDuration = 5.0
    public static let pauseThreshold = 0.5
    public static let compactScriptMaxUnitsPerSubtitle = 12
    public static let compactScriptPauseThreshold = 0.6
    public static let sentenceEndChars: Set<Character> = [".", "!", "?", "。", "！", "？"]
    public static let compactScriptSentenceEndChars: Set<Character> = [".", "!", "?", "。", "！", "？", "；", "：", "…"]
}

public protocol SubtitleStitchingStrategy: Sendable {
    var id: String { get }
    var maxUnitsPerSubtitle: Int { get }
    var maxDuration: Double { get }
    var pauseThreshold: Double { get }
    var sentenceEndChars: Set<Character> { get }
    func unitCount(for item: ForcedAlignItem) -> Int
}

public struct WordSubtitleStitchingStrategy: SubtitleStitchingStrategy {
    public let id = "word"
    public let maxUnitsPerSubtitle: Int
    public let maxDuration: Double
    public let pauseThreshold: Double
    public let sentenceEndChars: Set<Character>

    public init(
        maxUnitsPerSubtitle: Int = SubtitleFormattingDefaults.maxWordsPerLine,
        maxDuration: Double = SubtitleFormattingDefaults.maxDuration,
        pauseThreshold: Double = SubtitleFormattingDefaults.pauseThreshold,
        sentenceEndChars: Set<Character> = SubtitleFormattingDefaults.sentenceEndChars
    ) {
        self.maxUnitsPerSubtitle = maxUnitsPerSubtitle
        self.maxDuration = maxDuration
        self.pauseThreshold = pauseThreshold
        self.sentenceEndChars = sentenceEndChars
    }

    public func unitCount(for item: ForcedAlignItem) -> Int { 1 }
}

public struct CompactScriptSubtitleStitchingStrategy: SubtitleStitchingStrategy {
    public let id = "compact-script"
    public let maxUnitsPerSubtitle: Int
    public let maxDuration: Double
    public let pauseThreshold: Double
    public let sentenceEndChars: Set<Character>

    public init(
        maxUnitsPerSubtitle: Int = SubtitleFormattingDefaults.compactScriptMaxUnitsPerSubtitle,
        maxDuration: Double = SubtitleFormattingDefaults.maxDuration,
        pauseThreshold: Double = SubtitleFormattingDefaults.compactScriptPauseThreshold,
        sentenceEndChars: Set<Character> = SubtitleFormattingDefaults.compactScriptSentenceEndChars
    ) {
        self.maxUnitsPerSubtitle = maxUnitsPerSubtitle
        self.maxDuration = maxDuration
        self.pauseThreshold = pauseThreshold
        self.sentenceEndChars = sentenceEndChars
    }

    public func unitCount(for item: ForcedAlignItem) -> Int { 1 }
}

public struct SubtitleStitchingRegistry: Sendable {
    public let fallbackStrategy: any SubtitleStitchingStrategy
    public let exactStrategies: [String: any SubtitleStitchingStrategy]

    public init(
        fallbackStrategy: any SubtitleStitchingStrategy = WordSubtitleStitchingStrategy(),
        exactStrategies: [String: any SubtitleStitchingStrategy] = [
            "zh": CompactScriptSubtitleStitchingStrategy(),
            "yue": CompactScriptSubtitleStitchingStrategy(),
            "ja": CompactScriptSubtitleStitchingStrategy(),
        ]
    ) {
        self.fallbackStrategy = fallbackStrategy
        self.exactStrategies = exactStrategies
    }

    public func strategy(for language: String?) -> any SubtitleStitchingStrategy {
        guard let normalized = normalizeLanguageCode(language)?.lowercased() else {
            return fallbackStrategy
        }
        return exactStrategies[normalized] ?? fallbackStrategy
    }

    public static let `default` = SubtitleStitchingRegistry()
}

public struct SubtitleJSONSegment: Sendable, Codable {
    public let start: Double
    public let end: Double
    public let text: String

    public init(start: Double, end: Double, text: String) {
        self.start = start
        self.end = end
        self.text = text
    }
}

public struct SubtitleDebugPayload: Sendable, Codable {
    public let text: String
    public let language: String?
    public let duration: Double
    public let segments: [SubtitleJSONSegment]?
    public let debug: BatchSubtitleDebug?
    public let processingTime: Double?
    public let rtf: Double?
    public let speedMultiplier: Double?

    public init(
        text: String,
        language: String?,
        duration: Double,
        segments: [SubtitleJSONSegment]? = nil,
        debug: BatchSubtitleDebug? = nil,
        processingTime: Double? = nil,
        rtf: Double? = nil,
        speedMultiplier: Double? = nil
    ) {
        self.text = text
        self.language = language
        self.duration = duration
        self.segments = segments
        self.debug = debug
        self.processingTime = processingTime
        self.rtf = rtf
        self.speedMultiplier = speedMultiplier
    }
}

public func groupSubtitles(
    _ items: [ForcedAlignItem],
    language: String?,
    registry: SubtitleStitchingRegistry = .default
) -> [Subtitle] {
    groupSubtitles(items, strategy: registry.strategy(for: language))
}

public func groupSubtitles(
    _ items: [ForcedAlignItem],
    maxWordsPerLine: Int = SubtitleFormattingDefaults.maxWordsPerLine,
    maxDuration: Double = SubtitleFormattingDefaults.maxDuration,
    pauseThreshold: Double = SubtitleFormattingDefaults.pauseThreshold,
    sentenceEndChars: Set<Character> = SubtitleFormattingDefaults.sentenceEndChars
) -> [Subtitle] {
    groupSubtitles(
        items,
        strategy: WordSubtitleStitchingStrategy(
            maxUnitsPerSubtitle: maxWordsPerLine,
            maxDuration: maxDuration,
            pauseThreshold: pauseThreshold,
            sentenceEndChars: sentenceEndChars
        )
    )
}

public func groupSubtitles(
    _ items: [ForcedAlignItem],
    strategy: any SubtitleStitchingStrategy
) -> [Subtitle] {
    guard !items.isEmpty else { return [] }

    var subtitles: [Subtitle] = []
    var currentWords: [ForcedAlignItem] = []
    var currentUnits = 0
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
        currentUnits = 0
    }

    for (index, item) in items.enumerated() {
        currentWords.append(item)
        currentUnits += max(1, strategy.unitCount(for: item))

        let duration = item.endTime - (currentWords.first?.startTime ?? item.startTime)
        let atUnitLimit = currentUnits >= strategy.maxUnitsPerSubtitle
        let atDurationLimit = duration >= strategy.maxDuration
        let atSentenceEnd = item.text.last.map { strategy.sentenceEndChars.contains($0) } ?? false
        let hasPause = index + 1 < items.count && (items[index + 1].startTime - item.endTime) >= strategy.pauseThreshold

        if atUnitLimit || atDurationLimit || atSentenceEnd || hasPause {
            flush()
        }
    }
    flush()
    return subtitles
}

public func subtitleItems(from debug: BatchSubtitleDebug) -> [ForcedAlignItem] {
    debug.chunks.flatMap { chunk in
        chunk.items.map { item in
            ForcedAlignItem(text: item.text, startTime: item.start, endTime: item.end, alignText: item.alignText)
        }
    }
}

public func restitchSubtitles(
    from payload: SubtitleDebugPayload,
    language: String? = nil,
    registry: SubtitleStitchingRegistry = .default
) -> [Subtitle] {
    guard let debug = payload.debug else {
        return (payload.segments ?? []).enumerated().map { index, segment in
            Subtitle(index: index + 1, start: segment.start, end: segment.end, text: segment.text)
        }
    }
    return groupSubtitles(subtitleItems(from: debug), language: language ?? payload.language, registry: registry)
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
    subtitles: [Subtitle],
    debug: BatchSubtitleDebug? = nil
) -> Data {
    let segments: [[String: Any]] = subtitles.map { sub in
        ["start": sub.start, "end": sub.end, "text": sub.text]
    }
    var payload: [String: Any] = [
        "text": transcript,
        "language": normalizeLanguageCode(language) ?? language,
        "duration": duration,
        "segments": segments,
    ]
    if let debug, let debugData = try? JSONEncoder().encode(debug), let debugJSON = try? JSONSerialization.jsonObject(with: debugData) {
        payload["debug"] = debugJSON
    }
    return (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
}
