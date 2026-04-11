import Foundation
import NativeASR

public enum ASRServerLimits {
    public static let maxBodySize = 100 * 1024 * 1024
    public static let longAudioChunkThresholdSec = 10 * 60.0
    public static let longAudioChunkSec = 120.0
    public static let subtitleSimpleMaxAudioSec = 10 * 60.0
    public static let subtitleVADThresholdSec = 4 * 60.0
}

public struct HTTPRequest {
    public let method: String
    public let path: String
    public let headers: [String: String]
    public let body: Data

    public init(method: String, path: String, headers: [String: String], body: Data) {
        self.method = method
        self.path = path
        self.headers = headers
        self.body = body
    }
}

public struct HTTPResponse {
    public let status: Int
    public let contentType: String
    public let body: Data

    public init(status: Int, contentType: String, body: Data) {
        self.status = status
        self.contentType = contentType
        self.body = body
    }
}

public struct MultipartPart {
    public let name: String
    public let filename: String?
    public let contentType: String?
    public let body: Data

    public init(name: String, filename: String?, contentType: String?, body: Data) {
        self.name = name
        self.filename = filename
        self.contentType = contentType
        self.body = body
    }

    public var textValue: String? {
        String(data: body, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

public func parseMultipartBody(body: Data, contentTypeHeader: String) -> [MultipartPart]? {
    let boundaryPrefix = "boundary="
    guard let rawBoundary = contentTypeHeader
        .split(separator: ";")
        .map({ $0.trimmingCharacters(in: .whitespaces) })
        .first(where: { $0.hasPrefix(boundaryPrefix) })?
        .dropFirst(boundaryPrefix.count)
    else { return nil }

    let boundary = rawBoundary.trimmingCharacters(in: CharacterSet(charactersIn: "\""))
    let opening = Data("--\(boundary)\r\n".utf8)
    let nextBoundary = Data("\r\n--\(boundary)".utf8)
    let headerSeparator = Data("\r\n\r\n".utf8)
    guard body.starts(with: opening) else { return nil }

    var cursor = opening.endIndex
    var parts: [MultipartPart] = []

    while cursor <= body.endIndex {
        guard let headerRange = body.range(of: headerSeparator, in: cursor..<body.endIndex) else { return nil }
        let headerData = body[cursor..<headerRange.lowerBound]
        guard let headerText = String(data: headerData, encoding: .utf8) else { return nil }
        cursor = headerRange.upperBound

        guard let nextRange = body.range(of: nextBoundary, in: cursor..<body.endIndex) else { return nil }
        let partBody = Data(body[cursor..<nextRange.lowerBound])
        cursor = nextRange.upperBound

        var headers: [String: String] = [:]
        for line in headerText.split(separator: "\r\n", omittingEmptySubsequences: true) {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let key = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            headers[key] = value
        }

        guard let disposition = headers["content-disposition"] else { return nil }
        let dispositionParams = parseHeaderParameters(disposition)
        guard disposition.lowercased().hasPrefix("form-data"), let name = dispositionParams["name"] else { return nil }
        parts.append(MultipartPart(name: name, filename: dispositionParams["filename"], contentType: headers["content-type"], body: partBody))

        if body[cursor...].starts(with: Data("--".utf8)) {
            return parts
        }
        guard body[cursor...].starts(with: Data("\r\n".utf8)) else { return nil }
        cursor += 2
    }

    return nil
}

public func parseHeaderParameters(_ header: String) -> [String: String] {
    var out: [String: String] = [:]
    for segment in header.split(separator: ";").dropFirst() {
        let trimmed = segment.trimmingCharacters(in: .whitespaces)
        guard let eq = trimmed.firstIndex(of: "=") else { continue }
        let key = trimmed[..<eq].lowercased()
        let value = trimmed[trimmed.index(after: eq)...].trimmingCharacters(in: CharacterSet(charactersIn: "\""))
        out[String(key)] = value
    }
    return out
}

public func jsonResponse(status: Int, _ json: [String: Any]) -> HTTPResponse {
    let body = (try? JSONSerialization.data(withJSONObject: json)) ?? Data("{}".utf8)
    return HTTPResponse(status: status, contentType: "application/json", body: body)
}

public func textResponse(status: Int, _ text: String, contentType: String = "text/plain; charset=utf-8") -> HTTPResponse {
    HTTPResponse(status: status, contentType: contentType, body: Data(text.utf8))
}

public struct Subtitle: Equatable {
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
    let totalMs = Int(seconds * 1000)
    let ms = totalMs % 1000
    let s = (totalMs / 1000) % 60
    let m = (totalMs / 60000) % 60
    let h = totalMs / 3600000
    return String(format: "%02d:%02d:%02d%@%03d", h, m, s, separator, ms)
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

public func sanitizeFilename(_ filename: String) -> String {
    let trimmed = filename.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { return "upload.wav" }
    return URL(fileURLWithPath: trimmed).lastPathComponent
}

public func inferredFilename(contentType: String?) -> String {
    switch contentType?.lowercased() {
    case "audio/flac", "application/flac": return "upload.flac"
    case "audio/x-wav", "audio/wav", "audio/wave": return "upload.wav"
    case "audio/mpeg", "audio/mp3": return "upload.mp3"
    case "audio/mp4", "audio/m4a", "video/mp4": return "upload.m4a"
    default: return "upload.wav"
    }
}
