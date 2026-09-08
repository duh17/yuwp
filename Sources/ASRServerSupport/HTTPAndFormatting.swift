import Foundation
import NativeASR

public enum ASRServerLimits {
    public static let maxBodySize = 100 * 1024 * 1024
    public static let maxChunkSec = BatchTranscriptionDefaults.maxChunkDurationSec
}

public enum StreamContextualStringLimits {
    public static let maxPhraseCount = 100
    public static let maxPhraseUTF8ByteCount = 256
    public static let maxAggregateUTF8ByteCount = 8192
}

public enum StreamCreateBodyError: Error, Equatable, Sendable {
    case invalidJSON
    case bodyNotObject
    case streamConfigNotObject
    case contextualStringsNotArray
    case phraseNotString
    case emptyOrWhitespacePhrase
    case controlCharacters
    case tooManyPhrases
    case phraseTooLong
    case aggregateTooLong
}

public struct StreamCreateBody: Equatable, Sendable {
    public let contextualStrings: [String]

    public init(contextualStrings: [String]) {
        self.contextualStrings = contextualStrings
    }

    public var contextApplied: Bool { !contextualStrings.isEmpty }
}

/// Parse the optional stream-create JSON body.
/// Empty or whitespace-only bodies stay valid for existing clients.
/// Phrase strings are decoded with JSONDecoder so leading U+FEFF is preserved.
public func parseStreamCreateBody(_ data: Data) -> Result<StreamCreateBody, StreamCreateBodyError> {
    guard let text = String(data: data, encoding: .utf8) else {
        return data.isEmpty
            ? .success(StreamCreateBody(contextualStrings: []))
            : .failure(.invalidJSON)
    }
    // Payload presence only. Phrase blankness uses ASRContextualText, not this trim.
    if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        return .success(StreamCreateBody(contextualStrings: []))
    }

    let dto: StreamCreateRequestDTO
    do {
        dto = try JSONDecoder().decode(StreamCreateRequestDTO.self, from: data)
    } catch let error as StreamCreateBodyError {
        return .failure(error)
    } catch {
        return classifyStreamCreateDecodeError(data)
    }

    guard let phrases = dto.phrases else {
        return .success(StreamCreateBody(contextualStrings: []))
    }
    return validateContextualPhrases(phrases)
}

private func classifyStreamCreateDecodeError(_ data: Data) -> Result<StreamCreateBody, StreamCreateBodyError> {
    guard let parsed = try? JSONSerialization.jsonObject(with: data) else {
        return .failure(.invalidJSON)
    }
    if parsed is [String: Any] {
        return .failure(.invalidJSON)
    }
    return .failure(.bodyNotObject)
}

private func validateContextualPhrases(_ phrasesAny: [String]) -> Result<StreamCreateBody, StreamCreateBodyError> {
    if phrasesAny.count > StreamContextualStringLimits.maxPhraseCount {
        return .failure(.tooManyPhrases)
    }

    var phrases: [String] = []
    phrases.reserveCapacity(phrasesAny.count)
    var aggregateUTF8Bytes = 0
    for phrase in phrasesAny {
        // Hard limits and Cc checks run on the raw string before any trim.
        if phrase.unicodeScalars.contains(where: { $0.properties.generalCategory == .control }) {
            return .failure(.controlCharacters)
        }
        let utf8Bytes = phrase.utf8.count
        if utf8Bytes > StreamContextualStringLimits.maxPhraseUTF8ByteCount {
            return .failure(.phraseTooLong)
        }
        aggregateUTF8Bytes += utf8Bytes
        if aggregateUTF8Bytes > StreamContextualStringLimits.maxAggregateUTF8ByteCount {
            return .failure(.aggregateTooLong)
        }
        if ASRContextualText.isBlankPhrase(phrase) {
            return .failure(.emptyOrWhitespacePhrase)
        }
        phrases.append(phrase)
    }
    return .success(StreamCreateBody(contextualStrings: phrases))
}

private struct StreamCreateRequestDTO: Decodable {
    var phrases: [String]?

    enum CodingKeys: String, CodingKey {
        case streamConfig = "stream_config"
    }

    init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<CodingKeys>
        do {
            container = try decoder.container(keyedBy: CodingKeys.self)
        } catch {
            throw StreamCreateBodyError.bodyNotObject
        }

        guard container.contains(.streamConfig) else {
            phrases = nil
            return
        }
        if try container.decodeNil(forKey: .streamConfig) {
            throw StreamCreateBodyError.streamConfigNotObject
        }
        let config: StreamConfigDTO
        do {
            config = try container.decode(StreamConfigDTO.self, forKey: .streamConfig)
        } catch let error as StreamCreateBodyError {
            throw error
        } catch {
            throw StreamCreateBodyError.streamConfigNotObject
        }
        phrases = config.phrases
    }
}

private struct StreamConfigDTO: Decodable {
    var phrases: [String]?

    enum CodingKeys: String, CodingKey {
        case contextualStrings = "contextual_strings"
    }

    init(from decoder: Decoder) throws {
        let container: KeyedDecodingContainer<CodingKeys>
        do {
            container = try decoder.container(keyedBy: CodingKeys.self)
        } catch {
            throw StreamCreateBodyError.streamConfigNotObject
        }

        guard container.contains(.contextualStrings) else {
            phrases = nil
            return
        }
        if try container.decodeNil(forKey: .contextualStrings) {
            throw StreamCreateBodyError.contextualStringsNotArray
        }
        do {
            phrases = try container.decode([String].self, forKey: .contextualStrings)
        } catch {
            throw StreamCreateBodyError.contextualStringsNotArray
        }
    }
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
