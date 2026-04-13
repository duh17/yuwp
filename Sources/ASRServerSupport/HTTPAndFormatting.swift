import Foundation
import NativeASR

public enum ASRServerLimits {
    public static let maxBodySize = 100 * 1024 * 1024
    public static let maxChunkSec = BatchTranscriptionDefaults.maxChunkDurationSec
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
