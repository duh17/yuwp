import Foundation
import NativeASR

#if YUWP_INTERNAL_DIAGNOSTICS
let internalDiagnosticsEnabled = true
#else
let internalDiagnosticsEnabled = false
#endif

public protocol ASRServing: BatchTranscriptionServing, AnyObject, Sendable {
    func create(language: String?) -> String
    func feed(_ sid: String, pcmData: Data) -> [String: Any]?
    func stop(_ sid: String) -> [String: Any]?
    func transcribeAudio(audio: [Float], language: String?, temperature: Float) throws -> TranscriptionResult
}

public struct ASRRouteContext: Sendable {
    public let manager: any ASRServing
    public let aligner: ForcedAligner?
    public let vad: SileroVAD?
    public let streamingModelName: String
    public let activeModelID: String?
    public let batchModelName: String?
    public let batchRetranscribeEnabled: Bool
    public let loadAudio: @Sendable (URL) throws -> [Float]
    public let log: @Sendable (String) -> Void

    public init(
        manager: any ASRServing,
        aligner: ForcedAligner?,
        vad: SileroVAD?,
        streamingModelName: String,
        activeModelID: String? = nil,
        batchModelName: String?,
        batchRetranscribeEnabled: Bool,
        loadAudio: @escaping @Sendable (URL) throws -> [Float],
        log: @escaping @Sendable (String) -> Void = { _ in }
    ) {
        self.manager = manager
        self.aligner = aligner
        self.vad = vad
        self.streamingModelName = streamingModelName
        self.activeModelID = activeModelID
        self.batchModelName = batchModelName
        self.batchRetranscribeEnabled = batchRetranscribeEnabled
        self.loadAudio = loadAudio
        self.log = log
    }
}

private let streamRoutePrefix = "/v1/audio/transcriptions/stream"
private let batchRoutePaths = Set(["/v1/audio/transcriptions", "/audio/transcriptions"])

private func normalizedPath(_ rawPath: String) -> String {
    rawPath.split(separator: "?").first.map(String.init) ?? rawPath
}

private func queryValue(named name: String, in rawPath: String) -> String? {
    guard let queryStart = rawPath.firstIndex(of: "?") else { return nil }
    let query = String(rawPath[rawPath.index(after: queryStart)...])
    var components = URLComponents()
    components.percentEncodedQuery = query
    return components.queryItems?.first(where: { $0.name == name })?.value
}

public func routeRequest(_ req: HTTPRequest, context: ASRRouteContext) -> HTTPResponse {
    let path = normalizedPath(req.path)

    if path == "/v1/info" {
        return handleInfoRoute(req, context: context)
    }

    if batchRoutePaths.contains(path) {
        return handleBatchTranscriptionRequest(req, context: context)
    }

    if path == streamRoutePrefix || path.hasPrefix(streamRoutePrefix + "/") {
        return handleStreamRoute(req, path: path, context: context)
    }

    return jsonResponse(status: 404, ["error": "unknown endpoint: \(req.method) \(path)"])
}

// MARK: - Route Handlers

private func handleInfoRoute(_ req: HTTPRequest, context: ASRRouteContext) -> HTTPResponse {
    guard req.method == "GET" else { return jsonResponse(status: 405, ["error": "method not allowed"]) }

    var info: [String: Any] = [
        "model": context.streamingModelName,
        "streaming_model": context.streamingModelName, // backward compatibility
        "sample_rate": ASRAudio.sampleRate,
        "chunk_sec": 1.75,
        "final_accuracy_pass_enabled": context.batchRetranscribeEnabled,
        "batch_retranscribe": context.batchRetranscribeEnabled, // backward compatibility
        "internal_diagnostics": internalDiagnosticsEnabled,
        "status": "ready",
    ]
    if let activeModelID = context.activeModelID {
        info["model_id"] = activeModelID
    }
    if let batchModelName = context.batchModelName,
       batchModelName != context.streamingModelName {
        info["final_accuracy_pass_model"] = batchModelName
        info["batch_model"] = batchModelName // backward compatibility
    }
    info["aligner"] = context.aligner != nil
    info["vad"] = context.vad != nil
    return jsonResponse(status: 200, info)
}

private func handleStreamRoute(_ req: HTTPRequest, path: String, context: ASRRouteContext) -> HTTPResponse {
    if path == streamRoutePrefix {
        guard req.method == "POST" else { return jsonResponse(status: 405, ["error": "method not allowed"]) }
        let requestedLanguage = queryValue(named: "language", in: req.path)
            .flatMap { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .flatMap { $0.isEmpty ? nil : $0 }
        return jsonResponse(status: 200, ["session_id": context.manager.create(language: requestedLanguage)])
    }

    let sid = String(path.dropFirst(streamRoutePrefix.count + 1))
    guard !sid.isEmpty else { return jsonResponse(status: 400, ["error": "missing session_id"]) }

    switch req.method {
    case "POST":
        guard let result = context.manager.feed(sid, pcmData: req.body) else {
            return jsonResponse(status: 404, ["error": "session not found"])
        }
        return jsonResponse(status: 200, result)
    case "DELETE":
        guard let result = context.manager.stop(sid) else {
            return jsonResponse(status: 404, ["error": "session not found"])
        }
        return jsonResponse(status: 200, result)
    default:
        return jsonResponse(status: 405, ["error": "method not allowed"])
    }
}

private func handleBatchTranscriptionRequest(_ req: HTTPRequest, context: ASRRouteContext) -> HTTPResponse {
    guard req.method == "POST" else { return jsonResponse(status: 405, ["error": "method not allowed"]) }

    let fields: [String: String]
    let filePart: MultipartPart
    switch parseMultipartUpload(req) {
    case .success(let parsedFields, let parsedFilePart):
        fields = parsedFields
        filePart = parsedFilePart
    case .failure(let response):
        return response
    }

    let requestedFormat = (fields["response_format"] ?? "json").lowercased()
    if (fields["stream"] ?? "false").lowercased() == "true" {
        return invalidRequestResponse("stream=true is not supported on this endpoint", param: "stream")
    }
    guard ["json", "text", "srt", "vtt"].contains(requestedFormat) else {
        return invalidRequestResponse(
            "Unsupported response_format '\(requestedFormat)'. Supported values: text, json, srt, vtt.",
            param: "response_format"
        )
    }

    if let model = fields["model"]?.trimmingCharacters(in: .whitespacesAndNewlines), !model.isEmpty {
        let requestedModel = model.lowercased()
        guard YuwpModelSupport.isSupportedPublicModelID(requestedModel) else {
            return invalidRequestResponse(
                "Unsupported model '\(model)'. Supported models: \(YuwpModelSupport.supportedPublicModelIDs.joined(separator: ", ")).",
                param: "model"
            )
        }
        if let activeModelID = context.activeModelID, requestedModel != activeModelID.lowercased() {
            return invalidRequestResponse(
                "Model '\(model)' is supported but not loaded on this server. Active model: \(activeModelID).",
                param: "model"
            )
        }
    }

    let language = fields["language"].flatMap { $0.isEmpty ? nil : $0 }
    let temperature = Float(fields["temperature"] ?? "0") ?? 0
    let includeDebug = parseDebugFlag(fields["debug"])

    do {
        let audio = try loadAudio(filePart: filePart, using: context.loadAudio)

        switch requestedFormat {
        case "text":
            let result = try BatchTranscriptionPipeline.transcribe(
                using: context.manager,
                audio: audio,
                language: language,
                temperature: temperature,
                vad: context.vad,
                log: context.log
            )
            return textResponse(status: 200, result.text)
        case "srt", "vtt":
            guard let aligner = context.aligner else {
                return jsonResponse(status: 501, ["error": "aligner model not loaded"])
            }
            let subtitleResult = try BatchTranscriptionPipeline.subtitle(
                using: context.manager,
                audio: audio,
                transcript: nil,
                language: language,
                temperature: temperature,
                aligner: aligner,
                vad: context.vad,
                log: context.log
            )
            return makeSubtitleResponse(
                result: subtitleResult,
                format: requestedFormat,
                includeDebug: includeDebug
            )
        default:
            if let aligner = context.aligner {
                let subtitleResult = try BatchTranscriptionPipeline.subtitle(
                    using: context.manager,
                    audio: audio,
                    transcript: nil,
                    language: language,
                    temperature: temperature,
                    aligner: aligner,
                    vad: context.vad,
                    log: context.log
                )
                return makeSubtitleResponse(
                    result: subtitleResult,
                    format: "json",
                    includeDebug: includeDebug
                )
            }

            let result = try BatchTranscriptionPipeline.transcribe(
                using: context.manager,
                audio: audio,
                language: language,
                temperature: temperature,
                vad: context.vad,
                log: context.log
            )
            var payload: [String: Any] = [
                "text": result.text,
                "duration": result.audioDuration,
            ]
            if let language = result.language {
                payload["language"] = normalizeLanguageCode(language) ?? language
            }
            return jsonResponse(status: 200, payload)
        }
    } catch {
        return jsonResponse(status: 422, ["error": error.localizedDescription])
    }
}

private enum MultipartUploadParseResult {
    case success(fields: [String: String], filePart: MultipartPart)
    case failure(HTTPResponse)
}

private func parseMultipartUpload(_ req: HTTPRequest) -> MultipartUploadParseResult {
    guard let contentType = req.headers["content-type"],
          contentType.lowercased().contains("multipart/form-data"),
          let parts = parseMultipartBody(body: req.body, contentTypeHeader: contentType)
    else {
        return .failure(jsonResponse(status: 415, ["error": "expected multipart/form-data upload"]))
    }

    let fields = formFields(parts)
    guard let filePart = parts.first(where: { $0.name == "file" }) else {
        return .failure(jsonResponse(status: 400, ["error": "missing file field"]))
    }

    return .success(fields: fields, filePart: filePart)
}

private func formFields(_ parts: [MultipartPart]) -> [String: String] {
    var fields: [String: String] = [:]
    for part in parts where part.filename == nil {
        if let value = part.textValue {
            fields[part.name] = value
        }
    }
    return fields
}

private func invalidRequestResponse(_ message: String, param: String? = nil, status: Int = 400) -> HTTPResponse {
    var error: [String: Any] = [
        "message": message,
        "type": "invalid_request_error",
    ]
    if let param { error["param"] = param }
    return jsonResponse(status: status, ["error": error])
}

private func parseDebugFlag(_ raw: String?) -> Bool {
    guard let raw else { return false }
    switch raw.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() {
    case "1", "true", "yes", "on":
        return true
    default:
        return false
    }
}

private func makeSubtitleResponse(
    result: BatchSubtitleResult,
    format: String,
    includeDebug: Bool = false
) -> HTTPResponse {
    if format == "text" {
        return textResponse(status: 200, result.transcript)
    }

    let subtitles = groupSubtitles(result.items, language: result.language)

    switch format {
    case "json":
        let body = formatSubtitleJSON(
            transcript: result.transcript,
            language: result.language,
            duration: result.audioDuration,
            subtitles: subtitles,
            debug: includeDebug ? result.debug : nil
        )
        return HTTPResponse(status: 200, contentType: "application/json", body: body)
    case "vtt":
        return textResponse(status: 200, formatVTT(subtitles), contentType: "text/vtt; charset=utf-8")
    default:
        return textResponse(status: 200, formatSRT(subtitles), contentType: "text/srt; charset=utf-8")
    }
}

private func loadAudio(
    filePart: MultipartPart,
    using loader: @Sendable (URL) throws -> [Float]
) throws -> [Float] {
    let filename = filePart.filename ?? inferredFilename(contentType: filePart.contentType)
    let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    let tempURL = tempDir.appendingPathComponent(sanitizeFilename(filename))
    try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
    try filePart.body.write(to: tempURL, options: [.atomic])
    defer { try? FileManager.default.removeItem(at: tempDir) }
    return try loader(tempURL)
}

