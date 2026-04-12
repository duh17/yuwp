import Foundation
import NativeASR

#if YUWP_INTERNAL_DIAGNOSTICS
let internalDiagnosticsEnabled = true
#else
let internalDiagnosticsEnabled = false
#endif

private let vadLock = NSLock()
private let batchVADConfig = VADChunkingConfig(
    threshold: 0.6,
    minSpeechDuration: 0.25,
    minSilenceDuration: 0.08,
    speechPad: 0.02,
    splitMinSilenceDuration: 0.5,
    maxChunkDuration: ASRServerLimits.maxChunkSec,
    minChunkDuration: 30.0
)

private let batchEnergyConfig = EnergyChunkingConfig(
    maxChunkDuration: ASRServerLimits.maxChunkSec,
    minChunkDuration: 1.0,
    searchExpandDuration: 5.0,
    energyWindowDuration: 0.1,
    minProgressDuration: 1.0
)

public protocol ASRServing: AnyObject, Sendable {
    func create() -> String
    func feed(_ sid: String, pcmData: Data) -> [String: Any]?
    func stop(_ sid: String) -> [String: Any]?
    func transcribeAudio(audio: [Float], language: String?, temperature: Float) throws -> TranscriptionResult
    func transcribeChunk(audio: [Float], language: String?, temperature: Float) throws -> TranscriptionResult
    func subtitleItems(
        audio: [Float],
        transcript: String?,
        language: String?,
        temperature: Float,
        aligner: ForcedAligner
    ) throws -> (transcript: String, language: String, items: [ForcedAlignItem])
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

public func routeRequest(_ req: HTTPRequest, context: ASRRouteContext) -> HTTPResponse {
    let path = normalizedPath(req.path)

    if path == "/v1/info" {
        return handleInfoRoute(req, context: context)
    }

    if batchRoutePaths.contains(path) {
        return handleBatchTranscriptionRequest(req, context: context)
    }

    if path == "/v1/audio/subtitles" {
        return handleSubtitleRequest(req, context: context)
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
        "streaming_model": context.streamingModelName,
        "sample_rate": ASRAudio.sampleRate,
        "chunk_sec": 2.0,
        "batch_retranscribe": context.batchRetranscribeEnabled,
        "internal_diagnostics": internalDiagnosticsEnabled,
        "status": "ready",
    ]
    if let activeModelID = context.activeModelID {
        info["model"] = activeModelID
    }
    if let batchModelName = context.batchModelName {
        info["batch_model"] = batchModelName
    }
    info["aligner"] = context.aligner != nil
    info["vad"] = context.vad != nil
    return jsonResponse(status: 200, info)
}

private func handleStreamRoute(_ req: HTTPRequest, path: String, context: ASRRouteContext) -> HTTPResponse {
    if path == streamRoutePrefix {
        guard req.method == "POST" else { return jsonResponse(status: 405, ["error": "method not allowed"]) }
        return jsonResponse(status: 200, ["session_id": context.manager.create()])
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

    do {
        let audio = try loadAudio(filePart: filePart, using: context.loadAudio)
        let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)

        switch requestedFormat {
        case "text":
            let result = try transcribeChunked(
                manager: context.manager,
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
            let subtitleResult = try subtitleChunked(
                manager: context.manager,
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
                audioDuration: audioDuration,
                maxWordsPerLine: 8,
                maxDuration: 5.0,
                pauseThreshold: 0.5
            )
        default:
            if let aligner = context.aligner {
                let subtitleResult = try subtitleChunked(
                    manager: context.manager,
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
                    audioDuration: audioDuration,
                    maxWordsPerLine: 8,
                    maxDuration: 5.0,
                    pauseThreshold: 0.5
                )
            }

            let result = try transcribeChunked(
                manager: context.manager,
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

private func handleSubtitleRequest(_ req: HTTPRequest, context: ASRRouteContext) -> HTTPResponse {
    guard req.method == "POST" else { return jsonResponse(status: 405, ["error": "method not allowed"]) }
    guard let aligner = context.aligner else {
        return jsonResponse(status: 501, ["error": "aligner model not loaded"])
    }

    let fields: [String: String]
    let filePart: MultipartPart
    switch parseMultipartUpload(req) {
    case .success(let parsedFields, let parsedFilePart):
        fields = parsedFields
        filePart = parsedFilePart
    case .failure(let response):
        return response
    }

    let requestedLanguage = fields["language"].flatMap { $0.isEmpty ? nil : $0 }
    let responseFormat = (fields["response_format"] ?? "srt").lowercased()
    guard ["srt", "vtt", "json", "text"].contains(responseFormat) else {
        return jsonResponse(status: 400, ["error": "unsupported response_format: \(responseFormat). Use srt, vtt, json, or text"])
    }

    let maxWordsPerLine = Int(fields["max_words_per_line"] ?? "8") ?? 8
    let maxDuration = Double(fields["max_duration"] ?? "5.0") ?? 5.0
    let pauseThreshold = Double(fields["pause_threshold"] ?? "0.5") ?? 0.5
    let temperature = Float(fields["temperature"] ?? "0") ?? 0

    do {
        let audio = try loadAudio(filePart: filePart, using: context.loadAudio)
        let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
        let subtitleResult = try subtitleChunked(
            manager: context.manager,
            audio: audio,
            transcript: fields["text"],
            language: requestedLanguage,
            temperature: temperature,
            aligner: aligner,
            vad: context.vad,
            log: context.log
        )

        return makeSubtitleResponse(
            result: subtitleResult,
            format: responseFormat,
            audioDuration: audioDuration,
            maxWordsPerLine: maxWordsPerLine,
            maxDuration: maxDuration,
            pauseThreshold: pauseThreshold
        )
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

private func makeSubtitleResponse(
    result: (transcript: String, language: String, items: [ForcedAlignItem]),
    format: String,
    audioDuration: Double,
    maxWordsPerLine: Int,
    maxDuration: Double,
    pauseThreshold: Double
) -> HTTPResponse {
    if format == "text" {
        return textResponse(status: 200, result.transcript)
    }

    let subtitles = groupSubtitles(
        result.items,
        maxWordsPerLine: maxWordsPerLine,
        maxDuration: maxDuration,
        pauseThreshold: pauseThreshold
    )

    switch format {
    case "json":
        let body = formatSubtitleJSON(
            transcript: result.transcript,
            language: result.language,
            duration: audioDuration,
            subtitles: subtitles
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

private func formatVADChunkRanges(_ chunks: [VADAudioChunk]) -> String {
    chunks.enumerated().map { index, chunk in
        String(format: "%d:%.3f-%.3f", index + 1, chunk.startTime, chunk.endTime)
    }.joined(separator: ",")
}

private func chunkAudio(_ audio: [Float], vad: SileroVAD?) throws -> [AudioChunk] {
    if let vad {
        vadLock.lock()
        defer { vadLock.unlock() }
        return try vad.chunk(audio: audio, config: batchVADConfig)
    }

    return chunkAudioByEnergy(audio, sampleRate: ASRAudio.sampleRate, config: batchEnergyConfig)
}

private func transcribeChunked(
    manager: any ASRServing,
    audio: [Float],
    language: String?,
    temperature: Float,
    vad: SileroVAD?,
    log: @Sendable (String) -> Void
) throws -> TranscriptionResult {
    let startedAt = Date()
    let chunks = try chunkAudio(audio, vad: vad)
    let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
    let chunkMode = vad == nil ? "energy" : "VAD"
    log("\(chunkMode) chunking transcription: \(chunks.count) chunks from \(String(format: "%.1f", audioDuration))s")
    log("\(chunkMode) chunk ranges transcription: \(formatVADChunkRanges(chunks))")

    var texts: [String] = []
    var resolvedLanguage = language
    for chunk in chunks {
        let chunkResult = try manager.transcribeChunk(audio: chunk.audio, language: language, temperature: temperature)
        let trimmed = chunkResult.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { texts.append(trimmed) }
        if resolvedLanguage == nil { resolvedLanguage = chunkResult.language }
    }

    return TranscriptionResult(
        text: AlignedTextRenderer.render(segments: texts),
        language: resolvedLanguage,
        audioDuration: audioDuration,
        processingTime: Date().timeIntervalSince(startedAt)
    )
}

private func subtitleChunked(
    manager: any ASRServing,
    audio: [Float],
    transcript: String?,
    language: String?,
    temperature: Float,
    aligner: ForcedAligner,
    vad: SileroVAD?,
    log: @Sendable (String) -> Void
) throws -> (transcript: String, language: String, items: [ForcedAlignItem]) {
    let chunks = try chunkAudio(audio, vad: vad)
    let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
    let chunkMode = vad == nil ? "energy" : "VAD"
    log("\(chunkMode) chunking subtitles: \(chunks.count) chunks from \(String(format: "%.1f", audioDuration))s")
    log("\(chunkMode) chunk ranges subtitles: \(formatVADChunkRanges(chunks))")

    var allItems: [ForcedAlignItem] = []
    var transcriptParts: [String] = []
    var resolvedLanguage = language ?? "English"

    if let transcript, !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        let textParts = splitTextProportionally(transcript, chunkDurations: chunks.map(\.duration))
        for (chunk, textPart) in zip(chunks, textParts) {
            let part = try manager.subtitleItems(
                audio: chunk.audio,
                transcript: textPart,
                language: language,
                temperature: temperature,
                aligner: aligner
            )
            transcriptParts.append(part.transcript)
            resolvedLanguage = language ?? part.language
            allItems.append(contentsOf: offsetAlignmentItems(part.items, by: chunk.startTime))
        }
    } else {
        for chunk in chunks {
            let part = try manager.subtitleItems(
                audio: chunk.audio,
                transcript: nil,
                language: language,
                temperature: temperature,
                aligner: aligner
            )
            let trimmed = part.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { transcriptParts.append(trimmed) }
            resolvedLanguage = language ?? part.language
            allItems.append(contentsOf: offsetAlignmentItems(part.items, by: chunk.startTime))
        }
    }

    return (AlignedTextRenderer.render(segments: transcriptParts), resolvedLanguage, allItems)
}

func splitTextProportionally(_ text: String, chunkDurations: [Double]) -> [String] {
    guard !chunkDurations.isEmpty else { return [text] }
    let totalDuration = chunkDurations.reduce(0, +)
    guard totalDuration > 0 else { return [text] }

    let characters = Array(text)
    let totalCount = characters.count
    var parts: [String] = []
    var textPosition = 0

    for (index, duration) in chunkDurations.enumerated() {
        if index == chunkDurations.count - 1 {
            parts.append(String(characters[textPosition...]).trimmingCharacters(in: .whitespacesAndNewlines))
            break
        }

        let proportion = duration / totalDuration
        let charsForChunk = Int(Double(totalCount) * proportion)
        let endPosition = min(totalCount, textPosition + charsForChunk)
        let searchRange = max(20, Int(Double(charsForChunk) * 0.1))
        var bestPosition = endPosition
        let separators = Set(" 。．！？、，.!?,\n")

        outer: for offset in 0 ..< searchRange {
            for checkPosition in [endPosition + offset, endPosition - offset] {
                guard checkPosition >= 0, checkPosition < totalCount else { continue }
                if separators.contains(characters[checkPosition]) {
                    bestPosition = min(totalCount, checkPosition + 1)
                    break outer
                }
            }
        }

        if bestPosition <= textPosition {
            bestPosition = min(totalCount, endPosition)
        }
        parts.append(String(characters[textPosition ..< bestPosition]).trimmingCharacters(in: .whitespacesAndNewlines))
        textPosition = bestPosition
    }

    return parts
}

private func offsetAlignmentItems(_ items: [ForcedAlignItem], by offset: Double) -> [ForcedAlignItem] {
    items.map { item in
        ForcedAlignItem(text: item.text, startTime: item.startTime + offset, endTime: item.endTime + offset)
    }
}
