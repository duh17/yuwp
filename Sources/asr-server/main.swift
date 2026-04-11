// asr-server — Native streaming ASR HTTP server.
// Serves Yuwp (spawned as child) and any external HTTP client.
//
// Endpoints:
//   GET    /v1/info                                → server status
//   POST   /v1/audio/transcriptions                → OpenAI-style batch transcription
//   POST   /audio/transcriptions                   → OpenAI-style batch transcription alias
//   POST   /v1/audio/transcriptions/stream         → create session
//   POST   /v1/audio/transcriptions/stream/:id     → feed audio (raw s16le PCM)
//   DELETE /v1/audio/transcriptions/stream/:id     → stop session, get final text
//   POST   /v1/audio/subtitles                     → word-level alignment → SRT/VTT/JSON
//
// Usage: asr-server <streaming-model-dir> [--batch-model <dir>] [--aligner-model <dir>]
//                   [--disable-batch-retranscribe] [--port 9748] [--host 127.0.0.1] [--warmup]

#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import Foundation
import NativeASR

// MARK: - Shutdown State

nonisolated(unsafe) var serverSocket: Int32 = -1
nonisolated(unsafe) var shuttingDown = false
nonisolated(unsafe) var parentWatchTimer: DispatchSourceTimer?
private let vadLock = NSLock()

#if YUWP_INTERNAL_DIAGNOSTICS
let internalDiagnosticsEnabled = true
#else
let internalDiagnosticsEnabled = false
#endif

func handleShutdown(_: Int32) {
    guard !shuttingDown else { return }
    shuttingDown = true
    parentWatchTimer?.cancel()
    parentWatchTimer = nil
    let fd = serverSocket
    serverSocket = -1
    if fd >= 0 {
        Darwin.shutdown(fd, SHUT_RDWR)
        close(fd)
    }
}

func startParentWatch(expectedParentPID: Int32?) {
    guard let expectedParentPID else { return }
    let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .background))
    timer.schedule(deadline: .now() + .seconds(1), repeating: .seconds(1))
    timer.setEventHandler {
        let currentParentPID = getppid()
        guard currentParentPID == expectedParentPID else {
            log("Parent process \(expectedParentPID) disappeared (current ppid: \(currentParentPID)) — shutting down")
            handleShutdown(SIGTERM)
            return
        }
    }
    parentWatchTimer = timer
    timer.resume()
}

// MARK: - Session Manager

final class SessionManager: @unchecked Sendable {
    private let transcriber: Qwen3ASRTranscriber
    private let batchTranscriber: Qwen3ASRTranscriber?
    private let batchRetranscribeEnabled: Bool
    private var sessions: [String: StreamingSession] = [:]
    private var pendingAudio: [String: [Float]] = [:]
    private var lastActivity: [String: Date] = [:]
    private let stateLock = NSLock()
    private let inferenceLock = NSLock()  // MLX is single-threaded
    private let chunkSamples: Int
    private let sessionTimeout: TimeInterval = 300

    init(
        transcriber: Qwen3ASRTranscriber,
        batchTranscriber: Qwen3ASRTranscriber? = nil,
        batchRetranscribeEnabled: Bool = true,
        chunkSec: Double = 2.0
    ) {
        self.transcriber = transcriber
        self.batchTranscriber = batchTranscriber
        self.batchRetranscribeEnabled = batchRetranscribeEnabled
        self.chunkSamples = Int(chunkSec * Double(ASRAudio.sampleRate))
        // Cleanup timer
        DispatchQueue.global().async { [weak self] in
            while true {
                Thread.sleep(forTimeInterval: 30)
                self?.cleanupExpired()
            }
        }
    }

    func create() -> String {
        let sid = UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12).lowercased()
        let session = StreamingSession(
            transcriber: transcriber,
            batchTranscriber: batchTranscriber,
            config: StreamConfig(batchRetranscribe: batchRetranscribeEnabled)
        )
        stateLock.lock()
        sessions[String(sid)] = session
        pendingAudio[String(sid)] = []
        lastActivity[String(sid)] = Date()
        stateLock.unlock()
        log("Session created: \(sid)")
        return String(sid)
    }

    func feed(_ sid: String, pcmData: Data) -> [String: Any]? {
        stateLock.lock()
        guard let session = sessions[sid] else { stateLock.unlock(); return nil }
        var pending = pendingAudio[sid] ?? []
        pendingAudio[sid] = []  // Take ownership — prevents concurrent feed races
        lastActivity[sid] = Date()
        stateLock.unlock()

        // s16le PCM → Float32 (no lock needed, pure conversion)
        let samples = pcmData.withUnsafeBytes { buf -> [Float] in
            let int16s = buf.bindMemory(to: Int16.self)
            return int16s.map { Float($0) / 32768.0 }
        }
        pending.append(contentsOf: samples)

        // Process full chunks — serialized (MLX single-threaded)
        var batchCorrected = false
        inferenceLock.lock()
        while pending.count >= chunkSamples {
            let chunk = Array(pending.prefix(chunkSamples))
            pending = Array(pending.dropFirst(chunkSamples))
            let result = session.processChunk(chunk)
            if result.batchCorrected { batchCorrected = true }
#if YUWP_INTERNAL_DIAGNOSTICS
            log(
                "PERF sid=\(sid) chunk=\(session.processedChunkCount) "
                    + "samples=\(chunk.count) "
                    + "encode_ms=\(Int(result.encodeMs.rounded())) "
                    + "prefill_ms=\(Int(result.prefillMs.rounded())) "
                    + "decode_ms=\(Int(result.decodeMs.rounded())) "
                    + "total_ms=\(Int(result.totalMs.rounded())) "
                    + "reuse_pct=\(Int(result.reusePct.rounded())) "
                    + "text_len=\(result.text.count) "
                    + "batch_corrected=\(result.batchCorrected ? 1 : 0)"
            )
#endif
        }

        let response = transcriptPayload(
            session: session,
            kind: batchCorrected ? "segment_commit" : "partial",
            isFinal: false,
            batchCorrected: batchCorrected
        )
        inferenceLock.unlock()

        stateLock.lock()
        if sessions[sid] != nil {
            // Prepend remainder to any audio that arrived during inference
            pendingAudio[sid] = pending + (pendingAudio[sid] ?? [])
        }
        stateLock.unlock()

        return response
    }

    func stop(_ sid: String) -> [String: Any]? {
        stateLock.lock()
        guard let session = sessions.removeValue(forKey: sid) else { stateLock.unlock(); return nil }
        let pending = pendingAudio.removeValue(forKey: sid) ?? []
        lastActivity.removeValue(forKey: sid)
        stateLock.unlock()

        // Flush remaining audio under inference lock
        inferenceLock.lock()
        if !pending.isEmpty { _ = session.processChunk(pending) }
        let text = session.finalize()
        let response = transcriptPayload(
            session: session,
            kind: "final",
            isFinal: true,
            batchCorrected: false
        )
        inferenceLock.unlock()

        log("Session stopped (\(sid)): \(text.count) chars")
        return response
    }

    private func transcriptPayload(
        session: StreamingSession,
        kind: String,
        isFinal: Bool,
        batchCorrected: Bool
    ) -> [String: Any] {
        var resp: [String: Any] = [
            "text": session.finalText(),
            "committed_text": session.committedSegmentText(),
            "active_text": session.activeSegmentText(),
            "update_kind": kind,
            "is_final": isFinal,
        ]
        if batchCorrected { resp["batch_corrected"] = true }
        return resp
    }

    func transcribeFile(
        data: Data,
        filename: String,
        language: String? = nil,
        temperature: Float = 0.0
    ) throws -> TranscriptionResult {
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent(sanitizedFilename(filename))
        try FileManager.default.createDirectory(at: tempURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: tempURL, options: [.atomic])
        defer {
            try? FileManager.default.removeItem(at: tempURL.deletingLastPathComponent())
        }

        let audio = try loadAudioFile(tempURL)
        return try transcribeAudio(audio: audio, language: language, temperature: temperature)
    }

    func transcribeAudio(
        audio: [Float],
        language: String? = nil,
        temperature: Float = 0.0
    ) throws -> TranscriptionResult {
        let transcriber = batchTranscriber ?? self.transcriber
        let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)

        inferenceLock.lock()
        defer { inferenceLock.unlock() }

        if audioDuration <= longAudioChunkThresholdSec {
            return try transcriber.transcribe(audio: audio, language: language, temperature: temperature)
        }

        return try transcribeLongAudioLocked(
            audio: audio,
            language: language,
            temperature: temperature,
            transcriber: transcriber,
            audioDuration: audioDuration
        )
    }

    private func transcribeLongAudioLocked(
        audio: [Float],
        language: String?,
        temperature: Float,
        transcriber: Qwen3ASRTranscriber,
        audioDuration: Double
    ) throws -> TranscriptionResult {
        let startedAt = Date()
        let chunkSamples = Int(longAudioChunkSec * Double(ASRAudio.sampleRate))
        let totalChunks = (audio.count + chunkSamples - 1) / chunkSamples
        var texts: [String] = []
        var detectedLanguage: String?
        var offset = 0

        log(
            "Long batch transcription: \(String(format: "%.1f", audioDuration))s "
                + "audio -> \(totalChunks) fixed chunks of \(Int(longAudioChunkSec))s"
        )

        while offset < audio.count {
            let end = min(offset + chunkSamples, audio.count)
            let chunk = Array(audio[offset..<end])
            let result = try transcriber.transcribe(audio: chunk, language: language, temperature: temperature)
            let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty { texts.append(text) }
            if detectedLanguage == nil { detectedLanguage = result.language }
            offset = end
        }

        return TranscriptionResult(
            text: AlignedTextRenderer.render(segments: texts),
            language: language ?? detectedLanguage,
            audioDuration: audioDuration,
            processingTime: Date().timeIntervalSince(startedAt)
        )
    }

    func subtitleItems(
        audio: [Float],
        transcript: String?,
        language: String?,
        temperature: Float = 0.0,
        aligner: ForcedAligner
    ) throws -> (transcript: String, language: String, items: [ForcedAlignItem]) {
        let trimmedTranscript = transcript?.trimmingCharacters(in: .whitespacesAndNewlines)

        inferenceLock.lock()
        defer { inferenceLock.unlock() }

        let resolvedTranscript: String
        let resolvedLanguage: String
        if let trimmedTranscript, !trimmedTranscript.isEmpty {
            resolvedTranscript = trimmedTranscript
            resolvedLanguage = language ?? "English"
        } else {
            let transcriber = batchTranscriber ?? self.transcriber
            let result = try transcriber.transcribe(audio: audio, language: language, temperature: temperature)
            resolvedTranscript = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            resolvedLanguage = language ?? result.language ?? "English"
        }

        guard !resolvedTranscript.isEmpty else {
            return (resolvedTranscript, resolvedLanguage, [])
        }

        let items = aligner.align(audio: audio, text: resolvedTranscript, language: resolvedLanguage)
        return (resolvedTranscript, resolvedLanguage, items)
    }

    private func cleanupExpired() {
        let now = Date()
        stateLock.lock()
        let expired = lastActivity.filter { now.timeIntervalSince($0.value) > sessionTimeout }.map(\.key)
        for sid in expired {
            sessions.removeValue(forKey: sid)
            pendingAudio.removeValue(forKey: sid)
            lastActivity.removeValue(forKey: sid)
        }
        stateLock.unlock()
        if !expired.isEmpty { log("Expired \(expired.count) session(s)") }
    }
}

// MARK: - HTTP Parser (minimal, no deps)

private let maxBodySize = 100 * 1024 * 1024  // 100MB
private let longAudioChunkThresholdSec = 10 * 60.0
private let longAudioChunkSec = 120.0
private let subtitleSimpleMaxAudioSec = 10 * 60.0
private let subtitleVADThresholdSec = 4 * 60.0
private let longAudioVADConfig = VADChunkingConfig(
    threshold: 0.6,
    minSpeechDuration: 0.25,
    minSilenceDuration: 0.08,
    speechPad: 0.02,
    splitMinSilenceDuration: 0.5,
    maxChunkDuration: 120.0,
    minChunkDuration: 30.0
)

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data
}

struct HTTPResponse {
    let status: Int
    let contentType: String
    let body: Data
}

struct MultipartPart {
    let name: String
    let filename: String?
    let contentType: String?
    let body: Data

    var textValue: String? {
        String(data: body, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

func readHTTPRequest(fd: Int32) -> HTTPRequest? {
    // Recv timeout — don't block forever on dead connections
    var timeout = timeval(tv_sec: 30, tv_usec: 0)
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))

    var headerBuf = Data()
    var readBuf = [UInt8](repeating: 0, count: 8192)
    var headerEnd = -1

    // Read until \r\n\r\n (end of headers)
    while headerEnd < 0 {
        let n = recv(fd, &readBuf, readBuf.count, 0)
        if n == 0 { return nil }  // Client closed
        if n < 0 {
            if errno == EINTR { continue }
            return nil  // Timeout or error
        }
        headerBuf.append(contentsOf: readBuf[..<n])
        if let range = headerBuf.range(of: Data([0x0D, 0x0A, 0x0D, 0x0A])) {
            headerEnd = range.upperBound
        }
        if headerBuf.count > 65536 { return nil }  // Header too large
    }

    guard let headerStr = String(data: headerBuf[..<headerEnd], encoding: .utf8) else { return nil }
    let lines = headerStr.split(separator: "\r\n", omittingEmptySubsequences: false)
    guard let reqLine = lines.first else { return nil }
    let parts = reqLine.split(separator: " ", maxSplits: 2)
    guard parts.count >= 2 else { return nil }  // Malformed request line

    var headers: [String: String] = [:]
    for line in lines.dropFirst() {
        guard let colon = line.firstIndex(of: ":") else { continue }
        let key = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
        let val = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        headers[key] = val
    }

    // Read body (capped at maxBodySize)
    let contentLength = Int(headers["content-length"] ?? "0") ?? 0
    guard contentLength >= 0 else { return nil }
    if contentLength > maxBodySize {
        log("Rejected: body too large (\(contentLength) bytes)")
        writeResponse(
            fd: fd,
            response: writeJSONResponse(
                status: 413,
                ["error": "request body too large: \(contentLength) bytes (max \(maxBodySize))"]
            )
        )
        return nil
    }

    var body = Data(headerBuf[headerEnd...])
    while body.count < contentLength {
        let remain = contentLength - body.count
        let n = recv(fd, &readBuf, min(readBuf.count, remain), 0)
        if n == 0 { break }  // Client closed
        if n < 0 {
            if errno == EINTR { continue }
            break  // Timeout or error
        }
        body.append(contentsOf: readBuf[..<n])
    }

    return HTTPRequest(
        method: String(parts[0]), path: String(parts[1]),
        headers: headers, body: body
    )
}

func parseMultipartFormData(body: Data, contentTypeHeader: String) -> [MultipartPart]? {
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

func parseHeaderParameters(_ header: String) -> [String: String] {
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

func writeJSONResponse(status: Int, _ json: [String: Any]) -> HTTPResponse {
    let body = (try? JSONSerialization.data(withJSONObject: json)) ?? Data("{}".utf8)
    return HTTPResponse(status: status, contentType: "application/json", body: body)
}

func writeTextResponse(status: Int, _ text: String, contentType: String = "text/plain; charset=utf-8") -> HTTPResponse {
    HTTPResponse(status: status, contentType: contentType, body: Data(text.utf8))
}

func sendAll(fd: Int32, _ data: Data) {
    data.withUnsafeBytes { buf in
        guard let base = buf.baseAddress else { return }
        var offset = 0
        while offset < buf.count {
            let n = Darwin.send(fd, base + offset, buf.count - offset, 0)
            if n <= 0 { return }  // EPIPE or error (SIGPIPE already ignored)
            offset += n
        }
    }
}

func writeResponse(fd: Int32, response: HTTPResponse) {
    let statusText: String = switch response.status {
    case 200: "OK"
    case 400: "Bad Request"
    case 404: "Not Found"
    case 405: "Method Not Allowed"
    case 413: "Payload Too Large"
    case 415: "Unsupported Media Type"
    case 422: "Unprocessable Content"
    default: "Error"
    }
    let header = "HTTP/1.1 \(response.status) \(statusText)\r\nContent-Type: \(response.contentType)\r\nContent-Length: \(response.body.count)\r\nConnection: close\r\n\r\n"
    var resp = Data(header.utf8)
    resp.append(response.body)
    sendAll(fd: fd, resp)
}

// MARK: - Subtitle Grouping & Formatting

struct Subtitle {
    let index: Int
    let start: Double   // seconds
    let end: Double     // seconds
    let text: String
}

/// Group word-level alignment items into subtitle entries.
func groupSubtitles(
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

    for (i, item) in items.enumerated() {
        currentWords.append(item)

        let duration = item.endTime - (currentWords.first?.startTime ?? item.startTime)
        let atWordLimit = currentWords.count >= maxWordsPerLine
        let atDurationLimit = duration >= maxDuration
        let atSentenceEnd = item.text.last.map { sentenceEndChars.contains($0) } ?? false
        let hasPause = i + 1 < items.count && (items[i + 1].startTime - item.endTime) >= pauseThreshold

        if atWordLimit || atDurationLimit || atSentenceEnd || hasPause {
            flush()
        }
    }
    flush()
    return subtitles
}

/// Format timestamp as HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT).
private func formatTime(_ seconds: Double, separator: String = ",") -> String {
    let totalMs = Int(seconds * 1000)
    let ms = totalMs % 1000
    let s = (totalMs / 1000) % 60
    let m = (totalMs / 60000) % 60
    let h = totalMs / 3600000
    return String(format: "%02d:%02d:%02d%@%03d", h, m, s, separator, ms)
}

func formatSRT(_ subtitles: [Subtitle]) -> String {
    subtitles.map { sub in
        "\(sub.index)\n\(formatTime(sub.start)) --> \(formatTime(sub.end))\n\(sub.text)"
    }.joined(separator: "\n\n")
}

func formatVTT(_ subtitles: [Subtitle]) -> String {
    "WEBVTT\n\n" + subtitles.map { sub in
        "\(formatTime(sub.start, separator: ".")) --> \(formatTime(sub.end, separator: "."))\n\(sub.text)"
    }.joined(separator: "\n\n")
}

func normalizedLanguageCode(_ language: String?) -> String? {
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

func formatSubtitleJSON(
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
        "language": normalizedLanguageCode(language) ?? language,
        "duration": duration,
        "segments": segments,
    ]
    return (try? JSONSerialization.data(withJSONObject: payload)) ?? Data("{}".utf8)
}

private func splitTextProportionally(_ text: String, chunkDurations: [Double]) -> [String] {
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

private func formatVADChunkRanges(_ chunks: [VADAudioChunk]) -> String {
    chunks.enumerated().map { index, chunk in
        String(format: "%d:%.3f-%.3f", index + 1, chunk.startTime, chunk.endTime)
    }.joined(separator: ",")
}

private func chunkLongAudioWithVAD(_ audio: [Float], vad: SileroVAD) throws -> [VADAudioChunk] {
    vadLock.lock()
    defer { vadLock.unlock() }
    return try vad.chunk(audio: audio, config: longAudioVADConfig)
}

private func transcribeLongAudioWithVAD(
    mgr: SessionManager,
    audio: [Float],
    language: String?,
    temperature: Float,
    vad: SileroVAD
) throws -> TranscriptionResult {
    let startedAt = Date()
    let chunks = try chunkLongAudioWithVAD(audio, vad: vad)
    let transcriptionDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
    log("VAD chunking transcription: \(chunks.count) chunks from \(String(format: "%.1f", transcriptionDuration))s")
    log("VAD chunk ranges transcription: \(formatVADChunkRanges(chunks))")

    var texts: [String] = []
    var resolvedLanguage = language
    for chunk in chunks {
        let chunkResult = try mgr.transcribeAudio(audio: chunk.audio, language: language, temperature: temperature)
        let trimmed = chunkResult.text.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { texts.append(trimmed) }
        if resolvedLanguage == nil { resolvedLanguage = chunkResult.language }
    }

    return TranscriptionResult(
        text: AlignedTextRenderer.render(segments: texts),
        language: resolvedLanguage,
        audioDuration: Double(audio.count) / Double(ASRAudio.sampleRate),
        processingTime: Date().timeIntervalSince(startedAt)
    )
}

private func subtitleLongAudio(
    mgr: SessionManager,
    audio: [Float],
    transcript: String?,
    language: String?,
    temperature: Float,
    aligner: ForcedAligner,
    vad: SileroVAD
) throws -> (transcript: String, language: String, items: [ForcedAlignItem]) {
    let chunks = try chunkLongAudioWithVAD(audio, vad: vad)
    let subtitleDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
    log("VAD chunking subtitles: \(chunks.count) chunks from \(String(format: "%.1f", subtitleDuration))s")
    log("VAD chunk ranges subtitles: \(formatVADChunkRanges(chunks))")

    var allItems: [ForcedAlignItem] = []
    var transcriptParts: [String] = []
    var resolvedLanguage = language ?? "English"

    if let transcript, !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
        let textParts = splitTextProportionally(transcript, chunkDurations: chunks.map(\.duration))
        for (chunk, textPart) in zip(chunks, textParts) {
            let part = try mgr.subtitleItems(
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
            let part = try mgr.subtitleItems(
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

// MARK: - Router

func route(
    _ req: HTTPRequest,
    mgr: SessionManager,
    aligner: ForcedAligner?,
    vad: SileroVAD?,
    streamingModelName: String,
    batchModelName: String?,
    batchRetranscribeEnabled: Bool
) -> HTTPResponse {
    let path = req.path.split(separator: "?").first.map(String.init) ?? req.path
    let streamPrefix = "/v1/audio/transcriptions/stream"
    let batchPaths = Set(["/v1/audio/transcriptions", "/audio/transcriptions"])

    if path == "/v1/info" {
        guard req.method == "GET" else { return writeJSONResponse(status: 405, ["error": "method not allowed"]) }
        var info: [String: Any] = [
            "streaming_model": streamingModelName,
            "sample_rate": ASRAudio.sampleRate,
            "chunk_sec": 2.0,
            "batch_retranscribe": batchRetranscribeEnabled,
            "internal_diagnostics": internalDiagnosticsEnabled,
            "status": "ready",
        ]
        if let batchModelName { info["batch_model"] = batchModelName }
        info["aligner"] = aligner != nil
        info["vad"] = vad != nil
        return writeJSONResponse(status: 200, info)
    }

    if batchPaths.contains(path) {
        guard req.method == "POST" else { return writeJSONResponse(status: 405, ["error": "method not allowed"]) }
        guard let contentType = req.headers["content-type"],
              contentType.lowercased().contains("multipart/form-data"),
              let parts = parseMultipartFormData(body: req.body, contentTypeHeader: contentType)
        else {
            return writeJSONResponse(status: 415, ["error": "expected multipart/form-data upload"])
        }

        var fields: [String: String] = [:]
        for part in parts where part.filename == nil {
            if let value = part.textValue {
                fields[part.name] = value
            }
        }
        guard let filePart = parts.first(where: { $0.name == "file" }) else {
            return writeJSONResponse(status: 400, ["error": "missing file field"])
        }

        let requestedFormat = (fields["response_format"] ?? "json").lowercased()
        if (fields["stream"] ?? "false").lowercased() == "true" {
            return writeJSONResponse(status: 400, ["error": "stream=true is not supported on this endpoint"])
        }
        guard ["json", "text", "verbose_json"].contains(requestedFormat) else {
            return writeJSONResponse(status: 400, ["error": "unsupported response_format: \(requestedFormat)"])
        }

        let language = fields["language"].flatMap { $0.isEmpty ? nil : $0 }
        let temperature = Float(fields["temperature"] ?? "0") ?? 0
        let filename = filePart.filename ?? inferredFilename(contentType: filePart.contentType)
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let tempURL = tempDir.appendingPathComponent(sanitizedFilename(filename))

        do {
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
            try filePart.body.write(to: tempURL, options: [.atomic])
            defer { try? FileManager.default.removeItem(at: tempDir) }

            let audio = try loadAudioFile(tempURL)
            let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
            let result: TranscriptionResult
            if let vad, audioDuration > longAudioChunkThresholdSec {
                result = try transcribeLongAudioWithVAD(
                    mgr: mgr,
                    audio: audio,
                    language: language,
                    temperature: temperature,
                    vad: vad
                )
            } else {
                result = try mgr.transcribeAudio(audio: audio, language: language, temperature: temperature)
            }

            switch requestedFormat {
            case "text":
                return writeTextResponse(status: 200, result.text)
            case "verbose_json":
                var payload: [String: Any] = [
                    "text": result.text,
                    "duration": result.audioDuration,
                ]
                if let language = result.language {
                    payload["language"] = normalizedLanguageCode(language) ?? language
                }
                return writeJSONResponse(status: 200, payload)
            default:
                return writeJSONResponse(status: 200, ["text": result.text])
            }
        } catch {
            return writeJSONResponse(status: 422, ["error": error.localizedDescription])
        }
    }

    // MARK: Subtitle / Alignment endpoint

    if path == "/v1/audio/subtitles" {
        guard req.method == "POST" else { return writeJSONResponse(status: 405, ["error": "method not allowed"]) }
        guard let aligner else {
            return writeJSONResponse(status: 501, ["error": "aligner model not loaded (start server with --aligner-model)"])
        }
        guard let contentType = req.headers["content-type"],
              contentType.lowercased().contains("multipart/form-data"),
              let parts = parseMultipartFormData(body: req.body, contentTypeHeader: contentType)
        else {
            return writeJSONResponse(status: 415, ["error": "expected multipart/form-data upload"])
        }

        var fields: [String: String] = [:]
        for part in parts where part.filename == nil {
            if let value = part.textValue { fields[part.name] = value }
        }
        guard let filePart = parts.first(where: { $0.name == "file" }) else {
            return writeJSONResponse(status: 400, ["error": "missing file field"])
        }

        let requestedLanguage = fields["language"].flatMap { $0.isEmpty ? nil : $0 }
        let responseFormat = (fields["response_format"] ?? "srt").lowercased()
        guard ["srt", "vtt", "json", "text"].contains(responseFormat) else {
            return writeJSONResponse(status: 400, ["error": "unsupported response_format: \(responseFormat). Use srt, vtt, json, or text"])
        }

        let maxWordsPerLine = Int(fields["max_words_per_line"] ?? "8") ?? 8
        let maxDuration = Double(fields["max_duration"] ?? "5.0") ?? 5.0
        let pauseThreshold = Double(fields["pause_threshold"] ?? "0.5") ?? 0.5
        let temperature = Float(fields["temperature"] ?? "0") ?? 0

        // Decode audio to temp file, load as Float samples
        let filename = filePart.filename ?? inferredFilename(contentType: filePart.contentType)
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let tempURL = tempDir.appendingPathComponent(sanitizedFilename(filename))

        do {
            try FileManager.default.createDirectory(at: tempDir, withIntermediateDirectories: true)
            try filePart.body.write(to: tempURL, options: [.atomic])
            defer { try? FileManager.default.removeItem(at: tempDir) }

            let audio = try loadAudioFile(tempURL)
            let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
            let subtitleResult: (transcript: String, language: String, items: [ForcedAlignItem])

            if let vad, audioDuration > subtitleVADThresholdSec {
                let result = try subtitleLongAudio(
                    mgr: mgr,
                    audio: audio,
                    transcript: fields["text"],
                    language: requestedLanguage,
                    temperature: temperature,
                    aligner: aligner,
                    vad: vad
                )
                subtitleResult = (result.transcript, result.language, result.items)
            } else {
                guard audioDuration <= subtitleSimpleMaxAudioSec else {
                    return writeJSONResponse(
                        status: 422,
                        ["error": "long subtitle generation requires built-in VAD; keep subtitle audio under 10 minutes if VAD is unavailable"]
                    )
                }

                subtitleResult = try mgr.subtitleItems(
                    audio: audio,
                    transcript: fields["text"],
                    language: requestedLanguage,
                    temperature: temperature,
                    aligner: aligner
                )
            }

            switch responseFormat {
            case "text":
                return writeTextResponse(status: 200, subtitleResult.transcript)
            case "json":
                let subtitles = groupSubtitles(
                    subtitleResult.items,
                    maxWordsPerLine: maxWordsPerLine,
                    maxDuration: maxDuration,
                    pauseThreshold: pauseThreshold
                )
                let body = formatSubtitleJSON(
                    transcript: subtitleResult.transcript,
                    language: subtitleResult.language,
                    duration: audioDuration,
                    subtitles: subtitles
                )
                return HTTPResponse(status: 200, contentType: "application/json", body: body)
            case "vtt":
                let subtitles = groupSubtitles(
                    subtitleResult.items,
                    maxWordsPerLine: maxWordsPerLine,
                    maxDuration: maxDuration,
                    pauseThreshold: pauseThreshold
                )
                return writeTextResponse(status: 200, formatVTT(subtitles), contentType: "text/vtt; charset=utf-8")
            default: // srt
                let subtitles = groupSubtitles(
                    subtitleResult.items,
                    maxWordsPerLine: maxWordsPerLine,
                    maxDuration: maxDuration,
                    pauseThreshold: pauseThreshold
                )
                return writeTextResponse(status: 200, formatSRT(subtitles), contentType: "text/srt; charset=utf-8")
            }
        } catch {
            return writeJSONResponse(status: 422, ["error": error.localizedDescription])
        }
    }

    if path == streamPrefix {
        guard req.method == "POST" else { return writeJSONResponse(status: 405, ["error": "method not allowed"]) }
        return writeJSONResponse(status: 200, ["session_id": mgr.create()])
    }

    if path.hasPrefix(streamPrefix + "/") {
        let sid = String(path.dropFirst(streamPrefix.count + 1))
        guard !sid.isEmpty else { return writeJSONResponse(status: 400, ["error": "missing session_id"]) }

        switch req.method {
        case "POST":
            guard let result = mgr.feed(sid, pcmData: req.body) else {
                return writeJSONResponse(status: 404, ["error": "session not found"])
            }
            return writeJSONResponse(status: 200, result)
        case "DELETE":
            guard let result = mgr.stop(sid) else {
                return writeJSONResponse(status: 404, ["error": "session not found"])
            }
            return writeJSONResponse(status: 200, result)
        default:
            return writeJSONResponse(status: 405, ["error": "method not allowed"])
        }
    }

    return writeJSONResponse(status: 404, ["error": "unknown endpoint: \(req.method) \(path)"])
}

// MARK: - Server

func sanitizedFilename(_ filename: String) -> String {
    let trimmed = filename.trimmingCharacters(in: .whitespacesAndNewlines)
    if trimmed.isEmpty { return "upload.wav" }
    return URL(fileURLWithPath: trimmed).lastPathComponent
}

func inferredFilename(contentType: String?) -> String {
    switch contentType?.lowercased() {
    case "audio/flac", "application/flac": return "upload.flac"
    case "audio/x-wav", "audio/wav", "audio/wave": return "upload.wav"
    case "audio/mpeg", "audio/mp3": return "upload.mp3"
    case "audio/mp4", "audio/m4a", "video/mp4": return "upload.m4a"
    default: return "upload.wav"
    }
}

func startServer(
    host: String,
    port: UInt16,
    mgr: SessionManager,
    aligner: ForcedAligner?,
    vad: SileroVAD?,
    streamingModelName: String,
    batchModelName: String?,
    batchRetranscribeEnabled: Bool,
    parentPID: Int32?
) {
    let serverFd = socket(AF_INET, SOCK_STREAM, 0)
    guard serverFd >= 0 else { fputs("socket() failed\n", stderr); exit(1) }

    var opt: Int32 = 1
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))

    // Ignore SIGPIPE (client disconnect during send)
    signal(SIGPIPE, SIG_IGN)

    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = port.bigEndian
    addr.sin_addr.s_addr = inet_addr(host)

    let ok = withUnsafePointer(to: &addr) { ptr in
        ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { bind(serverFd, $0, socklen_t(MemoryLayout<sockaddr_in>.size)) }
    }
    guard ok == 0 else { fputs("bind() failed on \(host):\(port) — errno \(errno)\n", stderr); exit(1) }
    guard listen(serverFd, 32) == 0 else { fputs("listen() failed\n", stderr); exit(1) }

    // Register for graceful shutdown
    serverSocket = serverFd
    signal(SIGINT, handleShutdown)
    signal(SIGTERM, handleShutdown)
    startParentWatch(expectedParentPID: parentPID)

    log("Listening on http://\(host):\(port)")

    let inFlight = DispatchGroup()

    // Accept loop — dispatch each connection concurrently.
    // Inference is serialized by SessionManager.inferenceLock.
    while !shuttingDown {
        var clientAddr = sockaddr_in()
        var addrLen = socklen_t(MemoryLayout<sockaddr_in>.size)
        let clientFd = withUnsafeMutablePointer(to: &clientAddr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { accept(serverFd, $0, &addrLen) }
        }
        if clientFd < 0 {
            if errno == EINTR { continue }
            break  // Socket closed by signal handler or fatal error
        }

        inFlight.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            defer {
                close(clientFd)
                inFlight.leave()
            }
            if let req = readHTTPRequest(fd: clientFd) {
                let response = route(
                    req,
                    mgr: mgr,
                    aligner: aligner,
                    vad: vad,
                    streamingModelName: streamingModelName,
                    batchModelName: batchModelName,
                    batchRetranscribeEnabled: batchRetranscribeEnabled
                )
                writeResponse(fd: clientFd, response: response)
            }
        }
    }

    // Wait for in-flight connections (with timeout)
    log("Shutting down...")
    let result = inFlight.wait(timeout: .now() + 5)
    if result == .timedOut { log("Timed out waiting for in-flight requests") }

    // Defensive cleanup (signal handler may have already closed)
    parentWatchTimer?.cancel()
    parentWatchTimer = nil
    let fd = serverSocket
    serverSocket = -1
    if fd >= 0 { close(fd) }

    log("Server stopped")
}

// MARK: - Logging

func log(_ msg: String) {
    fputs("[asr-server] \(msg)\n", stderr)
}

// MARK: - Main

do {
    var args = Array(CommandLine.arguments.dropFirst())
    guard !args.isEmpty else {
        fputs("Usage: asr-server <streaming-model-dir> [--batch-model <dir>] [--aligner-model <dir>] [--disable-batch-retranscribe] [--port 9748] [--host 127.0.0.1] [--parent-pid <pid>] [--warmup]\n", stderr)
        exit(1)
    }

    let modelPath = args.removeFirst()
    var port: UInt16 = 9748
    var host = "127.0.0.1"
    var parentPID: Int32?
    var doWarmup = false
    var batchModelPath: String?
    var alignerModelPath: String?
    var batchRetranscribeEnabled = true

    while !args.isEmpty {
        switch args.removeFirst() {
        case "--port":
            guard !args.isEmpty, let p = UInt16(args.removeFirst()) else {
                fputs("--port requires a number\n", stderr); exit(1)
            }
            port = p
        case "--host":
            guard !args.isEmpty else { fputs("--host requires a value\n", stderr); exit(1) }
            host = args.removeFirst()
        case "--parent-pid":
            guard !args.isEmpty, let pid = Int32(args.removeFirst()) else {
                fputs("--parent-pid requires a pid\n", stderr); exit(1)
            }
            parentPID = pid
        case "--warmup": doWarmup = true
        case "--batch-model":
            guard !args.isEmpty else { fputs("--batch-model requires a path\n", stderr); exit(1) }
            batchModelPath = args.removeFirst()
        case "--aligner-model":
            guard !args.isEmpty else { fputs("--aligner-model requires a path\n", stderr); exit(1) }
            alignerModelPath = args.removeFirst()
        case "--disable-batch-retranscribe":
            batchRetranscribeEnabled = false
        case let flag: fputs("Unknown option: \(flag)\n", stderr); exit(1)
        }
    }

    let modelURL = URL(fileURLWithPath: modelPath)
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
        fputs("Model not found: \(modelPath)\n", stderr); exit(1)
    }

    let transcriber = try Qwen3ASRTranscriber.load(from: modelURL)
    let batchTranscriber: Qwen3ASRTranscriber?
    if batchRetranscribeEnabled, let batchModelPath {
        let batchURL = URL(fileURLWithPath: batchModelPath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: batchURL.path) else {
            fputs("Batch model not found: \(batchModelPath)\n", stderr)
            exit(1)
        }
        batchTranscriber = batchURL == modelURL.standardizedFileURL
            ? transcriber
            : try Qwen3ASRTranscriber.load(from: batchURL)
    } else {
        batchTranscriber = nil
    }
    // Load aligner model (optional)
    let aligner: ForcedAligner?
    if let alignerModelPath {
        let alignerURL = URL(fileURLWithPath: alignerModelPath)
        guard FileManager.default.fileExists(atPath: alignerURL.path) else {
            fputs("Aligner model not found: \(alignerModelPath)\n", stderr)
            exit(1)
        }
        log("Loading aligner model from \(alignerModelPath)...")
        aligner = try ForcedAligner.load(from: alignerURL)
        log("Aligner loaded (classify_num=\(aligner!.model.config.classifyNum))")
    } else {
        aligner = nil
    }

    let vad: SileroVAD?
    do {
        vad = try SileroVAD()
        log("Silero VAD loaded")
    } catch {
        log("Silero VAD unavailable: \(error.localizedDescription)")
        vad = nil
    }

    if doWarmup {
        try transcriber.warmup()
        if let batchTranscriber, batchTranscriber !== transcriber {
            try batchTranscriber.warmup()
        }
    }

    let mgr = SessionManager(
        transcriber: transcriber,
        batchTranscriber: batchTranscriber,
        batchRetranscribeEnabled: batchRetranscribeEnabled
    )
    startServer(
        host: host,
        port: port,
        mgr: mgr,
        aligner: aligner,
        vad: vad,
        streamingModelName: modelURL.lastPathComponent,
        batchModelName: batchTranscriber?.modelDirectory.lastPathComponent,
        batchRetranscribeEnabled: batchRetranscribeEnabled,
        parentPID: parentPID
    )
} catch {
    fputs("Error: \(error.localizedDescription)\n", stderr)
    exit(1)
}
