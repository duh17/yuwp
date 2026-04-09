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
//
// Usage: asr-server <streaming-model-dir> [--batch-model <dir>] [--disable-batch-retranscribe]
//                   [--port 9748] [--host 127.0.0.1] [--warmup]

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

#if YUWP_INTERNAL_DIAGNOSTICS
let internalDiagnosticsEnabled = true
#else
let internalDiagnosticsEnabled = false
#endif

func handleShutdown(_: Int32) {
    guard !shuttingDown else { return }
    shuttingDown = true
    let fd = serverSocket
    serverSocket = -1
    if fd >= 0 {
        Darwin.shutdown(fd, SHUT_RDWR)
        close(fd)
    }
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
        inferenceLock.unlock()

        stateLock.lock()
        if sessions[sid] != nil {
            // Prepend remainder to any audio that arrived during inference
            pendingAudio[sid] = pending + (pendingAudio[sid] ?? [])
        }
        stateLock.unlock()

        var resp: [String: Any] = ["text": session.finalText()]
        if batchCorrected { resp["batch_corrected"] = true }
        return resp
    }

    func stop(_ sid: String) -> String? {
        stateLock.lock()
        guard let session = sessions.removeValue(forKey: sid) else { stateLock.unlock(); return nil }
        let pending = pendingAudio.removeValue(forKey: sid) ?? []
        lastActivity.removeValue(forKey: sid)
        stateLock.unlock()

        // Flush remaining audio under inference lock
        inferenceLock.lock()
        if !pending.isEmpty { _ = session.processChunk(pending) }
        let text = session.finalize()
        inferenceLock.unlock()

        log("Session stopped (\(sid)): \(text.count) chars")
        return text
    }

    func transcribeFile(
        data: Data,
        filename: String,
        language: String? = nil,
        temperature: Float = 0.0
    ) throws -> TranscriptionResult {
        let transcriber = batchTranscriber ?? self.transcriber
        let tempURL = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString)
            .appendingPathComponent(sanitizedFilename(filename))
        try FileManager.default.createDirectory(at: tempURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: tempURL, options: [.atomic])
        defer {
            try? FileManager.default.removeItem(at: tempURL.deletingLastPathComponent())
        }

        inferenceLock.lock()
        defer { inferenceLock.unlock() }
        let audio = try loadAudioFile(tempURL)
        return try transcriber.transcribe(audio: audio, language: language, temperature: temperature)
    }

    private func sanitizedFilename(_ filename: String) -> String {
        let trimmed = filename.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return "upload.wav" }
        return URL(fileURLWithPath: trimmed).lastPathComponent
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
    guard contentLength >= 0, contentLength <= maxBodySize else {
        log("Rejected: body too large (\(contentLength) bytes)")
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

// MARK: - Router

func route(
    _ req: HTTPRequest,
    mgr: SessionManager,
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

        do {
            let result = try mgr.transcribeFile(data: filePart.body, filename: filename, language: language, temperature: temperature)
            switch requestedFormat {
            case "text":
                return writeTextResponse(status: 200, result.text)
            case "verbose_json":
                var payload: [String: Any] = [
                    "text": result.text,
                    "duration": result.audioDuration,
                ]
                if let language = result.language {
                    payload["language"] = language
                }
                return writeJSONResponse(status: 200, payload)
            default:
                return writeJSONResponse(status: 200, ["text": result.text])
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
            guard let text = mgr.stop(sid) else {
                return writeJSONResponse(status: 404, ["error": "session not found"])
            }
            return writeJSONResponse(status: 200, ["text": text])
        default:
            return writeJSONResponse(status: 405, ["error": "method not allowed"])
        }
    }

    return writeJSONResponse(status: 404, ["error": "unknown endpoint: \(req.method) \(path)"])
}

// MARK: - Server

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
    streamingModelName: String,
    batchModelName: String?,
    batchRetranscribeEnabled: Bool
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
        fputs("Usage: asr-server <streaming-model-dir> [--batch-model <dir>] [--disable-batch-retranscribe] [--port 9748] [--host 127.0.0.1] [--warmup]\n", stderr)
        exit(1)
    }

    let modelPath = args.removeFirst()
    var port: UInt16 = 9748
    var host = "127.0.0.1"
    var doWarmup = false
    var batchModelPath: String?
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
        case "--warmup": doWarmup = true
        case "--batch-model":
            guard !args.isEmpty else { fputs("--batch-model requires a path\n", stderr); exit(1) }
            batchModelPath = args.removeFirst()
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
        streamingModelName: modelURL.lastPathComponent,
        batchModelName: batchTranscriber?.modelDirectory.lastPathComponent,
        batchRetranscribeEnabled: batchRetranscribeEnabled
    )
} catch {
    fputs("Error: \(error.localizedDescription)\n", stderr)
    exit(1)
}
