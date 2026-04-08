// asr-server — Native streaming ASR HTTP server.
// Drop-in replacement for Sources/sidecar/transcribe.py.
// Serves both Yuwp (spawned as child) and Oppi server (HTTP client).
//
// Endpoints (identical to Python sidecar):
//   GET    /v1/info                                → server status
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

private let maxBodySize = 10 * 1024 * 1024  // 10MB

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let body: Data
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

    // Read body (capped at 10MB)
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

func writeResponse(fd: Int32, status: Int, json: [String: Any]) {
    let body = (try? JSONSerialization.data(withJSONObject: json)) ?? Data("{}".utf8)
    let statusText: String = switch status {
    case 200: "OK"
    case 400: "Bad Request"
    case 404: "Not Found"
    case 405: "Method Not Allowed"
    case 413: "Payload Too Large"
    default: "Error"
    }
    let header = "HTTP/1.1 \(status) \(statusText)\r\nContent-Type: application/json\r\nContent-Length: \(body.count)\r\nConnection: close\r\n\r\n"
    var resp = Data(header.utf8)
    resp.append(body)
    sendAll(fd: fd, resp)
}

// MARK: - Router

func route(
    _ req: HTTPRequest,
    mgr: SessionManager,
    streamingModelName: String,
    batchModelName: String?,
    batchRetranscribeEnabled: Bool
) -> (Int, [String: Any]) {
    let path = req.path.split(separator: "?").first.map(String.init) ?? req.path
    let streamPrefix = "/v1/audio/transcriptions/stream"

    if path == "/v1/info" {
        guard req.method == "GET" else { return (405, ["error": "method not allowed"]) }
        var info: [String: Any] = [
            "streaming_model": streamingModelName,
            "sample_rate": ASRAudio.sampleRate,
            "chunk_sec": 2.0,
            "batch_retranscribe": batchRetranscribeEnabled,
            "status": "ready",
        ]
        if let batchModelName { info["batch_model"] = batchModelName }
        return (200, info)
    }

    if path == streamPrefix {
        guard req.method == "POST" else { return (405, ["error": "method not allowed"]) }
        return (200, ["session_id": mgr.create()])
    }

    if path.hasPrefix(streamPrefix + "/") {
        let sid = String(path.dropFirst(streamPrefix.count + 1))
        guard !sid.isEmpty else { return (400, ["error": "missing session_id"]) }

        switch req.method {
        case "POST":
            guard let result = mgr.feed(sid, pcmData: req.body) else {
                return (404, ["error": "session not found"])
            }
            return (200, result)
        case "DELETE":
            guard let text = mgr.stop(sid) else {
                return (404, ["error": "session not found"])
            }
            return (200, ["text": text])
        default:
            return (405, ["error": "method not allowed"])
        }
    }

    return (404, ["error": "unknown endpoint: \(req.method) \(path)"])
}

// MARK: - Server

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
                let (status, json) = route(
                    req,
                    mgr: mgr,
                    streamingModelName: streamingModelName,
                    batchModelName: batchModelName,
                    batchRetranscribeEnabled: batchRetranscribeEnabled
                )
                writeResponse(fd: clientFd, status: status, json: json)
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
