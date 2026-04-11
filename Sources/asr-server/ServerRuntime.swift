#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import ASRServerSupport
import Foundation
import NativeASR

// MARK: - Shutdown State

final class ShutdownCoordinator: @unchecked Sendable {
    private let lock = NSLock()
    private var serverSocket: Int32 = -1
    private var shuttingDown = false
    private var parentWatchTimer: DispatchSourceTimer?
    private var signalSources: [DispatchSourceSignal] = []

    func setServerSocket(_ fd: Int32) {
        lock.lock()
        serverSocket = fd
        lock.unlock()
    }

    func isShutdownRequested() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return shuttingDown
    }

    func installSignalHandlers() {
        let signals = [SIGINT, SIGTERM]
        for signalNumber in signals {
            signal(signalNumber, SIG_IGN)
            let source = DispatchSource.makeSignalSource(
                signal: signalNumber,
                queue: DispatchQueue.global(qos: .userInitiated)
            )
            source.setEventHandler { [weak self] in
                _ = self?.requestShutdown(reason: "Received signal \(signalNumber) — shutting down")
            }
            source.resume()
            signalSources.append(source)
        }
    }

    func startParentWatch(expectedParentPID: Int32?) {
        guard let expectedParentPID else { return }
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .background))
        timer.schedule(deadline: .now() + .seconds(1), repeating: .seconds(1))
        timer.setEventHandler { [weak self] in
            let currentParentPID = getppid()
            guard currentParentPID == expectedParentPID else {
                _ = self?.requestShutdown(
                    reason: "Parent process \(expectedParentPID) disappeared (current ppid: \(currentParentPID)) — shutting down"
                )
                return
            }
        }

        lock.lock()
        parentWatchTimer?.cancel()
        parentWatchTimer = timer
        lock.unlock()

        timer.resume()
    }

    @discardableResult
    func requestShutdown(reason: String? = nil) -> Bool {
        lock.lock()
        guard !shuttingDown else {
            lock.unlock()
            return false
        }
        shuttingDown = true
        let timer = parentWatchTimer
        parentWatchTimer = nil
        let fd = serverSocket
        serverSocket = -1
        lock.unlock()

        if let reason { log(reason) }
        timer?.cancel()
        if fd >= 0 {
            Darwin.shutdown(fd, SHUT_RDWR)
            close(fd)
        }
        return true
    }

    func cleanup() {
        lock.lock()
        let timer = parentWatchTimer
        parentWatchTimer = nil
        let fd = serverSocket
        serverSocket = -1
        let sources = signalSources
        signalSources = []
        lock.unlock()

        timer?.cancel()
        for source in sources {
            source.cancel()
        }
        if fd >= 0 {
            close(fd)
        }
    }
}

// MARK: - Session Manager

final class StreamingSessionManager: @unchecked Sendable {
    private let transcriber: Qwen3ASRTranscriber
    private let batchTranscriber: Qwen3ASRTranscriber?
    private let batchRetranscribeEnabled: Bool
    private var sessions: [String: StreamingSession] = [:]
    private var pendingAudio: [String: [Float]] = [:]
    private var lastActivity: [String: Date] = [:]
    private let stateLock = NSLock()
    private let inferenceLock = NSLock()
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
        pendingAudio[sid] = []
        lastActivity[sid] = Date()
        stateLock.unlock()

        let samples = pcmData.withUnsafeBytes { buffer -> [Float] in
            let int16s = buffer.bindMemory(to: Int16.self)
            return int16s.map { Float($0) / 32768.0 }
        }
        pending.append(contentsOf: samples)

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
        var response: [String: Any] = [
            "text": session.finalText(),
            "committed_text": session.committedSegmentText(),
            "active_text": session.activeSegmentText(),
            "update_kind": kind,
            "is_final": isFinal,
        ]
        if batchCorrected { response["batch_corrected"] = true }
        return response
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

        if audioDuration <= ASRServerLimits.longAudioChunkThresholdSec {
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
        let chunkSamples = Int(ASRServerLimits.longAudioChunkSec * Double(ASRAudio.sampleRate))
        let totalChunks = (audio.count + chunkSamples - 1) / chunkSamples
        var texts: [String] = []
        var detectedLanguage: String?
        var offset = 0

        log(
            "Long batch transcription: \(String(format: "%.1f", audioDuration))s "
                + "audio -> \(totalChunks) fixed chunks of \(Int(ASRServerLimits.longAudioChunkSec))s"
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

extension StreamingSessionManager: ASRServing {}

// MARK: - HTTP Parsing / Response IO

func readRequest(fd: Int32) -> HTTPRequest? {
    var timeout = timeval(tv_sec: 30, tv_usec: 0)
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &timeout, socklen_t(MemoryLayout<timeval>.size))

    var headerBuf = Data()
    var readBuf = [UInt8](repeating: 0, count: 8192)
    var headerEnd = -1

    while headerEnd < 0 {
        let count = recv(fd, &readBuf, readBuf.count, 0)
        if count == 0 { return nil }
        if count < 0 {
            if errno == EINTR { continue }
            return nil
        }
        headerBuf.append(contentsOf: readBuf[..<count])
        if let range = headerBuf.range(of: Data([0x0D, 0x0A, 0x0D, 0x0A])) {
            headerEnd = range.upperBound
        }
        if headerBuf.count > 65536 { return nil }
    }

    guard let headerString = String(data: headerBuf[..<headerEnd], encoding: .utf8) else { return nil }
    let lines = headerString.split(separator: "\r\n", omittingEmptySubsequences: false)
    guard let requestLine = lines.first else { return nil }
    let parts = requestLine.split(separator: " ", maxSplits: 2)
    guard parts.count >= 2 else { return nil }

    var headers: [String: String] = [:]
    for line in lines.dropFirst() {
        guard let colon = line.firstIndex(of: ":") else { continue }
        let key = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
        let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
        headers[key] = value
    }

    let contentLength = Int(headers["content-length"] ?? "0") ?? 0
    guard contentLength >= 0 else { return nil }
    if contentLength > ASRServerLimits.maxBodySize {
        log("Rejected: body too large (\(contentLength) bytes)")
        writeResponse(
            fd: fd,
            response: jsonResponse(
                status: 413,
                ["error": "request body too large: \(contentLength) bytes (max \(ASRServerLimits.maxBodySize))"]
            )
        )
        return nil
    }

    var body = Data(headerBuf[headerEnd...])
    while body.count < contentLength {
        let remaining = contentLength - body.count
        let count = recv(fd, &readBuf, min(readBuf.count, remaining), 0)
        if count == 0 { break }
        if count < 0 {
            if errno == EINTR { continue }
            break
        }
        body.append(contentsOf: readBuf[..<count])
    }

    return HTTPRequest(
        method: String(parts[0]),
        path: String(parts[1]),
        headers: headers,
        body: body
    )
}

func sendAllBytes(fd: Int32, _ data: Data) {
    data.withUnsafeBytes { buffer in
        guard let base = buffer.baseAddress else { return }
        var offset = 0
        while offset < buffer.count {
            let count = Darwin.send(fd, base + offset, buffer.count - offset, 0)
            if count <= 0 { return }
            offset += count
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
    case 501: "Not Implemented"
    default: "Error"
    }
    let header = "HTTP/1.1 \(response.status) \(statusText)\r\nContent-Type: \(response.contentType)\r\nContent-Length: \(response.body.count)\r\nConnection: close\r\n\r\n"
    var responseData = Data(header.utf8)
    responseData.append(response.body)
    sendAllBytes(fd: fd, responseData)
}

// MARK: - Server

func startServer(
    host: String,
    port: UInt16,
    mgr: StreamingSessionManager,
    aligner: ForcedAligner?,
    vad: SileroVAD?,
    streamingModelName: String,
    batchModelName: String?,
    batchRetranscribeEnabled: Bool,
    parentPID: Int32?
) {
    let shutdown = ShutdownCoordinator()
    let serverFd = socket(AF_INET, SOCK_STREAM, 0)
    guard serverFd >= 0 else { fputs("socket() failed\n", stderr); exit(1) }

    var opt: Int32 = 1
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))
    signal(SIGPIPE, SIG_IGN)

    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = port.bigEndian
    addr.sin_addr.s_addr = inet_addr(host)

    let bindResult = withUnsafePointer(to: &addr) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(serverFd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    guard bindResult == 0 else { fputs("bind() failed on \(host):\(port) — errno \(errno)\n", stderr); exit(1) }
    guard listen(serverFd, 32) == 0 else { fputs("listen() failed\n", stderr); exit(1) }

    shutdown.setServerSocket(serverFd)
    shutdown.installSignalHandlers()
    shutdown.startParentWatch(expectedParentPID: parentPID)

    let routeContext = ASRRouteContext(
        manager: mgr,
        aligner: aligner,
        vad: vad,
        streamingModelName: streamingModelName,
        batchModelName: batchModelName,
        batchRetranscribeEnabled: batchRetranscribeEnabled,
        loadAudio: loadAudioFile,
        log: log
    )

    log("Listening on http://\(host):\(port)")

    let inFlight = DispatchGroup()
    while !shutdown.isShutdownRequested() {
        var clientAddr = sockaddr_in()
        var addrLen = socklen_t(MemoryLayout<sockaddr_in>.size)
        let clientFd = withUnsafeMutablePointer(to: &clientAddr) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { accept(serverFd, $0, &addrLen) }
        }
        if clientFd < 0 {
            if errno == EINTR { continue }
            break
        }

        inFlight.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            defer {
                close(clientFd)
                inFlight.leave()
            }
            if let request = readRequest(fd: clientFd) {
                let response = routeRequest(request, context: routeContext)
                writeResponse(fd: clientFd, response: response)
            }
        }
    }

    log("Shutting down...")
    if inFlight.wait(timeout: .now() + 5) == .timedOut {
        log("Timed out waiting for in-flight requests")
    }

    shutdown.cleanup()
    log("Server stopped")
}

// MARK: - Logging

func log(_ message: String) {
    fputs("[asr-server] \(message)\n", stderr)
}
