#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import ASRIPC
import Foundation
import MLX
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

private let vadLock = NSLock()

private func analyzeSpeechActivity(_ audio: [Float], vad: SileroVAD?) -> SpeechActivityHint? {
    guard let vad else { return nil }
    guard !audio.isEmpty else { return SpeechActivityHint(hasSpeech: false, speechDurationSec: 0) }

    let frameSize = SileroVAD.chunkSize
    var speechFrames = 0
    var offset = 0
    vadLock.lock()
    defer { vadLock.unlock() }
    vad.reset()

    while offset < audio.count {
        let end = min(offset + frameSize, audio.count)
        var frame = Array(audio[offset ..< end])
        if frame.count < frameSize {
            frame.append(contentsOf: repeatElement(0, count: frameSize - frame.count))
        }
        if let probability = try? vad.process(frame), probability >= 0.5 {
            speechFrames += 1
        }
        offset = end
    }

    let speechDurationSec = Double(speechFrames * frameSize) / Double(ASRAudio.sampleRate)
    return SpeechActivityHint(hasSpeech: speechFrames > 0, speechDurationSec: speechDurationSec)
}

private final class ManagedStreamingSession: @unchecked Sendable {
    let session: StreamingSession
    let gate = SessionOperationGate()
    var pendingAudio = StreamingAudioAccumulator()
    var preview = FirstPartialPreview()
    var lastActivity: Date
    var recordingData: Data?
    let recordingStartedAt: Date
    let language: String?
    let contextualStrings: [String]

    init(
        session: StreamingSession,
        startedAt: Date,
        language: String?,
        contextualStrings: [String],
        recordsAudio: Bool
    ) {
        self.session = session
        self.lastActivity = startedAt
        self.recordingData = recordsAudio ? Data() : nil
        self.recordingStartedAt = startedAt
        self.language = language
        self.contextualStrings = contextualStrings
    }
}

final class StreamingSessionManager: @unchecked Sendable {
    private let transcriber: Qwen3ASRTranscriber
    private let batchTranscriber: Qwen3ASRTranscriber?
    private let batchRetranscribeEnabled: Bool
    private var sessions: [String: ManagedStreamingSession] = [:]
    private let stateLock = NSLock()
    private let inferenceLock = NSLock()
    private let streamConfig: StreamConfig
    private let chunkSamples: Int
    private let bootstrapChunkSamples: Int
    private let vad: SileroVAD?
    private let batchVAD: SileroVAD?
    private let batchChunking: BatchChunkingMode
    private let sessionTimeout: TimeInterval = 300
    private let recordingConfiguration: ASRStreamRecordingConfiguration
    private var processedChunksSinceCacheTrim = 0
    private let cacheTrimChunkInterval = 8

    init(
        transcriber: Qwen3ASRTranscriber,
        batchTranscriber: Qwen3ASRTranscriber? = nil,
        batchRetranscribeEnabled: Bool = true,
        vad: SileroVAD? = nil,
        batchVAD: SileroVAD? = nil,
        batchChunking: BatchChunkingMode = .automatic,
        chunkSec: Double? = nil,
        recordingConfiguration: ASRStreamRecordingConfiguration? = nil
    ) {
        self.transcriber = transcriber
        self.batchTranscriber = batchTranscriber
        self.batchRetranscribeEnabled = batchRetranscribeEnabled
        self.vad = vad
        // Batch VAD is intentionally explicit. The live VAD instance is only
        // for streaming activity hints; stdio and energy-only batch paths do
        // not load or implicitly reuse it.
        self.batchVAD = batchVAD
        self.batchChunking = batchChunking
        var streamConfig = StreamConfig.forModel(
            at: transcriber.modelDirectory,
            batchRetranscribe: batchRetranscribeEnabled
        )
        if let chunkSec {
            streamConfig.chunkSec = chunkSec
        }
        self.streamConfig = streamConfig
        self.chunkSamples = Int(streamConfig.chunkSec * Double(ASRAudio.sampleRate))
        self.bootstrapChunkSamples = Int(min(streamConfig.chunkSec, 1.5) * Double(ASRAudio.sampleRate))
        self.recordingConfiguration = recordingConfiguration
            ?? .disabled(transcriptionModel: transcriber.modelDirectory.lastPathComponent)
        DispatchQueue.global().async { [weak self] in
            while true {
                Thread.sleep(forTimeInterval: 30)
                self?.cleanupExpired()
            }
        }
    }

    func create(language: String? = nil, contextualStrings: [String] = []) -> String {
        let sid = String(UUID().uuidString.replacingOccurrences(of: "-", with: "").prefix(12).lowercased())
        let session = StreamingSession(
            transcriber: transcriber,
            batchTranscriber: batchTranscriber,
            config: streamConfig,
            language: language,
            vocabularyHints: contextualStrings
        )
        let startedAt = Date()
        let managedSession = ManagedStreamingSession(
            session: session,
            startedAt: startedAt,
            language: language,
            contextualStrings: contextualStrings,
            recordsAudio: recordingConfiguration.enabled
        )
        stateLock.lock()
        sessions[sid] = managedSession
        stateLock.unlock()
        log("Session created: \(sid)")
        return sid
    }

    func feed(_ sid: String, pcmData: Data) -> [String: Any]? {
        guard let managedSession = sessionState(for: sid) else { return nil }

        return managedSession.gate.withActiveOperation {
            managedSession.lastActivity = Date()
            managedSession.recordingData?.append(pcmData)

            let samples = pcmData.withUnsafeBytes { buffer -> [Float] in
                let int16s = buffer.bindMemory(to: Int16.self)
                return int16s.map { Float($0) / 32768.0 }
            }
            managedSession.pendingAudio.append(contentsOf: samples)

            let session = managedSession.session
            var batchCorrected = false
            inferenceLock.lock()
            defer { inferenceLock.unlock() }

            while true {
                let currentChunkSamples = session.activeSegmentText().isEmpty
                    ? bootstrapChunkSamples
                    : chunkSamples
                guard let chunk = managedSession.pendingAudio.takePrefix(currentChunkSamples) else { break }

                let speechHint = analyzeSpeechActivity(chunk, vad: vad)
                let result = session.processChunk(chunk, speechHint: speechHint)
                managedSession.preview.canonicalChunkProcessed(text: result.text)
                if result.batchCorrected { batchCorrected = true }
                processedChunksSinceCacheTrim += 1
                if processedChunksSinceCacheTrim >= cacheTrimChunkInterval {
                    Self.trimMLXCache()
                    processedChunksSinceCacheTrim = 0
                }
#if YUWP_INTERNAL_DIAGNOSTICS
                log(
                    "PERF sid=\(sid) chunk=\(session.processedChunkCount) "
                        + "samples=\(chunk.count) "
                        + "speech_hint=\(speechHint.map { $0.hasSpeech ? 1 : 0 } ?? -1) "
                        + "speech_sec=\(speechHint?.speechDurationSec ?? -1) "
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

            attemptPreview(managedSession, sid: sid)
            return transcriptPayload(
                session: session,
                kind: batchCorrected ? "segment_commit" : "partial",
                isFinal: false,
                batchCorrected: batchCorrected,
                preview: managedSession.preview
            )
        }
    }

    /// Called only while the session gate and inferenceLock are held. The
    /// temporary decoder shares weights, never audio position or canonical KV.
    var chunkSec: Double { streamConfig.chunkSec }
    var decodeMode: StreamDecodeMode { streamConfig.decodeMode }

    private func attemptPreview(_ managedSession: ManagedStreamingSession, sid: String) {
        guard !streamConfig.isStablePrefix,
              let vad,
              managedSession.preview.reserveInspection(
                  pendingSamples: managedSession.pendingAudio.count,
                  canonicalChunkSamples: bootstrapChunkSamples
              ),
              let audio = managedSession.pendingAudio.peekPrefix(FirstPartialPreview.sampleCount),
              let speechHint = analyzeSpeechActivity(audio, vad: vad),
              managedSession.preview.reserveDecode(speechHint: speechHint)
        else { return }

        let previewSession = StreamingSession(
            transcriber: transcriber,
            batchTranscriber: batchTranscriber,
            config: StreamConfig(
                maxNewTokens: FirstPartialPreview.maxNewTokens,
                batchRetranscribe: batchRetranscribeEnabled
            ),
            language: managedSession.language,
            vocabularyHints: managedSession.contextualStrings
        )
        let result = previewSession.processChunk(audio, speechHint: speechHint)
        managedSession.preview.accept(text: result.text)
#if YUWP_INTERNAL_DIAGNOSTICS
        log(
            "PERF sid=\(sid) preview=1 samples=\(audio.count) "
                + "speech_hint=1 speech_sec=\(speechHint.speechDurationSec) "
                + "encode_ms=\(Int(result.encodeMs.rounded())) "
                + "prefill_ms=\(Int(result.prefillMs.rounded())) "
                + "decode_ms=\(Int(result.decodeMs.rounded())) "
                + "total_ms=\(Int(result.totalMs.rounded())) "
                + "text_len=\(result.text.count)"
        )
#endif
    }

    func stop(_ sid: String) -> [String: Any]? {
        guard let managedSession = sessionState(for: sid) else { return nil }

        let stopped = managedSession.gate.close { () -> (response: [String: Any], text: String, recording: Data?) in
            removeSession(sid, matching: managedSession)
            let pending = managedSession.pendingAudio.drain()
            let session = managedSession.session

            inferenceLock.lock()
            defer { inferenceLock.unlock() }
            let speechHint = pending.isEmpty ? nil : analyzeSpeechActivity(pending, vad: vad)
            let finished = session.finishOnStop(pendingAudio: pending, speechHint: speechHint)
            let text = finished.text
            let response = transcriptPayload(
                session: session,
                kind: "final",
                isFinal: true,
                batchCorrected: false
            )
#if YUWP_INTERNAL_DIAGNOSTICS
            log(
                "PERF sid=\(sid) stop_skip_provisional=\(finished.skippedProvisionalDecode ? 1 : 0) "
                    + "stop_fallback_decode=\(finished.usedFallbackDecode ? 1 : 0) "
                    + "pending_samples=\(finished.pendingSampleCount) "
                    + "session_samples=\(finished.sessionSampleCount)"
            )
#endif
            return (response, text, managedSession.recordingData)
        }
        guard let stopped else { return nil }

        if isSessionRegistryEmpty {
            inferenceLock.lock()
            Self.trimMLXCache()
            inferenceLock.unlock()
        }

        persistRecordingIfNeeded(
            sid: sid,
            pcmData: stopped.recording,
            transcript: stopped.text,
            language: managedSession.language,
            startedAt: managedSession.recordingStartedAt
        )

        log("Session stopped (\(sid)): \(stopped.text.count) chars")
        return stopped.response
    }

    private func sessionState(for sid: String) -> ManagedStreamingSession? {
        stateLock.lock()
        defer { stateLock.unlock() }
        return sessions[sid]
    }

    private func removeSession(_ sid: String, matching expected: ManagedStreamingSession) {
        stateLock.lock()
        if sessions[sid] === expected {
            sessions.removeValue(forKey: sid)
        }
        stateLock.unlock()
    }

    private var isSessionRegistryEmpty: Bool {
        stateLock.lock()
        defer { stateLock.unlock() }
        return sessions.isEmpty
    }

    private func transcriptPayload(
        session: StreamingSession,
        kind: String,
        isFinal: Bool,
        batchCorrected: Bool,
        preview: FirstPartialPreview? = nil
    ) -> [String: Any] {
        let canonicalText = session.finalText()
        let canonicalActiveText = isFinal ? "" : session.activeSegmentText()
        let visibleText = preview?.visibleText(canonicalText: canonicalText, isFinal: isFinal) ?? canonicalText
        let showsPreview = !isFinal && canonicalText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !visibleText.isEmpty
        var response: [String: Any] = [
            "text": visibleText,
            "committed_text": session.committedSegmentText(),
            "active_text": isFinal ? "" : (preview?.visibleText(canonicalText: canonicalActiveText, isFinal: false) ?? canonicalActiveText),
            "update_kind": showsPreview ? "partial" : kind,
            "is_final": isFinal,
        ]
        if showsPreview { response["batch_corrected"] = false }
        else if batchCorrected { response["batch_corrected"] = true }
        return response
    }

    func transcribeAudio(
        audio: [Float],
        language: String? = nil,
        temperature: Float = 0.0
    ) throws -> TranscriptionResult {
        try BatchTranscriptionPipeline.transcribe(
            using: self,
            audio: audio,
            language: language,
            temperature: temperature,
            vad: batchVAD,
            chunking: batchChunking,
            log: log
        )
    }

    public var hasR2T2BatchDelimiter: Bool {
        StreamConfig.isR2T2Model(at: (batchTranscriber ?? transcriber).modelDirectory)
    }

    func transcribeChunk(
        audio: [Float],
        language: String? = nil,
        temperature: Float = 0.0
    ) throws -> TranscriptionResult {
        let transcriber = batchTranscriber ?? self.transcriber
        inferenceLock.lock()
        defer { inferenceLock.unlock() }
        return try transcriber.transcribe(audio: audio, language: language, temperature: temperature)
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

    private func persistRecordingIfNeeded(
        sid: String,
        pcmData: Data?,
        transcript: String,
        language: String?,
        startedAt: Date?
    ) {
        guard recordingConfiguration.enabled,
              let pcmData,
              !pcmData.isEmpty
        else { return }

        let config = recordingConfiguration
        let date = startedAt ?? Date()
        DispatchQueue.global(qos: .utility).async {
            let context = ASRStreamRecordingContext(
                sessionID: sid,
                transcriptionModel: config.transcriptionModel,
                languageHint: language
            )
            do {
                let artifact = try ASRStreamRecordingArtifactWriter.writeRecording(
                    pcmData: pcmData,
                    directory: config.directory,
                    context: context,
                    date: date
                )
                try ASRStreamRecordingArtifactWriter.writeTranscript(
                    transcript,
                    for: artifact,
                    context: context
                )
                log(
                    "ASR recording saved: sid=\(sid) path=\(artifact.audioURL.path) "
                        + "(\(String(format: "%.1f", artifact.durationSeconds))s)"
                )
            } catch {
                log("ASR recording save failed: sid=\(sid) error=\(error.localizedDescription)")
            }
        }
    }

    private func cleanupExpired() {
        let now = Date()
        stateLock.lock()
        let snapshot = Array(sessions)
        stateLock.unlock()

        var expiredCount = 0
        for (sid, managedSession) in snapshot {
            let expired = managedSession.gate.closeIf(
                { now.timeIntervalSince(managedSession.lastActivity) > sessionTimeout },
                {
                    removeSession(sid, matching: managedSession)
                    managedSession.pendingAudio = StreamingAudioAccumulator()
                    managedSession.recordingData = nil
                    return true
                }
            ) ?? false
            if expired { expiredCount += 1 }
        }

        if expiredCount > 0 {
            log("Expired \(expiredCount) session(s)")
        }
        if expiredCount > 0, isSessionRegistryEmpty {
            inferenceLock.lock()
            Self.trimMLXCache()
            inferenceLock.unlock()
        }
    }

    private static func trimMLXCache() {
        MLX.Memory.clearCache()
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

private enum StdioFrameReadResult {
    case frame(ASRIPCFrame)
    case eof
    case invalidHeader
    case truncated
}

private func readExactBytes(fd: Int32, count: Int) -> Data? {
    guard count >= 0 else { return nil }
    if count == 0 { return Data() }

    var buffer = [UInt8](repeating: 0, count: count)
    var offset = 0

    while offset < count {
        let readCount = buffer.withUnsafeMutableBytes { rawBuffer -> Int in
            guard let base = rawBuffer.baseAddress else { return -1 }
            return Darwin.read(fd, base.advanced(by: offset), count - offset)
        }

        if readCount == 0 { return nil }
        if readCount < 0 {
            if errno == EINTR { continue }
            return nil
        }

        offset += readCount
    }

    return Data(buffer)
}

private func writeAllBytes(fd: Int32, data: Data) -> Bool {
    var offset = 0
    return data.withUnsafeBytes { rawBuffer in
        guard let base = rawBuffer.baseAddress else { return true }

        while offset < data.count {
            let written = Darwin.write(fd, base.advanced(by: offset), data.count - offset)
            if written < 0 {
                if errno == EINTR { continue }
                return false
            }
            if written == 0 { return false }
            offset += written
        }
        return true
    }
}

private func readStdioFrame() -> StdioFrameReadResult {
    guard let header = readExactBytes(fd: STDIN_FILENO, count: ASRIPCFrameCodec.headerSize) else {
        return .eof
    }
    guard let lengths = ASRIPCFrameCodec.decodeHeader(header) else {
        return .invalidHeader
    }

    guard let metadata = readExactBytes(fd: STDIN_FILENO, count: lengths.metadataLength),
          let binary = readExactBytes(fd: STDIN_FILENO, count: lengths.binaryLength)
    else {
        return .truncated
    }

    return .frame(ASRIPCFrame(metadata: metadata, binary: binary))
}

private func makeIPCResponse(id: UInt64, from payload: [String: Any]) -> ASRIPCResponse {
    ASRIPCResponse(
        id: id,
        ok: true,
        text: payload["text"] as? String,
        committedText: payload["committed_text"] as? String,
        activeText: payload["active_text"] as? String,
        updateKind: payload["update_kind"] as? String,
        batchCorrected: payload["batch_corrected"] as? Bool,
        isFinal: payload["is_final"] as? Bool
    )
}

private func writeIPCResponse(_ response: ASRIPCResponse) -> Bool {
    guard let frame = try? ASRIPCCodec.encode(response) else { return false }
    return writeAllBytes(fd: STDOUT_FILENO, data: frame)
}

func startStdioServer(
    mgr: StreamingSessionManager,
    streamingModelName: String,
    activeModelID: String?,
    batchRetranscribeEnabled: Bool
) {
    log("Listening on stdio")

    while true {
        let request: ASRIPCRequest
        switch readStdioFrame() {
        case .eof:
            log("Stdio input closed")
            return
        case .invalidHeader:
            let response = ASRIPCResponse(id: 0, ok: false, error: "invalid frame header")
            if !writeIPCResponse(response) { return }
            continue
        case .truncated:
            let response = ASRIPCResponse(id: 0, ok: false, error: "truncated stdio frame")
            _ = writeIPCResponse(response)
            return
        case .frame(let frame):
            do {
                request = try ASRIPCCodec.decodeRequest(metadata: frame.metadata)
            } catch {
                let response = ASRIPCResponse(id: 0, ok: false, error: "invalid request: \(error.localizedDescription)")
                if !writeIPCResponse(response) { return }
                continue
            }

            let response: ASRIPCResponse
            switch request.command {
            case .info:
                response = ASRIPCResponse(
                    id: request.id,
                    ok: true,
                    status: "ready",
                    model: activeModelID ?? streamingModelName,
                    sampleRate: ASRAudio.sampleRate,
                    chunkSec: mgr.chunkSec,
                    finalAccuracyPassEnabled: batchRetranscribeEnabled
                )
            case .create:
                response = ASRIPCResponse(
                    id: request.id,
                    ok: true,
                    sessionID: mgr.create(
                        language: request.language,
                        contextualStrings: request.contextualStrings
                    )
                )
            case .feed:
                guard let sid = request.sessionID else {
                    response = ASRIPCResponse(id: request.id, ok: false, error: "missing session_id")
                    break
                }
                guard let payload = mgr.feed(sid, pcmData: frame.binary) else {
                    response = ASRIPCResponse(id: request.id, ok: false, error: "session not found")
                    break
                }
                response = makeIPCResponse(id: request.id, from: payload)
            case .stop:
                guard let sid = request.sessionID else {
                    response = ASRIPCResponse(id: request.id, ok: false, error: "missing session_id")
                    break
                }
                guard let payload = mgr.stop(sid) else {
                    response = ASRIPCResponse(id: request.id, ok: false, error: "session not found")
                    break
                }
                response = makeIPCResponse(id: request.id, from: payload)
            }

            if !writeIPCResponse(response) {
                log("Failed to write stdio response")
                return
            }
        }
    }
}

func startServer(
    host: String,
    port: UInt16,
    mgr: StreamingSessionManager,
    aligner: ForcedAligner?,
    vad: SileroVAD?,
    batchVAD: SileroVAD?,
    batchChunking: BatchChunkingMode,
    streamingModelName: String,
    activeModelID: String?,
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
        batchVAD: batchVAD,
        batchChunking: batchChunking,
        streamingModelName: streamingModelName,
        activeModelID: activeModelID,
        batchModelName: batchModelName,
        batchRetranscribeEnabled: batchRetranscribeEnabled,
        chunkSec: mgr.chunkSec,
        decodeMode: mgr.decodeMode.rawValue,
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

private let diagnosticLoggingEnabled: Bool = {
    let raw = ProcessInfo.processInfo.environment["YUWP_DIAGNOSTIC_LOGGING"]?
        .trimmingCharacters(in: .whitespacesAndNewlines)
        .lowercased()
    return raw == "1" || raw == "true" || raw == "yes" || raw == "on"
}()

func log(_ message: String) {
    guard diagnosticLoggingEnabled else { return }
    fputs("[yuwp-asr] \(message)\n", stderr)
}
