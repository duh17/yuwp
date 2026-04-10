import Foundation

private final class LockedBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: T

    init(_ value: T) {
        storage = value
    }

    func set(_ value: T) {
        lock.lock()
        storage = value
        lock.unlock()
    }

    func get() -> T {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }
}

// MARK: - ASR Server State

/// State of the ASR server process, observed by the menu bar.
enum ASRServerState: Sendable, Equatable {
    case disabled       // server mode is off
    case stopped
    case starting       // process launched, model loading
    case ready          // accepting dictation
    case error(String)  // crashed or failed to start
}

// MARK: - Native ASR Provider

/// Manages the native ASR server process (asr-server).
/// Communicates via HTTP on localhost.
///
/// Launches `asr-server` as a child process, monitors its health,
/// and provides STT sessions via the HTTP streaming API.
final class NativeASRProvider: @unchecked Sendable, SttProvider {
    // SttProvider
    var isReady: Bool { state == .ready }
    var onReady: (@Sendable () -> Void)?
    var onError: (@Sendable (String) -> Void)?

    // State observation (for menu bar)
    private(set) var state: ASRServerState = .stopped
    var onStateChange: (@Sendable (ASRServerState) -> Void)?

    /// Transcription model spec: Hugging Face repo id or local model directory.
    var transcriptionModel: String = "mlx-community/Qwen3-ASR-0.6B-4bit"
    /// Whether pause/final batch commit is enabled.
    var batchCommitEnabled: Bool = true

    /// Hidden default aligner model used to power `/v1/audio/subtitles` when available locally.
    private static let defaultAlignerModel = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

    // Process management
    private var process: Process?
    private var readyPollTask: Task<Void, Never>?
    private var restartTask: Task<Void, Never>?

    // Crash recovery
    private var isIntentionalShutdown = false
    private var restartAttempts = 0
    private static let maxRestartAttempts = 5

    var port: UInt16
    var serverMode: ServerMode = .localhost

    private var bindHost: String? { serverMode.bindHost }
    private var clientHost: String { serverMode.clientHost }

    init(port: UInt16 = 9748) {
        self.port = port
    }

    // MARK: - Lifecycle

    func start() {
        isIntentionalShutdown = false

        guard let bindHost else {
            updateState(.disabled)
            yuwpLog("asr-server disabled")
            return
        }

        updateState(.starting)

        guard let transcriptionModelPath = Self.resolveModelPath(transcriptionModel) else {
            yuwpLog("Transcription model not found: \(transcriptionModel)")
            updateState(.error("Transcription model not found"))
            return
        }

        guard let serverBin = Self.findServerBinary() else {
            yuwpLog("asr-server binary not found — run: swift build -c release --product asr-server")
            updateState(.error("asr-server not found"))
            return
        }

        let proc = Process()
        let stderrPipe = Pipe()
        let alignerModelPath = Self.resolveModelPath(Self.defaultAlignerModel)

        proc.executableURL = URL(fileURLWithPath: serverBin)
        var arguments = [transcriptionModelPath, "--port", "\(port)", "--host", bindHost]
        if batchCommitEnabled {
            arguments += ["--batch-model", transcriptionModelPath]
        } else {
            arguments += ["--disable-batch-retranscribe"]
        }
        if let alignerModelPath {
            arguments += ["--aligner-model", alignerModelPath]
        } else {
            yuwpLog("Aligner model not found locally: \(Self.defaultAlignerModel) — subtitles disabled")
        }
        proc.arguments = arguments
        proc.standardInput = FileHandle.nullDevice
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = stderrPipe

        // Forward server stderr to our stderr
        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty { FileHandle.standardError.write(data) }
        }

        // Monitor for unexpected exit
        proc.terminationHandler = { [weak self] proc in
            guard let self, !self.isIntentionalShutdown else { return }
            let code = proc.terminationStatus
            yuwpLog("asr-server exited unexpectedly (code \(code))")
            self.process = nil
            self.updateState(.error("Server crashed (exit \(code))"))
            self.scheduleRestart()
        }

        do {
            try proc.run()
        } catch {
            yuwpLog("Failed to start asr-server: \(error)")
            updateState(.error("Failed to start server"))
            scheduleRestart()
            return
        }

        process = proc
        if let alignerModelPath {
            yuwpLog("asr-server started (PID: \(proc.processIdentifier), aligner: \(URL(fileURLWithPath: alignerModelPath).lastPathComponent))")
        } else {
            yuwpLog("asr-server started (PID: \(proc.processIdentifier))")
        }

        // Poll for readiness
        readyPollTask = Task.detached { [weak self] in
            for _ in 0..<60 {  // 30s timeout (60 × 500ms)
                try? await Task.sleep(for: .milliseconds(500))
                guard let self, !Task.isCancelled else { return }
                if self.checkReady() {
                    self.restartAttempts = 0
                    self.updateState(.ready)
                    return
                }
            }
            guard let self, !Task.isCancelled else { return }
            yuwpLog("asr-server failed to become ready within 30s")
            self.updateState(.error("Server startup timeout"))
        }
    }

    func shutdown() {
        isIntentionalShutdown = true
        readyPollTask?.cancel()
        readyPollTask = nil
        restartTask?.cancel()
        restartTask = nil

        if let proc = process, proc.isRunning {
            kill(proc.processIdentifier, SIGTERM)
            proc.waitUntilExit()
        }

        process = nil
        updateState(serverMode == .off ? .disabled : .stopped)
        yuwpLog(serverMode == .off ? "asr-server disabled" : "asr-server stopped")
    }

    // MARK: - SttProvider

    func makeSession() -> any SttSession {
        NativeASRSession(host: clientHost, port: port)
    }

    // MARK: - State

    private func updateState(_ newState: ASRServerState) {
        state = newState
        onStateChange?(newState)
        if case .ready = newState { onReady?() }
    }

    // MARK: - Crash Recovery

    private func scheduleRestart() {
        guard !isIntentionalShutdown else { return }
        guard restartAttempts < Self.maxRestartAttempts else {
            yuwpLog("Max restart attempts reached (\(Self.maxRestartAttempts))")
            updateState(.error("Server failed after \(Self.maxRestartAttempts) attempts"))
            return
        }
        restartAttempts += 1
        let delay = min(Double(1 << restartAttempts), 30.0)
        yuwpLog("Restarting asr-server in \(Int(delay))s (\(restartAttempts)/\(Self.maxRestartAttempts))")

        restartTask = Task.detached { [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            guard let self, !Task.isCancelled else { return }
            self.start()
        }
    }

    // MARK: - Health Check

    private func checkReady() -> Bool {
        guard let url = URL(string: "http://\(clientHost):\(port)/v1/info") else { return false }
        var req = URLRequest(url: url)
        req.timeoutInterval = 2
        let ready = LockedBox(false)
        let sema = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: req) { data, _, _ in
            defer { sema.signal() }
            guard let data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  json["status"] as? String == "ready" else { return }
            ready.set(true)
        }.resume()
        sema.wait()
        return ready.get()
    }

    // MARK: - Model + Binary Resolution

    /// Resolve a model spec (Hugging Face repo id or local directory) to a local directory path.
    static func resolveModelPath(_ spec: String) -> String? {
        ModelLocator.resolve(spec)?.path
    }

    /// Find the asr-server binary in expected locations.
    static func findServerBinary() -> String? {
        let candidates = [
            // App bundle
            Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/asr-server").path,
            // Development: build directory (release preferred)
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent(".build/arm64-apple-macosx/release/asr-server").path,
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent(".build/arm64-apple-macosx/debug/asr-server").path,
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
    }
}

// MARK: - Native ASR HTTP Session

/// HTTP-based STT session communicating with asr-server.
/// Audio feeds are serialized on a background queue to avoid blocking the audio thread.
final class NativeASRSession: SttSession, @unchecked Sendable {
    var onPartial: ((String) -> Void)?
    var onSegmentCommit: ((String) -> Void)?
    var onFinal: ((String) -> Void)?
    var onError: ((String) -> Void)?

    private let baseURL: String
    private var sessionId: String?
    private let queue = DispatchQueue(label: "yuwp.asr-session", qos: .userInitiated)

    init(host: String, port: UInt16) {
        self.baseURL = "http://\(host):\(port)/v1/audio/transcriptions/stream"
    }

    func begin(language: String?) {
        queue.async { [weak self] in
            guard let self else { return }
            guard let data = self.syncHTTP("POST", path: self.baseURL),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let sid = json["session_id"] as? String else {
                self.onError?("Failed to create ASR session")
                return
            }
            self.sessionId = sid
        }
    }

    func feedAudio(_ pcmData: Data) {
        queue.async { [weak self] in
            guard let self, let sid = self.sessionId else { return }
            guard let data = self.syncHTTP("POST", path: "\(self.baseURL)/\(sid)", body: pcmData),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let text = json["text"] as? String, !text.isEmpty else { return }
            if json["batch_corrected"] as? Bool == true {
                self.onSegmentCommit?(text)
            } else {
                self.onPartial?(text)
            }
        }
    }

    func end() {
        queue.async { [weak self] in
            guard let self else { return }
            guard let sid = self.sessionId else {
                self.onFinal?("")
                return
            }
            self.sessionId = nil
            guard let data = self.syncHTTP("DELETE", path: "\(self.baseURL)/\(sid)"),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let text = json["text"] as? String else {
                self.onFinal?("")
                return
            }
            self.onFinal?(text)
        }
    }

    /// Synchronous HTTP request (always called on the serial background queue).
    private func syncHTTP(_ method: String, path: String, body: Data? = nil) -> Data? {
        guard let url = URL(string: path) else { return nil }
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.timeoutInterval = 30
        req.httpBody = body
        let result = LockedBox<Data?>(nil)
        let sema = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: req) { data, response, _ in
            defer { sema.signal() }
            guard let data, let http = response as? HTTPURLResponse, http.statusCode == 200 else { return }
            result.set(data)
        }.resume()
        sema.wait()
        return result.get()
    }
}
