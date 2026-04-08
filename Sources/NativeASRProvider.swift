import Foundation

// MARK: - ASR Server State

/// State of the ASR server process, observed by the menu bar.
enum ASRServerState: Sendable, Equatable {
    case stopped
    case starting       // process launched, model loading
    case ready          // accepting dictation
    case error(String)  // crashed or failed to start
}

// MARK: - Native ASR Provider

/// Manages the native ASR server process (asr-server).
/// Communicates via HTTP on localhost — replaces the Python sidecar.
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

    /// HuggingFace model ID for streaming (resolved to local cache path).
    var streamingModel: String = "mlx-community/Qwen3-ASR-0.6B-4bit"
    /// Batch model (stored for API compat — native server uses same model for both).
    var batchModel: String = "mlx-community/Qwen3-ASR-1.7B-bf16"
    /// Batch retranscription (handled internally by native StreamingSession).
    var batchRetranscribeEnabled: Bool = true

    // Process management
    private var process: Process?
    private var readyPollTask: Task<Void, Never>?
    private var restartTask: Task<Void, Never>?

    // Crash recovery
    private var isIntentionalShutdown = false
    private var restartAttempts = 0
    private static let maxRestartAttempts = 5

    let port: UInt16
    let host: String

    init(port: UInt16 = 9748, host: String = "127.0.0.1") {
        self.port = port
        self.host = host
    }

    // MARK: - Lifecycle

    func start() {
        isIntentionalShutdown = false
        updateState(.starting)

        guard let modelPath = Self.resolveModelPath(streamingModel) else {
            yuwpLog("Model not found in HuggingFace cache: \(streamingModel)")
            updateState(.error("Model not found"))
            return
        }

        guard let serverBin = Self.findServerBinary() else {
            yuwpLog("asr-server binary not found — run: swift build -c release --product asr-server")
            updateState(.error("asr-server not found"))
            return
        }

        let proc = Process()
        let stderrPipe = Pipe()

        proc.executableURL = URL(fileURLWithPath: serverBin)
        proc.arguments = [modelPath, "--port", "\(port)", "--host", host]
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
        yuwpLog("asr-server started (PID: \(proc.processIdentifier))")

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
        updateState(.stopped)
        yuwpLog("asr-server stopped")
    }

    // MARK: - SttProvider

    func makeSession() -> any SttSession {
        NativeASRSession(host: host, port: port)
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
        guard let url = URL(string: "http://\(host):\(port)/v1/info") else { return false }
        var req = URLRequest(url: url)
        req.timeoutInterval = 2
        var ready = false
        let sema = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: req) { data, _, _ in
            defer { sema.signal() }
            guard let data,
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  json["status"] as? String == "ready" else { return }
            ready = true
        }.resume()
        sema.wait()
        return ready
    }

    // MARK: - Model + Binary Resolution

    /// Resolve a HuggingFace model ID (e.g. "mlx-community/Qwen3-ASR-0.6B-4bit")
    /// to a local cache directory path.
    static func resolveModelPath(_ modelId: String) -> String? {
        let parts = modelId.split(separator: "/")
        guard parts.count == 2 else { return nil }
        let cacheDir = NSString("~/.cache/huggingface/hub").expandingTildeInPath
        let snapshotsDir = "\(cacheDir)/models--\(parts[0])--\(parts[1])/snapshots"
        guard let snapshots = try? FileManager.default.contentsOfDirectory(atPath: snapshotsDir),
              let snapshot = snapshots.sorted().last else { return nil }
        return "\(snapshotsDir)/\(snapshot)"
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
            // Fallback: workspace path
            NSString("~/workspace/yuwp/.build/arm64-apple-macosx/release/asr-server")
                .expandingTildeInPath,
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
    }
}

// MARK: - Native ASR HTTP Session

/// HTTP-based STT session communicating with asr-server.
/// Audio feeds are serialized on a background queue to avoid blocking the audio thread.
final class NativeASRSession: SttSession, @unchecked Sendable {
    var onPartial: ((String) -> Void)?
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
            self.onPartial?(text)
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
        var result: Data?
        let sema = DispatchSemaphore(value: 0)
        URLSession.shared.dataTask(with: req) { data, response, _ in
            defer { sema.signal() }
            guard let data, let http = response as? HTTPURLResponse, http.statusCode == 200 else { return }
            result = data
        }.resume()
        sema.wait()
        return result
    }
}
