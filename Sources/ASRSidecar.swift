import Foundation

/// Sidecar process state, observed by the menu bar.
enum SidecarState: Sendable, Equatable {
    case stopped
    case starting       // process launched, model loading
    case ready          // accepting dictation
    case error(String)  // crashed or failed to start
}

/// Manages the Python ASR sidecar process.
/// Communicates via JSON lines over stdin/stdout.
/// Monitors process health and auto-restarts on unexpected exit.
///
/// Swift sends:  {"cmd": "start"}, {"cmd": "audio", "pcm_b64": "..."}, {"cmd": "stop"}
/// Python sends: {"type": "ready"}, {"type": "partial", "text": "..."}, {"type": "final", "text": "..."}
final class ASRSidecar: @unchecked Sendable, SttProvider {
    // SttProvider
    var isReady: Bool { state == .ready }
    var onReady: (@Sendable () -> Void)?
    var onError: (@Sendable (String) -> Void)?

    // State observation (for menu bar)
    private(set) var state: SidecarState = .stopped
    var onStateChange: (@Sendable (SidecarState) -> Void)?

    // Internal — forwarded to the active SttSession
    var onPartialResult: (@Sendable (String) -> Void)?
    var onFinalResult: (@Sendable (String) -> Void)?

    /// Model for streaming partials (low-latency).
    var streamingModel: String = "mlx-community/Qwen3-ASR-0.6B-4bit"
    /// Model for batch retranscription (high-quality correction).
    var batchModel: String = "mlx-community/Qwen3-ASR-1.7B-bf16"
    /// Whether to run batch retranscription for quality correction.
    var batchRetranscribeEnabled: Bool = true

    // Process management
    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var readTask: Task<Void, Never>?

    // Crash recovery
    private var isIntentionalShutdown = false
    private var restartAttempts = 0
    private var restartTask: Task<Void, Never>?
    private static let maxRestartAttempts = 5

    private let sidecarPath: String

    init() {
        // Locate the sidecar script relative to the executable or in the source tree
        let bundle = Bundle.main.bundlePath
        let candidates = [
            // Development: source tree
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .appendingPathComponent("sidecar/transcribe.py").path,
            // App bundle: Contents/Resources/sidecar/
            URL(fileURLWithPath: bundle)
                .deletingLastPathComponent()
                .appendingPathComponent("../Resources/sidecar/transcribe.py").standardized.path,
            // Installed alongside binary
            URL(fileURLWithPath: bundle)
                .deletingLastPathComponent()
                .appendingPathComponent("sidecar/transcribe.py").path,
            // Fallback: workspace path
            NSString("~/workspace/yuwp/Sources/sidecar/transcribe.py")
                .expandingTildeInPath,
        ]
        sidecarPath = candidates.first { FileManager.default.fileExists(atPath: $0) }
            ?? candidates.last!
    }

    // MARK: - Lifecycle

    func start() {
        isIntentionalShutdown = false
        updateState(.starting)

        let proc = Process()
        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()

        // Use uv to run the script with inline dependencies.
        // --serve starts the HTTP server alongside stdio for external clients.
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        var args = ["uv", "run", "--script", sidecarPath, streamingModel, "--serve"]
        if batchRetranscribeEnabled {
            args += ["--batch-model", batchModel]
        } else {
            args.append("--no-batch-retranscribe")
        }
        proc.arguments = args
        proc.standardInput = stdin
        proc.standardOutput = stdout
        proc.standardError = stderr

        // Forward sidecar stderr to our stderr
        stderr.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty {
                FileHandle.standardError.write(data)
            }
        }

        do {
            try proc.run()
        } catch {
            yuwpLog("Failed to start sidecar: \(error)")
            updateState(.error("Failed to start sidecar"))
            scheduleRestart()
            return
        }

        process = proc
        stdinPipe = stdin
        stdoutPipe = stdout

        yuwpLog("ASR sidecar started (PID: \(proc.processIdentifier))")

        // Read stdout line by line in a background task
        readTask = Task.detached { [weak self] in
            let handle = stdout.fileHandleForReading
            var buffer = Data()

            while let self, self.process?.isRunning == true {
                let chunk = handle.availableData
                if chunk.isEmpty { break } // EOF
                buffer.append(chunk)

                // Process complete lines
                while let newlineRange = buffer.range(of: Data([0x0A])) {
                    let lineData = buffer[buffer.startIndex..<newlineRange.lowerBound]
                    buffer.removeSubrange(buffer.startIndex...newlineRange.lowerBound)

                    guard let line = String(data: lineData, encoding: .utf8),
                          !line.isEmpty,
                          let json = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                          let type = json["type"] as? String else { continue }

                    switch type {
                    case "ready":
                        self.restartAttempts = 0
                        self.updateState(.ready)
                    case "partial":
                        if let text = json["text"] as? String {
                            self.onPartialResult?(text)
                        }
                    case "final":
                        if let text = json["text"] as? String {
                            self.onFinalResult?(text)
                        }
                    case "error":
                        let msg = json["message"] as? String ?? "Unknown sidecar error"
                        self.onError?(msg)
                    default:
                        break
                    }
                }
            }

            // Process exited — detect unexpected crash
            guard let self, !self.isIntentionalShutdown else { return }
            let code = self.process?.terminationStatus ?? -1
            yuwpLog("Sidecar exited unexpectedly (code \(code))")
            self.process = nil
            self.stdinPipe = nil
            self.stdoutPipe = nil
            self.updateState(.error("Sidecar crashed (exit \(code))"))
            self.scheduleRestart()
        }
    }

    func shutdown() {
        isIntentionalShutdown = true
        restartTask?.cancel()
        restartTask = nil
        readTask?.cancel()
        readTask = nil

        send(["cmd": "quit"])
        process?.terminate()
        process?.waitUntilExit() // fast after terminate

        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        updateState(.stopped)
        yuwpLog("ASR sidecar stopped")
    }

    // MARK: - State Management

    private func updateState(_ newState: SidecarState) {
        state = newState
        onStateChange?(newState)
        if case .ready = newState {
            onReady?()
        }
    }

    // MARK: - Crash Recovery

    private func scheduleRestart() {
        guard !isIntentionalShutdown else { return }
        guard restartAttempts < Self.maxRestartAttempts else {
            yuwpLog("Max restart attempts (\(Self.maxRestartAttempts)) reached")
            updateState(.error("Sidecar failed after \(Self.maxRestartAttempts) attempts"))
            return
        }

        restartAttempts += 1
        let delay = min(Double(1 << restartAttempts), 30.0) // 2, 4, 8, 16, 30s
        yuwpLog("Restarting sidecar in \(Int(delay))s (attempt \(restartAttempts)/\(Self.maxRestartAttempts))")

        restartTask = Task.detached { [weak self] in
            try? await Task.sleep(for: .seconds(delay))
            guard let self, !Task.isCancelled else { return }
            self.start()
        }
    }

    // MARK: - Session Commands

    func beginSession(language: String? = nil) {
        var msg: [String: Any] = ["cmd": "start"]
        if let language { msg["language"] = language }
        send(msg)
    }

    func sendAudio(_ pcmData: Data) {
        let b64 = pcmData.base64EncodedString()
        send(["cmd": "audio", "pcm_b64": b64])
    }

    func endSession() {
        send(["cmd": "stop"])
    }

    // MARK: - SttProvider

    func makeSession() -> any SttSession {
        SidecarSttSession(sidecar: self)
    }

    private func send(_ msg: [String: Any]) {
        guard let pipe = stdinPipe,
              let data = try? JSONSerialization.data(withJSONObject: msg),
              var line = String(data: data, encoding: .utf8) else { return }
        line += "\n"
        pipe.fileHandleForWriting.write(Data(line.utf8))
    }
}

// MARK: - Sidecar STT Session

/// Wraps ASRSidecar's per-session operations into the SttSession protocol.
/// The sidecar process is shared across sessions (one model load, many sessions).
final class SidecarSttSession: SttSession, @unchecked Sendable {
    var onPartial: ((String) -> Void)?
    var onFinal: ((String) -> Void)?
    var onError: ((String) -> Void)?

    private let sidecar: ASRSidecar

    init(sidecar: ASRSidecar) {
        self.sidecar = sidecar
    }

    func begin(language: String?) {
        // Wire sidecar callbacks to this session's callbacks
        sidecar.onPartialResult = { [weak self] text in
            Task { @MainActor in self?.onPartial?(text) }
        }
        sidecar.onFinalResult = { [weak self] text in
            Task { @MainActor in self?.onFinal?(text) }
        }
        sidecar.onError = { [weak self] msg in
            Task { @MainActor in self?.onError?(msg) }
        }
        sidecar.beginSession(language: language)
    }

    func feedAudio(_ pcmData: Data) {
        sidecar.sendAudio(pcmData)
    }

    func end() {
        sidecar.endSession()
    }
}
