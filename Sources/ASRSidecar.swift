import Foundation

/// Manages the Python ASR sidecar process.
/// Communicates via JSON lines over stdin/stdout.
///
/// Swift sends:  {"cmd": "start"}, {"cmd": "audio", "pcm_b64": "..."}, {"cmd": "stop"}
/// Python sends: {"type": "ready"}, {"type": "partial", "text": "..."}, {"type": "final", "text": "..."}
final class ASRSidecar: @unchecked Sendable, SttProvider {
    // SttProvider
    var isReady: Bool { _isReady }
    var onReady: (@Sendable () -> Void)?
    var onError: (@Sendable (String) -> Void)?

    // Internal — forwarded to the active SttSession
    var onPartialResult: (@Sendable (String) -> Void)?
    var onFinalResult: (@Sendable (String) -> Void)?

    private var _isReady = false

    private var process: Process?
    private var stdinPipe: Pipe?
    private var stdoutPipe: Pipe?
    private var readTask: Task<Void, Never>?

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

    func start() {
        let proc = Process()
        let stdin = Pipe()
        let stdout = Pipe()
        let stderr = Pipe()

        // Use uv to run the script with inline dependencies.
        // --serve starts the HTTP server alongside stdio for external clients.
        proc.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        proc.arguments = ["uv", "run", "--script", sidecarPath, "--serve"]
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
            onError?("Failed to start ASR sidecar: \(error)")
            return
        }

        process = proc
        stdinPipe = stdin
        stdoutPipe = stdout

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
                        self._isReady = true
                        self.onReady?()
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
        }

        yuwpLog("ASR sidecar started (PID: \(proc.processIdentifier))")
    }

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

    func shutdown() {
        send(["cmd": "quit"])
        readTask?.cancel()
        readTask = nil
        process?.terminate()
        process = nil
        stdinPipe = nil
        stdoutPipe = nil
        yuwpLog("ASR sidecar stopped")
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
