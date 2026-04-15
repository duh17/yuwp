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

private struct ASRServerConfiguration: Sendable, Equatable {
    var transcriptionModel: String = "mlx-community/Qwen3-ASR-0.6B-4bit"
    var batchCommitEnabled: Bool = true
    var diagnosticLoggingEnabled: Bool = false
    var port: UInt16
    var serverMode: ServerMode = .localhost

    var bindHost: String? { serverMode.bindHost }
    var clientHost: String { serverMode.clientHost }
}

private actor NativeASRServerLifecycle {
    private let stateSink: @Sendable (ASRServerState) -> Void

    private var process: Process?
    private var readyPollTask: Task<Void, Never>?
    private var restartTask: Task<Void, Never>?
    private var isIntentionalShutdown = false
    private var restartAttempts = 0
    private var launchGeneration: UInt64 = 0
    private var activeConfiguration: ASRServerConfiguration?

    private static let maxRestartAttempts = 5

    init(stateSink: @escaping @Sendable (ASRServerState) -> Void) {
        self.stateSink = stateSink
    }

    func start(configuration: ASRServerConfiguration) -> ASRServerState {
        launchGeneration &+= 1
        let generation = launchGeneration

        activeConfiguration = configuration
        isIntentionalShutdown = false
        cancelBackgroundTasks()

        guard let bindHost = configuration.bindHost else {
            process = nil
            yuwpLog("swift-mlx-asr-server disabled")
            return .disabled
        }

        guard let transcriptionModelPath = NativeASRProvider.resolveModelPath(configuration.transcriptionModel) else {
            yuwpLog("Transcription model not found: \(configuration.transcriptionModel)")
            process = nil
            return .error("Transcription model not found")
        }

        guard let serverBin = NativeASRProvider.findServerBinary() else {
            yuwpLog("swift-mlx-asr-server binary not found — run: swift build -c release --product swift-mlx-asr-server")
            process = nil
            return .error("swift-mlx-asr-server not found")
        }

        NativeASRProvider.cleanupOrphanedManagedServerIfNeeded(
            port: configuration.port,
            serverBinaryPath: serverBin
        )

        let proc = Process()
        let stderrPipe = Pipe()
        let alignerModelPath = NativeASRProvider.resolveModelPath(NativeASRProvider.defaultAlignerModel)

        proc.executableURL = URL(fileURLWithPath: serverBin)
        var arguments = [
            transcriptionModelPath,
            "--port", "\(configuration.port)",
            "--host", bindHost,
            "--parent-pid", "\(ProcessInfo.processInfo.processIdentifier)",
        ]
        if configuration.batchCommitEnabled {
            arguments += ["--batch-model", transcriptionModelPath]
        } else {
            arguments += ["--disable-batch-retranscribe"]
        }
        if let alignerModelPath {
            arguments += ["--aligner-model", alignerModelPath]
        } else {
            yuwpLog("Aligner model not found locally: \(NativeASRProvider.defaultAlignerModel) — subtitles disabled")
        }
        proc.arguments = arguments
        var childEnvironment = ProcessInfo.processInfo.environment
        childEnvironment["YUWP_DIAGNOSTIC_LOGGING"] = configuration.diagnosticLoggingEnabled ? "1" : "0"
        proc.environment = childEnvironment
        proc.standardInput = FileHandle.nullDevice
        proc.standardOutput = FileHandle.nullDevice
        proc.standardError = stderrPipe

        stderrPipe.fileHandleForReading.readabilityHandler = { handle in
            let data = handle.availableData
            if !data.isEmpty { FileHandle.standardError.write(data) }
        }

        proc.terminationHandler = { [weak self] proc in
            guard let self else { return }
            let pid = proc.processIdentifier
            let status = proc.terminationStatus
            Task.detached {
                await self.handleUnexpectedTermination(
                    processIdentifier: pid,
                    terminationStatus: status,
                    generation: generation,
                    port: configuration.port
                )
            }
        }

        do {
            try proc.run()
        } catch {
            process = nil
            yuwpLog("Failed to start swift-mlx-asr-server: \(error)")
            scheduleRestart(generation: generation)
            return .error("Failed to start server")
        }

        process = proc
        if let alignerModelPath {
            yuwpLog("swift-mlx-asr-server started (PID: \(proc.processIdentifier), aligner: \(URL(fileURLWithPath: alignerModelPath).lastPathComponent))")
        } else {
            yuwpLog("swift-mlx-asr-server started (PID: \(proc.processIdentifier))")
        }

        scheduleReadyPoll(generation: generation, configuration: configuration)
        return .starting
    }

    func shutdown(targetState: ASRServerState) -> ASRServerState {
        launchGeneration &+= 1
        isIntentionalShutdown = true
        activeConfiguration = nil
        restartAttempts = 0
        cancelBackgroundTasks()

        if let proc = process, proc.isRunning {
            kill(proc.processIdentifier, SIGTERM)
            proc.waitUntilExit()
        }

        process = nil
        yuwpLog(targetState == .disabled ? "swift-mlx-asr-server disabled" : "swift-mlx-asr-server stopped")
        return targetState
    }

    private func scheduleReadyPoll(generation: UInt64, configuration: ASRServerConfiguration) {
        readyPollTask?.cancel()
        readyPollTask = Task.detached { [weak self] in
            guard let self else { return }
            for _ in 0..<60 {
                try? await Task.sleep(for: .milliseconds(500))
                guard !Task.isCancelled else { return }
                if NativeASRProvider.checkReady(host: configuration.clientHost, port: configuration.port) {
                    await self.handleReady(generation: generation)
                    return
                }
            }
            guard !Task.isCancelled else { return }
            await self.handleStartupTimeout(generation: generation)
        }
    }

    private func handleReady(generation: UInt64) {
        guard generation == launchGeneration, !isIntentionalShutdown else { return }
        readyPollTask = nil
        restartAttempts = 0
        stateSink(.ready)
    }

    private func handleStartupTimeout(generation: UInt64) {
        guard generation == launchGeneration, !isIntentionalShutdown else { return }
        readyPollTask = nil
        yuwpLog("swift-mlx-asr-server failed to become ready within 30s")
        stateSink(.error("Server startup timeout"))
    }

    private func handleUnexpectedTermination(
        processIdentifier: Int32,
        terminationStatus: Int32,
        generation: UInt64,
        port: UInt16
    ) {
        guard generation == launchGeneration, !isIntentionalShutdown else { return }

        readyPollTask?.cancel()
        readyPollTask = nil
        if process?.processIdentifier == processIdentifier {
            process = nil
        }

        let code = terminationStatus
        if let listenerPID = NativeASRProvider.listeningPID(on: port), listenerPID != processIdentifier {
            let owner = NativeASRProvider.command(for: listenerPID) ?? "pid \(listenerPID)"
            yuwpLog("swift-mlx-asr-server failed to own port \(port); listener PID \(listenerPID): \(owner)")
            stateSink(.error("Port \(port) already in use"))
            return
        }

        yuwpLog("swift-mlx-asr-server exited unexpectedly (code \(code))")
        stateSink(.error("Server crashed (exit \(code))"))
        scheduleRestart(generation: generation)
    }

    private func scheduleRestart(generation: UInt64) {
        guard generation == launchGeneration, !isIntentionalShutdown else { return }
        guard let configuration = activeConfiguration else { return }
        guard restartAttempts < Self.maxRestartAttempts else {
            yuwpLog("Max restart attempts reached (\(Self.maxRestartAttempts))")
            stateSink(.error("Server failed after \(Self.maxRestartAttempts) attempts"))
            return
        }

        restartAttempts += 1
        let delay = min(Double(1 << restartAttempts), 30.0)
        yuwpLog("Restarting swift-mlx-asr-server in \(Int(delay))s (\(restartAttempts)/\(Self.maxRestartAttempts))")

        restartTask?.cancel()
        restartTask = Task.detached { [weak self] in
            guard let self else { return }
            try? await Task.sleep(for: .seconds(delay))
            guard !Task.isCancelled else { return }
            await self.restartIfNeeded(generation: generation, configuration: configuration)
        }
    }

    private func restartIfNeeded(generation: UInt64, configuration: ASRServerConfiguration) {
        guard generation == launchGeneration, !isIntentionalShutdown else { return }
        let state = start(configuration: configuration)
        stateSink(state)
    }

    private func cancelBackgroundTasks() {
        readyPollTask?.cancel()
        readyPollTask = nil
        restartTask?.cancel()
        restartTask = nil
    }
}

// MARK: - Native ASR Provider

/// Manages the native ASR server process (swift-mlx-asr-server).
/// Communicates via HTTP on localhost.
///
/// Launches `swift-mlx-asr-server` as a child process, monitors its health,
/// and provides STT sessions via the HTTP streaming API.
@MainActor
final class NativeASRProvider: SttProvider {
    // SttProvider
    var isReady: Bool { state == .ready }
    var onReady: (@MainActor @Sendable () -> Void)?
    var onError: (@MainActor @Sendable (String) -> Void)?

    // State observation (for menu bar)
    private(set) var state: ASRServerState = .stopped
    var onStateChange: (@MainActor @Sendable (ASRServerState) -> Void)?

    /// Default aligner model used to power timed transcription output when available locally.
    nonisolated static let defaultAlignerModel = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

    private var configuration: ASRServerConfiguration

    /// Transcription model spec: Hugging Face repo id or local model directory.
    var transcriptionModel: String {
        get { configuration.transcriptionModel }
        set { configuration.transcriptionModel = newValue }
    }

    /// Whether pause/final batch commit is enabled.
    var batchCommitEnabled: Bool {
        get { configuration.batchCommitEnabled }
        set { configuration.batchCommitEnabled = newValue }
    }

    /// Whether app/server diagnostic stderr logging is enabled.
    var diagnosticLoggingEnabled: Bool {
        get { configuration.diagnosticLoggingEnabled }
        set { configuration.diagnosticLoggingEnabled = newValue }
    }

    var port: UInt16 {
        get { configuration.port }
        set { configuration.port = newValue }
    }

    var serverMode: ServerMode {
        get { configuration.serverMode }
        set { configuration.serverMode = newValue }
    }

    private lazy var lifecycle = NativeASRServerLifecycle { [weak self] newState in
        Task { @MainActor [weak self] in
            self?.applyState(newState)
        }
    }

    private var lifecycleCommandTask: Task<Void, Never>?
    private var latestLifecycleCommandID: UInt64 = 0

    init(port: UInt16 = 9748) {
        self.configuration = ASRServerConfiguration(port: port)
    }

    // MARK: - Lifecycle

    func start() {
        let configuration = configuration
        applyState(preflightStartState(for: configuration))
        enqueueLifecycleCommand { lifecycle in
            await lifecycle.start(configuration: configuration)
        }
    }

    func shutdown() {
        let targetState: ASRServerState = configuration.serverMode == .off ? .disabled : .stopped
        applyState(targetState)
        enqueueLifecycleCommand { lifecycle in
            await lifecycle.shutdown(targetState: targetState)
        }
    }

    // MARK: - SttProvider

    func makeSession() -> any SttSession {
        NativeASRSession(host: configuration.clientHost, port: configuration.port)
    }

    // MARK: - State

    private func applyState(_ newState: ASRServerState) {
        guard state != newState else { return }

        state = newState
        onStateChange?(newState)
        if case .ready = newState {
            onReady?()
        }
        if case .error(let message) = newState {
            onError?(message)
        }
    }

    private func preflightStartState(for configuration: ASRServerConfiguration) -> ASRServerState {
        guard configuration.bindHost != nil else { return .disabled }
        guard Self.resolveModelPath(configuration.transcriptionModel) != nil else {
            return .error("Transcription model not found")
        }
        guard Self.findServerBinary() != nil else {
            return .error("swift-mlx-asr-server not found")
        }
        return .starting
    }

    private func enqueueLifecycleCommand(
        _ operation: @escaping @Sendable (NativeASRServerLifecycle) async -> ASRServerState
    ) {
        latestLifecycleCommandID &+= 1
        let commandID = latestLifecycleCommandID
        let previousTask = lifecycleCommandTask

        lifecycleCommandTask = Task { @MainActor [weak self, previousTask] in
            _ = await previousTask?.value
            guard let self else { return }

            let newState = await operation(lifecycle)
            guard commandID == latestLifecycleCommandID else { return }
            applyState(newState)
        }
    }

    // MARK: - Model + Binary Resolution

    /// Resolve a model spec (Hugging Face repo id or local directory) to a local directory path.
    nonisolated static func resolveModelPath(_ spec: String) -> String? {
        ModelLocator.resolve(spec)?.path
    }

    /// Find the swift-mlx-asr-server binary in expected locations.
    nonisolated static func findServerBinary() -> String? {
        let candidates = [
            // App bundle
            Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/swift-mlx-asr-server").path,
            // Development: build directory (release preferred)
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent(".build/arm64-apple-macosx/release/swift-mlx-asr-server").path,
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent(".build/arm64-apple-macosx/debug/swift-mlx-asr-server").path,
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
    }

    nonisolated fileprivate static func cleanupOrphanedManagedServerIfNeeded(port: UInt16, serverBinaryPath: String) {
        guard let listenerPID = Self.listeningPID(on: port) else { return }
        guard let parentPID = Self.parentPID(for: listenerPID), parentPID == 1 else { return }
        guard let command = Self.command(for: listenerPID), command.contains(serverBinaryPath) else { return }

        yuwpLog("Found orphaned swift-mlx-asr-server on port \(port) (PID: \(listenerPID)) — terminating")
        kill(listenerPID, SIGTERM)
        Self.waitForListener(on: port, toExit: listenerPID, timeout: 1.5)

        if Self.listeningPID(on: port) == listenerPID {
            yuwpLog("Orphaned swift-mlx-asr-server \(listenerPID) ignored SIGTERM — sending SIGKILL")
            kill(listenerPID, SIGKILL)
            Self.waitForListener(on: port, toExit: listenerPID, timeout: 1.0)
        }
    }

    nonisolated fileprivate static func checkReady(host: String, port: UInt16) -> Bool {
        guard let url = URL(string: "http://\(host):\(port)/v1/info") else { return false }
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

    nonisolated fileprivate static func listeningPID(on port: UInt16) -> Int32? {
        guard let output = toolOutput(
            launchPath: "/usr/sbin/lsof",
            arguments: ["-tiTCP:\(port)", "-sTCP:LISTEN"]
        ) else { return nil }
        guard let line = output.split(whereSeparator: \.isNewline).first,
              let pid = Int32(line.trimmingCharacters(in: .whitespacesAndNewlines)) else { return nil }
        return pid
    }

    nonisolated private static func parentPID(for pid: Int32) -> Int32? {
        guard let output = toolOutput(launchPath: "/bin/ps", arguments: ["-o", "ppid=", "-p", "\(pid)"])
        else { return nil }
        return Int32(output.trimmingCharacters(in: .whitespacesAndNewlines))
    }

    nonisolated fileprivate static func command(for pid: Int32) -> String? {
        toolOutput(launchPath: "/bin/ps", arguments: ["-o", "command=", "-p", "\(pid)"])?
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    nonisolated private static func toolOutput(launchPath: String, arguments: [String]) -> String? {
        let process = Process()
        let stdout = Pipe()
        process.executableURL = URL(fileURLWithPath: launchPath)
        process.arguments = arguments
        process.standardOutput = stdout
        process.standardError = Pipe()
        do {
            try process.run()
        } catch {
            return nil
        }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        let data = stdout.fileHandleForReading.readDataToEndOfFile()
        return String(data: data, encoding: .utf8)
    }

    nonisolated private static func waitForListener(on port: UInt16, toExit pid: Int32, timeout: TimeInterval) {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if listeningPID(on: port) != pid { return }
            Thread.sleep(forTimeInterval: 0.1)
        }
    }
}

// MARK: - Native ASR HTTP Session

/// HTTP-based STT session communicating with swift-mlx-asr-server.
/// Audio feeds are serialized on a background queue to avoid blocking the audio thread.
final class NativeASRSession: SttSession, @unchecked Sendable {
    var onUpdate: ((TranscriptUpdate) -> Void)?
    var onError: ((String) -> Void)?
    var debugSessionID: String? { sessionIDBox.get() }

    private let baseURL: String
    private var sessionId: String?
    private let sessionIDBox = LockedBox<String?>(nil)
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
            self.sessionIDBox.set(sid)
        }
    }

    func feedAudio(_ pcmData: Data) {
        queue.async { [weak self] in
            guard let self, let sid = self.sessionId else { return }
            guard let data = self.syncHTTP("POST", path: "\(self.baseURL)/\(sid)", body: pcmData),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let update = Self.parseTranscriptUpdate(json, fallbackKind: .partial),
                  !update.text.isEmpty else { return }
            self.onUpdate?(update)
        }
    }

    func end() {
        queue.async { [weak self] in
            guard let self else { return }
            guard let sid = self.sessionId else {
                self.onUpdate?(TranscriptUpdate(kind: .final, text: ""))
                return
            }
            self.sessionId = nil
            guard let data = self.syncHTTP("DELETE", path: "\(self.baseURL)/\(sid)"),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let update = Self.parseTranscriptUpdate(json, fallbackKind: .final) else {
                self.onUpdate?(TranscriptUpdate(kind: .final, text: ""))
                return
            }
            self.onUpdate?(update)
        }
    }

    static func parseTranscriptUpdate(
        _ json: [String: Any],
        fallbackKind: TranscriptUpdateKind
    ) -> TranscriptUpdate? {
        let kind: TranscriptUpdateKind
        if let rawKind = json["update_kind"] as? String,
           let parsedKind = TranscriptUpdateKind(rawValue: rawKind) {
            kind = parsedKind
        } else if fallbackKind == .partial, json["batch_corrected"] as? Bool == true {
            kind = .segmentCommit
        } else {
            kind = fallbackKind
        }

        guard let text = json["text"] as? String else { return nil }
        return TranscriptUpdate(
            kind: kind,
            text: text,
            committedText: json["committed_text"] as? String,
            activeText: json["active_text"] as? String
        )
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
