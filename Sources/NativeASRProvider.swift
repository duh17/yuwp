import ASRIPC
import Darwin
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

private final class NativeASRStdioBridge: @unchecked Sendable {
    private enum BridgeError: Error {
        case closed
        case timeout
        case invalidResponse(String)
        case server(String)
    }

    private let inputHandle: FileHandle
    private let outputHandle: FileHandle
    private let queue = DispatchQueue(label: "yuwp.asr-stdio-bridge", qos: .userInitiated)

    private var nextRequestID: UInt64 = 1
    private var closed = false

    init(inputHandle: FileHandle, outputHandle: FileHandle) {
        self.inputHandle = inputHandle
        self.outputHandle = outputHandle
    }

    deinit {
        close()
    }

    func close() {
        queue.sync {
            guard !closed else { return }
            closed = true
            inputHandle.closeFile()
            outputHandle.closeFile()
        }
    }

    func isReady(timeout: TimeInterval = 2) -> Bool {
        do {
            let response = try request(command: .info, sessionID: nil, language: nil, binary: Data(), timeout: timeout)
            return response.status == "ready"
        } catch {
            return false
        }
    }

    func createSession(language: String? = nil, timeout: TimeInterval = 10) -> String? {
        perform(command: .create, sessionID: nil, language: language, binary: Data(), timeout: timeout)?.sessionID
    }

    func feed(sessionID: String, pcmData: Data, timeout: TimeInterval = 30) -> TranscriptUpdate? {
        guard let response = perform(command: .feed, sessionID: sessionID, language: nil, binary: pcmData, timeout: timeout) else {
            return nil
        }
        return Self.makeTranscriptUpdate(from: response, fallbackKind: .partial)
    }

    func stop(sessionID: String, timeout: TimeInterval = 30) -> TranscriptUpdate? {
        guard let response = perform(command: .stop, sessionID: sessionID, language: nil, binary: Data(), timeout: timeout) else {
            return nil
        }
        return Self.makeTranscriptUpdate(from: response, fallbackKind: .final)
    }

    private func perform(
        command: ASRIPCCommand,
        sessionID: String?,
        language: String?,
        binary: Data,
        timeout: TimeInterval
    ) -> ASRIPCResponse? {
        do {
            return try request(command: command, sessionID: sessionID, language: language, binary: binary, timeout: timeout)
        } catch BridgeError.server(let message) {
            yuwpLog("ASR stdio request failed (\(command.rawValue)): \(message)")
            return nil
        } catch {
            close()
            yuwpLog("ASR stdio transport failed (\(command.rawValue)): \(error)")
            return nil
        }
    }

    private func request(
        command: ASRIPCCommand,
        sessionID: String?,
        language: String?,
        binary: Data,
        timeout: TimeInterval
    ) throws -> ASRIPCResponse {
        try queue.sync {
            guard !closed else { throw BridgeError.closed }

            let requestID = nextRequestID
            nextRequestID &+= 1

            let request = ASRIPCRequest(id: requestID, command: command, sessionID: sessionID, language: language)
            let frame = try ASRIPCCodec.encode(request, binary: binary)
            try writeAll(fd: inputHandle.fileDescriptor, data: frame)

            let deadline = Date().addingTimeInterval(timeout)
            while true {
                let header = try readExact(fd: outputHandle.fileDescriptor, count: ASRIPCFrameCodec.headerSize, deadline: deadline)
                guard let lengths = ASRIPCFrameCodec.decodeHeader(header) else {
                    throw BridgeError.invalidResponse("invalid response header")
                }
                let metadata = try readExact(fd: outputHandle.fileDescriptor, count: lengths.metadataLength, deadline: deadline)
                _ = try readExact(fd: outputHandle.fileDescriptor, count: lengths.binaryLength, deadline: deadline)

                let response = try ASRIPCCodec.decodeResponse(metadata: metadata)
                guard response.id == requestID else {
                    yuwpLog("ASR stdio out-of-order response id=\(response.id), expected=\(requestID)")
                    continue
                }
                if !response.ok {
                    throw BridgeError.server(response.error ?? "request failed")
                }
                return response
            }
        }
    }

    private func writeAll(fd: Int32, data: Data) throws {
        var offset = 0
        let success = data.withUnsafeBytes { rawBuffer -> Bool in
            guard let base = rawBuffer.baseAddress else { return true }

            while offset < data.count {
                let count = Darwin.write(fd, base.advanced(by: offset), data.count - offset)
                if count < 0 {
                    if errno == EINTR { continue }
                    return false
                }
                if count == 0 { return false }
                offset += count
            }
            return true
        }

        guard success else { throw BridgeError.closed }
    }

    private func readExact(fd: Int32, count: Int, deadline: Date) throws -> Data {
        guard count >= 0 else {
            throw BridgeError.invalidResponse("negative length")
        }
        if count == 0 { return Data() }

        var buffer = [UInt8](repeating: 0, count: count)
        var offset = 0

        while offset < count {
            let remaining = deadline.timeIntervalSinceNow
            guard remaining > 0 else { throw BridgeError.timeout }
            guard pollReadable(fd: fd, timeout: remaining) else { throw BridgeError.timeout }

            let readCount = buffer.withUnsafeMutableBytes { rawBuffer -> Int in
                guard let base = rawBuffer.baseAddress else { return -1 }
                return Darwin.read(fd, base.advanced(by: offset), count - offset)
            }

            if readCount == 0 { throw BridgeError.closed }
            if readCount < 0 {
                if errno == EINTR { continue }
                throw BridgeError.closed
            }
            offset += readCount
        }

        return Data(buffer)
    }

    private func pollReadable(fd: Int32, timeout: TimeInterval) -> Bool {
        let timeoutMS = Int32(min(Double(Int32.max), max(1, timeout * 1000.0)))
        var descriptor = pollfd(fd: fd, events: Int16(POLLIN), revents: 0)

        while true {
            let result = Darwin.poll(&descriptor, 1, timeoutMS)
            if result > 0 { return true }
            if result == 0 { return false }
            if errno == EINTR { continue }
            return false
        }
    }

    private static func makeTranscriptUpdate(
        from response: ASRIPCResponse,
        fallbackKind: TranscriptUpdateKind
    ) -> TranscriptUpdate? {
        guard let text = response.text else { return nil }
        let kind = response.updateKind.flatMap(TranscriptUpdateKind.init(rawValue:)) ?? fallbackKind
        return TranscriptUpdate(
            kind: kind,
            text: text,
            committedText: response.committedText,
            activeText: response.activeText
        )
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
    var saveRecordings: Bool = false
    var recordingsDir: URL = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
        .appendingPathComponent("Yuwp/recordings", isDirectory: true)
    var port: UInt16
    var serverMode: ServerMode = .localhost
    var asrTransport: ASRIPCTransport = .stdio

    var bindHost: String? { serverMode.bindHost }
    var clientHost: String { serverMode.clientHost }

    var launchTransport: ASRIPCTransport {
        ServerRuntimePolicy.effectiveTransport(
            serverMode: serverMode,
            requestedTransport: asrTransport
        )
    }
}

private actor NativeASRServerLifecycle {
    private let stateSink: @Sendable (ASRServerState) -> Void
    private let stdioBridgeBox: LockedBox<NativeASRStdioBridge?>

    private var process: Process?
    private var readyPollTask: Task<Void, Never>?
    private var restartTask: Task<Void, Never>?
    private var isIntentionalShutdown = false
    private var restartAttempts = 0
    private var launchGeneration: UInt64 = 0
    private var activeConfiguration: ASRServerConfiguration?

    private static let maxRestartAttempts = 5

    init(
        stateSink: @escaping @Sendable (ASRServerState) -> Void,
        stdioBridgeBox: LockedBox<NativeASRStdioBridge?>
    ) {
        self.stateSink = stateSink
        self.stdioBridgeBox = stdioBridgeBox
    }

    func start(configuration: ASRServerConfiguration) -> ASRServerState {
        launchGeneration &+= 1
        let generation = launchGeneration

        activeConfiguration = configuration
        isIntentionalShutdown = false
        cancelBackgroundTasks()

        stdioBridgeBox.set(nil)

        guard let bindHost = configuration.bindHost else {
            process = nil
            yuwpLog("yuwp-asr serve disabled")
            return .disabled
        }

        let transport = configuration.launchTransport
        if configuration.asrTransport == .stdio, configuration.serverMode != .localhost {
            yuwpLog("ASR stdio transport requested but unavailable for \(configuration.serverMode.description); falling back to HTTP")
        }

        guard let transcriptionModelPath = NativeASRProvider.resolveModelPath(configuration.transcriptionModel) else {
            yuwpLog("Transcription model not found: \(configuration.transcriptionModel)")
            process = nil
            return .error("Transcription model not found")
        }

        guard let serverBin = NativeASRProvider.findServerBinary() else {
            yuwpLog("yuwp-asr binary not found — run: swift build -c release --product yuwp-asr")
            process = nil
            return .error("yuwp-asr not found")
        }

        if transport == .http {
            NativeASRProvider.cleanupOrphanedManagedServerIfNeeded(
                port: configuration.port,
                serverBinaryPath: serverBin
            )
        }

        let proc = Process()
        let stderrPipe = Pipe()
        let stdinPipe = transport == .stdio ? Pipe() : nil
        let stdoutPipe = transport == .stdio ? Pipe() : nil

        proc.executableURL = URL(fileURLWithPath: serverBin)
        var arguments = [
            "serve",
            "--model", transcriptionModelPath,
            "--transport", transport.rawValue,
            "--port", "\(configuration.port)",
            "--host", bindHost,
            "--parent-pid", "\(ProcessInfo.processInfo.processIdentifier)",
        ]
        if configuration.batchCommitEnabled {
            arguments += ["--batch-model", transcriptionModelPath]
        } else {
            arguments += ["--disable-batch-retranscribe"]
        }
        proc.arguments = arguments
        var childEnvironment = ProcessInfo.processInfo.environment
        childEnvironment["YUWP_DIAGNOSTIC_LOGGING"] = configuration.diagnosticLoggingEnabled ? "1" : "0"
        childEnvironment["YUWP_ASR_SAVE_RECORDINGS"] = configuration.saveRecordings ? "1" : "0"
        childEnvironment["YUWP_ASR_RECORDINGS_DIR"] = configuration.recordingsDir.path
        proc.environment = childEnvironment
        proc.standardInput = stdinPipe ?? FileHandle.nullDevice
        proc.standardOutput = stdoutPipe ?? FileHandle.nullDevice
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
                    port: configuration.port,
                    transport: transport
                )
            }
        }

        do {
            try proc.run()
        } catch {
            process = nil
            stdioBridgeBox.set(nil)
            yuwpLog("Failed to start yuwp-asr serve: \(error)")
            scheduleRestart(generation: generation)
            return .error("Failed to start server")
        }

        process = proc

        if transport == .stdio, let stdinPipe, let stdoutPipe {
            let bridge = NativeASRStdioBridge(
                inputHandle: stdinPipe.fileHandleForWriting,
                outputHandle: stdoutPipe.fileHandleForReading
            )
            stdioBridgeBox.set(bridge)
        } else {
            stdioBridgeBox.set(nil)
        }

        yuwpLog("yuwp-asr serve started (PID: \(proc.processIdentifier), transport: \(transport.rawValue))")

        scheduleReadyPoll(generation: generation, configuration: configuration, transport: transport)
        return .starting
    }

    func shutdown(targetState: ASRServerState) -> ASRServerState {
        launchGeneration &+= 1
        isIntentionalShutdown = true
        activeConfiguration = nil
        restartAttempts = 0
        cancelBackgroundTasks()

        if let bridge = stdioBridgeBox.get() {
            bridge.close()
            stdioBridgeBox.set(nil)
        }

        if let proc = process, proc.isRunning {
            kill(proc.processIdentifier, SIGTERM)
            proc.waitUntilExit()
        }

        process = nil
        yuwpLog(targetState == .disabled ? "yuwp-asr serve disabled" : "yuwp-asr serve stopped")
        return targetState
    }

    private func scheduleReadyPoll(
        generation: UInt64,
        configuration: ASRServerConfiguration,
        transport: ASRIPCTransport
    ) {
        readyPollTask?.cancel()
        readyPollTask = Task.detached { [weak self] in
            guard let self else { return }
            for _ in 0..<60 {
                try? await Task.sleep(for: .milliseconds(500))
                guard !Task.isCancelled else { return }

                let ready = await self.checkReady(configuration: configuration, transport: transport)
                if ready {
                    await self.handleReady(generation: generation)
                    return
                }
            }
            guard !Task.isCancelled else { return }
            await self.handleStartupTimeout(generation: generation)
        }
    }

    private func checkReady(configuration: ASRServerConfiguration, transport: ASRIPCTransport) -> Bool {
        switch transport {
        case .http:
            return NativeASRProvider.checkReady(host: configuration.clientHost, port: configuration.port)
        case .stdio:
            guard let bridge = stdioBridgeBox.get() else { return false }
            return bridge.isReady(timeout: 1.5)
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
        yuwpLog("yuwp-asr serve failed to become ready within 30s")
        stateSink(.error("Server startup timeout"))
    }

    private func handleUnexpectedTermination(
        processIdentifier: Int32,
        terminationStatus: Int32,
        generation: UInt64,
        port: UInt16,
        transport: ASRIPCTransport
    ) {
        guard generation == launchGeneration, !isIntentionalShutdown else { return }

        readyPollTask?.cancel()
        readyPollTask = nil
        if process?.processIdentifier == processIdentifier {
            process = nil
        }
        if let bridge = stdioBridgeBox.get() {
            bridge.close()
            stdioBridgeBox.set(nil)
        }

        let code = terminationStatus
        if transport == .http,
           let listenerPID = NativeASRProvider.listeningPID(on: port), listenerPID != processIdentifier {
            let owner = NativeASRProvider.command(for: listenerPID) ?? "pid \(listenerPID)"
            yuwpLog("yuwp-asr serve failed to own port \(port); listener PID \(listenerPID): \(owner)")
            stateSink(.error("Port \(port) already in use"))
            return
        }

        yuwpLog("yuwp-asr serve exited unexpectedly (code \(code))")
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
        yuwpLog("Restarting yuwp-asr serve in \(Int(delay))s (\(restartAttempts)/\(Self.maxRestartAttempts))")

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

/// Manages the native ASR server process (`yuwp-asr serve`).
///
/// Launches `yuwp-asr serve` as a child process, monitors its health,
/// and provides STT sessions over either localhost HTTP or stdio IPC.
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

    var saveRecordings: Bool {
        get { configuration.saveRecordings }
        set { configuration.saveRecordings = newValue }
    }

    var recordingsDir: URL {
        get { configuration.recordingsDir }
        set { configuration.recordingsDir = newValue.standardizedFileURL }
    }

    var serverMode: ServerMode {
        get { configuration.serverMode }
        set { configuration.serverMode = newValue }
    }

    var asrTransport: ASRIPCTransport {
        get { configuration.asrTransport }
        set { configuration.asrTransport = newValue }
    }

    private let stdioBridgeBox = LockedBox<NativeASRStdioBridge?>(nil)

    private lazy var lifecycle = NativeASRServerLifecycle(
        stateSink: { [weak self] newState in
            Task { @MainActor [weak self] in
                self?.applyState(newState)
            }
        },
        stdioBridgeBox: stdioBridgeBox
    )

    private var lifecycleCommandTask: Task<Void, Never>?
    private var latestLifecycleCommandID: UInt64 = 0

    init(port: UInt16 = ASRIPCDefaults.defaultHTTPPort) {
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
        switch configuration.launchTransport {
        case .http:
            return NativeASRSession(host: configuration.clientHost, port: configuration.port)
        case .stdio:
            if let bridge = stdioBridgeBox.get() {
                return NativeASRStdioSession(bridge: bridge)
            }
            let message = "ASR stdio bridge unavailable — restart Yuwp or switch App Transport to HTTP"
            yuwpLog(message)
            return NativeASRUnavailableSession(reason: message)
        }
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
            return .error("yuwp-asr not found")
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

    /// Find the yuwp-asr binary in expected locations.
    nonisolated static func findServerBinary() -> String? {
        let candidates = [
            // App bundle
            Bundle.main.bundleURL.appendingPathComponent("Contents/MacOS/yuwp-asr").path,
            // Development: build directory (release preferred)
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent(".build/arm64-apple-macosx/release/yuwp-asr").path,
            URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent(".build/arm64-apple-macosx/debug/yuwp-asr").path,
        ]
        return candidates.first { FileManager.default.fileExists(atPath: $0) }
    }

    nonisolated fileprivate static func cleanupOrphanedManagedServerIfNeeded(port: UInt16, serverBinaryPath: String) {
        guard let listenerPID = Self.listeningPID(on: port) else { return }
        guard let parentPID = Self.parentPID(for: listenerPID), parentPID == 1 else { return }
        guard let command = Self.command(for: listenerPID), command.contains(serverBinaryPath) else { return }

        yuwpLog("Found orphaned yuwp-asr serve on port \(port) (PID: \(listenerPID)) — terminating")
        kill(listenerPID, SIGTERM)
        Self.waitForListener(on: port, toExit: listenerPID, timeout: 1.5)

        if Self.listeningPID(on: port) == listenerPID {
            yuwpLog("Orphaned yuwp-asr serve \(listenerPID) ignored SIGTERM — sending SIGKILL")
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

// MARK: - Native ASR Unavailable Session

/// Fails immediately when a transport-specific session cannot be created.
fileprivate final class NativeASRUnavailableSession: SttSession, @unchecked Sendable {
    var onUpdate: ((TranscriptUpdate) -> Void)?
    var onError: ((String) -> Void)?
    var debugSessionID: String? { nil }

    private let reason: String

    init(reason: String) {
        self.reason = reason
    }

    @MainActor
    func begin(language: String?) {
        onError?(reason)
    }

    func feedAudio(_ pcmData: Data) {
        // Intentionally no-op: no backing transport is available.
    }

    func end() {
        onUpdate?(TranscriptUpdate(kind: .final, text: ""))
    }
}

// MARK: - Native ASR Stdio Session

/// Stdio-based STT session communicating with yuwp-asr over framed stdin/stdout.
/// Audio feeds are serialized on a background queue to avoid blocking the audio thread.
fileprivate final class NativeASRStdioSession: SttSession, @unchecked Sendable {
    var onUpdate: ((TranscriptUpdate) -> Void)?
    var onError: ((String) -> Void)?
    var debugSessionID: String? { sessionIDBox.get() }

    private let bridge: NativeASRStdioBridge
    private var sessionId: String?
    private var pendingChunks: [Data] = []
    private let maxPendingChunks = 8
    private let sessionIDBox = LockedBox<String?>(nil)
    private let queue = DispatchQueue(label: "yuwp.asr-stdio-session", qos: .userInitiated)

    init(bridge: NativeASRStdioBridge) {
        self.bridge = bridge
    }

    func begin(language: String?) {
        queue.async { [weak self] in
            guard let self else { return }
            let trimmedLanguage = language?.trimmingCharacters(in: .whitespacesAndNewlines)
            let normalizedLanguage = (trimmedLanguage?.isEmpty == false) ? trimmedLanguage : nil
            guard let sid = self.bridge.createSession(language: normalizedLanguage) else {
                self.onError?("Failed to create ASR session")
                return
            }
            self.sessionId = sid
            self.sessionIDBox.set(sid)

            if !self.pendingChunks.isEmpty {
                let buffered = self.pendingChunks
                self.pendingChunks.removeAll(keepingCapacity: true)
                for chunk in buffered {
                    _ = self.sendChunk(sid: sid, pcmData: chunk)
                }
            }
        }
    }

    func feedAudio(_ pcmData: Data) {
        queue.async { [weak self] in
            guard let self else { return }
            guard let sid = self.sessionId else {
                if self.pendingChunks.count >= self.maxPendingChunks {
                    self.pendingChunks.removeFirst(self.pendingChunks.count - self.maxPendingChunks + 1)
                }
                self.pendingChunks.append(pcmData)
                return
            }
            _ = self.sendChunk(sid: sid, pcmData: pcmData)
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
            self.pendingChunks.removeAll(keepingCapacity: false)
            if let update = self.bridge.stop(sessionID: sid) {
                self.onUpdate?(update)
            } else {
                self.onUpdate?(TranscriptUpdate(kind: .final, text: ""))
            }
        }
    }

    private func sendChunk(sid: String, pcmData: Data) -> Bool {
        guard let update = bridge.feed(sessionID: sid, pcmData: pcmData), !update.text.isEmpty else {
            return false
        }
        self.onUpdate?(update)
        return true
    }
}

// MARK: - Native ASR HTTP Session

/// HTTP-based STT session communicating with yuwp-asr.
/// Audio feeds are serialized on a background queue to avoid blocking the audio thread.
final class NativeASRSession: SttSession, @unchecked Sendable {
    var onUpdate: ((TranscriptUpdate) -> Void)?
    var onError: ((String) -> Void)?
    var debugSessionID: String? { sessionIDBox.get() }

    private let baseURL: String
    private var sessionId: String?
    private var pendingChunks: [Data] = []
    private let maxPendingChunks = 8
    private let sessionIDBox = LockedBox<String?>(nil)
    private let queue = DispatchQueue(label: "yuwp.asr-session", qos: .userInitiated)

    init(host: String, port: UInt16) {
        self.baseURL = "http://\(host):\(port)/v1/audio/transcriptions/stream"
    }

    func begin(language: String?) {
        queue.async { [weak self] in
            guard let self else { return }
            let createPath = self.createSessionPath(language: language)
            guard let data = self.syncHTTP("POST", path: createPath),
                  let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let sid = json["session_id"] as? String else {
                self.onError?("Failed to create ASR session")
                return
            }
            self.sessionId = sid
            self.sessionIDBox.set(sid)

            // Flush audio captured while session creation was in flight.
            if !self.pendingChunks.isEmpty {
                let buffered = self.pendingChunks
                self.pendingChunks.removeAll(keepingCapacity: true)
                for chunk in buffered {
                    _ = self.sendChunk(sid: sid, pcmData: chunk)
                }
            }
        }
    }

    func feedAudio(_ pcmData: Data) {
        queue.async { [weak self] in
            guard let self else { return }
            guard let sid = self.sessionId else {
                if self.pendingChunks.count >= self.maxPendingChunks {
                    self.pendingChunks.removeFirst(self.pendingChunks.count - self.maxPendingChunks + 1)
                }
                self.pendingChunks.append(pcmData)
                return
            }
            _ = self.sendChunk(sid: sid, pcmData: pcmData)
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
            self.pendingChunks.removeAll(keepingCapacity: false)
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

    private func sendChunk(sid: String, pcmData: Data) -> Bool {
        guard let data = self.syncHTTP("POST", path: "\(self.baseURL)/\(sid)", body: pcmData),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let update = Self.parseTranscriptUpdate(json, fallbackKind: .partial),
              !update.text.isEmpty else { return false }
        self.onUpdate?(update)
        return true
    }

    private func createSessionPath(language: String?) -> String {
        let trimmed = language?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let language = trimmed, !language.isEmpty,
              var components = URLComponents(string: baseURL)
        else {
            return baseURL
        }

        components.queryItems = [URLQueryItem(name: "language", value: language)]
        return components.string ?? baseURL
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
