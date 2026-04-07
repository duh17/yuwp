import Foundation

// MARK: - STT Provider Protocols

/// Manages the lifecycle of an STT backend (model loading, connections, etc.).
/// One provider lives for the app's lifetime. Creates sessions for each dictation.
///
/// Implementations:
///   - `ASRSidecar` — local Python sidecar (Qwen3-ASR via mlx-audio)
///   - Future: Apple Speech, remote Whisper API, external clients, etc.
/// Not actor-isolated — implementations handle their own thread safety.
/// ASRSidecar is `@unchecked Sendable`; future providers may be actors.
protocol SttProvider: AnyObject, Sendable {
    var isReady: Bool { get }
    var onReady: (@Sendable () -> Void)? { get set }
    var onError: (@Sendable (String) -> Void)? { get set }
    func start()
    func shutdown()
    @MainActor func makeSession() -> any SttSession
}

/// A single dictation recording cycle. Created by `SttProvider.makeSession()`.
///
/// Feeds raw PCM audio and emits partial/final transcripts via callbacks.
/// Each session is used once (begin → feed → end), then discarded.
/// `feedAudio` is not actor-isolated — called from the real-time audio thread.
protocol SttSession: AnyObject, Sendable {
    var onPartial: ((String) -> Void)? { get set }
    var onFinal: ((String) -> Void)? { get set }
    var onError: ((String) -> Void)? { get set }
    @MainActor func begin(language: String?)
    func feedAudio(_ pcmData: Data)
    func end()
}

// MARK: - Audio & Text Injection Protocols

/// Audio capture abstraction. Real implementation: `AudioCapture`.
/// Not actor-isolated — AudioCapture is `@unchecked Sendable`.
protocol AudioCapturing: AnyObject, Sendable {
    var onAudioLevel: (@Sendable (Float) -> Void)? { get set }
    @discardableResult
    func start(onBuffer: @escaping @Sendable (Data) -> Void) -> Bool
    func stop() -> Data?
}

/// Text injection abstraction. Real implementation: `TextInjector`.
@MainActor
protocol TextInjecting: AnyObject {
    var targetPosition: NSPoint { get }
    var isLiveInjecting: Bool { get }
    func captureTarget()
    func inject(_ text: String)
    func commit(_ text: String)
    func release()
}

// MARK: - Session Events

/// Events emitted by DictationSession for UI consumption.
enum DictationEvent: Equatable {
    /// AX injection verified — switch to compact indicator
    case liveInjectionVerified(caretPosition: NSPoint)
    /// Partial transcript available (clipboard mode — show in panel)
    case partialTranscript(String)
    /// Caret moved during AX injection — reposition compact indicator
    case caretMoved(NSPoint)
    /// Audio level update for waveform visualization
    case audioLevel(Float)
    /// Final transcript committed — session is done
    case finished
}

/// Orchestrates a single dictation session: hotkey toggle → record → transcribe → inject.
///
/// Owns the dictation state machine but not the UI. Emits `DictationEvent`s
/// that the caller (AppDelegate) maps to MicPanel and status icon updates.
///
/// Testable: inject mock ASRSidecar, TextInjector, AudioCapture.
@MainActor
final class DictationSession {
    let textInjector: any TextInjecting
    private let audioCapture: any AudioCapturing
    private let sttSession: any SttSession
    private let typewriter = TypewriterAnimator()
    private var typewriterDriveTask: Task<Void, Never>?
    private var finalTimeoutTask: Task<Void, Never>?
    private var maxDurationTask: Task<Void, Never>?
    private static let maxDurationSeconds: UInt64 = 5 * 60 // 5 minutes

    /// Callback for events — set by AppDelegate to update UI.
    var onEvent: ((DictationEvent) -> Void)?

    /// Called when session needs external stop (e.g., max duration).
    /// Set by AppDelegate to trigger stopDictation().
    var onRequestStop: (() -> Void)?

    private(set) var isActive = false

    init(
        sttSession: any SttSession,
        textInjector: any TextInjecting,
        audioCapture: any AudioCapturing
    ) {
        self.sttSession = sttSession
        self.textInjector = textInjector
        self.audioCapture = audioCapture
    }

    /// Start a dictation session. Captures the focused element, begins audio + STT.
    func start() {
        guard !isActive else { return }
        isActive = true

        textInjector.captureTarget()

        // Wire STT callbacks
        sttSession.onPartial = { [weak self] text in
            Task { @MainActor in self?.handlePartial(text) }
        }
        sttSession.onFinal = { [weak self] text in
            Task { @MainActor in self?.handleFinal(text) }
        }
        sttSession.onError = { msg in
            Task { @MainActor in yuwpLog("STT error: \(msg)") }
        }
        sttSession.begin(language: nil)

        audioCapture.onAudioLevel = { [weak self] level in
            Task { @MainActor in
                self?.onEvent?(.audioLevel(level))
            }
        }

        let started = audioCapture.start { [weak self] buffer in
            self?.sttSession.feedAudio(buffer)
        }

        if !started {
            yuwpLog("Audio capture failed to start — aborting session")
            isActive = false
            sttSession.end()
            finalize()
            return
        }

        yuwpLog("Listening...")

        // Safety net: auto-stop after max duration
        maxDurationTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: Self.maxDurationSeconds * 1_000_000_000)
            guard let self, self.isActive else { return }
            yuwpLog("Max session duration (\(Self.maxDurationSeconds)s) reached — auto-stopping")
            self.onRequestStop?()
        }
    }

    /// Stop recording and wait for the final transcription.
    /// Returns the captured PCM data for archival (WAV saving).
    @discardableResult
    func stop() -> Data? {
        guard isActive else { return nil }
        isActive = false

        maxDurationTask?.cancel()
        maxDurationTask = nil
        let pcmData = audioCapture.stop()
        sttSession.end()
        typewriterDriveTask?.cancel()
        typewriterDriveTask = nil

        yuwpLog("Waiting for final result...")

        // Timeout: if STT doesn't send final within 10s, clean up
        finalTimeoutTask = Task { @MainActor [weak self] in
            try? await Task.sleep(nanoseconds: 10_000_000_000)
            guard let self, !self.isActive else { return }
            yuwpLog("Final result timeout — releasing injector")
            self.finalize()
        }

        return pcmData
    }

    // MARK: - STT Callbacks

    private func handlePartial(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed.lowercased() != "none" else { return }

        typewriter.update(fullText: text)
        textInjector.inject(typewriter.displayText)

        if textInjector.isLiveInjecting {
            onEvent?(.liveInjectionVerified(caretPosition: textInjector.targetPosition))
        } else {
            onEvent?(.partialTranscript(typewriter.displayText))
            driveTypewriterDisplay()
        }
    }

    private func handleFinal(_ text: String) {
        finalTimeoutTask?.cancel()
        finalTimeoutTask = nil

        typewriter.commitCurrentAnimation()
        textInjector.commit(text)

        finalize()
    }

    // MARK: - Private

    private func finalize() {
        typewriter.reset()
        textInjector.release()
        onEvent?(.finished)
    }

    private func driveTypewriterDisplay() {
        guard typewriter.isAnimating else { return }
        typewriterDriveTask?.cancel()
        typewriterDriveTask = Task { @MainActor in
            while typewriter.isAnimating {
                try? await Task.sleep(nanoseconds: 16_000_000)
                guard !Task.isCancelled else { break }
                let display = typewriter.displayText
                onEvent?(.partialTranscript(display))
                textInjector.inject(display)
            }
        }
    }
}

// MARK: - WAV Writer

enum WAVWriter {
    /// Encode 16kHz mono s16le PCM data as a WAV file.
    static func write(_ pcmData: Data, to url: URL) throws {
        let sampleRate: UInt32 = 16000
        let channels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = sampleRate * UInt32(channels) * UInt32(bitsPerSample / 8)
        let blockAlign = channels * (bitsPerSample / 8)
        let dataSize = UInt32(pcmData.count)

        var wav = Data()
        wav.append(contentsOf: "RIFF".utf8)
        wav.append(withUnsafeBytes(of: (36 + dataSize).littleEndian) { Data($0) })
        wav.append(contentsOf: "WAVE".utf8)
        wav.append(contentsOf: "fmt ".utf8)
        wav.append(withUnsafeBytes(of: UInt32(16).littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: UInt16(1).littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: channels.littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: sampleRate.littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: byteRate.littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: blockAlign.littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: bitsPerSample.littleEndian) { Data($0) })
        wav.append(contentsOf: "data".utf8)
        wav.append(withUnsafeBytes(of: dataSize.littleEndian) { Data($0) })
        wav.append(pcmData)

        try wav.write(to: url)
    }
}
