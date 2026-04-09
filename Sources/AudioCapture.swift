@preconcurrency import AVFoundation
import Foundation

/// Warnings emitted by audio capture for the session to handle.
enum AudioCaptureWarning: Sendable, Equatable {
    /// Audio hardware route changed (mic plugged/unplugged).
    /// The engine may still work, or it may need a restart.
    case routeChanged
    /// Consecutive silent buffers detected — mic may be dead.
    /// Fires once after `seconds` of silence, not repeatedly.
    case silentInput(seconds: Double)
    /// Speech was detected, then silence for `seconds`. Used for auto-stop mode.
    case speechPause(seconds: Double)
}

/// Captures microphone audio at 16kHz mono PCM and delivers raw buffers.
/// Also computes real-time RMS audio level for waveform visualization.
final class AudioCapture: @unchecked Sendable, AudioCapturing {  // AudioCapturing conformance
    private var engine = AVAudioEngine()
    private var isRunning = false
    private let targetSampleRate: Double = 16000
    private var converter: AVAudioConverter?

    /// Accumulated raw PCM for saving
    private var recordingBuffer = Data()
    private let bufferLock = NSLock()

    // Silent buffer detection (dead mic)
    private var consecutiveSilentBuffers = 0
    private var silenceWarningFired = false
    /// RMS below this for a 100ms buffer means functionally silent (codec conflict, dead mic)
    private static let deadMicRmsThreshold: Float = 0.0005
    /// Fire warning after this many seconds of dead silence
    private static let silenceWarningSeconds: Double = 2.0
    /// Buffers per second at ~100ms tap interval
    private static let buffersPerSecond: Int = 10

    // Speech pause detection (auto-stop mode)
    private var hasDetectedSpeech = false
    private var consecutivePauseBuffers = 0
    private var speechPauseFired = false
    /// RMS below this after speech = "user stopped talking"
    private static let speechPauseRmsThreshold: Float = 0.005
    /// RMS above this = speech detected
    private static let speechDetectRmsThreshold: Float = 0.01
    /// Seconds of post-speech silence before firing .speechPause. nil = disabled.
    var speechPauseTimeout: Double?

    // Route change observation
    private var routeChangeObserver: NSObjectProtocol?
    private var startTime: CFAbsoluteTime = 0
    /// Ignore route changes during the first second — engine startup can trigger spurious notifications.
    private static let routeChangeGracePeriod: Double = 1.0

    /// Audio level callback — fires with normalized RMS (0.0–1.0) per tap (~100ms).
    /// Called on the audio thread; dispatch to main if needed.
    var onAudioLevel: (@Sendable (Float) -> Void)?

    /// Warning callback — fires for route changes and sustained silence.
    /// Called on the audio thread (route change) or audio render thread (silence).
    var onWarning: (@Sendable (AudioCaptureWarning) -> Void)?

    /// Start capturing. `onBuffer` is called with raw Int16 PCM data at 16kHz mono.
    /// Returns true if capture started successfully.
    @discardableResult
    func start(onBuffer: @escaping @Sendable (Data) -> Void) -> Bool {
        guard !isRunning else {
            yuwpLog("Audio capture already running — skipping start")
            return false
        }

        // Reset silence tracking
        consecutiveSilentBuffers = 0
        silenceWarningFired = false
        hasDetectedSpeech = false
        consecutivePauseBuffers = 0
        speechPauseFired = false

        // Check for a usable audio input device before touching the engine.
        // Accessing engine.inputNode with no input device can crash.
        let inputDevices = AVCaptureDevice.DiscoverySession(
            deviceTypes: [.microphone, .external],
            mediaType: .audio,
            position: .unspecified
        ).devices
        if inputDevices.isEmpty {
            yuwpLog("No audio input device found — cannot start capture")
            return false
        }

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

        // Sanity-check the input format (invalid when device disappeared mid-access)
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
            yuwpLog("Invalid input format (\(inputFormat.sampleRate)Hz, \(inputFormat.channelCount)ch) — no usable mic?")
            return false
        }

        // Target format: 16kHz, mono, Int16
        guard let targetFormat = AVAudioFormat(
            commonFormat: .pcmFormatInt16,
            sampleRate: targetSampleRate,
            channels: 1,
            interleaved: true
        ) else {
            yuwpLog("Failed to create target audio format")
            return false
        }

        // Create converter from mic format to target format
        guard let conv = AVAudioConverter(from: inputFormat, to: targetFormat) else {
            yuwpLog("Failed to create audio converter: \(inputFormat) -> \(targetFormat)")
            return false
        }
        converter = conv

        // Observe audio route changes (mic plugged/unplugged, Bluetooth disconnect)
        startRouteChangeObserver()

        // Buffer size: ~100ms at input sample rate
        let bufferSize = AVAudioFrameCount(inputFormat.sampleRate * 0.1)

        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: inputFormat) {
            [weak self] buffer, _ in
            guard let self, let converter = self.converter else { return }

            // Convert to 16kHz mono Int16
            let frameCapacity = AVAudioFrameCount(
                Double(buffer.frameLength) * self.targetSampleRate / inputFormat.sampleRate
            )
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: frameCapacity + 16
            ) else { return }

            var error: NSError?
            var allConsumed = false
            converter.convert(to: outputBuffer, error: &error) { _, outStatus in
                if allConsumed {
                    outStatus.pointee = .noDataNow
                    return nil
                }
                allConsumed = true
                outStatus.pointee = .haveData
                return buffer
            }

            if let error {
                yuwpLog("Audio conversion error: \(error)")
                return
            }

            guard let channelData = outputBuffer.int16ChannelData else { return }
            let frameCount = Int(outputBuffer.frameLength)
            let data = Data(bytes: channelData[0], count: frameCount * 2)

            // Compute RMS from the Int16 samples
            var rms: Float = 0
            if frameCount > 0 {
                let ptr = channelData[0]
                var sumSq: Float = 0
                for i in 0..<frameCount {
                    let sample = Float(ptr[i]) / 32768.0
                    sumSq += sample * sample
                }
                rms = sqrt(sumSq / Float(frameCount))

                // Fire level callback (normalized: speech RMS ~0.02-0.15 → 0.0-1.0)
                self.onAudioLevel?(min(rms / 0.12, 1.0))
            }

            // Silent buffer detection — Bluetooth codec conflicts, dead mics
            self.trackSilence(rms: rms)
            self.trackSpeechPause(rms: rms)

            // Accumulate for recording (bulk append, not per-sample)
            self.bufferLock.lock()
            self.recordingBuffer.append(data)
            self.bufferLock.unlock()

            onBuffer(data)
        }

        do {
            try engine.start()
            isRunning = true
            startTime = CFAbsoluteTimeGetCurrent()
            yuwpLog("Audio capture started (\(Int(inputFormat.sampleRate))Hz -> 16kHz mono)")
            return true
        } catch {
            yuwpLog("Failed to start audio engine: \(error)")
            // Clean up the tap we just installed
            inputNode.removeTap(onBus: 0)
            converter = nil
            stopRouteChangeObserver()
            return false
        }
    }

    func stop() -> Data? {
        guard isRunning else { return nil }
        engine.inputNode.removeTap(onBus: 0)
        engine.stop()
        engine.reset()
        converter = nil
        isRunning = false
        stopRouteChangeObserver()

        bufferLock.lock()
        let pcmData = recordingBuffer
        recordingBuffer = Data()
        bufferLock.unlock()

        let sampleCount = pcmData.count / 2
        yuwpLog("Audio capture stopped (\(sampleCount) samples, \(String(format: "%.1f", Double(sampleCount) / targetSampleRate))s)")

        return pcmData.isEmpty ? nil : pcmData
    }

    // MARK: - Silent Buffer Detection

    private func trackSilence(rms: Float) {
        if rms < Self.deadMicRmsThreshold {
            consecutiveSilentBuffers += 1
            let threshold = Int(Self.silenceWarningSeconds) * Self.buffersPerSecond
            if consecutiveSilentBuffers >= threshold, !silenceWarningFired {
                silenceWarningFired = true
                yuwpLog("Warning: \(Self.silenceWarningSeconds)s of dead silence — mic may not be working")
                onWarning?(.silentInput(seconds: Self.silenceWarningSeconds))
            }
        } else {
            consecutiveSilentBuffers = 0
            // Don't reset silenceWarningFired — only warn once per session
        }
    }

    // MARK: - Speech Pause Detection

    private func trackSpeechPause(rms: Float) {
        guard let timeout = speechPauseTimeout, !speechPauseFired else { return }

        if rms >= Self.speechDetectRmsThreshold {
            hasDetectedSpeech = true
            consecutivePauseBuffers = 0
            return
        }

        guard hasDetectedSpeech, rms < Self.speechPauseRmsThreshold else { return }

        consecutivePauseBuffers += 1
        let threshold = Int(timeout) * Self.buffersPerSecond
        if consecutivePauseBuffers >= threshold {
            speechPauseFired = true
            yuwpLog("Speech pause detected (\(timeout)s silence after speech)")
            onWarning?(.speechPause(seconds: timeout))
        }
    }

    // MARK: - Audio Route Change

    private func startRouteChangeObserver() {
        routeChangeObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: nil
        ) { [weak self] _ in
            guard let self, self.isRunning else { return }
            let elapsed = CFAbsoluteTimeGetCurrent() - self.startTime
            if elapsed < Self.routeChangeGracePeriod {
                yuwpLog("Audio route change ignored (startup, \(String(format: "%.1f", elapsed))s)")
                return
            }
            yuwpLog("Audio route changed mid-session")
            self.onWarning?(.routeChanged)
        }
    }

    private func stopRouteChangeObserver() {
        if let observer = routeChangeObserver {
            NotificationCenter.default.removeObserver(observer)
            routeChangeObserver = nil
        }
    }
}
