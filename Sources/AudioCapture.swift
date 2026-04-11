@preconcurrency import AVFoundation
import AudioToolbox
import CoreAudio
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
    struct WarningTracker {
        /// RMS below this for a 100ms buffer means functionally silent (codec conflict, dead mic)
        static let deadMicRmsThreshold: Float = 0.0005
        /// Fire warning after this many seconds of dead silence
        static let silenceWarningSeconds: Double = 2.0
        /// Buffers per second at ~100ms tap interval
        static let buffersPerSecond: Int = 10
        /// RMS below this after speech = "user stopped talking"
        static let speechPauseRmsThreshold: Float = 0.005
        /// RMS above this = speech detected
        static let speechDetectRmsThreshold: Float = 0.01
        /// Ignore route changes during the first second — engine startup can trigger spurious notifications.
        static let routeChangeGracePeriod: Double = 1.0

        var speechPauseTimeout: Double?

        private(set) var consecutiveSilentBuffers = 0
        private(set) var silenceWarningFired = false
        private(set) var hasDetectedSpeech = false
        private(set) var consecutivePauseBuffers = 0
        private(set) var speechPauseFired = false

        mutating func reset() {
            consecutiveSilentBuffers = 0
            silenceWarningFired = false
            hasDetectedSpeech = false
            consecutivePauseBuffers = 0
            speechPauseFired = false
        }

        mutating func ingest(rms: Float) -> [AudioCaptureWarning] {
            var warnings: [AudioCaptureWarning] = []
            if let warning = trackSilence(rms: rms) {
                warnings.append(warning)
            }
            if let warning = trackSpeechPause(rms: rms) {
                warnings.append(warning)
            }
            return warnings
        }

        static func warningForRouteChange(elapsedSinceStart: Double) -> AudioCaptureWarning? {
            elapsedSinceStart >= routeChangeGracePeriod ? .routeChanged : nil
        }

        static func bufferThreshold(for seconds: Double) -> Int {
            max(1, Int(ceil(seconds * Double(buffersPerSecond))))
        }

        private mutating func trackSilence(rms: Float) -> AudioCaptureWarning? {
            guard !hasDetectedSpeech else {
                consecutiveSilentBuffers = 0
                return nil
            }

            if rms < Self.deadMicRmsThreshold {
                consecutiveSilentBuffers += 1
                let threshold = Self.bufferThreshold(for: Self.silenceWarningSeconds)
                if consecutiveSilentBuffers >= threshold, !silenceWarningFired {
                    silenceWarningFired = true
                    return .silentInput(seconds: Self.silenceWarningSeconds)
                }
            } else {
                consecutiveSilentBuffers = 0
                // Don't reset silenceWarningFired — only warn once per session.
            }
            return nil
        }

        private mutating func trackSpeechPause(rms: Float) -> AudioCaptureWarning? {
            if rms >= Self.speechDetectRmsThreshold {
                hasDetectedSpeech = true
                consecutiveSilentBuffers = 0
                consecutivePauseBuffers = 0
                return nil
            }

            guard let timeout = speechPauseTimeout, !speechPauseFired, hasDetectedSpeech else { return nil }

            guard rms < Self.speechPauseRmsThreshold else {
                consecutivePauseBuffers = 0
                return nil
            }

            consecutivePauseBuffers += 1
            let threshold = Self.bufferThreshold(for: timeout)
            if consecutivePauseBuffers >= threshold {
                speechPauseFired = true
                return .speechPause(seconds: timeout)
            }

            return nil
        }
    }

    private var engine = AVAudioEngine()
    private var isRunning = false
    private let targetSampleRate: Double = 16000
    private var converter: AVAudioConverter?

    private let inputCatalog: any AudioInputCatalog

    /// Accumulated raw PCM for saving
    private var recordingBuffer = Data()
    private let bufferLock = NSLock()
    private var warningTracker = WarningTracker()

    var inputSelection: AudioInputSelection = .systemDefault

    /// Seconds of post-speech silence before firing .speechPause. nil = disabled.
    var speechPauseTimeout: Double? {
        get { warningTracker.speechPauseTimeout }
        set { warningTracker.speechPauseTimeout = newValue }
    }

    // Route change observation
    private var routeChangeObserver: NSObjectProtocol?
    private var startTime: CFAbsoluteTime = 0

    /// Audio level callback — fires with normalized RMS (0.0–1.0) per tap (~100ms).
    /// Called on the audio thread; dispatch to main if needed.
    var onAudioLevel: (@Sendable (Float) -> Void)?

    /// Warning callback — fires for route changes and sustained silence.
    /// Called on the audio thread (route change) or audio render thread (silence).
    var onWarning: (@Sendable (AudioCaptureWarning) -> Void)?

    init(inputCatalog: any AudioInputCatalog = SystemAudioInputCatalog()) {
        self.inputCatalog = inputCatalog
    }

    /// Start capturing. `onBuffer` is called with raw Int16 PCM data at 16kHz mono.
    /// Returns true if capture started successfully.
    @discardableResult
    func start(onBuffer: @escaping @Sendable (Data) -> Void) -> Bool {
        guard !isRunning else {
            yuwpLog("Audio capture already running — skipping start")
            return false
        }

        // Reset the engine on every session. Route/sample-rate changes can leave
        // the previous graph advertising a stale client format, which can crash
        // installTap() when the live hardware format no longer matches.
        engine.stop()
        engine.reset()
        engine = AVAudioEngine()

        // Reset warning tracking for the new session.
        warningTracker.reset()

        let availableInputs = inputCatalog.availableInputDevices()
        guard !availableInputs.isEmpty else {
            yuwpLog("No audio input device found — cannot start capture")
            return false
        }

        let inputNode = engine.inputNode
        let selectedDevice = selectInputDevice(on: inputNode, availableInputs: availableInputs)
        let hardwareFormat = inputNode.inputFormat(forBus: 0)
        let clientFormat = inputNode.outputFormat(forBus: 0)

        // Sanity-check the live hardware format (invalid when device disappeared mid-access)
        guard hardwareFormat.sampleRate > 0, hardwareFormat.channelCount > 0 else {
            yuwpLog("Invalid hardware input format (\(hardwareFormat.sampleRate)Hz, \(hardwareFormat.channelCount)ch) — no usable mic?")
            return false
        }

        if hardwareFormat.sampleRate != clientFormat.sampleRate || hardwareFormat.channelCount != clientFormat.channelCount {
            yuwpLog(
                "Audio format mismatch detected before tap install — using hardware format " +
                "(hw: \(Int(hardwareFormat.sampleRate))Hz/\(hardwareFormat.channelCount)ch, " +
                "client: \(Int(clientFormat.sampleRate))Hz/\(clientFormat.channelCount)ch)"
            )
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

        // Convert from the live hardware capture format, not the engine's stale client format.
        guard let conv = AVAudioConverter(from: hardwareFormat, to: targetFormat) else {
            yuwpLog("Failed to create audio converter: \(hardwareFormat) -> \(targetFormat)")
            return false
        }
        converter = conv

        // Observe audio route changes (mic plugged/unplugged, Bluetooth disconnect)
        startRouteChangeObserver()

        // Buffer size: ~100ms at input sample rate
        let bufferSize = AVAudioFrameCount(hardwareFormat.sampleRate * 0.1)

        inputNode.installTap(onBus: 0, bufferSize: bufferSize, format: hardwareFormat) {
            [weak self] buffer, _ in
            guard let self, let converter = self.converter else { return }

            // Convert to 16kHz mono Int16
            let frameCapacity = AVAudioFrameCount(
                Double(buffer.frameLength) * self.targetSampleRate / hardwareFormat.sampleRate
            )
            guard let outputBuffer = AVAudioPCMBuffer(
                pcmFormat: targetFormat,
                frameCapacity: frameCapacity + 16
            ) else { return }

            var error: NSError?
            nonisolated(unsafe) var allConsumed = false
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

            for warning in self.warningTracker.ingest(rms: rms) {
                self.emitWarning(warning)
            }

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
            let deviceLabel = selectedDevice.map { " [\($0.name)]" } ?? ""
            yuwpLog(
                "Audio capture started\(deviceLabel) (hw \(Int(hardwareFormat.sampleRate))Hz/\(hardwareFormat.channelCount)ch, " +
                "client \(Int(clientFormat.sampleRate))Hz/\(clientFormat.channelCount)ch -> 16kHz mono)"
            )
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

    private func selectInputDevice(
        on inputNode: AVAudioInputNode,
        availableInputs: [AudioInputDeviceDescriptor]
    ) -> AudioInputDeviceDescriptor? {
        let requestedDevice = inputCatalog.resolve(inputSelection)
        let defaultDevice = availableInputs.first(where: \.isDefault) ?? availableInputs.first

        let targetDevice: AudioInputDeviceDescriptor?
        switch inputSelection {
        case .systemDefault:
            targetDevice = defaultDevice
        case .device(let uid):
            if let requestedDevice {
                targetDevice = requestedDevice
            } else {
                yuwpLog("Selected input device unavailable (\(uid)) — falling back to system default")
                targetDevice = defaultDevice
            }
        }

        guard let targetDevice else { return nil }
        guard let audioUnit = inputNode.audioUnit else {
            yuwpLog("Input audio unit unavailable — using current system input device")
            return nil
        }

        var deviceID = targetDevice.audioObjectID
        let size = UInt32(MemoryLayout<AudioObjectID>.size)
        let status = AudioUnitSetProperty(
            audioUnit,
            kAudioOutputUnitProperty_CurrentDevice,
            kAudioUnitScope_Global,
            0,
            &deviceID,
            size
        )
        guard status == noErr else {
            yuwpLog("Failed to select input device \(targetDevice.name) (status \(status)) — using current system input")
            return nil
        }

        if case .device = inputSelection {
            yuwpLog("Using input device: \(targetDevice.detailText)")
        }
        return targetDevice
    }

    private func emitWarning(_ warning: AudioCaptureWarning) {
        switch warning {
        case .routeChanged:
            yuwpLog("Audio route changed mid-session")
        case .silentInput(let seconds):
            yuwpLog("Warning: \(seconds)s of dead silence — mic may not be working")
        case .speechPause(let seconds):
            yuwpLog("Speech pause detected (\(seconds)s silence after speech)")
        }
        onWarning?(warning)
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
            guard let warning = WarningTracker.warningForRouteChange(elapsedSinceStart: elapsed) else {
                yuwpLog("Audio route change ignored (startup, \(String(format: "%.1f", elapsed))s)")
                return
            }
            self.emitWarning(warning)
        }
    }

    private func stopRouteChangeObserver() {
        if let observer = routeChangeObserver {
            NotificationCenter.default.removeObserver(observer)
            routeChangeObserver = nil
        }
    }
}
