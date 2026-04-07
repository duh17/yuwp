@preconcurrency import AVFoundation
import Foundation

/// Captures microphone audio at 16kHz mono PCM and delivers raw buffers.
/// Also computes real-time RMS audio level for waveform visualization.
final class AudioCapture: @unchecked Sendable, AudioCapturing {  // AudioCapturing conformance
    private var engine = AVAudioEngine()
    private var isRunning = false
    private let targetSampleRate: Double = 16000
    private var converter: AVAudioConverter?

    /// Accumulated raw PCM for saving
    private var recordingBuffer: [Int16] = []
    private let bufferLock = NSLock()

    /// Audio level callback — fires with normalized RMS (0.0–1.0) per tap (~100ms).
    /// Called on the audio thread; dispatch to main if needed.
    var onAudioLevel: (@Sendable (Float) -> Void)?

    /// Start capturing. `onBuffer` is called with raw Int16 PCM data at 16kHz mono.
    /// Returns true if capture started successfully.
    @discardableResult
    func start(onBuffer: @escaping @Sendable (Data) -> Void) -> Bool {
        guard !isRunning else {
            yuwpLog("Audio capture already running — skipping start")
            return false
        }

        let inputNode = engine.inputNode
        let inputFormat = inputNode.outputFormat(forBus: 0)

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
            let byteCount = frameCount * 2
            let data = Data(bytes: channelData[0], count: byteCount)

            // Compute RMS from the Int16 samples and fire level callback
            if let levelCallback = self.onAudioLevel, frameCount > 0 {
                let ptr = channelData[0]
                var sumSq: Float = 0
                for i in 0..<frameCount {
                    let sample = Float(ptr[i]) / 32768.0
                    sumSq += sample * sample
                }
                let rms = sqrt(sumSq / Float(frameCount))
                // Normalize: typical speech RMS ~0.02-0.15, map to 0.0-1.0
                let normalized = min(rms / 0.12, 1.0)
                levelCallback(normalized)
            }

            // Accumulate for saving
            self.bufferLock.lock()
            let ptr = channelData[0]
            for i in 0..<frameCount {
                self.recordingBuffer.append(ptr[i])
            }
            self.bufferLock.unlock()

            onBuffer(data)
        }

        do {
            try engine.start()
            isRunning = true
            yuwpLog("Audio capture started (\(Int(inputFormat.sampleRate))Hz -> 16kHz mono)")
            return true
        } catch {
            yuwpLog("Failed to start audio engine: \(error)")
            // Clean up the tap we just installed
            inputNode.removeTap(onBus: 0)
            converter = nil
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

        bufferLock.lock()
        let pcm = recordingBuffer
        recordingBuffer = []
        bufferLock.unlock()

        yuwpLog("Audio capture stopped (\(pcm.count) samples, \(String(format: "%.1f", Double(pcm.count) / targetSampleRate))s)")

        guard !pcm.isEmpty else { return nil }
        return pcm.withUnsafeBufferPointer { ptr in
            Data(buffer: ptr)
        }
    }
}
