import CoreML
import Foundation

/// Tiny CoreML wrapper for the Silero VAD v6 model.
///
/// `SileroVAD.swift` is local Yuwp code. The bundled
/// `Sources/NativeASR/Resources/silero_vad.mlmodelc` resource is derived from
/// FluidInference's `silero-vad-coreml` CoreML conversion (MIT), which in turn
/// is based on the original `snakers4/silero-vad` model (MIT).
/// See `THIRD_PARTY_NOTICES.md` for the acknowledgement trail we keep in-repo.
public final class SileroVAD: @unchecked Sendable {
    public static let sampleRate = 16_000
    public static let chunkSize = 576   // 36 ms at 16 kHz

    private let model: MLModel
    private var hiddenState: MLMultiArray
    private var cellState: MLMultiArray
    static let stateLength = 128
    private static let stateShape: [NSNumber] = [1, NSNumber(value: stateLength)]

    /// LSTM state plus unprocessed samples shorter than `chunkSize`.
    public struct StreamState: Sendable, Equatable {
        var hidden: [Float]
        var cell: [Float]
        public var leftover: [Float]

        public init() {
            self.init(
                hidden: Array(repeating: 0, count: SileroVAD.stateLength),
                cell: Array(repeating: 0, count: SileroVAD.stateLength),
                leftover: []
            )
        }

        init(hidden: [Float], cell: [Float], leftover: [Float]) {
            self.hidden = hidden
            self.cell = cell
            self.leftover = leftover
        }
    }

    public struct StreamActivity: Sendable, Equatable {
        public var hasSpeech: Bool
        public var speechDurationSec: Double
        public var processedFrameCount: Int
        public var speechFrameCount: Int
    }

    public init(modelURL: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .all
        model = try MLModel(contentsOf: modelURL, configuration: config)
        hiddenState = try MLMultiArray(shape: Self.stateShape, dataType: .float32)
        cellState = try MLMultiArray(shape: Self.stateShape, dataType: .float32)
    }

    public convenience init() throws {
        try self.init(bundle: .module)
    }

    public convenience init(bundle: Bundle) throws {
        guard let url = bundle.url(forResource: "silero_vad", withExtension: "mlmodelc") else {
            throw SileroVADError.modelNotFound
        }
        try self.init(modelURL: url)
    }

    public func reset() {
        hiddenState = (try? MLMultiArray(shape: Self.stateShape, dataType: .float32)) ?? hiddenState
        cellState = (try? MLMultiArray(shape: Self.stateShape, dataType: .float32)) ?? cellState
    }

    public func process(_ samples: [Float]) throws -> Float {
        let prediction = try predict(samples: samples, hidden: hiddenState, cell: cellState)
        hiddenState = prediction.hidden
        cellState = prediction.cell
        return prediction.probability
    }

    /// One 36 ms frame with caller-owned LSTM state. Does not use or update
    /// the instance hidden/cell buffers, so a shared model can serve many streams.
    public func process(_ samples: [Float], state: StreamState) throws -> (Float, StreamState) {
        guard state.hidden.count == Self.stateLength, state.cell.count == Self.stateLength else {
            throw SileroVADError.inferenceError
        }
        let hidden = try Self.floatsToMultiArray(state.hidden, shape: Self.stateShape)
        let cell = try Self.floatsToMultiArray(state.cell, shape: Self.stateShape)
        let prediction = try predict(samples: samples, hidden: hidden, cell: cell)
        var next = state
        if let copied = Self.floats(from: prediction.hidden) {
            next.hidden = copied
        }
        if let copied = Self.floats(from: prediction.cell) {
            next.cell = copied
        }
        return (prediction.probability, next)
    }

    /// Full 576-sample frames only. Remainder is carried in `state.leftover`
    /// instead of zero-padding a partial frame. `padFinalFrame` scores one
    /// zero-padded frame only when ending and leftover plus samples cannot
    /// form a full frame. Live chunks stay unpadded.
    public func processStream(
        _ samples: [Float],
        state: StreamState,
        padFinalFrame: Bool = false
    ) -> (StreamActivity, StreamState) {
        var (frames, leftover) = Self.takeFullFrames(
            leftover: state.leftover, samples: samples
        )
        if padFinalFrame, frames.isEmpty, !leftover.isEmpty {
            leftover.append(contentsOf: repeatElement(0, count: Self.chunkSize - leftover.count))
            frames = [leftover]
            leftover = []
        }
        var current = state
        current.leftover = leftover
        var speechFrames = 0
        for frame in frames {
            do {
                let (probability, next) = try process(frame, state: current)
                current = next
                if probability >= 0.5 {
                    speechFrames += 1
                }
            } catch {
                // Keep the previous LSTM state. Do not treat a failed frame as
                // scored silence.
            }
        }
        current.leftover = leftover
        let activity = StreamActivity(
            hasSpeech: speechFrames > 0,
            speechDurationSec: Self.streamSpeechDurationSec(speechFrameCount: speechFrames),
            processedFrameCount: frames.count,
            speechFrameCount: speechFrames
        )
        return (activity, current)
    }

    private struct VADPrediction {
        var probability: Float
        var hidden: MLMultiArray
        var cell: MLMultiArray
    }

    private func predict(
        samples: [Float],
        hidden: MLMultiArray,
        cell: MLMultiArray
    ) throws -> VADPrediction {
        guard samples.count == Self.chunkSize else {
            throw SileroVADError.invalidChunkSize(expected: Self.chunkSize, got: samples.count)
        }
        let audioInput = try Self.floatsToMultiArray(samples, shape: [1, NSNumber(value: Self.chunkSize)])
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "audio_input": MLFeatureValue(multiArray: audioInput),
            "hidden_state": MLFeatureValue(multiArray: hidden),
            "cell_state": MLFeatureValue(multiArray: cell),
        ])
        let result = try model.prediction(from: provider)
        let nextHidden = result.featureValue(for: "new_hidden_state")?.multiArrayValue ?? hidden
        let nextCell = result.featureValue(for: "new_cell_state")?.multiArrayValue ?? cell
        guard let output = result.featureValue(for: "vad_output")?.multiArrayValue else {
            throw SileroVADError.inferenceError
        }
        return VADPrediction(
            probability: output[0].floatValue,
            hidden: nextHidden,
            cell: nextCell
        )
    }

    static func takeFullFrames(
        leftover: [Float],
        samples: [Float],
        chunkSize: Int = chunkSize
    ) -> (frames: [[Float]], leftover: [Float]) {
        let totalCount = leftover.count + samples.count
        guard totalCount >= chunkSize else {
            if leftover.isEmpty { return ([], samples) }
            if samples.isEmpty { return ([], leftover) }
            return ([], leftover + samples)
        }
        var combined: [Float]
        if leftover.isEmpty {
            combined = samples
        } else {
            combined = leftover
            combined.append(contentsOf: samples)
        }
        let frameCount = combined.count / chunkSize
        var frames: [[Float]] = []
        frames.reserveCapacity(frameCount)
        var offset = 0
        for _ in 0 ..< frameCount {
            frames.append(Array(combined[offset ..< (offset + chunkSize)]))
            offset += chunkSize
        }
        let remaining = offset < combined.count ? Array(combined[offset...]) : []
        return (frames, remaining)
    }

    static func streamSpeechDurationSec(
        speechFrameCount: Int,
        chunkSize: Int = chunkSize,
        sampleRate: Int = sampleRate
    ) -> Double {
        Double(speechFrameCount * chunkSize) / Double(sampleRate)
    }

    private static func floatsToMultiArray(_ values: [Float], shape: [NSNumber]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        for i in 0 ..< values.count {
            pointer[i] = values[i]
        }
        return array
    }

    private static func floats(from array: MLMultiArray) -> [Float]? {
        guard array.count == stateLength else { return nil }
        return (0 ..< stateLength).map { array[$0].floatValue }
    }
}

public enum SileroVADError: LocalizedError {
    case modelNotFound
    case invalidChunkSize(expected: Int, got: Int)
    case inferenceError

    public var errorDescription: String? {
        switch self {
        case .modelNotFound:
            return "SileroVAD model not found in package resources"
        case let .invalidChunkSize(expected, got):
            return "SileroVAD expected \(expected) samples, got \(got)"
        case .inferenceError:
            return "SileroVAD inference failed"
        }
    }
}

public struct AudioChunk: Sendable {
    public let audio: [Float]
    public let startTime: Double
    public let endTime: Double

    public init(audio: [Float], startTime: Double, endTime: Double) {
        self.audio = audio
        self.startTime = startTime
        self.endTime = endTime
    }

    public var duration: Double { endTime - startTime }
}

public typealias VADAudioChunk = AudioChunk

public struct VADChunkingConfig: Sendable {
    public let threshold: Float
    public let negativeThreshold: Float
    public let minSpeechDuration: Double
    public let minSilenceDuration: Double
    public let speechPad: Double
    public let splitMinSilenceDuration: Double
    public let maxChunkDuration: Double?
    public let minChunkDuration: Double

    public init(
        threshold: Float = 0.5,
        negativeThreshold: Float? = nil,
        minSpeechDuration: Double = 0.25,
        minSilenceDuration: Double = 0.1,
        speechPad: Double = 0.03,
        splitMinSilenceDuration: Double = 2.0,
        maxChunkDuration: Double? = 120.0,
        minChunkDuration: Double = 30.0
    ) {
        self.threshold = threshold
        self.negativeThreshold = negativeThreshold ?? max(0, threshold - 0.15)
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.speechPad = speechPad
        self.splitMinSilenceDuration = splitMinSilenceDuration
        self.maxChunkDuration = maxChunkDuration
        self.minChunkDuration = minChunkDuration
    }
}

public struct EnergyChunkingConfig: Sendable {
    public let maxChunkDuration: Double
    public let minChunkDuration: Double
    public let searchExpandDuration: Double
    public let energyWindowDuration: Double
    public let minProgressDuration: Double

    public init(
        maxChunkDuration: Double = 1200.0,
        minChunkDuration: Double = 1.0,
        searchExpandDuration: Double = 5.0,
        energyWindowDuration: Double = 0.1,
        minProgressDuration: Double = 1.0
    ) {
        self.maxChunkDuration = maxChunkDuration
        self.minChunkDuration = minChunkDuration
        self.searchExpandDuration = searchExpandDuration
        self.energyWindowDuration = energyWindowDuration
        self.minProgressDuration = minProgressDuration
    }
}

public func chunkAudioByEnergy(
    _ audio: [Float],
    sampleRate: Int = SileroVAD.sampleRate,
    config: EnergyChunkingConfig = EnergyChunkingConfig(),
    strictMaxDuration: Bool = false
) -> [AudioChunk] {
    let audioDuration = Double(audio.count) / Double(sampleRate)
    if audio.isEmpty || audioDuration <= config.maxChunkDuration {
        return [AudioChunk(audio: audio, startTime: 0, endTime: audioDuration)]
    }

    let totalSamples = audio.count
    let maxChunkSamples = Int(config.maxChunkDuration * Double(sampleRate))
    let searchSamples = Int(config.searchExpandDuration * Double(sampleRate))
    let energyWindowSamples = max(1, Int(config.energyWindowDuration * Double(sampleRate)))
    let minProgressSamples = max(1, Int(config.minProgressDuration * Double(sampleRate)))
    let minChunkSamples = max(1, Int(config.minChunkDuration * Double(sampleRate)))

    var chunks: [AudioChunk] = []
    var startSample = 0

    while startSample < totalSamples {
        let endSample = min(startSample + maxChunkSamples, totalSamples)
        if endSample >= totalSamples {
            chunks.append(AudioChunk(
                audio: Array(audio[startSample ..< totalSamples]),
                startTime: Double(startSample) / Double(sampleRate),
                endTime: Double(totalSamples) / Double(sampleRate)
            ))
            break
        }

        // Subtitle ASR must never search past the hard cap. Move the last cut
        // backward when necessary so it does not leave a sub-second tail.
        let remainingTail = totalSamples - endSample
        let latestCut = strictMaxDuration && remainingTail > 0 && remainingTail < minChunkSamples
            ? totalSamples - minChunkSamples : endSample
        let searchStart = max(startSample, latestCut - searchSamples)
        let searchEnd = strictMaxDuration ? latestCut : min(totalSamples, endSample + searchSamples)
        let searchRegion = Array(audio[searchStart ..< searchEnd])

        var cutSample = endSample
        if searchRegion.count > energyWindowSamples {
            var prefix = [Double](repeating: 0, count: searchRegion.count + 1)
            for (index, sample) in searchRegion.enumerated() {
                let value = Double(sample)
                prefix[index + 1] = prefix[index] + value * value
            }

            var bestIndex = 0
            var bestEnergy = Double.greatestFiniteMagnitude
            let limit = searchRegion.count - energyWindowSamples
            if limit >= 0 {
                for windowStart in 0 ... limit {
                    let windowEnd = windowStart + energyWindowSamples
                    let energy = (prefix[windowEnd] - prefix[windowStart]) / Double(energyWindowSamples)
                    if energy < bestEnergy {
                        bestEnergy = energy
                        bestIndex = windowStart
                    }
                }
                cutSample = searchStart + bestIndex + energyWindowSamples / 2
            }
        }

        cutSample = max(cutSample, startSample + minProgressSamples)
        cutSample = min(cutSample, strictMaxDuration ? latestCut : totalSamples)

        chunks.append(AudioChunk(
            audio: Array(audio[startSample ..< cutSample]),
            startTime: Double(startSample) / Double(sampleRate),
            endTime: Double(cutSample) / Double(sampleRate)
        ))
        startSample = cutSample
    }

    return chunks
}

private struct VADSpeechSpan {
    let startSample: Int
    let endSample: Int
}

extension SileroVAD {
    /// Split 16kHz mono audio using Silero VAD speech/silence boundaries.
    ///
    /// When `maxChunkDuration` is set, oversized regions are force-split as a
    /// safety fallback. When it is `nil`, chunking is VAD-only and boundaries
    /// come exclusively from detected silence gaps, with only short-chunk merges.
    public func chunk(audio: [Float], config: VADChunkingConfig = VADChunkingConfig()) throws -> [VADAudioChunk] {
        let audioDuration = Double(audio.count) / Double(Self.sampleRate)

        let speechSpans = try detectSpeechSpans(in: audio, config: config)
        guard !speechSpans.isEmpty else {
            return [VADAudioChunk(audio: audio, startTime: 0, endTime: audioDuration)]
        }

        let splitMinSilenceSamples = Int(config.splitMinSilenceDuration * Double(Self.sampleRate))
        let maxChunkSamples = config.maxChunkDuration.map { Int($0 * Double(Self.sampleRate)) }
        let minChunkSamples = Int(config.minChunkDuration * Double(Self.sampleRate))

        var splitPoints: [Int] = []
        for i in 0 ..< max(0, speechSpans.count - 1) {
            let gapStart = speechSpans[i].endSample
            let gapEnd = speechSpans[i + 1].startSample
            let gapDuration = gapEnd - gapStart
            if gapDuration >= splitMinSilenceSamples {
                splitPoints.append((gapStart + gapEnd) / 2)
            }
        }

        var ranges: [(Int, Int)] = []
        var chunkStart = 0

        func appendForcedRanges(until end: Int) {
            guard let maxChunkSamples else { return }

            var cursor = chunkStart
            while end - cursor > maxChunkSamples {
                let forcedEnd = cursor + maxChunkSamples
                ranges.append((cursor, forcedEnd))
                cursor = forcedEnd
            }
            chunkStart = cursor
        }

        for splitPoint in splitPoints {
            appendForcedRanges(until: splitPoint)
            let chunkDuration = splitPoint - chunkStart
            if chunkDuration >= minChunkSamples {
                ranges.append((chunkStart, splitPoint))
                chunkStart = splitPoint
            }
        }

        appendForcedRanges(until: audio.count)
        if chunkStart < audio.count {
            ranges.append((chunkStart, audio.count))
        }

        if ranges.count > 1 {
            var merged: [(Int, Int)] = []
            for range in ranges {
                let duration = range.1 - range.0
                let canMergeWithPrevious: Bool
                if let last = merged.last, duration < minChunkSamples {
                    if let maxChunkSamples {
                        canMergeWithPrevious = (range.1 - last.0) <= maxChunkSamples
                    } else {
                        canMergeWithPrevious = true
                    }
                } else {
                    canMergeWithPrevious = false
                }

                if canMergeWithPrevious, let last = merged.last {
                    merged[merged.count - 1] = (last.0, range.1)
                } else {
                    merged.append(range)
                }
            }
            ranges = merged
        }

        return ranges.map { start, end in
            VADAudioChunk(
                audio: Array(audio[start ..< end]),
                startTime: Double(start) / Double(Self.sampleRate),
                endTime: Double(end) / Double(Self.sampleRate)
            )
        }
    }

    private func detectSpeechSpans(in audio: [Float], config: VADChunkingConfig) throws -> [VADSpeechSpan] {
        reset()

        let chunkSize = Self.chunkSize
        let sampleRate = Self.sampleRate
        let minSpeechSamples = Int(config.minSpeechDuration * Double(sampleRate))
        let minSilenceSamples = Int(config.minSilenceDuration * Double(sampleRate))
        let speechPadSamples = Int(config.speechPad * Double(sampleRate))

        let paddedCount = ((audio.count + chunkSize - 1) / chunkSize) * chunkSize
        var paddedAudio = audio
        if paddedCount > audio.count {
            paddedAudio.append(contentsOf: repeatElement(0, count: paddedCount - audio.count))
        }

        var spans: [VADSpeechSpan] = []
        var triggered = false
        var currentStart: Int?
        var possibleEnd: Int?

        for frameStart in stride(from: 0, to: paddedAudio.count, by: chunkSize) {
            let frame = Array(paddedAudio[frameStart ..< (frameStart + chunkSize)])
            let probability = try process(frame)

            if !triggered {
                if probability >= config.threshold {
                    triggered = true
                    currentStart = frameStart
                    possibleEnd = nil
                }
                continue
            }

            if probability < config.negativeThreshold {
                if possibleEnd == nil {
                    possibleEnd = frameStart
                }
                if let end = possibleEnd, frameStart - end >= minSilenceSamples {
                    if let start = currentStart {
                        spans.append(VADSpeechSpan(startSample: start, endSample: end))
                    }
                    triggered = false
                    currentStart = nil
                    possibleEnd = nil
                }
            } else {
                possibleEnd = nil
            }
        }

        if triggered, let start = currentStart {
            spans.append(VADSpeechSpan(startSample: start, endSample: audio.count))
        }

        spans = spans.filter { ($0.endSample - $0.startSample) >= minSpeechSamples }
        guard !spans.isEmpty else { return [] }

        var paddedSpans: [VADSpeechSpan] = []
        for span in spans {
            let start = max(0, span.startSample - speechPadSamples)
            let end = min(audio.count, span.endSample + speechPadSamples)
            if let last = paddedSpans.last, start <= last.endSample {
                paddedSpans[paddedSpans.count - 1] = VADSpeechSpan(startSample: last.startSample, endSample: max(last.endSample, end))
            } else {
                paddedSpans.append(VADSpeechSpan(startSample: start, endSample: end))
            }
        }
        return paddedSpans
    }
}
