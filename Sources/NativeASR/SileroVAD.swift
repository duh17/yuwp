import CoreML
import Foundation

/// Tiny CoreML wrapper for the Silero VAD v6 model.
///
/// Vendored and adapted from paean-ai/silero-vad-swift (MIT), trimmed to the
/// minimum surface we need for server-side long-audio chunking.
public final class SileroVAD: @unchecked Sendable {
    public static let sampleRate = 16_000
    public static let chunkSize = 576   // 36 ms at 16 kHz

    private let model: MLModel
    private var hiddenState: MLMultiArray
    private var cellState: MLMultiArray
    private static let stateShape: [NSNumber] = [1, 128]

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
        guard samples.count == Self.chunkSize else {
            throw SileroVADError.invalidChunkSize(expected: Self.chunkSize, got: samples.count)
        }

        let audioInput = try Self.floatsToMultiArray(samples, shape: [1, NSNumber(value: Self.chunkSize)])
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "audio_input": MLFeatureValue(multiArray: audioInput),
            "hidden_state": MLFeatureValue(multiArray: hiddenState),
            "cell_state": MLFeatureValue(multiArray: cellState),
        ])
        let result = try model.prediction(from: provider)

        if let newHiddenState = result.featureValue(for: "new_hidden_state")?.multiArrayValue {
            hiddenState = newHiddenState
        }
        if let newCellState = result.featureValue(for: "new_cell_state")?.multiArrayValue {
            cellState = newCellState
        }
        guard let output = result.featureValue(for: "vad_output")?.multiArrayValue else {
            throw SileroVADError.inferenceError
        }
        return output[0].floatValue
    }

    private static func floatsToMultiArray(_ values: [Float], shape: [NSNumber]) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape, dataType: .float32)
        let pointer = array.dataPointer.bindMemory(to: Float.self, capacity: values.count)
        for i in 0 ..< values.count {
            pointer[i] = values[i]
        }
        return array
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

public struct VADAudioChunk: Sendable {
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
        if let maxChunkDuration = config.maxChunkDuration, audioDuration <= maxChunkDuration {
            return [VADAudioChunk(audio: audio, startTime: 0, endTime: audioDuration)]
        }

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
