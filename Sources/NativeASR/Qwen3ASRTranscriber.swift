// NativeASR — Batch Transcriber
// Load a Qwen3-ASR model and transcribe a WAV file or audio samples.
// Phase 1: batch-only (no streaming). Phase 2 will add the StreamingSession.

@preconcurrency import AVFoundation
import Foundation
import MLX
import MLXRandom
import MLXNN

// MARK: - Result

public struct TranscriptionResult: Sendable {
    public let text: String
    public let language: String?
    public let audioDuration: Double
    public let processingTime: Double

    public init(text: String, language: String?, audioDuration: Double, processingTime: Double) {
        self.text = text
        self.language = language
        self.audioDuration = audioDuration
        self.processingTime = processingTime
    }

    public var rtf: Double { processingTime / max(audioDuration, 1e-6) }
    public var speedMultiplier: Double { audioDuration / max(processingTime, 1e-6) }
}

// MARK: - Transcriber

public final class Qwen3ASRTranscriber: @unchecked Sendable {
    public let model: Qwen3ASRModel
    public let tokenizer: Qwen3ASRTokenizer
    public let modelDirectory: URL

    private init(model: Qwen3ASRModel, tokenizer: Qwen3ASRTokenizer, modelDirectory: URL) {
        self.model = model
        self.tokenizer = tokenizer
        self.modelDirectory = modelDirectory
    }

#if YUWP_INTERNAL_DIAGNOSTICS
    /// Tiny in-memory transcriber for stop-tail session fixtures. Not a product path.
    static func testingTranscriber(
        model: Qwen3ASRModel,
        tokenizer: Qwen3ASRTokenizer,
        modelDirectory: URL = URL(fileURLWithPath: "/tmp/yuwp-stop-tail-test", isDirectory: true)
    ) -> Qwen3ASRTranscriber {
        Qwen3ASRTranscriber(model: model, tokenizer: tokenizer, modelDirectory: modelDirectory)
    }
#endif

    // MARK: - Loading

    /// Load model and tokenizer from a local directory.
    /// - Parameter directory: Path to directory with config.json, model.safetensors, vocab.json, merges.txt
    public static func load(from directory: URL) throws -> Qwen3ASRTranscriber {
        fputs("[NativeASR] Loading model from \(directory.path)...\n", stderr)
        let t0 = Date()

        let model = try Qwen3ASRModel.load(from: directory)
        let tokenizer = try Qwen3ASRTokenizer.load(from: directory)

        let elapsed = Date().timeIntervalSince(t0)
        fputs("[NativeASR] Model loaded in \(String(format: "%.1f", elapsed))s\n", stderr)

        return Qwen3ASRTranscriber(model: model, tokenizer: tokenizer, modelDirectory: directory)
    }

    // MARK: - Metal Warmup

    /// Run a dummy transcription to pre-compile Metal shaders.
    /// Call once after load() to eliminate JIT latency on first real transcription.
    public func warmup() throws {
        fputs("[NativeASR] Warming up Metal shaders...\n", stderr)
        let t0 = Date()
        let sampleCount = ASRAudio.sampleRate * 4  // 4 seconds of noise
        var noise = [Float](repeating: 0, count: sampleCount)
        for i in 0..<sampleCount { noise[i] = Float.random(in: -0.01...0.01) }
        _ = try transcribe(audio: noise, maxTokens: 16)
        MLX.Memory.clearCache()
        let elapsed = Date().timeIntervalSince(t0)
        fputs("[NativeASR] Warmup complete in \(String(format: "%.1f", elapsed))s\n", stderr)
    }

    // MARK: - Transcription

    /// Transcribe audio from a WAV file.
    public func transcribe(
        file url: URL,
        language: String? = nil,
        maxTokens: Int = 4096
    ) throws -> TranscriptionResult {
        let audio = try loadAudioFile(url)
        return try transcribe(audio: audio, language: language, maxTokens: maxTokens)
    }

    /// Transcribe raw audio samples (float32, 16kHz mono).
    public func transcribe(
        audio: [Float],
        language: String? = nil,
        maxTokens: Int = 4096,
        temperature: Float = 0.0,
        vocabularyHints: [String] = [],
        draftText: String? = nil
    ) throws -> TranscriptionResult {
        guard audio.count >= ASRAudio.nFft else {
            return TranscriptionResult(text: "", language: language, audioDuration: Double(audio.count) / Double(ASRAudio.sampleRate), processingTime: 0)
        }

        let t0 = Date()
        let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
        let tokenCap = min(maxTokens, Int(ceil(audioDuration * 20.0)) + 64)

        // Phase 1: Mel spectrogram
        let audioArray = MLXArray(audio)
        let melSpec = logMelSpectrogram(audio: audioArray)
        // Shape: (nFrames, nMels) → transpose to (nMels, nFrames) → add batch dim → (1, nMels, nFrames)
        let inputFeatures = melSpec.T.expandedDimensions(axis: 0)
        // Compute nFrames from audio length directly (avoid eval for shape)
        let nFrames = (audio.count + ASRAudio.nFft) / ASRAudio.hopLength
        let attnMask = MLXArray(Array(repeating: Int32(1), count: nFrames)).expandedDimensions(axis: 0)

        // Phase 2: Audio encoding (mel + encoder, lazy — will be evaluated when needed)
        let audioFeatures = model.getAudioFeatures(inputFeatures: inputFeatures, featureAttentionMask: attnMask)

        // Phase 3: Build prompt and fuse embeddings
        // numAudioTokens computed from audio length — no shape query needed
        let numAudioTokens = audioEncoderOutputLength(nFrames)
        let prompt = tokenizer.buildPromptTokens(
            numAudioTokens: numAudioTokens,
            language: language,
            vocabularyHints: vocabularyHints
        )
        let inputIds = MLXArray(prompt.tokenIds.map { Int32($0) }).expandedDimensions(axis: 0)
        let inputEmbeds = model.buildInputsEmbeds(
            inputIds: inputIds, audioFeatures: audioFeatures,
            numAudioTokens: numAudioTokens, audioStartIndex: prompt.audioPadStartIndex
        )

        // Phase 4: The draft is an acceleration hint, not a prompt prefix. Only
        // greedy decoding can use exact argmax verification; sampling is unchanged.
        let draftTokens = temperature <= 0 ? draftText.map(tokenizer.encodeData) ?? [] : []
        let generatedTokens: [Int]
        if !draftTokens.isEmpty, tokenCap > 0 {
            generatedTokens = decodeWithDraft(
                inputIds: inputIds, inputEmbeds: inputEmbeds,
                tokenCap: tokenCap, draftTokens: draftTokens
            )
        } else {
            generatedTokens = decodeSerial(
                inputIds: inputIds, inputEmbeds: inputEmbeds,
                tokenCap: tokenCap, temperature: temperature
            )
        }

        let cleanedText = language == nil
            ? tokenizer.cleanTokenOutput(generatedTokens)
            : tokenizer.cleanOutput(tokenizer.decode(generatedTokens))
        return TranscriptionResult(
            text: Self.trimPathologicalRepetition(in: cleanedText),
            language: language,
            audioDuration: audioDuration,
            processingTime: Date().timeIntervalSince(t0)
        )
    }

    private func decodeSerial(
        inputIds: MLXArray, inputEmbeds: MLXArray, tokenCap: Int, temperature: Float
    ) -> [Int] {
        // Double-buffer pattern matching Python's mlx_lm generate_step:
        // 1. Prefill: run full prompt through model
        // 2. Sample first token, queue next step, async eval
        // 3. Loop: read previous token (.item blocks), queue next step, async eval
        var (logits, cache) = model(inputIds: inputIds, inputEmbeddings: inputEmbeds, cache: nil)

        // First sample + queue next
        var y = sampleToken(logits: logits, temperature: temperature)
        var nextId = y.asType(.int32).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
        var nextEmbed = model.model.embedTokens(nextId)
        (logits, cache) = model(inputIds: nextId, inputEmbeddings: nextEmbed, cache: cache)
        var nextY = sampleToken(logits: logits, temperature: temperature)
        asyncEval(nextY)

        var generatedTokens: [Int] = []
        var repetitionCount = 0
        var lastToken = -1

        // First token — y.item() blocks until ready; asyncEval(nextY)
        // already ensures y is computed as a dependency.
        var n = 0
        while n < tokenCap {
            // Read current token (blocks until ready)
            let tokenId = y.item(Int.self)

            if tokenizer.isEOS(tokenId) { break }

            if tokenId == lastToken {
                repetitionCount += 1
                if repetitionCount >= 10 { break }
            } else {
                repetitionCount = 0
                lastToken = tokenId
            }

            generatedTokens.append(tokenId)
            n += 1

            if n >= tokenCap { break }

            // Advance: current = next, queue new next
            y = nextY
            nextId = y.asType(.int32).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            nextEmbed = model.model.embedTokens(nextId)
            (logits, cache) = model(inputIds: nextId, inputEmbeddings: nextEmbed, cache: cache)
            nextY = sampleToken(logits: logits, temperature: temperature)
            asyncEval(nextY)
        }

        return generatedTokens
    }

    private func decodeWithDraft(
        inputIds: MLXArray, inputEmbeds: MLXArray, tokenCap: Int, draftTokens: [Int]
    ) -> [Int] {
        let (logits, cache) = model(inputIds: inputIds, inputEmbeddings: inputEmbeds)
        let firstToken = sampleToken(logits: logits, temperature: 0).item(Int.self)
        var forwardCalls = 0
        var verifiedPositions = 0
        var rejectedPositions = 0
        let result = TranscriptDraft.greedyDecode(
            firstToken: firstToken,
            maxTokens: tokenCap,
            eosTokens: Qwen3ASRTokenizer.eosTokens,
            draft: TranscriptDraft(tokens: draftTokens),
            verify: { inputs in
                let ids = MLXArray(inputs.map { Int32($0) }).expandedDimensions(axis: 0)
                let (blockLogits, _) = model(
                    inputIds: ids, cache: cache, logitPositions: inputs.count
                )
                let choices = MLX.argMax(blockLogits, axis: -1)
                eval(choices)
                forwardCalls += 1
                verifiedPositions += inputs.count
                return choices.asArray(Int32.self).map(Int.init)
            },
            rewind: { count in
                rejectedPositions += count
                for layer in cache { layer.trim(n: count) }
            }
        )
        #if YUWP_INTERNAL_DIAGNOSTICS
        fputs("[NativeASR] DRAFT calls=\(forwardCalls) positions=\(verifiedPositions) rejected=\(rejectedPositions) emitted=\(result.count)\n", stderr)
        #endif
        return result
    }

    private static func trimPathologicalRepetition(in text: String) -> String {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 24 else { return trimmed }

        let chars = Array(trimmed)
        let maxPeriod = min(24, chars.count / 4)
        guard maxPeriod >= 3 else { return trimmed }

        for period in 3 ... maxPeriod {
            let patternStart = chars.count - period
            let pattern = Array(chars[patternStart ..< chars.count])
            guard pattern.contains(where: { $0.isLetter || $0.isNumber }) else { continue }

            var repeatStart = patternStart
            var repeats = 1
            while repeatStart - period >= 0 {
                let candidate = Array(chars[(repeatStart - period) ..< repeatStart])
                if candidate == pattern {
                    repeats += 1
                    repeatStart -= period
                } else {
                    break
                }
            }

            if repeats >= 4 {
                let collapsed = String(chars[..<repeatStart] + pattern)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                if collapsed.count < trimmed.count {
                    fputs("[NativeASR] Batch repetition trimmed \(trimmed.count)→\(collapsed.count) chars\n", stderr)
                    return collapsed
                }
            }
        }

        return trimmed
    }

    // MARK: - Token Sampling

    private func sampleToken(logits: MLXArray, temperature: Float) -> MLXArray {
        let lastLogits = logits[0, -1, 0...]
        if temperature <= 0 {
            return MLX.argMax(lastLogits, axis: -1)
        }
        let probs = MLX.softmax(lastLogits / temperature, axis: -1)
        let uniform = MLXRandom.uniform(low: Float(0), high: Float(1), probs.shape)
        let gumbel = -MLX.log(-MLX.log(uniform + 1e-10) + 1e-10)
        return MLX.argMax(MLX.log(probs + 1e-10) + gumbel, axis: -1)
    }
}

// MARK: - Audio Loading

/// Load a WAV or other audio file and resample to 16kHz mono.
public func loadAudioFile(_ url: URL) throws -> [Float] {
    let file = try AVAudioFile(forReading: url)
    let format = file.processingFormat
    let frameCount = AVAudioFrameCount(file.length)

    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
        throw Qwen3ASRError.audioLoadFailed("Failed to create buffer")
    }
    try file.read(into: buffer)

    guard let channelData = buffer.floatChannelData else {
        throw Qwen3ASRError.audioLoadFailed("No float channel data")
    }

    let length = Int(buffer.frameLength)
    let channelCount = Int(format.channelCount)
    var samples: [Float]

    if channelCount == 1 {
        samples = Array(UnsafeBufferPointer(start: channelData[0], count: length))
    } else {
        samples = [Float](repeating: 0, count: length)
        for ch in 0 ..< channelCount {
            let ch_data = UnsafeBufferPointer(start: channelData[ch], count: length)
            for i in 0 ..< length { samples[i] += ch_data[i] }
        }
        let scale = 1.0 / Float(channelCount)
        for i in 0 ..< length { samples[i] *= scale }
    }

    // Resample if needed
    let srcRate = format.sampleRate
    if abs(srcRate - Double(ASRAudio.sampleRate)) > 1.0 {
        samples = try resampleAudio(samples, from: srcRate, to: Double(ASRAudio.sampleRate))
    }
    return samples
}

private final class ResampleInputState: @unchecked Sendable {
    var didProvide = false
}

private func resampleAudio(_ samples: [Float], from srcRate: Double, to dstRate: Double) throws -> [Float] {
    let srcFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: srcRate, channels: 1, interleaved: false)!
    let dstFmt = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: dstRate, channels: 1, interleaved: false)!

    guard let converter = AVAudioConverter(from: srcFmt, to: dstFmt) else {
        throw Qwen3ASRError.audioLoadFailed("Failed to create resampler")
    }

    let inCount = AVAudioFrameCount(samples.count)
    let outCount = AVAudioFrameCount(Double(inCount) * dstRate / srcRate)

    guard let inBuf = AVAudioPCMBuffer(pcmFormat: srcFmt, frameCapacity: inCount),
          let outBuf = AVAudioPCMBuffer(pcmFormat: dstFmt, frameCapacity: outCount)
    else { throw Qwen3ASRError.audioLoadFailed("Failed to create resample buffers") }

    inBuf.frameLength = inCount
    samples.withUnsafeBufferPointer { ptr in
        inBuf.floatChannelData![0].update(from: ptr.baseAddress!, count: samples.count)
    }

    let state = ResampleInputState()
    var convError: NSError?
    converter.convert(to: outBuf, error: &convError) { _, status in
        if state.didProvide { status.pointee = .endOfStream; return nil }
        state.didProvide = true
        status.pointee = .haveData
        return inBuf
    }
    if let e = convError { throw Qwen3ASRError.audioLoadFailed("Resample error: \(e)") }

    return Array(UnsafeBufferPointer(start: outBuf.floatChannelData![0], count: Int(outBuf.frameLength)))
}
