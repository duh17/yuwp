// NativeASR — Qwen3-ForcedAligner
// Word-level audio-text alignment via timestamp classification.
// Port of mlx_audio.stt.models.qwen3_asr.qwen3_forced_aligner.
//
// Architecture: Same AudioEncoder + Qwen3TextModel as the ASR model,
// but lm_head outputs classify_num (5000) timestamp classes instead of vocab tokens.
// Single forward pass → argmax at <timestamp> positions × 80ms → word-level times.

import Foundation
import MLX
import MLXNN

// MARK: - Config

private struct AlignerThinkerConfig: Codable {
    let audioConfig: AudioEncoderConfig
    let textConfig: TextDecoderConfig
    let audioTokenId: Int
    let audioStartTokenId: Int
    let audioEndTokenId: Int
    let classifyNum: Int

    enum CodingKeys: String, CodingKey {
        case audioConfig = "audio_config"
        case textConfig = "text_config"
        case audioTokenId = "audio_token_id"
        case audioStartTokenId = "audio_start_token_id"
        case audioEndTokenId = "audio_end_token_id"
        case classifyNum = "classify_num"
    }
}

public struct ForcedAlignerConfig: Sendable {
    public let audioConfig: AudioEncoderConfig
    public let textConfig: TextDecoderConfig
    public let classifyNum: Int              // 5000 timestamp classes
    public let timestampSegmentTime: Float   // 80.0 ms per class
    public let timestampTokenId: Int         // 151705
    public let audioTokenId: Int             // 151676
    public let audioStartTokenId: Int        // 151669
    public let audioEndTokenId: Int          // 151670
    public let quantizationConfig: Qwen3ASRQuantizationConfig?
}

extension ForcedAlignerConfig: Decodable {
    enum CodingKeys: String, CodingKey {
        case thinkerConfig = "thinker_config"
        case timestampSegmentTime = "timestamp_segment_time"
        case timestampTokenId = "timestamp_token_id"
        case quantizationConfig = "quantization_config"
        case quantization
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)

        let thinker = try container.decode(AlignerThinkerConfig.self, forKey: .thinkerConfig)
        audioConfig = thinker.audioConfig
        textConfig = thinker.textConfig
        classifyNum = thinker.classifyNum
        audioTokenId = thinker.audioTokenId
        audioStartTokenId = thinker.audioStartTokenId
        audioEndTokenId = thinker.audioEndTokenId

        timestampSegmentTime = try container.decode(Float.self, forKey: .timestampSegmentTime)
        timestampTokenId = try container.decode(Int.self, forKey: .timestampTokenId)

        let q1 = try container.decodeIfPresent(Qwen3ASRQuantizationConfig.self, forKey: .quantizationConfig)
        let q2 = try container.decodeIfPresent(Qwen3ASRQuantizationConfig.self, forKey: .quantization)
        quantizationConfig = q1 ?? q2
    }

    public static func load(from directory: URL) throws -> ForcedAlignerConfig {
        let url = directory.appendingPathComponent("config.json")
        let data = try Data(contentsOf: url)
        return try JSONDecoder().decode(ForcedAlignerConfig.self, from: data)
    }
}

// MARK: - Model

public final class ForcedAlignerModel: Module {
    public let config: ForcedAlignerConfig

    @ModuleInfo(key: "audio_tower") var audioTower: AudioEncoder
    @ModuleInfo var model: Qwen3TextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    public init(config: ForcedAlignerConfig) {
        self.config = config
        _audioTower.wrappedValue = AudioEncoder(config: config.audioConfig)
        _model.wrappedValue = Qwen3TextModel(config: config.textConfig)
        _lmHead.wrappedValue = Linear(config.textConfig.hiddenSize, config.classifyNum, bias: false)
    }

    /// Forward pass: audio + text → timestamp class logits [1, seqLen, classifyNum].
    public func callAsFunction(
        inputIds: MLXArray,
        inputFeatures: MLXArray,
        featureAttentionMask: MLXArray? = nil
    ) -> MLXArray {
        // Encode audio
        let audioFeatures = audioTower(inputFeatures, featureAttentionMask: featureAttentionMask)

        // Fuse audio features into text embeddings at audio-pad positions
        let inputsEmbeds = model.embedTokens(inputIds)
        let audioTyped = audioFeatures.asType(inputsEmbeds.dtype)
        let seqLen = inputsEmbeds.shape[1]
        let hiddenDim = inputsEmbeds.shape[2]

        let audioMask = (inputIds .== MLXArray(Int32(config.audioTokenId)))
        let flatMask = audioMask.reshaped([seqLen])
        let flatEmbeds = inputsEmbeds.reshaped([seqLen, hiddenDim])
        let numAudio = audioTyped.shape[0]
        let audioCumsum = MLX.cumsum(flatMask.asType(.int32), axis: 0) - 1
        let audioIndices = MLX.clip(audioCumsum, min: 0, max: numAudio - 1)
        let gathered = audioTyped[audioIndices]
        let mask = flatMask.expandedDimensions(axis: -1).asType(gathered.dtype)
        let fused = mask * gathered + (1.0 - mask) * flatEmbeds
        let embeds = fused.reshaped([1, seqLen, hiddenDim])

        // Decode (single pass, no KV cache reuse)
        let (hidden, _) = model(inputEmbeddings: embeds, cache: nil)

        // Classify to timestamp classes
        return lmHead(hidden)
    }

    // MARK: - Weight Loading

    /// Sanitize weights from HuggingFace format.
    /// Unlike the ASR model, lm_head is NOT skipped (aligner uses it for timestamp classification).
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        let hasThinkerPrefix = weights.keys.contains { $0.hasPrefix("thinker.") }

        for (key, value) in weights {
            var k = key
            var v = value

            if k.hasPrefix("thinker.") {
                k = String(k.dropFirst("thinker.".count))
            }

            // Transpose Conv2d weights OIHW → OHWI (only for original HF format)
            if hasThinkerPrefix, k.contains("conv2d"), k.hasSuffix(".weight"), v.ndim == 4 {
                v = v.transposed(0, 2, 3, 1)
            }

            result[k] = v
        }
        return result
    }

    /// Load model from a directory with config.json + model.safetensors + vocab.json + merges.txt.
    public static func load(from directory: URL) throws -> ForcedAlignerModel {
        let config = try ForcedAlignerConfig.load(from: directory)
        let model = ForcedAlignerModel(config: config)

        let weightsURL = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw Qwen3ASRError.modelLoadFailed("model.safetensors not found at \(weightsURL.path)")
        }

        let rawWeights = try MLX.loadArrays(url: weightsURL)
        let weights = sanitize(weights: rawWeights)

        let isQuantized = weights.keys.contains { $0.contains(".scales") }
        if isQuantized {
            let groupSize = config.quantizationConfig?.groupSize ?? 64
            let bits = config.quantizationConfig?.bits ?? 8
            MLXNN.quantize(model: model, groupSize: groupSize, bits: bits) { path, _ in
                weights["\(path).scales"] != nil
            }
        }

        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .noUnusedKeys)
        model.train(false)
        eval(model)
        return model
    }
}

// MARK: - Result Types

public struct ForcedAlignItem: Sendable {
    public let text: String
    public let startTime: Double   // seconds
    public let endTime: Double     // seconds
    public let alignText: String?

    public init(text: String, startTime: Double, endTime: Double, alignText: String? = nil) {
        self.text = text
        self.startTime = startTime
        self.endTime = endTime
        self.alignText = alignText
    }
}

// MARK: - Text Preprocessing & Timestamp Logic

enum AlignmentProcessor {
    struct AlignmentWord: Equatable {
        var text: String
        let alignText: String
    }

    // MARK: Word Tokenization

    /// Split text into alignment words based on language.
    static func tokenizeWords(_ text: String, language: String) -> [String] {
        prepareWords(text, language: language).map(\.alignText)
    }

    static func prepareWords(_ text: String, language: String) -> [AlignmentWord] {
        switch language.lowercased() {
        case "chinese", "cantonese":
            return tokenizeChineseMixedWords(text)
        case "japanese":
            // Character-level for CJK, grouped for Latin (no nagisa dependency)
            return tokenizeChineseMixedWords(text)
        default:
            return tokenizeSpaceLangWords(text)
        }
    }

    /// Space-separated languages: preserve display text, but align on cleaned tokens.
    static func tokenizeSpaceLangWords(_ text: String) -> [AlignmentWord] {
        var words: [AlignmentWord] = []

        func appendWord(displayText: String, alignText: String) {
            guard !alignText.isEmpty else { return }
            words.append(AlignmentWord(text: displayText, alignText: alignText))
        }

        func appendDisplayOnly(_ rawText: String) {
            guard !rawText.isEmpty, !words.isEmpty else { return }
            words[words.count - 1].text.append(rawText)
        }

        for segment in text.split(omittingEmptySubsequences: true, whereSeparator: { $0.isWhitespace }) {
            let rawSegment = String(segment)
            for piece in splitCJKDisplay(rawSegment) {
                let cleaned = cleanToken(piece)
                if cleaned.isEmpty {
                    appendDisplayOnly(piece)
                } else {
                    appendWord(displayText: piece, alignText: cleaned)
                }
            }
        }

        return words
    }

    /// Chinese mixed: each CJK character is its own token, Latin characters are grouped.
    static func tokenizeChineseMixedWords(_ text: String) -> [AlignmentWord] {
        var tokens: [AlignmentWord] = []
        var latinBuf: [Character] = []

        func appendDisplayOnly(_ rawText: String) {
            guard !rawText.isEmpty, !tokens.isEmpty else { return }
            tokens[tokens.count - 1].text.append(rawText)
        }

        func flushLatin() {
            if !latinBuf.isEmpty {
                let raw = String(latinBuf)
                let cleaned = cleanToken(raw)
                if !cleaned.isEmpty { tokens.append(AlignmentWord(text: raw, alignText: cleaned)) }
                latinBuf.removeAll()
            }
        }

        for ch in text {
            if ch.unicodeScalars.contains(where: isCJK) {
                flushLatin()
                tokens.append(AlignmentWord(text: String(ch), alignText: String(ch)))
            } else if isKeptChar(ch) {
                latinBuf.append(ch)
            } else {
                flushLatin()
                appendDisplayOnly(String(ch))
            }
        }
        flushLatin()
        return tokens
    }

    /// Keep only letters, numbers, and apostrophes.
    static func cleanToken(_ token: String) -> String {
        String(token.filter { isKeptChar($0) })
    }

    static func isKeptChar(_ ch: Character) -> Bool {
        ch == "'" || ch.isLetter || ch.isNumber
    }

    static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        ScriptClassifier.isCJK(scalar)
    }

    /// Split a segment at CJK character boundaries, preserving original punctuation inside each piece.
    static func splitCJKDisplay(_ text: String) -> [String] {
        var tokens: [String] = []
        var buf = ""
        for ch in text {
            if ch.unicodeScalars.contains(where: isCJK) {
                if !buf.isEmpty { tokens.append(buf); buf = "" }
                tokens.append(String(ch))
            } else {
                buf.append(ch)
            }
        }
        if !buf.isEmpty { tokens.append(buf) }
        return tokens
    }

    // MARK: Input Construction

    /// Build input token IDs for forced alignment.
    /// Format: <audio_start> <audio_pad>×N <audio_end> word1_tokens <ts><ts> word2_tokens <ts><ts> ...
    static func buildInputIds(
        words: [String],
        numAudioTokens: Int,
        tokenizer: Qwen3ASRTokenizer,
        config: ForcedAlignerConfig
    ) -> [Int32] {
        var ids: [Int32] = []

        // Audio header
        ids.append(Int32(config.audioStartTokenId))
        ids.append(contentsOf: [Int32](repeating: Int32(config.audioTokenId), count: numAudioTokens))
        ids.append(Int32(config.audioEndTokenId))

        // Words interleaved with timestamp markers
        let tsId = Int32(config.timestampTokenId)
        for word in words {
            let wordTokens = tokenizer.encode(word)
            ids.append(contentsOf: wordTokens.map { Int32($0) })
            ids.append(tsId)
            ids.append(tsId)
        }

        return ids
    }

    // MARK: Timestamp Fixing

    /// Fix non-monotonic timestamps using Longest Increasing Subsequence + interpolation.
    /// Port of ForceAlignProcessor.fix_timestamp from Python.
    static func fixTimestamps(_ data: [Int]) -> [Int] {
        let n = data.count
        if n == 0 { return [] }
        if n == 1 { return data }

        // O(n²) DP to find LIS (non-strictly increasing)
        var dp = [Int](repeating: 1, count: n)
        var parent = [Int](repeating: -1, count: n)

        for i in 1 ..< n {
            for j in 0 ..< i {
                if data[j] <= data[i], dp[j] + 1 > dp[i] {
                    dp[i] = dp[j] + 1
                    parent[i] = j
                }
            }
        }

        let maxLen = dp.max()!
        let maxIdx = dp.firstIndex(of: maxLen)!

        // Reconstruct LIS indices
        var lisIndices: [Int] = []
        var idx = maxIdx
        while idx != -1 {
            lisIndices.append(idx)
            idx = parent[idx]
        }
        lisIndices.reverse()

        var isNormal = [Bool](repeating: false, count: n)
        for i in lisIndices { isNormal[i] = true }

        var result = data
        var i = 0

        while i < n {
            if !isNormal[i] {
                // Find end of anomaly run
                var j = i
                while j < n, !isNormal[j] { j += 1 }
                let count = j - i

                // Find nearest valid neighbors
                var leftVal: Int?
                for k in stride(from: i - 1, through: 0, by: -1) {
                    if isNormal[k] { leftVal = result[k]; break }
                }
                var rightVal: Int?
                for k in j ..< n {
                    if isNormal[k] { rightVal = result[k]; break }
                }

                if count <= 2 {
                    // Small anomaly: nearest neighbor
                    for k in i ..< j {
                        if leftVal == nil {
                            result[k] = rightVal!
                        } else if rightVal == nil {
                            result[k] = leftVal!
                        } else {
                            result[k] = (k - (i - 1)) <= (j - k) ? leftVal! : rightVal!
                        }
                    }
                } else {
                    // Large anomaly: linear interpolation
                    if let lv = leftVal, let rv = rightVal {
                        let step = Double(rv - lv) / Double(count + 1)
                        for k in i ..< j {
                            result[k] = lv + Int(step * Double(k - i + 1))
                        }
                    } else if let lv = leftVal {
                        for k in i ..< j { result[k] = lv }
                    } else if let rv = rightVal {
                        for k in i ..< j { result[k] = rv }
                    }
                }

                i = j
            } else {
                i += 1
            }
        }

        return result
    }
}

// MARK: - High-Level API

public final class ForcedAligner: @unchecked Sendable {
    public let model: ForcedAlignerModel
    public let tokenizer: Qwen3ASRTokenizer

    public init(model: ForcedAlignerModel, tokenizer: Qwen3ASRTokenizer) {
        self.model = model
        self.tokenizer = tokenizer
    }

    /// Load aligner from a model directory.
    public static func load(from directory: URL) throws -> ForcedAligner {
        let model = try ForcedAlignerModel.load(from: directory)
        let tokenizer = try Qwen3ASRTokenizer.load(from: directory)
        return ForcedAligner(model: model, tokenizer: tokenizer)
    }

    /// Align audio with transcript, returning word-level timestamps.
    /// - Parameters:
    ///   - audio: Raw audio samples at 16 kHz.
    ///   - text: Transcript to align.
    ///   - language: Language name (e.g. "English", "Chinese").
    /// - Returns: Per-word start/end times in seconds.
    public func align(audio: [Float], text: String, language: String = "English") -> [ForcedAlignItem] {
        let words = AlignmentProcessor.prepareWords(text, language: language)
        guard !words.isEmpty else { return [] }

        // Mel spectrogram → (nMels, nFrames) → (1, nMels, nFrames)
        let melSpec = logMelSpectrogram(audio: MLXArray(audio))
        let inputFeatures = melSpec.T.expandedDimensions(axis: 0)

        let nFrames = (audio.count + ASRAudio.nFft) / ASRAudio.hopLength
        let attnMask = MLXArray([Int32](repeating: 1, count: nFrames)).expandedDimensions(axis: 0)
        let numAudioTokens = audioEncoderOutputLength(nFrames)

        // Build input: audio header + word tokens interleaved with <timestamp> markers
        let inputIdValues = AlignmentProcessor.buildInputIds(
            words: words.map(\.alignText),
            numAudioTokens: numAudioTokens,
            tokenizer: tokenizer,
            config: model.config
        )
        let inputIds = MLXArray(inputIdValues).expandedDimensions(axis: 0)

        // Single forward pass
        let logits = model(inputIds: inputIds, inputFeatures: inputFeatures, featureAttentionMask: attnMask)
        let outputIds = MLX.argMax(logits, axis: -1).squeezed(axis: 0)
        eval(outputIds)

        // Extract predicted timestamps at <timestamp> token positions
        let tsId = Int32(model.config.timestampTokenId)
        let segTime = model.config.timestampSegmentTime

        var tsPositions: [Int] = []
        for (i, id) in inputIdValues.enumerated() {
            if id == tsId { tsPositions.append(i) }
        }

        var rawTimestamps: [Int] = []
        for pos in tsPositions {
            let classIdx = outputIds[pos].item(Int.self)
            rawTimestamps.append(Int(Float(classIdx) * segTime))
        }

        // Fix non-monotonic predictions
        let fixed = AlignmentProcessor.fixTimestamps(rawTimestamps)

        // Build result: 2 timestamps per word (start, end)
        var items: [ForcedAlignItem] = []
        for (i, word) in words.enumerated() {
            let startMs = fixed[i * 2]
            let endMs = fixed[i * 2 + 1]
            items.append(ForcedAlignItem(
                text: word.text,
                startTime: round(Double(startMs) / 1000.0 * 1000) / 1000,
                endTime: round(Double(endMs) / 1000.0 * 1000) / 1000,
                alignText: word.alignText == word.text ? nil : word.alignText
            ))
        }

        MLX.Memory.clearCache()
        return items
    }
}
