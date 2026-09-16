// NativeASR — Qwen3-ASR main model
// Ties AudioEncoder + Qwen3TextModel + LM head.
// Weight sanitization handles thinker. prefix and Conv2d transposition.

import Foundation
import MLX
import MLXNN

// MARK: - Model

public final class Qwen3ASRModel: Module {
    public let config: Qwen3ASRConfig

    @ModuleInfo(key: "audio_tower") var audioTower: AudioEncoder
    @ModuleInfo var model: Qwen3TextModel
    @ModuleInfo(key: "lm_head") var lmHead: Linear?

    public init(config: Qwen3ASRConfig) {
        self.config = config

        _audioTower.wrappedValue = AudioEncoder(config: config.audioConfig)
        _model.wrappedValue = Qwen3TextModel(config: config.textConfig)

        if !config.textConfig.tieWordEmbeddings {
            _lmHead.wrappedValue = Linear(config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false)
        } else {
            _lmHead.wrappedValue = nil
        }
    }

    public var numLayers: Int { config.textConfig.numHiddenLayers }

    // MARK: - Audio Encoding

    /// Encode mel spectrogram features through the audio tower.
    /// - Returns: shape (seqLen, outputDim)
    public func getAudioFeatures(
        inputFeatures: MLXArray,
        featureAttentionMask: MLXArray? = nil
    ) -> MLXArray {
        audioTower(inputFeatures, featureAttentionMask: featureAttentionMask)
    }

    // MARK: - Embedding Fusion

    /// Replace audio placeholder tokens in inputIds with actual audio features.
    /// - Parameters:
    ///   - inputIds: shape (1, seqLen)
    ///   - audioFeatures: shape (numAudioTokens, outputDim)
    /// - Returns: shape (1, seqLen, hiddenSize)
    /// Build input embeddings by direct concatenation.
    /// `audioStart` is the index of the first audio-pad token in the prompt.
    /// If not provided, falls back to the mask-based approach.
    public func buildInputsEmbeds(
        inputIds: MLXArray,
        audioFeatures: MLXArray,
        numAudioTokens: Int? = nil,
        audioStartIndex: Int? = nil
    ) -> MLXArray {
        // Fast path: when we know the audio token boundaries from prompt construction,
        // directly concatenate prefix_embeds + audio_features + suffix_embeds.
        // This avoids embedding N audio-pad tokens + cumsum/gather/mask overhead.
        if let start = audioStartIndex, let nAudio = numAudioTokens {
            let prefixIds = inputIds[0..., 0 ..< start]
            let suffixIds = inputIds[0..., (start + nAudio)...]
            let prefixEmbed = model.embedTokens(prefixIds).squeezed(axis: 0)
            let suffixEmbed = model.embedTokens(suffixIds).squeezed(axis: 0)
            let audioTyped = audioFeatures.asType(prefixEmbed.dtype)
            let combined = MLX.concatenated([prefixEmbed, audioTyped, suffixEmbed], axis: 0)
            return combined.expandedDimensions(axis: 0)
        }

        // Fallback: mask-based approach
        let inputsEmbeds = model.embedTokens(inputIds)
        let audioTyped = audioFeatures.asType(inputsEmbeds.dtype)
        let (_, seqLen, hiddenDim) = (inputsEmbeds.shape[0], inputsEmbeds.shape[1], inputsEmbeds.shape[2])
        let audioMask = (inputIds .== MLXArray(config.audioTokenId))
        let flatMask = audioMask.reshaped([seqLen])
        let flatEmbeds = inputsEmbeds.reshaped([seqLen, hiddenDim])
        let numAudioFeatures = numAudioTokens ?? audioTyped.shape[0]
        let audioCumsum = MLX.cumsum(flatMask.asType(.int32), axis: 0) - 1
        let audioIndices = MLX.clip(audioCumsum, min: 0, max: numAudioFeatures - 1)
        let gatheredAudio = audioTyped[audioIndices]
        let expandedMask = flatMask.expandedDimensions(axis: -1).asType(gatheredAudio.dtype)
        let result = expandedMask * gatheredAudio + (1.0 - expandedMask) * flatEmbeds
        return result.reshaped([1, seqLen, hiddenDim])
    }

    // MARK: - Forward Pass

    /// Full forward pass.
    /// - Returns: Last-position logits shaped `[batch, 1, vocabulary]` and the
    ///   updated KV caches. Set `logitPositions` to verify a bounded draft block;
    ///   the default still projects only the last position, including for prefill.
    ///
    /// PERFORMANCE: The LM head (vocab projection) is only computed for the last
    /// token position. All callers sample from `logits[0, -1]`, so computing
    /// logits for every prefill position wastes (seqLen-1) × hiddenSize × vocabSize
    /// multiply-adds. For a 163-token prefill with vocab=151936, this saves ~99%
    /// of the LM head compute.
    public func callAsFunction(
        inputIds: MLXArray,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]? = nil,
        logitPositions: Int = 1
    ) -> (MLXArray, [KVCache]) {
        let embeds = inputEmbeddings ?? model.embedTokens(inputIds)

        let (hidden, newCache) = model(inputEmbeddings: embeds, cache: cache)

        // Never project the entire audio prefill just to verify a short draft.
        let seqLen = hidden.shape[1]
        precondition(logitPositions > 0 && logitPositions <= seqLen)
        let lastHidden = seqLen > logitPositions
            ? hidden[0..., (seqLen - logitPositions) ..< seqLen, 0...]
            : hidden

        return (project(lastHidden), newCache)
    }

    private func project(_ hidden: MLXArray) -> MLXArray {
        if config.textConfig.tieWordEmbeddings {
            return model.embedTokens.asLinear(hidden)
        } else if let lmHead {
            return lmHead(hidden)
        } else {
            fatalError("No LM head and embeddings not tied")
        }
    }

    /// Create fresh KV caches for all decoder layers.
    public func makeCache() -> [KVCache] {
        (0 ..< numLayers).map { _ in KVCache() }
    }

    // MARK: - Weight Loading

    /// Sanitize weights from HuggingFace format:
    /// 1. Strip `thinker.` prefix
    /// 2. Transpose Conv2d weights OIHW → OHWI
    /// 3. Skip `lm_head.weight` (tied embeddings)
    public static func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        var result: [String: MLXArray] = [:]
        let hasThinkerPrefix = weights.keys.contains { $0.hasPrefix("thinker.") }

        for (key, value) in weights {
            var k = key
            var v = value

            // Strip thinker. prefix
            if k.hasPrefix("thinker.") {
                k = String(k.dropFirst("thinker.".count))
            }

            // Skip tied lm_head weight
            if k == "lm_head.weight" { continue }

            // Transpose Conv2d weights OIHW → OHWI (only for original HF format)
            if hasThinkerPrefix, k.contains("conv2d"), k.hasSuffix(".weight"), v.ndim == 4 {
                v = v.transposed(0, 2, 3, 1)
            }

            result[k] = v
        }

        return result
    }

    /// Load model from a local directory containing config.json + model.safetensors.
    public static func load(from directory: URL) throws -> Qwen3ASRModel {
        let config = try Qwen3ASRConfig.load(from: directory)
        let model = Qwen3ASRModel(config: config)

        let weightsURL = directory.appendingPathComponent("model.safetensors")
        guard FileManager.default.fileExists(atPath: weightsURL.path) else {
            throw Qwen3ASRError.modelLoadFailed("model.safetensors not found at \(weightsURL.path)")
        }

        let rawWeights = try MLX.loadArrays(url: weightsURL)
        let weights = sanitize(weights: rawWeights)

        // Detect quantized weights (have .scales keys)
        let isQuantized = weights.keys.contains { $0.contains(".scales") }
        if isQuantized {
            let groupSize = config.quantizationConfig?.groupSize ?? 64
            let bits = config.quantizationConfig?.bits ?? 4
            // Only quantize layers that actually have scales in the weights
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

// MARK: - Output Length Helper

/// Compute audio encoder output sequence length from mel frame count.
func audioEncoderOutputLength(_ melFrames: Int) -> Int {
    let leave = melFrames % 100
    let feat = leave > 0 ? (leave - 1) / 2 + 1 : 0
    let s1 = feat > 0 ? (feat - 1) / 2 + 1 : 0
    let s2 = s1 > 0 ? (s1 - 1) / 2 + 1 : 0
    return s2 + (melFrames / 100) * 13
}
