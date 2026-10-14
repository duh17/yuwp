import Foundation
@preconcurrency import MLX
import MLXRandom

/// Device-side YuE2 sampling with the native protocol's filters over phase-local logits.
/// Nothing here reads back to the host, so the decode loop can queue the next step first.
struct YuE2Sampler {
    let phase: YuE2Phase
    let sampling: YuE2Sampling
    /// Historical `off` sampling keeps BF16 scores and protects the top three ranks in top-p.
    let legacyOff: Bool
    let size: Int
    private var key: MLXArray
    /// Last `penaltyWindow` sampled local ids, oldest first; -1 marks empty slots.
    private(set) var window: MLXArray
    private let endMask: MLXArray
    /// Top-k, top-p and the draw run on the `poolSize` highest scores instead of the whole phase
    /// vocabulary. The margin over k keeps every token tied at the k-th value in the pool.
    private let poolSize: Int
    private let rankFloor: MLXArray

    init(phase: YuE2Phase, sampling: YuE2Sampling, seed: UInt64, legacyOff: Bool) {
        self.phase = phase
        self.sampling = sampling
        self.legacyOff = legacyOff
        size = YuE2Protocol.phaseSize(phase)
        // MLX keys are two uint32 words: [seed >> 32, seed & 0xffffffff].
        key = MLXRandom.key(seed)
        window = MLXArray([Int32](repeating: -1, count: sampling.penaltyWindow))
        endMask = MLXArray(Int32(YuE2Protocol.endIndex(phase))) .== MLXArray(0 ..< Int32(size))
        poolSize = min(size, min(sampling.topK, size) + 256)
        rankFloor = MLXArray(0 ..< Int32(poolSize)) .>= MLXArray(Int32(legacyOff ? 3 : 1))
    }

    /// Phase-local `[1]` int32 token from bf16 `[1, size]` logits at decode `step`.
    mutating func sample(_ logits: MLXArray, step: Int) -> MLXArray {
        var scores = legacyOff ? logits : logits.asType(.float32)
        let negInf = MLXArray(-Float.infinity).asType(scores.dtype)
        if step < sampling.minTokens {
            scores = MLX.where(endMask, negInf, scores)
        }
        scores = penalize(scores)
        if sampling.temperature == 0 {
            return argMax(scores, axis: -1).asType(.int32)
        }
        if sampling.temperature != 1 {
            scores = (scores.asType(.float32) / sampling.temperature).asType(scores.dtype)
        }
        // Ascending ids give ties the same order a stable sort of the full vocabulary would.
        let pool = sorted(argPartition(-scores, kth: poolSize - 1, axis: -1)[0..., ..<poolSize], axis: -1)
        var values = takeAlong(scores, pool, axis: -1)
        let k = min(sampling.topK, size)
        let threshold = top(values, k: k, axis: -1).min(axis: -1, keepDims: true)
        values = MLX.where(values .< threshold, negInf, values)
        if sampling.topP < 1 {
            let order = argSort(-values, axis: -1)
            var ranked = takeAlong(values, order, axis: -1)
            let probabilities = softmax(ranked, axis: -1, precise: true)
            let preceding = (cumsum(probabilities, axis: -1) - probabilities).asType(scores.dtype)
            let removed = (preceding.asType(.float32) .> sampling.topP) .&& rankFloor
            ranked = MLX.where(removed, negInf, ranked)
            values = putAlong(ranked, order, values: ranked, axis: -1)
        }
        let keys = MLXRandom.split(key: key, into: 2)
        key = keys[0]
        // categorical() is argmax(log p + Gumbel noise over the full vocabulary); drawing the full-size
        // noise from the same key and reading it at the pool ids picks the identical token.
        let logProbabilities = log(softmax(values, axis: -1, precise: true).asType(.float32))
        let noise = takeAlong(MLXRandom.gumbel([1, size], dtype: .float32, key: keys[1]), pool, axis: -1)
        let choice = argMax(logProbabilities + noise, axis: -1, keepDims: true)
        return takeAlong(pool, choice, axis: -1).reshaped(1).asType(.int32)
    }

    /// Record a sampled content token in the repetition window (lazy, device-side).
    mutating func remember(_ token: MLXArray) {
        window = concatenated([window[1...], token.reshaped(1)], axis: 0)
    }

    /// YuE2's frequency-powered penalty: a token seen n times is scaled by penalty^n.
    private func penalize(_ scores: MLXArray) -> MLXArray {
        guard sampling.repetitionPenalty != 1 else { return scores }
        let valid = (window .>= 0).asType(.float32)
        let counts = MLXArray.zeros([size], dtype: .float32).at[maximum(window, 0)].add(valid)
        let alpha = pow(MLXArray(sampling.repetitionPenalty), counts).asType(scores.dtype)
        return MLX.where(scores .< 0, (scores * alpha).asType(scores.dtype), (scores / alpha).asType(scores.dtype))
    }
}
