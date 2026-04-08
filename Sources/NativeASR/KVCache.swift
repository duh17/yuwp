// NativeASR — KV Cache
// Simple per-layer key/value cache for autoregressive text generation.
//
// DESIGN:
// - Keys/values are stored as MLXArrays, shape [B, nHeads, seqLen, headDim]
// - offset tracks how many tokens have been cached (used for RoPE position)
// - update() appends new K/V and returns the full cache
// - For Phase 2 (streaming), setting offset = reuseLen before update() enables
//   delta prefill: new K/V overwrites cache from reuseLen position onwards.

import Foundation
import MLX

public final class KVCache: @unchecked Sendable {
    private var storedKeys: MLXArray?
    private var storedValues: MLXArray?

    /// Current number of cached token positions.
    /// Setting this to a smaller value enables delta prefill:
    /// the next update() will overwrite from this position.
    public var offset: Int = 0

    public init() {}

    /// Append new keys/values to the cache and return the full K/V history.
    ///
    /// If offset < storedKeys.seqLen, only the first `offset` positions from
    /// the existing cache are kept before appending — enabling delta prefill.
    ///
    /// - Parameters:
    ///   - newKeys: shape [B, nHeads, newLen, headDim]
    ///   - newValues: shape [B, nHeads, newLen, headDim]
    /// - Returns: (fullKeys, fullValues) with shape [B, nHeads, offset+newLen, headDim]
    public func update(keys newKeys: MLXArray, values newValues: MLXArray) -> (MLXArray, MLXArray) {
        if let existing = storedKeys, let existingV = storedValues {
            let existingLen = existing.shape[2]
            let k: MLXArray
            let v: MLXArray
            if offset < existingLen {
                // Delta-prefill: truncate cache to `offset` before appending
                let prefix = existing[0..., 0..., 0 ..< offset, 0...]
                let prefixV = existingV[0..., 0..., 0 ..< offset, 0...]
                k = MLX.concatenated([prefix, newKeys], axis: 2)
                v = MLX.concatenated([prefixV, newValues], axis: 2)
            } else {
                k = MLX.concatenated([existing, newKeys], axis: 2)
                v = MLX.concatenated([existingV, newValues], axis: 2)
            }
            storedKeys = k
            storedValues = v
            offset = k.shape[2]
            return (k, v)
        } else {
            storedKeys = newKeys
            storedValues = newValues
            offset = newKeys.shape[2]
            return (newKeys, newValues)
        }
    }

    /// Remove the last `n` token positions from the cache.
    public func trim(n: Int) {
        guard n > 0, let k = storedKeys, let v = storedValues else { return }
        let newLen = max(0, k.shape[2] - n)
        if newLen > 0 {
            storedKeys = k[0..., 0..., 0 ..< newLen, 0...]
            storedValues = v[0..., 0..., 0 ..< newLen, 0...]
        } else {
            storedKeys = nil
            storedValues = nil
        }
        offset = newLen
    }

    /// Reset the cache entirely.
    public func reset() {
        storedKeys = nil
        storedValues = nil
        offset = 0
    }
}
