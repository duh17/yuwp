// NativeASR — Text Decoder (Qwen3 LLM)
// Grouped Query Attention + QK-norm + RoPE + SwiGLU MLP.
//
// GOTCHAS:
// - QK-norm applied BEFORE RoPE (Qwen3-specific, missing it → garbage output)
// - GQA: nKVHeads < nHeads, MLXFast.SDPA handles the broadcast
// - .causal mask for prefill (seqLen > 1), .none for single-token decode (seq=1)
// - Our KVCache, not MLXLMCommon.KVCacheSimple

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Qwen3 Attention

final class Qwen3Attention: Module {
    let nHeads: Int
    let nKVHeads: Int
    let headDim: Int
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    let rope: RoPE

    init(config: TextDecoderConfig) {
        nHeads = config.numAttentionHeads
        nKVHeads = config.numKeyValueHeads
        headDim = config.headDim
        scale = pow(Float(headDim), -0.5)

        let dim = config.hiddenSize
        _qProj.wrappedValue = Linear(dim, nHeads * headDim, bias: config.attentionBias)
        _kProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.attentionBias)
        _vProj.wrappedValue = Linear(dim, nKVHeads * headDim, bias: config.attentionBias)
        _oProj.wrappedValue = Linear(nHeads * headDim, dim, bias: config.attentionBias)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.rmsNormEps)

        rope = RoPE(dimensions: headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil
    ) -> MLXArray {
        let (B, L, _) = (x.shape[0], x.shape[1], x.shape[2])

        var queries = qProj(x).reshaped([B, L, nHeads, headDim]).transposed(0, 2, 1, 3)
        var keys = kProj(x).reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)
        var values = vProj(x).reshaped([B, L, nKVHeads, headDim]).transposed(0, 2, 1, 3)

        // QK-norm BEFORE RoPE (Qwen3-specific — skip this and output is garbage)
        queries = qNorm(queries)
        keys = kNorm(keys)

        // RoPE with cache offset for correct position encoding
        let offset = cache?.offset ?? 0
        queries = rope(queries, offset: offset)
        keys = rope(keys, offset: offset)

        // KV cache update
        if let cache {
            (keys, values) = cache.update(keys: keys, values: values)
        }

        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: scale, mask: mask
        )

        return oProj(out.transposed(0, 2, 1, 3).reshaped([B, L, -1]))
    }
}

// MARK: - Qwen3 MLP (SwiGLU)

final class Qwen3MLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(config: TextDecoderConfig) {
        let d = config.hiddenSize
        let h = config.intermediateSize
        _gateProj.wrappedValue = Linear(d, h, bias: false)
        _upProj.wrappedValue = Linear(d, h, bias: false)
        _downProj.wrappedValue = Linear(h, d, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(silu(gateProj(x)) * upProj(x))
    }
}

// MARK: - Qwen3 Decoder Layer

final class Qwen3DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Qwen3Attention
    @ModuleInfo var mlp: Qwen3MLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttnNorm: RMSNorm

    init(config: TextDecoderConfig) {
        _selfAttn.wrappedValue = Qwen3Attention(config: config)
        _mlp.wrappedValue = Qwen3MLP(config: config)
        _inputNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
        _postAttnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    func callAsFunction(
        _ x: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode = .none,
        cache: KVCache? = nil
    ) -> MLXArray {
        var h = x
        let r1 = h
        h = selfAttn(inputNorm(h), mask: mask, cache: cache)
        h = r1 + h

        let r2 = h
        h = mlp(postAttnNorm(h))
        return r2 + h
    }
}

// MARK: - Qwen3 Text Model

final class Qwen3TextModel: Module {
    let config: TextDecoderConfig

    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [Qwen3DecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(config: TextDecoderConfig) {
        self.config = config
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.vocabSize, dimensions: config.hiddenSize)
        _layers.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in Qwen3DecoderLayer(config: config) }
        _norm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.rmsNormEps)
    }

    /// Forward pass.
    /// - Parameters:
    ///   - inputIds: Token IDs, shape (B, L). Mutually exclusive with inputEmbeddings.
    ///   - inputEmbeddings: Pre-computed embeddings (for audio-text fusion), shape (B, L, hidden).
    ///   - cache: Per-layer KV caches. Pass nil on first call (caches are created and returned).
    /// - Returns: (hidden states, updated KV caches)
    func callAsFunction(
        inputIds: MLXArray? = nil,
        inputEmbeddings: MLXArray? = nil,
        cache: [KVCache]? = nil
    ) -> (MLXArray, [KVCache]) {
        var h: MLXArray
        if let emb = inputEmbeddings {
            h = emb
        } else if let ids = inputIds {
            h = embedTokens(ids)
        } else {
            fatalError("Either inputIds or inputEmbeddings must be provided")
        }

        // PERFORMANCE: .causal for prefill, .none for single-token decode with KV cache
        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = h.shape[1] > 1 ? .causal : .none

        let cacheList = cache ?? (0 ..< layers.count).map { _ in KVCache() }

        for (i, layer) in layers.enumerated() {
            h = layer(h, mask: maskMode, cache: cacheList[i])
        }

        return (norm(h), cacheList)
    }
}
