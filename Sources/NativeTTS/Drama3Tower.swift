import Foundation
@preconcurrency import MLX
import MLXFast
import MLXLMCommon
import MLXNN

// The Drama 3 codebook tower: shared enrichment layers then one residual layer per remaining codebook.
// Every tower layer is a causal transformer layer over the frame sequence with its own KV cache;
// layer i mixes the embedding of code i into the hidden state and head i predicts code i + 1.

final class Drama3TowerAttention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let rotaryDim: Int
    let ropeBase: Float

    @ModuleInfo(key: "wqkv") var wqkv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ config: Drama3Config.Tower) {
        heads = config.heads
        kvHeads = config.kvHeads
        headDim = config.headDim
        rotaryDim = config.rotaryDim
        ropeBase = config.ropeBase
        _wqkv.wrappedValue = Linear(config.dim, 2 * heads * headDim + 2 * kvHeads * headDim, bias: false)
        _wo.wrappedValue = Linear(heads * headDim, config.dim, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: config.normEps)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: KVCacheSimple) -> MLXArray {
        let B = x.dim(0)
        let L = x.dim(1)
        let queryWidth = heads * headDim
        let kvWidth = kvHeads * headDim
        // Checkpoint row order: [query | gate | key | value].
        let parts = split(wqkv(x), indices: [queryWidth, 2 * queryWidth, 2 * queryWidth + kvWidth], axis: -1)
        var queries = qNorm(parts[0].reshaped(B, L, heads, headDim)).transposed(0, 2, 1, 3)
        var keys = kNorm(parts[2].reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let values = parts[3].reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        let offset = cache.offset
        queries = MLXFast.RoPE(queries, dimensions: rotaryDim, traditional: true, base: ropeBase, scale: 1, offset: offset)
        keys = MLXFast.RoPE(keys, dimensions: rotaryDim, traditional: true, base: ropeBase, scale: 1, offset: offset)
        let (allKeys, allValues) = cache.update(keys: keys, values: values)
        let output = MLXFast.scaledDotProductAttention(
            queries: queries, keys: allKeys, values: allValues,
            scale: 1 / Float(headDim).squareRoot(), mask: L > 1 ? .causal : .none
        )
        .transposed(0, 2, 1, 3)
        .reshaped(B, L, queryWidth)
        return wo(output * sigmoid(parts[1]))
    }
}

final class Drama3TowerFeedForward: Module {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(dim: Int, hidden: Int) {
        _w1.wrappedValue = Linear(dim, hidden, bias: false)
        _w2.wrappedValue = Linear(hidden, dim, bias: false)
        _w3.wrappedValue = Linear(dim, hidden, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

final class Drama3TowerLayer: Module {
    let codebookSize: Int

    @ModuleInfo(key: "token_embedding") var tokenEmbedding: Embedding
    @ModuleInfo(key: "enorm") var enorm: RMSNorm
    @ModuleInfo(key: "hnorm") var hnorm: RMSNorm
    @ModuleInfo(key: "eh_proj") var ehProj: Linear
    @ModuleInfo(key: "attention") var attention: Drama3TowerAttention
    @ModuleInfo(key: "attention_norm") var attentionNorm: RMSNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: RMSNorm
    @ModuleInfo(key: "feed_forward") var feedForward: Drama3TowerFeedForward
    @ModuleInfo(key: "final_layernorm") var finalNorm: RMSNorm

    init(_ config: Drama3Config.Tower) {
        codebookSize = config.codebookSize
        _tokenEmbedding.wrappedValue = Embedding(embeddingCount: config.codebookSize, dimensions: config.dim)
        _enorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _hnorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _ehProj.wrappedValue = Linear(2 * config.dim, config.dim, bias: false)
        _attention.wrappedValue = Drama3TowerAttention(config)
        _attentionNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _ffnNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        _feedForward.wrappedValue = Drama3TowerFeedForward(dim: config.dim, hidden: config.intermediateSize)
        _finalNorm.wrappedValue = RMSNorm(dimensions: config.dim, eps: config.normEps)
        super.init()
    }

    /// `codes` [B, L] are this layer's input code; rows whose label is not semantic embed zeros.
    func callAsFunction(
        _ hidden: MLXArray, codes: MLXArray, labelIsSemantic: MLXArray, extraInput: MLXArray?, cache: KVCacheSimple
    ) -> MLXArray {
        var embeddings = tokenEmbedding(clip(codes, min: 0, max: codebookSize - 1))
        embeddings = MLX.where(labelIsSemantic[.ellipsis, .newAxis], embeddings, MLXArray(0, dtype: embeddings.dtype))
        if let extraInput {
            embeddings = embeddings + extraInput.asType(embeddings.dtype)
        }
        let mixed = ehProj(concatenated([enorm(embeddings), hnorm(hidden)], axis: -1))
        let residual = mixed + attention(attentionNorm(mixed), cache: cache)
        return finalNorm(feedForward(ffnNorm(residual)) + residual)
    }
}

final class Drama3Tower: Module {
    let config: Drama3Config.Tower
    let semanticBeginID: Int
    let semanticEndID: Int

    @ModuleInfo(key: "codebook_embeddings") var codebookEmbeddings: Embedding
    @ModuleInfo(key: "shared") var shared: [Drama3TowerLayer]
    @ModuleInfo(key: "residual") var residual: [Drama3TowerLayer]
    @ModuleInfo(key: "heads") var heads: [Linear]

    init(_ config: Drama3Config.Tower, semanticBeginID: Int, semanticEndID: Int) {
        self.config = config
        self.semanticBeginID = semanticBeginID
        self.semanticEndID = semanticEndID
        _codebookEmbeddings.wrappedValue = Embedding(
            embeddingCount: config.codebookSize * config.codebookCount, dimensions: config.dim
        )
        _shared.wrappedValue = (0 ..< config.sharedLayerCount).map { _ in Drama3TowerLayer(config) }
        _residual.wrappedValue = (0 ..< config.codebookCount - 1).map { _ in Drama3TowerLayer(config) }
        _heads.wrappedValue = (0 ..< config.codebookCount - 1).map { _ in
            Linear(config.dim, config.codebookSize, bias: false)
        }
        super.init()
    }

    var layerCount: Int { shared.count + residual.count }

    func isSemantic(_ tokens: MLXArray) -> MLXArray {
        (tokens .>= semanticBeginID) .&& (tokens .<= semanticEndID)
    }

    /// Summed embeddings of codes `first..<end` of each frame; `frames` [N, 1 + codebooks].
    func codebookEmbeddingSum(_ frames: MLXArray, first: Int, end: Int) -> MLXArray {
        let codes = clip(frames[0..., (1 + first) ..< (1 + end)], min: 0, max: config.codebookSize - 1)
        let offsets = MLXArray((first ..< end).map { Int32($0 * config.codebookSize) })
        return codebookEmbeddings(codes.asType(.int32) + offsets).sum(axis: 1)
    }

    /// Backbone input rows: semantic frames add the backbone codebooks to the token embedding.
    func backboneInput(frames: MLXArray, tokenEmbeddings: MLXArray) -> MLXArray {
        let count = config.backboneCodebookCount
        let mixed = (tokenEmbeddings
            + codebookEmbeddingSum(frames, first: 1, end: 1 + count).asType(tokenEmbeddings.dtype))
            * MLXArray(1 / Float(1 + count).squareRoot()).asType(tokenEmbeddings.dtype)
        return MLX.where(isSemantic(frames[0..., 0])[.ellipsis, .newAxis], mixed, tokenEmbeddings)
    }

    /// Extra input of the first shared layer: the codes the backbone did not see, of semantic input frames.
    func sharedInput(frames: MLXArray) -> MLXArray {
        let sum = codebookEmbeddingSum(frames, first: 1 + config.backboneCodebookCount, end: config.codebookCount)
        return MLX.where(isSemantic(frames[0..., 0])[.ellipsis, .newAxis], sum, MLXArray(0, dtype: sum.dtype))
    }

    func newCaches() -> [KVCacheSimple] {
        (0 ..< layerCount).map { _ in KVCacheSimple() }
    }

    /// Teacher-forced pass over prompt rows: row p is labelled by frame p + 1. Only fills the caches.
    func prefill(hidden: MLXArray, inputFrames: MLXArray, nextFrames: MLXArray, caches: [KVCacheSimple]) {
        let labelIsSemantic = isSemantic(nextFrames[0..., 0])[.newAxis]
        let extra = sharedInput(frames: inputFrames)[.newAxis]
        var h = hidden[.newAxis]
        let codeZero = nextFrames[0..., 1][.newAxis]
        for (index, layer) in shared.enumerated() {
            h = layer(h, codes: codeZero, labelIsSemantic: labelIsSemantic, extraInput: index == 0 ? extra : nil, cache: caches[index])
        }
        for (index, layer) in residual.enumerated() {
            h = layer(
                h, codes: nextFrames[0..., 1 + index][.newAxis], labelIsSemantic: labelIsSemantic,
                extraInput: nil, cache: caches[shared.count + index]
            )
        }
        eval(h)
    }

    /// One decode step for a single row: given the sampled main token and code 0, samples codes 1..<N.
    func decode(
        hidden: MLXArray, inputFrame: MLXArray, mainToken: MLXArray, codeZero: MLXArray,
        caches: [KVCacheSimple], sampleCode: (MLXArray, Int) -> MLXArray
    ) -> [MLXArray] {
        let labelIsSemantic = isSemantic(mainToken)[.newAxis]
        var h = hidden[.newAxis]
        let extra = sharedInput(frames: inputFrame)[.newAxis]
        let codeZeroRow = codeZero[.newAxis]
        for (index, layer) in shared.enumerated() {
            h = layer(h, codes: codeZeroRow, labelIsSemantic: labelIsSemantic, extraInput: index == 0 ? extra : nil, cache: caches[index])
        }
        var codes = [codeZero]
        for (index, layer) in residual.enumerated() {
            h = layer(
                h, codes: codes[codes.count - 1][.newAxis], labelIsSemantic: labelIsSemantic,
                extraInput: nil, cache: caches[shared.count + index]
            )
            codes.append(sampleCode(heads[index](h[0]), index))
        }
        return codes
    }
}
