import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

/// Generator config (`config.json`). YuE2-3B is a Qwen3-shaped trunk whose layers carry an AR
/// weight set (text, ABC, codec ids) and a NAR weight set (latent canvas).
public struct YuE2Config: Decodable, Sendable {
    public let hiddenSize: Int
    public let intermediateSize: Int
    public let layers: Int
    public let heads: Int
    public let kvHeads: Int
    public let headDim: Int
    public let vocabSize: Int
    public let rmsNormEps: Float
    public let ropeTheta: Float
    public let maxPositions: Int
    public let latentDim: Int
    public let maxLatentFrames: Int
    public let timestepShift: Float

    enum CodingKeys: String, CodingKey {
        case hiddenSize = "hidden_size"
        case intermediateSize = "intermediate_size"
        case layers = "num_hidden_layers"
        case heads = "num_attention_heads"
        case kvHeads = "num_key_value_heads"
        case headDim = "head_dim"
        case vocabSize = "vocab_size"
        case rmsNormEps = "rms_norm_eps"
        case ropeTheta = "rope_theta"
        case maxPositions = "max_position_embeddings"
        case latentDim = "latent_dim"
        case maxLatentFrames = "max_latent_frames"
        case timestepShift = "timestep_shift"
    }

    public static func load(directory: URL) throws -> YuE2Config {
        let config = try JSONDecoder().decode(YuE2Config.self, from: Data(contentsOf: directory.appendingPathComponent("config.json")))
        guard config.vocabSize == YuE2Protocol.vocabSize, config.maxPositions == YuE2Protocol.context,
              config.latentDim == YuE2Protocol.latentDim, config.heads % config.kvHeads == 0
        else { throw YuE2Error.invalidConfig("config.json is not a YuE2-3B generator config") }
        return config
    }
}

final class YuE2Attention: Module {
    let heads: Int
    let kvHeads: Int
    let headDim: Int
    let ropeTheta: Float
    let scale: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ c: YuE2Config) {
        heads = c.heads
        kvHeads = c.kvHeads
        headDim = c.headDim
        ropeTheta = c.ropeTheta
        scale = 1 / Float(c.headDim).squareRoot()
        _qProj.wrappedValue = Linear(c.hiddenSize, c.heads * c.headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hiddenSize, c.kvHeads * c.headDim, bias: false)
        _oProj.wrappedValue = Linear(c.heads * c.headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        super.init()
    }

    /// Normed, rotated `[B, H, L, D]` projections at absolute positions `offset ..< offset + L`.
    func project(_ x: MLXArray, offset: Int) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let (B, L) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(B, L, heads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(B, L, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(B, L, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: ropeTheta, scale: 1, offset: offset)
        k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: ropeTheta, scale: 1, offset: offset)
        return (q, k, v)
    }

    func output(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let (B, L) = (q.dim(0), q.dim(2))
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(o.transposed(0, 2, 1, 3).reshaped(B, L, heads * headDim))
    }
}

private let swiGLU = compile(shapeless: true) { (gate: MLXArray, up: MLXArray) -> MLXArray in
    silu(gate) * up
}

final class YuE2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    init(_ c: YuE2Config) {
        _gateProj.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _upProj.wrappedValue = Linear(c.hiddenSize, c.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downProj(swiGLU(gateProj(x), upProj(x)))
    }
}

/// Fixed-capacity append-only K/V for one AR branch: `[1, kvHeads, capacity, headDim]`.
final class YuE2KVCache {
    private var keys: MLXArray
    private var values: MLXArray
    private(set) var offset = 0
    let capacity: Int

    init(_ c: YuE2Config, capacity: Int) {
        self.capacity = capacity
        keys = MLXArray.zeros([1, c.kvHeads, capacity, c.headDim], dtype: .bfloat16)
        values = MLXArray.zeros([1, c.kvHeads, capacity, c.headDim], dtype: .bfloat16)
    }

    func append(_ k: MLXArray, _ v: MLXArray) throws -> (MLXArray, MLXArray) {
        let end = offset + k.dim(2)
        guard end <= capacity else { throw YuE2Error.invalidInput("KV cache capacity \(capacity) exceeded") }
        keys[0..., 0..., offset ..< end, 0...] = k.asType(.bfloat16)
        values[0..., 0..., offset ..< end, 0...] = v.asType(.bfloat16)
        offset = end
        return (keys[0..., 0..., ..<end, 0...], values[0..., 0..., ..<end, 0...])
    }

    var state: [MLXArray] { [keys, values] }
}

final class YuE2ARLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: YuE2Attention
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: YuE2MLP

    init(_ c: YuE2Config) {
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _attention.wrappedValue = YuE2Attention(c)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _mlp.wrappedValue = YuE2MLP(c)
        super.init()
    }

    func callAsFunction(_ x: MLXArray, cache: YuE2KVCache) throws -> MLXArray {
        let (q, k, v) = attention.project(inputNorm(x), offset: cache.offset)
        let (keys, values) = try cache.append(k, v)
        let h = x + attention.output(q, keys, values, mask: x.dim(1) > 1 ? .causal : .none)
        return h + mlp(postNorm(h))
    }

    /// Causal pass over a whole sequence from position 0, returning this layer's K/V for NAR conditioning.
    func conditioning(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let (q, k, v) = attention.project(inputNorm(x), offset: 0)
        let h = x + attention.output(q, k, v, mask: .causal)
        return (h + mlp(postNorm(h)), k, v)
    }
}

final class YuE2ARBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "layers") var layers: [YuE2ARLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ c: YuE2Config) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabSize, dimensions: c.hiddenSize)
        _layers.wrappedValue = (0 ..< c.layers).map { _ in YuE2ARLayer(c) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        super.init()
    }
}

/// The AR half: Qwen3 backbone plus an untied LM head (bf16, or affine 8/4-bit linears).
final class YuE2ARModel: Module {
    @ModuleInfo(key: "model") var model: YuE2ARBackbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    let config: YuE2Config

    init(_ c: YuE2Config) {
        config = c
        _model.wrappedValue = YuE2ARBackbone(c)
        _lmHead.wrappedValue = Linear(c.hiddenSize, c.vocabSize, bias: false)
        super.init()
    }

    /// Final hidden state of the last position after appending `ids` to `caches`.
    func step(_ ids: MLXArray, caches: [YuE2KVCache]) throws -> MLXArray {
        var x = model.embedTokens(ids)
        for (layer, cache) in zip(model.layers, caches) {
            x = try layer(x, cache: cache)
        }
        return model.norm(x[0..., (x.dim(1) - 1)...])
    }

    static func load(directory: URL, file: String, config: YuE2Config) throws -> YuE2ARModel {
        let url = directory.appendingPathComponent(file)
        guard FileManager.default.fileExists(atPath: url.path) else { throw YuE2Error.missingWeights(url.path) }
        let weights = try MLX.loadArrays(url: url)
        let model = YuE2ARModel(config)
        if weights["lm_head.scales"] != nil {
            quantize(model: model) { path, module -> (groupSize: Int, bits: Int)? in
                guard module is Linear, let scales = weights[path + ".scales"], let weight = weights[path + ".weight"] else { return nil }
                let inputs = scales.dim(-1) * 64
                return (64, weight.dim(-1) * 32 / inputs)
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.train(false)
        eval(model.parameters())
        return model
    }
}

/// Rows of the LM head a phase may sample, prebuilt contiguous so each step is one matmul.
struct YuE2PhaseHead {
    let weight: MLXArray
    let scales: MLXArray?
    let biases: MLXArray?
    let groupSize: Int
    let bits: Int

    init(_ head: Linear, phase: YuE2Phase) {
        func rows(_ a: MLXArray) -> MLXArray {
            switch phase {
            case .abc:
                concatenated([a[..<YuE2Protocol.eod], a[YuE2Protocol.abcEnd ..< YuE2Protocol.abcEnd + 1]], axis: 0)
            case .semantic:
                a[YuE2Protocol.musicEnd ..< YuE2Protocol.codecOffset + YuE2Protocol.codecSize]
            }
        }
        if let q = head as? QuantizedLinear {
            weight = rows(q.weight)
            scales = rows(q.scales)
            biases = q.biases.map(rows)
            groupSize = q.groupSize
            bits = q.bits
        } else {
            weight = rows(head.weight)
            scales = nil
            biases = nil
            groupSize = 0
            bits = 0
        }
        eval([weight] + [scales, biases].compactMap { $0 })
    }

    /// `[1, 1, hidden]` → bf16 `[1, rows]` logits.
    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let h = hidden.reshaped(1, -1)
        let logits = if let scales {
            quantizedMatmul(h, weight, scales: scales, biases: biases, transpose: true, groupSize: groupSize, bits: bits)
        } else {
            matmul(h, weight.T)
        }
        return logits.asType(.bfloat16)
    }
}

final class YuE2NARLayer: Module {
    @ModuleInfo(key: "nar_input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "nar_self_attn") var attention: YuE2Attention
    @ModuleInfo(key: "nar_pre_mlp_layernorm") var preMLPNorm: RMSNorm
    @ModuleInfo(key: "nar_mlp") var mlp: YuE2MLP

    init(_ c: YuE2Config) {
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _attention.wrappedValue = YuE2Attention(c)
        _preMLPNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _mlp.wrappedValue = YuE2MLP(c)
        super.init()
    }
}

final class YuE2NARBackbone: Module {
    @ModuleInfo(key: "layers") var layers: [YuE2NARLayer]

    init(_ c: YuE2Config) {
        _layers.wrappedValue = (0 ..< c.layers).map { _ in YuE2NARLayer(c) }
        super.init()
    }
}

final class YuE2TimestepEmbedder: Module {
    @ModuleInfo(key: "mlp") var mlp: [Linear]
    /// Derived constant, not a checkpoint parameter (kept out of the MLXArray parameter tree).
    let frequencies: [Float]

    init(hidden: Int) {
        _mlp.wrappedValue = [Linear(256, hidden), Linear(hidden, hidden)]
        let half = 128
        frequencies = (0 ..< half).map { Foundation.exp(-Foundation.log(Float(10000)) * Float($0) / Float(half)) }
        super.init()
    }

    /// Sinusoid in FP32 (cos first) of the bf16-rounded timestep, then the checkpoint's BF16 MLP.
    /// Returns `[1, hidden]`.
    func callAsFunction(_ t: Float) -> MLXArray {
        let args = MLXArray(t).asType(.bfloat16).asType(.float32) * MLXArray(frequencies)
        let embedding = concatenated([cos(args), sin(args)], axis: -1).asType(.bfloat16).reshaped(1, -1)
        return mlp[1](silu(mlp[0](embedding)))
    }
}

final class YuE2LatentPositions: Module {
    @ParameterInfo(key: "pe") var pe: MLXArray

    init(frames: Int, hidden: Int) {
        _pe.wrappedValue = MLXArray.zeros([frames, hidden], dtype: .bfloat16)
        super.init()
    }
}

/// The NAR half: acoustic experts plus latent projections, conditioned on the bf16 AR path's K/V.
final class YuE2AcousticModel: Module {
    @ModuleInfo(key: "model") var model: YuE2NARBackbone
    @ModuleInfo(key: "llm2vae") var llm2vae: Linear
    @ModuleInfo(key: "vae2llm") var vae2llm: Linear
    @ModuleInfo(key: "time_embedder") var timeEmbedder: YuE2TimestepEmbedder
    @ModuleInfo(key: "latent_pos_embed") var latentPositions: YuE2LatentPositions
    let config: YuE2Config

    init(_ c: YuE2Config) {
        config = c
        _model.wrappedValue = YuE2NARBackbone(c)
        _llm2vae.wrappedValue = Linear(c.hiddenSize, c.latentDim)
        _vae2llm.wrappedValue = Linear(c.latentDim, c.hiddenSize)
        _timeEmbedder.wrappedValue = YuE2TimestepEmbedder(hidden: c.hiddenSize)
        _latentPositions.wrappedValue = YuE2LatentPositions(frames: c.maxLatentFrames, hidden: c.hiddenSize)
        super.init()
    }

    static func load(directory: URL, config: YuE2Config) throws -> YuE2AcousticModel {
        let url = directory.appendingPathComponent("nar-bf16.safetensors")
        guard FileManager.default.fileExists(atPath: url.path) else { throw YuE2Error.missingWeights(url.path) }
        var weights: [String: MLXArray] = [:]
        // The checkpoint's time MLP is Sequential(Linear, SiLU, Linear): indices 0 and 2.
        for (key, value) in try MLX.loadArrays(url: url) {
            weights[key.replacingOccurrences(of: "time_embedder.mlp.2.", with: "time_embedder.mlp.1.")] = value
        }
        let model = YuE2AcousticModel(config)
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.train(false)
        eval(model.parameters())
        return model
    }
}
