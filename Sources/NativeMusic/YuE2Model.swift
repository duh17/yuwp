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

    /// q, k and v projections stacked into one matmul (see `fuseProjections`).
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ c: YuE2Config) {
        heads = c.heads
        kvHeads = c.kvHeads
        headDim = c.headDim
        ropeTheta = c.ropeTheta
        scale = 1 / Float(c.headDim).squareRoot()
        _qkvProj.wrappedValue = Linear(c.hiddenSize, (c.heads + 2 * c.kvHeads) * c.headDim, bias: false)
        _oProj.wrappedValue = Linear(c.heads * c.headDim, c.hiddenSize, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        _kNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.rmsNormEps)
        super.init()
    }

    /// Normed, rotated `[B, H, L, D]` projections at absolute positions `offset ..< offset + L`.
    func project(_ x: MLXArray, offset: Int) -> (q: MLXArray, k: MLXArray, v: MLXArray) {
        let (B, L) = (x.dim(0), x.dim(1))
        let qkv = qkvProj(x).reshaped(B, L, heads + 2 * kvHeads, headDim)
        var q = qNorm(qkv[0..., 0..., ..<heads]).transposed(0, 2, 1, 3)
        var k = kNorm(qkv[0..., 0..., heads ..< heads + kvHeads]).transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., (heads + kvHeads)...].transposed(0, 2, 1, 3)
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
    /// gate and up projections stacked into one matmul (see `fuseProjections`).
    @ModuleInfo(key: "gate_up_proj") var gateUpProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let intermediate: Int

    init(_ c: YuE2Config) {
        intermediate = c.intermediateSize
        _gateUpProj.wrappedValue = Linear(c.hiddenSize, 2 * c.intermediateSize, bias: false)
        _downProj.wrappedValue = Linear(c.intermediateSize, c.hiddenSize, bias: false)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gateUp = gateUpProj(x)
        return downProj(swiGLU(gateUp[.ellipsis, ..<intermediate], gateUp[.ellipsis, intermediate...]))
    }
}

/// Stack checkpoint q/k/v and gate/up tensors (weights and any quantization scales/biases) so each
/// group runs as one matmul. Rows are only concatenated: quantization groups run along the input
/// dimension, so fused 8/4-bit weights are the original rows unchanged.
func fuseProjections(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var fused = weights
    let groups = [("q_proj", "k_proj", "v_proj", "qkv_proj"), ("gate_proj", "up_proj", nil, "gate_up_proj")]
    for key in weights.keys {
        for (first, second, third, target) in groups where key.hasSuffix(".\(first).weight") {
            let prefix = String(key.dropLast("\(first).weight".count))
            for suffix in ["weight", "scales", "biases"] {
                let names = [first, second, third].compactMap { $0 }.map { "\(prefix)\($0).\(suffix)" }
                let parts = names.compactMap { weights[$0] }
                guard parts.count == names.count else { continue }
                fused["\(prefix)\(target).\(suffix)"] = concatenated(parts, axis: 0)
                names.forEach { fused[$0] = nil }
            }
        }
    }
    return fused
}

/// Append-only K/V for one AR branch, `[1, kvHeads, allocated, headDim]`, grown in 256-token
/// blocks so the buffer tracks the used length instead of the whole context.
final class YuE2KVCache {
    private var keys: MLXArray?
    private var values: MLXArray?
    private(set) var offset = 0
    let capacity: Int
    private let kvHeads: Int
    private let headDim: Int
    private static let block = 256

    init(_ c: YuE2Config, capacity: Int) {
        self.capacity = capacity
        kvHeads = c.kvHeads
        headDim = c.headDim
    }

    func append(_ k: MLXArray, _ v: MLXArray) throws -> (MLXArray, MLXArray) {
        let end = offset + k.dim(2)
        guard end <= capacity else { throw YuE2Error.invalidInput("KV cache capacity \(capacity) exceeded") }
        if end > (keys?.dim(2) ?? 0) {
            let grow = (end - offset + Self.block - 1) / Self.block * Self.block
            // Separate arrays: subscript writes mutate an MLXArray in place, so K and V must not share one.
            let blankKeys = MLXArray.zeros([1, kvHeads, grow, headDim], dtype: .bfloat16)
            let blankValues = MLXArray.zeros([1, kvHeads, grow, headDim], dtype: .bfloat16)
            if let keys, let values {
                self.keys = concatenated([keys[0..., 0..., ..<offset, 0...], blankKeys], axis: 2)
                self.values = concatenated([values[0..., 0..., ..<offset, 0...], blankValues], axis: 2)
            } else {
                keys = blankKeys
                values = blankValues
            }
        }
        guard let keys, let values else { throw YuE2Error.invalidInput("KV cache was not allocated") }
        keys[0..., 0..., offset ..< end, 0...] = k.asType(.bfloat16)
        values[0..., 0..., offset ..< end, 0...] = v.asType(.bfloat16)
        offset = end
        return (keys[0..., 0..., ..<end, 0...], values[0..., 0..., ..<end, 0...])
    }

    var state: [MLXArray] { [keys, values].compactMap { $0 } }
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

    /// Load `ar-<bits>bit.safetensors`, or `ar-bf16.safetensors` when `bits` is nil. A quantized file
    /// that is not on disk is made from the bf16 weights with the same recipe (affine, group 64,
    /// linear layers only; embeddings stay bf16).
    static func load(directory: URL, bits: Int?, config: YuE2Config) throws -> YuE2ARModel {
        let bf16 = directory.appendingPathComponent("ar-bf16.safetensors")
        let quantized = bits.map { directory.appendingPathComponent("ar-\($0)bit.safetensors") }
        let url = quantized.flatMap { FileManager.default.fileExists(atPath: $0.path) ? $0 : nil } ?? bf16
        guard FileManager.default.fileExists(atPath: url.path) else { throw YuE2Error.missingWeights((quantized ?? bf16).path) }
        let weights = fuseProjections(try MLX.loadArrays(url: url))
        let model = YuE2ARModel(config)
        if weights["lm_head.scales"] != nil {
            quantize(model: model) { path, module -> (groupSize: Int, bits: Int)? in
                guard module is Linear, let scales = weights[path + ".scales"], let weight = weights[path + ".weight"] else { return nil }
                let inputs = scales.dim(-1) * 64
                return (64, weight.dim(-1) * 32 / inputs)
            }
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        if let bits, weights["lm_head.scales"] == nil {
            quantize(model: model, groupSize: 64, bits: bits, mode: .affine) { _, module in module is Linear }
        }
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
        try model.update(parameters: ModuleParameters.unflattened(fuseProjections(weights)), verify: .all)
        model.train(false)
        eval(model.parameters())
        return model
    }
}
