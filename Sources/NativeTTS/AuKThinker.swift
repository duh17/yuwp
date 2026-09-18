import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

final class AuKThinkerRMSNorm: Module {
    let eps: Float
    let weight: MLXArray

    init(dimensions: Int, eps: Float) {
        self.eps = eps
        self.weight = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

final class AuKTextAttention: Module {
    let nHeads: Int
    let nKV: Int
    let dimHead: Int
    let scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear

    override init() {
        nHeads = AuKFlashConfig.thinkerHeads
        nKV = AuKFlashConfig.thinkerKVHeads
        dimHead = AuKFlashConfig.thinkerHidden / nHeads
        scale = pow(Float(dimHead), -0.5)
        _qProj.wrappedValue = Linear(AuKFlashConfig.thinkerHidden, nHeads * dimHead, bias: true)
        _kProj.wrappedValue = Linear(AuKFlashConfig.thinkerHidden, nKV * dimHead, bias: true)
        _vProj.wrappedValue = Linear(AuKFlashConfig.thinkerHidden, nKV * dimHead, bias: true)
        _oProj.wrappedValue = Linear(nHeads * dimHead, AuKFlashConfig.thinkerHidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let tokens = x.dim(1)
        var q = qProj(x).reshaped(batch, tokens, nHeads, dimHead).transposed(0, 2, 1, 3)
        var k = kProj(x).reshaped(batch, tokens, nKV, dimHead).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(batch, tokens, nKV, dimHead).transposed(0, 2, 1, 3)
        q = applyHalfSplitRoPE(q, cos: cos, sin: sin)
        k = applyHalfSplitRoPE(k, cos: cos, sin: sin)
        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .causal
        )
        return oProj(output.transposed(0, 2, 1, 3).reshaped(batch, tokens, -1))
    }
}

final class AuKTextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear

    override init() {
        _gateProj.wrappedValue = Linear(AuKFlashConfig.thinkerHidden, AuKFlashConfig.thinkerIntermediate, bias: false)
        _upProj.wrappedValue = Linear(AuKFlashConfig.thinkerHidden, AuKFlashConfig.thinkerIntermediate, bias: false)
        _downProj.wrappedValue = Linear(AuKFlashConfig.thinkerIntermediate, AuKFlashConfig.thinkerHidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gateProj(x)
        return downProj((gate * sigmoid(gate)) * upProj(x))
    }
}

final class AuKTextLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: AuKTextAttention
    @ModuleInfo var mlp: AuKTextMLP
    @ModuleInfo(key: "input_layernorm") var inputLayernorm: AuKThinkerRMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayernorm: AuKThinkerRMSNorm

    override init() {
        _selfAttn.wrappedValue = AuKTextAttention()
        _mlp.wrappedValue = AuKTextMLP()
        _inputLayernorm.wrappedValue = AuKThinkerRMSNorm(
            dimensions: AuKFlashConfig.thinkerHidden,
            eps: AuKFlashConfig.thinkerRMSNormEps
        )
        _postAttentionLayernorm.wrappedValue = AuKThinkerRMSNorm(
            dimensions: AuKFlashConfig.thinkerHidden,
            eps: AuKFlashConfig.thinkerRMSNormEps
        )
    }

    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        var h = x + selfAttn(inputLayernorm(x), cos: cos, sin: sin)
        h = h + mlp(postAttentionLayernorm(h))
        return h
    }
}

final class AuKAudioAttention: Module {
    let nHeads: Int
    let dimHead: Int
    let scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    override init() {
        nHeads = AuKFlashConfig.audioHeads
        dimHead = AuKFlashConfig.audioDModel / nHeads
        scale = pow(Float(dimHead), -0.5)
        _qProj.wrappedValue = Linear(AuKFlashConfig.audioDModel, AuKFlashConfig.audioDModel, bias: true)
        _kProj.wrappedValue = Linear(AuKFlashConfig.audioDModel, AuKFlashConfig.audioDModel, bias: false)
        _vProj.wrappedValue = Linear(AuKFlashConfig.audioDModel, AuKFlashConfig.audioDModel, bias: true)
        _outProj.wrappedValue = Linear(AuKFlashConfig.audioDModel, AuKFlashConfig.audioDModel, bias: true)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let batch = x.dim(0)
        let tokens = x.dim(1)
        let shape = [batch, tokens, nHeads, dimHead]
        let q = qProj(x).reshaped(shape).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped(shape).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(shape).transposed(0, 2, 1, 3)
        let output = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: .array(mask)
        )
        return outProj(output.transposed(0, 2, 1, 3).reshaped(batch, tokens, -1))
    }
}

final class AuKAudioLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: AuKAudioAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnLayerNorm: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: LayerNorm

    override init() {
        _selfAttn.wrappedValue = AuKAudioAttention()
        _selfAttnLayerNorm.wrappedValue = LayerNorm(dimensions: AuKFlashConfig.audioDModel)
        _fc1.wrappedValue = Linear(AuKFlashConfig.audioDModel, AuKFlashConfig.audioFFN)
        _fc2.wrappedValue = Linear(AuKFlashConfig.audioFFN, AuKFlashConfig.audioDModel)
        _finalLayerNorm.wrappedValue = LayerNorm(dimensions: AuKFlashConfig.audioDModel)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + selfAttn(selfAttnLayerNorm(x), mask: mask)
        let inner = finalLayerNorm(h)
        return h + fc2(gelu(fc1(inner)))
    }
}

final class AuKAudioTower: Module {
    let nWindow: Int
    @ModuleInfo var conv1: Conv1d
    @ModuleInfo var conv2: Conv1d
    @ModuleInfo var layers: [AuKAudioLayer]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo var proj: Linear
    private var positional: MLXArray?

    override init() {
        nWindow = AuKFlashConfig.audioWindow
        _conv1.wrappedValue = Conv1d(
            inputChannels: AuKFlashConfig.audioMelBins,
            outputChannels: AuKFlashConfig.audioDModel,
            kernelSize: 3,
            padding: 1
        )
        _conv2.wrappedValue = Conv1d(
            inputChannels: AuKFlashConfig.audioDModel,
            outputChannels: AuKFlashConfig.audioDModel,
            kernelSize: 3,
            stride: 2,
            padding: 1
        )
        _layers.wrappedValue = (0 ..< AuKFlashConfig.audioLayers).map { _ in AuKAudioLayer() }
        _lnPost.wrappedValue = LayerNorm(dimensions: AuKFlashConfig.audioDModel)
        _proj.wrappedValue = Linear(AuKFlashConfig.audioDModel, AuKFlashConfig.audioOutputDim)
    }

    private func positionalEncoding(_ n: Int) -> MLXArray {
        if let positional, positional.dim(0) >= n {
            return positional[0 ..< n]
        }
        let length = max(n, nWindow)
        let channels = AuKFlashConfig.audioDModel
        let logTS = log(10_000.0) / Double(channels / 2 - 1)
        var values = [Float](repeating: 0, count: length * channels)
        for pos in 0 ..< length {
            for j in 0 ..< (channels / 2) {
                let scaled = Double(pos) * exp(-logTS * Double(j))
                values[pos * channels + j] = Float(Foundation.sin(scaled))
                values[pos * channels + channels / 2 + j] = Float(Foundation.cos(scaled))
            }
        }
        let table = MLXArray(values).reshaped([length, channels])
        positional = table
        eval(table)
        return table[0 ..< n]
    }

    func callAsFunction(_ mel: MLXArray, featureLen: Int?) -> MLXArray {
        let total = featureLen ?? mel.dim(1)
        let clipped = mel[0..., 0 ..< total, 0...]
        let win = nWindow * 2
        var lengths: [Int] = []
        if total == 0 {
            lengths = [0]
        } else {
            lengths = Array(repeating: win, count: total / win)
            let rest = total % win
            if rest > 0 {
                lengths.append(rest)
            } else if lengths.isEmpty {
                lengths = [total]
            }
        }

        var embeds: [MLXArray] = []
        var keep: [Int] = []
        var offset = 0
        for length in lengths {
            var chunk = clipped[0..., offset ..< (offset + length), 0...]
            if length < win {
                chunk = padded(chunk, widths: [IntOrPair(0), IntOrPair((0, win - length)), IntOrPair(0)])
            }
            var e = gelu(conv1(chunk))
            if length < win {
                var mask = [Float](repeating: 0, count: win)
                for i in 0 ..< length { mask[i] = 1 }
                e = e * MLXArray(mask).reshaped([1, win, 1])
            }
            e = gelu(conv2(e))
            e = e + positionalEncoding(e.dim(1)).expandedDimensions(axis: 0)
            embeds.append(e)
            keep.append((length + 1) / 2)
            offset += length
        }

        let cropped = zip(embeds, keep).map { embed, k in embed[0..., 0 ..< k, 0...] }
        var h = concatenated(cropped, axis: 1)
        let n = h.dim(1)
        var block = [Float](repeating: -.infinity, count: n * n)
        var pos = 0
        for k in keep {
            for i in 0 ..< k {
                for j in 0 ..< k {
                    block[(pos + i) * n + (pos + j)] = 0
                }
            }
            pos += k
        }
        let mask = MLXArray(block).reshaped([1, 1, n, n])
        for layer in layers {
            h = layer(h, mask: mask)
        }

        h = h[0]
        let n2 = (h.dim(0) / 2) * 2
        if n2 == 0 {
            return proj(lnPost(MLXArray.zeros([0, AuKFlashConfig.audioDModel])))
        }
        h = h[0 ..< n2].reshaped([n2 / 2, 2, -1]).mean(axis: 1)
        return proj(lnPost(h))
    }
}

public final class AuKThinkerEncoder: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo var layers: [AuKTextLayer]
    @ModuleInfo var norm: AuKThinkerRMSNorm
    @ModuleInfo(key: "audio_tower") var audioTower: AuKAudioTower
    private var ropeCacheN = 0
    private var ropeCos: MLXArray?
    private var ropeSin: MLXArray?

    public override init() {
        _embedTokens.wrappedValue = Embedding(
            embeddingCount: AuKFlashConfig.thinkerVocab,
            dimensions: AuKFlashConfig.thinkerHidden
        )
        _layers.wrappedValue = (0 ..< AuKFlashConfig.thinkerLayers).map { _ in AuKTextLayer() }
        _norm.wrappedValue = AuKThinkerRMSNorm(
            dimensions: AuKFlashConfig.thinkerHidden,
            eps: AuKFlashConfig.thinkerRMSNormEps
        )
        _audioTower.wrappedValue = AuKAudioTower()
    }

    private func rope(n: Int) -> (MLXArray, MLXArray) {
        if let ropeCos, let ropeSin, ropeCacheN >= n {
            return (ropeCos[0..., 0..., 0 ..< n, 0...], ropeSin[0..., 0..., 0 ..< n, 0...])
        }
        let dimHead = AuKFlashConfig.thinkerHidden / AuKFlashConfig.thinkerHeads
        let inv = analyticRoPEInvFreq(dimHead: dimHead, base: Double(AuKFlashConfig.thinkerRopeTheta))
        var cosVals = [Float](repeating: 0, count: n * dimHead)
        var sinVals = [Float](repeating: 0, count: n * dimHead)
        for pos in 0 ..< n {
            for j in 0 ..< inv.count {
                let ang = Double(pos) * inv[j]
                let c = Float(Foundation.cos(ang))
                let s = Float(Foundation.sin(ang))
                cosVals[pos * dimHead + j] = c
                cosVals[pos * dimHead + inv.count + j] = c
                sinVals[pos * dimHead + j] = s
                sinVals[pos * dimHead + inv.count + j] = s
            }
        }
        let cos = MLXArray(cosVals).reshaped([1, 1, n, dimHead])
        let sin = MLXArray(sinVals).reshaped([1, 1, n, dimHead])
        ropeCos = cos
        ropeSin = sin
        ropeCacheN = n
        eval(cos, sin)
        return (cos, sin)
    }

    public func callAsFunction(
        inputIds: MLXArray,
        audioFeatures: MLXArray? = nil,
        audioTokenMask: MLXArray? = nil,
        audioFeatureLen: Int? = nil
    ) throws -> [MLXArray] {
        var h = embedTokens(inputIds)
        if let audioFeatures {
            let audioEmb = audioTower(audioFeatures, featureLen: audioFeatureLen)
            h = try scatterAudio(h: h, audioEmb: audioEmb, mask: audioTokenMask)
        }
        let n = h.dim(1)
        let (cos, sin) = rope(n: n)
        var hidden: [MLXArray] = [h]
        for layer in layers {
            h = layer(h, cos: cos, sin: sin)
            hidden.append(h)
        }
        if let last = hidden.indices.last {
            hidden[last] = norm(hidden[last])
        }
        return hidden
    }
}

private func scatterAudio(h: MLXArray, audioEmb: MLXArray, mask: MLXArray?) throws -> MLXArray {
    guard let mask else {
        throw AuKError.invalidInput("audio_token_mask is required when passing audio features")
    }
    let seqLen = h.dim(1)
    let hiddenDim = h.dim(2)
    let flatMask = mask.reshaped([seqLen]).asType(.int32)
    eval(flatMask)
    let flags = flatMask.asArray(Int32.self)
    let count = flags.reduce(0) { $0 + Int($1) }
    if count != audioEmb.dim(0) {
        throw AuKError.invalidInput(
            "audio placeholder count (\(count)) != audio tower output length (\(audioEmb.dim(0)))"
        )
    }
    let audioCumsum = cumsum(flatMask, axis: 0) - 1
    let audioIndices = clip(audioCumsum, min: 0, max: max(0, audioEmb.dim(0) - 1))
    let flatEmbeds = h.reshaped([seqLen, hiddenDim])
    let gathered = audioEmb[audioIndices]
    let expanded = flatMask.expandedDimensions(axis: -1).asType(gathered.dtype)
    let mixed = expanded * gathered + (1 - expanded) * flatEmbeds
    return mixed.reshaped([1, seqLen, hiddenDim])
}
