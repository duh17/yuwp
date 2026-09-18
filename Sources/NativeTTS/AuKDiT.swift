import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

final class AuKRopeTable {
    let invFreq: [Double]
    let dimHead: Int
    private var cos: MLXArray?
    private var sin: MLXArray?
    private var cachedN = 0

    init(invFreq: [Float]) {
        self.invFreq = invFreq.map(Double.init)
        self.dimHead = invFreq.count * 2
    }

    convenience init(analyticDimHead: Int, base: Double = 10_000) {
        self.init(invFreq: analyticRoPEInvFreq(dimHead: analyticDimHead, base: base).map(Float.init))
    }

    func apply(_ x: MLXArray) -> MLXArray {
        let n = x.dim(-2)
        grow(to: n)
        guard let cos, let sin else { return x }
        return applyInterleavedRoPE(x, cos: cos[0 ..< n], sin: sin[0 ..< n])
    }

    private func grow(to n: Int) {
        if cos != nil, n <= cachedN { return }
        var cosVals = [Float](repeating: 0, count: n * dimHead)
        var sinVals = [Float](repeating: 0, count: n * dimHead)
        for pos in 0 ..< n {
            for j in 0 ..< invFreq.count {
                let ang = Double(pos) * invFreq[j]
                let c = Float(Foundation.cos(ang))
                let s = Float(Foundation.sin(ang))
                let even = pos * dimHead + j * 2
                cosVals[even] = c
                cosVals[even + 1] = c
                sinVals[even] = s
                sinVals[even + 1] = s
            }
        }
        let cosArr = MLXArray(cosVals).reshaped([n, dimHead])
        let sinArr = MLXArray(sinVals).reshaped([n, dimHead])
        cos = cosArr
        sin = sinArr
        cachedN = n
        eval(cosArr, sinArr)
    }
}

final class AuKDiTRMSNorm: Module {
    let eps: Float
    let weight: MLXArray

    init(dimensions: Int, eps: Float = AuKFlashConfig.torchFloat32Eps) {
        self.eps = eps
        self.weight = MLXArray.ones([dimensions])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: weight, eps: eps)
    }
}

func aukLayerNorm(_ x: MLXArray, eps: Float = 1e-6) -> MLXArray {
    MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps)
}

final class AuKAdaLayerNorm: Module {
    @ModuleInfo var linear: Linear

    init(dim: Int) {
        _linear.wrappedValue = Linear(dim, dim * 6)
    }

    func callAsFunction(_ x: MLXArray, emb: MLXArray) -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray) {
        let e = linear(aukSilu(emb))
        let parts = e.split(parts: 6, axis: 1)
        let shiftMSA = parts[0]
        let scaleMSA = parts[1]
        let gateMSA = parts[2]
        let shiftMLP = parts[3]
        let scaleMLP = parts[4]
        let gateMLP = parts[5]
        let normed = aukLayerNorm(x) * (1 + scaleMSA.expandedDimensions(axis: 1)) + shiftMSA.expandedDimensions(axis: 1)
        return (normed, gateMSA, shiftMLP, scaleMLP, gateMLP)
    }
}

final class AuKAdaLayerNormFinal: Module {
    @ModuleInfo var linear: Linear

    init(dim: Int) {
        _linear.wrappedValue = Linear(dim, dim * 2)
    }

    func callAsFunction(_ x: MLXArray, emb: MLXArray) -> MLXArray {
        let e = linear(aukSilu(emb))
        let parts = e.split(parts: 2, axis: 1)
        let scale = parts[0]
        let shift = parts[1]
        return aukLayerNorm(x) * (1 + scale).expandedDimensions(axis: 1) + shift.expandedDimensions(axis: 1)
    }
}

final class AuKSwiGLUFeedForward: Module {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear

    init(dim: Int, mult: Float = 2.0) {
        let inner = Int(Float(dim) * mult)
        _linearIn.wrappedValue = Linear(dim, inner * 2, bias: false)
        _linearOut.wrappedValue = Linear(inner, dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = linearIn(x).split(parts: 2, axis: -1)
        return linearOut(aukSilu(parts[0]) * parts[1])
    }
}

private func aukHeads(_ x: MLXArray, heads: Int) -> MLXArray {
    let batch = x.dim(0)
    let tokens = x.dim(1)
    return x.reshaped(batch, tokens, heads, -1).transposed(0, 2, 1, 3)
}

private func aukUnheads(_ x: MLXArray) -> MLXArray {
    let batch = x.dim(0)
    let heads = x.dim(1)
    let tokens = x.dim(2)
    let dim = x.dim(3)
    return x.transposed(0, 2, 1, 3).reshaped(batch, tokens, heads * dim)
}

final class AuKAttention: Module {
    let heads: Int
    let scale: Float
    let hasContext: Bool
    @ModuleInfo(key: "to_qkv") var toQKV: Linear
    @ModuleInfo(key: "q_norm") var qNorm: AuKDiTRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: AuKDiTRMSNorm
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "to_qkv_c") var toQKVC: Linear?
    @ModuleInfo(key: "c_q_norm") var cQNorm: AuKDiTRMSNorm?
    @ModuleInfo(key: "c_k_norm") var cKNorm: AuKDiTRMSNorm?
    @ModuleInfo(key: "to_out_c") var toOutC: Linear?

    init(dim: Int, heads: Int, dimHead: Int, contextDim: Int? = nil) {
        self.heads = heads
        self.scale = pow(Float(dimHead), -0.5)
        let inner = heads * dimHead
        _toQKV.wrappedValue = Linear(dim, 3 * inner)
        _qNorm.wrappedValue = AuKDiTRMSNorm(dimensions: dimHead)
        _kNorm.wrappedValue = AuKDiTRMSNorm(dimensions: dimHead)
        _toOut.wrappedValue = [Linear(inner, dim)]
        if let contextDim {
            self.hasContext = true
            _toQKVC.wrappedValue = Linear(contextDim, 3 * inner)
            _cQNorm.wrappedValue = AuKDiTRMSNorm(dimensions: dimHead)
            _cKNorm.wrappedValue = AuKDiTRMSNorm(dimensions: dimHead)
            _toOutC.wrappedValue = Linear(inner, contextDim)
        } else {
            self.hasContext = false
            _toQKVC.wrappedValue = nil
            _cQNorm.wrappedValue = nil
            _cKNorm.wrappedValue = nil
            _toOutC.wrappedValue = nil
        }
    }

    private func qkv(_ x: MLXArray, proj: Linear, qn: AuKDiTRMSNorm, kn: AuKDiTRMSNorm) -> (MLXArray, MLXArray, MLXArray) {
        let parts = proj(x).split(parts: 3, axis: -1)
        return (qn(aukHeads(parts[0], heads: heads)), kn(aukHeads(parts[1], heads: heads)), aukHeads(parts[2], heads: heads))
    }

    func single(_ x: MLXArray, rope: AuKRopeTable) -> MLXArray {
        var qkvVals = qkv(x, proj: toQKV, qn: qNorm, kn: kNorm)
        qkvVals.0 = rope.apply(qkvVals.0)
        qkvVals.1 = rope.apply(qkvVals.1)
        let output = MLXFast.scaledDotProductAttention(
            queries: qkvVals.0,
            keys: qkvVals.1,
            values: qkvVals.2,
            scale: scale,
            mask: .none
        )
        return toOut[0](aukUnheads(output))
    }

    func joint(_ x: MLXArray, context: MLXArray, rope: AuKRopeTable) -> (MLXArray, MLXArray) {
        guard let toQKVC, let cQNorm, let cKNorm, let toOutC else {
            return (single(x, rope: rope), context)
        }
        var audio = qkv(x, proj: toQKV, qn: qNorm, kn: kNorm)
        var text = qkv(context, proj: toQKVC, qn: cQNorm, kn: cKNorm)
        audio.0 = rope.apply(audio.0)
        audio.1 = rope.apply(audio.1)
        text.0 = rope.apply(text.0)
        text.1 = rope.apply(text.1)
        let q = concatenated([audio.0, text.0], axis: 2)
        let k = concatenated([audio.1, text.1], axis: 2)
        let v = concatenated([audio.2, text.2], axis: 2)
        let output = aukUnheads(
            MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        )
        let audioLen = x.dim(1)
        return (toOut[0](output[0..., 0 ..< audioLen, 0...]), toOutC(output[0..., audioLen..., 0...]))
    }
}

final class AuKDiTBlock: Module {
    @ModuleInfo(key: "attn_norm") var attnNorm: AuKAdaLayerNorm
    @ModuleInfo var attn: AuKAttention
    @ModuleInfo var ff: AuKSwiGLUFeedForward

    init(dim: Int, heads: Int, dimHead: Int, ffMult: Float) {
        _attnNorm.wrappedValue = AuKAdaLayerNorm(dim: dim)
        _attn.wrappedValue = AuKAttention(dim: dim, heads: heads, dimHead: dimHead)
        _ff.wrappedValue = AuKSwiGLUFeedForward(dim: dim, mult: ffMult)
    }

    func callAsFunction(_ x: MLXArray, t: MLXArray, rope: AuKRopeTable) -> MLXArray {
        let (norm, gateMSA, shiftMLP, scaleMLP, gateMLP) = attnNorm(x, emb: t)
        let h = x + gateMSA.expandedDimensions(axis: 1) * attn.single(norm, rope: rope)
        let ffNorm = aukLayerNorm(h) * (1 + scaleMLP.expandedDimensions(axis: 1)) + shiftMLP.expandedDimensions(axis: 1)
        return h + gateMLP.expandedDimensions(axis: 1) * ff(ffNorm)
    }
}

final class AuKMMDiTBlock: Module {
    @ModuleInfo(key: "attn_norm_c") var attnNormC: AuKAdaLayerNorm
    @ModuleInfo(key: "attn_norm_x") var attnNormX: AuKAdaLayerNorm
    @ModuleInfo var attn: AuKAttention
    @ModuleInfo(key: "ff_c") var ffC: AuKSwiGLUFeedForward
    @ModuleInfo(key: "ff_x") var ffX: AuKSwiGLUFeedForward

    init(dim: Int, heads: Int, dimHead: Int, ffMult: Float) {
        _attnNormC.wrappedValue = AuKAdaLayerNorm(dim: dim)
        _attnNormX.wrappedValue = AuKAdaLayerNorm(dim: dim)
        _attn.wrappedValue = AuKAttention(dim: dim, heads: heads, dimHead: dimHead, contextDim: dim)
        _ffC.wrappedValue = AuKSwiGLUFeedForward(dim: dim, mult: ffMult)
        _ffX.wrappedValue = AuKSwiGLUFeedForward(dim: dim, mult: ffMult)
    }

    func callAsFunction(_ x: MLXArray, c: MLXArray, t: MLXArray, rope: AuKRopeTable) -> (MLXArray, MLXArray) {
        let cNorm = attnNormC(c, emb: t)
        let xNorm = attnNormX(x, emb: t)
        let (xAttn, cAttn) = attn.joint(xNorm.0, context: cNorm.0, rope: rope)

        var text = c + cNorm.1.expandedDimensions(axis: 1) * cAttn
        let textFF = aukLayerNorm(text) * (1 + cNorm.3.expandedDimensions(axis: 1)) + cNorm.2.expandedDimensions(axis: 1)
        text = text + cNorm.4.expandedDimensions(axis: 1) * ffC(textFF)

        var audio = x + xNorm.1.expandedDimensions(axis: 1) * xAttn
        let audioFF = aukLayerNorm(audio) * (1 + xNorm.3.expandedDimensions(axis: 1)) + xNorm.2.expandedDimensions(axis: 1)
        audio = audio + xNorm.4.expandedDimensions(axis: 1) * ffX(audioFF)
        return (text, audio)
    }
}

final class AuKConvPositionEmbedding: Module {
    @ModuleInfo var conv1d: [AuKConv1d]

    init(dim: Int, kernelSize: Int = 31, groups: Int = 16) {
        _conv1d.wrappedValue = [
            AuKConv1d(inChannels: dim, outChannels: dim, kernelSize: kernelSize, groups: groups, padding: kernelSize / 2),
            AuKConv1d(inChannels: dim, outChannels: dim, kernelSize: kernelSize, groups: groups, padding: kernelSize / 2),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1d[0](x)
        h = aukMish(h)
        h = conv1d[1](h)
        return aukMish(h)
    }
}

final class AuKAudioPromptEmbedding: Module {
    @ModuleInfo var linear: Linear
    @ModuleInfo(key: "conv_pos_embed") var convPosEmbed: AuKConvPositionEmbedding

    init(inDim: Int, outDim: Int) {
        _linear.wrappedValue = Linear(inDim, outDim)
        _convPosEmbed.wrappedValue = AuKConvPositionEmbedding(dim: outDim)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = linear(x)
        return convPosEmbed(h) + h
    }
}

final class AuKTimestepEmbedding: Module {
    let freqEmbedDim: Int
    @ModuleInfo(key: "time_mlp") var timeMLP: [Linear]

    init(dim: Int, freqEmbedDim: Int = 256) {
        self.freqEmbedDim = freqEmbedDim
        _timeMLP.wrappedValue = [Linear(freqEmbedDim, dim), Linear(dim, dim)]
    }

    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let half = freqEmbedDim / 2
        let scale = log(Float(10_000)) / Float(half - 1)
        let freqs = exp(MLXArray(0 ..< half).asType(.float32) * -scale)
        let args = Float(1000.0) * t.expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        let embedding = concatenated([sin(args), cos(args)], axis: -1)
        return timeMLP[1](aukSilu(timeMLP[0](embedding)))
    }
}

public final class AuKFlux2Edit: Module {
    let cfg: (dim: Int, heads: Int, dimHead: Int)
    let rope: AuKRopeTable
    @ModuleInfo(key: "time_embed") var timeEmbed: AuKTimestepEmbedding
    @ModuleInfo(key: "txt_norm") var txtNorm: AuKDiTRMSNorm
    @ModuleInfo(key: "txt_proj") var txtProj: Linear
    @ModuleInfo(key: "audio_embed") var audioEmbed: AuKAudioPromptEmbedding
    @ModuleInfo(key: "transformer_blocks") var transformerBlocks: [AuKMMDiTBlock]
    @ModuleInfo(key: "single_transformer_blocks") var singleTransformerBlocks: [AuKDiTBlock]
    @ModuleInfo(key: "norm_out") var normOut: AuKAdaLayerNormFinal
    @ModuleInfo(key: "proj_out") var projOut: Linear
    private var cachedTextCond: MLXArray?
    private var cachedTextUncond: MLXArray?

    public init(invFreq: [Float]?) {
        let dim = AuKFlashConfig.ditDim
        let heads = AuKFlashConfig.ditHeads
        let dimHead = AuKFlashConfig.ditDimHead
        self.cfg = (dim, heads, dimHead)
        self.rope = invFreq.map { AuKRopeTable(invFreq: $0) } ?? AuKRopeTable(analyticDimHead: dimHead)
        _timeEmbed.wrappedValue = AuKTimestepEmbedding(dim: dim)
        _txtNorm.wrappedValue = AuKDiTRMSNorm(dimensions: dim)
        _txtProj.wrappedValue = Linear(AuKFlashConfig.textHiddenDim, dim)
        _audioEmbed.wrappedValue = AuKAudioPromptEmbedding(inDim: AuKFlashConfig.latentDim, outDim: dim)
        _transformerBlocks.wrappedValue = (0 ..< AuKFlashConfig.ditLayers).map { _ in
            AuKMMDiTBlock(dim: dim, heads: heads, dimHead: dimHead, ffMult: AuKFlashConfig.ditFFMult)
        }
        _singleTransformerBlocks.wrappedValue = (0 ..< AuKFlashConfig.ditSingleLayers).map { _ in
            AuKDiTBlock(dim: dim, heads: heads, dimHead: dimHead, ffMult: AuKFlashConfig.ditFFMult)
        }
        _normOut.wrappedValue = AuKAdaLayerNormFinal(dim: dim)
        _projOut.wrappedValue = Linear(dim, AuKFlashConfig.latentDim)
    }

    public func clearCache() {
        cachedTextCond = nil
        cachedTextUncond = nil
    }

    func projectText(_ text: MLXArray, dropText: Bool) -> MLXArray {
        let c = txtNorm(txtProj(text))
        return dropText ? MLXArray.zeros(c.shape) : c
    }

    public func callAsFunction(
        x: MLXArray,
        text: MLXArray,
        t: MLXArray,
        ref: MLXArray?,
        cfgInfer: Bool = false,
        cache: Bool = true
    ) -> MLXArray {
        var temb = timeEmbed(t)
        let hasRef = if let ref { ref.dim(1) > 0 } else { false }

        func embed(dropAudioCond: Bool) -> (MLXArray, Int) {
            let xEmb = audioEmbed(x)
            guard hasRef, let ref else { return (xEmb, 0) }
            let r = dropAudioCond ? MLXArray.zeros(ref.shape) : ref
            let refEmb = audioEmbed(r)
            return (concatenated([refEmb, xEmb], axis: 1), refEmb.dim(1))
        }

        let h: MLXArray
        let c: MLXArray
        let promptLen: Int
        if cfgInfer {
            let cCond: MLXArray
            let cUncond: MLXArray
            if cache, let cachedTextCond, let cachedTextUncond {
                cCond = cachedTextCond
                cUncond = cachedTextUncond
            } else {
                cCond = projectText(text, dropText: false)
                cUncond = projectText(text, dropText: true)
                if cache {
                    cachedTextCond = cCond
                    cachedTextUncond = cUncond
                }
            }
            let cond = embed(dropAudioCond: false)
            let uncond = embed(dropAudioCond: true)
            h = concatenated([cond.0, uncond.0], axis: 0)
            c = concatenated([cCond, cUncond], axis: 0)
            temb = concatenated([temb, temb], axis: 0)
            promptLen = cond.1
        } else {
            c = projectText(text, dropText: false)
            let embedded = embed(dropAudioCond: false)
            h = embedded.0
            promptLen = embedded.1
        }

        var audio = h
        var textStates = c
        for block in transformerBlocks {
            let out = block(audio, c: textStates, t: temb, rope: rope)
            textStates = out.0
            audio = out.1
        }

        let textLen = textStates.dim(1)
        var combined = concatenated([textStates, audio], axis: 1)
        for block in singleTransformerBlocks {
            combined = block(combined, t: temb, rope: rope)
        }
        let start = textLen + promptLen
        let sliced = combined[0..., start..., 0...]
        return projOut(normOut(sliced, emb: temb))
    }
}
