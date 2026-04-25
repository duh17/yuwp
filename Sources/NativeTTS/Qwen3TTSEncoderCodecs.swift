import Foundation
@preconcurrency import MLX
@preconcurrency import MLXLMCommon
import MLXNN

// Minimal Mimi/Seanet encoder pieces from mlx-audio-swift, adapted into NativeTTS.
// These are used only for Qwen3-TTS reference-audio voice cloning.

// MARK: - NCL Conv wrappers

final class TTSCodecConv1d: Module {
    var weight: MLXArray
    var bias: MLXArray?

    let padding: Int
    let groups: Int
    let stride: Int
    let dilation: Int

    init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        let scale = 1.0 / Float(inChannels * ksize)
        self.weight = MLXRandom.uniform(low: -scale, high: scale, [outChannels, ksize, inChannels / groups])
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
        self.padding = padding
        self.groups = groups
        self.stride = stride
        self.dilation = dilation
    }

    func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        let xsNLC = swappedAxes(xsNCL, 1, 2)
        var y = conv1d(
            xsNLC,
            weight,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups
        )
        if let bias { y = y + bias }
        return swappedAxes(y, 1, 2)
    }
}

final class TTSCodecNormConv1d: Module {
    @ModuleInfo var conv: TTSCodecConv1d

    init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int = 1,
        padding: Int = 0,
        groups: Int = 1,
        dilation: Int = 1,
        bias: Bool = true
    ) {
        _conv.wrappedValue = TTSCodecConv1d(
            inChannels: inChannels,
            outChannels: outChannels,
            ksize: ksize,
            stride: stride,
            padding: padding,
            groups: groups,
            dilation: dilation,
            bias: bias
        )
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray { conv(xs) }
}

@inline(__always)
private func qwen3ExtraPaddingForConv1d(xs: MLXArray, ksize: Int, stride: Int, paddingTotal: Int) -> Int {
    let length = xs.shape[2]
    let nFrames = max(length + paddingTotal - ksize, 0)
    let nf = Double(nFrames) / Double(stride) + 1.0
    let idealLength = (Int(ceil(nf)) - 1) * stride + ksize - paddingTotal
    return max(0, idealLength - length)
}

final class TTSCodecStreamableConv1d: Module {
    private let causal: Bool
    private let padMode: MLX.PadMode
    private let ksizeBase: Int
    @ModuleInfo var conv: TTSCodecNormConv1d

    init(
        inChannels: Int,
        outChannels: Int,
        ksize: Int,
        stride: Int,
        dilation: Int,
        groups: Int,
        bias: Bool,
        causal: Bool,
        padMode: MLX.PadMode
    ) {
        self.causal = causal
        self.padMode = padMode
        self.ksizeBase = ksize
        _conv.wrappedValue = TTSCodecNormConv1d(
            inChannels: inChannels,
            outChannels: outChannels,
            ksize: ksize,
            stride: stride,
            groups: groups,
            dilation: dilation,
            bias: bias
        )
    }

    func resetState() {}

    func callAsFunction(_ xsNCL: MLXArray) -> MLXArray {
        let dilation = conv.conv.dilation
        let effectiveKernelSize = (ksizeBase - 1) * dilation + 1
        let paddingTotal = effectiveKernelSize - conv.conv.stride
        let extra = qwen3ExtraPaddingForConv1d(
            xs: xsNCL,
            ksize: effectiveKernelSize,
            stride: conv.conv.stride,
            paddingTotal: paddingTotal
        )
        let pad: (Int, Int) = if causal {
            (paddingTotal, 0)
        } else {
            (paddingTotal - paddingTotal / 2, paddingTotal / 2)
        }
        let paddedInput = padded(
            xsNCL,
            widths: [.init(0), .init(0), .init((pad.0, pad.1 + extra))],
            mode: padMode
        )
        return conv(paddedInput)
    }
}

final class TTSCodecConvDownsample1d: Module {
    @ModuleInfo var conv: TTSCodecStreamableConv1d

    init(stride: Int, dim: Int, causal: Bool) {
        _conv.wrappedValue = TTSCodecStreamableConv1d(
            inChannels: dim,
            outChannels: dim,
            ksize: 2 * stride,
            stride: stride,
            dilation: 1,
            groups: 1,
            bias: false,
            causal: causal,
            padMode: .edge
        )
    }

    func resetState() { conv.resetState() }
    func callAsFunction(_ xs: MLXArray) -> MLXArray { conv(xs) }
}

// MARK: - Seanet encoder

struct TTSCodecSeanetConfig {
    let dimension: Int
    let channels: Int
    let causal: Bool
    let nfilters: Int
    let nresidualLayers: Int
    let ratios: [Int]
    let ksize: Int
    let residualKsize: Int
    let lastKsize: Int
    let dilationBase: Int
    let padMode: MLX.PadMode
    let trueSkip: Bool
    let compress: Int
}

final class TTSCodecSeanetResnetBlock: Module {
    @ModuleInfo var block: [TTSCodecStreamableConv1d]
    @ModuleInfo var shortcut: TTSCodecStreamableConv1d?

    init(cfg: TTSCodecSeanetConfig, dim: Int, ksizesAndDilations: [(Int, Int)]) {
        let hidden = dim / cfg.compress
        _block.wrappedValue = ksizesAndDilations.enumerated().map { index, item in
            let (ksize, dilation) = item
            let inChannels = index == 0 ? dim : hidden
            let outChannels = index == ksizesAndDilations.count - 1 ? dim : hidden
            return TTSCodecStreamableConv1d(
                inChannels: inChannels,
                outChannels: outChannels,
                ksize: ksize,
                stride: 1,
                dilation: dilation,
                groups: 1,
                bias: true,
                causal: cfg.causal,
                padMode: cfg.padMode
            )
        }
        _shortcut.wrappedValue = cfg.trueSkip
            ? nil
            : TTSCodecStreamableConv1d(
                inChannels: dim,
                outChannels: dim,
                ksize: 1,
                stride: 1,
                dilation: 1,
                groups: 1,
                bias: true,
                causal: cfg.causal,
                padMode: cfg.padMode
            )
    }

    func resetState() {
        shortcut?.resetState()
        for layer in block { layer.resetState() }
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var x = xs
        for layer in block {
            x = layer(elu(x, alpha: 1.0))
        }
        if let shortcut {
            return x + shortcut(xs)
        }
        return x + xs
    }
}

final class TTSCodecEncoderLayer: Module {
    @ModuleInfo var residuals: [TTSCodecSeanetResnetBlock]
    @ModuleInfo var downsample: TTSCodecStreamableConv1d

    init(cfg: TTSCodecSeanetConfig, ratio: Int, mult: Int) {
        var dilation = 1
        var residualLayers = [TTSCodecSeanetResnetBlock]()
        for _ in 0 ..< cfg.nresidualLayers {
            residualLayers.append(TTSCodecSeanetResnetBlock(
                cfg: cfg,
                dim: mult * cfg.nfilters,
                ksizesAndDilations: [(cfg.residualKsize, dilation), (1, 1)]
            ))
            dilation *= cfg.dilationBase
        }
        _residuals.wrappedValue = residualLayers
        _downsample.wrappedValue = TTSCodecStreamableConv1d(
            inChannels: mult * cfg.nfilters,
            outChannels: mult * cfg.nfilters * 2,
            ksize: ratio * 2,
            stride: ratio,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: true,
            padMode: cfg.padMode
        )
    }

    func resetState() {
        downsample.resetState()
        for residual in residuals { residual.resetState() }
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var x = xs
        for residual in residuals { x = residual(x) }
        return downsample(elu(x, alpha: 1.0))
    }
}

final class TTSCodecSeanetEncoder: Module {
    @ModuleInfo var init_conv1d: TTSCodecStreamableConv1d
    @ModuleInfo var layers: [TTSCodecEncoderLayer]
    @ModuleInfo var final_conv1d: TTSCodecStreamableConv1d

    init(cfg: TTSCodecSeanetConfig) {
        var mult = 1
        _init_conv1d.wrappedValue = TTSCodecStreamableConv1d(
            inChannels: cfg.channels,
            outChannels: mult * cfg.nfilters,
            ksize: cfg.ksize,
            stride: 1,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: cfg.causal,
            padMode: cfg.padMode
        )

        var encoderLayers = [TTSCodecEncoderLayer]()
        for ratio in cfg.ratios.reversed() {
            encoderLayers.append(TTSCodecEncoderLayer(cfg: cfg, ratio: ratio, mult: mult))
            mult *= 2
        }
        _layers.wrappedValue = encoderLayers

        _final_conv1d.wrappedValue = TTSCodecStreamableConv1d(
            inChannels: mult * cfg.nfilters,
            outChannels: cfg.dimension,
            ksize: cfg.lastKsize,
            stride: 1,
            dilation: 1,
            groups: 1,
            bias: true,
            causal: cfg.causal,
            padMode: cfg.padMode
        )
    }

    func resetState() {
        init_conv1d.resetState()
        final_conv1d.resetState()
        for layer in layers { layer.resetState() }
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        var x = init_conv1d(xs)
        for layer in layers { x = layer(x) }
        x = elu(x, alpha: 1.0)
        return final_conv1d(x)
    }
}

// MARK: - Encoder transformer

struct TTSCodecTransformerConfig {
    let dModel: Int
    let numHeads: Int
    let numLayers: Int
    let causal: Bool
    let normFirst: Bool
    let biasFF: Bool
    let biasAttn: Bool
    let layerScale: Float?
    let positionalEmbedding: String
    let context: Int
    let maxPeriod: Int
    let kvRepeat: Int
    let dimFeedforward: Int
    let convLayout: Bool

    var headDim: Int { dModel / numHeads }
}

final class TTSCodecId: Module, UnaryLayer {
    func callAsFunction(_ xs: MLXArray) -> MLXArray { xs }
}

final class TTSCodecLayerScale: Module, UnaryLayer {
    var scale: MLXArray

    init(dim: Int) {
        self.scale = MLXArray.ones([dim])
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray { xs * scale }
}

final class TTSCodecAttention: Module {
    private let cfg: TTSCodecTransformerConfig
    private let scale: Float
    @ModuleInfo var in_proj: Linear
    @ModuleInfo var out_proj: Linear
    @ModuleInfo var rope: RoPE?

    init(cfg: TTSCodecTransformerConfig) {
        self.cfg = cfg
        let numKV = cfg.numHeads / cfg.kvRepeat
        let outDim = cfg.dModel + 2 * numKV * (cfg.dModel / cfg.numHeads)
        self.scale = 1.0 / Float(Double(cfg.headDim).squareRoot())
        _in_proj.wrappedValue = Linear(cfg.dModel, outDim, bias: cfg.biasAttn)
        _out_proj.wrappedValue = Linear(cfg.dModel, cfg.dModel, bias: cfg.biasAttn)
        _rope.wrappedValue = cfg.positionalEmbedding == "rope"
            ? RoPE(dimensions: cfg.headDim, traditional: true, base: Float(cfg.maxPeriod))
            : nil
    }

    func callAsFunction(_ xs: MLXArray, cache: any KVCache, mask: MLXArray? = nil) -> MLXArray {
        let batch = xs.shape[0]
        let time = xs.shape[1]
        let hidden = xs.shape[2]
        let qkv = in_proj(xs).reshaped([batch, time, 3, cfg.numHeads, cfg.headDim])

        var q = swappedAxes(qkv[0..<qkv.shape[0], 0..<qkv.shape[1], 0, 0..<qkv.shape[3], 0..<qkv.shape[4]], 1, 2)
        var k = swappedAxes(qkv[0..<qkv.shape[0], 0..<qkv.shape[1], 1, 0..<qkv.shape[3], 0..<qkv.shape[4]], 1, 2)
        var v = swappedAxes(qkv[0..<qkv.shape[0], 0..<qkv.shape[1], 2, 0..<qkv.shape[3], 0..<qkv.shape[4]], 1, 2)

        if let rope {
            q = rope(q, offset: cache.offset)
            k = rope(k, offset: cache.offset)
        }

        (k, v) = cache.update(keys: k, values: v)
        let kLen = k.shape[2]
        let targetLen = time + min(cfg.context, kLen - time)
        if targetLen < kLen {
            let start = kLen - targetLen
            k = split(k, indices: [start], axis: 2)[1]
            v = split(v, indices: [start], axis: 2)[1]
        }

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = if let mask {
            .array(mask)
        } else if cfg.causal {
            createAttentionMask(h: xs, cache: cache)
        } else {
            .none
        }

        var out = MLXFast.scaledDotProductAttention(
            queries: q,
            keys: k,
            values: v,
            scale: scale,
            mask: maskMode
        )
        out = swappedAxes(out, 1, 2).reshaped([batch, time, hidden])
        return out_proj(out)
    }
}

final class TTSCodecMlpNoGating: Module, UnaryLayer {
    @ModuleInfo var linear1: Linear
    @ModuleInfo var linear2: Linear

    init(cfg: TTSCodecTransformerConfig) {
        _linear1.wrappedValue = Linear(cfg.dModel, cfg.dimFeedforward, bias: cfg.biasFF)
        _linear2.wrappedValue = Linear(cfg.dimFeedforward, cfg.dModel, bias: cfg.biasFF)
    }

    func callAsFunction(_ xs: MLXArray) -> MLXArray {
        linear2(gelu(linear1(xs)))
    }
}

final class TTSCodecTransformerLayer: Module {
    @ModuleInfo var gating: TTSCodecMlpNoGating
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var layer_scale_1: TTSCodecLayerScale
    @ModuleInfo var layer_scale_2: TTSCodecLayerScale
    @ModuleInfo var self_attn: TTSCodecAttention

    init(cfg: TTSCodecTransformerConfig) {
        _gating.wrappedValue = TTSCodecMlpNoGating(cfg: cfg)
        _norm1.wrappedValue = LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
        _norm2.wrappedValue = LayerNorm(dimensions: cfg.dModel, eps: 1e-5)
        _layer_scale_1.wrappedValue = TTSCodecLayerScale(dim: cfg.dModel)
        _layer_scale_2.wrappedValue = TTSCodecLayerScale(dim: cfg.dModel)
        _self_attn.wrappedValue = TTSCodecAttention(cfg: cfg)
    }

    func callAsFunction(_ xs: MLXArray, cache: any KVCache) -> MLXArray {
        var x = xs
        var n1 = norm1(x)
        n1 = self_attn(n1, cache: cache)
        x = x + layer_scale_1(n1)
        x = x + layer_scale_2(gating(norm2(x)))
        return x
    }
}

final class TTSCodecTransformer: Module {
    @ModuleInfo var layers: [TTSCodecTransformerLayer]

    init(cfg: TTSCodecTransformerConfig) {
        _layers.wrappedValue = (0 ..< cfg.numLayers).map { _ in TTSCodecTransformerLayer(cfg: cfg) }
    }

    func callAsFunction(_ xs: MLXArray, cache: [any KVCache]) -> MLXArray {
        var x = xs
        for (layer, cacheEntry) in zip(layers, cache) {
            x = layer(x, cache: cacheEntry)
        }
        return x
    }

    func makeCache() -> [any KVCache] {
        (0 ..< layers.count).map { _ in KVCacheSimple() as any KVCache }
    }
}

final class TTSCodecProjectedTransformer: Module {
    private let convLayout: Bool
    @ModuleInfo var transformer: TTSCodecTransformer
    @ModuleInfo var input_proj: Linear?
    @ModuleInfo var output_projs: [Linear?]

    init(cfg: TTSCodecTransformerConfig, inputDim: Int, outputDims: [Int]) {
        self.convLayout = cfg.convLayout
        _transformer.wrappedValue = TTSCodecTransformer(cfg: cfg)
        _input_proj.wrappedValue = inputDim == cfg.dModel ? nil : Linear(inputDim, cfg.dModel, bias: false)
        _output_projs.wrappedValue = outputDims.map { dim in
            dim == cfg.dModel ? nil : Linear(cfg.dModel, dim, bias: false)
        }
    }

    func callAsFunction(_ xsIn: MLXArray, cache: [any KVCache]) -> [MLXArray] {
        var xs = xsIn
        if convLayout { xs = swappedAxes(xs, 1, 2) }
        if let input_proj { xs = input_proj(xs) }
        xs = transformer(xs, cache: cache)

        if output_projs.compactMap({ $0 }).isEmpty {
            return [convLayout ? swappedAxes(xs, 1, 2) : xs]
        }

        return output_projs.compactMap { projection in
            guard let projection else { return nil }
            var out = projection(xs)
            if convLayout { out = swappedAxes(out, 1, 2) }
            return out
        }
    }

    func makeCache() -> [any KVCache] { transformer.makeCache() }
}

// MARK: - Encoder quantizer

final class TTSCodecEuclideanCodebook: Module {
    private let epsilon: Float = 1e-5
    private let dim: Int

    var initialized: MLXArray
    var embedding_sum: MLXArray
    var cluster_usage: MLXArray
    private(set) var _embedding: MLXArray
    private(set) var _c2: MLXArray

    init(dim: Int, codebookSize: Int) {
        self.dim = dim
        self.initialized = MLXArray.zeros([1], dtype: .float32)
        self.embedding_sum = MLXArray.zeros([codebookSize, dim], dtype: .float32)
        self.cluster_usage = MLXArray.zeros([codebookSize], dtype: .float32)
        let safeUsage = maximum(cluster_usage, epsilon).reshaped([codebookSize, 1])
        self._embedding = embedding_sum / safeUsage
        self._c2 = _embedding.square().sum(axis: -1) / 2
    }

    func updateInPlace() {
        let safeUsage = maximum(cluster_usage, epsilon).reshaped([cluster_usage.shape[0], 1])
        _embedding = embedding_sum / safeUsage
        _c2 = _embedding.square().sum(axis: -1) / 2
    }

    override func update(
        parameters: ModuleParameters,
        verify: Module.VerifyUpdate,
        path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        try super.update(parameters: parameters, verify: verify, path: path, modulePath: modulePath)
        updateInPlace()
        return self
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        let targetShape = Array(xs.shape.dropLast())
        let flat = xs.reshaped([-1, dim])
        let dotProduct = flat.matmul(swappedAxes(_embedding, -1, -2))
        let distances = _c2 - dotProduct
        return argMin(distances, axis: -1).reshaped(targetShape)
    }

    func decode(_ xs: MLXArray) -> MLXArray {
        let targetShape = xs.shape + [dim]
        let taken = take(_embedding, xs.flattened(), axis: 0)
        return taken.reshaped(targetShape)
    }
}

final class TTSCodecVectorQuantization: Module {
    @ModuleInfo var project_in: Linear?
    @ModuleInfo var project_out: Linear?
    @ModuleInfo var codebook: TTSCodecEuclideanCodebook

    init(dim: Int, codebookSize: Int, codebookDim: Int?) {
        let cbDim = codebookDim ?? dim
        _project_in.wrappedValue = dim == cbDim ? nil : Linear(dim, cbDim)
        _project_out.wrappedValue = dim == cbDim ? nil : Linear(cbDim, dim)
        _codebook.wrappedValue = TTSCodecEuclideanCodebook(dim: cbDim, codebookSize: codebookSize)
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var x = swappedAxes(xs, -1, -2)
        if let project_in { x = project_in(x) }
        return codebook.encode(x)
    }

    func decode(_ xs: MLXArray) -> MLXArray {
        var x = codebook.decode(xs)
        if let project_out { x = project_out(x) }
        return swappedAxes(x, -1, -2)
    }
}

final class TTSCodecResidualVectorQuantization: Module {
    @ModuleInfo var layers: [TTSCodecVectorQuantization]

    init(nq: Int, dim: Int, codebookSize: Int, codebookDim: Int? = nil) {
        _layers.wrappedValue = (0 ..< nq).map { _ in
            TTSCodecVectorQuantization(dim: dim, codebookSize: codebookSize, codebookDim: codebookDim)
        }
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var codes = [MLXArray]()
        var residual = xs
        for layer in layers {
            let indices = layer.encode(residual)
            let quantized = layer.decode(indices)
            residual = residual - quantized
            codes.append(indices)
        }
        return stacked(codes, axis: 0)
    }

    func decode(_ xs: MLXArray) -> MLXArray {
        var quantized = layers[0].decode(xs[0])
        for i in 1 ..< xs.shape[0] {
            quantized = quantized + layers[i].decode(xs[i])
        }
        return quantized
    }
}

final class TTSCodecResidualVectorQuantizer: Module {
    @ModuleInfo var input_proj: TTSCodecConv1d?
    @ModuleInfo var output_proj: TTSCodecConv1d?
    @ModuleInfo var vq: TTSCodecResidualVectorQuantization

    init(dim: Int, inputDim: Int?, outputDim: Int?, nq: Int, bins: Int, forceProjection: Bool) {
        let inDim = inputDim ?? dim
        let outDim = outputDim ?? dim
        _input_proj.wrappedValue = inDim == dim && !forceProjection
            ? nil
            : TTSCodecConv1d(inChannels: inDim, outChannels: dim, ksize: 1, bias: false)
        _output_proj.wrappedValue = outDim == dim && !forceProjection
            ? nil
            : TTSCodecConv1d(inChannels: dim, outChannels: outDim, ksize: 1, bias: false)
        _vq.wrappedValue = TTSCodecResidualVectorQuantization(nq: nq, dim: dim, codebookSize: bins)
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var x = xs
        if let input_proj { x = input_proj(x) }
        return swappedAxes(vq.encode(x), 0, 1)
    }

    func decode(_ xs: MLXArray) -> MLXArray {
        let x = swappedAxes(xs, 0, 1)
        var quantized = vq.decode(x)
        if let output_proj { quantized = output_proj(quantized) }
        return quantized
    }
}

final class TTSCodecSplitResidualVectorQuantizer: Module {
    private let nq: Int
    @ModuleInfo var rvq_first: TTSCodecResidualVectorQuantizer
    @ModuleInfo var rvq_rest: TTSCodecResidualVectorQuantizer

    init(dim: Int, inputDim: Int?, outputDim: Int?, nq: Int, bins: Int) {
        self.nq = nq
        _rvq_first.wrappedValue = TTSCodecResidualVectorQuantizer(
            dim: dim,
            inputDim: inputDim,
            outputDim: outputDim,
            nq: 1,
            bins: bins,
            forceProjection: true
        )
        _rvq_rest.wrappedValue = TTSCodecResidualVectorQuantizer(
            dim: dim,
            inputDim: inputDim,
            outputDim: outputDim,
            nq: max(nq - 1, 0),
            bins: bins,
            forceProjection: true
        )
    }

    func encode(_ xs: MLXArray) -> MLXArray {
        var codes = rvq_first.encode(xs)
        if nq > 1 {
            let rest = rvq_rest.encode(xs)
            codes = concatenated([codes, rest], axis: 1)
        }
        return codes
    }
}
