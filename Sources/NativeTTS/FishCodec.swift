import Foundation
@preconcurrency import MLX
import MLXFast
import MLXNN

/// Fish Audio's causal modded-DAC codec (S2-Pro / Drama 3): 44.1 kHz audio ↔ 10 codebooks at
/// 2048 samples per frame (1 semantic codebook of 4096, 9 residual codebooks of 1024).
///
/// Weights come from `Drama3Convert.convertCodec` (weight norm folded, MLX conv layout). All tensors are
/// NLC: [batch, time, channels].
public final class FishCodec: @unchecked Sendable {
    public static let sampleRate = 44_100
    public static let hopLength = 512
    public static let frameLength = 2_048
    public static let codebookCount = 10

    let encoder: Encoder
    let quantizer: Quantizer
    let decoder: Decoder

    public static func load(url: URL) throws -> FishCodec {
        let bag = WeightBag(try MLX.loadArrays(url: url))
        let codec = try FishCodec(bag: bag)
        let unused = bag.unused
        guard unused.isEmpty else {
            throw Drama3Error.missingWeights("codec has unused tensors: \(unused.sorted().prefix(6))")
        }
        eval(bag.consumedArrays)
        return codec
    }

    init(bag: WeightBag) throws {
        encoder = try Encoder(bag, prefix: "encoder")
        quantizer = try Quantizer(bag, prefix: "quantizer")
        decoder = try Decoder(bag, prefix: "decoder")
    }

    /// Mono 44.1 kHz samples → codes [codebooks, frames].
    public func encode(samples: MLXArray) -> MLXArray {
        let length = samples.dim(-1)
        let padded = (length + Self.frameLength - 1) / Self.frameLength * Self.frameLength
        var audio = samples.reshaped(1, length, 1).asType(.float32)
        if padded > length {
            audio = MLX.padded(audio, widths: [.init((0, 0)), .init((0, padded - length)), .init((0, 0))])
        }
        let latent = encoder(audio)
        return quantizer.encode(latent)[0].transposed(1, 0)
    }

    /// Codes [codebooks, frames] → mono samples [frames * 2048].
    public func decode(codes: MLXArray) -> MLXArray {
        let latent = quantizer.decode(codes.transposed(1, 0)[.newAxis])
        return decoder(latent).reshaped(-1)
    }

    // MARK: Weight access

    final class WeightBag {
        private let weights: [String: MLXArray]
        private(set) var consumed = Set<String>()

        init(_ weights: [String: MLXArray]) {
            self.weights = weights
        }

        func take(_ key: String) throws -> MLXArray {
            guard let value = weights[key] else { throw Drama3Error.missingWeights("codec tensor \(key)") }
            consumed.insert(key)
            return value
        }

        func has(_ key: String) -> Bool { weights[key] != nil }

        var unused: Set<String> { Set(weights.keys).subtracting(consumed) }
        var consumedArrays: [MLXArray] { consumed.compactMap { weights[$0] } }
    }

    // MARK: Layers

    struct Snake {
        let alpha: MLXArray
        let inverse: MLXArray

        init(_ bag: WeightBag, _ prefix: String) throws {
            alpha = try bag.take(prefix + ".alpha").reshaped(-1)
            inverse = 1 / (alpha + 1e-9)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let s = sin(alpha * x)
            return x + inverse * s * s
        }
    }

    /// Causal conv: left pad (kernel - stride), right pad to complete the last stride.
    struct CausalConv {
        let weight: MLXArray
        let bias: MLXArray?
        let stride: Int
        let dilation: Int
        let groups: Int
        let effectiveKernel: Int

        init(_ bag: WeightBag, _ prefix: String, stride: Int = 1, dilation: Int = 1, groups: Int = 1) throws {
            weight = try bag.take(prefix + ".weight")
            bias = bag.has(prefix + ".bias") ? try bag.take(prefix + ".bias") : nil
            self.stride = stride
            self.dilation = dilation
            self.groups = groups
            effectiveKernel = (weight.dim(1) - 1) * dilation + 1
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let length = x.dim(1)
            let padding = effectiveKernel - stride
            let frames = Double(length - effectiveKernel + padding) / Double(stride) + 1
            let ideal = (Int(frames.rounded(.up)) - 1) * stride + (effectiveKernel - padding)
            let extra = max(0, ideal - length)
            let input = padding + extra > 0
                ? MLX.padded(x, widths: [.init((0, 0)), .init((padding, extra)), .init((0, 0))])
                : x
            var y = conv1d(input, weight, stride: stride, padding: 0, dilation: dilation, groups: groups)
            if let bias { y = y + bias }
            return y
        }
    }

    /// Causal transposed conv: full transposed conv, then drop the trailing (kernel - stride) samples.
    struct CausalTransposedConv {
        let weight: MLXArray
        let bias: MLXArray?
        let stride: Int

        init(_ bag: WeightBag, _ prefix: String, stride: Int) throws {
            weight = try bag.take(prefix + ".weight")
            bias = bag.has(prefix + ".bias") ? try bag.take(prefix + ".bias") : nil
            self.stride = stride
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var y = convTransposed1d(x, weight, stride: stride, padding: 0)
            if let bias { y = y + bias }
            let trim = weight.dim(1) - stride
            return trim > 0 ? y[0..., ..<(y.dim(1) - trim), 0...] : y
        }
    }

    struct ResidualUnit {
        let snake1: Snake
        let conv1: CausalConv
        let snake2: Snake
        let conv2: CausalConv

        init(_ bag: WeightBag, _ prefix: String, dilation: Int) throws {
            snake1 = try Snake(bag, prefix + ".block.0")
            conv1 = try CausalConv(bag, prefix + ".block.1.conv", dilation: dilation)
            snake2 = try Snake(bag, prefix + ".block.2")
            conv2 = try CausalConv(bag, prefix + ".block.3.conv")
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let y = conv2(snake2(conv1(snake1(x))))
            let trim = x.dim(1) - y.dim(1)
            return (trim > 0 ? x[0..., ..<(x.dim(1) - trim), 0...] : x) + y
        }
    }

    struct ConvNeXtBlock {
        let dwconv: CausalConv
        let normWeight: MLXArray
        let normBias: MLXArray
        let pw1Weight: MLXArray
        let pw1Bias: MLXArray
        let pw2Weight: MLXArray
        let pw2Bias: MLXArray
        let gamma: MLXArray

        init(_ bag: WeightBag, _ prefix: String, channels: Int) throws {
            dwconv = try CausalConv(bag, prefix + ".dwconv.conv", groups: channels)
            normWeight = try bag.take(prefix + ".norm.weight")
            normBias = try bag.take(prefix + ".norm.bias")
            pw1Weight = try bag.take(prefix + ".pwconv1.weight")
            pw1Bias = try bag.take(prefix + ".pwconv1.bias")
            pw2Weight = try bag.take(prefix + ".pwconv2.weight")
            pw2Bias = try bag.take(prefix + ".pwconv2.bias")
            gamma = try bag.take(prefix + ".gamma")
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var y = MLXFast.layerNorm(dwconv(x), weight: normWeight, bias: normBias, eps: 1e-6)
            y = gelu(matmul(y, pw1Weight.T) + pw1Bias)
            y = matmul(y, pw2Weight.T) + pw2Bias
            return x + gamma * y
        }
    }

    /// Window-limited causal transformer with RMSNorm, layer scale, and the checkpoint's RoPE table.
    struct WindowTransformer {
        struct Layer {
            let wqkv: MLXArray
            let wo: MLXArray
            let w1: MLXArray
            let w2: MLXArray
            let w3: MLXArray
            let attentionNorm: MLXArray
            let ffnNorm: MLXArray
            let attentionScale: MLXArray
            let ffnScale: MLXArray
        }

        let layers: [Layer]
        let norm: MLXArray
        let freqs: MLXArray
        let heads: Int
        let headDim: Int
        let window: Int

        init(_ bag: WeightBag, _ prefix: String, layerCount: Int, heads: Int, window: Int) throws {
            layers = try (0 ..< layerCount).map { index in
                let p = "\(prefix).layers.\(index)"
                return Layer(
                    wqkv: try bag.take(p + ".attention.wqkv.weight"),
                    wo: try bag.take(p + ".attention.wo.weight"),
                    w1: try bag.take(p + ".feed_forward.w1.weight"),
                    w2: try bag.take(p + ".feed_forward.w2.weight"),
                    w3: try bag.take(p + ".feed_forward.w3.weight"),
                    attentionNorm: try bag.take(p + ".attention_norm.weight"),
                    ffnNorm: try bag.take(p + ".ffn_norm.weight"),
                    attentionScale: try bag.take(p + ".attention_layer_scale.gamma"),
                    ffnScale: try bag.take(p + ".ffn_layer_scale.gamma")
                )
            }
            norm = try bag.take(prefix + ".norm.weight")
            freqs = try bag.take(prefix + ".freqs_cis")
            self.heads = heads
            headDim = freqs.dim(1) * 2
            self.window = window
        }

        func rotate(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
            // x [B, T, H, D] with interleaved (even, odd) pairs.
            let shape = x.shape
            let pairs = x.reshaped(shape[0], shape[1], shape[2], headDim / 2, 2)
            let even = pairs[.ellipsis, 0]
            let odd = pairs[.ellipsis, 1]
            return stacked([even * cos - odd * sin, odd * cos + even * sin], axis: -1).reshaped(shape)
        }

        func callAsFunction(_ input: MLXArray) -> MLXArray {
            let B = input.dim(0)
            let T = input.dim(1)
            precondition(T <= freqs.dim(0), "codec transformer supports at most \(freqs.dim(0)) positions")
            let table = freqs[..<T]
            let cos = table[.newAxis, 0..., .newAxis, 0..., 0]
            let sin = table[.newAxis, 0..., .newAxis, 0..., 1]
            let rows = MLXArray.arange(T, dtype: .int32)[0..., .newAxis]
            let cols = MLXArray.arange(T, dtype: .int32)[.newAxis, 0...]
            let allowed = (cols .<= rows) .&& (cols .> (rows - Int32(window)))
            let mask = MLX.where(allowed, MLXArray(Float(0)), MLXArray(Float(-1e9)))
            var x = input
            for layer in layers {
                let h = MLXFast.rmsNorm(x, weight: layer.attentionNorm, eps: 1e-5)
                let qkv = split(matmul(h, layer.wqkv.T), parts: 3, axis: -1)
                let q = rotate(qkv[0].reshaped(B, T, heads, headDim), cos: cos, sin: sin).transposed(0, 2, 1, 3)
                let k = rotate(qkv[1].reshaped(B, T, heads, headDim), cos: cos, sin: sin).transposed(0, 2, 1, 3)
                let v = qkv[2].reshaped(B, T, heads, headDim).transposed(0, 2, 1, 3)
                let attended = MLXFast.scaledDotProductAttention(
                    queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(), mask: .array(mask)
                )
                .transposed(0, 2, 1, 3).reshaped(B, T, heads * headDim)
                x = x + layer.attentionScale * matmul(attended, layer.wo.T)
                let f = MLXFast.rmsNorm(x, weight: layer.ffnNorm, eps: 1e-5)
                let ff = matmul(silu(matmul(f, layer.w1.T)) * matmul(f, layer.w3.T), layer.w2.T)
                x = x + layer.ffnScale * ff
            }
            return MLXFast.rmsNorm(x, weight: norm, eps: 1e-5)
        }
    }

    // MARK: Encoder

    struct Encoder {
        struct Block {
            let units: [ResidualUnit]
            let snake: Snake
            let conv: CausalConv
            let transformer: WindowTransformer?
        }

        let inputConv: CausalConv
        let blocks: [Block]
        let snake: Snake
        let outputConv: CausalConv

        init(_ bag: WeightBag, prefix: String) throws {
            inputConv = try CausalConv(bag, "\(prefix).block.0.conv")
            let strides = [2, 4, 8, 8]
            var dim = 64
            var blocks: [Block] = []
            for (index, stride) in strides.enumerated() {
                dim *= 2
                let p = "\(prefix).block.\(index + 1).block"
                let transformerPrefix = "\(p).5"
                blocks.append(Block(
                    units: try [1, 3, 9].enumerated().map { try ResidualUnit(bag, "\(p).\($0.offset)", dilation: $0.element) },
                    snake: try Snake(bag, "\(p).3"),
                    conv: try CausalConv(bag, "\(p).4.conv", stride: stride),
                    transformer: bag.has("\(transformerPrefix).norm.weight")
                        ? try WindowTransformer(bag, transformerPrefix, layerCount: 4, heads: dim / 64, window: 512)
                        : nil
                ))
            }
            self.blocks = blocks
            snake = try Snake(bag, "\(prefix).block.5")
            outputConv = try CausalConv(bag, "\(prefix).block.6.conv")
        }

        func callAsFunction(_ audio: MLXArray) -> MLXArray {
            var x = inputConv(audio)
            for block in blocks {
                for unit in block.units { x = unit(x) }
                x = block.conv(block.snake(x))
                if let transformer = block.transformer { x = transformer(x) }
            }
            return outputConv(snake(x))
        }
    }

    // MARK: Quantizer

    struct VectorQuantizer {
        let inProjWeight: MLXArray
        let inProjBias: MLXArray
        let outProjWeight: MLXArray
        let outProjBias: MLXArray
        let codebook: MLXArray
        let normalizedCodebook: MLXArray

        init(_ bag: WeightBag, _ prefix: String) throws {
            inProjWeight = try bag.take(prefix + ".in_proj.weight").squeezed(axis: 1)
            inProjBias = try bag.take(prefix + ".in_proj.bias")
            outProjWeight = try bag.take(prefix + ".out_proj.weight").squeezed(axis: 1)
            outProjBias = try bag.take(prefix + ".out_proj.bias")
            codebook = try bag.take(prefix + ".codebook.weight")
            normalizedCodebook = codebook / maximum(sqrt((codebook * codebook).sum(axis: 1, keepDims: true)), 1e-12)
        }

        var size: Int { codebook.dim(0) }

        func decode(_ codes: MLXArray) -> MLXArray {
            matmul(codebook[clip(codes, min: 0, max: size - 1)], outProjWeight.T) + outProjBias
        }

        /// Nearest code (cosine) of each latent row; returns (codes [B, T], quantized latent [B, T, C]).
        func encode(_ z: MLXArray) -> (MLXArray, MLXArray) {
            let projected = matmul(z, inProjWeight.T) + inProjBias
            let normalized = projected / maximum(sqrt((projected * projected).sum(axis: -1, keepDims: true)), 1e-12)
            let codes = argMax(matmul(normalized, normalizedCodebook.T), axis: -1)
            return (codes, decode(codes))
        }
    }

    struct Quantizer {
        let semantic: VectorQuantizer
        let residual: [VectorQuantizer]
        let downsample: [(CausalConv, ConvNeXtBlock)]
        let upsample: [(CausalTransposedConv, ConvNeXtBlock)]
        let preModule: WindowTransformer
        let postModule: WindowTransformer

        init(_ bag: WeightBag, prefix: String) throws {
            semantic = try VectorQuantizer(bag, "\(prefix).semantic_quantizer.quantizers.0")
            residual = try (0 ..< FishCodec.codebookCount - 1).map {
                try VectorQuantizer(bag, "\(prefix).quantizer.quantizers.\($0)")
            }
            downsample = try (0 ..< 2).map {
                (try CausalConv(bag, "\(prefix).downsample.\($0).0.conv", stride: 2),
                 try ConvNeXtBlock(bag, "\(prefix).downsample.\($0).1", channels: 1024))
            }
            upsample = try (0 ..< 2).map {
                (try CausalTransposedConv(bag, "\(prefix).upsample.\($0).0.conv", stride: 2),
                 try ConvNeXtBlock(bag, "\(prefix).upsample.\($0).1", channels: 1024))
            }
            preModule = try WindowTransformer(bag, "\(prefix).pre_module", layerCount: 8, heads: 16, window: 128)
            postModule = try WindowTransformer(bag, "\(prefix).post_module", layerCount: 8, heads: 16, window: 128)
        }

        /// Latent [1, T, 1024] at hop 512 → codes [1, T / 4, codebooks].
        func encode(_ latent: MLXArray) -> MLXArray {
            var z = latent
            for (conv, block) in downsample { z = block(conv(z)) }
            z = preModule(z)
            let (semanticCodes, semanticZ) = semantic.encode(z)
            var remainder = z - semanticZ
            var codes = [semanticCodes]
            for quantizer in residual {
                let (indices, quantized) = quantizer.encode(remainder)
                codes.append(indices)
                remainder = remainder - quantized
            }
            return stacked(codes, axis: -1)
        }

        /// Codes [1, T, codebooks] → latent [1, 4T, 1024].
        func decode(_ codes: MLXArray) -> MLXArray {
            var z = semantic.decode(codes[0..., 0..., 0])
            for (index, quantizer) in residual.enumerated() where index + 1 < codes.dim(-1) {
                z = z + quantizer.decode(codes[0..., 0..., index + 1])
            }
            z = postModule(z)
            for (conv, block) in upsample { z = block(conv(z)) }
            return z
        }
    }

    // MARK: Decoder

    struct Decoder {
        struct Block {
            let snake: Snake
            let upsample: CausalTransposedConv
            let units: [ResidualUnit]
        }

        let inputConv: CausalConv
        let blocks: [Block]
        let snake: Snake
        let outputConv: CausalConv

        init(_ bag: WeightBag, prefix: String) throws {
            inputConv = try CausalConv(bag, "\(prefix).model.0.conv")
            blocks = try [8, 8, 4, 2].enumerated().map { index, stride in
                let p = "\(prefix).model.\(index + 1).block"
                return Block(
                    snake: try Snake(bag, "\(p).0"),
                    upsample: try CausalTransposedConv(bag, "\(p).1.conv", stride: stride),
                    units: try [1, 3, 9].enumerated().map { try ResidualUnit(bag, "\(p).\($0.offset + 2)", dilation: $0.element) }
                )
            }
            snake = try Snake(bag, "\(prefix).model.5")
            outputConv = try CausalConv(bag, "\(prefix).model.6.conv")
        }

        func callAsFunction(_ latent: MLXArray) -> MLXArray {
            var x = inputConv(latent)
            for block in blocks {
                x = block.upsample(block.snake(x))
                for unit in block.units { x = unit(x) }
            }
            return tanh(outputConv(snake(x)))
        }
    }
}
