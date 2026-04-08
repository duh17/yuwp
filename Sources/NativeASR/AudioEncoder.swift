// NativeASR — Audio Encoder
// Conv2d stem + sinusoidal position encoding + transformer with bidirectional windowed attention.
//
// GOTCHAS:
// - Conv2d in mlx-swift expects NHWC (channels-last). Input must be [batch, freq, time, 1].
// - Weights from PyTorch OIHW are transposed to OHWI during loading (sanitize()).
// - Windowed attention via block-diagonal additive mask passed to SDPA.
// - LayerNorm (with bias) — encoder uses LayerNorm, not RMSNorm.

import Foundation
import MLX
import MLXFast
import MLXNN

// MARK: - Sinusoidal Position Embedding

final class SinusoidalPositionEmbedding: Module {
    let channels: Int
    let invTimescales: MLXArray

    init(channels: Int, maxTimescale: Float = 10000.0) {
        precondition(channels % 2 == 0, "channels must be even")
        self.channels = channels
        let logStep = log(maxTimescale) / Float(channels / 2 - 1)
        invTimescales = MLX.exp(
            -logStep * MLXArray((0 ..< (channels / 2)).map { Float($0) })
        )
    }

    func callAsFunction(_ seqLen: Int) -> MLXArray {
        let positions = MLXArray((0 ..< seqLen).map { Float($0) }).expandedDimensions(axis: 1)
        let scaled = positions * invTimescales.expandedDimensions(axis: 0)
        return MLX.concatenated([MLX.sin(scaled), MLX.cos(scaled)], axis: 1)
    }
}

// MARK: - Audio Attention

final class AudioAttention: Module {
    let numHeads: Int
    let headDim: Int
    let scaling: Float

    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(config: AudioEncoderConfig) {
        let embedDim = config.dModel
        numHeads = config.encoderAttentionHeads
        headDim = embedDim / numHeads
        scaling = pow(Float(headDim), -0.5)
        precondition(headDim * numHeads == embedDim, "d_model must be divisible by encoder_attention_heads")

        _qProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        _kProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        _vProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
        _outProj.wrappedValue = Linear(embedDim, embedDim, bias: true)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        let (bsz, seqLen, embedDim) = (x.shape[0], x.shape[1], x.shape[2])

        let q = (qProj(x) * scaling)
            .reshaped(bsz, seqLen, numHeads, headDim)
            .transposed(0, 2, 1, 3)
        let k = kProj(x)
            .reshaped(bsz, seqLen, numHeads, headDim)
            .transposed(0, 2, 1, 3)
        let v = vProj(x)
            .reshaped(bsz, seqLen, numHeads, headDim)
            .transposed(0, 2, 1, 3)

        let maskMode: MLXFast.ScaledDotProductAttentionMaskMode = mask.map { .array($0) } ?? .none
        let out = MLXFast.scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0, mask: maskMode
        )

        return outProj(
            out.transposed(0, 2, 1, 3).reshaped(bsz, seqLen, embedDim)
        )
    }
}

// MARK: - Audio Encoder Layer

final class AudioEncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: AudioAttention
    @ModuleInfo(key: "self_attn_layer_norm") var selfAttnNorm: LayerNorm
    @ModuleInfo var fc1: Linear
    @ModuleInfo var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm

    init(config: AudioEncoderConfig) {
        let d = config.dModel
        _selfAttn.wrappedValue = AudioAttention(config: config)
        _selfAttnNorm.wrappedValue = LayerNorm(dimensions: d)
        _fc1.wrappedValue = Linear(d, config.encoderFfnDim)
        _fc2.wrappedValue = Linear(config.encoderFfnDim, d)
        _finalNorm.wrappedValue = LayerNorm(dimensions: d)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray? = nil) -> MLXArray {
        var h = x
        let r1 = h
        h = selfAttnNorm(h)
        h = selfAttn(h, mask: mask)
        h = r1 + h

        let r2 = h
        h = finalNorm(h)
        h = gelu(fc1(h))
        h = fc2(h)
        return r2 + h
    }
}

// MARK: - Audio Encoder

final class AudioEncoder: Module {
    let config: AudioEncoderConfig
    let nWindow: Int
    let nWindowInfer: Int

    @ModuleInfo var conv2d1: Conv2d
    @ModuleInfo var conv2d2: Conv2d
    @ModuleInfo var conv2d3: Conv2d
    @ModuleInfo(key: "conv_out") var convOut: Linear
    @ModuleInfo(key: "positional_embedding") var positionalEmbedding: SinusoidalPositionEmbedding
    @ModuleInfo var layers: [AudioEncoderLayer]
    @ModuleInfo(key: "ln_post") var lnPost: LayerNorm
    @ModuleInfo var proj1: Linear
    @ModuleInfo var proj2: Linear

    init(config: AudioEncoderConfig) {
        self.config = config
        nWindow = config.nWindow
        nWindowInfer = config.nWindowInfer

        let d = config.dModel
        let ch = config.downsampleHiddenSize

        _conv2d1.wrappedValue = Conv2d(inputChannels: 1, outputChannels: ch, kernelSize: IntOrPair(3), stride: IntOrPair(2), padding: IntOrPair(1))
        _conv2d2.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: IntOrPair(3), stride: IntOrPair(2), padding: IntOrPair(1))
        _conv2d3.wrappedValue = Conv2d(inputChannels: ch, outputChannels: ch, kernelSize: IntOrPair(3), stride: IntOrPair(2), padding: IntOrPair(1))

        // freq dim after 3 stride-2 convolutions over 128 mel bins
        let freqAfterConv = ((((config.numMelBins + 1) / 2) + 1) / 2 + 1) / 2
        _convOut.wrappedValue = Linear(ch * freqAfterConv, d, bias: false)

        _positionalEmbedding.wrappedValue = SinusoidalPositionEmbedding(channels: d)

        _layers.wrappedValue = (0 ..< config.encoderLayers).map { _ in AudioEncoderLayer(config: config) }

        _lnPost.wrappedValue = LayerNorm(dimensions: d)
        _proj1.wrappedValue = Linear(d, d)
        _proj2.wrappedValue = Linear(d, config.outputDim)
    }

    /// Output length after 3 stride-2 convolutions for a single chunk of `inputLen` frames.
    private static func convOutputLength(_ inputLen: Int) -> Int {
        let leave = inputLen % 100
        let feat = leave > 0 ? (leave - 1) / 2 + 1 : 0
        let s1 = feat > 0 ? (feat - 1) / 2 + 1 : 0
        let s2 = s1 > 0 ? (s1 - 1) / 2 + 1 : 0
        return s2 + (inputLen / 100) * 13
    }

    /// Block-diagonal additive attention mask restricting attention to within windows.
    /// Built on GPU using MLX operations instead of CPU loop (O(N²) CPU work was bottleneck for long audio).
    private func makeBlockMask(seqLen: Int, cuSeqlens: [Int], dtype: DType) -> MLXArray {
        // Assign each position to its block index
        var blockIds = [Int32](repeating: 0, count: seqLen)
        for i in 0 ..< max(0, cuSeqlens.count - 1) {
            let start = cuSeqlens[i], end = min(cuSeqlens[i + 1], seqLen)
            for pos in start ..< end {
                blockIds[pos] = Int32(i)
            }
        }
        // Build mask on GPU: positions in same block = 0, different block = -1e9
        let ids = MLXArray(blockIds)
        let rowIds = ids.expandedDimensions(axis: 1)  // (seqLen, 1)
        let colIds = ids.expandedDimensions(axis: 0)  // (1, seqLen)
        let sameBlock = (rowIds .== colIds).asType(dtype)  // 1.0 where same block
        let mask = MLX.where(sameBlock .== 1, MLXArray(Float(0.0)).asType(dtype), MLXArray(Float(-1e9)).asType(dtype))
        return mask.expandedDimensions(axis: 0).expandedDimensions(axis: 0)
    }

    func callAsFunction(_ inputFeatures: MLXArray, featureAttentionMask: MLXArray? = nil) -> MLXArray {
        let batchSize = inputFeatures.shape[0]
        let totalFrames = inputFeatures.shape[2]
        let chunkSize = max(1, nWindow * 2)

        var hiddenPerSample: [MLXArray] = []
        var afterCnnLens: [Int] = []
        var maxLenAfterCnn = 1

        for sampleIdx in 0 ..< batchSize {
            let sampleFeatures = inputFeatures[sampleIdx]
            // For batch transcription with all-1s mask, featureLength == totalFrames.
            // Skip the .sum().item() sync point when mask is all-1s (our common case).
            let featureLength = totalFrames

            // Split into chunks
            var chunkLengths: [Int] = []
            var chunks: [MLXArray] = []
            var pos = 0
            while pos < featureLength {
                let end = min(pos + chunkSize, featureLength)
                chunkLengths.append(end - pos)
                chunks.append(sampleFeatures[0..., pos ..< end])
                pos = end
            }

            // Pad chunks to uniform length for batched conv
            let maxChunkLen = chunkLengths.max() ?? chunkSize
            let paddedChunks: [MLXArray] = chunks.enumerated().map { (i, chunk) in
                let len = chunkLengths[i]
                if len < maxChunkLen {
                    return MLX.padded(chunk, widths: [IntOrPair((0, 0)), IntOrPair((0, maxChunkLen - len))])
                }
                return chunk
            }

            // Batched conv stem — input shape [numChunks, freq, time, 1] (NHWC)
            var x = MLX.stacked(paddedChunks, axis: 0).expandedDimensions(axis: -1)
            x = gelu(conv2d1(x))
            x = gelu(conv2d2(x))
            x = gelu(conv2d3(x))

            // Compute conv output dims from input dims (avoid .shape queries on lazy arrays)
            let numChunks = paddedChunks.count
            let channels = config.downsampleHiddenSize
            // freq dim after 3 stride-2 convolutions: ((((128+1)/2)+1)/2+1)/2 = 16
            let freqAfterConv = ((((config.numMelBins + 1) / 2) + 1) / 2 + 1) / 2
            // time dim after 3 stride-2 convolutions
            func convDim(_ d: Int) -> Int { (d + 2 * 1 - 3) / 2 + 1 }
            let timeAfterConv = convDim(convDim(convDim(maxChunkLen)))
            maxLenAfterCnn = max(maxLenAfterCnn, timeAfterConv)

            // Reshape from NHWC to [chunks, time, channels*freq] → project → add pos emb
            x = x.transposed(0, 2, 3, 1).reshaped([numChunks, timeAfterConv, channels * freqAfterConv])
            x = convOut(x)
            x = x + positionalEmbedding(timeAfterConv).expandedDimensions(axis: 0)

            // Collect valid outputs per chunk
            let chunkHidden: [MLXArray] = chunkLengths.enumerated().map { (i, len) in
                let validLen = max(1, Self.convOutputLength(len))
                return x[i, 0 ..< validLen, 0...]
            }

            let sampleHidden = chunkHidden.count == 1
                ? chunkHidden[0]
                : MLX.concatenated(chunkHidden, axis: 0)
            hiddenPerSample.append(sampleHidden)
            // Compute total from chunkLengths instead of shape query
            let sampleLen = chunkLengths.reduce(0) { $0 + max(1, Self.convOutputLength($1)) }
            afterCnnLens.append(sampleLen)
        }

        let hiddenFlat = hiddenPerSample.count == 1
            ? hiddenPerSample[0]
            : MLX.concatenated(hiddenPerSample, axis: 0)

        // Build block attention mask for windowed attention
        let inferScale = max(1, nWindowInfer / max(1, nWindow * 2))
        let windowAfterCnn = max(1, maxLenAfterCnn * inferScale)

        var cuChunkLens: [Int] = [0]
        for cnnLen in afterCnnLens {
            let full = cnnLen / windowAfterCnn
            if full > 0 { cuChunkLens.append(contentsOf: Array(repeating: windowAfterCnn, count: full)) }
            let rem = cnnLen % windowAfterCnn
            if rem > 0 { cuChunkLens.append(rem) }
        }
        var cuSeqlens: [Int] = []
        var running = 0
        for c in cuChunkLens { running += c; cuSeqlens.append(running) }

        let seqLen = afterCnnLens.reduce(0, +)
        let attnMask = makeBlockMask(seqLen: seqLen, cuSeqlens: cuSeqlens, dtype: .bfloat16)

        var h = hiddenFlat.expandedDimensions(axis: 0)
        for layer in layers {
            h = layer(h, mask: attnMask)
        }
        h = h.squeezed(axis: 0)
        h = lnPost(h)
        h = gelu(proj1(h))
        h = proj2(h)
        return h
    }
}
