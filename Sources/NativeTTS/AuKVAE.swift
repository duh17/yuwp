import Foundation
@preconcurrency import MLX
import MLXNN

final class AuKResStackPair: Module {
    @ModuleInfo var conv1: AuKConv1d
    @ModuleInfo var conv2: AuKConv1d

    init(channels: Int, kernelSize: Int, dilation: Int) {
        _conv1.wrappedValue = AuKConv1d(
            inChannels: channels,
            outChannels: channels,
            kernelSize: kernelSize,
            dilation: dilation,
            padding: dilation
        )
        _conv2.wrappedValue = AuKConv1d(
            inChannels: channels,
            outChannels: channels,
            kernelSize: kernelSize,
            dilation: 1,
            padding: 1
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(aukLeakyReLU(x, slope: 0.01))
        h = conv2(aukLeakyReLU(h, slope: 0.01))
        return x + h
    }
}

final class AuKResStack: Module {
    @ModuleInfo var layers: [AuKResStackPair]

    init(channels: Int, kernelSize: Int = 3, base: Int = 3, nums: Int = 4) {
        _layers.wrappedValue = (0 ..< nums).map { i in
            let dilation = Int(pow(Double(base), Double(i)))
            return AuKResStackPair(channels: channels, kernelSize: kernelSize, dilation: dilation)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in layers {
            h = layer(h)
        }
        return h
    }
}

final class AuKVAEEncoderStage: Module {
    @ModuleInfo var down: AuKConv1d
    @ModuleInfo var stack: AuKResStack

    init(inChannels: Int, outChannels: Int, factor: Int) {
        let kernel = factor * 2
        _down.wrappedValue = AuKConv1d(
            inChannels: inChannels,
            outChannels: outChannels,
            kernelSize: kernel,
            stride: factor,
            padding: (kernel - 1) / 2
        )
        _stack.wrappedValue = AuKResStack(channels: outChannels, kernelSize: 3, base: 2, nums: 6)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = down(x)
        h = stack(h)
        return aukLeakyReLU(h, slope: 0.2)
    }
}

final class AuKVAEEncoder: Module {
    @ModuleInfo var pre: AuKConv1d
    @ModuleInfo var stages: [AuKVAEEncoderStage]
    @ModuleInfo var post: AuKConv1d

    override init() {
        let channels = AuKFlashConfig.downsampleChannels
        let rates = AuKFlashConfig.downsampleRates
        _pre.wrappedValue = AuKConv1d(inChannels: 1, outChannels: channels[0], kernelSize: 3, padding: 1)
        var built: [AuKVAEEncoderStage] = []
        built.reserveCapacity(rates.count)
        for (index, factor) in rates.enumerated() {
            built.append(
                AuKVAEEncoderStage(
                    inChannels: channels[index],
                    outChannels: channels[index + 1],
                    factor: factor
                )
            )
        }
        _stages.wrappedValue = built
        _post.wrappedValue = AuKConv1d(
            inChannels: channels[channels.count - 1],
            outChannels: AuKFlashConfig.latentDim * 2,
            kernelSize: 3,
            padding: 1
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = aukLeakyReLU(pre(x), slope: 0.2)
        for stage in stages {
            h = stage(h)
        }
        return post(h)
    }
}

final class AuKAMPBlock1: Module {
    @ModuleInfo var convs1: [AuKConv1d]
    @ModuleInfo var convs2: [AuKConv1d]
    @ModuleInfo var activations: [AuKActivation1d]

    init(channels: Int, kernelSize: Int, dilation: [Int]) {
        _convs1.wrappedValue = dilation.map { d in
            AuKConv1d(
                inChannels: channels,
                outChannels: channels,
                kernelSize: kernelSize,
                dilation: d,
                causal: true
            )
        }
        _convs2.wrappedValue = dilation.map { _ in
            AuKConv1d(
                inChannels: channels,
                outChannels: channels,
                kernelSize: kernelSize,
                dilation: 1,
                causal: true
            )
        }
        _activations.wrappedValue = (0 ..< (2 * dilation.count)).map { _ in
            AuKActivation1d(channels: channels, alphaLogscale: true, causal: true)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        let act1 = stride(from: 0, to: activations.count, by: 2).compactMap { i in
            i < activations.count ? activations[i] : nil
        }
        let act2 = stride(from: 1, to: activations.count, by: 2).compactMap { i in
            i < activations.count ? activations[i] : nil
        }
        for i in 0 ..< convs1.count {
            var residual = act1[i](h)
            residual = convs1[i](residual)
            residual = act2[i](residual)
            residual = convs2[i](residual)
            h = h + residual
        }
        return h
    }
}

final class AuKVAEDecoder: Module {
    let numKernels: Int
    let numUpsamples: Int
    @ModuleInfo(key: "conv_pre") var convPre: AuKConv1d
    @ModuleInfo var ups: [AuKConvTranspose1d]
    @ModuleInfo var resblocks: [AuKAMPBlock1]
    @ModuleInfo(key: "activation_post") var activationPost: AuKActivation1d
    @ModuleInfo(key: "conv_post") var convPost: AuKConv1d

    override init() {
        numKernels = AuKFlashConfig.resblockKernelSizes.count
        numUpsamples = AuKFlashConfig.upsampleRates.count
        var channels = AuKFlashConfig.upsampleInitialChannel
        _convPre.wrappedValue = AuKConv1d(
            inChannels: AuKFlashConfig.latentDim,
            outChannels: channels,
            kernelSize: 7,
            causal: AuKFlashConfig.convPreIsCausal
        )

        var upsBuilt: [AuKConvTranspose1d] = []
        var resBuilt: [AuKAMPBlock1] = []
        upsBuilt.reserveCapacity(numUpsamples)
        for i in 0 ..< numUpsamples {
            let next = channels / 2
            upsBuilt.append(
                AuKConvTranspose1d(
                    inChannels: channels,
                    outChannels: next,
                    kernelSize: AuKFlashConfig.upsampleKernelSizes[i],
                    stride: AuKFlashConfig.upsampleRates[i],
                    causal: true
                )
            )
            channels = next
            for (kernel, dilation) in zip(AuKFlashConfig.resblockKernelSizes, AuKFlashConfig.resblockDilationSizes) {
                resBuilt.append(AuKAMPBlock1(channels: channels, kernelSize: kernel, dilation: dilation))
            }
        }
        _ups.wrappedValue = upsBuilt
        _resblocks.wrappedValue = resBuilt
        _activationPost.wrappedValue = AuKActivation1d(channels: channels, alphaLogscale: true, causal: true)
        _convPost.wrappedValue = AuKConv1d(
            inChannels: channels,
            outChannels: 1,
            kernelSize: 7,
            bias: false,
            causal: true
        )
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convPre(x)
        for i in 0 ..< numUpsamples {
            h = ups[i](h)
            var acc: MLXArray?
            for j in 0 ..< numKernels {
                let r = resblocks[i * numKernels + j](h)
                acc = acc.map { $0 + r } ?? r
            }
            if let acc {
                h = acc / Float(numKernels)
            }
        }
        h = convPost(activationPost(h))
        return clip(h, min: -1, max: 1)
    }
}

public final class AuKBigVGANFlowVAE: Module {
    @ModuleInfo(key: "audio_encoder") var audioEncoder: AuKVAEEncoder
    @ModuleInfo var decoder: AuKVAEDecoder
    let global_mean: MLXArray
    let global_log_std: MLXArray

    public override init() {
        _audioEncoder.wrappedValue = AuKVAEEncoder()
        _decoder.wrappedValue = AuKVAEDecoder()
        self.global_mean = MLXArray.zeros([AuKFlashConfig.latentDim])
        self.global_log_std = MLXArray.ones([AuKFlashConfig.latentDim])
    }

    public func encode(_ wav: MLXArray) -> MLXArray {
        let stats = audioEncoder(wav)
        let dim = AuKFlashConfig.latentDim
        let mean = stats[.ellipsis, ..<dim]
        return (mean - global_mean) / sqrt(global_log_std)
    }

    public func denormalize(_ latents: MLXArray) -> MLXArray {
        latents * sqrt(global_log_std) + global_mean
    }

    public func decode(_ latents: MLXArray) -> MLXArray {
        decoder(denormalize(latents))
    }
}
