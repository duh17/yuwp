import Foundation
@preconcurrency import MLX
import MLXNN

func aukModifiedBesselI0(_ x: Double) -> Double {
    var total = 1.0
    var term = 1.0
    let quarterSq = (x * x) / 4.0
    for k in 1 ..< 40 {
        term *= quarterSq / (Double(k) * Double(k))
        total += term
    }
    return total
}

func aukKaiserSincFilter1d(cutoff: Float, halfWidth: Float, kernelSize: Int) -> [Float] {
    guard kernelSize > 0 else { return [] }
    if cutoff == 0 {
        return [Float](repeating: 0, count: kernelSize)
    }

    let even = kernelSize % 2 == 0
    let halfSize = kernelSize / 2
    let deltaF = 4.0 * Double(halfWidth)
    let a = 2.285 * Double(halfSize - 1) * Double.pi * deltaF + 7.95
    let beta: Double
    if a > 50 {
        beta = 0.1102 * (a - 8.7)
    } else if a >= 21 {
        beta = 0.5842 * pow(a - 21, 0.4) + 0.07886 * (a - 21)
    } else {
        beta = 0
    }

    var window = [Double](repeating: 0, count: kernelSize)
    let denom = Double(kernelSize - 1) / 2.0
    let i0Beta = aukModifiedBesselI0(beta)
    for n in 0 ..< kernelSize {
        let r = (Double(n) - Double(kernelSize - 1) / 2.0) / denom
        let arg = beta * sqrt(max(1.0 - r * r, 0))
        window[n] = aukModifiedBesselI0(arg) / i0Beta
    }

    var time = [Double](repeating: 0, count: kernelSize)
    if even {
        for i in 0 ..< kernelSize {
            time[i] = Double(i - halfSize) + 0.5
        }
    } else {
        for i in 0 ..< kernelSize {
            time[i] = Double(i - halfSize)
        }
    }

    var filt = [Double](repeating: 0, count: kernelSize)
    var sum = 0.0
    for i in 0 ..< kernelSize {
        let arg2 = 2.0 * Double(cutoff) * time[i]
        let sinc: Double
        if abs(arg2) < 1e-12 {
            sinc = 1
        } else {
            let piArg = Double.pi * arg2
            sinc = abs(piArg) < 1e-12 ? 1 : sin(piArg) / piArg
        }
        filt[i] = 2.0 * Double(cutoff) * window[i] * sinc
        sum += filt[i]
    }
    if sum != 0 {
        for i in 0 ..< kernelSize {
            filt[i] /= sum
        }
    }
    return filt.map(Float.init)
}

final class AuKConv1d: Module {
    let stride: Int
    let dilation: Int
    let groups: Int
    let causal: Bool
    let leftPadding: Int
    let padding: Int
    let weight: MLXArray
    let bias: MLXArray?

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        dilation: Int = 1,
        groups: Int = 1,
        bias: Bool = true,
        causal: Bool = false,
        padding: Int? = nil
    ) {
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.causal = causal
        if causal {
            self.leftPadding = dilation * (kernelSize - 1)
            self.padding = 0
        } else {
            self.leftPadding = 0
            self.padding = padding ?? Int((kernelSize * dilation - dilation) / 2)
        }
        self.weight = MLXArray.zeros([outChannels, kernelSize, inChannels / max(groups, 1)])
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var input = x
        if causal && leftPadding > 0 {
            input = padded(input, widths: [IntOrPair(0), IntOrPair((leftPadding, 0)), IntOrPair(0)])
        }
        var y = conv1d(
            input,
            weight,
            stride: stride,
            padding: padding,
            dilation: dilation,
            groups: groups
        )
        if let bias {
            y = y + bias
        }
        return y
    }
}

final class AuKConvTranspose1d: Module {
    let stride: Int
    let causal: Bool
    let padding: Int
    let weight: MLXArray
    let bias: MLXArray?

    init(
        inChannels: Int,
        outChannels: Int,
        kernelSize: Int,
        stride: Int = 1,
        bias: Bool = true,
        causal: Bool = false,
        padding: Int? = nil
    ) {
        self.stride = stride
        self.causal = causal
        if causal {
            self.padding = 0
        } else {
            self.padding = padding ?? (kernelSize - stride) / 2
        }
        self.weight = MLXArray.zeros([outChannels, kernelSize, inChannels])
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = convTransposed1d(x, weight, stride: stride, padding: padding)
        if let bias {
            y = y + bias
        }
        if causal {
            let time = y.dim(1)
            let keep = max(0, time - stride)
            if keep < time {
                y = y[0..., 0 ..< keep, 0...]
            }
        }
        return y
    }
}

final class AuKSnakeBeta: Module {
    let alphaLogscale: Bool
    let alpha: MLXArray
    let beta: MLXArray

    init(channels: Int, alphaLogscale: Bool = true) {
        self.alphaLogscale = alphaLogscale
        self.alpha = MLXArray.zeros([channels])
        self.beta = MLXArray.zeros([channels])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var a = alpha
        var b = beta
        if alphaLogscale {
            a = exp(a)
            b = exp(b)
        }
        let s = sin(x * a)
        return x + (1.0 / (b + 1e-9)) * s * s
    }
}

struct AuKUpSample1d {
    let ratio: Int
    let kernelSize: Int
    let causal: Bool
    let pad: Int
    let padLeft: Int
    let padRight: Int
    let filter: MLXArray

    init(ratio: Int = 2, kernelSize: Int = 12, causal: Bool = false) {
        self.ratio = ratio
        self.kernelSize = kernelSize
        self.causal = causal
        if causal {
            self.pad = 0
            self.padLeft = 0
            self.padRight = 0
        } else {
            self.pad = kernelSize / ratio - 1
            self.padLeft = pad * ratio + (kernelSize - ratio) / 2
            self.padRight = pad * ratio + (kernelSize - ratio + 1) / 2
        }
        self.filter = MLXArray(aukKaiserSincFilter1d(cutoff: 0.5 / Float(ratio), halfWidth: 0.6 / Float(ratio), kernelSize: kernelSize))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let channels = x.dim(-1)
        var input = x
        if pad > 0 {
            input = padded(input, widths: [IntOrPair(0), IntOrPair((pad, pad)), IntOrPair(0)], mode: .edge)
        }
        let weight = broadcast(filter.reshaped([1, kernelSize, 1]), to: [channels, kernelSize, 1])
        var y = Float(ratio) * convTransposed1d(input, weight, stride: ratio, groups: channels)
        if causal {
            let trim = kernelSize - ratio
            let keep = max(0, y.dim(1) - trim)
            y = y[0..., 0 ..< keep, 0...]
        } else {
            let end = y.dim(1) - padRight
            if padLeft < end {
                y = y[0..., padLeft ..< end, 0...]
            }
        }
        return y
    }
}

struct AuKDownSample1d {
    let ratio: Int
    let kernelSize: Int
    let padLeft: Int
    let padRight: Int
    let filter: MLXArray

    init(ratio: Int = 2, kernelSize: Int = 12, causal: Bool = false) {
        self.ratio = ratio
        self.kernelSize = kernelSize
        if causal {
            self.padLeft = kernelSize - 1
            self.padRight = 0
        } else {
            let even = kernelSize % 2 == 0
            self.padLeft = kernelSize / 2 - (even ? 1 : 0)
            self.padRight = kernelSize / 2
        }
        self.filter = MLXArray(aukKaiserSincFilter1d(cutoff: 0.5 / Float(ratio), halfWidth: 0.6 / Float(ratio), kernelSize: kernelSize))
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let channels = x.dim(-1)
        let input = padded(
            x,
            widths: [IntOrPair(0), IntOrPair((padLeft, padRight)), IntOrPair(0)],
            mode: .edge
        )
        let weight = broadcast(filter.reshaped([1, kernelSize, 1]), to: [channels, kernelSize, 1])
        return conv1d(input, weight, stride: ratio, groups: channels)
    }
}

final class AuKActivation1d: Module {
    @ModuleInfo var act: AuKSnakeBeta
    let upsample: AuKUpSample1d
    let downsample: AuKDownSample1d

    init(channels: Int, alphaLogscale: Bool = true, causal: Bool = false) {
        _act.wrappedValue = AuKSnakeBeta(channels: channels, alphaLogscale: alphaLogscale)
        self.upsample = AuKUpSample1d(ratio: 2, kernelSize: 12, causal: false)
        self.downsample = AuKDownSample1d(ratio: 2, kernelSize: 12, causal: causal)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        downsample(act(upsample(x)))
    }
}

func aukSilu(_ x: MLXArray) -> MLXArray {
    x * sigmoid(x)
}

func aukLeakyReLU(_ x: MLXArray, slope: Float) -> MLXArray {
    which(x .>= 0, x, x * slope)
}

func aukMish(_ x: MLXArray) -> MLXArray {
    x * tanh(logAddExp(x, MLXArray.zeros(x.shape)))
}
