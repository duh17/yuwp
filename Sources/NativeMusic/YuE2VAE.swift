import Foundation
@preconcurrency import MLX

/// YuE2-Vae Oobleck decoder: 64-dim latents at 25 Hz → 48 kHz stereo, FP32, channels-last.
/// Weight norm is folded at load; tiles decode with a halo wider than the receptive field.
public final class YuE2VAEDecoder: @unchecked Sendable {
    struct Config: Decodable {
        let channels: Int
        let cMults: [Int]
        let strides: [Int]
        let latentDim: Int
        let outChannels: Int
        let useSnake: Bool
        let finalTanh: Bool
        let snakeType: String?

        enum CodingKeys: String, CodingKey {
            case channels, strides
            case cMults = "c_mults"
            case latentDim = "latent_dim"
            case outChannels = "out_channels"
            case useSnake = "use_snake"
            case finalTanh = "final_tanh"
            case snakeType = "snake_type"
        }
    }

    private struct Snake {
        let scale: MLXArray // exp(alpha)
        let inverseBeta: MLXArray // 1 / (exp(beta) + 1e-9)

        static let kernel = compile(shapeless: true) { (x: MLXArray, scale: MLXArray, inverseBeta: MLXArray) -> MLXArray in
            x + square(sin(x * scale)) * inverseBeta
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray { Self.kernel(x, scale, inverseBeta) }
    }

    private struct Conv {
        let weight: MLXArray // [out, k, in]
        let bias: MLXArray?
        let padding: Int
        let dilation: Int

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let y = conv1d(x, weight, stride: 1, padding: padding, dilation: dilation)
            return bias.map { y + $0 } ?? y
        }
    }

    private struct ConvTranspose {
        let weight: MLXArray // [out, k, in]
        let bias: MLXArray
        let stride: Int

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            convTransposed1d(x, weight, stride: stride, padding: (stride + 1) / 2) + bias
        }
    }

    private struct ResidualUnit {
        let snake1: Snake
        let conv1: Conv
        let snake2: Snake
        let conv2: Conv

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            x + conv2(snake2(conv1(snake1(x))))
        }
    }

    private struct Block {
        let snake: Snake
        let up: ConvTranspose
        let units: [ResidualUnit]

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            units.reduce(up(snake(x))) { $1($0) }
        }
    }

    private let input: Conv
    private let blocks: [Block]
    private let outSnake: Snake
    private let output: Conv
    private let finalTanh: Bool
    /// Applied in order (largest stride first).
    let strides: [Int]
    public let latentDim: Int

    public init(directory: URL) throws {
        let configData = try Data(contentsOf: directory.appendingPathComponent("config.json"))
        struct Root: Decodable { let decoderConfig: Config
            enum CodingKeys: String, CodingKey { case decoderConfig = "decoder_config" }
        }
        let config = try JSONDecoder().decode(Root.self, from: configData).decoderConfig
        guard config.useSnake, (config.snakeType ?? "vanilla") == "vanilla" else {
            throw YuE2Error.invalidConfig("only vanilla SnakeBeta Oobleck decoders are supported")
        }
        var raw = try MLX.loadArrays(url: directory.appendingPathComponent("model.safetensors"))
        raw = raw.filter { $0.key.hasPrefix("decoder.") }

        func tensor(_ key: String) throws -> MLXArray {
            guard let value = raw["decoder." + key] else { throw YuE2Error.missingWeights("VAE decoder.\(key)") }
            return value.asType(.float32)
        }
        /// PyTorch weight_norm(dim=0): w = g * v / ||v|| over all but dim 0.
        func folded(_ key: String, transposed: Bool) throws -> MLXArray {
            let v = try tensor(key + ".weight_v")
            let g = try tensor(key + ".weight_g")
            let w = v * (g / sqrt(sum(v * v, axes: [1, 2], keepDims: true)))
            // Conv [out, in, k] → [out, k, in]; ConvTranspose [in, out, k] → [out, k, in].
            return transposed ? w.transposed(1, 2, 0) : w.transposed(0, 2, 1)
        }
        func snake(_ key: String) throws -> Snake {
            Snake(scale: exp(try tensor(key + ".alpha")), inverseBeta: 1 / (exp(try tensor(key + ".beta")) + 1e-9))
        }
        func conv(_ key: String, padding: Int, dilation: Int = 1, bias: Bool = true) throws -> Conv {
            Conv(weight: try folded(key, transposed: false), bias: bias ? try tensor(key + ".bias") : nil, padding: padding, dilation: dilation)
        }

        let mults = [1] + config.cMults
        input = try conv("layers.0", padding: 3)
        var blocks: [Block] = []
        var strides: [Int] = []
        for (n, i) in stride(from: mults.count - 1, to: 0, by: -1).enumerated() {
            let p = "layers.\(n + 1).layers"
            let s = config.strides[i - 1]
            let units = try [1, 3, 9].enumerated().map { j, d in
                ResidualUnit(
                    snake1: try snake("\(p).\(j + 2).layers.0"),
                    conv1: try conv("\(p).\(j + 2).layers.1", padding: 3 * d, dilation: d),
                    snake2: try snake("\(p).\(j + 2).layers.2"),
                    conv2: try conv("\(p).\(j + 2).layers.3", padding: 0)
                )
            }
            blocks.append(Block(
                snake: try snake("\(p).0"),
                up: ConvTranspose(weight: try folded("\(p).1", transposed: true), bias: try tensor("\(p).1.bias"), stride: s),
                units: units
            ))
            strides.append(s)
        }
        self.blocks = blocks
        self.strides = strides
        outSnake = try snake("layers.\(blocks.count + 1)")
        output = try conv("layers.\(blocks.count + 2)", padding: 3, bias: false)
        finalTanh = config.finalTanh
        latentDim = config.latentDim
        eval(input.weight, blocks.flatMap { [$0.up.weight] + $0.units.flatMap { [$0.conv1.weight, $0.conv2.weight] } }, output.weight)
    }

    /// Samples produced for `frames` latent frames (odd strides drop one sample each).
    public func outputLength(frames: Int) -> Int {
        strides.reduce(frames) { $0 * $1 - $1 % 2 }
    }

    private func forward(_ z: MLXArray) -> MLXArray {
        var x = input(z)
        for block in blocks { x = block(x) }
        x = output(outSnake(x))
        return finalTanh ? tanh(x) : x
    }

    /// `[T, 64]` latents → `[samples, 2]` FP32 audio, decoded in `coreFrames` tiles with a 16-frame halo.
    public func decode(_ latents: MLXArray, coreFrames: Int = 256, haloFrames: Int = 16, onProgress: ((Int, Int) -> Void)? = nil) throws -> MLXArray {
        guard latents.ndim == 2, latents.dim(1) == latentDim, latents.dim(0) >= 1 else {
            throw YuE2Error.invalidInput("expected latents [T, \(latentDim)]")
        }
        guard coreFrames >= 1, haloFrames >= 16 else { throw YuE2Error.invalidInput("need coreFrames >= 1 and haloFrames >= 16") }
        let frames = latents.dim(0)
        let ratio = strides.reduce(1, *)
        let total = outputLength(frames: frames)
        let z = latents.asType(.float32)
        var pieces: [MLXArray] = []
        let count = (frames + coreFrames - 1) / coreFrames
        for (index, start) in stride(from: 0, to: frames, by: coreFrames).enumerated() {
            let end = min(frames, start + coreFrames)
            let left = max(0, start - haloFrames)
            let right = min(frames, end + haloFrames)
            let tile = forward(z[left ..< right].expandedDimensions(axis: 0))[0]
            let offset = (start - left) * ratio
            let length = min(end * ratio, total) - start * ratio
            let piece = tile[offset ..< offset + length]
            eval(piece)
            pieces.append(piece)
            onProgress?(index + 1, count)
        }
        let audio = concatenated(pieces, axis: 0)
        guard audio.dim(0) == total, audio.dim(1) == 2 else { throw YuE2Error.numerical("VAE produced \(audio.shape), expected [\(total), 2]") }
        return audio
    }
}
