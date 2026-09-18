import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom
import Tokenizers

public final class AuKEngine: @unchecked Sendable {
    public let sampleRate = AuKFlashConfig.sampleRate
    public let modelName = AuKFlashConfig.name

    private let vae: AuKBigVGANFlowVAE
    private let dit: AuKFlux2Edit
    private let thinker: AuKThinkerEncoder
    private let tokenizer: any Tokenizer
    private let layerWeights: MLXArray
    private let layerScale: MLXArray
    private let lock = NSLock()

    init(
        vae: AuKBigVGANFlowVAE,
        dit: AuKFlux2Edit,
        thinker: AuKThinkerEncoder,
        tokenizer: any Tokenizer,
        layerWeights: MLXArray,
        layerScale: MLXArray
    ) {
        self.vae = vae
        self.dit = dit
        self.thinker = thinker
        self.tokenizer = tokenizer
        self.layerWeights = layerWeights
        self.layerScale = layerScale
    }

    public static func load(
        modelDirectory: URL,
        thinkerDirectory: URL? = nil,
        bits: Int? = nil
    ) async throws -> AuKEngine {
        switch inspectAuKModelDirectory(modelDirectory) {
        case .pytorchSource:
            throw AuKError.unconvertedCheckpoint(
                "\(modelDirectory.path) looks like official PyTorch AuK-Flash weights. Convert once with `yuwp-tts convert-auk --src <AuK-Flash> --thinker-src <Qwen2.5-Omni-3B> --out <mlx-dir>`."
            )
        case .unknown:
            throw AuKError.missingWeights("No AuK-Flash MLX weights found in \(modelDirectory.path)")
        case .converted:
            break
        }

        let fusionURL = modelDirectory.appendingPathComponent("fusion_flash.safetensors")
        guard FileManager.default.fileExists(atPath: fusionURL.path) else {
            throw AuKError.missingWeights("fusion_flash.safetensors is required")
        }
        let fusion = try MLX.loadArrays(url: fusionURL)
        guard let inv = fusion["inv_freq"], let layerWeights = fusion["layer_weights"], let layerScale = fusion["layer_scale"] else {
            throw AuKError.missingWeights("fusion_flash.safetensors must contain inv_freq, layer_weights, layer_scale")
        }
        eval(inv)
        let invFreq = inv.asArray(Float.self)

        let vae = AuKBigVGANFlowVAE()
        let vaeWeights = try MLX.loadArrays(url: modelDirectory.appendingPathComponent("vae.safetensors"))
        try aukProveCompleteLoad(vae, weights: vaeWeights, label: "VAE")
        vae.train(false)
        eval(vae.parameters())

        let dit = AuKFlux2Edit(invFreq: invFreq)
        try loadQuantizable(
            dit,
            directory: modelDirectory,
            baseName: "dit_flash",
            bits: bits
        )

        let thinker = AuKThinkerEncoder()
        let thinkerDir = modelDirectory.appendingPathComponent("thinker")
        try loadQuantizable(
            thinker,
            directory: thinkerDir,
            baseName: "thinker",
            bits: bits
        )

        let tokenizerDir = tokenizerDirectory(modelDirectory: modelDirectory, thinkerDirectory: thinkerDirectory)
        let tokenizer: any Tokenizer
        do {
            tokenizer = try await AutoTokenizer.from(modelFolder: tokenizerDir)
        } catch {
            throw AuKError.tokenizerFailed("Could not load Qwen2.5-Omni tokenizer from \(tokenizerDir.path): \(error.localizedDescription)")
        }

        return AuKEngine(
            vae: vae,
            dit: dit,
            thinker: thinker,
            tokenizer: tokenizer,
            layerWeights: layerWeights.asType(.float32),
            layerScale: layerScale.asType(.float32)
        )
    }

    public func generate(
        instruction: String,
        referenceAudioURL: URL?,
        genSeconds: Double?,
        seed: UInt64?
    ) throws -> (samples: [Float], sampleRate: Int) {
        lock.lock()
        defer { lock.unlock() }

        let resolvedSeconds = try aukResolveGenSeconds(genSeconds, hasReferenceAudio: referenceAudioURL != nil)

        var refLatent = MLXArray.zeros([1, 0, AuKFlashConfig.latentDim])
        var refLen = 0
        if let referenceAudioURL {
            let wav = try AuKProcessor.loadMonoWaveform(
                from: referenceAudioURL,
                targetSampleRate: AuKFlashConfig.sampleRate
            )
            let wavArray = MLXArray(wav).reshaped([1, wav.count, 1])
            refLatent = vae.encode(wavArray)
            eval(refLatent)
            refLen = refLatent.dim(1)
        }

        let genLen: Int
        if let resolvedSeconds {
            genLen = AuKFlashConfig.latentFrames(seconds: resolvedSeconds)
        } else {
            genLen = max(1, refLen)
        }

        let prompt = try AuKProcessor.encode(
            instruction: instruction,
            tokenizer: tokenizer,
            referenceAudioURL: referenceAudioURL
        )
        let hidden = try thinker(
            inputIds: prompt.inputIds,
            audioFeatures: prompt.audioFeatures,
            audioTokenMask: prompt.audioTokenMask,
            audioFeatureLen: prompt.audioFeatureLen
        )
        let textEmbed = fuseLayers(hidden)
        eval(textEmbed)

        if let seed {
            MLXRandom.seed(seed)
        }
        let genLatent = sample(textEmbed: textEmbed, refLatent: refLatent, genLen: genLen)
        eval(genLatent)
        if !isFinite(genLatent).all().item(Bool.self) {
            throw AuKError.generationFailed("generated latent contains NaN/Inf")
        }

        let wav = vae.decode(genLatent)
        eval(wav)
        let samples = wav.reshaped([-1]).asArray(Float.self)
        if samples.contains(where: { !$0.isFinite }) {
            throw AuKError.generationFailed("generated audio contains NaN/Inf")
        }
        return (samples, sampleRate)
    }

    private func fuseLayers(_ hidden: [MLXArray]) -> MLXArray {
        let layers = hidden.dropFirst().map { aukLayerNorm($0, eps: 1e-5) }
        let stacked = stacked(Array(layers), axis: 0)
        let weights = softmax(layerWeights, axis: 0)
        return (stacked * weights.reshaped([-1, 1, 1, 1])).sum(axis: 0) * layerScale
    }

    private func sample(textEmbed: MLXArray, refLatent: MLXArray, genLen: Int) -> MLXArray {
        let resolved = AuKSampling.resolve(nfe: 4, cfgStrength: 0, sway: nil, tGrid: nil)
        let t = MLXArray(resolved.tGrid)
        var y = MLXRandom.normal([1, genLen, AuKFlashConfig.latentDim])
        dit.clearCache()
        let steps = resolved.tGrid.count - 1
        for i in 0 ..< steps {
            let ti = t[i ..< (i + 1)]
            let dt = t[i + 1] - t[i]
            let v = dit(x: y, text: textEmbed, t: ti, ref: refLatent, cfgInfer: false, cache: true)
            y = y + v * dt
            eval(y)
        }
        dit.clearCache()
        return y
    }

    private static func tokenizerDirectory(modelDirectory: URL, thinkerDirectory: URL?) -> URL {
        let local = modelDirectory.appendingPathComponent("tokenizer.json")
        if FileManager.default.fileExists(atPath: local.path) {
            return modelDirectory
        }
        if let thinkerDirectory {
            return thinkerDirectory
        }
        return modelDirectory
    }

    private static func loadQuantizable(
        _ model: Module,
        directory: URL,
        baseName: String,
        bits: Int?
    ) throws {
        let resolved = resolveAuKWeightFile(directory: directory, baseName: baseName, bits: bits)
        guard FileManager.default.fileExists(atPath: resolved.url.path) else {
            throw AuKError.missingWeights("\(resolved.url.lastPathComponent) not found in \(directory.path)")
        }
        let weights = try MLX.loadArrays(url: resolved.url)
        let quantizedOnDisk = weights.keys.contains { $0.contains(".scales") }
        if quantizedOnDisk {
            guard let quantBits = resolved.bits, quantBits == 4 || quantBits == 8 else {
                throw AuKError.missingWeights("refusing to auto-load quantized \(resolved.url.lastPathComponent); pass --bits \(quantBitsForFile(weights)) or use fp32 weights")
            }
            quantize(model: model, groupSize: 64, bits: quantBits) { path, _ in
                weights["\(path).scales"] != nil
            }
        }
        try aukProveCompleteLoad(model, weights: weights, label: baseName)
        if !quantizedOnDisk, let bits, bits == 4 || bits == 8 {
            quantize(model: model, groupSize: 64, bits: bits)
        }
        model.train(false)
        eval(model.parameters())
    }
}

private func quantBitsForFile(_ weights: [String: MLXArray]) -> Int {
    weights.keys.contains(where: { $0.contains(".q4") }) ? 4 : 8
}
