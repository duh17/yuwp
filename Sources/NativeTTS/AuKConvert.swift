import Foundation
@preconcurrency import MLX
import MLXNN

public struct AuKConvertResult: Sendable {
    public var ditTensors: Int
    public var vaeTensors: Int
    public var thinkerTensors: Int
    public var outputDirectory: URL
}

public enum AuKConvert {
    /// Tensor counts from a complete official AuK-Flash conversion onto the native graphs.
    public static let expectedDiTTensorCount = 417
    public static let expectedThinkerTensorCount = 922

    static func foldWeightNormParameters(_ sd: [String: MLXArray]) throws -> [String: MLXArray] {
        var folded: [String: MLXArray] = [:]
        var consumed = Set<String>()
        for key in sd.keys where key.hasSuffix(".weight_g") {
            let base = String(key.dropLast(".weight_g".count))
            let vKey = base + ".weight_v"
            guard let g = sd[key], let v = sd[vKey] else {
                throw AuKError.conversionFailed("weight-norm pair missing for \(base)")
            }
            folded[base + ".weight"] = foldWeightNorm(g: g, v: v)
            consumed.insert(key)
            consumed.insert(vKey)
        }
        for (key, value) in sd {
            if consumed.contains(key) { continue }
            if key.hasSuffix(".weight_v") { continue }
            if folded[key] == nil {
                folded[key] = value
            }
        }
        return folded
    }

    static func convertVAE(_ src: [String: MLXArray]) throws -> [String: MLXArray] {
        let folded = try foldWeightNormParameters(src)
        var out: [String: MLXArray] = [:]
        for (key, value) in folded {
            guard let mapped = remapVAEKey(key) else { continue }
            if mapped.hasSuffix(".weight"), isVAEConvWeight(mapped) {
                if mapped.contains(".ups.") {
                    out[mapped] = transposeConvTranspose1dWeight(value)
                } else {
                    out[mapped] = transposeConv1dWeight(value)
                }
            } else {
                out[mapped] = value.asType(.float32)
            }
        }
        try aukRequireConvertedGraph(
            fileKeys: Set(out.keys),
            requiredKeys: [
                "audio_encoder.pre.weight",
                "audio_encoder.post.weight",
                "decoder.conv_pre.weight",
                "decoder.conv_post.weight",
                "global_mean",
                "global_log_std",
            ],
            expectedCount: aukExpectedVAETensorCount(),
            label: "VAE"
        )
        return out
    }

    static func convertDiT(_ src: [String: MLXArray]) throws -> (dit: [String: MLXArray], fusion: [String: MLXArray]) {
        var dit: [String: MLXArray] = [:]
        var fusion: [String: MLXArray] = [:]
        for (key, value) in src {
            switch remapDiTKey(key) {
            case .skip:
                continue
            case .fusion(let name):
                fusion[name] = value.asType(.float32)
            case .dit(let name):
                if name.contains(".conv1d."), name.hasSuffix(".weight") {
                    dit[name] = transposeConv1dWeight(value)
                } else {
                    dit[name] = value.asType(.float32)
                }
            case .thinker:
                continue
            }
        }
        guard fusion["inv_freq"] != nil, fusion["layer_weights"] != nil, fusion["layer_scale"] != nil else {
            throw AuKError.conversionFailed("DiT conversion missed fusion tensors (inv_freq/layer_weights/layer_scale)")
        }
        try aukRequireConvertedGraph(
            fileKeys: Set(dit.keys),
            requiredKeys: [
                "txt_proj.weight",
                "audio_embed.linear.weight",
                "transformer_blocks.0.attn.to_qkv.weight",
                "single_transformer_blocks.0.attn.to_qkv.weight",
                "proj_out.weight",
            ],
            expectedCount: AuKConvert.expectedDiTTensorCount,
            label: "DiT"
        )
        return (dit, fusion)
    }

    static func convertThinker(_ src: [String: MLXArray]) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in src {
            switch remapThinkerKey(key) {
            case .thinker(let name):
                if name.hasPrefix("audio_tower.conv"), name.hasSuffix(".weight") {
                    out[name] = transposeConv1dWeight(value)
                } else {
                    out[name] = value.asType(.float32)
                }
            default:
                continue
            }
        }
        try aukRequireConvertedGraph(
            fileKeys: Set(out.keys),
            requiredKeys: [
                "embed_tokens.weight",
                "norm.weight",
                "layers.0.self_attn.q_proj.weight",
                "audio_tower.conv1.weight",
                "audio_tower.proj.weight",
            ],
            expectedCount: AuKConvert.expectedThinkerTensorCount,
            label: "Thinker"
        )
        return out
    }

    public static func convertDirectory(
        source: URL,
        thinkerSource: URL,
        output: URL,
        bits: Int? = nil
    ) throws -> AuKConvertResult {
        let fm = FileManager.default
        try fm.createDirectory(at: output, withIntermediateDirectories: true)

        let vaeSrc = source.appendingPathComponent("vae.safetensors")
        let ditSrc = source.appendingPathComponent("auk_flash.safetensors")
        guard fm.fileExists(atPath: vaeSrc.path) else {
            throw AuKError.missingWeights("vae.safetensors not found in \(source.path)")
        }
        guard fm.fileExists(atPath: ditSrc.path) else {
            throw AuKError.missingWeights("auk_flash.safetensors not found in \(source.path)")
        }

        let vaeOut = try convertVAE(try MLX.loadArrays(url: vaeSrc))
        try aukProveCompleteLoad(AuKBigVGANFlowVAE(), weights: vaeOut, label: "VAE")
        try MLX.save(arrays: vaeOut, url: output.appendingPathComponent("vae.safetensors"))

        let ditConverted = try convertDiT(try MLX.loadArrays(url: ditSrc))
        guard let inv = ditConverted.fusion["inv_freq"] else {
            throw AuKError.conversionFailed("DiT fusion missing inv_freq")
        }
        eval(inv)
        try aukProveCompleteLoad(AuKFlux2Edit(invFreq: inv.asArray(Float.self)), weights: ditConverted.dit, label: "DiT")
        try MLX.save(arrays: ditConverted.dit, url: output.appendingPathComponent("dit_flash.safetensors"))
        try MLX.save(arrays: ditConverted.fusion, url: output.appendingPathComponent("fusion_flash.safetensors"))

        var thinkerWeights: [String: MLXArray] = [:]
        let thinkerFiles = try fm.contentsOfDirectory(at: thinkerSource, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        if thinkerFiles.isEmpty {
            throw AuKError.missingWeights("no safetensors found in thinker source \(thinkerSource.path)")
        }
        for file in thinkerFiles {
            let shard = try MLX.loadArrays(url: file)
            for (key, value) in shard {
                switch remapThinkerKey(key) {
                case .thinker:
                    thinkerWeights[key] = value
                default:
                    continue
                }
            }
        }
        let thinkerOut = try convertThinker(thinkerWeights)
        try aukProveCompleteLoad(AuKThinkerEncoder(), weights: thinkerOut, label: "Thinker")
        let thinkerDir = output.appendingPathComponent("thinker")
        try fm.createDirectory(at: thinkerDir, withIntermediateDirectories: true)
        try MLX.save(arrays: thinkerOut, url: thinkerDir.appendingPathComponent("thinker.safetensors"))
        let thinkerConfig: [String: Any] = [
            "text": [
                "hidden_size": AuKFlashConfig.thinkerHidden,
                "num_hidden_layers": AuKFlashConfig.thinkerLayers,
                "num_attention_heads": AuKFlashConfig.thinkerHeads,
                "num_key_value_heads": AuKFlashConfig.thinkerKVHeads,
                "intermediate_size": AuKFlashConfig.thinkerIntermediate,
                "vocab_size": AuKFlashConfig.thinkerVocab,
                "rope_theta": AuKFlashConfig.thinkerRopeTheta,
                "rms_norm_eps": AuKFlashConfig.thinkerRMSNormEps,
            ],
            "audio": [
                "d_model": AuKFlashConfig.audioDModel,
                "encoder_layers": AuKFlashConfig.audioLayers,
                "encoder_attention_heads": AuKFlashConfig.audioHeads,
                "encoder_ffn_dim": AuKFlashConfig.audioFFN,
                "output_dim": AuKFlashConfig.audioOutputDim,
                "n_window": AuKFlashConfig.audioWindow,
                "num_mel_bins": AuKFlashConfig.audioMelBins,
                "scale_embedding": false,
            ],
        ]
        let thinkerConfigData = try JSONSerialization.data(withJSONObject: thinkerConfig, options: [.prettyPrinted, .sortedKeys])
        try thinkerConfigData.write(to: thinkerDir.appendingPathComponent("thinker_config.json"))

        try copyProcessorFiles(from: thinkerSource, to: output)
        let yamlSrc = source.appendingPathComponent("config.yaml")
        if fm.fileExists(atPath: yamlSrc.path) {
            try copyReplacing(yamlSrc, output.appendingPathComponent("config.yaml"))
        }
        let meta: [String: Any] = [
            "name": AuKFlashConfig.name,
            "variant": AuKFlashConfig.variant,
            "sample_rate": AuKFlashConfig.sampleRate,
            "downsample_rate": AuKFlashConfig.downsampleRate,
            "latent_dim": AuKFlashConfig.latentDim,
        ]
        let metaData = try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
        try metaData.write(to: output.appendingPathComponent("auk_config.json"))

        if let bits, bits == 4 || bits == 8 {
            try quantizeConvertedWeights(at: output, bits: bits, fusion: ditConverted.fusion)
        }

        return AuKConvertResult(
            ditTensors: ditConverted.dit.count,
            vaeTensors: vaeOut.count,
            thinkerTensors: thinkerOut.count,
            outputDirectory: output
        )
    }

    private static func isVAEConvWeight(_ mapped: String) -> Bool {
        if mapped.hasSuffix(".act.alpha") || mapped.hasSuffix(".act.beta") { return false }
        if mapped == "global_mean" || mapped == "global_log_std" { return false }
        return mapped.hasSuffix(".weight")
    }

    private static let requiredProcessorFiles = [
        "tokenizer.json",
        "tokenizer_config.json",
        "config.json",
    ]
    private static let optionalProcessorFiles = [
        "vocab.json",
        "merges.txt",
        "preprocessor_config.json",
        "chat_template.json",
        "added_tokens.json",
        "special_tokens_map.json",
    ]

    private static func copyProcessorFiles(from source: URL, to dest: URL) throws {
        for name in requiredProcessorFiles {
            let src = source.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: src.path) else {
                throw AuKError.conversionFailed("missing required processor file \(name) in \(source.path)")
            }
            try copyReplacing(src, dest.appendingPathComponent(name))
        }
        for name in optionalProcessorFiles {
            let src = source.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: src.path) {
                try copyReplacing(src, dest.appendingPathComponent(name))
            }
        }
    }

    private static func quantizeConvertedWeights(at output: URL, bits: Int, fusion: [String: MLXArray]) throws {
        guard let inv = fusion["inv_freq"] else {
            throw AuKError.conversionFailed("missing inv_freq for quantization")
        }
        eval(inv)
        let invFreq = inv.asArray(Float.self)
        let dit = AuKFlux2Edit(invFreq: invFreq)
        try aukProveCompleteLoad(
            dit,
            weights: try MLX.loadArrays(url: output.appendingPathComponent("dit_flash.safetensors")),
            label: "DiT"
        )
        quantize(model: dit, groupSize: 64, bits: bits)
        try MLX.save(
            arrays: flattenedParameterDict(dit.parameters()),
            url: output.appendingPathComponent("dit_flash.q\(bits).safetensors")
        )

        let thinker = AuKThinkerEncoder()
        try aukProveCompleteLoad(
            thinker,
            weights: try MLX.loadArrays(url: output.appendingPathComponent("thinker/thinker.safetensors")),
            label: "Thinker"
        )
        quantize(model: thinker, groupSize: 64, bits: bits)
        try MLX.save(
            arrays: flattenedParameterDict(thinker.parameters()),
            url: output.appendingPathComponent("thinker/thinker.q\(bits).safetensors")
        )
    }
}

func flattenedParameterDict(_ parameters: ModuleParameters) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    for (key, value) in parameters.flattened() {
        out[key] = value
    }
    return out
}

func aukProveCompleteLoad(_ model: Module, weights: [String: MLXArray], label: String) throws {
    let modelKeys = Set(model.parameters().flattened().map(\.0))
    try aukRequireExactKeys(fileKeys: Set(weights.keys), modelKeys: modelKeys, label: label)
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
}

func copyReplacing(_ source: URL, _ destination: URL) throws {
    let fm = FileManager.default
    if fm.fileExists(atPath: destination.path) {
        try fm.removeItem(at: destination)
    }
    do {
        try fm.copyItem(at: source, to: destination)
    } catch {
        throw AuKError.conversionFailed("failed to copy \(source.lastPathComponent): \(error.localizedDescription)")
    }
}
