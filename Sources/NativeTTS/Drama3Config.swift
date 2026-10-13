import Foundation

public enum Drama3Error: Error, LocalizedError, Equatable {
    case missingWeights(String)
    case unconvertedCheckpoint(String)
    case invalidConfig(String)
    case invalidInput(String)
    case conversionFailed(String)
    case generationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .missingWeights(let message): "Missing Drama 3 weights: \(message)"
        case .unconvertedCheckpoint(let message): "Drama 3 checkpoint is not converted to MLX: \(message)"
        case .invalidConfig(let message): "Invalid Drama 3 config: \(message)"
        case .invalidInput(let message): "Invalid Drama 3 input: \(message)"
        case .conversionFailed(let message): "Drama 3 conversion failed: \(message)"
        case .generationFailed(let message): "Drama 3 generation failed: \(message)"
        }
    }
}

/// Fish Audio Drama 3 (`model_type: fish_omni`): a Qwen3.5-MoE backbone that samples the main
/// (semantic) token of each audio frame plus a codebook tower that predicts the remaining codes.
public struct Drama3Config: Sendable, Equatable {
    public struct Backbone: Sendable, Equatable {
        public var hiddenSize: Int
        public var layerCount: Int
        public var vocabularySize: Int
        public var attentionHeads: Int
        public var kvHeads: Int
        public var headDim: Int
        public var rotaryDim: Int
        public var ropeTheta: Float
        public var rmsNormEps: Float
        public var fullAttentionInterval: Int
        public var linearKeyHeads: Int
        public var linearValueHeads: Int
        public var linearKeyHeadDim: Int
        public var linearValueHeadDim: Int
        public var linearConvKernel: Int
        public var expertCount: Int
        public var expertsPerToken: Int
        public var moeIntermediateSize: Int
        public var sharedExpertIntermediateSize: Int
        public var normTopKProbability: Bool

        public func isLinearLayer(_ index: Int) -> Bool {
            (index + 1) % fullAttentionInterval != 0
        }
    }

    public struct Tower: Sendable, Equatable {
        public var dim: Int
        public var heads: Int
        public var kvHeads: Int
        public var headDim: Int
        public var rotaryDim: Int
        public var ropeBase: Float
        public var intermediateSize: Int
        public var normEps: Float
        public var sharedLayerCount: Int
        public var codebookCount: Int
        public var codebookSize: Int
        public var backboneCodebookCount: Int
    }

    public var backbone: Backbone
    public var tower: Tower
    public var eosTokenID: Int
    public var semanticBeginID: Int
    public var semanticEndID: Int
    /// Present in converted directories: affine quantization applied to linear/embedding weights.
    public var quantizationBits: Int?
    public var quantizationGroupSize: Int

    public static let sampleRate = 44_100
    public static let samplesPerFrame = 2_048

    public static func load(directory: URL) throws -> Drama3Config {
        let url = directory.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: url) else {
            throw Drama3Error.missingWeights("config.json not found in \(directory.path)")
        }
        return try parse(data)
    }

    static func parse(_ data: Data) throws -> Drama3Config {
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Drama3Error.invalidConfig("config.json is not an object")
        }
        guard root["model_type"] as? String == "fish_omni" else {
            throw Drama3Error.invalidConfig("model_type must be fish_omni")
        }
        guard let text = root["text_config"] as? [String: Any],
              let decoder = root["audio_decoder_config"] as? [String: Any]
        else { throw Drama3Error.invalidConfig("text_config and audio_decoder_config are required") }

        func int(_ dict: [String: Any], _ key: String) throws -> Int {
            guard let value = dict[key] as? NSNumber else { throw Drama3Error.invalidConfig("\(key) missing") }
            return value.intValue
        }
        func float(_ dict: [String: Any], _ key: String) throws -> Float {
            guard let value = dict[key] as? NSNumber else { throw Drama3Error.invalidConfig("\(key) missing") }
            return value.floatValue
        }

        guard let layerTypes = text["layer_types"] as? [String],
              let firstFull = layerTypes.firstIndex(of: "full_attention")
        else { throw Drama3Error.invalidConfig("text_config.layer_types needs a full_attention layer") }
        let interval = firstFull + 1
        for (index, type) in layerTypes.enumerated() where (type == "full_attention") != ((index + 1) % interval == 0) {
            throw Drama3Error.invalidConfig("text_config.layer_types is not periodic")
        }
        let rope = text["rope_parameters"] as? [String: Any] ?? [:]
        let headDim = try int(text, "head_dim")
        let partial = (rope["partial_rotary_factor"] as? NSNumber)?.floatValue ?? 0.25
        guard text["attn_output_gate"] as? Bool ?? true, text["shared_expert_gate"] as? Bool ?? true else {
            throw Drama3Error.invalidConfig("backbone must use gated attention output and a gated shared expert")
        }
        let backbone = Backbone(
            hiddenSize: try int(text, "hidden_size"),
            layerCount: layerTypes.count,
            vocabularySize: try int(text, "vocab_size"),
            attentionHeads: try int(text, "num_attention_heads"),
            kvHeads: try int(text, "num_key_value_heads"),
            headDim: headDim,
            rotaryDim: Int(Float(headDim) * partial),
            ropeTheta: (rope["rope_theta"] as? NSNumber)?.floatValue ?? 10_000_000,
            rmsNormEps: try float(text, "rms_norm_eps"),
            fullAttentionInterval: interval,
            linearKeyHeads: try int(text, "linear_num_key_heads"),
            linearValueHeads: try int(text, "linear_num_value_heads"),
            linearKeyHeadDim: try int(text, "linear_key_head_dim"),
            linearValueHeadDim: try int(text, "linear_value_head_dim"),
            linearConvKernel: try int(text, "linear_conv_kernel_dim"),
            expertCount: try int(text, "num_experts"),
            expertsPerToken: try int(text, "num_experts_per_tok"),
            moeIntermediateSize: try int(text, "moe_intermediate_size"),
            sharedExpertIntermediateSize: try int(text, "shared_expert_intermediate_size"),
            normTopKProbability: text["norm_top_k_prob"] as? Bool ?? true
        )
        guard layerTypes.count == backbone.layerCount else {
            throw Drama3Error.invalidConfig("layer_types length mismatch")
        }

        let towerHeadDim = try int(decoder, "head_dim")
        let tower = Tower(
            dim: try int(decoder, "dim"),
            heads: try int(decoder, "n_head"),
            kvHeads: try int(decoder, "n_local_heads"),
            headDim: towerHeadDim,
            rotaryDim: Int(Float(towerHeadDim) * (try float(decoder, "rotary_percent"))),
            ropeBase: try float(decoder, "rope_base"),
            intermediateSize: try int(decoder, "intermediate_size"),
            normEps: try float(decoder, "norm_eps"),
            sharedLayerCount: try int(decoder, "shared_enrichment_layers"),
            codebookCount: try int(decoder, "num_codebooks"),
            codebookSize: try int(decoder, "vocab_size"),
            backboneCodebookCount: try int(decoder, "vq_backbone_codebooks")
        )
        guard decoder["layernorm_zero_centered_gamma"] as? Bool ?? true,
              decoder["decoder_type"] as? String == "codebook_mtp_shared",
              tower.backboneCodebookCount >= 0,
              tower.backboneCodebookCount < tower.codebookCount
        else { throw Drama3Error.invalidConfig("unsupported audio_decoder_config") }

        var bits: Int?
        var groupSize = 64
        if let quant = root["mlx_quantization"] as? [String: Any] {
            bits = (quant["bits"] as? NSNumber)?.intValue
            groupSize = (quant["group_size"] as? NSNumber)?.intValue ?? 64
        }
        let config = Drama3Config(
            backbone: backbone,
            tower: tower,
            eosTokenID: try int(root, "eos_token_id"),
            semanticBeginID: try int(root, "semantic_start_token_id"),
            semanticEndID: try int(root, "semantic_end_token_id"),
            quantizationBits: bits,
            quantizationGroupSize: groupSize
        )
        guard config.semanticEndID - config.semanticBeginID + 1 == tower.codebookSize else {
            throw Drama3Error.invalidConfig("semantic token band must span the codebook size")
        }
        return config
    }
}

public enum Drama3ModelLayout: Equatable, Sendable {
    case converted
    case officialSource
    case unknown
}

public func inspectDrama3ModelDirectory(_ url: URL) -> Drama3ModelLayout {
    let fm = FileManager.default
    guard let data = try? Data(contentsOf: url.appendingPathComponent("config.json")),
          let root = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
          root["model_type"] as? String == "fish_omni"
    else { return .unknown }
    if fm.fileExists(atPath: url.appendingPathComponent(Drama3Convert.codecFileName).path),
       fm.fileExists(atPath: url.appendingPathComponent(Drama3Convert.weightIndexFileName).path) {
        return .converted
    }
    if fm.fileExists(atPath: url.appendingPathComponent("codec.pth").path) {
        return .officialSource
    }
    return .unknown
}
