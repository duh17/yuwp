import Foundation
@preconcurrency import MLX

public struct Drama3ConvertResult: Sendable {
    public var tensorCount: Int
    public var codecTensorCount: Int
    public var outputDirectory: URL
}

/// Converts the official `fishaudio/drama-3` checkpoint (block-FP8 experts, bf16 elsewhere, PyTorch
/// `codec.pth`) into an MLX directory the Swift runtime loads directly.
///
/// - FP8 E4M3 experts are dequantized with their 128×128 `weight_scale_inv` blocks and stacked per layer.
/// - Zero-centered RMSNorm gammas are shifted by +1 so the runtime uses plain `RMSNorm`.
/// - Optional affine quantization (`bits` 4 or 8, group 64) of linear and embedding weights;
///   MoE routers stay 8-bit.
/// - The codec has its weight norm folded and its conv weights moved to MLX layout.
public enum Drama3Convert {
    public static let codecFileName = "codec.safetensors"
    public static let weightIndexFileName = "drama3-mlx.safetensors.index.json"
    static let shardBytes = 5 << 30
    static let fp8BlockSize = 128
    static let quantGroupSize = 64

    public static func convertDirectory(source: URL, output: URL, bits: Int?) throws -> Drama3ConvertResult {
        if let bits, bits != 4 && bits != 8 {
            throw Drama3Error.invalidInput("--bits must be 4 or 8")
        }
        guard inspectDrama3ModelDirectory(source) == .officialSource else {
            throw Drama3Error.missingWeights("\(source.path) is not an official Drama 3 checkpoint (config.json + codec.pth)")
        }
        let config = try Drama3Config.load(directory: source)
        let fm = FileManager.default
        try fm.createDirectory(at: output, withIntermediateDirectories: true)

        let sourceShards = try sourceShardFiles(source)
        var sourceTensors: [String: MLXArray] = [:]
        for shard in sourceShards {
            for (key, value) in try MLX.loadArrays(url: shard) {
                sourceTensors[key] = value
            }
        }

        var writer = ShardWriter(directory: output)
        var consumed = Set<String>()

        // Non-expert tensors first, in a stable order.
        for key in sourceTensors.keys.sorted() where !key.contains(".mlp.experts.") {
            guard let value = sourceTensors[key], let mapped = try mapKey(key) else { continue }
            consumed.insert(key)
            let transformed = try transform(mapped: mapped, value: value)
            try writer.add(quantizedTensors(name: mapped, value: transformed, bits: bits))
        }

        for layer in 0 ..< config.backbone.layerCount {
            let prefix = "model.language_model.layers.\(layer).mlp.experts"
            for projection in ["gate_proj", "up_proj", "down_proj"] {
                var experts: [MLXArray] = []
                experts.reserveCapacity(config.backbone.expertCount)
                for expert in 0 ..< config.backbone.expertCount {
                    let weightKey = "\(prefix).\(expert).\(projection).weight"
                    let scaleKey = "\(prefix).\(expert).\(projection).weight_scale_inv"
                    guard let weight = sourceTensors[weightKey], let scale = sourceTensors[scaleKey] else {
                        throw Drama3Error.conversionFailed("missing \(weightKey) or its scale")
                    }
                    experts.append(dequantizeFP8(weight, scaleInverse: scale))
                    consumed.insert(weightKey)
                    consumed.insert(scaleKey)
                }
                let stacked = stacked(experts)
                eval(stacked)
                let name = "backbone.layers.\(layer).mlp.switch_mlp.\(projection).weight"
                try writer.add(quantizedTensors(name: name, value: stacked, bits: bits))
            }
            fputs("Drama 3: converted experts of layer \(layer + 1)/\(config.backbone.layerCount)\n", stderr)
        }

        let leftovers = Set(sourceTensors.keys).subtracting(consumed)
        guard leftovers.isEmpty else {
            throw Drama3Error.conversionFailed("unmapped source tensors: \(leftovers.sorted().prefix(8))")
        }
        let weightMap = try writer.finish(indexName: weightIndexFileName)

        let codec = try convertCodec(TorchCheckpointReader.load(url: source.appendingPathComponent("codec.pth")))
        try MLX.save(arrays: codec, url: output.appendingPathComponent(codecFileName))

        for file in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja", "LICENSE"] {
            let from = source.appendingPathComponent(file)
            let to = output.appendingPathComponent(file)
            guard fm.fileExists(atPath: from.path) else { continue }
            if fm.fileExists(atPath: to.path) { try fm.removeItem(at: to) }
            try fm.copyItem(at: from.resolvingSymlinksInPath(), to: to)
        }
        try writeConfig(source: source, output: output, bits: bits)

        return Drama3ConvertResult(tensorCount: weightMap, codecTensorCount: codec.count, outputDirectory: output)
    }

    static func sourceShardFiles(_ source: URL) throws -> [URL] {
        let indexURL = source.appendingPathComponent("model.safetensors.index.json")
        guard let data = try? Data(contentsOf: indexURL),
              let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = root["weight_map"] as? [String: String]
        else { throw Drama3Error.missingWeights("model.safetensors.index.json") }
        return Set(weightMap.values).sorted().map { source.appendingPathComponent($0) }
    }

    // MARK: Key mapping

    static let zeroCenteredNormSuffixes = [
        ".input_layernorm.weight", ".post_attention_layernorm.weight", ".q_norm.weight", ".k_norm.weight",
        ".enorm.weight", ".hnorm.weight", ".final_layernorm.weight", ".attention_norm.weight", ".ffn_norm.weight",
    ]

    /// Maps an official checkpoint key to its converted name; nil skips it.
    static func mapKey(_ key: String) throws -> String? {
        if key == "lm_head.weight" { return key }
        if key.hasPrefix("model.language_model.") {
            return "backbone." + key.dropFirst("model.language_model.".count)
        }
        if key == "audio_decoder.codebook_embeddings.weight" {
            return "tower.codebook_embeddings.weight"
        }
        let towerPrefixes = [
            ("audio_decoder_mtp_shared.layers.", "tower.shared."),
            ("audio_decoder_mtp.layers.", "tower.residual."),
            ("audio_decoder_mtp.heads.", "tower.heads."),
        ]
        for (from, to) in towerPrefixes where key.hasPrefix(from) {
            var name = to + key.dropFirst(from.count)
            name = name.replacingOccurrences(of: ".semantic_embedding.", with: ".token_embedding.")
            name = name.replacingOccurrences(of: ".codebook_embedding.", with: ".token_embedding.")
            name = name.replacingOccurrences(of: ".transformer_layer.", with: ".")
            return name
        }
        throw Drama3Error.conversionFailed("unexpected checkpoint key \(key)")
    }

    static func isZeroCenteredNorm(_ name: String) -> Bool {
        if name == "backbone.norm.weight" { return true }
        return zeroCenteredNormSuffixes.contains { name.hasSuffix($0) }
    }

    static func transform(mapped: String, value: MLXArray) throws -> MLXArray {
        if value.dtype == .uint8 {
            throw Drama3Error.conversionFailed("\(mapped) is FP8 outside the routed experts")
        }
        if isZeroCenteredNorm(mapped) {
            return (value.asType(.float32) + 1).asType(value.dtype)
        }
        if mapped.hasSuffix("conv1d.weight"), value.ndim == 3, value.dim(-1) != 1 {
            return value.movedAxis(source: 2, destination: 1)
        }
        return value
    }

    // MARK: FP8

    /// E4M3 (bias 7, no infinities, 0x7F/0xFF = NaN) decode table.
    static let e4m3Table = MLXArray(e4m3Values)

    static let e4m3Values: [Float] = {
        var values = [Float](repeating: 0, count: 256)
        for byte in 0 ..< 256 {
            let sign: Float = byte & 0x80 != 0 ? -1 : 1
            let exponent = (byte >> 3) & 0xF
            let mantissa = Float(byte & 0x7)
            if exponent == 0xF && byte & 0x7 == 0x7 {
                values[byte] = .nan
            } else if exponent == 0 {
                values[byte] = sign * mantissa / 8 * pow(2, -6)
            } else {
                values[byte] = sign * (1 + mantissa / 8) * pow(2, Float(exponent - 7))
            }
        }
        return values
    }()

    static func dequantizeFP8(_ weight: MLXArray, scaleInverse: MLXArray) -> MLXArray {
        let rows = weight.dim(0)
        let columns = weight.dim(1)
        let decoded = e4m3Table[weight.asType(.int32)]
        var scale = repeated(scaleInverse.asType(.float32), count: fp8BlockSize, axis: 0)
        scale = repeated(scale, count: fp8BlockSize, axis: 1)
        scale = scale[..<rows, ..<columns]
        return (decoded * scale).asType(.bfloat16)
    }

    // MARK: Quantization

    static func quantizationBits(for name: String, requested: Int?) -> Int? {
        guard let requested, name.hasSuffix(".weight") else { return nil }
        if name.hasSuffix(".mlp.gate.weight") || name.hasSuffix(".shared_expert_gate.weight") {
            return 8
        }
        let quantizable = [
            "_proj.weight", ".wqkv.weight", ".wo.weight", ".w1.weight", ".w2.weight", ".w3.weight",
            "embed_tokens.weight", "lm_head.weight", ".eh_proj.weight",
        ]
        if name.hasPrefix("tower.heads.") { return requested }
        return quantizable.contains { name.hasSuffix($0) } ? requested : nil
    }

    static func quantizedTensors(name: String, value: MLXArray, bits: Int?) -> [String: MLXArray] {
        guard let bits = quantizationBits(for: name, requested: bits),
              value.ndim >= 2, value.dim(-1) % quantGroupSize == 0
        else { return [name: value] }
        let (wq, scales, biases) = quantized(value, groupSize: quantGroupSize, bits: bits)
        let base = String(name.dropLast(".weight".count))
        var out = [name: wq, base + ".scales": scales]
        if let biases { out[base + ".biases"] = biases }
        return out
    }

    // MARK: Output

    struct ShardWriter {
        let directory: URL
        var pending: [String: MLXArray] = [:]
        var pendingBytes = 0
        var shardNames: [String] = []
        var weightMap: [String: String] = [:]

        init(directory: URL) {
            self.directory = directory
        }

        mutating func add(_ tensors: [String: MLXArray]) throws {
            for (name, value) in tensors {
                pending[name] = value
                pendingBytes += value.nbytes
            }
            if pendingBytes >= Drama3Convert.shardBytes {
                try flush()
            }
        }

        mutating func flush() throws {
            guard !pending.isEmpty else { return }
            let name = String(format: "drama3-%05d.safetensors", shardNames.count + 1)
            eval(Array(pending.values))
            try MLX.save(arrays: pending, url: directory.appendingPathComponent(name))
            for key in pending.keys { weightMap[key] = name }
            shardNames.append(name)
            pending = [:]
            pendingBytes = 0
        }

        mutating func finish(indexName: String) throws -> Int {
            try flush()
            let index: [String: Any] = ["metadata": ["format": "mlx"], "weight_map": weightMap]
            let data = try JSONSerialization.data(withJSONObject: index, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: directory.appendingPathComponent(indexName))
            return weightMap.count
        }
    }

    static func writeConfig(source: URL, output: URL, bits: Int?) throws {
        let data = try Data(contentsOf: source.appendingPathComponent("config.json"))
        guard var root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw Drama3Error.invalidConfig("config.json")
        }
        root.removeValue(forKey: "quantization_config")
        if let bits {
            root["mlx_quantization"] = ["bits": bits, "group_size": quantGroupSize]
        } else {
            root.removeValue(forKey: "mlx_quantization")
        }
        let out = try JSONSerialization.data(withJSONObject: root, options: [.prettyPrinted, .sortedKeys])
        try out.write(to: output.appendingPathComponent("config.json"))
    }

    // MARK: Codec

    /// Folds weight norm (`parametrizations.weight.original0/1` and legacy `weight_g/weight_v`), drops
    /// recomputable causal-mask buffers, and moves conv weights to MLX layout:
    /// Conv1d [out, in, k] → [out, k, in]; ConvTranspose1d [in, out, k] → [out, k, in].
    /// The transformers' bf16 `freqs_cis` tables are kept: the reference runtime rotates with them.
    static func convertCodec(_ source: [String: MLXArray]) throws -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        var consumed = Set<String>()
        let markers = [
            (".parametrizations.weight.original0", ".parametrizations.weight.original1"),
            (".weight_g", ".weight_v"),
        ]
        for (gMarker, vMarker) in markers {
            for key in source.keys where key.hasSuffix(gMarker) {
                let base = String(key.dropLast(gMarker.count))
                guard let g = source[key], let v = source[base + vMarker] else {
                    throw Drama3Error.conversionFailed("codec weight-norm pair missing for \(base)")
                }
                let v32 = v.asType(.float32)
                let norm = sqrt((v32 * v32).sum(axes: Array(1 ..< v32.ndim), keepDims: true))
                out[base + ".weight"] = g.asType(.float32) * v32 / norm
                consumed.insert(key)
                consumed.insert(base + vMarker)
            }
        }
        for (key, value) in source where !consumed.contains(key) && !key.hasSuffix(".causal_mask") {
            out[key] = value.asType(.float32)
        }
        var converted: [String: MLXArray] = [:]
        for (key, value) in out {
            if key.hasSuffix(".weight"), value.ndim == 3 {
                converted[key] = isTransposedConv(key) ? value.transposed(1, 2, 0) : value.transposed(0, 2, 1)
            } else {
                converted[key] = value
            }
        }
        eval(Array(converted.values))
        return converted
    }

    /// ConvTranspose1d weights in the Fish codec: decoder upsampling blocks (`decoder.model.N.block.1`)
    /// and the quantizer's upsample stages (`quantizer.upsample.N.0`).
    static func isTransposedConv(_ key: String) -> Bool {
        let parts = key.split(separator: ".")
        if parts.count >= 5, parts[0] == "decoder", parts[1] == "model", parts[3] == "block", parts[4] == "1" {
            return true
        }
        if parts.count >= 4, parts[0] == "quantizer", parts[1] == "upsample", parts[3] == "0" {
            return true
        }
        return false
    }
}
