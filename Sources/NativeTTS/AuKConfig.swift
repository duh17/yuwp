import Foundation
@preconcurrency import MLX

public enum AuKError: Error, LocalizedError, Equatable {
    case unsupportedVariant(String)
    case missingWeights(String)
    case unconvertedCheckpoint(String)
    case invalidInput(String)
    case generationFailed(String)
    case conversionFailed(String)
    case tokenizerFailed(String)

    public var errorDescription: String? {
        switch self {
        case .unsupportedVariant(let message):
            "Unsupported AuK variant: \(message)"
        case .missingWeights(let message):
            "Missing AuK weights: \(message)"
        case .unconvertedCheckpoint(let message):
            "AuK checkpoint is not converted to MLX: \(message)"
        case .invalidInput(let message):
            "Invalid AuK input: \(message)"
        case .generationFailed(let message):
            "AuK generation failed: \(message)"
        case .conversionFailed(let message):
            "AuK conversion failed: \(message)"
        case .tokenizerFailed(let message):
            "AuK tokenizer failed: \(message)"
        }
    }
}

public enum TTSBackendKind: Equatable, Sendable {
    case aukFlash
    case qwen3TTS
}

public enum AuKModelLayout: Equatable, Sendable {
    case converted(variant: String)
    case pytorchSource
    case unknown
}

public enum AuKMappedKey: Equatable, Sendable {
    case dit(String)
    case fusion(String)
    case thinker(String)
    case skip
}

public struct AuKFlashConfig: Sendable {
    public static let name = "AuK-Flash"
    public static let variant = "flash"
    public static let sampleRate = 24_000
    public static let thinkerSampleRate = 16_000
    public static let downsampleRate = 480
    public static let latentDim = 64
    public static let ditDim = 1536
    public static let ditHeads = 24
    public static let ditDimHead = 64
    public static let ditFFMult: Float = 2.0
    public static let textHiddenDim = 2048
    public static let ditLayers = 10
    public static let ditSingleLayers = 20
    public static let thinkerLayers = 36
    public static let thinkerHeads = 16
    public static let thinkerKVHeads = 2
    public static let thinkerHidden = 2048
    public static let thinkerIntermediate = 11_008
    public static let thinkerVocab = 151_936
    public static let thinkerRopeTheta: Float = 1_000_000
    public static let thinkerRMSNormEps: Float = 1e-6
    public static let audioDModel = 1280
    public static let audioLayers = 32
    public static let audioHeads = 20
    public static let audioFFN = 5120
    public static let audioOutputDim = 2048
    public static let audioWindow = 100
    public static let audioMelBins = 128
    public static let audioTokenId = 151_646
    public static let audioBosId = 151_647
    public static let audioEosId = 151_648
    public static let nFft = 400
    public static let hopLength = 160
    public static let convPreIsCausal = false
    public static let torchFloat32Eps: Float = 1.1920928955078125e-07
    public static let upsampleRates = [5, 4, 3, 2, 2, 2]
    public static let upsampleKernelSizes = [10, 8, 6, 4, 4, 4]
    public static let upsampleInitialChannel = 1536
    public static let resblockKernelSizes = [3, 7, 11]
    public static let resblockDilationSizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
    public static let downsampleRates = [2, 2, 2, 3, 4, 5]
    public static let downsampleChannels = [12, 24, 48, 96, 192, 384, 768]
    public static let tGrid: [Float] = [
        0.0,
        0.07612049579620361,
        0.2928932309150696,
        0.6173166036605835,
        1.0,
    ]

    public static var hopSize: Int {
        downsampleRates.reduce(1, *)
    }

    public static func latentFrames(seconds: Double) -> Int {
        max(1, Int(ceil(seconds * Double(sampleRate) / Double(downsampleRate))))
    }
}

public struct AuKSampling: Sendable, Equatable {
    public var nfe: Int
    public var cfgStrength: Float
    public var sway: Float?
    public var tGrid: [Float]

    public var usesCFG: Bool { cfgStrength >= 1e-5 }

    public static func resolve(
        nfe: Int,
        cfgStrength: Float,
        sway: Float?,
        tGrid: [Float]?
    ) -> AuKSampling {
        _ = nfe
        _ = cfgStrength
        _ = sway
        _ = tGrid
        return AuKSampling(
            nfe: 4,
            cfgStrength: 0,
            sway: nil,
            tGrid: AuKFlashConfig.tGrid
        )
    }
}

func aukModelName(fromYAML yaml: String) -> String? {
    for line in yaml.split(whereSeparator: \.isNewline) {
        let trimmed = line.trimmingCharacters(in: .whitespaces)
        guard trimmed.hasPrefix("name:") else { continue }
        var value = trimmed.dropFirst("name:".count).trimmingCharacters(in: .whitespaces)
        value = value.trimmingCharacters(in: CharacterSet(charactersIn: "\"'"))
        if !value.isEmpty {
            return value
        }
    }
    return nil
}

func aukRequireFlashVariant(named name: String) throws -> String {
    if name == AuKFlashConfig.name || name.lowercased() == AuKFlashConfig.variant {
        return AuKFlashConfig.variant
    }
    throw AuKError.unsupportedVariant(
        "Yuwp's native port supports AuK-Flash only (fixed 4 steps, CFG off); got \(name)"
    )
}

public func inspectAuKModelDirectory(_ url: URL) -> AuKModelLayout {
    let fm = FileManager.default
    let yamlURL = url.appendingPathComponent("config.yaml")
    let jsonURL = url.appendingPathComponent("auk_config.json")
    var namedFlash = false
    if let yaml = try? String(contentsOf: yamlURL, encoding: .utf8),
       let name = aukModelName(fromYAML: yaml),
       name == AuKFlashConfig.name {
        namedFlash = true
    }
    if let data = try? Data(contentsOf: jsonURL),
       let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
       let name = obj["name"] as? String,
       name == AuKFlashConfig.name {
        namedFlash = true
    }

    let convertedDit = fm.fileExists(atPath: url.appendingPathComponent("dit_flash.safetensors").path)
        || fm.fileExists(atPath: url.appendingPathComponent("dit_flash.q8.safetensors").path)
        || fm.fileExists(atPath: url.appendingPathComponent("dit_flash.q4.safetensors").path)
    let convertedFusion = fm.fileExists(atPath: url.appendingPathComponent("fusion_flash.safetensors").path)
    let hasVAE = fm.fileExists(atPath: url.appendingPathComponent("vae.safetensors").path)

    if convertedDit && convertedFusion && hasVAE {
        return .converted(variant: AuKFlashConfig.variant)
    }

    let pytorchDit = fm.fileExists(atPath: url.appendingPathComponent("auk_flash.safetensors").path)
    if namedFlash || pytorchDit {
        return .pytorchSource
    }
    return .unknown
}

public func detectTTSBackend(at url: URL) -> TTSBackendKind {
    switch inspectAuKModelDirectory(url) {
    case .converted, .pytorchSource:
        return .aukFlash
    case .unknown:
        return .qwen3TTS
    }
}

func remapDiTKey(_ src: String) -> AuKMappedKey {
    if src.hasPrefix("text_encoder.") {
        return .skip
    }
    if src == "layer_weights" || src == "layer_scale" {
        return .fusion(src)
    }
    var name = src
    if name.hasPrefix("transformer.") {
        name = String(name.dropFirst("transformer.".count))
    }
    if name == "rotary_embed.inv_freq" {
        return .fusion("inv_freq")
    }
    name = name.replacingOccurrences(of: "time_mlp.2.", with: "time_mlp.1.")
    name = name.replacingOccurrences(of: "conv_pos_embed.conv1d.2.", with: "conv_pos_embed.conv1d.1.")
    return .dit(name)
}

func remapThinkerKey(_ src: String) -> AuKMappedKey {
    guard src.hasPrefix("thinker.") else { return .skip }
    var name = String(src.dropFirst("thinker.".count))
    if name.hasPrefix("visual.") || name == "lm_head.weight" || name.hasPrefix("audio_tower.audio_bos_eos_token") {
        return .skip
    }
    if name.hasPrefix("model.") {
        name = String(name.dropFirst("model.".count))
    }
    return .thinker(name)
}

func remapVAEKey(_ src: String) -> String? {
    if src.hasPrefix("flow.") { return nil }
    if src.contains(".upsample.") || src.contains(".downsample.") || src.contains(".lowpass.") {
        return nil
    }
    if src.hasPrefix("activation_post.upsample") || src.hasPrefix("activation_post.downsample") {
        return nil
    }
    if src == "global_mean" || src == "global_log_std" {
        return src
    }

    let encoderPrefix = "audio_encoder.generator."
    if src.hasPrefix(encoderPrefix) {
        let rest = String(src.dropFirst(encoderPrefix.count))
        let parts = rest.split(separator: ".", omittingEmptySubsequences: false).map(String.init)
        guard let index = parts.first.flatMap(Int.init) else { return nil }

        if index == 0, parts.count >= 3, parts[1] == "layer" {
            return "audio_encoder.pre." + parts.dropFirst(2).joined(separator: ".")
        }
        if index == 20, parts.count >= 3, parts[1] == "layer" {
            return "audio_encoder.post." + parts.dropFirst(2).joined(separator: ".")
        }

        let downIdx = [2, 5, 8, 11, 14, 17]
        let stackIdx = [3, 6, 9, 12, 15, 18]
        if let stage = downIdx.firstIndex(of: index), parts.count >= 3, parts[1] == "layer" {
            return "audio_encoder.stages.\(stage).down." + parts.dropFirst(2).joined(separator: ".")
        }
        if let stage = stackIdx.firstIndex(of: index),
           parts.count >= 5,
           parts[1] == "layers",
           let inner = Int(parts[3]) {
            let mappedInner: String
            switch inner {
            case 1: mappedInner = "conv1"
            case 3: mappedInner = "conv2"
            default: return nil
            }
            let param = parts.dropFirst(4).joined(separator: ".")
            return "audio_encoder.stages.\(stage).stack.layers.\(parts[2]).\(mappedInner).\(param)"
        }
        return nil
    }

    if src.hasPrefix("conv_pre.") || src.hasPrefix("conv_post.") {
        return "decoder." + src
    }
    if src.hasPrefix("ups.") {
        let parts = src.split(separator: ".").map(String.init)
        guard parts.count >= 4 else { return nil }
        return "decoder.ups.\(parts[1]).\(parts[3])"
    }
    if src.hasPrefix("resblocks.") || src.hasPrefix("activation_post.act.") {
        return "decoder." + src
    }
    return nil
}

func transposeConv1dWeightCPU(_ values: [Float], outChannels: Int, inChannels: Int, kernel: Int) -> [Float] {
    var out = [Float](repeating: 0, count: values.count)
    for o in 0 ..< outChannels {
        for i in 0 ..< inChannels {
            for k in 0 ..< kernel {
                let src = (o * inChannels + i) * kernel + k
                let dst = (o * kernel + k) * inChannels + i
                out[dst] = values[src]
            }
        }
    }
    return out
}

func transposeConvTranspose1dWeightCPU(_ values: [Float], inChannels: Int, outChannels: Int, kernel: Int) -> [Float] {
    var out = [Float](repeating: 0, count: values.count)
    for i in 0 ..< inChannels {
        for o in 0 ..< outChannels {
            for k in 0 ..< kernel {
                let src = (i * outChannels + o) * kernel + k
                let dst = (o * kernel + k) * inChannels + i
                out[dst] = values[src]
            }
        }
    }
    return out
}

func foldWeightNormCPU(g: [Float], v: [Float], rows: Int) -> [Float] {
    let width = v.count / rows
    var out = [Float](repeating: 0, count: v.count)
    for r in 0 ..< rows {
        var norm: Float = 0
        for j in 0 ..< width {
            let x = v[r * width + j]
            norm += x * x
        }
        norm = sqrt(norm)
        let scale = g[r] / norm
        for j in 0 ..< width {
            out[r * width + j] = scale * v[r * width + j]
        }
    }
    return out
}

func applyInterleavedRoPECPU(_ x: [Float], cos: [Float], sin: [Float]) -> [Float] {
    var out = [Float](repeating: 0, count: x.count)
    var i = 0
    while i + 1 < x.count {
        let x1 = x[i]
        let x2 = x[i + 1]
        out[i] = x[i] * cos[i] + (-x2) * sin[i]
        out[i + 1] = x[i + 1] * cos[i + 1] + x1 * sin[i + 1]
        i += 2
    }
    return out
}

func applyHalfSplitRoPECPU(_ x: [Float], cos: [Float], sin: [Float]) -> [Float] {
    let half = x.count / 2
    var rot = [Float](repeating: 0, count: x.count)
    for i in 0 ..< half {
        rot[i] = -x[half + i]
        rot[half + i] = x[i]
    }
    var out = [Float](repeating: 0, count: x.count)
    for i in 0 ..< x.count {
        out[i] = x[i] * cos[i] + rot[i] * sin[i]
    }
    return out
}

func foldWeightNorm(g: MLXArray, v: MLXArray) -> MLXArray {
    let g32 = g.asType(.float32)
    let v32 = v.asType(.float32)
    let axes = Array(1 ..< v32.ndim)
    let norm = sqrt(v32.square().sum(axes: axes, keepDims: true))
    return (g32 / norm) * v32
}

func transposeConv1dWeight(_ weight: MLXArray) -> MLXArray {
    weight.asType(.float32).transposed(0, 2, 1)
}

func transposeConvTranspose1dWeight(_ weight: MLXArray) -> MLXArray {
    weight.asType(.float32).transposed(1, 2, 0)
}

func analyticRoPEInvFreq(dimHead: Int, base: Double = 10_000) -> [Double] {
    stride(from: 0, to: dimHead, by: 2).map { i in
        pow(base, -Double(i) / Double(dimHead))
    }
}

func applyInterleavedRoPE(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let shape = x.shape
    guard let last = shape.last, last % 2 == 0 else { return x }
    let paired = x.reshaped(shape.dropLast() + [last / 2, 2])
    let x1 = paired[.ellipsis, 0]
    let x2 = paired[.ellipsis, 1]
    let rot = stacked([-x2, x1], axis: -1).reshaped(shape)
    return x * cos + rot * sin
}

func applyHalfSplitRoPE(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let half = x.dim(-1) / 2
    let x1 = x[.ellipsis, ..<half]
    let x2 = x[.ellipsis, half...]
    let rot = concatenated([-x2, x1], axis: -1)
    return x * cos + rot * sin
}

func aukApplyChatTemplate(instruction: String, hasAudio: Bool) -> String {
    let user: String
    if hasAudio {
        user = instruction + "<|audio_bos|><|AUDIO|><|audio_eos|>"
    } else {
        let marker = "|<no_prompt_audio>|"
        user = instruction.hasSuffix(marker) ? instruction : instruction + marker
    }
    return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n\(user)<|im_end|>\n<|im_start|>assistant\n"
}

func aukRejectUnsupportedMultimodal(contentTypes: [String]) throws {
    if contentTypes.contains(where: { $0 == "image" || $0 == "video" }) {
        throw AuKError.invalidInput("AuK-Flash native inference rejects image/video input")
    }
}

func aukAudioTowerTokenCount(melFrames: Int, nWindow: Int = AuKFlashConfig.audioWindow) -> Int {
    guard melFrames > 0 else { return 0 }
    let win = nWindow * 2
    var keepSum = 0
    var offset = 0
    while offset < melFrames {
        let length = min(win, melFrames - offset)
        keepSum += (length + 1) / 2
        offset += length
    }
    return keepSum / 2
}

func aukWhisperMelFrameCount(
    sampleCount: Int,
    nFft: Int = AuKFlashConfig.nFft,
    hopLength: Int = AuKFlashConfig.hopLength
) -> Int {
    guard sampleCount > 0, hopLength > 0 else { return 0 }
    let paddedLen = sampleCount + nFft
    let numFrames = 1 + (paddedLen - nFft) / hopLength
    return max(0, numFrames - 1)
}

func aukDownmixToMono(channels: [[Float]]) -> [Float] {
    guard let first = channels.first else { return [] }
    guard channels.count > 1 else { return first }
    let frames = first.count
    var mono = [Float](repeating: 0, count: frames)
    let scale = 1.0 / Float(channels.count)
    for channel in channels {
        let n = min(frames, channel.count)
        for i in 0 ..< n {
            mono[i] += channel[i]
        }
    }
    for i in 0 ..< frames {
        mono[i] *= scale
    }
    return mono
}

func aukExpandAudioTokens(_ ids: [Int], count: Int, audioTokenId: Int = AuKFlashConfig.audioTokenId) -> [Int] {
    guard count > 0, let index = ids.firstIndex(of: audioTokenId) else { return ids }
    var out = Array(ids.prefix(index))
    out.append(contentsOf: Array(repeating: audioTokenId, count: count))
    out.append(contentsOf: ids.suffix(from: index + 1))
    return out
}

func aukPeriodicHann(size: Int) -> [Float] {
    guard size > 0 else { return [] }
    if size == 1 { return [1] }
    return (0 ..< size).map { n in
        0.5 - 0.5 * cos(2 * Float.pi * Float(n) / Float(size))
    }
}

func aukRemoveFrameDC(_ frame: [Float]) -> [Float] {
    guard !frame.isEmpty else { return frame }
    let mean = frame.reduce(0, +) / Float(frame.count)
    return frame.map { $0 - mean }
}

public func aukResolveInstruction(instruction: String?, instructions: String?, input: String?) -> String? {
    // AuK accepts a native full instruction, while OpenAI-shaped requests put
    // the speech content in `input` and delivery guidance in `instructions`.
    for candidate in [instruction, input, instructions] {
        if let trimmed = candidate?.trimmingCharacters(in: .whitespacesAndNewlines), !trimmed.isEmpty {
            return trimmed
        }
    }
    return nil
}

public func aukResolveGenSeconds(_ value: Double?, hasReferenceAudio: Bool) throws -> Double? {
    if let value {
        guard value.isFinite, value > 0 else {
            throw AuKError.invalidInput("gen_seconds must be a positive finite duration")
        }
        return value
    }
    if hasReferenceAudio {
        return nil
    }
    throw AuKError.invalidInput("Instruct TTS requires gen_seconds when no reference audio is provided")
}

func aukRequireExactKeys(fileKeys: Set<String>, modelKeys: Set<String>, label: String) throws {
    let missing = modelKeys.subtracting(fileKeys)
    let extra = fileKeys.subtracting(modelKeys)
    if !missing.isEmpty || !extra.isEmpty {
        let missingSample = missing.sorted().prefix(8).joined(separator: ", ")
        let extraSample = extra.sorted().prefix(8).joined(separator: ", ")
        throw AuKError.conversionFailed(
            "\(label) weight graph is incomplete: missing \(missing.count), extra \(extra.count). missing=\(missingSample) extra=\(extraSample)"
        )
    }
}

func aukRequireConvertedGraph(
    fileKeys: Set<String>,
    requiredKeys: Set<String>,
    expectedCount: Int,
    label: String
) throws {
    let missingRequired = requiredKeys.subtracting(fileKeys)
    if !missingRequired.isEmpty || fileKeys.count != expectedCount {
        throw AuKError.conversionFailed(
            "\(label) conversion missed required tensors (have \(fileKeys.count), expected \(expectedCount), missing \(missingRequired.sorted().joined(separator: ", ")))"
        )
    }
}

func aukExpectedVAETensorCount() -> Int {
    let stages = AuKFlashConfig.downsampleRates.count
    let stackPairs = 6
    let encoder = 2 + stages * (2 + stackPairs * 2 * 2) + 2
    let kernels = AuKFlashConfig.resblockKernelSizes.count
    let ups = AuKFlashConfig.upsampleRates.count
    let dilations = AuKFlashConfig.resblockDilationSizes[0].count
    let actsPerBlock = 2 * dilations
    let resblocks = ups * kernels
    let decoder = 2 + ups * 2 + resblocks * (dilations * 2 * 2 + actsPerBlock * 2) + 2 + 1
    return encoder + decoder + 2
}

func resolveAuKWeightFile(directory: URL, baseName: String, bits: Int?) -> (url: URL, bits: Int?) {
    let fm = FileManager.default
    if let bits {
        let quantized = directory.appendingPathComponent("\(baseName).q\(bits).safetensors")
        if fm.fileExists(atPath: quantized.path) {
            return (quantized, bits)
        }
        return (directory.appendingPathComponent("\(baseName).safetensors"), bits)
    }
    return (directory.appendingPathComponent("\(baseName).safetensors"), nil)
}
