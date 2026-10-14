import Foundation
@preconcurrency import MLX
import MLXRandom

public enum YuE2Precision: String, Sendable, CaseIterable {
    case bf16
    case int8 = "8bit"
    case int4 = "4bit"

    var arFile: String { "ar-\(rawValue).safetensors" }
}

public struct YuE2StageTiming: Sendable, Codable {
    public var seconds: Double
    public var prefillSeconds: Double
    public var prefixTokens: Int
    public var outputTokens: Int
    public var tokensPerSecond: Double
}

public struct YuE2Plan: Sendable {
    public let request: YuE2Request
    /// The rendered score: YuE2's own plan, the supplied ABC, or nil for `cot=off`.
    public let abc: String?
    public let abcIDs: [Int]?
    /// The semantic-phase prefix: prompt, score and MUSIC_START.
    public let prefix: [Int]
    public let timing: YuE2StageTiming?
    public let truncated: Bool
}

public struct YuE2Semantic: Sendable {
    public let plan: YuE2Plan
    /// Codec-local ids in `0 ..< 32768`, 25 per second of audio.
    public let codes: [Int]
    public let timing: YuE2StageTiming
    public let truncated: Bool
}

/// YuE2-3B on MLX: plan an ABC score, generate semantic codec ids, solve acoustic latents with
/// midpoint flow matching, and decode 48 kHz stereo through the Oobleck VAE.
public final class YuE2Engine: @unchecked Sendable {
    public let config: YuE2Config
    public let tokenizer: YuE2Tokenizer
    public let precision: YuE2Precision
    let directory: URL
    let vaeDirectory: URL
    private var decoder: YuE2ARModel?
    private var heads: [YuE2Phase: YuE2PhaseHead] = [:]
    private var conditioner: YuE2ARModel?
    private var acoustic: YuE2AcousticModel?
    private var vae: YuE2VAEDecoder?

    /// `directory` is a converted mlx-Yue2-3B layout: `config.json`, `qwen.tiktoken`,
    /// `ar-{bf16,8bit,4bit}.safetensors`, `nar-bf16.safetensors`, and the VAE in `vae/` by default.
    public init(directory: URL, precision: YuE2Precision = .int8, vaeDirectory: URL? = nil) throws {
        self.directory = directory
        self.precision = precision
        self.vaeDirectory = vaeDirectory ?? directory.appendingPathComponent("vae")
        config = try YuE2Config.load(directory: directory)
        tokenizer = try YuE2Tokenizer(url: directory.appendingPathComponent("qwen.tiktoken"))
    }

    // MARK: Model residency

    private func decodeModel() throws -> YuE2ARModel {
        if let decoder { return decoder }
        let model = precision == .bf16 ? try conditioningModel() : try YuE2ARModel.load(directory: directory, file: precision.arFile, config: config)
        decoder = model
        return model
    }

    private func head(_ phase: YuE2Phase) throws -> YuE2PhaseHead {
        if let head = heads[phase] { return head }
        let head = YuE2PhaseHead(try decodeModel().lmHead, phase: phase)
        heads[phase] = head
        return head
    }

    /// The NAR attends over the bf16 AR path; quantized AR weights never condition it.
    private func conditioningModel() throws -> YuE2ARModel {
        if let conditioner { return conditioner }
        let model = try YuE2ARModel.load(directory: directory, file: YuE2Precision.bf16.arFile, config: config)
        conditioner = model
        return model
    }

    private func acousticModel() throws -> YuE2AcousticModel {
        if let acoustic { return acoustic }
        let model = try YuE2AcousticModel.load(directory: directory, config: config)
        acoustic = model
        return model
    }

    public func vaeDecoder() throws -> YuE2VAEDecoder {
        if let vae { return vae }
        let model = try YuE2VAEDecoder(directory: vaeDirectory)
        vae = model
        return model
    }

    /// Drop the quantized decode weights once token generation is done.
    public func releaseDecoder() {
        if precision != .bf16 { decoder = nil }
        heads = [:]
        MLX.Memory.clearCache()
    }

    // MARK: Stages

    public func plan(_ request: YuE2Request, sampling: YuE2Sampling = .abcDefault, onToken: ((Int) -> Void)? = nil) throws -> YuE2Plan {
        try request.validate()
        if request.cot == .off {
            return YuE2Plan(request: request, abc: nil, abcIDs: nil, prefix: request.prefix(tokenizer: tokenizer, abcIDs: nil), timing: nil, truncated: false)
        }
        if let abc = request.abc {
            let ids = tokenizer.encode(abc)
            return YuE2Plan(request: request, abc: abc, abcIDs: ids, prefix: request.prefix(tokenizer: tokenizer, abcIDs: ids), timing: nil, truncated: false)
        }
        let (ids, timing, truncated) = try generate(
            prefix: request.prefix(tokenizer: tokenizer, abcIDs: nil), negative: nil, guidance: 1,
            sampling: sampling, seed: request.seed, phase: .abc, legacyOff: false, onToken: onToken
        )
        return YuE2Plan(request: request, abc: tokenizer.decode(ids), abcIDs: ids, prefix: request.prefix(tokenizer: tokenizer, abcIDs: ids), timing: timing, truncated: truncated)
    }

    public func semantic(_ plan: YuE2Plan, sampling: YuE2Sampling = .semanticDefault, onToken: ((Int) -> Void)? = nil) throws -> YuE2Semantic {
        let request = plan.request
        let guidance = request.guidance
        let negative = guidance == 1 ? nil : request.negativePrefix(tokenizer: tokenizer, abcIDs: plan.abcIDs)
        let (ids, timing, truncated) = try generate(
            prefix: plan.prefix, negative: negative, guidance: guidance, sampling: sampling,
            seed: request.seed, phase: .semantic, legacyOff: request.cot == .off, onToken: onToken
        )
        return YuE2Semantic(plan: plan, codes: ids.map { $0 - YuE2Protocol.codecOffset }, timing: timing, truncated: truncated)
    }

    /// Request-local FP32 solver noise `[frames, 64]`. Not bit-compatible with mlx-Yue's NumPy PCG64 noise.
    public static func noise(frames: Int, seed: UInt64) -> MLXArray {
        MLXRandom.normal([frames, YuE2Protocol.latentDim], dtype: .float32, key: MLXRandom.key(seed))
    }

    // MARK: AR token generation

    /// Sample phase-local tokens; returns native ids of the content (end token excluded).
    /// The next step is queued on the GPU before the current token is read back.
    func generate(
        prefix: [Int], negative: [Int]?, guidance: Float, sampling: YuE2Sampling, seed: UInt64,
        phase: YuE2Phase, legacyOff: Bool, onToken: ((Int) -> Void)?
    ) throws -> ([Int], YuE2StageTiming, Bool) {
        try sampling.validate()
        guard prefix.count + sampling.maxTokens <= YuE2Protocol.context else {
            throw YuE2Error.invalidInput("prefix (\(prefix.count)) + max tokens (\(sampling.maxTokens)) exceeds the 24576 context")
        }
        let model = try decodeModel()
        let phaseHead = try head(phase)
        let started = Date()

        func prefill(_ ids: [Int]) throws -> (MLXArray, [YuE2KVCache]) {
            let caches = (0 ..< config.layers).map { _ in YuE2KVCache(config, capacity: ids.count + sampling.maxTokens) }
            let input = MLXArray(ids.map(Int32.init)).reshaped(1, -1)
            var hidden: MLXArray?
            for start in stride(from: 0, to: ids.count, by: 1024) {
                let stop = min(start + 1024, ids.count)
                hidden = try model.step(input[0..., start ..< stop], caches: caches)
                if stop < ids.count { eval(caches.flatMap(\.state)) }
            }
            guard let hidden else { throw YuE2Error.invalidInput("empty prefix") }
            return (phaseHead(hidden), caches)
        }

        var (conditional, positive) = try prefill(prefix)
        var unconditional: MLXArray?
        var negativeCaches: [YuE2KVCache] = []
        if let negative {
            let (logits, caches) = try prefill(negative)
            unconditional = logits
            negativeCaches = caches
        }
        func guided(_ c: MLXArray, _ u: MLXArray?) -> MLXArray {
            guard let u else { return c }
            let delta = (c - u).asType(.bfloat16)
            return (u + (delta.asType(.float32) * guidance).asType(.bfloat16)).asType(.bfloat16)
        }
        var sampler = YuE2Sampler(phase: phase, sampling: sampling, seed: seed, legacyOff: legacyOff)
        var token = sampler.sample(guided(conditional, unconditional), step: 0)
        asyncEval(token)
        eval(positive.flatMap(\.state))
        let prefillSeconds = Date().timeIntervalSince(started)

        let end = YuE2Protocol.endIndex(phase)
        var output: [Int] = []
        output.reserveCapacity(sampling.maxTokens)
        var finished = false
        for step in 0 ..< sampling.maxTokens {
            var next: MLXArray?
            if step + 1 < sampling.maxTokens {
                // Queue step+1 assuming `token` is content; discarded if it turns out to be the end.
                sampler.remember(token)
                let native = phase == .semantic
                    ? token + Int32(YuE2Protocol.musicEnd)
                    : MLX.where(token .== Int32(YuE2Protocol.eod), MLXArray(Int32(YuE2Protocol.abcEnd)), token)
                let input = native.reshaped(1, 1)
                conditional = phaseHead(try model.step(input, caches: positive))
                if !negativeCaches.isEmpty {
                    unconditional = phaseHead(try model.step(input, caches: negativeCaches))
                }
                let sampled = sampler.sample(guided(conditional, unconditional), step: step + 1)
                asyncEval(sampled)
                next = sampled
            }
            let local = token.item(Int.self)
            if local == end {
                finished = true
                break
            }
            let id = YuE2Protocol.nativeToken(local: local, phase: phase)
            output.append(id)
            onToken?(id)
            guard let next else { break }
            token = next
        }
        let seconds = Date().timeIntervalSince(started)
        let produced = output.count + (finished ? 1 : 0)
        let timing = YuE2StageTiming(
            seconds: seconds, prefillSeconds: prefillSeconds, prefixTokens: prefix.count,
            outputTokens: produced, tokensPerSecond: Double(produced) / max(seconds - prefillSeconds, 1e-9)
        )
        return (output, timing, !finished)
    }

    // MARK: Acoustic flow matching

    /// Midpoint ODE from noise (t = 1) to latents (t = 0), one context chunk at a time.
    /// `noise` is FP32 `[codes.count, 64]`; returns FP32 latents of the same shape.
    public func synthesize(prefix: [Int], codes: [Int], noise: MLXArray, steps: Int, onProgress: ((Int, Int) -> Void)? = nil) throws -> MLXArray {
        guard steps >= 1 else { throw YuE2Error.invalidInput("steps must be positive") }
        guard noise.shape == [codes.count, YuE2Protocol.latentDim] else {
            throw YuE2Error.invalidInput("noise must be [\(codes.count), 64], got \(noise.shape)")
        }
        guard codes.allSatisfy({ (0 ..< YuE2Protocol.codecSize).contains($0) }) else {
            throw YuE2Error.invalidInput("codec ids must be in 0..<32768")
        }
        let ranges = try YuE2Protocol.chunkRanges(frames: codes.count, prefixTokens: prefix.count)
        let ar = try conditioningModel()
        let model = try acousticModel()
        var pieces: [MLXArray] = []
        let dt = 1 / Float(steps)
        for (chunkIndex, range) in ranges.enumerated() {
            let arTokens = prefix + codes[range].map { $0 + YuE2Protocol.codecOffset } + [YuE2Protocol.musicEnd]
            let context = conditioning(ar, tokens: arTokens)
            var state = noise[range].asType(.bfloat16)
            for step in 0 ..< steps {
                let t = 1 - Float(step) * dt
                let first = velocity(model, ar: ar, state: state, rawT: Self.logit(t), context: context, arLength: arTokens.count)
                let midpoint = state - (first.asType(.float32) * (dt / 2)).asType(.bfloat16)
                let second = velocity(model, ar: ar, state: midpoint, rawT: Self.logit(t - dt / 2), context: context, arLength: arTokens.count)
                state = state - (second.asType(.float32) * dt).asType(.bfloat16)
                eval(state)
                onProgress?(chunkIndex * steps + step + 1, ranges.count * steps)
            }
            pieces.append(state.asType(.float32))
        }
        let latents = pieces.count == 1 ? pieces[0] : concatenated(pieces, axis: 0)
        guard isFinite(latents).all().item(Bool.self) else { throw YuE2Error.numerical("acoustic latents are not finite") }
        return latents
    }

    /// Causal bf16 AR pass over prompt + codec ids, keeping each layer's rotated K/V.
    private func conditioning(_ ar: YuE2ARModel, tokens: [Int]) -> [(MLXArray, MLXArray)] {
        var x = ar.model.embedTokens(MLXArray(tokens.map(Int32.init)).reshaped(1, -1))
        var context: [(MLXArray, MLXArray)] = []
        for layer in ar.model.layers {
            let (h, k, v) = layer.conditioning(x)
            eval(h, k, v)
            context.append((k, v))
            x = h
        }
        return context
    }

    private func velocity(_ model: YuE2AcousticModel, ar: YuE2ARModel, state: MLXArray, rawT: Float, context: [(MLXArray, MLXArray)], arLength: Int) -> MLXArray {
        let length = state.dim(0) + 2
        let boundary = padded(state, widths: [IntOrPair((1, 1)), IntOrPair((0, 0))])
        let sigmoidT = 1 / (1 + Foundation.exp(-rawT))
        let shift = config.timestepShift
        let t = shift * sigmoidT / (1 + (shift - 1) * sigmoidT)
        var x = model.vae2llm(boundary.expandedDimensions(axis: 0))
        x = x + model.timeEmbedder(t).expandedDimensions(axis: 0)
        x = x + model.latentPositions.pe[..<length].expandedDimensions(axis: 0)
        for (layer, (arKeys, arValues)) in zip(model.model.layers, context) {
            let (q, k, v) = layer.attention.project(layer.inputNorm(x), offset: arLength)
            let keys = concatenated([arKeys, k], axis: 2)
            let values = concatenated([arValues, v], axis: 2)
            x = x + layer.attention.output(q, keys, values, mask: .none)
            x = x + layer.mlp(layer.preMLPNorm(x))
        }
        return model.llm2vae(ar.model.norm(x))[0, 1 ..< (length - 1)]
    }

    static func logit(_ t: Float) -> Float {
        if t <= 0 { return -20 }
        if t >= 1 { return 20 }
        return max(-20, min(20, Foundation.log(t / (1 - t))))
    }
}
