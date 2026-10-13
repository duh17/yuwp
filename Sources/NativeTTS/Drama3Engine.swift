import Foundation
@preconcurrency import MLX
import MLXNN
import MLXRandom
import Tokenizers

public struct Drama3Sampling: Sendable, Equatable {
    public var temperature: Float = 1.0
    public var topP: Float = 0.9
    public var topK: Int = 30
    public var maxFrames: Int = 2_048
    public var seed: UInt64?

    public init() {}
}

public struct Drama3Reference: Sendable {
    public var transcript: String
    /// Mono samples at `FishCodec.sampleRate`.
    public var samples: [Float]

    public init(transcript: String, samples: [Float]) {
        self.transcript = transcript
        self.samples = samples
    }
}

public struct Drama3Result: Sendable {
    public var samples: [Float]
    public var frameCount: Int
    public var promptRowCount: Int
    public var generationSeconds: Double
    public var finishedWithEndOfSpeech: Bool
}

final class Drama3Model: Module {
    @ModuleInfo(key: "backbone") var backbone: Drama3Backbone
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "tower") var tower: Drama3Tower

    init(_ config: Drama3Config) {
        _backbone.wrappedValue = Drama3Backbone(config.backbone)
        _lmHead.wrappedValue = Linear(config.backbone.hiddenSize, config.backbone.vocabularySize, bias: false)
        _tower.wrappedValue = Drama3Tower(
            config.tower, semanticBeginID: config.semanticBeginID, semanticEndID: config.semanticEndID
        )
        super.init()
    }
}

/// Swift/MLX runtime for Fish Audio Drama 3 (converted with `Drama3Convert`).
public final class Drama3Engine: @unchecked Sendable {
    public let config: Drama3Config
    public let sampleRate = FishCodec.sampleRate
    public let modelName = "Fish Audio Drama 3"

    let model: Drama3Model
    let codec: FishCodec
    let tokenizer: any Tokenizer
    let candidateIDs: MLXArray
    private let lock = NSLock()

    static let rasWindowFrames = 16
    static let rasTemperature: Float = 1.0
    static let rasTopP: Float = 0.95
    static let rasTopK = 30
    static let maxSpeakerID = 15

    init(config: Drama3Config, model: Drama3Model, codec: FishCodec, tokenizer: any Tokenizer) {
        self.config = config
        self.model = model
        self.codec = codec
        self.tokenizer = tokenizer
        let ids = [config.eosTokenID] + Array(config.semanticBeginID ... config.semanticEndID)
        candidateIDs = MLXArray(ids.map(Int32.init))
    }

    public static func load(directory: URL) async throws -> Drama3Engine {
        switch inspectDrama3ModelDirectory(directory) {
        case .officialSource:
            throw Drama3Error.unconvertedCheckpoint(
                "\(directory.path) holds the official PyTorch checkpoint. Convert once with `yuwp-tts convert-drama3 --src <drama-3-dir> --out <mlx-dir>`."
            )
        case .unknown:
            throw Drama3Error.missingWeights("no converted Drama 3 model in \(directory.path)")
        case .converted:
            break
        }
        let config = try Drama3Config.load(directory: directory)
        let model = Drama3Model(config)
        let weights = try loadConvertedWeights(directory)
        if config.quantizationBits != nil {
            quantize(model: model) { path, _ -> (groupSize: Int, bits: Int)? in
                guard let scales = weights[path + ".scales"], let weight = weights[path + ".weight"] else { return nil }
                let inputs = scales.dim(-1) * config.quantizationGroupSize
                return (config.quantizationGroupSize, weight.dim(-1) * 32 / inputs)
            }
        }
        let modelKeys = Set(model.parameters().flattened().map(\.0))
        let fileKeys = Set(weights.keys)
        guard modelKeys == fileKeys else {
            let missing = modelKeys.subtracting(fileKeys).sorted().prefix(6)
            let extra = fileKeys.subtracting(modelKeys).sorted().prefix(6)
            throw Drama3Error.missingWeights("weights do not match the model graph; missing \(Array(missing)), unexpected \(Array(extra))")
        }
        try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        model.train(false)
        eval(model.parameters())

        let codec = try FishCodec.load(url: directory.appendingPathComponent(Drama3Convert.codecFileName))
        let tokenizer = try await AutoTokenizer.from(modelFolder: directory)
        guard tokenizer.convertTokenToId("<|semantic:0|>") == config.semanticBeginID else {
            throw Drama3Error.invalidConfig("tokenizer <|semantic:0|> does not match semantic_start_token_id")
        }
        return Drama3Engine(config: config, model: model, codec: codec, tokenizer: tokenizer)
    }

    static func loadConvertedWeights(_ directory: URL) throws -> [String: MLXArray] {
        let indexURL = directory.appendingPathComponent(Drama3Convert.weightIndexFileName)
        let data = try Data(contentsOf: indexURL)
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let weightMap = root["weight_map"] as? [String: String]
        else { throw Drama3Error.missingWeights(Drama3Convert.weightIndexFileName) }
        var weights: [String: MLXArray] = [:]
        for shard in Set(weightMap.values).sorted() {
            for (key, value) in try MLX.loadArrays(url: directory.appendingPathComponent(shard)) {
                weights[key] = value
            }
        }
        return weights
    }

    // MARK: Prompt

    static let structuralTokens = ["<|im_start|>", "<|im_end|>", "<|voice|>", "<|text|>", "<|interleave|>", "<|semantic:"]

    static func validateCallerText(_ text: String, location: String) throws {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw Drama3Error.invalidInput("\(location): text must not be empty")
        }
        for token in structuralTokens where text.contains(token) {
            throw Drama3Error.invalidInput("\(location): \(token) is reserved for the prompt builder")
        }
        var search = text.startIndex
        while let marker = text.range(of: "<|speaker", range: search ..< text.endIndex) {
            guard leadingSpeakerTag(String(text[marker.lowerBound...])) != nil else {
                throw Drama3Error.invalidInput("\(location): speaker tags must be <|speaker:N|> with N in 0...\(maxSpeakerID)")
            }
            search = marker.upperBound
        }
    }

    /// The `<|speaker:N|>` tag at the start of `text`, if any.
    static func leadingSpeakerTag(_ text: String) -> String? {
        guard text.hasPrefix("<|speaker:"), let close = text.range(of: "|>") else { return nil }
        let digits = text[text.index(text.startIndex, offsetBy: 10) ..< close.lowerBound]
        guard let id = Int(digits), (0 ... maxSpeakerID).contains(id), String(id) == digits else { return nil }
        return String(text[..<close.upperBound])
    }

    static func withSpeakerTag(_ text: String) -> String {
        let trimmed = String(text.drop(while: { $0.isWhitespace }))
        return leadingSpeakerTag(trimmed) == nil ? "<|speaker:0|> \(text)" : text
    }

    enum PromptSegment {
        case text(String)
        case codes(MLXArray)
    }

    /// Prompt frame rows [rows, 1 + codebooks] in the `reference-bracket-v1` layout.
    func promptRows(text: String, references: [(transcript: String, codes: MLXArray)]) throws -> MLXArray {
        var segments: [PromptSegment] = [
            .text("<|im_start|>system\n"),
            .text(references.isEmpty
                ? "convert the provided text to speech"
                : "convert the provided text to speech reference to the following:"),
        ]
        for (index, reference) in references.enumerated() {
            try Self.validateCallerText(reference.transcript, location: "reference \(index + 1)")
            let tagged = Self.withSpeakerTag(reference.transcript)
            let body = String(tagged.drop(while: { $0.isWhitespace }))
            let leading = String(tagged.prefix(tagged.count - body.count))
            let tag = Self.leadingSpeakerTag(body) ?? "<|speaker:0|>"
            let transcript = tag + leading + body.dropFirst(tag.count)
            segments.append(.text("\n\n[Reference \(index + 1)]\nText: \(transcript)\nSpeech: "))
            segments.append(.codes(reference.codes))
        }
        try Self.validateCallerText(text, location: "input")
        segments += [
            .text("<|im_end|>\n"), .text("<|im_start|>user\n"), .text(Self.withSpeakerTag(text)),
            .text("<|im_end|>\n"), .text("<|im_start|>assistant\n<|voice|>"),
        ]
        let width = 1 + config.tower.codebookCount
        var rows: [MLXArray] = []
        for segment in segments {
            switch segment {
            case .text(let string):
                let ids = tokenizer.encode(text: string, addSpecialTokens: false)
                guard !ids.isEmpty else { continue }
                let tokens = MLXArray(ids.map(Int32.init)).reshaped(ids.count, 1)
                rows.append(concatenated([tokens, MLXArray.zeros([ids.count, width - 1], dtype: .int32)], axis: 1))
            case .codes(let codes):
                let codes32 = codes.asType(.int32)
                let main = codes32[0 ..< 1] + Int32(config.semanticBeginID)
                rows.append(concatenated([main, codes32], axis: 0).transposed(1, 0))
            }
        }
        return concatenated(rows, axis: 0)
    }

    // MARK: Sampling

    /// One token per row: drop ranks >= topK and tokens whose inclusive cumulative probability exceeds
    /// topP (rank 0 always stays), then draw. Returns int32 [rows].
    static func sample(_ logits: MLXArray, temperature: Float, topK: Int, topP: Float) -> MLXArray {
        let scaled = logits.asType(.float32) / max(temperature, 1e-5)
        let order = argSort(-scaled, axis: -1)
        let sorted = takeAlong(scaled, order, axis: -1)
        let cumulative = cumsum(softmax(sorted, axis: -1), axis: -1)
        let ranks = MLXArray.arange(sorted.dim(-1), dtype: .int32)
        let drop = ((cumulative .> topP) .|| (ranks .>= Int32(topK))) .&& (ranks .> 0)
        let masked = MLX.where(drop, MLXArray(-Float.infinity), sorted)
        let choice = MLXRandom.categorical(masked, axis: -1)
        return takeAlong(order, choice[0..., .newAxis].asType(.int32), axis: -1).squeezed(axis: -1).asType(.int32)
    }

    // MARK: Generation

    public func encodeReference(_ reference: Drama3Reference) -> MLXArray {
        lock.lock()
        defer { lock.unlock() }
        let codes = codec.encode(samples: MLXArray(reference.samples))
        eval(codes)
        return codes
    }

    public func synthesize(
        text: String,
        references: [Drama3Reference] = [],
        sampling: Drama3Sampling = Drama3Sampling(),
        onFrame: ((Int) -> Void)? = nil
    ) throws -> Drama3Result {
        let referenceCodes = references.map { (transcript: $0.transcript, codes: encodeReference($0)) }
        lock.lock()
        defer { lock.unlock() }
        if let seed = sampling.seed { MLXRandom.seed(seed) }
        let rows = try promptRows(text: text, references: referenceCodes)
        let start = Date()
        let (frames, sawEnd) = generateFrames(rows: rows, sampling: sampling, onFrame: onFrame)
        let generationSeconds = Date().timeIntervalSince(start)
        guard !frames.isEmpty else {
            throw Drama3Error.generationFailed("the model ended speech before producing any audio frame")
        }
        let codes = concatenated(frames, axis: 0)[0..., 1...].transposed(1, 0)
        let audio = codec.decode(codes: codes)
        eval(audio)
        return Drama3Result(
            samples: audio.asArray(Float.self),
            frameCount: frames.count,
            promptRowCount: rows.dim(0),
            generationSeconds: generationSeconds,
            finishedWithEndOfSpeech: sawEnd
        )
    }

    func generateFrames(rows: MLXArray, sampling: Drama3Sampling, onFrame: ((Int) -> Void)?) -> ([MLXArray], Bool) {
        let backbone = model.backbone
        let tower = model.tower
        let backboneCache = Drama3BackboneCache(config: config.backbone)
        let towerCaches = tower.newCaches()
        let rowCount = rows.dim(0)

        func backboneInput(_ frames: MLXArray) -> MLXArray {
            tower.backboneInput(frames: frames, tokenEmbeddings: backbone.embedTokens(frames[0..., 0]))[.newAxis]
        }

        let prefillHidden = backbone(embeddings: backboneInput(rows), cache: backboneCache)[0]
        if rowCount > 1 {
            tower.prefill(
                hidden: prefillHidden[..<(rowCount - 1)],
                inputFrames: rows[..<(rowCount - 1)],
                nextFrames: rows[1...],
                caches: towerCaches
            )
        }
        var hidden = prefillHidden[(rowCount - 1)...]
        var inputFrame = rows[(rowCount - 1)...]
        // The RAS window starts with the main tokens of the last prompt rows, left-padded with -1.
        let promptTail = rows[max(0, rowCount - Self.rasWindowFrames)..., 0].asType(.int32)
        var window = concatenated([
            MLXArray([Int32](repeating: -1, count: Self.rasWindowFrames - promptTail.dim(0))),
            promptTail,
        ], axis: 0)
        var frames: [MLXArray] = []
        let semanticBegin = Int32(config.semanticBeginID)
        let codebookLimit = config.tower.codebookSize - 1

        for _ in 0 ..< sampling.maxFrames {
            let candidateLogits = model.lmHead(hidden)[0..., candidateIDs]
            let first = candidateIDs[Self.sample(candidateLogits, temperature: sampling.temperature, topK: sampling.topK, topP: sampling.topP)]
            let rasDraw = candidateIDs[Self.sample(candidateLogits, temperature: Self.rasTemperature, topK: Self.rasTopK, topP: Self.rasTopP)]
            let isRepeat = (window[.newAxis] .== first[0..., .newAxis]).any(axis: -1)
            let mainToken = MLX.where(isRepeat .&& tower.isSemantic(first), rasDraw, first)
            let codeZero = clip(mainToken - semanticBegin, min: 0, max: codebookLimit)
            let codes = tower.decode(
                hidden: hidden, inputFrame: inputFrame, mainToken: mainToken, codeZero: codeZero,
                caches: towerCaches
            ) { logits, _ in
                Self.sample(logits, temperature: sampling.temperature, topK: sampling.topK, topP: sampling.topP)
            }
            let frame = concatenated([mainToken] + codes, axis: 0).asType(.int32)[.newAxis]
            eval(frame)
            let main = frame[0, 0].item(Int.self)
            if main == config.eosTokenID {
                return (frames, true)
            }
            frames.append(frame)
            onFrame?(frames.count)
            window = concatenated([window[1...], frame[0, 0 ..< 1]], axis: 0)
            inputFrame = frame
            hidden = backbone(embeddings: backboneInput(frame), cache: backboneCache)[0]
        }
        return (frames, false)
    }
}
