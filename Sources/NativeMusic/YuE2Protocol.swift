import Foundation

/// Checkpoint-native YuE2 token layout, prompts and generation defaults (`yue2-native-v1`).
public enum YuE2Protocol {
    public static let eod = 151_643
    public static let abcStart = 151_847
    public static let abcEnd = 151_848
    public static let musicStart = 151_851
    public static let musicEnd = 151_852
    public static let codecOffset = 151_853
    public static let codecSize = 32768
    public static let vocabSize = 184_704
    public static let context = 24576
    /// One semantic id covers 1920 samples at 48 kHz.
    public static let frameRate = 25
    public static let sampleRate = 48000
    public static let latentDim = 64

    /// Local rows sampled per phase: ABC is ordinary text `[0, EOD)` plus ABC_END;
    /// semantic is MUSIC_END followed by the 32768 codec ids.
    public static func phaseSize(_ phase: YuE2Phase) -> Int {
        phase == .abc ? eod + 1 : codecSize + 1
    }

    public static func endIndex(_ phase: YuE2Phase) -> Int {
        phase == .abc ? eod : 0
    }

    public static func nativeToken(local: Int, phase: YuE2Phase) -> Int {
        switch phase {
        case .abc: local < eod ? local : abcEnd
        case .semantic: musicEnd + local
        }
    }

    /// Acoustic chunks the NAR solves independently: half the context left after the prefix.
    public static func chunkRanges(frames: Int, prefixTokens: Int) throws -> [Range<Int>] {
        let size = min((context - prefixTokens - 3) / 2, context)
        guard frames >= 1, size >= 1 else {
            throw YuE2Error.invalidInput("empty codec or a prefix that leaves no acoustic context")
        }
        return stride(from: 0, to: frames, by: size).map { $0 ..< min($0 + size, frames) }
    }
}

public enum YuE2Phase: String, Sendable {
    case abc
    case semantic
}

/// Score planning: `full` writes melody + chords, `melody` a chord-free melody, `off` skips the score.
public enum YuE2Cot: String, Sendable, CaseIterable {
    case off
    case melody
    case full

    public var instruction: String {
        switch self {
        case .off: "Generate music with codec tokens from the given conditions."
        case .melody: "Generate a melody-only ABC transcription without chord symbols, then generate music with codec tokens from the given conditions."
        case .full: "Generate a chord-annotated ABC transcription, then generate music with codec tokens from the given conditions."
        }
    }
}

public struct YuE2Sampling: Sendable, Equatable {
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var repetitionPenalty: Float
    public var penaltyWindow: Int
    public var minTokens: Int
    public var maxTokens: Int

    public init(temperature: Float, topP: Float, topK: Int, repetitionPenalty: Float, penaltyWindow: Int, minTokens: Int, maxTokens: Int) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.repetitionPenalty = repetitionPenalty
        self.penaltyWindow = penaltyWindow
        self.minTokens = minTokens
        self.maxTokens = maxTokens
    }

    public static let abcDefault = YuE2Sampling(temperature: 0.7, topP: 0.9, topK: 30, repetitionPenalty: 1.005, penaltyWindow: 100, minTokens: 32, maxTokens: 4096)
    public static let semanticDefault = YuE2Sampling(temperature: 1.0, topP: 0.95, topK: 100, repetitionPenalty: 1.2, penaltyWindow: 50, minTokens: 200, maxTokens: 9000)

    func validate() throws {
        guard temperature.isFinite, (0 ... 5).contains(temperature), topP > 0, topP <= 1, topK >= 1,
              repetitionPenalty > 0, (1 ... 100).contains(penaltyWindow),
              minTokens >= 0, minTokens <= maxTokens, maxTokens >= 1
        else { throw YuE2Error.invalidInput("invalid sampling settings \(self)") }
    }
}

public struct YuE2Request: Sendable {
    public var style: String
    public var lyrics: String
    public var cot: YuE2Cot
    public var seed: UInt64
    /// Supplied ABC score; YuE2 renders it instead of planning one (cot must be melody or full).
    public var abc: String?
    /// Classifier-free guidance; nil uses the protocol default (1.01 for `off`, otherwise 1.0).
    public var cfgScale: Float?

    public init(style: String, lyrics: String, cot: YuE2Cot = .full, seed: UInt64 = 831_001, abc: String? = nil, cfgScale: Float? = nil) {
        self.style = style
        self.lyrics = lyrics
        self.cot = cot
        self.seed = seed
        self.abc = abc
        self.cfgScale = cfgScale
    }

    public var guidance: Float { cfgScale ?? (cot == .off ? 1.01 : 1.0) }

    public var promptText: String {
        "\(cot.instruction)\n[Tags]\n\(style)\n[Lyrics]\n\(lyrics)\n"
    }

    func validate() throws {
        guard seed < (UInt64(1) << 63) else { throw YuE2Error.invalidInput("seed must be below 2^63") }
        if let abc {
            guard cot != .off, !abc.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw YuE2Error.invalidInput("a supplied ABC score needs nonempty text and --mode melody or full")
            }
        }
        if let cfgScale, !(cfgScale.isFinite && (0 ... 20).contains(cfgScale)) {
            throw YuE2Error.invalidInput("cfg scale must be in 0...20")
        }
    }

    /// Positive prefix. Without `abcIDs` and without a supplied score it stops at ABC_START (planning).
    func prefix(tokenizer: YuE2Tokenizer, abcIDs: [Int]?) -> [Int] {
        let base = [YuE2Protocol.eod] + tokenizer.encode(promptText)
        if cot == .off {
            return base + [YuE2Protocol.abcStart, YuE2Protocol.abcEnd, YuE2Protocol.musicStart]
        }
        guard let abcIDs else { return base + [YuE2Protocol.abcStart] }
        return base + [YuE2Protocol.abcStart] + abcIDs + [YuE2Protocol.abcEnd, YuE2Protocol.musicStart]
    }

    /// Unconditional CFG branch: the instruction only, with the exact positive-branch ABC ids.
    func negativePrefix(tokenizer: YuE2Tokenizer, abcIDs: [Int]?) -> [Int] {
        let base = [YuE2Protocol.eod] + tokenizer.encode(cot.instruction)
        if cot == .off { return base + [YuE2Protocol.musicStart] }
        return base + [YuE2Protocol.abcStart] + (abcIDs ?? []) + [YuE2Protocol.abcEnd, YuE2Protocol.musicStart]
    }
}

public enum YuE2Error: Error, CustomStringConvertible {
    case invalidInput(String)
    case missingWeights(String)
    case invalidConfig(String)
    case numerical(String)

    public var description: String {
        switch self {
        case let .invalidInput(message): "invalid input: \(message)"
        case let .missingWeights(message): "missing weights: \(message)"
        case let .invalidConfig(message): "invalid config: \(message)"
        case let .numerical(message): "numerical failure: \(message)"
        }
    }
}
