// NativeASR — Qwen3-ASR Tokenizer
// Self-contained BPE tokenizer loading from vocab.json + merges.txt.
// Does not require swift-transformers or tokenizer_config.json.
//
// DESIGN:
// - Decode: vocab.json (ID→string) + GPT-2 byte-to-unicode inversion → UTF-8 output
// - Encode: BPE encoding for arbitrary text + special token handling
// - buildPrompt: hardcoded token IDs for the audio transcription template
//
// PROMPT TOKEN IDS (verified with Qwen3-ASR-0.6B tokenizer):
// <|im_start|>=151644, system=8948, \n=198, <|im_end|>=151645
// user=872, assistant=77091, <|audio_start|>=151669
// <|audio_pad|>=151676, <|audio_end|>=151670
// language English=[11528, 6364], <asr_text>=151704

import Foundation

public final class Qwen3ASRTokenizer: @unchecked Sendable {
    // MARK: - Special token IDs

    public static let imStart: Int = 151644
    public static let imEnd: Int = 151645
    public static let endOfText: Int = 151643
    public static let audioStart: Int = 151669
    public static let audioEnd: Int = 151670
    public static let audioPad: Int = 151676
    public static let asrText: Int = 151704

    public static let eosTokens: Set<Int> = [151645, 151643]
    public static let allLangTokens: Set<Int> = [11528, 6364, 8453, 22574, 44923, 151704]

    // MARK: - Internal state

    private let idToToken: [Int: String]
    private let tokenToId: [String: Int]
    private let bpeRanks: [BPEPair: Int]

    // GPT-2 byte-to-unicode decoder: unicode char → original byte
    private static let bytesDecoder: [Character: UInt8] = buildBytesDecoder()
    private static let bytesEncoder: [UInt8: Character] = {
        var enc: [UInt8: Character] = [:]
        for (c, b) in bytesDecoder { enc[b] = c }
        return enc
    }()

    // MARK: - Init

    private init(idToToken: [Int: String], tokenToId: [String: Int], bpeRanks: [BPEPair: Int]) {
        self.idToToken = idToToken
        self.tokenToId = tokenToId
        self.bpeRanks = bpeRanks
    }

    /// Load tokenizer from a model directory containing vocab.json + merges.txt.
    public static func load(from directory: URL) throws -> Qwen3ASRTokenizer {
        // Load vocab.json
        let vocabURL = directory.appendingPathComponent("vocab.json")
        guard FileManager.default.fileExists(atPath: vocabURL.path) else {
            throw Qwen3ASRError.tokenizerLoadFailed("vocab.json not found at \(vocabURL.path)")
        }
        let vocabData = try Data(contentsOf: vocabURL)
        let vocab = try JSONDecoder().decode([String: Int].self, from: vocabData)
        let idToToken = Dictionary(uniqueKeysWithValues: vocab.map { ($1, $0) })

        // Load merges.txt
        let mergesURL = directory.appendingPathComponent("merges.txt")
        guard FileManager.default.fileExists(atPath: mergesURL.path) else {
            throw Qwen3ASRError.tokenizerLoadFailed("merges.txt not found at \(mergesURL.path)")
        }
        let mergesText = try String(contentsOf: mergesURL, encoding: .utf8)
        var bpeRanks: [BPEPair: Int] = [:]
        var rank = 0
        for line in mergesText.split(separator: "\n") {
            let trimmed = String(line).trimmingCharacters(in: .whitespaces)
            if trimmed.isEmpty || trimmed.hasPrefix("#") { continue }
            let parts = trimmed.split(separator: " ", maxSplits: 1)
            if parts.count == 2 {
                bpeRanks[BPEPair(String(parts[0]), String(parts[1]))] = rank
                rank += 1
            }
        }

        return Qwen3ASRTokenizer(idToToken: idToToken, tokenToId: vocab, bpeRanks: bpeRanks)
    }

    // MARK: - Prompt Building

    /// Build the audio transcription prompt token IDs.
    ///
    /// Reference behavior from mlx-audio:
    /// - `language != nil` → assistant prefix is `language <lang><asr_text>`
    /// - `language == nil` → no assistant prefix at all (model auto-detects language)
    /// - no trailing newline after the assistant prefix
    public func buildPrompt(numAudioTokens: Int, language: String? = nil) -> [Int] {
        var tokens = [
            151644, 8948, 198, 151645, 198,     // <|im_start|>system\n<|im_end|>\n
            151644, 872, 198, 151669,            // <|im_start|>user\n<|audio_start|>
        ]
        tokens += Array(repeating: 151676, count: numAudioTokens)  // audio pads
        tokens += [
            151670, 151645, 198,                // <|audio_end|><|im_end|>\n
            151644, 77091, 198,                 // <|im_start|>assistant\n
        ]

        let lang = language?.trimmingCharacters(in: .whitespacesAndNewlines)
        if let lang, !lang.isEmpty, lang.lowercased() != "auto" {
            if lang.lowercased() == "english" || lang.lowercased() == "en" {
                tokens += [11528, 6364, 151704] // language English<asr_text>
            } else {
                tokens += encode("language \(lang)")
                tokens += [151704]
            }
        }

        return tokens
    }

    // MARK: - Decoding

    /// Decode token IDs to a UTF-8 string.
    public func decode(_ tokens: [Int]) -> String {
        var bytes: [UInt8] = []
        let dec = Self.bytesDecoder
        for tokenId in tokens {
            guard let tokenStr = idToToken[tokenId] else { continue }
            for char in tokenStr {
                if let byte = dec[char] {
                    bytes.append(byte)
                }
                // Characters not in bytesDecoder are dropped (shouldn't happen for valid BPE tokens)
            }
        }
        return String(bytes: bytes, encoding: .utf8) ?? String(bytes: bytes, encoding: .isoLatin1) ?? ""
    }

    /// Drop the auto-language prefix emitted by the model when language is nil.
    /// Expected prefix tokens are `language <lang> <asr_text>`.
    public func stripAutoLanguagePrefix(_ tokens: [Int]) -> [Int] {
        var index = 0
        while index < tokens.count, Self.allLangTokens.contains(tokens[index]) {
            index += 1
        }
        return Array(tokens[index...])
    }

    /// Clean ASR output: strip special tokens and extract transcribed text.
    public func cleanOutput(_ text: String) -> String {
        var cleaned = text

        // In auto-language mode, the model may emit `language <detected><asr_text><text>`.
        // Split on the delimiter before removing it so we keep the actual transcript.
        if let range = cleaned.range(of: "<asr_text>") {
            cleaned = String(cleaned[range.upperBound...])
        }

        let specials = [
            "<|im_start|>", "<|im_end|>", "<|endoftext|>",
            "<|audio_start|>", "<|audio_end|>", "<|audio_pad|>",
            "<asr_text>", "</asr_text>",
        ]
        for s in specials { cleaned = cleaned.replacingOccurrences(of: s, with: "") }

        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func isEOS(_ tokenId: Int) -> Bool {
        Self.eosTokens.contains(tokenId)
    }

    // MARK: - Encoding

    /// Encode arbitrary text to token IDs using BPE.
    /// Handles special tokens by looking them up directly in the vocab.
    public func encode(_ text: String) -> [Int] {
        // Fast path: single special token
        if let id = tokenToId[text] { return [id] }

        var result: [Int] = []
        // Simple word-level split (GPT-2 style: split on spaces, keep space with next word)
        let words = splitGPT2Style(text)
        for word in words {
            let bpeTokens = applyBPE(word)
            for token in bpeTokens {
                if let id = tokenToId[token] {
                    result.append(id)
                }
                // Unknown tokens are dropped
            }
        }
        return result
    }

    // MARK: - BPE Implementation

    private func applyBPE(_ word: String) -> [String] {
        // Convert word to byte-level representation
        let enc = Self.bytesEncoder
        var symbols: [String] = word.utf8.compactMap { byte in
            enc[byte].map { String($0) }
        }

        if symbols.isEmpty { return [] }
        if symbols.count == 1 { return symbols }

        while true {
            var bestPair: BPEPair? = nil
            var bestRank = Int.max

            for i in 0 ..< symbols.count - 1 {
                let pair = BPEPair(symbols[i], symbols[i + 1])
                if let rank = bpeRanks[pair], rank < bestRank {
                    bestRank = rank
                    bestPair = pair
                }
            }

            guard let best = bestPair else { break }

            var merged: [String] = []
            var i = 0
            while i < symbols.count {
                if i < symbols.count - 1,
                   symbols[i] == best.first,
                   symbols[i + 1] == best.second
                {
                    merged.append(best.first + best.second)
                    i += 2
                } else {
                    merged.append(symbols[i])
                    i += 1
                }
            }
            symbols = merged
            if symbols.count == 1 { break }
        }

        return symbols
    }

    /// Split text GPT-2 style: space is prepended to words except the first.
    private func splitGPT2Style(_ text: String) -> [String] {
        var words: [String] = []
        var current = ""
        for char in text {
            if char == " " {
                if !current.isEmpty {
                    words.append(current)
                }
                current = "\u{0120}" // Ġ (GPT-2 space prefix)
            } else {
                current.append(char)
            }
        }
        if !current.isEmpty { words.append(current) }
        return words
    }

    // MARK: - GPT-2 Byte Encoding

    private static func buildBytesDecoder() -> [Character: UInt8] {
        var decoder: [Character: UInt8] = [:]
        // Printable ASCII: ! (33) to ~ (126)
        for b: UInt8 in 33...126 { decoder[Character(UnicodeScalar(b))] = b }
        // ¡ (161) to ¬ (172)
        for b: UInt8 in 161...172 { decoder[Character(UnicodeScalar(b))] = b }
        // ® (174) to ÿ (255)
        for b: UInt8 in 174...255 { decoder[Character(UnicodeScalar(b))] = b }

        // Remaining bytes (0-32, 127-160, 173) → U+0100 onwards
        var next: UInt32 = 0x0100
        let mapped: Set<UInt8> = Set(Array(33...126) + Array(161...172) + Array(174...255))
        for b in 0...255 {
            let byte = UInt8(b)
            if !mapped.contains(byte) {
                if let scalar = Unicode.Scalar(next) {
                    decoder[Character(scalar)] = byte
                }
                next += 1
            }
        }

        return decoder
    }
}

// MARK: - BPEPair

struct BPEPair: Hashable {
    let first: String
    let second: String
    init(_ first: String, _ second: String) {
        self.first = first
        self.second = second
    }
}
