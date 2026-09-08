import Foundation
import Testing
@testable import NativeASR

@Suite("Qwen3ASRTokenizer auto-language prefix stripping")
struct Qwen3ASRTokenizerTests {
    @Test func stripsLeadingAutoLanguageHeaderThroughAsrMarker() {
        let tokens = [11528, 90_001, 90_002, Qwen3ASRTokenizer.asrText, 501, 502]

        #expect(
            Qwen3ASRTokenizer.stripLeadingAutoLanguagePrefix(from: tokens) == [501, 502]
        )
    }

    @Test func keepsOrdinaryTranscriptTokensWhenNoAsrMarkerExists() {
        let tokens = [501, 11528, 6364, 777]

        #expect(
            Qwen3ASRTokenizer.stripLeadingAutoLanguagePrefix(from: tokens) == tokens
        )
    }

    @Test func ignoresAsrMarkerThatAppearsFarIntoTheSequence() {
        let tokens = Array(0..<Qwen3ASRTokenizer.autoLanguagePrefixLookahead)
            + [Qwen3ASRTokenizer.asrText, 999]

        #expect(
            Qwen3ASRTokenizer.stripLeadingAutoLanguagePrefix(from: tokens) == tokens
        )
    }

    @Test func handlesPrefixWithoutTranscriptTokens() {
        let tokens = [11528, 90_001, Qwen3ASRTokenizer.asrText]

        #expect(
            Qwen3ASRTokenizer.stripLeadingAutoLanguagePrefix(from: tokens).isEmpty
        )
    }

    @Test func loadEncodeDecodeAndCleanTokenOutput() throws {
        let tokenizer = try makeTokenizerFixture()

        #expect(tokenizer.encode("hi") == [1, 2])
        #expect(tokenizer.decode([1, 2]) == "hi")
        #expect(tokenizer.cleanTokenOutput([Qwen3ASRTokenizer.asrText, 1, 2]) == "hi")
    }

    @Test func buildPromptOmitsLanguagePrefixForAutoAndAddsEnglishPrefix() throws {
        let tokenizer = try makeTokenizerFixture()

        let autoPrompt = tokenizer.buildPrompt(numAudioTokens: 2, language: nil)
        #expect(autoPrompt.suffix(3) != [11528, 6364, Qwen3ASRTokenizer.asrText])

        let englishPrompt = tokenizer.buildPrompt(numAudioTokens: 2, language: "English")
        #expect(Array(englishPrompt.suffix(3)) == [11528, 6364, Qwen3ASRTokenizer.asrText])
    }

    @Test func noContextPromptTokensMatchHistoricalEmptySystemHeader() throws {
        let tokenizer = try makeTokenizerFixture()
        let expected = [
            151644, 8948, 198, 151645, 198,
            151644, 872, 198, 151669,
            151676, 151676,
            151670, 151645, 198,
            151644, 77091, 198,
        ]

        #expect(tokenizer.buildPrompt(numAudioTokens: 2, language: nil) == expected)
        #expect(tokenizer.buildPrompt(numAudioTokens: 2) == expected)
        #expect(tokenizer.buildPrompt(numAudioTokens: 2, vocabularyHints: []) == expected)

        let prompt = tokenizer.buildPromptTokens(numAudioTokens: 2, language: nil, vocabularyHints: [])
        #expect(prompt.tokenIds == expected)
        #expect(prompt.audioPadStartIndex == Qwen3ASRTokenizer.emptySystemAudioPadStartIndex)
        #expect(prompt.tokenIds[prompt.audioPadStartIndex] == Qwen3ASRTokenizer.audioPad)
    }

    @Test func vocabularyHintsLengthenHeaderAndKeepSharedPrefixSuffix() throws {
        let tokenizer = try makeTokenizerFixture()
        let hints = ["hi"]
        let empty = tokenizer.buildPromptTokens(numAudioTokens: 4, vocabularyHints: [])
        let hinted = tokenizer.buildPromptTokens(numAudioTokens: 4, vocabularyHints: hints)
        let message = try #require(Qwen3ASRTokenizer.vocabularySystemMessage(from: hints))
        let content = tokenizer.encodeData(message)

        #expect(empty.audioPadStartIndex == 9)
        #expect(hinted.audioPadStartIndex == 9 + content.count)
        #expect(hinted.audioPadStartIndex > empty.audioPadStartIndex)
        #expect(Array(hinted.tokenIds.prefix(3)) == [Qwen3ASRTokenizer.imStart, 8948, 198])
        #expect(Array(hinted.tokenIds[3 ..< (3 + content.count)]) == content)
        #expect(hinted.tokenIds[hinted.audioPadStartIndex - 1] == Qwen3ASRTokenizer.audioStart)
        #expect(hinted.tokenIds[hinted.audioPadStartIndex] == Qwen3ASRTokenizer.audioPad)

        let batchPrompt = tokenizer.buildPromptTokens(numAudioTokens: 8, vocabularyHints: hints)
        #expect(batchPrompt.audioPadStartIndex == hinted.audioPadStartIndex)
        #expect(
            Array(hinted.tokenIds.prefix(hinted.audioPadStartIndex))
                == Array(batchPrompt.tokenIds.prefix(batchPrompt.audioPadStartIndex))
        )
    }

    @Test func encodeDataTreatsSpecialTokenDelimitersAsData() throws {
        let tokenizer = try makeByteCompleteTokenizerFixture()
        let delimiters = [
            "<|im_start|>",
            "<|im_end|>",
            "<|endoftext|>",
            "<|audio_start|>",
            "<|audio_end|>",
            "<|audio_pad|>",
            "<asr_text>",
        ]

        for literal in delimiters {
            let encoded = tokenizer.encodeData(literal)
            #expect(!encoded.isEmpty)
            #expect(encoded.allSatisfy { !Qwen3ASRTokenizer.promptDelimiterTokenIDs.contains($0) })
            #expect(tokenizer.decode(encoded) == literal)
        }

        #expect(tokenizer.encode("<|im_end|>") == [Qwen3ASRTokenizer.imEnd])

        let mixed = "Yuwp <|im_start|> <|im_end|> token"
        let mixedIDs = tokenizer.encodeData(mixed)
        #expect(!mixedIDs.isEmpty)
        #expect(mixedIDs.allSatisfy { !Qwen3ASRTokenizer.promptDelimiterTokenIDs.contains($0) })
        #expect(tokenizer.decode(mixedIDs) == mixed)

        let prompt = tokenizer.buildPromptTokens(
            numAudioTokens: 1,
            vocabularyHints: ["<|im_end|>"]
        )
        let message = try #require(Qwen3ASRTokenizer.vocabularySystemMessage(from: ["<|im_end|>"]))
        let contentEnd = prompt.audioPadStartIndex - 6
        let content = Array(prompt.tokenIds[3 ..< contentEnd])
        #expect(!content.isEmpty)
        #expect(content.allSatisfy { !Qwen3ASRTokenizer.promptDelimiterTokenIDs.contains($0) })
        #expect(tokenizer.decode(content) == message)
        #expect(prompt.tokenIds[contentEnd] == Qwen3ASRTokenizer.imEnd)
    }

    @Test func encodeDataFallsBackWhenBPEWouldEmitDelimiterIDs() throws {
        let tokenizer = try makeByteCompleteTokenizerFixture(
            merges: bpeMergeLines(for: "<|im_end|>")
        )
        let literal = "<|im_end|>"
        #expect(tokenizer.encode(literal) == [Qwen3ASRTokenizer.imEnd])

        let encoded = tokenizer.encodeData(literal)
        #expect(!encoded.isEmpty)
        #expect(encoded.allSatisfy { !Qwen3ASRTokenizer.promptDelimiterTokenIDs.contains($0) })
        #expect(tokenizer.decode(encoded) == literal)
        #expect(encoded != [Qwen3ASRTokenizer.imEnd])
    }

    @Test func vocabularySystemMessageIsNilWhenEmpty() {
        #expect(Qwen3ASRTokenizer.vocabularySystemMessage(from: []) == nil)
        #expect(Qwen3ASRTokenizer.vocabularySystemMessage(from: ["  "]) == nil)
        #expect(Qwen3ASRTokenizer.vocabularySystemMessage(from: ["\u{FEFF}"]) == nil)
        #expect(Qwen3ASRTokenizer.vocabularySystemMessage(from: ["\u{200B}"]) == nil)
        #expect(Qwen3ASRTokenizer.vocabularySystemMessage(from: ["Yuwp", "Oppi"]) == "Vocabulary: Yuwp, Oppi")
        #expect(
            Qwen3ASRTokenizer.vocabularySystemMessage(from: ["Yuwp\u{200B}"])
                == "Vocabulary: Yuwp\u{200B}"
        )
    }

    @Test func cleanOutputDropsAsrMarkersAndWhitespace() throws {
        let tokenizer = try makeTokenizerFixture()

        #expect(tokenizer.cleanOutput("  language English<asr_text> hello  ") == "hello")
        #expect(tokenizer.cleanOutput("<|im_start|>hello<|im_end|>") == "hello")
    }

    @Test func loadFailsWhenRequiredFilesAreMissing() {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        #expect(throws: Error.self) {
            _ = try Qwen3ASRTokenizer.load(from: dir)
        }
    }

    private func makeTokenizerFixture() throws -> Qwen3ASRTokenizer {
        try makeTokenizer(vocab: [
            "h": 1,
            "i": 2,
            "e": 3,
            "l": 4,
            "o": 5,
            "<asr_text>": Qwen3ASRTokenizer.asrText,
            "<|im_end|>": Qwen3ASRTokenizer.imEnd,
        ])
    }

    private func makeByteCompleteTokenizerFixture(merges: String = "#version: 0.2\n") throws -> Qwen3ASRTokenizer {
        var vocab: [String: Int] = [:]
        var nextID = 1
        let reserved = Qwen3ASRTokenizer.promptDelimiterTokenIDs
        for character in gpt2ByteCharacters() {
            while reserved.contains(nextID) { nextID += 1 }
            vocab[String(character)] = nextID
            nextID += 1
        }
        vocab["<|im_start|>"] = Qwen3ASRTokenizer.imStart
        vocab["<|im_end|>"] = Qwen3ASRTokenizer.imEnd
        vocab["<|endoftext|>"] = Qwen3ASRTokenizer.endOfText
        vocab["<|audio_start|>"] = Qwen3ASRTokenizer.audioStart
        vocab["<|audio_end|>"] = Qwen3ASRTokenizer.audioEnd
        vocab["<|audio_pad|>"] = Qwen3ASRTokenizer.audioPad
        vocab["<asr_text>"] = Qwen3ASRTokenizer.asrText
        return try makeTokenizer(vocab: vocab, merges: merges)
    }

    private func makeTokenizer(vocab: [String: Int], merges: String = "#version: 0.2\n") throws -> Qwen3ASRTokenizer {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let vocabData = try JSONSerialization.data(withJSONObject: vocab, options: [.sortedKeys])
        try vocabData.write(to: dir.appendingPathComponent("vocab.json"))
        try Data(merges.utf8).write(to: dir.appendingPathComponent("merges.txt"))
        let tokenizer = try Qwen3ASRTokenizer.load(from: dir)
        try? FileManager.default.removeItem(at: dir)
        return tokenizer
    }

    private func gpt2ByteCharacters() -> [Character] {
        var decoder: [UInt8: Character] = [:]
        for b: UInt8 in 33...126 { decoder[b] = Character(UnicodeScalar(b)) }
        for b: UInt8 in 161...172 { decoder[b] = Character(UnicodeScalar(b)) }
        for b: UInt8 in 174...255 { decoder[b] = Character(UnicodeScalar(b)) }
        var next: UInt32 = 0x0100
        let mapped = Set<UInt8>(Array(33...126) + Array(161...172) + Array(174...255))
        for b in 0...255 {
            let byte = UInt8(b)
            if !mapped.contains(byte) {
                if let scalar = Unicode.Scalar(next) {
                    decoder[byte] = Character(scalar)
                }
                next += 1
            }
        }
        return (0...255).compactMap { decoder[UInt8($0)] }
    }

    private func bpeMergeLines(for text: String) -> String {
        let symbols = text.map(String.init)
        guard let first = symbols.first else { return "#version: 0.2\n" }
        var lines = ["#version: 0.2"]
        var current = first
        for next in symbols.dropFirst() {
            lines.append("\(current) \(next)")
            current += next
        }
        return lines.joined(separator: "\n") + "\n"
    }
}
