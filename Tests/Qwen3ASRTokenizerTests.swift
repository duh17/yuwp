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
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)

        let vocab: [String: Int] = [
            "h": 1,
            "i": 2,
            "e": 3,
            "l": 4,
            "o": 5,
            "<asr_text>": Qwen3ASRTokenizer.asrText,
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: vocab, options: [.sortedKeys])
        try vocabData.write(to: dir.appendingPathComponent("vocab.json"))
        try Data("#version: 0.2\n".utf8).write(to: dir.appendingPathComponent("merges.txt"))

        let tokenizer = try Qwen3ASRTokenizer.load(from: dir)
        try? FileManager.default.removeItem(at: dir)
        return tokenizer
    }
}
