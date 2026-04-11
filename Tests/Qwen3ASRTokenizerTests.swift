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
}
