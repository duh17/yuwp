import Foundation
import Testing
@testable import NativeASR

@Suite("StablePrefixCommitter")
struct StablePrefixCommitterTests {
    @Test func cutsAtFirstPipeAndRollsBackUnfixedScalars() {
        let result = commit("", generated: "And so my| extra", unfixed: 1)

        #expect(!result.committedText.contains("|"))
        #expect(!result.unstableTail.contains("|"))
        #expect(result.committedText == "And so m")
        #expect(result.unstableTail == "y")
        #expect(result.committedDelta == "And so m")
    }

    @Test func generatedThatAlreadyIncludesPrefixIsNotDoubled() {
        let result = commit("And so", generated: "And so, my fellow|", unfixed: 1)

        #expect(result.committedText.hasPrefix("And so"))
        #expect(result.committedText == "And so, my fello")
        #expect(result.unstableTail == "w")
    }

    @Test func appendOnlyGuardKeepsPrefixWhenHypothesisDiverges() {
        let result = commit("And so, my fellow Americans", generated: "hello world|", unfixed: 1)

        #expect(result.committedText == "And so, my fellow Americans")
        #expect(result.unstableTail.isEmpty)
        #expect(result.committedDelta.isEmpty)
    }

    @Test func doesNotShrinkAlreadyCommittedPrefix() {
        let result = commit("And so my", generated: "|", unfixed: 1)

        #expect(result.committedText == "And so my")
        #expect(result.unstableTail.isEmpty)
    }

    @Test func stripsAsrTextMetaBeforeThePipeCut() {
        let result = commit("", generated: "language English<asr_text>甚至出现|", unfixed: 1)

        #expect(result.committedText == "甚至出")
        #expect(result.unstableTail == "现")
        #expect(!result.committedText.contains("<asr_text>"))
    }

    @Test func truncatesAtReplacementCharacterInsteadOfCommittingIt() {
        let result = commit("甚至", generated: "出现\u{FFFD}交易|", unfixed: 0)

        #expect(result.committedText == "甚至出现")
        #expect(!result.committedText.contains("\u{FFFD}"))
        #expect(!result.committedText.contains("交易"))
    }

    @Test func zeroUnfixedCommitsEverythingBeforePipe() {
        let result = commit("甚至出现", generated: "交易几乎停滞的情况。|", unfixed: 0)

        #expect(result.committedText == "甚至出现交易几乎停滞的情况。")
        #expect(result.unstableTail.isEmpty)
        #expect(result.committedDelta == "交易几乎停滞的情况。")
    }

    @Test func whitespaceContinuationIsConcatenated() {
        let result = commit("And so", generated: " my fellow|", unfixed: 1)

        #expect(result.committedText == "And so my fello")
        #expect(result.unstableTail == "w")
        #expect(result.committedDelta == " my fello")
    }

    @Test func languageHeaderOnlyIsDetectedBeforeAsrMarker() {
        #expect(StablePrefixCommitter.isLanguageHeaderOnly("language"))
        #expect(StablePrefixCommitter.isLanguageHeaderOnly("language English"))
        #expect(StablePrefixCommitter.isLanguageHeaderOnly("language None"))
        #expect(!StablePrefixCommitter.isLanguageHeaderOnly("language English And so"))
        #expect(!StablePrefixCommitter.isLanguageHeaderOnly("language NoneSo what about the ownership"))
        #expect(!StablePrefixCommitter.isLanguageHeaderOnly("And so my"))
        #expect(!StablePrefixCommitter.isLanguageHeaderOnly("language is a tool"))
    }

    @Test func stripsGluedAutoLanguageHeaderWithoutAsrMarker() {
        #expect(
            StablePrefixCommitter.stripMeta("language NoneSo what about the ownership")
                == "So what about the ownership"
        )
        #expect(StablePrefixCommitter.stripMeta("language None").isEmpty)
        #expect(StablePrefixCommitter.stripMeta("language English And so") == "And so")
        #expect(StablePrefixCommitter.stripMeta("language is a tool") == "language is a tool")
    }

    @Test func commitStripsLanguageNoneGluedToTranscript() {
        let first = commit("", generated: "language NoneSo what about the ownership|", unfixed: 0)

        #expect(first.committedText == "So what about the ownership")
        #expect(!first.committedText.contains("language"))
        #expect(!first.committedText.contains("None"))

        let next = commit(
            first.committedText,
            generated: "language NoneSo what about the ownership of this|",
            unfixed: 0
        )
        #expect(next.committedText == "So what about the ownership of this")
        #expect(next.committedDelta == " of this")
    }

    @Test func visibleTranscriptIsAppendOnlyAfterHeaderStrip() {
        let first = StablePrefixCommitter.visibleTranscript(
            prefixText: "",
            decoded: "language NoneSo what about"
        )
        #expect(first == "So what about")

        let next = StablePrefixCommitter.visibleTranscript(
            prefixText: first,
            decoded: "language NoneSo what about the ownership"
        )
        #expect(next == "So what about the ownership")

        let divergent = StablePrefixCommitter.visibleTranscript(
            prefixText: next,
            decoded: "hello world|"
        )
        #expect(divergent == next)
    }

    @Test func emptyGeneratedKeepsPrefix() {
        let result = commit("And so", generated: "", unfixed: 1)

        #expect(result.committedText == "And so")
        #expect(result.unstableTail.isEmpty)
        #expect(result.committedDelta.isEmpty)
    }

    @Test func r2t2ModelPathsSelectStablePrefixAndQwenKeepsRollback() {
        let r2t2 = URL(fileURLWithPath: "/tmp/mlx-community--Confucius4-R2T2-bf16")
        let eightBit = URL(fileURLWithPath: "/models/Confucius4-R2T2-8bit")
        let qwen = URL(fileURLWithPath: "/tmp/mlx-community--Qwen3-ASR-1.7B-bf16")

        #expect(StreamConfig.isR2T2Model(at: r2t2))
        #expect(StreamConfig.isR2T2Model(at: eightBit))
        #expect(!StreamConfig.isR2T2Model(at: qwen))

        let r2t2Config = StreamConfig.forModel(at: r2t2)
        #expect(r2t2Config.decodeMode == .stablePrefix)
        #expect(r2t2Config.chunkSec == 0.16)
        #expect(r2t2Config.batchRetranscribe == false)

        let qwenConfig = StreamConfig.forModel(at: qwen, batchRetranscribe: true)
        #expect(qwenConfig.decodeMode == .rollbackBatch)
        #expect(qwenConfig.chunkSec == 1.75)
        #expect(qwenConfig.rollback == 5)
        #expect(qwenConfig.unfixedChunks == 2)
        #expect(qwenConfig.maxPrefixTokens == 20)
        #expect(qwenConfig.repetitionPenalty == 1.3)
        #expect(qwenConfig.batchRetranscribe == true)
    }

    @Test func stablePrefixConfigDisablesLiveRewriteKnobs() {
        let config = StreamConfig.stablePrefix(chunkSec: 0.16)

        #expect(config.decodeMode == .stablePrefix)
        #expect(config.chunkSec == 0.16)
        #expect(config.rollback == 1)
        #expect(config.unfixedChunks == 0)
        #expect(config.maxNewTokens == 4)
        #expect(config.maxPrefixTokens == 0)
        #expect(config.batchRetranscribe == false)
        #expect(config.repetitionPenalty == 1.0)
    }
}

private func commit(
    _ prefix: String,
    generated: String,
    unfixed: Int
) -> StablePrefixCommitter.Result {
    StablePrefixCommitter.commit(
        prefixText: prefix,
        generatedText: generated,
        unfixedTokenCount: unfixed,
        encode: scalarEncode,
        decode: scalarDecode
    )
}

private func scalarEncode(_ text: String) -> [Int] {
    text.unicodeScalars.map { Int($0.value) }
}

private func scalarDecode(_ ids: [Int]) -> String {
    String(String.UnicodeScalarView(ids.compactMap(UnicodeScalar.init)))
}
