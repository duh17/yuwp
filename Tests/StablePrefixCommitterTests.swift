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

    @Test func leadingLanguageIsCapturedBeforeCleaningBothHeaderFormats() {
        #expect(StablePrefixCommitter.leadingLanguage(in: "language Japanese<asr_text>こんにちは") == "Japanese")
        #expect(StablePrefixCommitter.leadingLanguage(in: "language Chinese你好|") == "Chinese")
        #expect(StablePrefixCommitter.leadingLanguage(in: "language NoneSo what") == nil)
        #expect(StablePrefixCommitter.leadingLanguage(in: "We discuss language Japanese") == nil)
    }

    @Test func midTranscriptLanguageSwitchKeepsEnglishAndAppendsChinese() {
        let decoded = "In addition, language Chinese<asr_text>我来说中文现在怎么样"
        #expect(
            StablePrefixCommitter.stripMeta(decoded)
                == "In addition, 我来说中文现在怎么样"
        )
        let visible = StablePrefixCommitter.visibleTranscript(
            prefixText: "In addition,",
            decoded: decoded
        )
        #expect(visible.contains("In addition,"))
        #expect(visible.contains("我来说中文"))
        #expect(!visible.contains("language"))
        #expect(!visible.contains("<asr_text>"))
    }

    @Test func markerlessLanguageSwitchDoesNotLeakIntoLiveOrCommittedText() {
        let english = "In addition, "
        let switching = "In addition, language Chinese我来说中文"
        #expect(StablePrefixCommitter.stripMeta(switching) == "In addition, 我来说中文")
        #expect(
            StablePrefixCommitter.visibleTranscript(prefixText: english, decoded: switching)
                == "In addition, 我来说中文"
        )
        let result = commit(english, generated: switching + "|", unfixed: 0)
        #expect(result.committedText == "In addition, 我来说中文")
        #expect(result.committedDelta == "我来说中文")

        // A sampled header in progress must not flash as an unfixed live tail.
        for partial in [
            "In addition, language", "In addition, language ",
            "In addition, language Chin", "In addition, language Chinese",
        ] {
            #expect(StablePrefixCommitter.visibleTranscript(prefixText: english, decoded: partial) == english)
        }
        // Endpoint/stop (no holdback) keeps a finished spoken "language",
        // but still strips `language ` plus a name or partial name.
        #expect(commit(english, generated: "In addition, language", unfixed: 0).committedText == "In addition, language")
        for header in ["In addition, language ", "In addition, language Chin", "In addition, language Chinese"] {
            #expect(commit(english, generated: header, unfixed: 0).committedText == english)
        }
        #expect(StablePrefixCommitter.stripMeta("language NoneSo what") == "So what")
        #expect(StablePrefixCommitter.stripMeta("In addition, language NoneSo what") == "In addition, So what")
        #expect(StablePrefixCommitter.stripMeta("language is a tool") == "language is a tool")
        #expect(
            StreamingSession.preferStopBatch(
                streamed: "In addition, 我", batch: "In addition, language Chinese我来说中文"
            ) == "In addition, 我来说中文"
        )
    }

    @Test func cjkContinuesAfterEnglishPunctuation() {
        let visible = StablePrefixCommitter.visibleTranscript(
            prefixText: "In addition,",
            decoded: "我来说中文"
        )
        #expect(visible == "In addition,我来说中文")
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

    @Test func stopBatchKeepsLongerStreamedText() {
        #expect(StreamingSession.preferStopBatch(streamed: "hello world today", batch: "hello") == "hello world today")
        #expect(StreamingSession.preferStopBatch(streamed: "hello", batch: "hello world today") == "hello world today")
        #expect(StreamingSession.preferStopBatch(streamed: "", batch: "hello") == "hello")
        #expect(StreamingSession.preferStopBatch(streamed: "hello", batch: "") == "hello")
    }

    @Test func r2t2StopDoesNotFullSessionBatchALongSession() {
        let speech = SpeechEvidence(vadSpeechDurationSec: SpeechEvidence.minimumVADSpeechDurationSec)
        let overCap = StreamingSession.maxLiveBatchSegmentSamples + 1
        let fortyFiveSeconds = StreamingSession.maxSessionContextSamples

        #expect(overCap > StreamingSession.maxLiveBatchSegmentSamples)
        #expect(fortyFiveSeconds > StreamingSession.maxLiveBatchSegmentSamples)
        #expect(
            StreamingSession.stablePrefixStopBatchStrategy(
                sessionAudioSampleCount: overCap,
                activeSpeechEvidence: speech
            ) == .none
        )
        #expect(
            StreamingSession.stablePrefixStopBatchStrategy(
                sessionAudioSampleCount: fortyFiveSeconds,
                activeSpeechEvidence: speech
            ) == .none
        )
    }

    @Test func r2t2ShortEligibleSessionMayFullSessionBatch() {
        let speech = SpeechEvidence(vadSpeechDurationSec: SpeechEvidence.minimumVADSpeechDurationSec)
        let silent = SpeechEvidence()
        let short = ASRAudio.sampleRate * 4
        let atCap = StreamingSession.maxLiveBatchSegmentSamples
        let r2t2 = StreamConfig.stablePrefix()

        #expect(r2t2.decodeMode == .stablePrefix)
        #expect(r2t2.batchRetranscribe == false)
        #expect(
            StreamingSession.stopBatchStrategy(
                config: r2t2,
                sessionAudioSampleCount: short,
                activeAudioSampleCount: short,
                activeSpeechEvidence: speech
            ) == .none
        )
        #expect(
            StreamingSession.stablePrefixStopBatchStrategy(
                sessionAudioSampleCount: short,
                activeSpeechEvidence: speech
            ) == .fullSession
        )
        #expect(
            StreamingSession.stablePrefixStopBatchStrategy(
                sessionAudioSampleCount: atCap,
                activeSpeechEvidence: speech
            ) == .fullSession
        )
        #expect(
            StreamingSession.stablePrefixStopBatchStrategy(
                sessionAudioSampleCount: short,
                activeSpeechEvidence: silent
            ) == .none
        )
    }

    @Test func qwenStopBatchStrategyStaysRollbackBatch() {
        let speech = SpeechEvidence(vadSpeechDurationSec: SpeechEvidence.minimumVADSpeechDurationSec)
        let qwen = StreamConfig(batchRetranscribe: true)
        #expect(qwen.decodeMode == .rollbackBatch)
        #expect(
            StreamingSession.stopBatchStrategy(
                config: qwen,
                sessionAudioSampleCount: ASRAudio.sampleRate * 6,
                activeAudioSampleCount: ASRAudio.sampleRate * 2,
                activeSpeechEvidence: speech
            ) == .activeSegmentOnly
        )
        #expect(
            StreamingSession.stopBatchStrategy(
                config: qwen,
                sessionAudioSampleCount: StreamingSession.maxSessionContextSamples,
                activeAudioSampleCount: ASRAudio.sampleRate * 2,
                activeSpeechEvidence: speech
            ) == .activeSegmentOnly
        )
    }

    @Test func trailingLanguageWordSurvivesEndpointCommitAndHidesWhileLive() {
        let spoken = "until I switched language"
        #expect(StablePrefixCommitter.stripMeta(spoken) == spoken)
        #expect(
            StablePrefixCommitter.stripMeta(spoken, hidePartialLanguageHeader: true)
                == "until I switched "
        )

        #expect(
            StablePrefixCommitter.visibleTranscript(
                prefixText: "until I switched ",
                decoded: spoken
            ) == "until I switched "
        )
        #expect(
            StablePrefixCommitter.visibleTranscript(
                prefixText: "until I switched ",
                decoded: spoken,
                hidePartialLanguageHeader: false
            ) == spoken
        )
        #expect(
            StablePrefixCommitter.visibleTranscript(
                prefixText: spoken,
                decoded: spoken
            ) == spoken
        )

        #expect(StablePrefixCommitter.stripLanguageHeader("language", hidePartialLanguageHeader: true) == "")
        #expect(StablePrefixCommitter.stripLanguageHeader("language", hidePartialLanguageHeader: false) == "language")
        #expect(
            StablePrefixCommitter.visibleTranscript(
                prefixText: "",
                decoded: "language",
                hidePartialLanguageHeader: true
            ).isEmpty
        )
        #expect(
            StablePrefixCommitter.visibleTranscript(
                prefixText: "",
                decoded: "language",
                hidePartialLanguageHeader: false
            ) == "language"
        )

        let endpoint = commit("", generated: spoken + "|", unfixed: 0)
        #expect(endpoint.committedText == spoken)
        #expect(endpoint.unstableTail.isEmpty)

        #expect(
            StreamingSession.preferStablePrefixStopText(
                displayText: spoken,
                flushedText: "until I switched "
            ) == spoken
        )
    }

    @Test func stablePrefixDisplayTextUpdateRule() {
        #expect(
            StreamingSession.stablePrefixUpdatedDisplayText(
                displayText: "hello",
                committedText: "hello",
                candidate: "hello world"
            ) == "hello world"
        )
        #expect(
            StreamingSession.stablePrefixUpdatedDisplayText(
                displayText: "hello world",
                committedText: "hello",
                candidate: "hello"
            ) == "hello world"
        )
        #expect(
            StreamingSession.stablePrefixUpdatedDisplayText(
                displayText: "hello world",
                committedText: "hello",
                candidate: "hello there"
            ) == "hello there"
        )
        #expect(
            StreamingSession.stablePrefixUpdatedDisplayText(
                displayText: "hello world",
                committedText: "hello world",
                candidate: "goodbye"
            ) == "hello world"
        )
        #expect(
            StreamingSession.stablePrefixUpdatedDisplayText(
                displayText: "hello",
                committedText: "hello world",
                candidate: "hello"
            ) == "hello world"
        )
        #expect(
            StreamingSession.stablePrefixUpdatedDisplayText(
                displayText: "hello world",
                committedText: "hello",
                candidate: "hello world?"
            ) == "hello world?"
        )
    }

    @Test func stablePrefixLookaheadIgnoresUpdatesWithoutNewLetterOrNumber() {
        let question = StreamingSession.stablePrefixLookaheadDisplayUpdate(
            displayText: "hello world",
            committedText: "hello",
            candidate: "hello world?"
        )
        #expect(question.text == "hello world")
        #expect(!question.displayChanged)

        let period = StreamingSession.stablePrefixLookaheadDisplayUpdate(
            displayText: "hello world",
            committedText: "hello",
            candidate: "hello world."
        )
        #expect(period.text == "hello world")
        #expect(!period.displayChanged)

        let spaceAndMark = StreamingSession.stablePrefixLookaheadDisplayUpdate(
            displayText: "hello world",
            committedText: "hello",
            candidate: "hello world ?"
        )
        #expect(spaceAndMark.text == "hello world")
        #expect(!spaceAndMark.displayChanged)

        let tailQuestion = StreamingSession.stablePrefixLookaheadDisplayUpdate(
            displayText: "hello world",
            committedText: "hello",
            candidate: "hello?"
        )
        #expect(tailQuestion.text == "hello world")
        #expect(!tailQuestion.displayChanged)

        let cjkTailQuestion = StreamingSession.stablePrefixLookaheadDisplayUpdate(
            displayText: "你好世界",
            committedText: "你好",
            candidate: "你好？"
        )
        #expect(cjkTailQuestion.text == "你好世界")
        #expect(!cjkTailQuestion.displayChanged)

        let bareQuestion = StreamingSession.stablePrefixLookaheadDisplayUpdate(
            displayText: "hello",
            committedText: "hello",
            candidate: "hello?"
        )
        #expect(bareQuestion.text == "hello")
        #expect(!bareQuestion.displayChanged)

        let cjkBareQuestion = StreamingSession.stablePrefixLookaheadDisplayUpdate(
            displayText: "你好",
            committedText: "你好",
            candidate: "你好？"
        )
        #expect(cjkBareQuestion.text == "你好")
        #expect(!cjkBareQuestion.displayChanged)

        let words = StreamingSession.stablePrefixLookaheadDisplayUpdate(
            displayText: "hello",
            committedText: "hello",
            candidate: "hello world"
        )
        #expect(words.text == "hello world")
        #expect(words.displayChanged)

        let wordsThenMark = StreamingSession.stablePrefixLookaheadDisplayUpdate(
            displayText: "hello",
            committedText: "hello",
            candidate: "hello world?"
        )
        #expect(wordsThenMark.text == "hello world?")
        #expect(wordsThenMark.displayChanged)

        let rewriteWords = StreamingSession.stablePrefixLookaheadDisplayUpdate(
            displayText: "hello world",
            committedText: "hello",
            candidate: "hello there"
        )
        #expect(rewriteWords.text == "hello there")
        #expect(rewriteWords.displayChanged)

        let cjkExtend = StreamingSession.stablePrefixLookaheadDisplayUpdate(
            displayText: "你",
            committedText: "你",
            candidate: "你好"
        )
        #expect(cjkExtend.text == "你好")
        #expect(cjkExtend.displayChanged)
    }

    @Test func stablePrefixQuietAdmitDecision() {
        let leadingQuiet = StreamingSession.stablePrefixChunkDecision(
            chunkHasSpeech: false, rms: 0.001, hasSpeech: false
        )
        #expect(leadingQuiet.action == .waitForSpeech)
        #expect(!leadingQuiet.appendAudio)

        let leadingNoisy = StreamingSession.stablePrefixChunkDecision(
            chunkHasSpeech: false, rms: 0.004, hasSpeech: false
        )
        #expect(leadingNoisy.action == .waitForSpeech)
        #expect(leadingNoisy.appendAudio)

        let speech = StreamingSession.stablePrefixChunkDecision(
            chunkHasSpeech: true, rms: 0.05, hasSpeech: false
        )
        #expect(speech.action == .decode)
        #expect(speech.appendAudio)
        #expect(speech.hasSpeech)

        let noisyPause = StreamingSession.stablePrefixChunkDecision(
            chunkHasSpeech: false, rms: 0.004, hasSpeech: true
        )
        #expect(noisyPause.action == .decode)
        #expect(noisyPause.appendAudio)

        let quietPause = StreamingSession.stablePrefixChunkDecision(
            chunkHasSpeech: false, rms: 0.001, hasSpeech: true
        )
        #expect(quietPause.action == .skipQuietPause)
        #expect(!quietPause.appendAudio)
        #expect(quietPause.hasSpeech)

        let noisyAfterSkip = StreamingSession.stablePrefixChunkDecision(
            chunkHasSpeech: false, rms: 0.004, hasSpeech: true
        )
        #expect(noisyAfterSkip.action == .decode)
        #expect(noisyAfterSkip.appendAudio)
    }

    @Test func stablePrefixLookaheadBudgetTransitions() {
        let idle = StreamingSession.StablePrefixLookaheadPause.idle

        let firstQuiet = StreamingSession.stablePrefixLookaheadAdmit(
            chunkHasSpeech: false, rms: 0.001, hasSpeech: true, pause: idle
        )
        #expect(firstQuiet.runLookahead)
        #expect(!firstQuiet.resetPause)

        let afterChanged = StreamingSession.stablePrefixLookaheadAfterDecode(
            pause: idle, displayChanged: true
        )
        #expect(afterChanged.used == 1)
        #expect(afterChanged.allowMore)
        let nextQuiet = StreamingSession.stablePrefixLookaheadAdmit(
            chunkHasSpeech: false, rms: 0.001, hasSpeech: true, pause: afterChanged
        )
        #expect(nextQuiet.runLookahead)
        #expect(!nextQuiet.resetPause)

        let afterUnchanged = StreamingSession.stablePrefixLookaheadAfterDecode(
            pause: afterChanged, displayChanged: false
        )
        #expect(afterUnchanged.used == 2)
        #expect(!afterUnchanged.allowMore)
        let stopped = StreamingSession.stablePrefixLookaheadAdmit(
            chunkHasSpeech: false, rms: 0.001, hasSpeech: true, pause: afterUnchanged
        )
        #expect(!stopped.runLookahead)
        #expect(!stopped.resetPause)

        var capped = idle
        for step in 1 ... StreamingSession.StablePrefixLookaheadPause.maxDecodes {
            let admit = StreamingSession.stablePrefixLookaheadAdmit(
                chunkHasSpeech: false, rms: 0.001, hasSpeech: true, pause: capped
            )
            #expect(admit.runLookahead)
            capped = StreamingSession.stablePrefixLookaheadAfterDecode(
                pause: capped, displayChanged: true
            )
            #expect(capped.used == step)
        }
        #expect(!capped.allowMore)
        #expect(
            !StreamingSession.stablePrefixLookaheadAdmit(
                chunkHasSpeech: false, rms: 0.001, hasSpeech: true, pause: capped
            ).runLookahead
        )

        let speechResets = StreamingSession.stablePrefixLookaheadAdmit(
            chunkHasSpeech: true, rms: 0.05, hasSpeech: true, pause: afterUnchanged
        )
        #expect(speechResets.resetPause)
        #expect(!speechResets.runLookahead)
        #expect(
            StreamingSession.stablePrefixLookaheadAdmit(
                chunkHasSpeech: false, rms: 0.001, hasSpeech: true, pause: idle
            ).runLookahead
        )

        let noisyResets = StreamingSession.stablePrefixLookaheadAdmit(
            chunkHasSpeech: false, rms: 0.004, hasSpeech: true, pause: afterUnchanged
        )
        #expect(noisyResets.resetPause)
        #expect(!noisyResets.runLookahead)

        let leadingQuiet = StreamingSession.stablePrefixLookaheadAdmit(
            chunkHasSpeech: false, rms: 0.001, hasSpeech: false, pause: idle
        )
        #expect(!leadingQuiet.runLookahead)
        #expect(!leadingQuiet.resetPause)
    }

    @Test func visibleTranscriptCommitsHeldLastWordWithoutReencode() {
        let live = "but he did very well when he started writing for other"
        let withLast = "but he did very well when he started writing for other people"
        #expect(StablePrefixCommitter.visibleTranscript(prefixText: live, decoded: withLast) == withLast)
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
