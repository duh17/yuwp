#if YUWP_INTERNAL_DIAGNOSTICS
import Foundation
import MLX
#endif
import Testing
@testable import NativeASR

@Suite("Stop-time provisional decode skip", .serialized)
struct StopTailFinalizationTests {
    private func eligibleInputs(
        pending: Int = ASRAudio.sampleRate / 2,
        batchRetranscribe: Bool = true,
        wouldDecode: Bool = true,
        strategy: StopBatchStrategy = .activeSegmentOnly,
        sessionSamples: Int = ASRAudio.sampleRate * 4,
        activeSamples: Int = ASRAudio.sampleRate * 2,
        hasEnoughSpeech: Bool = true
    ) -> StopTailPlanInputs {
        StopTailPlanInputs(
            pendingSampleCount: pending,
            batchRetranscribe: batchRetranscribe,
            wouldProvisionallyDecode: wouldDecode,
            strategy: strategy,
            sessionAudioSampleCountAfterIngest: sessionSamples,
            activeAudioSampleCountAfterIngest: activeSamples,
            maxSessionContextSamples: StreamingSession.maxSessionContextSamples,
            maxLiveBatchSegmentSamples: StreamingSession.maxLiveBatchSegmentSamples,
            hasEnoughSpeechAfterIngest: hasEnoughSpeech
        )
    }

    private func speechPending(samples: Int = ASRAudio.sampleRate / 10) -> [Float] {
        Array(repeating: 0.04, count: samples)
    }

    private func liveSpeechHint(durationSec: Double = 0.05) -> SpeechActivityHint {
        SpeechActivityHint(hasSpeech: true, speechDurationSec: durationSec)
    }

    private func energyRecoveryEvidence(vadSec: Double = 0.05) -> SpeechEvidence {
        SpeechEvidence(
            hasSpeechActivityHints: true,
            vadSpeechDurationSec: vadSec,
            energySpeechDurationSec: SpeechEvidence.minimumEnergySpeechDurationSec,
            peakAmplitude: SpeechEvidence.minimumPeakAmplitude
        )
    }

    private func enoughVADEvidence() -> SpeechEvidence {
        SpeechEvidence(
            hasSpeechActivityHints: true,
            vadSpeechDurationSec: SpeechEvidence.minimumVADSpeechDurationSec,
            energySpeechDurationSec: SpeechEvidence.minimumEnergySpeechDurationSec,
            peakAmplitude: SpeechEvidence.minimumPeakAmplitude
        )
    }

    private func sessionPlan(
        pending: [Float],
        hint: SpeechActivityHint?,
        evidence: SpeechEvidence,
        preIngestSamples: Int,
        config: StreamConfig = StreamConfig(),
        hasSpeech: Bool = true
    ) -> StopTailPlan {
        StreamingSession.stopTailPlan(
            pendingAudio: pending,
            speechHint: hint,
            activeSpeechEvidence: evidence,
            consecutiveSilence: 0,
            batchDoneForPause: false,
            hasRawTokens: true,
            hasSpeech: hasSpeech,
            sessionAudioSampleCount: preIngestSamples,
            activeAudioSampleCount: preIngestSamples,
            config: config
        )
    }

    @Test func emptyPendingOnlyFinalizes() {
        #expect(StopTailExecutor.plan(eligibleInputs(pending: 0)) == .finalizeOnly)
    }

    @Test func disabledFinalAccuracyKeepsOriginalProcessThenFinalize() {
        #expect(
            StopTailExecutor.plan(eligibleInputs(batchRetranscribe: false, strategy: .none))
                == .processThenFinalize
        )
    }

    @Test func silenceOrPausePathsDoNotSkip() {
        #expect(
            StopTailExecutor.plan(eligibleInputs(wouldDecode: false))
                == .processThenFinalize
        )
    }

    @Test func ineligibleBatchStrategyDoesNotSkip() {
        #expect(
            StopTailExecutor.plan(eligibleInputs(strategy: .none))
                == .processThenFinalize
        )
    }

    @Test func longAudioGuardsKeepOriginalProvisionalDecode() {
        let plan = StopTailExecutor.plan(
            eligibleInputs(
                sessionSamples: StreamingSession.maxSessionContextSamples + 1,
                activeSamples: StreamingSession.maxLiveBatchSegmentSamples + 1
            )
        )
        #expect(plan == .processThenFinalize)
    }

    @Test func sessionContextOrActiveBatchCanSkip() {
        #expect(StopTailExecutor.plan(eligibleInputs()) == .skipProvisionalDecode)
        #expect(
            StopTailExecutor.plan(eligibleInputs(strategy: .fullSession))
                == .skipProvisionalDecode
        )
        #expect(
            StopTailExecutor.plan(
                eligibleInputs(
                    sessionSamples: StreamingSession.maxSessionContextSamples + 1,
                    activeSamples: StreamingSession.maxLiveBatchSegmentSamples
                )
            ) == .skipProvisionalDecode
        )
    }

    @Test func skipIngestsOnceThenBatchesWithoutDecoding() {
        var events: [String] = []
        let (text, execution) = StopTailExecutor.execute(
            plan: .skipProvisionalDecode,
            processThenFinalize: {
                events.append("process")
                return "processed"
            },
            finalizeOnly: {
                events.append("finalize-only")
                return "final-only"
            },
            ingestWithoutDecode: { events.append("ingest") },
            finalizeBatch: {
                events.append("batch")
                return "batch tail"
            },
            fallbackDecodeThenText: {
                events.append("decode")
                return "decoded tail"
            }
        )

        #expect(events == ["ingest", "batch"])
        #expect(text == "batch tail")
        #expect(execution == .skippedProvisionalDecode)
    }

    @Test(arguments: [Optional<String>.none, Optional(""), Optional("   ")])
    func unusableBatchDecodesTheIngestedTailNotAPriorPartial(unusable: String?) {
        let priorPartial = "stale words without tail"
        var ingested = 0
        var events: [String] = []
        let (text, execution) = StopTailExecutor.execute(
            plan: .skipProvisionalDecode,
            processThenFinalize: { priorPartial },
            finalizeOnly: { priorPartial },
            ingestWithoutDecode: {
                ingested += 1
                events.append("ingest")
            },
            finalizeBatch: {
                events.append("batch")
                return unusable
            },
            fallbackDecodeThenText: {
                #expect(ingested == 1)
                events.append("decode")
                return "includes tail speech"
            }
        )

        #expect(events == ["ingest", "batch", "decode"])
        #expect(ingested == 1)
        #expect(text == "includes tail speech")
        #expect(text != priorPartial)
        #expect(execution == .skippedThenFallbackDecoded)
    }

    @Test func ineligiblePlanNeverUsesTheSkipSeam() {
        var events: [String] = []
        let (text, execution) = StopTailExecutor.execute(
            plan: .processThenFinalize,
            processThenFinalize: {
                events.append("process")
                return "original"
            },
            finalizeOnly: {
                events.append("finalize-only")
                return "final-only"
            },
            ingestWithoutDecode: { events.append("ingest") },
            finalizeBatch: {
                events.append("batch")
                return "batch"
            },
            fallbackDecodeThenText: {
                events.append("decode")
                return "decoded"
            }
        )

        #expect(events == ["process"])
        #expect(text == "original")
        #expect(execution == .processedThenFinalized)
    }

    @Test func speechTailWouldProvisionallyDecode() {
        #expect(
            StreamingSession.chunkWouldProvisionallyDecode(
                chunkHasSpeech: true,
                rms: 0.05,
                consecutiveSilenceAfterChunk: 0,
                batchRetranscribe: true,
                batchDoneForPause: false,
                hasRawTokens: true,
                hasEnoughSpeechAfterIngest: true,
                alreadyHasSpeech: true
            )
        )
        #expect(
            StreamingSession.chunkWouldProvisionallyDecode(
                chunkHasSpeech: true,
                rms: 0.05,
                consecutiveSilenceAfterChunk: 0,
                batchRetranscribe: true,
                batchDoneForPause: false,
                hasRawTokens: false,
                hasEnoughSpeechAfterIngest: true,
                alreadyHasSpeech: false
            )
        )
    }

    @Test func pauseAttemptAndQuietSilenceWouldNotDecode() {
        #expect(
            !StreamingSession.chunkWouldProvisionallyDecode(
                chunkHasSpeech: false,
                rms: 0.001,
                consecutiveSilenceAfterChunk: StreamingSession.pauseChunks,
                batchRetranscribe: true,
                batchDoneForPause: false,
                hasRawTokens: true,
                hasEnoughSpeechAfterIngest: true,
                alreadyHasSpeech: true
            )
        )
        #expect(
            !StreamingSession.chunkWouldProvisionallyDecode(
                chunkHasSpeech: false,
                rms: StreamingSession.silenceRMS / 2,
                consecutiveSilenceAfterChunk: 0,
                batchRetranscribe: true,
                batchDoneForPause: true,
                hasRawTokens: true,
                hasEnoughSpeechAfterIngest: true,
                alreadyHasSpeech: true
            )
        )
        #expect(
            !StreamingSession.chunkWouldProvisionallyDecode(
                chunkHasSpeech: false,
                rms: 0.05,
                consecutiveSilenceAfterChunk: 0,
                batchRetranscribe: false,
                batchDoneForPause: false,
                hasRawTokens: false,
                hasEnoughSpeechAfterIngest: false,
                alreadyHasSpeech: false
            )
        )
    }

    @Test func energyRecoveryEligibilityRefusesSkipOnlyPastTheLiveBatchCap() {
        let cap = StreamingSession.maxLiveBatchSegmentSamples
        #expect(
            StopTailExecutor.plan(
                eligibleInputs(
                    sessionSamples: cap + 1,
                    activeSamples: cap + 1,
                    hasEnoughSpeech: false
                )
            ) == .processThenFinalize
        )
        #expect(
            StopTailExecutor.plan(
                eligibleInputs(
                    sessionSamples: cap,
                    activeSamples: cap,
                    hasEnoughSpeech: false
                )
            ) == .skipProvisionalDecode
        )
        #expect(
            StopTailExecutor.plan(
                eligibleInputs(
                    sessionSamples: cap + 1,
                    activeSamples: cap + 1,
                    hasEnoughSpeech: true
                )
            ) == .skipProvisionalDecode
        )
        #expect(
            StopTailExecutor.plan(eligibleInputs(hasEnoughSpeech: false))
                == .skipProvisionalDecode
        )
        #expect(
            StopTailExecutor.plan(
                eligibleInputs(
                    strategy: .fullSession,
                    sessionSamples: cap + 1,
                    activeSamples: cap + 1,
                    hasEnoughSpeech: false
                )
            ) == .skipProvisionalDecode
        )
    }

    @Test func sessionPlanMatchesFirstSegmentEnergyRecoveryTrigger() {
        let pending = speechPending()
        let hint = liveSpeechHint()
        let energy = energyRecoveryEvidence()
        let cap = StreamingSession.maxLiveBatchSegmentSamples
        #expect(!energy.hasEnoughSpeech)
        #expect(energy.hasRecoverableEnergySpeech)

        #expect(
            sessionPlan(
                pending: pending, hint: hint, evidence: energy, preIngestSamples: cap
            ) == .processThenFinalize
        )
        #expect(
            sessionPlan(
                pending: pending,
                hint: hint,
                evidence: energy,
                preIngestSamples: cap - pending.count
            ) == .skipProvisionalDecode
        )
        #expect(
            sessionPlan(
                pending: pending,
                hint: hint,
                evidence: enoughVADEvidence(),
                preIngestSamples: cap
            ) == .skipProvisionalDecode
        )
        #expect(
            sessionPlan(
                pending: pending,
                hint: hint,
                evidence: energy,
                preIngestSamples: ASRAudio.sampleRate * 2
            ) == .skipProvisionalDecode
        )
        #expect(
            sessionPlan(
                pending: pending,
                hint: hint,
                evidence: energy,
                preIngestSamples: cap,
                config: StreamConfig(finalizationPass: .fullSessionRetranscribe)
            ) == .skipProvisionalDecode
        )
    }

    @Test func skipFallbackDropsEnergyOnlyLongSegmentWordsThatOriginalFinalizeKeeps() {
        let decoded = "recovered tail words"
        let evidence = energyRecoveryEvidence()
        let config = StreamConfig()
        let cap = StreamingSession.maxLiveBatchSegmentSamples
        #expect(!evidence.hasEnoughSpeech)
        #expect(
            !StreamingSession.shouldAppendActiveTextFallback(
                config: config, activeText: decoded, activeSpeechEvidence: evidence
            )
        )

        let (skipText, skipExecution) = StopTailExecutor.execute(
            plan: .skipProvisionalDecode,
            processThenFinalize: { decoded },
            finalizeOnly: { "" },
            ingestWithoutDecode: {},
            finalizeBatch: { nil },
            fallbackDecodeThenText: {
                StreamingSession.shouldAppendActiveTextFallback(
                    config: config, activeText: decoded, activeSpeechEvidence: evidence
                ) ? decoded : ""
            }
        )
        #expect(skipExecution == .skippedThenFallbackDecoded)
        #expect(skipText.isEmpty)

        let (originalText, originalExecution) = StopTailExecutor.execute(
            plan: .processThenFinalize,
            processThenFinalize: {
                let streamed = decoded.trimmingCharacters(in: .whitespacesAndNewlines)
                if cap + 1 > cap, !streamed.isEmpty {
                    return streamed
                }
                return StreamingSession.shouldAppendActiveTextFallback(
                    config: config, activeText: decoded, activeSpeechEvidence: evidence
                ) ? decoded : ""
            },
            finalizeOnly: { "" },
            ingestWithoutDecode: {},
            finalizeBatch: { nil },
            fallbackDecodeThenText: { "" }
        )
        #expect(originalExecution == .processedThenFinalized)
        #expect(originalText == decoded)
    }

#if YUWP_INTERNAL_DIAGNOSTICS
    @Test(arguments: [
        StreamingSession.StopTailTestSessionContext.unavailable,
        StreamingSession.StopTailTestSessionContext.empty,
    ])
    func finishOnStopLongEnergyRecoveryKeepsStreamingWordsWhenSessionContextUnusable(
        sessionContext: StreamingSession.StopTailTestSessionContext
    ) throws {
        let decoded = "recovered tail words"
        let pending = speechPending()
        let hint = liveSpeechHint()
        let session = try makeStopTailSession()
        session.stopTailTestForcedActiveText = decoded
        session.stopTailTestSessionContext = sessionContext
        session.seedStopTailTestState(
            sessionAndActiveSampleCount: StreamingSession.maxLiveBatchSegmentSamples,
            evidence: energyRecoveryEvidence()
        )

        let result = session.finishOnStop(pendingAudio: pending, speechHint: hint)

        #expect(!result.skippedProvisionalDecode)
        #expect(!result.usedFallbackDecode)
        #expect(result.text == decoded)
        #expect(!result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        #expect(result.sessionSampleCount == StreamingSession.maxLiveBatchSegmentSamples + pending.count)
    }

    private func makeStopTailSession(config: StreamConfig = StreamConfig()) throws -> StreamingSession {
        let tokenizer = try makeTokenizerFixture()
        let transcriber = Qwen3ASRTranscriber.testingTranscriber(
            model: Qwen3ASRModel(config: Qwen3ASRConfig(
                audioConfig: AudioEncoderConfig(
                    encoderLayers: 0, encoderAttentionHeads: 2, encoderFfnDim: 64,
                    dModel: 32, outputDim: 32, downsampleHiddenSize: 8
                ),
                textConfig: TextDecoderConfig(
                    vocabSize: 64, hiddenSize: 32, intermediateSize: 64,
                    numHiddenLayers: 1, numAttentionHeads: 4, numKeyValueHeads: 2,
                    headDim: 8, tieWordEmbeddings: true
                )
            )),
            tokenizer: tokenizer
        )
        return StreamingSession(transcriber: transcriber, config: config)
    }

    private func makeTokenizerFixture() throws -> Qwen3ASRTokenizer {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let vocab: [String: Int] = [
            "h": 1,
            "i": 2,
            "<asr_text>": Qwen3ASRTokenizer.asrText,
        ]
        let vocabData = try JSONSerialization.data(withJSONObject: vocab, options: [.sortedKeys])
        try vocabData.write(to: dir.appendingPathComponent("vocab.json"))
        try Data("#version: 0.2\n".utf8).write(to: dir.appendingPathComponent("merges.txt"))
        return try Qwen3ASRTokenizer.load(from: dir)
    }
#endif
}
