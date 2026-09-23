// NativeASR — Streaming Session

import Foundation
import MLX

/// Streaming configuration.
public enum FinalizationPass: String, Sendable {
    case activeSegmentOnly
    case fullSessionRetranscribe
}

public enum StreamDecodeMode: String, Sendable {
    case rollbackBatch
    case stablePrefix
}

public struct StreamConfig: Sendable {
    public var chunkSec: Double, rollback: Int, unfixedChunks: Int
    public var maxNewTokens: Int, maxEncWindows: Int, maxPrefixTokens: Int
    public var batchRetranscribe: Bool
    public var finalizationPass: FinalizationPass
    public var decodeMode: StreamDecodeMode
    public var repetitionPenalty: Float

    public init(
        chunkSec: Double = 1.75, rollback: Int = 5, unfixedChunks: Int = 2,
        maxNewTokens: Int = 32, maxEncWindows: Int = 4, maxPrefixTokens: Int = 20,
        batchRetranscribe: Bool = true,
        finalizationPass: FinalizationPass = .activeSegmentOnly,
        decodeMode: StreamDecodeMode = .rollbackBatch,
        repetitionPenalty: Float = 1.3
    ) {
        self.chunkSec = chunkSec; self.rollback = rollback; self.unfixedChunks = unfixedChunks
        self.maxNewTokens = maxNewTokens; self.maxEncWindows = maxEncWindows
        self.maxPrefixTokens = maxPrefixTokens; self.batchRetranscribe = batchRetranscribe
        self.finalizationPass = finalizationPass
        self.decodeMode = decodeMode
        self.repetitionPenalty = repetitionPenalty
    }

    /// R2T2 Longest-Stable-Prefix loop. `maxPrefixTokens == 0` means uncapped.
    public static func stablePrefix(
        chunkSec: Double = 0.16,
        unfixedTokenCount: Int = 1,
        maxNewTokens: Int = 4,
        finalizationPass: FinalizationPass = .activeSegmentOnly
    ) -> StreamConfig {
        StreamConfig(
            chunkSec: chunkSec,
            rollback: unfixedTokenCount,
            unfixedChunks: 0,
            maxNewTokens: maxNewTokens,
            maxPrefixTokens: 0,
            batchRetranscribe: false,
            finalizationPass: finalizationPass,
            decodeMode: .stablePrefix,
            repetitionPenalty: 1.0
        )
    }

    public var isStablePrefix: Bool { decodeMode == .stablePrefix }

    public static func isR2T2Model(at directory: URL) -> Bool {
        let path = directory.path
        let name = directory.lastPathComponent
        return path.range(of: "r2t2", options: .caseInsensitive) != nil
            || name.range(of: "r2t2", options: .caseInsensitive) != nil
    }

    public static func forModel(
        at directory: URL,
        batchRetranscribe: Bool = true
    ) -> StreamConfig {
        if isR2T2Model(at: directory) {
            return .stablePrefix()
        }
        return StreamConfig(batchRetranscribe: batchRetranscribe)
    }
}

public struct SpeechActivityHint: Sendable {
    public let hasSpeech: Bool
    public let speechDurationSec: Double

    public init(hasSpeech: Bool, speechDurationSec: Double) {
        self.hasSpeech = hasSpeech
        self.speechDurationSec = speechDurationSec
    }
}

struct AudioEnergyStats: Sendable, Equatable {
    let rms: Float
    let peakAmplitude: Float
    let durationSec: Double
    let speechLikeDurationSec: Double

    init(
        rms: Float,
        peakAmplitude: Float,
        durationSec: Double,
        speechLikeDurationSec: Double? = nil
    ) {
        self.rms = rms
        self.peakAmplitude = peakAmplitude
        self.durationSec = durationSec
        self.speechLikeDurationSec = min(
            durationSec,
            max(0, speechLikeDurationSec ?? durationSec)
        )
    }
}

struct SpeechEvidence: Sendable, Equatable {
    static let minimumVADSpeechDurationSec = 0.18
    static let minimumEnergySpeechDurationSec = 0.30
    static let energySpeechRMS: Float = 0.008
    static let minimumPeakAmplitude: Float = 0.015
    static let energyWindowSamples = ASRAudio.sampleRate / 50  // 20 ms

    var hasSpeechActivityHints: Bool
    var vadSpeechDurationSec: Double
    var energySpeechDurationSec: Double
    var peakAmplitude: Float

    init(
        hasSpeechActivityHints: Bool = false,
        vadSpeechDurationSec: Double = 0,
        energySpeechDurationSec: Double = 0,
        peakAmplitude: Float = 0
    ) {
        self.hasSpeechActivityHints = hasSpeechActivityHints
        self.vadSpeechDurationSec = vadSpeechDurationSec
        self.energySpeechDurationSec = energySpeechDurationSec
        self.peakAmplitude = peakAmplitude
    }

    var hasRecoverableEnergySpeech: Bool {
        energySpeechDurationSec >= Self.minimumEnergySpeechDurationSec
            && peakAmplitude >= Self.minimumPeakAmplitude
    }

    var hasEnoughSpeech: Bool {
        if vadSpeechDurationSec >= Self.minimumVADSpeechDurationSec {
            return true
        }
        if hasSpeechActivityHints {
            return false
        }
        return hasRecoverableEnergySpeech
    }

    mutating func ingest(stats: AudioEnergyStats, speechHint: SpeechActivityHint?) {
        if let speechHint {
            hasSpeechActivityHints = true
            if speechHint.hasSpeech {
                vadSpeechDurationSec += max(0, speechHint.speechDurationSec)
            }
        }

        peakAmplitude = max(peakAmplitude, stats.peakAmplitude)
        energySpeechDurationSec += stats.speechLikeDurationSec
    }
}

struct LiveBatchRefreshPolicy: Sendable, Equatable {
    let minimumNoGrowthChunks: Int
    let minimumAudioSamples: Int
    let maximumAudioSamples: Int
    let retryIntervalChunks: Int

    static let `default` = LiveBatchRefreshPolicy(
        minimumNoGrowthChunks: 1,
        minimumAudioSamples: ASRAudio.sampleRate * 2,
        maximumAudioSamples: ASRAudio.sampleRate * 12,
        retryIntervalChunks: 2
    )

    func shouldAttempt(
        batchRetranscribeEnabled: Bool,
        textChanged: Bool,
        hasSpeech: Bool,
        speechActive: Bool,
        consecutiveNoGrowthChunks: Int,
        activeAudioSampleCount: Int,
        chunksSinceLastAttempt: Int
    ) -> Bool {
        batchRetranscribeEnabled
            && !textChanged
            && hasSpeech
            && speechActive
            && consecutiveNoGrowthChunks >= minimumNoGrowthChunks
            && activeAudioSampleCount >= minimumAudioSamples
            && activeAudioSampleCount <= maximumAudioSamples
            && chunksSinceLastAttempt >= retryIntervalChunks
    }
}

public struct ChunkResult: Sendable {
    public let text: String
    public let isPartial: Bool
    public var batchCorrected: Bool = false
    public var encodeMs: Double = 0, prefillMs: Double = 0, decodeMs: Double = 0
    public var totalMs: Double = 0, reusePct: Double = 0
}

public enum StopBatchStrategy: Sendable, Equatable {
    case none
    case activeSegmentOnly
    case fullSession
}

/// Live prefill and pause/stall/final/full-session batch correction share this context.
struct SessionASRContext: Equatable, Sendable {
    let language: String?
    let vocabularyHints: [String]
}

public final class StreamingSession: @unchecked Sendable {
    private let transcriber: Qwen3ASRTranscriber
    private let batchTranscriber: Qwen3ASRTranscriber?
    private let config: StreamConfig
    private let asrContext: SessionASRContext
    private var audioBuffer: [Float] = []
    private var sessionAudioBuffer: [Float] = []
    private var encWindowCache: [MLXArray] = []
    private var nextWindowStart: Int = 0
    /// Number of complete encoder windows evicted from the front of the cache.
    private var encoderCacheOrigin: Int = 0
    private let encWindowSamples: Int
    private var kvCache: [KVCache]
    private var prevPrefillEmbeds: MLXArray?
    /// Number of encoder tokens from cached windows in the previous chunk.
    /// Used for structural reuse-length estimation (avoids GPU sync).
    private var prevCachedEncTokenCount: Int = 0
    /// Cache origin from the previous chunk. A change means encoder token
    /// positions shifted and only the fixed prompt header remains reusable.
    private var prevEncoderCacheOrigin: Int = 0
    private var rawTokens: [Int] = []
    private var chunkIdx: Int = 0
    private var lastText: String = ""
    private var consecutiveSilence: Int = 0
    private var batchDoneForPause: Bool = false
    private var hasSpeech: Bool = false
    private var activeSpeechEvidence = SpeechEvidence()
    private var consecutiveSpeechNoGrowth: Int = 0
    private var lastStallBatchAttemptChunk: Int = -1_000_000
    /// Concatenated text from all previously committed segments. Frozen — never
    /// rewritten by streaming or batch passes after a commit fires.
    private var committedText: String = ""
    /// Token ids for `committedText` in stable-prefix mode. Re-encoding the
    /// string each chunk drifts from the sampled ids and stalls English.
    private var committedPrefixTokens: [Int] = []
#if YUWP_INTERNAL_DIAGNOSTICS
    enum StopTailTestSessionContext: Equatable, Sendable {
        case real
        case unavailable
        case empty
        case text(String)
    }

    /// Injected session-context result for stop-tail fixtures. Default keeps the real path.
    var stopTailTestSessionContext: StopTailTestSessionContext = .real
    /// When set, stop-tail fixtures skip MLX decode/extract and use this streaming text.
    var stopTailTestForcedActiveText: String?
#endif
    static let silenceRMS: Float = 0.003
    static let speechStartRMS: Float = 0.010
    private static let pauseRMS: Float = 0.020
    static let pauseChunks = 1
    private static let stallRefreshRMS: Float = 0.008
    private static let liveBatchRefreshPolicy = LiveBatchRefreshPolicy.default
    // Session-context retranscribe walks the whole session buffer. Cap that path
    // for long sessions to avoid multi-second stalls during live dictation.
    static let maxSessionContextSamples = ASRAudio.sampleRate * 45
    // Cap expensive mid-session batch correction windows. For longer active
    // segments we commit the streaming text directly and keep moving.
    static let maxLiveBatchSegmentSamples = ASRAudio.sampleRate * 12

    public init(
        transcriber: Qwen3ASRTranscriber,
        batchTranscriber: Qwen3ASRTranscriber? = nil,
        config: StreamConfig = StreamConfig(),
        language: String? = nil,
        vocabularyHints: [String] = []
    ) {
        self.transcriber = transcriber
        self.batchTranscriber = batchTranscriber
        self.config = config
        self.asrContext = Self.sessionASRContext(language: language, vocabularyHints: vocabularyHints)
        self.encWindowSamples = transcriber.model.config.audioConfig.nWindowInfer * ASRAudio.hopLength
        self.kvCache = transcriber.model.makeCache()
    }

    /// Same language and vocabulary header for live prompt construction and
    /// pause/stall/final/full-session batch correction.
    static func sessionASRContext(
        language: String?,
        vocabularyHints: [String]
    ) -> SessionASRContext {
        SessionASRContext(
            language: language?.trimmingCharacters(in: .whitespacesAndNewlines),
            vocabularyHints: vocabularyHints
        )
    }

    private enum ChunkAdmission {
        case finished(ChunkResult)
        case admitted(rms: Float)
    }

    /// Process one chunk of audio. Returns partial transcription result.
    public func processChunk(_ audioChunk: [Float], speechHint: SpeechActivityHint? = nil) -> ChunkResult {
        let t0 = Date()
        switch admitChunk(audioChunk, speechHint: speechHint, startedAt: t0) {
        case .finished(let result):
            return result
        case .admitted(let rms):
            return decodeAdmittedChunk(startedAt: t0, speechHint: speechHint, rms: rms)
        }
    }

    static func chunkWouldProvisionallyDecode(
        chunkHasSpeech: Bool,
        rms: Float,
        consecutiveSilenceAfterChunk: Int,
        batchRetranscribe: Bool,
        batchDoneForPause: Bool,
        hasRawTokens: Bool,
        hasEnoughSpeechAfterIngest: Bool,
        alreadyHasSpeech: Bool
    ) -> Bool {
        if batchRetranscribe
            && consecutiveSilenceAfterChunk >= pauseChunks
            && !batchDoneForPause
            && hasRawTokens
            && hasEnoughSpeechAfterIngest
        {
            return false
        }
        if !chunkHasSpeech && rms < silenceRMS {
            return false
        }
        return alreadyHasSpeech || chunkHasSpeech
    }

    /// Append session audio and speech evidence. Decode is a separate step so
    /// stop can skip an unexposed tail pass when final accuracy will replace it.
    private func admitChunk(
        _ audioChunk: [Float],
        speechHint: SpeechActivityHint?,
        startedAt t0: Date
    ) -> ChunkAdmission {
        sessionAudioBuffer.append(contentsOf: audioChunk)
        let stats = Self.computeAudioStats(audioChunk)
        activeSpeechEvidence.ingest(stats: stats, speechHint: speechHint)
        let rms = stats.rms
        let chunkHasSpeech = speechHint?.hasSpeech ?? (rms >= Self.speechStartRMS)

        if chunkHasSpeech { consecutiveSilence = 0; batchDoneForPause = false }
        else { consecutiveSilence += 1 }

        // Segment commit on pause: batch retranscribe the active segment,
        // append it to committedText, then reset streaming state so the next
        // chunks build a fresh active segment. Committed text is never rewritten.
        if config.batchRetranscribe
            && consecutiveSilence >= Self.pauseChunks && !batchDoneForPause && !rawTokens.isEmpty
            && activeSpeechEvidence.hasEnoughSpeech
        {
            batchDoneForPause = true
            if let segmentText = batchCommitSegmentText() {
                committedText = Self.appendSegment(committedText, segmentText)
                resetActiveSegment()
                lastText = committedText
                chunkIdx += 1
                return .finished(ChunkResult(
                    text: committedText, isPartial: true, batchCorrected: true,
                    totalMs: Date().timeIntervalSince(t0) * 1000
                ))
            }
        }

        if !chunkHasSpeech && rms < Self.silenceRMS {
            chunkIdx += 1
            if config.isStablePrefix {
                // Do not keep running inference on pause — that hitch is what
                // feels like the app halting after a word. Flush the held token
                // so the last word appears without a GPU pass.
                if hasSpeech { flushStablePrefixHeldTokens() }
                lastText = committedText
                return .finished(ChunkResult(
                    text: committedText, isPartial: true, totalMs: Date().timeIntervalSince(t0) * 1000
                ))
            }
            let activeText = rawTokens.isEmpty ? "" : extractText(rawTokens)
            let combined = Self.appendSegment(committedText, activeText)
            lastText = combined
            return .finished(ChunkResult(
                text: combined, isPartial: true, totalMs: Date().timeIntervalSince(t0) * 1000
            ))
        }

        if chunkHasSpeech { hasSpeech = true }
        if !hasSpeech {
            audioBuffer.append(contentsOf: audioChunk)
            chunkIdx += 1
            return .finished(ChunkResult(
                text: committedText, isPartial: true, totalMs: Date().timeIntervalSince(t0) * 1000
            ))
        }

        audioBuffer.append(contentsOf: audioChunk)
        return .admitted(rms: rms)
    }

    private func decodeAdmittedChunk(
        startedAt t0: Date,
        speechHint: SpeechActivityHint?,
        rms: Float
    ) -> ChunkResult {
#if YUWP_INTERNAL_DIAGNOSTICS
        if let forced = stopTailTestForcedActiveText {
            if rawTokens.isEmpty { rawTokens = [1] }
            let combined = Self.appendSegment(committedText, forced)
            lastText = combined
            chunkIdx += 1
            return ChunkResult(
                text: combined, isPartial: true,
                totalMs: Date().timeIntervalSince(t0) * 1000
            )
        }
#endif
        // 1. Encode
        let encT0 = Date()
        let encOutput = encodeIncremental()
        let numEncTokens = encOutput.shape[0]
        let encodeMs = Date().timeIntervalSince(encT0) * 1000

        if numEncTokens == 0 {
            chunkIdx += 1
            return ChunkResult(text: committedText, isPartial: true, totalMs: Date().timeIntervalSince(t0) * 1000)
        }

        // 2. Prefix tokens (rollback, or committed text in stable-prefix mode)
        var prefixTokens: [Int] = []
        if config.isStablePrefix {
            prefixTokens = committedPrefixTokens
        } else if chunkIdx >= config.unfixedChunks && !rawTokens.isEmpty {
            let nPrefix = max(0, rawTokens.count - config.rollback)
            prefixTokens = Array(rawTokens.prefix(nPrefix))
            prefixTokens = transcriber.tokenizer.stripAutoLanguagePrefix(prefixTokens)
            if config.maxPrefixTokens > 0, prefixTokens.count > config.maxPrefixTokens {
                prefixTokens = Array(prefixTokens.suffix(config.maxPrefixTokens))
            }
        }

        // 3. Input embeddings
        let (inputEmbeds, headerLength) = buildInputEmbeds(
            encOutput: encOutput, numEncTokens: numEncTokens, prefixTokens: prefixTokens
        )
        // No explicit eval here — the model forward pass triggers lazy evaluation.
        // Previously needed for computeReuseLength's GPU comparison; structural
        // reuse estimation avoids that sync.

        // 4. Delta prefill
        let prefillT0 = Date()
        let totalLen = inputEmbeds.shape[1]
        let prefillLen = totalLen - 1

        // Structural reuse estimation: prompt header + cached encoder windows
        // are bit-identical across chunks. Header length follows the actual
        // system+user prefix, including vocabulary tokens. Avoids the GPU sync
        // that computeReuseLength required (element-wise diff + argMax + eval).
        let reuseLen = Self.estimateReuseLength(
            hasPreviousPrefill: prevPrefillEmbeds != nil,
            encoderCacheOriginChanged: encoderCacheOrigin != prevEncoderCacheOrigin,
            previousCachedEncoderTokenCount: prevCachedEncTokenCount,
            prefillLength: prefillLen,
            headerLength: headerLength
        )

        for c in kvCache { c.offset = reuseLen }

        let deltaLen = prefillLen - reuseLen

        // If embedding shrank, correct the offset
        if deltaLen < 0 {
            for c in kvCache { c.offset = prefillLen }
        }

        // Single forward pass: when deltaLen > 0, process delta + last position
        // together (causal mask). Otherwise just the last position (seqLen=1).
        // This merges what was previously two separate 28-layer decoder calls.
        let dummyIds = MLXArray([Int32(0)]).expandedDimensions(axis: 0)
        let startIdx = deltaLen > 0 ? reuseLen : prefillLen
        let prefillEmbeds = inputEmbeds[0..., startIdx ..< (prefillLen + 1), 0...]
        let (logits, _) = transcriber.model(
            inputIds: dummyIds,
            inputEmbeddings: prefillEmbeds, cache: kvCache
        )
        eval(logits)

        let prefillMs = Date().timeIntervalSince(prefillT0) * 1000
        let reusePct = Double(reuseLen) / Double(max(totalLen, 1)) * 100

        prevPrefillEmbeds = inputEmbeds[0..., 0 ..< prefillLen, 0...]
        prevCachedEncTokenCount = encWindowCache.reduce(0) { $0 + $1.shape[0] }
        prevEncoderCacheOrigin = encoderCacheOrigin

        // 5. Decode
        let decodeT0 = Date()
        let newTokens = decodeTokens(logits: logits, maxTokens: config.maxNewTokens)
        let decodeMs = Date().timeIntervalSince(decodeT0) * 1000

        // 6. Update tokens
        if config.isStablePrefix {
            applyStablePrefixUpdate(newTokens: newTokens)
            chunkIdx += 1
            return ChunkResult(
                text: committedText, isPartial: true,
                encodeMs: encodeMs, prefillMs: prefillMs, decodeMs: decodeMs,
                totalMs: Date().timeIntervalSince(t0) * 1000, reusePct: reusePct
            )
        }

        if !newTokens.isEmpty {
            var uncappedPrefix: [Int] = []
            if chunkIdx >= config.unfixedChunks && !rawTokens.isEmpty {
                let n = rawTokens.count - config.rollback
                if n > 0 { uncappedPrefix = Array(rawTokens.prefix(n)) }
            }
            rawTokens = transcriber.tokenizer.stripAutoLanguagePrefix(uncappedPrefix + newTokens)
        }

        var activeText = extractText(rawTokens)
        if let trimmed = trimRepetition() { activeText = trimmed }
        var combined = Self.appendSegment(committedText, activeText)
        let speechActiveForRefresh = speechHint?.hasSpeech ?? (rms >= Self.stallRefreshRMS)
        var textChanged = combined != lastText
        let projectedNoGrowthRun = (!textChanged && hasSpeech && speechActiveForRefresh)
            ? (consecutiveSpeechNoGrowth + 1)
            : 0

        let refreshPolicy = Self.liveBatchRefreshPolicy
        if refreshPolicy.shouldAttempt(
            batchRetranscribeEnabled: config.batchRetranscribe,
            textChanged: textChanged,
            hasSpeech: hasSpeech,
            speechActive: speechActiveForRefresh,
            consecutiveNoGrowthChunks: projectedNoGrowthRun,
            activeAudioSampleCount: audioBuffer.count,
            chunksSinceLastAttempt: chunkIdx - lastStallBatchAttemptChunk
        ) {
            // Record the attempt even if refresh fails so a difficult segment
            // cannot monopolize the serialized inference path.
            lastStallBatchAttemptChunk = chunkIdx

            if let refreshedText = batchRetranscribeActiveSegment(reason: "stall refresh") {
                let refreshedCombined = Self.appendSegment(committedText, refreshedText)
                if refreshedCombined != combined {
                    activeText = refreshedText
                    combined = refreshedCombined
                    textChanged = true
                }
            }
        }

        if textChanged {
            consecutiveSpeechNoGrowth = 0
            lastStallBatchAttemptChunk = -1_000_000
        } else if hasSpeech && speechActiveForRefresh {
            consecutiveSpeechNoGrowth = projectedNoGrowthRun
        } else {
            consecutiveSpeechNoGrowth = 0
            lastStallBatchAttemptChunk = -1_000_000
        }

        lastText = combined
        chunkIdx += 1

        return ChunkResult(
            text: combined, isPartial: true,
            encodeMs: encodeMs, prefillMs: prefillMs, decodeMs: decodeMs,
            totalMs: Date().timeIntervalSince(t0) * 1000, reusePct: reusePct
        )
    }

    /// Reset everything tied to the active (provisional) segment, but keep
    /// `committedText`. Called after a segment is committed at a pause boundary.
    private func resetActiveSegment() {
        audioBuffer = []
        encWindowCache = []
        nextWindowStart = 0
        encoderCacheOrigin = 0
        kvCache = transcriber.model.makeCache()
        prevPrefillEmbeds = nil
        prevCachedEncTokenCount = 0
        prevEncoderCacheOrigin = 0
        rawTokens = []
        chunkIdx = 0
        consecutiveSilence = 0
        batchDoneForPause = false
        hasSpeech = false
        activeSpeechEvidence = SpeechEvidence()
        consecutiveSpeechNoGrowth = 0
        lastStallBatchAttemptChunk = -1_000_000
    }

    public var processedChunkCount: Int { chunkIdx }

    public func finalText() -> String { lastText }

    public func committedSegmentText() -> String { committedText }

    public func activeSegmentText() -> String {
        // Stable-prefix commits are append-only. Publishing the full hypothesis
        // as active_text makes the client join it onto committedText and inject
        // a duplicated, rewriting string.
        if config.isStablePrefix { return "" }
        guard !rawTokens.isEmpty else { return "" }
        return extractText(rawTokens).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    public func finishOnStop(
        pendingAudio: [Float],
        speechHint: SpeechActivityHint?
    ) -> StopFinishResult {
        let plan = stopTailPlan(pendingAudio: pendingAudio, speechHint: speechHint)
        var admission: ChunkAdmission?
        let startedAt = Date()
        let (text, execution) = StopTailExecutor.execute(
            plan: plan,
            processThenFinalize: {
                _ = processChunk(pendingAudio, speechHint: speechHint)
                return finalize()
            },
            finalizeOnly: { finalize() },
            ingestWithoutDecode: {
                admission = admitChunk(pendingAudio, speechHint: speechHint, startedAt: startedAt)
            },
            finalizeBatch: {
                guard case .admitted = admission else { return nil }
                return tryUsableStopBatchCorrection()
            },
            fallbackDecodeThenText: {
                if case .admitted(let rms) = admission {
                    _ = decodeAdmittedChunk(
                        startedAt: startedAt, speechHint: speechHint, rms: rms
                    )
                }
                return appendActiveTextFallback()
            }
        )
        return StopFinishResult(
            text: text,
            skippedProvisionalDecode: execution == .skippedProvisionalDecode
                || execution == .skippedThenFallbackDecoded,
            usedFallbackDecode: execution == .skippedThenFallbackDecoded,
            pendingSampleCount: pendingAudio.count,
            sessionSampleCount: sessionAudioBuffer.count
        )
    }

    private func stopTailPlan(
        pendingAudio: [Float],
        speechHint: SpeechActivityHint?
    ) -> StopTailPlan {
        Self.stopTailPlan(
            pendingAudio: pendingAudio,
            speechHint: speechHint,
            activeSpeechEvidence: activeSpeechEvidence,
            consecutiveSilence: consecutiveSilence,
            batchDoneForPause: batchDoneForPause,
            hasRawTokens: !rawTokens.isEmpty,
            hasSpeech: hasSpeech,
            sessionAudioSampleCount: sessionAudioBuffer.count,
            activeAudioSampleCount: audioBuffer.count,
            config: config
        )
    }

    /// Same eligibility mapping `finishOnStop` uses. Exposed for stop-tail fixtures.
    static func stopTailPlan(
        pendingAudio: [Float],
        speechHint: SpeechActivityHint?,
        activeSpeechEvidence: SpeechEvidence,
        consecutiveSilence: Int,
        batchDoneForPause: Bool,
        hasRawTokens: Bool,
        hasSpeech: Bool,
        sessionAudioSampleCount: Int,
        activeAudioSampleCount: Int,
        config: StreamConfig
    ) -> StopTailPlan {
        guard !pendingAudio.isEmpty else { return .finalizeOnly }
        let stats = Self.computeAudioStats(pendingAudio)
        var evidence = activeSpeechEvidence
        evidence.ingest(stats: stats, speechHint: speechHint)
        let chunkHasSpeech = speechHint?.hasSpeech ?? (stats.rms >= Self.speechStartRMS)
        let consecutive = chunkHasSpeech ? 0 : consecutiveSilence + 1
        let wouldDecode = Self.chunkWouldProvisionallyDecode(
            chunkHasSpeech: chunkHasSpeech,
            rms: stats.rms,
            consecutiveSilenceAfterChunk: consecutive,
            batchRetranscribe: config.batchRetranscribe,
            batchDoneForPause: batchDoneForPause,
            hasRawTokens: hasRawTokens,
            hasEnoughSpeechAfterIngest: evidence.hasEnoughSpeech,
            alreadyHasSpeech: hasSpeech
        )
        let sessionAfter = sessionAudioSampleCount + pendingAudio.count
        let activeAfter = wouldDecode ? activeAudioSampleCount + pendingAudio.count : activeAudioSampleCount
        let strategy = Self.stopBatchStrategy(
            config: config,
            sessionAudioSampleCount: sessionAfter,
            activeAudioSampleCount: activeAfter,
            activeSpeechEvidence: evidence
        )
        return StopTailExecutor.plan(
            StopTailPlanInputs(
                pendingSampleCount: pendingAudio.count,
                batchRetranscribe: config.batchRetranscribe,
                wouldProvisionallyDecode: wouldDecode,
                strategy: strategy,
                sessionAudioSampleCountAfterIngest: sessionAfter,
                activeAudioSampleCountAfterIngest: activeAfter,
                maxSessionContextSamples: Self.maxSessionContextSamples,
                maxLiveBatchSegmentSamples: Self.maxLiveBatchSegmentSamples,
                hasEnoughSpeechAfterIngest: evidence.hasEnoughSpeech
            )
        )
    }

#if YUWP_INTERNAL_DIAGNOSTICS
    func seedStopTailTestState(
        sessionAndActiveSampleCount: Int,
        evidence: SpeechEvidence,
        hasSpeech: Bool = true
    ) {
        let audio = [Float](repeating: 0, count: sessionAndActiveSampleCount)
        sessionAudioBuffer = audio
        audioBuffer = audio
        activeSpeechEvidence = evidence
        self.hasSpeech = hasSpeech
        committedText = ""
        committedPrefixTokens = []
        lastText = ""
        rawTokens = []
        consecutiveSilence = 0
        batchDoneForPause = false
    }
#endif

    private func tryUsableStopBatchCorrection() -> String? {
#if YUWP_INTERNAL_DIAGNOSTICS
        if ProcessInfo.processInfo.environment["YUWP_TEST_FAIL_STOP_BATCH"] == "1" {
            return nil
        }
#endif
        switch Self.stopBatchStrategy(
            config: config,
            sessionAudioSampleCount: sessionAudioBuffer.count,
            activeAudioSampleCount: audioBuffer.count,
            activeSpeechEvidence: activeSpeechEvidence
        ) {
        case .fullSession:
            guard let fullText = batchRetranscribeFullSession(),
                  StopTailExecutor.isUsableBatchText(fullText) else { return nil }
            committedText = fullText
            rawTokens = []
            lastText = committedText
            return committedText
        case .activeSegmentOnly:
            guard let segmentText = batchFinalizeSegmentText(
                allowStreamingLongSegmentFallback: false
            ), StopTailExecutor.isUsableBatchText(segmentText) else { return nil }
            committedText = Self.appendSegment(committedText, segmentText)
            lastText = committedText
            return committedText
        case .none:
            return nil
        }
    }

    private func appendActiveTextFallback() -> String {
        let activeText = rawTokens.isEmpty ? "" : extractText(rawTokens)
        if Self.shouldAppendActiveTextFallback(
            config: config,
            activeText: activeText,
            activeSpeechEvidence: activeSpeechEvidence
        ) {
            committedText = Self.appendSegment(committedText, activeText)
        }
        lastText = committedText
        return committedText
    }

    /// Finalize the session on stop. By default we batch only the trailing
    /// active segment and preserve every previously committed segment verbatim.
    /// Full-session retranscribe is available only as an explicit opt-in for
    /// A/B comparisons.
    public func finalize() -> String {
        if config.isStablePrefix {
            return finalizeStablePrefix()
        }
        switch Self.stopBatchStrategy(
            config: config,
            sessionAudioSampleCount: sessionAudioBuffer.count,
            activeAudioSampleCount: audioBuffer.count,
            activeSpeechEvidence: activeSpeechEvidence
        ) {
        case .fullSession:
            if let fullText = batchRetranscribeFullSession() {
                committedText = fullText
                rawTokens = []
                lastText = committedText
                return committedText
            }

        case .activeSegmentOnly:
            if let segmentText = batchFinalizeSegmentText() {
                committedText = Self.appendSegment(committedText, segmentText)
                lastText = committedText
                return committedText
            }

        case .none:
            break
        }

        return appendActiveTextFallback()
    }

    static func stopBatchStrategy(
        config: StreamConfig,
        sessionAudioSampleCount: Int,
        activeAudioSampleCount: Int,
        activeSpeechEvidence: SpeechEvidence = SpeechEvidence()
    ) -> StopBatchStrategy {
        guard config.batchRetranscribe else { return .none }

        switch config.finalizationPass {
        case .fullSessionRetranscribe:
            return sessionAudioSampleCount >= ASRAudio.sampleRate ? .fullSession : .none

        case .activeSegmentOnly:
            guard sessionAudioSampleCount >= ASRAudio.sampleRate else { return .none }
            guard activeAudioSampleCount > 0 else { return .none }

            if activeSpeechEvidence.hasEnoughSpeech {
                return .activeSegmentOnly
            }

            let isFirstSegment = sessionAudioSampleCount == activeAudioSampleCount

            // Preserve the first-utterance recovery path when Silero misses a
            // soft start. In the live server path every chunk carries a VAD
            // hint, so `hasEnoughSpeech` becomes VAD-gated. If we have at least
            // one second of the very first segment plus strong energy evidence,
            // still allow the final batch pass on stop.
            if isFirstSegment,
               activeAudioSampleCount >= ASRAudio.sampleRate,
               activeSpeechEvidence.hasRecoverableEnergySpeech {
                return .activeSegmentOnly
            }

            // For later segments, keep the stricter speech gate so trailing
            // room noise does not trigger a batch hallucination.
            return .none
        }
    }

    static func shouldAppendActiveTextFallback(
        config: StreamConfig,
        activeText: String,
        activeSpeechEvidence: SpeechEvidence
    ) -> Bool {
        guard !activeText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return false }
        guard config.batchRetranscribe else { return true }
        return activeSpeechEvidence.hasEnoughSpeech
    }

    static func encoderWindowEvictionCount(cachedWindowCount: Int, maximum: Int) -> Int {
        max(0, cachedWindowCount - maximum)
    }

    static func estimateReuseLength(
        hasPreviousPrefill: Bool,
        encoderCacheOriginChanged: Bool,
        previousCachedEncoderTokenCount: Int,
        prefillLength: Int,
        headerLength: Int
    ) -> Int {
        guard hasPreviousPrefill else { return 0 }
        let reusableLength = encoderCacheOriginChanged
            ? headerLength
            : headerLength + previousCachedEncoderTokenCount
        return min(reusableLength, prefillLength)
    }

    /// Concatenate two segment texts with a single space, handling empty inputs
    /// and avoiding double-spaces.
    static func appendSegment(_ committed: String, _ segment: String) -> String {
        let trimmedSegment = segment.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedSegment.isEmpty { return committed }
        if committed.isEmpty { return trimmedSegment }
        return committed + " " + trimmedSegment
    }

    static func deriveActiveText(fromSessionText fullText: String, committedText: String) -> String? {
        let trimmedFull = fullText.trimmingCharacters(in: .whitespacesAndNewlines)
        let trimmedCommitted = committedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedCommitted.isEmpty else { return trimmedFull }
        if trimmedFull == trimmedCommitted { return "" }

        let separator = trimmedCommitted + " "
        guard trimmedFull.hasPrefix(separator) else { return nil }

        let candidate = String(trimmedFull.dropFirst(separator.count))
        if looksLikeDuplicatedCommittedPrefix(candidate, committedText: trimmedCommitted) {
            return nil
        }
        return candidate
    }

    private static func looksLikeDuplicatedCommittedPrefix(_ candidate: String, committedText: String) -> Bool {
        let trimmedCandidate = candidate.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedCandidate.isEmpty else { return false }
        guard committedText.count >= 24 else { return false }
        guard committedText.split(whereSeparator: \.isWhitespace).count >= 5 else { return false }
        return trimmedCandidate == committedText || trimmedCandidate.hasPrefix(committedText + " ")
    }

    private func encodeSegment(_ audio: [Float]) -> MLXArray {
        let melSpec = logMelSpectrogram(audio: MLXArray(audio))
        let inputFeatures = melSpec.T.expandedDimensions(axis: 0)
        let nFrames = (audio.count + ASRAudio.nFft) / ASRAudio.hopLength
        let attnMask = MLXArray(Array(repeating: Int32(1), count: nFrames))
            .expandedDimensions(axis: 0)
        return transcriber.model.getAudioFeatures(
            inputFeatures: inputFeatures, featureAttentionMask: attnMask
        )
    }

    private func encodeIncremental() -> MLXArray {
        let total = audioBuffer.count

        // Encode new complete windows
        while nextWindowStart + encWindowSamples <= total {
            let window = Array(audioBuffer[nextWindowStart ..< nextWindowStart + encWindowSamples])
            let enc = encodeSegment(window)
            eval(enc)
            encWindowCache.append(enc)
            nextWindowStart += encWindowSamples
        }

        // Evict oldest only after the cache exceeds capacity. Reaching capacity
        // does not shift encoder positions and must preserve structural reuse.
        let evictionCount = Self.encoderWindowEvictionCount(
            cachedWindowCount: encWindowCache.count,
            maximum: config.maxEncWindows
        )
        if evictionCount > 0 {
            encWindowCache.removeFirst(evictionCount)
            encoderCacheOrigin += evictionCount
        }

        // Encode tail (partial window)
        var tailEnc: MLXArray?
        if nextWindowStart < total {
            let tail = Array(audioBuffer[nextWindowStart...])
            if !tail.isEmpty {
                let enc = encodeSegment(tail)
                eval(enc)
                tailEnc = enc
            }
        }

        // Concatenate cached + tail
        var parts = encWindowCache
        if let t = tailEnc { parts.append(t) }

        if parts.isEmpty {
            return MLXArray.zeros([0, transcriber.model.config.audioConfig.outputDim])
        }
        return parts.count == 1 ? parts[0] : MLX.concatenated(parts, axis: 0)
    }

    private func buildInputEmbeds(
        encOutput: MLXArray, numEncTokens: Int, prefixTokens: [Int]
    ) -> (embeds: MLXArray, headerLength: Int) {
        let prompt = transcriber.tokenizer.buildPromptTokens(
            numAudioTokens: numEncTokens,
            language: asrContext.language,
            vocabularyHints: asrContext.vocabularyHints
        )
        let inputIds = MLXArray(prompt.tokenIds.map { Int32($0) }).expandedDimensions(axis: 0)

        var embeds = transcriber.model.buildInputsEmbeds(
            inputIds: inputIds, audioFeatures: encOutput,
            numAudioTokens: numEncTokens, audioStartIndex: prompt.audioPadStartIndex
        )

        if !prefixTokens.isEmpty {
            let pfxIds = MLXArray(prefixTokens.map { Int32($0) }).expandedDimensions(axis: 0)
            let pfxEmbed = transcriber.model.model.embedTokens(pfxIds)
            embeds = MLX.concatenated([embeds, pfxEmbed], axis: 1)
        }

        return (embeds, prompt.audioPadStartIndex)
    }

    private func transcribeWithSessionContext(
        _ audio: [Float], draftText: String
    ) throws -> TranscriptionResult {
        let batcher = batchTranscriber ?? transcriber
        return try batcher.transcribe(
            audio: audio,
            language: asrContext.language,
            vocabularyHints: asrContext.vocabularyHints,
            draftText: draftText
        )
    }

    /// Decode with double-buffer asyncEval pattern:
    /// Sample current token while next forward pass runs on GPU.
    private func decodeTokens(logits: MLXArray, maxTokens: Int) -> [Int] {
        let eos = Qwen3ASRTokenizer.eosTokens
        let repPenalty = config.repetitionPenalty
        let stopOnPipe = config.isStablePrefix
        let repWindow = 8
        var tokens: [Int] = []

        // Sample first token, queue next forward
        var y = sampleWithPenalty(logits: logits, recent: [], penalty: repPenalty)
        var curLogits = logits

        for _ in 0 ..< maxTokens {
            // Prepare the next graph while the current sample finishes.
            let tokId = y.asType(.int32).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            let tokEmbed = transcriber.model.model.embedTokens(tokId)
            (curLogits, _) = transcriber.model(
                inputIds: tokId, inputEmbeddings: tokEmbed, cache: kvCache
            )
            let nextY = sampleWithPenalty(
                logits: curLogits, recent: Array(tokens.suffix(repWindow)), penalty: repPenalty
            )
            let token = y.item(Int.self)
            if eos.contains(token) { break }
            tokens.append(token)
            if stopOnPipe, transcriber.tokenizer.decode(tokens).contains("|") { break }
            if tokens.count >= maxTokens
                || (tokens.count >= 4 && Set(tokens.suffix(4)).count == 1) { break }

            // Do not submit a speculative forward that cannot be consumed.
            asyncEval(nextY)
            y = nextY
        }
        return tokens
    }

    private func applyStablePrefixUpdate(newTokens: [Int]) {
        if newTokens.isEmpty { return }

        var kept = committedPrefixTokens
        for token in newTokens {
            let trial = kept + [token]
            let decoded = transcriber.tokenizer.decode(trial)
            if decoded.contains("|") { break }
            if decoded.contains("\u{FFFD}") { break }
            kept = trial
        }

        let floor = committedPrefixTokens.count
        let unfixed = max(0, config.rollback)
        let proposedFixed = max(0, kept.count - unfixed)
        let fixedCount = max(floor, proposedFixed)
        committedPrefixTokens = Array(kept.prefix(fixedCount))
        let decodedCommitted = committedPrefixTokens.isEmpty
            ? ""
            : transcriber.tokenizer.decode(committedPrefixTokens)
        committedText = StablePrefixCommitter.visibleTranscript(
            prefixText: committedText,
            decoded: decodedCommitted
        )
        rawTokens = kept
        lastText = committedText
    }

    private func flushStablePrefixHeldTokens() {
        guard !rawTokens.isEmpty else { return }
        committedText = StablePrefixCommitter.visibleTranscript(
            prefixText: committedText,
            decoded: transcriber.tokenizer.decode(rawTokens)
        )
        committedPrefixTokens = rawTokens
        lastText = committedText
    }

    private func finalizeStablePrefix() -> String {
        // Commit the held-back token from sampled ids. Re-encoding the decoded
        // string drifts off those ids and drops the last word.
        flushStablePrefixHeldTokens()
        rawTokens = []
        // Default stop result is flushed live text. Full-session batch is a
        // stop-only overlay for short sessions with speech evidence; live
        // pause-batch stays off because `batchRetranscribe` is false.
        if Self.stablePrefixStopBatchStrategy(
            sessionAudioSampleCount: sessionAudioBuffer.count,
            activeSpeechEvidence: activeSpeechEvidence
        ) == .fullSession,
           let batchText = batchRetranscribeFullSession(),
           StopTailExecutor.isUsableBatchText(batchText) {
            let chosen = Self.preferStopBatch(streamed: committedText, batch: batchText)
            if chosen != committedText {
                committedText = chosen
                committedPrefixTokens = []
            }
        }
        lastText = committedText
        return committedText
    }

    /// R2T2 stop-only accuracy pass. Live decode stays append-only because
    /// `batchRetranscribe` is false; this gate is independent of Qwen pause-batch.
    static func stablePrefixStopBatchStrategy(
        sessionAudioSampleCount: Int,
        activeSpeechEvidence: SpeechEvidence
    ) -> StopBatchStrategy {
        guard sessionAudioSampleCount > 0,
              sessionAudioSampleCount <= maxLiveBatchSegmentSamples,
              activeSpeechEvidence.hasEnoughSpeech else {
            return .none
        }
        return .fullSession
    }

    /// Stop-batch must never wipe a longer live transcript. A shorter batch is
    /// how the on-screen text looks reset or cut off.
    static func preferStopBatch(streamed: String, batch: String) -> String {
        let streamed = streamed.trimmingCharacters(in: .whitespacesAndNewlines)
        let batch = batch.trimmingCharacters(in: .whitespacesAndNewlines)
        if streamed.isEmpty { return batch }
        if batch.isEmpty { return streamed }
        if batch.count < streamed.count { return streamed }
        return batch
    }

    private func sampleWithPenalty(logits: MLXArray, recent: [Int], penalty: Float) -> MLXArray {
        let lastLogits = logits[0, -1, 0...]
        let recentSet = Set(recent)
        if recentSet.isEmpty || penalty <= 1.0 {
            return MLX.argMax(lastLogits, axis: -1)
        }
        let idxArr = MLXArray(Array(recentSet).map { Int32($0) })
        let vals = lastLogits[idxArr]
        let penalized = MLX.where(vals .> 0, vals / MLXArray(penalty), vals * MLXArray(penalty))
        lastLogits[idxArr] = penalized
        return MLX.argMax(lastLogits, axis: -1)
    }

    private func extractText(_ tokens: [Int]) -> String {
#if YUWP_INTERNAL_DIAGNOSTICS
        if let forced = stopTailTestForcedActiveText {
            return forced
        }
#endif
        let cleaned = transcriber.tokenizer.cleanTokenOutput(tokens)
        return cleaned == "None" ? "" : cleaned
    }

    private func detectRepetition(_ text: String, minPeriod: Int = 30) -> Int? {
        let n = text.count
        if n < minPeriod * 2 { return nil }

        let chars = Array(text)
        for p in minPeriod ... (n / 2) {
            let a = String(chars[(n - 2 * p) ..< (n - p)])
            let b = String(chars[(n - p)...])
            if a == b {
                if let range = text.range(of: b) {
                    let firstEnd = text.distance(from: text.startIndex, to: range.upperBound)
                    if firstEnd < n - p { return firstEnd }
                }
            }
        }
        return nil
    }

    private func trimRepetition() -> String? {
        let text = extractText(rawTokens)
        guard let trimAt = detectRepetition(text) else { return nil }

        let trimmed = String(text.prefix(trimAt)).trimmingCharacters(in: .whitespaces)
        fputs("[StreamingSession] Repetition trimmed \(text.count)→\(trimmed.count) chars\n", stderr)

        rawTokens = transcriber.tokenizer.encode(trimmed)
        return trimmed
    }

    private func batchRetranscribeActiveSegment(reason: String) -> String? {
        guard audioBuffer.count >= ASRAudio.sampleRate else { return nil }

        do {
            let result = try transcribeWithSessionContext(
                audioBuffer, draftText: activeSegmentText()
            )
            let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                rawTokens = transcriber.tokenizer.encode(text)
            }
            // Reset cache — batch changed the token sequence
            kvCache = transcriber.model.makeCache()
            prevPrefillEmbeds = nil
            prevCachedEncTokenCount = 0
            prevEncoderCacheOrigin = encoderCacheOrigin
            fputs("[StreamingSession] Active-segment \(reason): \(audioBuffer.count / ASRAudio.sampleRate)s audio → \(text.count) chars\n", stderr)
            return text
        } catch {
            fputs("[StreamingSession] Active-segment \(reason) error: \(error)\n", stderr)
            return nil
        }
    }

    private func batchCommitSegmentText() -> String? {
        if let activeText = sessionContextActiveText(), !activeText.isEmpty {
            fputs("[StreamingSession] Session-context pause commit: \(sessionAudioBuffer.count / ASRAudio.sampleRate)s audio → \(activeText.count) chars\n", stderr)
            return activeText
        }

        // Keep live UX responsive: avoid long synchronous batch passes in the
        // middle of dictation for very large active segments.
        if audioBuffer.count > Self.maxLiveBatchSegmentSamples {
            let streamed = extractText(rawTokens).trimmingCharacters(in: .whitespacesAndNewlines)
            if !streamed.isEmpty {
                fputs("[StreamingSession] Pause commit using streaming text (segment too long: \(audioBuffer.count / ASRAudio.sampleRate)s)\n", stderr)
                return streamed
            }
        }

        return batchRetranscribeActiveSegment(reason: "pause commit")
    }

    private func batchFinalizeSegmentText(
        allowStreamingLongSegmentFallback: Bool = true
    ) -> String? {
        if let activeText = sessionContextActiveText(), !activeText.isEmpty {
            fputs("[StreamingSession] Session-context final segment: \(sessionAudioBuffer.count / ASRAudio.sampleRate)s audio → \(activeText.count) chars\n", stderr)
            return activeText
        }

        // Hard guard: avoid giant end-of-session batch retranscribes that can
        // spike GPU memory after long uninterrupted dictation.
        if audioBuffer.count > Self.maxLiveBatchSegmentSamples {
            guard allowStreamingLongSegmentFallback else { return nil }
            let streamed = extractText(rawTokens).trimmingCharacters(in: .whitespacesAndNewlines)
            if !streamed.isEmpty {
                fputs("[StreamingSession] Finalize using streaming text (segment too long: \(audioBuffer.count / ASRAudio.sampleRate)s)\n", stderr)
                return streamed
            }
            return nil
        }

        return batchRetranscribeActiveSegment(reason: "finalization")
    }

    private func sessionContextActiveText() -> String? {
#if YUWP_INTERNAL_DIAGNOSTICS
        switch stopTailTestSessionContext {
        case .real:
            break
        case .unavailable:
            return nil
        case .empty:
            return ""
        case .text(let text):
            return text
        }
#endif
        if sessionAudioBuffer.count > Self.maxSessionContextSamples {
            fputs("[StreamingSession] Session-context skipped (session too long: \(sessionAudioBuffer.count / ASRAudio.sampleRate)s)\n", stderr)
            return nil
        }

        do {
            let result = try transcribeWithSessionContext(
                sessionAudioBuffer,
                draftText: Self.appendSegment(committedText, activeSegmentText())
            )
            let fullText = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if let activeText = Self.deriveActiveText(fromSessionText: fullText, committedText: committedText) {
                return activeText
            }
            if committedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return fullText
            }
            fputs("[StreamingSession] Session-context split failed; refusing to append full transcript to avoid duplication\n", stderr)
            return nil
        } catch {
            fputs("[StreamingSession] Session-context batch error: \(error)\n", stderr)
            return nil
        }
    }

    private func batchRetranscribeFullSession() -> String? {
        do {
            let result = try transcribeWithSessionContext(
                sessionAudioBuffer,
                draftText: Self.appendSegment(committedText, activeSegmentText())
            )
            let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            fputs("[StreamingSession] Final full-session batch: \(sessionAudioBuffer.count / ASRAudio.sampleRate)s audio → \(text.count) chars\n", stderr)
            return text
        } catch {
            fputs("[StreamingSession] Final full-session batch error: \(error)\n", stderr)
            return nil
        }
    }

    static func computeAudioStats(_ audio: [Float]) -> AudioEnergyStats {
        guard !audio.isEmpty else {
            return AudioEnergyStats(rms: 0, peakAmplitude: 0, durationSec: 0, speechLikeDurationSec: 0)
        }

        var sum: Float = 0
        var peak: Float = 0
        var speechLikeSamples = 0

        var windowSum: Float = 0
        var windowPeak: Float = 0
        var windowSampleCount = 0

        for sample in audio {
            let amplitude = Swift.abs(sample)
            sum += sample * sample
            peak = max(peak, amplitude)

            windowSum += sample * sample
            windowPeak = max(windowPeak, amplitude)
            windowSampleCount += 1

            if windowSampleCount == SpeechEvidence.energyWindowSamples {
                if isEnergySpeechWindow(sumSquares: windowSum, peakAmplitude: windowPeak, sampleCount: windowSampleCount) {
                    speechLikeSamples += windowSampleCount
                }
                windowSum = 0
                windowPeak = 0
                windowSampleCount = 0
            }
        }

        if windowSampleCount > 0,
           isEnergySpeechWindow(sumSquares: windowSum, peakAmplitude: windowPeak, sampleCount: windowSampleCount) {
            speechLikeSamples += windowSampleCount
        }

        return AudioEnergyStats(
            rms: sqrt(sum / Float(audio.count)),
            peakAmplitude: peak,
            durationSec: Double(audio.count) / Double(ASRAudio.sampleRate),
            speechLikeDurationSec: Double(speechLikeSamples) / Double(ASRAudio.sampleRate)
        )
    }

    private static func isEnergySpeechWindow(
        sumSquares: Float,
        peakAmplitude: Float,
        sampleCount: Int
    ) -> Bool {
        guard sampleCount > 0 else { return false }
        let rms = sqrt(sumSquares / Float(sampleCount))
        return rms >= SpeechEvidence.energySpeechRMS
            && peakAmplitude >= SpeechEvidence.minimumPeakAmplitude
    }
}
