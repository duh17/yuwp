// NativeASR — Streaming Session

import Foundation
import MLX

/// Streaming configuration.
public enum FinalizationPass: String, Sendable {
    case activeSegmentOnly
    case fullSessionRetranscribe
}

public struct StreamConfig: Sendable {
    public var chunkSec: Double, rollback: Int, unfixedChunks: Int
    public var maxNewTokens: Int, maxEncWindows: Int, maxPrefixTokens: Int
    public var batchRetranscribe: Bool
    public var finalizationPass: FinalizationPass

    public init(
        chunkSec: Double = 1.75, rollback: Int = 5, unfixedChunks: Int = 2,
        maxNewTokens: Int = 32, maxEncWindows: Int = 4, maxPrefixTokens: Int = 20,
        batchRetranscribe: Bool = true,
        finalizationPass: FinalizationPass = .activeSegmentOnly
    ) {
        self.chunkSec = chunkSec; self.rollback = rollback; self.unfixedChunks = unfixedChunks
        self.maxNewTokens = maxNewTokens; self.maxEncWindows = maxEncWindows
        self.maxPrefixTokens = maxPrefixTokens; self.batchRetranscribe = batchRetranscribe
        self.finalizationPass = finalizationPass
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

public final class StreamingSession: @unchecked Sendable {
    private let transcriber: Qwen3ASRTranscriber
    private let batchTranscriber: Qwen3ASRTranscriber?
    private let config: StreamConfig
    private let language: String?
    private var audioBuffer: [Float] = []
    private var sessionAudioBuffer: [Float] = []
    private var encWindowCache: [MLXArray] = []
    private var nextWindowStart: Int = 0
    private let encWindowSamples: Int
    private var kvCache: [KVCache]
    private var prevPrefillEmbeds: MLXArray?
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
    private static let silenceRMS: Float = 0.003
    private static let speechStartRMS: Float = 0.010
    private static let pauseRMS: Float = 0.020
    private static let pauseChunks = 1
    private static let stallRefreshRMS: Float = 0.008
    private static let liveBatchRefreshPolicy = LiveBatchRefreshPolicy.default
    // Session-context retranscribe walks the whole session buffer. Cap that path
    // for long sessions to avoid multi-second stalls during live dictation.
    private static let maxSessionContextSamples = ASRAudio.sampleRate * 45
    // Cap expensive mid-session batch correction windows. For longer active
    // segments we commit the streaming text directly and keep moving.
    private static let maxLiveBatchSegmentSamples = ASRAudio.sampleRate * 12

    public init(
        transcriber: Qwen3ASRTranscriber,
        batchTranscriber: Qwen3ASRTranscriber? = nil,
        config: StreamConfig = StreamConfig(),
        language: String? = nil
    ) {
        self.transcriber = transcriber
        self.batchTranscriber = batchTranscriber
        self.config = config
        self.language = language?.trimmingCharacters(in: .whitespacesAndNewlines)
        self.encWindowSamples = transcriber.model.config.audioConfig.nWindowInfer * ASRAudio.hopLength
        self.kvCache = transcriber.model.makeCache()
    }

    /// Process one chunk of audio. Returns partial transcription result.
    public func processChunk(_ audioChunk: [Float], speechHint: SpeechActivityHint? = nil) -> ChunkResult {
        let t0 = Date()
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
                return ChunkResult(
                    text: committedText, isPartial: true, batchCorrected: true,
                    totalMs: Date().timeIntervalSince(t0) * 1000
                )
            }
        }

        if !chunkHasSpeech && rms < Self.silenceRMS {
            chunkIdx += 1
            let activeText = rawTokens.isEmpty ? "" : extractText(rawTokens)
            let combined = Self.appendSegment(committedText, activeText)
            lastText = combined
            return ChunkResult(text: combined, isPartial: true, totalMs: Date().timeIntervalSince(t0) * 1000)
        }

        if chunkHasSpeech { hasSpeech = true }
        if !hasSpeech {
            audioBuffer.append(contentsOf: audioChunk)
            chunkIdx += 1
            return ChunkResult(text: committedText, isPartial: true, totalMs: Date().timeIntervalSince(t0) * 1000)
        }

        audioBuffer.append(contentsOf: audioChunk)

        // 1. Encode
        let encT0 = Date()
        let encOutput = encodeIncremental()
        let numEncTokens = encOutput.shape[0]
        let encodeMs = Date().timeIntervalSince(encT0) * 1000

        if numEncTokens == 0 {
            chunkIdx += 1
            return ChunkResult(text: committedText, isPartial: true, totalMs: Date().timeIntervalSince(t0) * 1000)
        }

        // 2. Prefix tokens (rollback)
        var prefixTokens: [Int] = []
        if chunkIdx >= config.unfixedChunks && !rawTokens.isEmpty {
            let nPrefix = max(0, rawTokens.count - config.rollback)
            prefixTokens = Array(rawTokens.prefix(nPrefix))
            prefixTokens = transcriber.tokenizer.stripAutoLanguagePrefix(prefixTokens)
            if prefixTokens.count > config.maxPrefixTokens {
                prefixTokens = Array(prefixTokens.suffix(config.maxPrefixTokens))
            }
        }

        // 3. Input embeddings
        let inputEmbeds = buildInputEmbeds(
            encOutput: encOutput, numEncTokens: numEncTokens, prefixTokens: prefixTokens
        )
        eval(inputEmbeds)

        // 4. Delta prefill
        let prefillT0 = Date()
        let reuseLen = computeReuseLength(prevEmbeds: prevPrefillEmbeds, newEmbeds: inputEmbeds)
        let totalLen = inputEmbeds.shape[1]
        let prefillLen = totalLen - 1

        for c in kvCache { c.offset = reuseLen }

        let deltaLen = prefillLen - reuseLen

        // If embedding shrank, correct the offset
        if deltaLen < 0 {
            for c in kvCache { c.offset = prefillLen }
        } else if deltaLen > 0 {
            let deltaEmbeds = inputEmbeds[0..., reuseLen ..< prefillLen, 0...]
            let dummyIds = MLXArray([Int32(0)]).expandedDimensions(axis: 0)
            let (dl, _) = transcriber.model(
                inputIds: dummyIds,
                inputEmbeddings: deltaEmbeds, cache: kvCache
            )
            eval(dl)
        }

        let dummyIds = MLXArray([Int32(0)]).expandedDimensions(axis: 0)
        let lastEmbed = inputEmbeds[0..., prefillLen ..< (prefillLen + 1), 0...]
        let (logits, _) = transcriber.model(
            inputIds: dummyIds,
            inputEmbeddings: lastEmbed, cache: kvCache
        )
        eval(logits)

        let prefillMs = Date().timeIntervalSince(prefillT0) * 1000
        let reusePct = Double(reuseLen) / Double(max(totalLen, 1)) * 100

        prevPrefillEmbeds = inputEmbeds[0..., 0 ..< prefillLen, 0...]

        // 5. Decode
        let decodeT0 = Date()
        let newTokens = decodeTokens(logits: logits, maxTokens: config.maxNewTokens)
        let decodeMs = Date().timeIntervalSince(decodeT0) * 1000

        // 6. Update tokens
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
        kvCache = transcriber.model.makeCache()
        prevPrefillEmbeds = nil
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
        guard !rawTokens.isEmpty else { return "" }
        return extractText(rawTokens).trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Finalize the session on stop. By default we batch only the trailing
    /// active segment and preserve every previously committed segment verbatim.
    /// Full-session retranscribe is available only as an explicit opt-in for
    /// A/B comparisons.
    public func finalize() -> String {
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

        // Evict oldest
        while encWindowCache.count > config.maxEncWindows {
            encWindowCache.removeFirst()
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
    ) -> MLXArray {
        let promptIds = transcriber.tokenizer.buildPrompt(numAudioTokens: numEncTokens, language: language)
        let inputIds = MLXArray(promptIds.map { Int32($0) }).expandedDimensions(axis: 0)

        var embeds = transcriber.model.buildInputsEmbeds(
            inputIds: inputIds, audioFeatures: encOutput,
            numAudioTokens: numEncTokens, audioStartIndex: 9
        )

        if !prefixTokens.isEmpty {
            let pfxIds = MLXArray(prefixTokens.map { Int32($0) }).expandedDimensions(axis: 0)
            let pfxEmbed = transcriber.model.model.embedTokens(pfxIds)
            embeds = MLX.concatenated([embeds, pfxEmbed], axis: 1)
        }

        return embeds
    }

    private func computeReuseLength(prevEmbeds: MLXArray?, newEmbeds: MLXArray) -> Int {
        guard let prev = prevEmbeds else { return 0 }
        let cmpLen = min(prev.shape[1], newEmbeds.shape[1])
        if cmpLen == 0 { return 0 }

        let prevSlice = prev[0, 0 ..< cmpLen].asType(.float32)
        let newSlice = newEmbeds[0, 0 ..< cmpLen].asType(.float32)
        let diff = MLX.abs(prevSlice - newSlice).sum(axis: -1) // shape: (cmpLen,)
        let mask = (diff .> MLXArray(Float(1e-4))).asType(.int32)
        let total = mask.sum()
        let first = MLX.argMax(mask)
        eval(total, first)

        if total.item(Int.self) == 0 { return cmpLen }
        return first.item(Int.self)
    }

    /// Decode with double-buffer asyncEval pattern:
    /// Sample current token while next forward pass runs on GPU.
    private func decodeTokens(logits: MLXArray, maxTokens: Int) -> [Int] {
        let eos = Qwen3ASRTokenizer.eosTokens
        let repPenalty = Float(1.3)
        let repWindow = 8
        var tokens: [Int] = []

        // Sample first token, queue next forward
        var y = sampleWithPenalty(logits: logits, recent: [], penalty: repPenalty)
        var curLogits = logits

        for _ in 0 ..< maxTokens {
            // Queue next forward while waiting for current sample
            let tokId = y.asType(.int32).expandedDimensions(axis: 0).expandedDimensions(axis: 0)
            let tokEmbed = transcriber.model.model.embedTokens(tokId)
            (curLogits, _) = transcriber.model(
                inputIds: tokId, inputEmbeddings: tokEmbed, cache: kvCache
            )
            let nextY = sampleWithPenalty(
                logits: curLogits, recent: Array(tokens.suffix(repWindow)), penalty: repPenalty
            )
            asyncEval(nextY)

            // Now read current token (blocks until y is ready)
            let token = y.item(Int.self)
            if eos.contains(token) { break }
            tokens.append(token)
            if tokens.count >= 4 && Set(tokens.suffix(4)).count == 1 { break }

            y = nextY
        }
        return tokens
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
            let batcher = batchTranscriber ?? transcriber
            let result = try batcher.transcribe(audio: audioBuffer, language: language)
            let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                rawTokens = transcriber.tokenizer.encode(text)
            }
            // Reset cache — batch changed the token sequence
            kvCache = transcriber.model.makeCache()
            prevPrefillEmbeds = nil
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

    private func batchFinalizeSegmentText() -> String? {
        if let activeText = sessionContextActiveText(), !activeText.isEmpty {
            fputs("[StreamingSession] Session-context final segment: \(sessionAudioBuffer.count / ASRAudio.sampleRate)s audio → \(activeText.count) chars\n", stderr)
            return activeText
        }

        // Hard guard: avoid giant end-of-session batch retranscribes that can
        // spike GPU memory after long uninterrupted dictation.
        if audioBuffer.count > Self.maxLiveBatchSegmentSamples {
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
        if sessionAudioBuffer.count > Self.maxSessionContextSamples {
            fputs("[StreamingSession] Session-context skipped (session too long: \(sessionAudioBuffer.count / ASRAudio.sampleRate)s)\n", stderr)
            return nil
        }

        do {
            let batcher = batchTranscriber ?? transcriber
            let result = try batcher.transcribe(audio: sessionAudioBuffer, language: language)
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
            let batcher = batchTranscriber ?? transcriber
            let result = try batcher.transcribe(audio: sessionAudioBuffer, language: language)
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
