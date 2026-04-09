// NativeASR — Streaming Session

import Foundation
import MLX

/// Streaming configuration.
public struct StreamConfig: Sendable {
    public var chunkSec: Double, rollback: Int, unfixedChunks: Int
    public var maxNewTokens: Int, maxEncWindows: Int, maxPrefixTokens: Int
    public var batchRetranscribe: Bool

    public init(
        chunkSec: Double = 2.0, rollback: Int = 5, unfixedChunks: Int = 2,
        maxNewTokens: Int = 32, maxEncWindows: Int = 4, maxPrefixTokens: Int = 20,
        batchRetranscribe: Bool = true
    ) {
        self.chunkSec = chunkSec; self.rollback = rollback; self.unfixedChunks = unfixedChunks
        self.maxNewTokens = maxNewTokens; self.maxEncWindows = maxEncWindows
        self.maxPrefixTokens = maxPrefixTokens; self.batchRetranscribe = batchRetranscribe
    }
}

public struct ChunkResult: Sendable {
    public let text: String
    public let isPartial: Bool
    public var batchCorrected: Bool = false
    public var encodeMs: Double = 0, prefillMs: Double = 0, decodeMs: Double = 0
    public var totalMs: Double = 0, reusePct: Double = 0
}

public final class StreamingSession: @unchecked Sendable {
    private let transcriber: Qwen3ASRTranscriber
    private let batchTranscriber: Qwen3ASRTranscriber?
    private let config: StreamConfig
    private var audioBuffer: [Float] = []
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
    /// Concatenated text from all previously committed segments. Frozen — never
    /// rewritten by streaming or batch passes after a commit fires.
    private var committedText: String = ""
    private static let silenceRMS: Float = 0.003
    private static let pauseRMS: Float = 0.020
    private static let pauseChunks = 2

    public init(
        transcriber: Qwen3ASRTranscriber,
        batchTranscriber: Qwen3ASRTranscriber? = nil,
        config: StreamConfig = StreamConfig()
    ) {
        self.transcriber = transcriber
        self.batchTranscriber = batchTranscriber
        self.config = config
        self.encWindowSamples = transcriber.model.config.audioConfig.nWindowInfer * ASRAudio.hopLength
        self.kvCache = transcriber.model.makeCache()
    }

    /// Process one chunk of audio. Returns partial transcription result.
    public func processChunk(_ audioChunk: [Float]) -> ChunkResult {
        let t0 = Date()
        let rms = Self.computeRMS(audioChunk)

        if rms < Self.pauseRMS { consecutiveSilence += 1 }
        else { consecutiveSilence = 0; batchDoneForPause = false }

        // Segment commit on pause: batch retranscribe the active segment,
        // append it to committedText, then reset streaming state so the next
        // chunks build a fresh active segment. Committed text is never rewritten.
        if config.batchRetranscribe
            && consecutiveSilence >= Self.pauseChunks && !batchDoneForPause && !rawTokens.isEmpty
        {
            batchDoneForPause = true
            if let segmentText = batchRetranscribe() {
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

        if rms < Self.silenceRMS {
            chunkIdx += 1
            let activeText = rawTokens.isEmpty ? "" : extractText(rawTokens)
            let combined = Self.appendSegment(committedText, activeText)
            lastText = combined
            return ChunkResult(text: combined, isPartial: true, totalMs: Date().timeIntervalSince(t0) * 1000)
        }

        if rms >= Self.pauseRMS { hasSpeech = true }
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
            prefixTokens = prefixTokens.filter { !Qwen3ASRTokenizer.allLangTokens.contains($0) }
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
            rawTokens = uncappedPrefix + newTokens
        }

        var activeText = extractText(rawTokens)
        if let trimmed = trimRepetition() { activeText = trimmed }
        let combined = Self.appendSegment(committedText, activeText)

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
    }

    public var processedChunkCount: Int { chunkIdx }

    public func finalText() -> String { lastText }

    /// Finalize the session on stop. Batch retranscribes the active segment
    /// (if any), appends it to committed text, and returns the full transcript.
    /// The committed prefix is never re-batched.
    public func finalize() -> String {
        // For the *first* segment of a session (no commits yet), match the old
        // behavior: batch any audio >= 1s, no speech check. This preserves
        // transcription of quiet speech that never crosses pauseRMS.
        //
        // For *post-commit* trailing segments, require `hasSpeech == true`.
        // Pure-silence trailing audio after a commit can hallucinate ("None",
        // "I guess", etc) — confirmed by offline experiments. The committed
        // prefix is always preserved either way.
        let isPostCommit = !committedText.isEmpty
        let canBatch = config.batchRetranscribe
            && audioBuffer.count >= ASRAudio.sampleRate
            && (!isPostCommit || hasSpeech)

        var activeText = ""
        if canBatch, let segmentText = batchRetranscribe() {
            activeText = segmentText
        } else if !rawTokens.isEmpty {
            // Batch disabled, audio too short, or batch failed — fall back to
            // streaming output if any tokens were decoded
            activeText = extractText(rawTokens)
        }
        committedText = Self.appendSegment(committedText, activeText)
        lastText = committedText
        return committedText
    }

    /// Concatenate two segment texts with a single space, handling empty inputs
    /// and avoiding double-spaces.
    static func appendSegment(_ committed: String, _ segment: String) -> String {
        let trimmedSegment = segment.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmedSegment.isEmpty { return committed }
        if committed.isEmpty { return trimmedSegment }
        return committed + " " + trimmedSegment
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
        let promptIds = transcriber.tokenizer.buildPrompt(numAudioTokens: numEncTokens)
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
        let filtered = tokens.filter { !Qwen3ASRTokenizer.allLangTokens.contains($0) }
        let raw = transcriber.tokenizer.decode(filtered)
        let cleaned = transcriber.tokenizer.cleanOutput(raw)
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
            .filter { !Qwen3ASRTokenizer.allLangTokens.contains($0) }
        return trimmed
    }

    private func batchRetranscribe() -> String? {
        guard audioBuffer.count >= ASRAudio.sampleRate else { return nil }

        do {
            let batcher = batchTranscriber ?? transcriber
            let result = try batcher.transcribe(audio: audioBuffer)
            let text = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !text.isEmpty {
                rawTokens = transcriber.tokenizer.encode(text)
                    .filter { !Qwen3ASRTokenizer.allLangTokens.contains($0) }
            }
            // Reset cache — batch changed the token sequence
            kvCache = transcriber.model.makeCache()
            prevPrefillEmbeds = nil
            fputs("[StreamingSession] Batch retranscribe: \(audioBuffer.count / ASRAudio.sampleRate)s audio → \(text.count) chars\n", stderr)
            return text
        } catch {
            fputs("[StreamingSession] Batch retranscribe error: \(error)\n", stderr)
            return nil
        }
    }

    private static func computeRMS(_ audio: [Float]) -> Float {
        guard !audio.isEmpty else { return 0 }
        var sum: Float = 0
        for s in audio { sum += s * s }
        return sqrt(sum / Float(audio.count))
    }
}
