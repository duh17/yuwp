import Foundation

public enum BatchChunkingMode: String, Codable, CaseIterable, Sendable {
    case automatic
    case vad
    case energy

    /// Resolve the requested batch policy for one decoded audio duration.
    /// Explicit VAD safely falls back to energy when the VAD model is unavailable.
    public func resolved(audioDuration: Double, hasVAD: Bool) -> BatchChunkingMode {
        switch self {
        case .automatic:
            return audioDuration > BatchTranscriptionDefaults.maxChunkDurationSec && hasVAD
                ? .energy
                : (hasVAD ? .vad : .energy)
        case .vad:
            return hasVAD ? .vad : .energy
        case .energy:
            return .energy
        }
    }
}

public enum BatchTranscriptionDefaults {
    public static let maxChunkDurationSec = 120.0

    public static let vadConfig = VADChunkingConfig(
        threshold: 0.6,
        minSpeechDuration: 0.25,
        minSilenceDuration: 0.08,
        speechPad: 0.02,
        splitMinSilenceDuration: 0.5,
        maxChunkDuration: maxChunkDurationSec,
        // Keep clearly separated utterances as distinct chunks so batch ASR
        // does not flatten mixed-language segments into a single dominant
        // language when the clip is still well under the hard max length.
        minChunkDuration: 2.0
    )

    public static let energyConfig = EnergyChunkingConfig(
        maxChunkDuration: maxChunkDurationSec,
        minChunkDuration: 1.0,
        searchExpandDuration: 5.0,
        energyWindowDuration: 0.1,
        minProgressDuration: 1.0
    )

    /// Subtitle ASR and alignment share these audio windows. Ordinary batch stays at 120s.
    public static let subtitleAlignmentConfig = EnergyChunkingConfig(
        maxChunkDuration: 30.0,
        minChunkDuration: 1.0,
        searchExpandDuration: 2.0,
        energyWindowDuration: 0.1,
        minProgressDuration: 1.0
    )
}

public protocol BatchTranscriptionServing: AnyObject, Sendable {
    var hasR2T2BatchDelimiter: Bool { get }
    func transcribeChunk(audio: [Float], language: String?, temperature: Float) throws -> TranscriptionResult
    func subtitleItems(
        audio: [Float],
        transcript: String?,
        language: String?,
        temperature: Float,
        aligner: ForcedAligner
    ) throws -> (transcript: String, language: String, items: [ForcedAlignItem])
}

public extension BatchTranscriptionServing {
    var hasR2T2BatchDelimiter: Bool { false }
}

public struct BatchAlignmentItemDebug: Sendable, Codable {
    public let text: String
    public let alignText: String?
    public let start: Double
    public let end: Double

    public init(text: String, alignText: String? = nil, start: Double, end: Double) {
        self.text = text
        self.alignText = alignText
        self.start = start
        self.end = end
    }
}

public struct BatchSubtitleChunkDebug: Sendable, Codable {
    public let index: Int
    public let start: Double
    public let end: Double
    public let duration: Double
    public let transcript: String
    public let items: [BatchAlignmentItemDebug]

    public init(index: Int, start: Double, end: Double, duration: Double, transcript: String, items: [BatchAlignmentItemDebug]) {
        self.index = index
        self.start = start
        self.end = end
        self.duration = duration
        self.transcript = transcript
        self.items = items
    }
}

public struct BatchSubtitleDebug: Sendable, Codable {
    public let chunkingMode: String
    public let chunkCount: Int
    public let chunks: [BatchSubtitleChunkDebug]

    public init(chunkingMode: String, chunkCount: Int, chunks: [BatchSubtitleChunkDebug]) {
        self.chunkingMode = chunkingMode
        self.chunkCount = chunkCount
        self.chunks = chunks
    }
}

public struct BatchSubtitleResult: Sendable {
    public let transcript: String
    public let language: String
    public let items: [ForcedAlignItem]
    public let audioDuration: Double
    public let processingTime: Double
    public let debug: BatchSubtitleDebug?

    public init(
        transcript: String,
        language: String,
        items: [ForcedAlignItem],
        audioDuration: Double,
        processingTime: Double,
        debug: BatchSubtitleDebug? = nil
    ) {
        self.transcript = transcript
        self.language = language
        self.items = items
        self.audioDuration = audioDuration
        self.processingTime = processingTime
        self.debug = debug
    }
}

public enum BatchTranscriptionPipeline {
    private static let vadLock = NSLock()

    public static func transcribe(
        using service: any BatchTranscriptionServing,
        audio: [Float],
        language: String?,
        temperature: Float,
        vad: SileroVAD?,
        chunking: BatchChunkingMode = .automatic,
        log: @Sendable (String) -> Void = { _ in }
    ) throws -> TranscriptionResult {
        let startedAt = Date()
        let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
        let resolvedChunking = chunking.resolved(audioDuration: audioDuration, hasVAD: vad != nil)
        let chunks = try chunkAudio(audio, vad: vad, chunking: resolvedChunking)
        let chunkMode = chunkingLabel(resolvedChunking)
        log("\(chunkMode) chunking transcription: \(chunks.count) chunks from \(String(format: "%.1f", audioDuration))s")
        log("\(chunkMode) chunk ranges transcription: \(formatChunkRanges(chunks))")

        var texts: [String] = []
        var resolvedLanguage = language
        for chunk in chunks {
            let chunkResult = try service.transcribeChunk(audio: chunk.audio, language: language, temperature: temperature)
            let trimmed = chunkResult.text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { texts.append(trimmed) }
            if resolvedLanguage == nil { resolvedLanguage = chunkResult.language }
        }

        return TranscriptionResult(
            text: AlignedTextRenderer.render(segments: texts),
            language: resolvedLanguage,
            audioDuration: audioDuration,
            processingTime: Date().timeIntervalSince(startedAt)
        )
    }

    public static func subtitle(
        using service: any BatchTranscriptionServing,
        audio: [Float],
        transcript: String?,
        language: String?,
        temperature: Float,
        aligner: ForcedAligner,
        vad: SileroVAD?,
        chunking: BatchChunkingMode = .automatic,
        log: @Sendable (String) -> Void = { _ in }
    ) throws -> BatchSubtitleResult {
        try subtitle(
            using: service, audio: audio, transcript: transcript, language: language,
            temperature: temperature, vad: vad, chunking: chunking, log: log,
            alignItems: { window, text, language, temperature in
                try service.subtitleItems(audio: window, transcript: text, language: language,
                                          temperature: temperature, aligner: aligner)
            }
        )
    }

    // Inject only the alignment boundary so windowing and ASR granularity can be
    // tested without loading an MLX model. Production always uses subtitleItems.
    static func subtitle(
        using service: any BatchTranscriptionServing,
        audio: [Float],
        transcript: String?,
        language: String?,
        temperature: Float,
        vad: SileroVAD?,
        chunking: BatchChunkingMode = .automatic,
        log: @Sendable (String) -> Void = { _ in },
        alignItems: ([Float], String, String?, Float) throws -> (transcript: String, language: String, items: [ForcedAlignItem])
    ) throws -> BatchSubtitleResult {
        let startedAt = Date()
        let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
        let resolvedChunking = subtitleChunkingMode(chunking, audioDuration: audioDuration, hasVAD: vad != nil)
        let asrChunks = try subtitleAudioChunks(audio, vad: vad, chunking: resolvedChunking)
        let chunkMode = chunkingLabel(resolvedChunking)
        log("\(chunkMode) pre-ASR subtitle chunks: \(asrChunks.count) from \(String(format: "%.1f", audioDuration))s")
        log("\(chunkMode) pre-ASR subtitle ranges: \(formatChunkRanges(asrChunks))")

        var allItems: [ForcedAlignItem] = []
        var transcriptParts: [String] = []
        var resolvedLanguage: String? = language
        var debugChunks: [BatchSubtitleChunkDebug] = []
        let suppliedText = transcript?.trimmingCharacters(in: .whitespacesAndNewlines)
        let hasSuppliedText = !(suppliedText ?? "").isEmpty
        if hasSuppliedText, resolvedLanguage == nil {
            resolvedLanguage = inferredSubtitleLanguage(suppliedText ?? "")
        }
        // User-supplied text has no clip-level transcript. This legacy path must
        // approximate its placement; ASR-generated text is never split by length.
        let suppliedParts = hasSuppliedText
            ? splitTextProportionally(suppliedText ?? "", chunkDurations: asrChunks.map(\.duration))
            : []

        for (index, chunk) in asrChunks.enumerated() {
            let text: String
            let alignmentLanguage: String?
            if hasSuppliedText {
                text = suppliedParts[index]
                alignmentLanguage = resolvedLanguage
            } else {
                let recognized = try service.transcribeChunk(audio: chunk.audio, language: language, temperature: temperature)
                let raw = recognized.text.trimmingCharacters(in: .whitespacesAndNewlines)
                text = service.hasR2T2BatchDelimiter
                    ? StablePrefixCommitter.pipeCut(StablePrefixCommitter.stripMeta(raw)).trimmingCharacters(in: .whitespacesAndNewlines)
                    : raw
                alignmentLanguage = language ?? recognized.language ?? inferredSubtitleLanguage(text) ?? resolvedLanguage
                if resolvedLanguage == nil { resolvedLanguage = alignmentLanguage }
                if !text.isEmpty { transcriptParts.append(text) }
            }

            // The production subtitleItems implementation treats empty text as
            // a request to run ASR; do not transcribe silent or empty splits twice.
            var items: [ForcedAlignItem] = []
            if !text.isEmpty {
                let part = try alignItems(chunk.audio, text, alignmentLanguage, temperature)
                items = boundedAlignmentItems(collapsePinnedItems(part.items, duration: chunk.duration),
                                              duration: chunk.duration)
            }
            let offsetItems = offsetAlignmentItems(items, by: chunk.startTime)
            allItems.append(contentsOf: offsetItems)
            debugChunks.append(makeDebugChunk(index: index + 1, chunk: chunk,
                                               transcript: text, items: offsetItems))
        }

        return BatchSubtitleResult(
            transcript: hasSuppliedText ? (suppliedText ?? "") : AlignedTextRenderer.render(segments: transcriptParts),
            language: resolvedLanguage ?? "English",
            items: allItems,
            audioDuration: audioDuration,
            processingTime: Date().timeIntervalSince(startedAt),
            debug: BatchSubtitleDebug(chunkingMode: "\(chunkMode)-pre-asr-cap", chunkCount: debugChunks.count, chunks: debugChunks)
        )
    }

    static func subtitleChunkingMode(_ mode: BatchChunkingMode, audioDuration: Double, hasVAD: Bool) -> BatchChunkingMode {
        mode == .automatic ? (hasVAD ? .vad : .energy)
            : mode.resolved(audioDuration: audioDuration, hasVAD: hasVAD)
    }

    /// Only infer from scripts that are unambiguous enough for the aligner.
    /// A caller can pass a language hint for mixed or all-Kanji Japanese text.
    static func inferredSubtitleLanguage(_ text: String) -> String? {
        if text.unicodeScalars.contains(where: { (0x3040 ... 0x30FF).contains($0.value) }) { return "Japanese" }
        if text.unicodeScalars.contains(where: { (0xAC00 ... 0xD7AF).contains($0.value) }) { return "Korean" }
        if text.unicodeScalars.contains(where: { ScriptClassifier.isCJK($0) }) { return "Chinese" }
        return nil
    }

    /// Preserve VAD pauses (or explicit energy mode), then enforce a hard 30s
    /// bound on every clip before either ASR or forced alignment sees it.
    static func subtitleAudioChunks(_ audio: [Float], vad: SileroVAD?, chunking: BatchChunkingMode) throws -> [AudioChunk] {
        let parents = try chunkAudio(audio, vad: vad, chunking: chunking)
        return parents.flatMap { parent in
            chunkAudioByEnergy(parent.audio, sampleRate: ASRAudio.sampleRate,
                               config: BatchTranscriptionDefaults.subtitleAlignmentConfig,
                               strictMaxDuration: true).map { local in
                AudioChunk(audio: local.audio, startTime: parent.startTime + local.startTime,
                           endTime: parent.startTime + local.endTime)
            }
        }
    }

    public static func chunkAudio(
        _ audio: [Float],
        vad: SileroVAD?,
        chunking: BatchChunkingMode = .automatic
    ) throws -> [AudioChunk] {
        let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
        switch chunking.resolved(audioDuration: audioDuration, hasVAD: vad != nil) {
        case .vad:
            guard let vad else {
                return chunkAudioByEnergy(audio, sampleRate: ASRAudio.sampleRate, config: BatchTranscriptionDefaults.energyConfig)
            }
            vadLock.lock()
            defer { vadLock.unlock() }
            return try vad.chunk(audio: audio, config: BatchTranscriptionDefaults.vadConfig)
        case .automatic, .energy:
            return chunkAudioByEnergy(audio, sampleRate: ASRAudio.sampleRate, config: BatchTranscriptionDefaults.energyConfig)
        }
    }

    public static func splitTextProportionally(_ text: String, chunkDurations: [Double]) -> [String] {
        guard !chunkDurations.isEmpty else { return [text] }
        let totalDuration = chunkDurations.reduce(0, +)
        guard totalDuration > 0 else { return [text] }

        let characters = Array(text)
        let totalCount = characters.count
        var parts: [String] = []
        var textPosition = 0

        var elapsedDuration = 0.0
        for (index, duration) in chunkDurations.enumerated() {
            if index == chunkDurations.count - 1 {
                parts.append(String(characters[textPosition...]).trimmingCharacters(in: .whitespacesAndNewlines))
                break
            }

            elapsedDuration += duration
            // Round cumulative share upward: a short transcript must not be
            // postponed to the final alignment window by repeated floor(0).
            let endPosition = min(totalCount, max(textPosition,
                Int((Double(totalCount) * elapsedDuration / totalDuration).rounded(.up))))
            let charsForChunk = endPosition - textPosition
            let searchRange = max(2, min(20, Int(Double(charsForChunk) * 0.1)))
            var bestPosition = endPosition
            let separators = Set(" 。．！？、，.!?,\n")

            outer: for offset in 0 ... searchRange {
                for checkPosition in [endPosition + offset, endPosition - offset] {
                    guard checkPosition >= 0, checkPosition < totalCount else { continue }
                    if checkPosition > textPosition, separators.contains(characters[checkPosition]) {
                        bestPosition = min(totalCount, checkPosition + 1)
                        break outer
                    }
                }
            }

            if bestPosition <= textPosition {
                bestPosition = min(totalCount, endPosition)
            }
            parts.append(String(characters[textPosition ..< bestPosition]).trimmingCharacters(in: .whitespacesAndNewlines))
            textPosition = bestPosition
        }

        return parts
    }
}

public func splitTextProportionally(_ text: String, chunkDurations: [Double]) -> [String] {
    BatchTranscriptionPipeline.splitTextProportionally(text, chunkDurations: chunkDurations)
}

extension BatchTranscriptionPipeline {
    private static func chunkingLabel(_ mode: BatchChunkingMode) -> String {
        mode == .vad ? "VAD" : "energy"
    }

    private static func formatChunkRanges(_ chunks: [AudioChunk]) -> String {
        chunks.enumerated().map { index, chunk in
            String(format: "%d:%.3f-%.3f", index + 1, chunk.startTime, chunk.endTime)
        }.joined(separator: ",")
    }

    /// Clamp model predictions to the local audio and keep output ordered.
    static func boundedAlignmentItems(_ items: [ForcedAlignItem], duration: Double) -> [ForcedAlignItem] {
        var previousEnd = 0.0
        return items.map { item in
            let start = item.startTime.isFinite ? item.startTime : previousEnd
            let end = item.endTime.isFinite ? item.endTime : start
            let boundedStart = min(max(start, previousEnd), duration)
            let boundedEnd = min(max(end, boundedStart), duration)
            previousEnd = boundedEnd
            return ForcedAlignItem(text: item.text, startTime: boundedStart, endTime: boundedEnd,
                                   alignText: item.alignText)
        }
    }

    /// A shared positive interval is evidence for the whole phrase, not for
    /// artificial word boundaries inside it. Leave zero or invalid pins untimed.
    static func collapsePinnedItems(_ items: [ForcedAlignItem], duration: Double) -> [ForcedAlignItem] {
        guard items.count > 1, let first = items.first,
              first.startTime.isFinite, first.endTime.isFinite,
              first.startTime >= 0, first.endTime > first.startTime, first.endTime <= duration,
              items.allSatisfy({ $0.startTime == first.startTime && $0.endTime == first.endTime }) else {
            return items
        }
        return [ForcedAlignItem(text: AlignedTextRenderer.render(tokens: items.map(\.text)),
                                startTime: first.startTime, endTime: first.endTime)]
    }

    private static func offsetAlignmentItems(_ items: [ForcedAlignItem], by offset: Double) -> [ForcedAlignItem] {
        items.map { item in
            ForcedAlignItem(text: item.text, startTime: item.startTime + offset, endTime: item.endTime + offset, alignText: item.alignText)
        }
    }

    private static func makeDebugChunk(index: Int, chunk: AudioChunk, transcript: String, items: [ForcedAlignItem]) -> BatchSubtitleChunkDebug {
        BatchSubtitleChunkDebug(
            index: index,
            start: chunk.startTime,
            end: chunk.endTime,
            duration: chunk.duration,
            transcript: transcript,
            items: items.map { BatchAlignmentItemDebug(text: $0.text, alignText: $0.alignText, start: $0.startTime, end: $0.endTime) }
        )
    }
}
