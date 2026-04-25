import Foundation

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
}

public protocol BatchTranscriptionServing: AnyObject, Sendable {
    func transcribeChunk(audio: [Float], language: String?, temperature: Float) throws -> TranscriptionResult
    func subtitleItems(
        audio: [Float],
        transcript: String?,
        language: String?,
        temperature: Float,
        aligner: ForcedAligner
    ) throws -> (transcript: String, language: String, items: [ForcedAlignItem])
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
        log: @Sendable (String) -> Void = { _ in }
    ) throws -> TranscriptionResult {
        let startedAt = Date()
        let chunks = try chunkAudio(audio, vad: vad)
        let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
        let chunkMode = vad == nil ? "energy" : "VAD"
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
        log: @Sendable (String) -> Void = { _ in }
    ) throws -> BatchSubtitleResult {
        let startedAt = Date()
        let chunks = try chunkAudio(audio, vad: vad)
        let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
        let chunkMode = vad == nil ? "energy" : "VAD"
        log("\(chunkMode) chunking subtitles: \(chunks.count) chunks from \(String(format: "%.1f", audioDuration))s")
        log("\(chunkMode) chunk ranges subtitles: \(formatChunkRanges(chunks))")

        var allItems: [ForcedAlignItem] = []
        var transcriptParts: [String] = []
        var resolvedLanguage = language ?? "English"
        var debugChunks: [BatchSubtitleChunkDebug] = []

        if let transcript, !transcript.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let textParts = splitTextProportionally(transcript, chunkDurations: chunks.map(\.duration))
            for (index, pair) in zip(chunks.indices, zip(chunks, textParts)) {
                let (chunk, textPart) = pair
                let part = try service.subtitleItems(
                    audio: chunk.audio,
                    transcript: textPart,
                    language: language,
                    temperature: temperature,
                    aligner: aligner
                )
                transcriptParts.append(part.transcript)
                resolvedLanguage = language ?? part.language
                let offsetItems = offsetAlignmentItems(part.items, by: chunk.startTime)
                allItems.append(contentsOf: offsetItems)
                debugChunks.append(makeDebugChunk(index: index + 1, chunk: chunk, transcript: part.transcript, items: offsetItems))
            }
        } else {
            for (index, chunk) in chunks.enumerated() {
                let part = try service.subtitleItems(
                    audio: chunk.audio,
                    transcript: nil,
                    language: language,
                    temperature: temperature,
                    aligner: aligner
                )
                let trimmed = part.transcript.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { transcriptParts.append(trimmed) }
                resolvedLanguage = language ?? part.language
                let offsetItems = offsetAlignmentItems(part.items, by: chunk.startTime)
                allItems.append(contentsOf: offsetItems)
                debugChunks.append(makeDebugChunk(index: index + 1, chunk: chunk, transcript: part.transcript, items: offsetItems))
            }
        }

        return BatchSubtitleResult(
            transcript: AlignedTextRenderer.render(segments: transcriptParts),
            language: resolvedLanguage,
            items: allItems,
            audioDuration: audioDuration,
            processingTime: Date().timeIntervalSince(startedAt),
            debug: BatchSubtitleDebug(chunkingMode: chunkMode, chunkCount: chunks.count, chunks: debugChunks)
        )
    }

    public static func chunkAudio(_ audio: [Float], vad: SileroVAD?) throws -> [AudioChunk] {
        if let vad {
            vadLock.lock()
            defer { vadLock.unlock() }
            return try vad.chunk(audio: audio, config: BatchTranscriptionDefaults.vadConfig)
        }

        return chunkAudioByEnergy(audio, sampleRate: ASRAudio.sampleRate, config: BatchTranscriptionDefaults.energyConfig)
    }

    public static func splitTextProportionally(_ text: String, chunkDurations: [Double]) -> [String] {
        guard !chunkDurations.isEmpty else { return [text] }
        let totalDuration = chunkDurations.reduce(0, +)
        guard totalDuration > 0 else { return [text] }

        let characters = Array(text)
        let totalCount = characters.count
        var parts: [String] = []
        var textPosition = 0

        for (index, duration) in chunkDurations.enumerated() {
            if index == chunkDurations.count - 1 {
                parts.append(String(characters[textPosition...]).trimmingCharacters(in: .whitespacesAndNewlines))
                break
            }

            let proportion = duration / totalDuration
            let charsForChunk = Int(Double(totalCount) * proportion)
            let endPosition = min(totalCount, textPosition + charsForChunk)
            let searchRange = max(20, Int(Double(charsForChunk) * 0.1))
            var bestPosition = endPosition
            let separators = Set(" 。．！？、，.!?,\n")

            outer: for offset in 0 ..< searchRange {
                for checkPosition in [endPosition + offset, endPosition - offset] {
                    guard checkPosition >= 0, checkPosition < totalCount else { continue }
                    if separators.contains(characters[checkPosition]) {
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

private extension BatchTranscriptionPipeline {
    static func formatChunkRanges(_ chunks: [AudioChunk]) -> String {
        chunks.enumerated().map { index, chunk in
            String(format: "%d:%.3f-%.3f", index + 1, chunk.startTime, chunk.endTime)
        }.joined(separator: ",")
    }

    static func offsetAlignmentItems(_ items: [ForcedAlignItem], by offset: Double) -> [ForcedAlignItem] {
        items.map { item in
            ForcedAlignItem(text: item.text, startTime: item.startTime + offset, endTime: item.endTime + offset, alignText: item.alignText)
        }
    }

    static func makeDebugChunk(index: Int, chunk: AudioChunk, transcript: String, items: [ForcedAlignItem]) -> BatchSubtitleChunkDebug {
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
