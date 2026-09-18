import Foundation

public struct AuKSpeechPlanRequest: Equatable, Sendable {
    public var task: String?
    public var mode: String?
    public var input: String?
    public var instruction: String?
    public var instructions: String?
    public var hasSourceAudio: Bool
    public var autoChunk: Bool?
    public var chunkTargetCharacters: Int?
    public var chunkHardCharacterLimit: Int?
    public var interChunkPauseSeconds: Double?
    public var genSeconds: Double?

    public init(
        task: String? = nil,
        mode: String? = nil,
        input: String? = nil,
        instruction: String? = nil,
        instructions: String? = nil,
        hasSourceAudio: Bool = false,
        autoChunk: Bool? = nil,
        chunkTargetCharacters: Int? = nil,
        chunkHardCharacterLimit: Int? = nil,
        interChunkPauseSeconds: Double? = nil,
        genSeconds: Double? = nil
    ) {
        self.task = task
        self.mode = mode
        self.input = input
        self.instruction = instruction
        self.instructions = instructions
        self.hasSourceAudio = hasSourceAudio
        self.autoChunk = autoChunk
        self.chunkTargetCharacters = chunkTargetCharacters
        self.chunkHardCharacterLimit = chunkHardCharacterLimit
        self.interChunkPauseSeconds = interChunkPauseSeconds
        self.genSeconds = genSeconds
    }

    public var requestsAutoChunk: Bool {
        autoChunk == true || chunkTargetCharacters != nil || chunkHardCharacterLimit != nil
    }
}

public struct AuKPreparedChunk: Equatable, Sendable {
    public var instruction: String
    public var genSeconds: Double?

    public init(instruction: String, genSeconds: Double?) {
        self.instruction = instruction
        self.genSeconds = genSeconds
    }
}

public struct AuKSpeechPlan: Equatable, Sendable {
    public enum Kind: String, Equatable, Sendable {
        case tts
        case singleTask
    }

    public var kind: Kind
    public var chunks: [AuKPreparedChunk]
    public var pauseSeconds: Double

    public init(kind: Kind, chunks: [AuKPreparedChunk], pauseSeconds: Double) {
        self.kind = kind
        self.chunks = chunks
        self.pauseSeconds = pauseSeconds
    }

    public var isMultiChunk: Bool { chunks.count > 1 }
}

public struct AuKChunkingRejection: Equatable, Sendable {
    public var code: String
    public var message: String
    public var options: [String: String]

    public init(code: String, message: String, options: [String: String]) {
        self.code = code
        self.message = message
        self.options = options
    }

    public var jsonObject: [String: Any] {
        [
            "error": message,
            "code": code,
            "options": options,
        ]
    }
}

public enum AuKSpeechPlanResult: Equatable, Sendable {
    case plan(AuKSpeechPlan)
    case rejected(AuKChunkingRejection)
}

public struct AuKExtractedSpeakText: Equatable, Sendable {
    public var prefix: String
    public var text: String
    public var suffix: String

    public init(prefix: String, text: String, suffix: String) {
        self.prefix = prefix
        self.text = text
        self.suffix = suffix
    }

    public func wrap(_ chunk: String) -> String {
        prefix + chunk + suffix
    }
}

public struct AuKSpeechJSON: Decodable, Equatable, Sendable {
    public var input: String?
    public var instruction: String?
    public var instructions: String?
    public var task: String?
    public var mode: String?
    public var autoChunk: Bool?
    public var nfe: Int?
    public var cfg: Float?
    public var cfgStrength: Float?
    public var sway: Float?
    public var genSeconds: Double?
    public var refAudio: String?
    public var chunkTargetCharacters: Int?
    public var chunkHardCharacterLimit: Int?
    public var chunkPauseMs: Double?
    public var interChunkPauseSeconds: Double?

    enum CodingKeys: String, CodingKey {
        case input, instruction, instructions, task, mode, nfe, cfg, sway
        case autoChunk = "auto_chunk"
        case cfgStrength = "cfg_strength"
        case genSeconds = "gen_seconds"
        case refAudio = "ref_audio"
        case chunkTargetCharacters = "chunk_target_characters"
        case chunkHardCharacterLimit = "chunk_hard_character_limit"
        case chunkPauseMs = "chunk_pause_ms"
        case interChunkPauseSeconds = "inter_chunk_pause_seconds"
    }

    public var resolvedCFG: Float? { cfgStrength ?? cfg }

    public var resolvedPauseSeconds: Double? {
        if let interChunkPauseSeconds { return interChunkPauseSeconds }
        if let chunkPauseMs { return chunkPauseMs / 1000 }
        return nil
    }

    public func planRequest(hasSourceAudio: Bool) -> AuKSpeechPlanRequest {
        AuKSpeechPlanRequest(
            task: task,
            mode: mode,
            input: input,
            instruction: instruction,
            instructions: instructions,
            hasSourceAudio: hasSourceAudio || (refAudio?.isEmpty == false),
            autoChunk: autoChunk,
            chunkTargetCharacters: chunkTargetCharacters,
            chunkHardCharacterLimit: chunkHardCharacterLimit,
            interChunkPauseSeconds: resolvedPauseSeconds,
            genSeconds: genSeconds
        )
    }
}

public struct AuKStreamMetadataEvent: Equatable, Sendable {
    public var format: String
    public var sampleRate: Int
    public var channels: Int
    public var encoding: String
    public var backend: String
    public var variant: String

    public init(
        format: String = "pcm_s16le",
        sampleRate: Int,
        channels: Int = 1,
        encoding: String = "base64",
        backend: String,
        variant: String
    ) {
        self.format = format
        self.sampleRate = sampleRate
        self.channels = channels
        self.encoding = encoding
        self.backend = backend
        self.variant = variant
    }
}

public struct AuKStreamAudioEvent: Equatable, Sendable {
    public var chunk: Int
    public var samples: Int
    public var seconds: Double
    public var elapsedSeconds: Double
    public var pcm: [Float]

    public init(chunk: Int, samples: Int, seconds: Double, elapsedSeconds: Double, pcm: [Float]) {
        self.chunk = chunk
        self.samples = samples
        self.seconds = seconds
        self.elapsedSeconds = elapsedSeconds
        self.pcm = pcm
    }
}

public struct AuKStreamDoneEvent: Equatable, Sendable {
    public var firstAudioSeconds: Double
    public var audioDurationSeconds: Double
    public var wallSeconds: Double
    public var chunks: Int

    public init(firstAudioSeconds: Double, audioDurationSeconds: Double, wallSeconds: Double, chunks: Int) {
        self.firstAudioSeconds = firstAudioSeconds
        self.audioDurationSeconds = audioDurationSeconds
        self.wallSeconds = wallSeconds
        self.chunks = chunks
    }
}

public enum AuKStreamEvent: Equatable, Sendable {
    case metadata(AuKStreamMetadataEvent)
    case audio(AuKStreamAudioEvent)
    case done(AuKStreamDoneEvent)
    case error(String)
}

public enum AuKSpeechPlanner {
    public static let defaultPauseSeconds = 0.2
    public static let defaultChunkTargetCharacters = 220
    public static let defaultChunkHardCharacterLimit = 320
    public static let minimumChunkSeconds = 0.25

    public static func plan(_ request: AuKSpeechPlanRequest) -> AuKSpeechPlanResult {
        let pause = request.interChunkPauseSeconds ?? defaultPauseSeconds
        if !pause.isFinite || pause < 0 {
            return .rejected(
                AuKChunkingRejection(
                    code: "invalid_chunk_pause",
                    message: "inter-chunk pause must be a finite duration >= 0",
                    options: ["inter_chunk_pause_seconds": "0.2"]
                )
            )
        }

        let resolved = aukResolveInstruction(
            instruction: request.instruction,
            instructions: request.instructions,
            input: request.input
        )
        let autoChunkDisabled = request.autoChunk == false

        switch classify(request) {
        case .tts(let target, let wrap):
            let speak = target.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !speak.isEmpty else {
                return .rejected(missingInstructionRejection())
            }
            let texts: [String]
            if autoChunkDisabled {
                texts = [speak]
            } else {
                let targetChars = request.chunkTargetCharacters ?? defaultChunkTargetCharacters
                let hardLimit = request.chunkHardCharacterLimit ?? defaultChunkHardCharacterLimit
                let chunks = LongFormTTSChunker.chunk(
                    speak,
                    targetCharacters: targetChars,
                    hardCharacterLimit: hardLimit
                )
                texts = chunks.isEmpty ? [speak] : chunks
            }
            let seconds = aukDistributeGenSeconds(request.genSeconds, weights: texts.map(\.count))
            let prepared = zip(texts, seconds).map { text, duration in
                AuKPreparedChunk(instruction: wrap(text), genSeconds: duration)
            }
            return .plan(AuKSpeechPlan(kind: .tts, chunks: prepared, pauseSeconds: pause))

        case .single:
            if request.requestsAutoChunk {
                return .rejected(autoChunkUnsupportedRejection())
            }
            guard let resolved, !resolved.isEmpty else {
                return .rejected(missingInstructionRejection())
            }
            return .plan(
                AuKSpeechPlan(
                    kind: .singleTask,
                    chunks: [AuKPreparedChunk(instruction: resolved, genSeconds: request.genSeconds)],
                    pauseSeconds: pause
                )
            )
        }
    }

    private enum Classification {
        case tts(target: String, wrap: @Sendable (String) -> String)
        case single
    }

    private static func classify(_ request: AuKSpeechPlanRequest) -> Classification {
        if aukExplicitNonTTSTask(request.task, request.mode) {
            return .single
        }

        let instruction = trimmedNonempty(request.instruction)
        let instructions = trimmedNonempty(request.instructions)
        let input = trimmedNonempty(request.input)
        let style = instruction ?? instructions

        if aukExplicitTTSTask(request.task, request.mode) {
            if let instruction, let extracted = aukExtractTTSTemplate(instruction) {
                return .tts(target: extracted.text, wrap: { extracted.wrap($0) })
            }
            if let input {
                return .tts(target: input, wrap: wrapClosure(style: style, hasSourceAudio: request.hasSourceAudio))
            }
            if let instruction {
                if aukLooksLikeNonTTSInstruction(instruction) {
                    return .single
                }
                return .tts(target: instruction, wrap: { $0 })
            }
            return .single
        }

        if let instruction, let extracted = aukExtractTTSTemplate(instruction) {
            return .tts(target: extracted.text, wrap: { extracted.wrap($0) })
        }

        if let input {
            if let style, aukLooksLikeNonTTSInstruction(style) {
                return .single
            }
            if let style, aukExtractTTSTemplate(style) == nil {
                return .tts(target: input, wrap: wrapClosure(style: style, hasSourceAudio: request.hasSourceAudio))
            }
            if style == nil {
                if request.hasSourceAudio {
                    return .tts(target: input, wrap: wrapClosure(style: nil, hasSourceAudio: true))
                }
                return .tts(target: input, wrap: { $0 })
            }
        }

        return .single
    }

    private static func wrapClosure(style: String?, hasSourceAudio: Bool) -> @Sendable (String) -> String {
        { chunk in
            aukWrapTTSChunk(chunk, style: style, hasSourceAudio: hasSourceAudio)
        }
    }

    private static func missingInstructionRejection() -> AuKChunkingRejection {
        AuKChunkingRejection(
            code: "missing_instruction",
            message: "instruction/input is required",
            options: [:]
        )
    }

    private static func autoChunkUnsupportedRejection() -> AuKChunkingRejection {
        AuKChunkingRejection(
            code: "auto_chunk_unsupported",
            message: "Auto-chunk is only supported for TTS. Existing-audio edits, enhancement, and separation stay a single instruction plus source audio.",
            options: [
                "task": "tts",
                "input": "<text to speak>",
                "instruction": "<voice/style, or an official zero-shot/instruct TTS template>",
                "auto_chunk": "omit or false for edit/enhancement/separation",
                "hint": "Set task=tts, pass input as speak-text with a voice/style instruction, or use an official TTS template so only the target text is chunked.",
            ]
        )
    }
}

public enum AuKStreamSession {
    public static func run(
        sampleRate: Int,
        backend: String,
        variant: String,
        plan: AuKSpeechPlan,
        generate: (AuKPreparedChunk) throws -> [Float],
        now: () -> TimeInterval = { Date().timeIntervalSinceReferenceDate },
        emit: (AuKStreamEvent) -> Void
    ) throws {
        let start = now()
        emit(
            .metadata(
                AuKStreamMetadataEvent(
                    sampleRate: sampleRate,
                    backend: backend,
                    variant: variant
                )
            )
        )

        var firstAudio: TimeInterval?
        var totalSamples = 0
        var chunkCount = 0
        let silence = aukInterChunkSilence(sampleRate: sampleRate, pauseSeconds: plan.pauseSeconds)

        for (index, chunk) in plan.chunks.enumerated() {
            var samples = try generate(chunk)
            if index < plan.chunks.count - 1 {
                samples.append(contentsOf: silence)
            }
            guard !samples.isEmpty else { continue }
            chunkCount += 1
            if firstAudio == nil {
                firstAudio = now() - start
            }
            totalSamples += samples.count
            emit(
                .audio(
                    AuKStreamAudioEvent(
                        chunk: chunkCount,
                        samples: samples.count,
                        seconds: Double(samples.count) / Double(sampleRate),
                        elapsedSeconds: now() - start,
                        pcm: samples
                    )
                )
            )
        }

        emit(
            .done(
                AuKStreamDoneEvent(
                    firstAudioSeconds: firstAudio ?? -1,
                    audioDurationSeconds: Double(totalSamples) / Double(max(sampleRate, 1)),
                    wallSeconds: now() - start,
                    chunks: chunkCount
                )
            )
        )
    }
}

public func aukSpeechStreamJSON(_ event: AuKStreamEvent, audioBase64: String? = nil) -> [String: Any] {
    switch event {
    case .metadata(let metadata):
        return [
            "event": "metadata",
            "format": metadata.format,
            "sample_rate": metadata.sampleRate,
            "channels": metadata.channels,
            "encoding": metadata.encoding,
            "backend": metadata.backend,
            "variant": metadata.variant,
        ]
    case .audio(let audio):
        var object: [String: Any] = [
            "event": "audio",
            "chunk": audio.chunk,
            "samples": audio.samples,
            "seconds": audio.seconds,
            "elapsed_seconds": audio.elapsedSeconds,
        ]
        if let audioBase64 {
            object["audio"] = audioBase64
        }
        return object
    case .done(let done):
        return [
            "event": "done",
            "first_audio_seconds": done.firstAudioSeconds,
            "audio_duration_seconds": done.audioDurationSeconds,
            "wall_seconds": done.wallSeconds,
            "chunks": done.chunks,
        ]
    case .error(let message):
        return [
            "event": "error",
            "error": message,
        ]
    }
}

public func aukInterChunkSilence(sampleRate: Int, pauseSeconds: Double) -> [Float] {
    guard sampleRate > 0, pauseSeconds.isFinite, pauseSeconds > 0 else { return [] }
    let count = Int((pauseSeconds * Double(sampleRate)).rounded())
    guard count > 0 else { return [] }
    return [Float](repeating: 0, count: count)
}

public func aukConcatenateTTSChunks(_ chunks: [[Float]], sampleRate: Int, pauseSeconds: Double) -> [Float] {
    guard !chunks.isEmpty else { return [] }
    let silence = aukInterChunkSilence(sampleRate: sampleRate, pauseSeconds: pauseSeconds)
    var output: [Float] = []
    for (index, chunk) in chunks.enumerated() {
        output.append(contentsOf: chunk)
        if index < chunks.count - 1 {
            output.append(contentsOf: silence)
        }
    }
    return output
}

public func aukDistributeGenSeconds(_ total: Double?, weights: [Int]) -> [Double?] {
    guard !weights.isEmpty else { return [] }
    guard let total, total.isFinite, total > 0 else {
        return Array(repeating: total, count: weights.count)
    }
    if weights.count == 1 {
        return [total]
    }
    let sum = Double(weights.map { max(1, $0) }.reduce(0, +))
    return weights.map { weight in
        max(AuKSpeechPlanner.minimumChunkSeconds, total * Double(max(1, weight)) / sum)
    }
}

func trimmedNonempty(_ value: String?) -> String? {
    guard let value else { return nil }
    let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
    return trimmed.isEmpty ? nil : trimmed
}

func aukExplicitTTSTask(_ task: String?, _ mode: String?) -> Bool {
    aukNormalizedTaskValues(task, mode).contains("tts")
}

func aukExplicitNonTTSTask(_ task: String?, _ mode: String?) -> Bool {
    let blocked: Set<String> = [
        "edit", "enhance", "enhancement", "separate", "separation", "clone-edit",
    ]
    return aukNormalizedTaskValues(task, mode).contains(where: { blocked.contains($0) })
}

func aukNormalizedTaskValues(_ task: String?, _ mode: String?) -> [String] {
    [task, mode].compactMap { value in
        trimmedNonempty(value)?.lowercased()
    }
}

func aukWrapTTSChunk(_ chunk: String, style: String?, hasSourceAudio: Bool) -> String {
    if let style, let extracted = aukExtractTTSTemplate(style) {
        return extracted.wrap(chunk)
    }
    if hasSourceAudio {
        if let style, !style.isEmpty {
            return "\(style)\nSay the following with the same voice: \"\(chunk)\""
        }
        return "Say the following with the same voice: \"\(chunk)\""
    }
    if let style, !style.isEmpty {
        return "Generate speech based on the following description: \"\(style)\". The content to speak is: \"\(chunk)\"."
    }
    return chunk
}

public func aukExtractTTSTemplate(_ instruction: String) -> AuKExtractedSpeakText? {
    if let zeroShot = aukExtractAfterMarker("Say the following with the same voice:", in: instruction) {
        return zeroShot
    }
    let instructMarkers = [
        "The content to speak is:",
        "generate speech content",
        "生成语音内容",
    ]
    for marker in instructMarkers {
        if let extracted = aukExtractAfterMarker(marker, in: instruction) {
            return extracted
        }
    }
    return nil
}

func aukExtractAfterMarker(_ marker: String, in instruction: String) -> AuKExtractedSpeakText? {
    guard let markerRange = instruction.range(of: marker, options: .caseInsensitive) else {
        return nil
    }
    let afterMarker = instruction[markerRange.upperBound...]
    let leadingWhitespace = afterMarker.prefix { $0.isWhitespace }
    let body = afterMarker[leadingWhitespace.endIndex...]
    guard let first = body.first else { return nil }

    let pairs: [(Character, Character)] = [
        ("'", "'"),
        ("\"", "\""),
        ("“", "”"),
        ("‘", "’"),
        ("「", "」"),
    ]
    if let pair = pairs.first(where: { $0.0 == first }) {
        let afterOpen = body.dropFirst()
        guard let closeIndex = afterOpen.lastIndex(of: pair.1) else { return nil }
        let text = String(afterOpen[..<closeIndex])
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
        let afterClose = afterOpen[afterOpen.index(after: closeIndex)...]
        let prefix = String(instruction[..<markerRange.upperBound]) + String(leadingWhitespace) + String(pair.0)
        return AuKExtractedSpeakText(
            prefix: prefix,
            text: text,
            suffix: String(pair.1) + String(afterClose)
        )
    }

    let trimmed = String(body).trimmingCharacters(in: .whitespacesAndNewlines)
    let stripped = trimmed.trimmingCharacters(in: CharacterSet(charactersIn: ".。"))
    guard !stripped.isEmpty else { return nil }
    return AuKExtractedSpeakText(
        prefix: String(instruction[..<markerRange.upperBound]) + String(leadingWhitespace),
        text: stripped,
        suffix: ""
    )
}

public func aukLooksLikeNonTTSInstruction(_ instruction: String) -> Bool {
    if aukExtractTTSTemplate(instruction) != nil {
        return false
    }
    let text = instruction.trimmingCharacters(in: .whitespacesAndNewlines)
    let lower = text.lowercased()
    let english = [
        "replace '",
        "replace \"",
        "change \"",
        "change '",
        "raise the pitch",
        "lower the pitch",
        "adjust the speech speed",
        "increase the volume",
        "decrease the volume",
        "change the emotion",
        "keep the spoken content unchanged",
        "change the timbre",
        "remove the regional accent",
        "remove all ",
        "convert this speech into a soft whisper",
        "convert this whispered",
        "remove only the background noise",
        "remove only the room reverberation",
        "remove noise and reverberation",
        "repair the ",
        "keep only the",
        "keep the clean singing",
        "keep only the singing",
        "keep all human voices",
        "remove all other speakers",
        "drop all other audio",
        "output audio of the same length",
        "output clean speech of the same length",
    ]
    if english.contains(where: { lower.contains($0) }) {
        return true
    }
    let localized = [
        "把‘", "改成", "将音调", "将语速", "将音量", "将情感",
        "请只去除", "请保留所有说话人", "只保留第", "请只保留歌声",
        "请只保留说", "请修复这段音频", "去掉其余说话人", "其余声音都去掉",
    ]
    return localized.contains(where: { text.contains($0) })
}
