// asr-stream-test — Replay a WAV file through StreamingSession,
// compare against a batch baseline, and emit quality metrics.
//
// Usage: asr-stream-test <wav-file> [model-dir] [options]

import Foundation
import MLX
import NativeASR

private let yuwpDefaultsDomain = "com.yuwp.app"
private let requiredModelFiles = ["config.json", "model.safetensors", "vocab.json", "merges.txt"]
private let speechRMS: Float = 0.020
private let vadSpeechThreshold: Float = 0.5

private struct ToolConfig: Codable {
    let wavFile: String
    let modelDir: String
    let chunkSec: Double
    let decodeMode: String
    let warmup: Bool
    let batchBaselineEnabled: Bool
    let batchRetranscribeEnabled: Bool
    let finalizationPass: String
}

private struct TimingBreakdown: Codable {
    let encodeMs: Double
    let prefillMs: Double
    let decodeMs: Double
    let totalMs: Double
    let reusePct: Double
}

private struct ChunkTrace: Codable {
    let index: Int
    let audioStartSec: Double
    let audioEndSec: Double
    let durationSec: Double
    let rms: Double
    let speechActive: Bool
    let speechDurationSec: Double
    let textChanged: Bool
    let charDelta: Int
    let wordDelta: Int
    let batchCorrected: Bool
    let committedText: String
    let activeText: String
    let text: String
    let timing: TimingBreakdown
}

private struct MemoryReport: Codable {
    let activeMB: Int
    let peakMB: Int
    let deltaMB: Int
}

private struct SpeechActivity {
    let hasSpeech: Bool
    let speechDurationSec: Double
}

private struct StreamingReport: Codable {
    let elapsedSec: Double
    let finalizationSec: Double
    let chunkCount: Int
    let segmentCommitCount: Int
    let firstSpeechChunk: Int?
    let firstSpeechAudioSec: Double?
    let firstTextChunk: Int?
    let firstTextAudioSec: Double?
    let speechToFirstTextSec: Double?
    let speechChunksWithoutGrowth: Int
    let speechSecondsWithoutGrowth: Double
    let maxConsecutiveSpeechChunksWithoutGrowth: Int
    let maxConsecutiveSpeechNoGrowthSec: Double
    let recoveredStallEvents: Int
    let recoveredSpeechNoGrowthChunks: Int
    let recoveredSpeechNoGrowthSec: Double
    let maxRecoveredSpeechNoGrowthSec: Double
    let finalizationAddedChars: Int
    let finalizationAddedWords: Int
    let preFinalizeText: String
    let finalText: String
    let memory: MemoryReport
    let chunks: [ChunkTrace]
}

private struct BatchReport: Codable {
    let elapsedSec: Double
    let audioDurationSec: Double
    let text: String
}

private struct EditCounts: Codable {
    let substitutions: Int
    let deletions: Int
    let insertions: Int
}

private struct AccuracyMetrics: Codable {
    let exactMatch: Bool
    let normalizedExactMatch: Bool
    let wordErrorRate: Double
    let charErrorRate: Double
    let wordEdits: EditCounts
    let charEdits: EditCounts
    let batchWordCount: Int
    let streamWordCount: Int
    let batchCharCount: Int
    let streamCharCount: Int
}

private struct SpeechFrame: Codable {
    let startSec: Double
    let endSec: Double
    let rms: Double
    let probability: Float
}

private struct EvalReport: Codable {
    let speechFrames: [SpeechFrame]?
    let config: ToolConfig
    let streaming: StreamingReport
    let batch: BatchReport?
    let accuracy: AccuracyMetrics?
}

private struct EditSummary {
    let substitutions: Int
    let deletions: Int
    let insertions: Int

    var total: Int { substitutions + deletions + insertions }
}

private struct ParsedArgs {
    let wavPath: String
    let modelSpec: String?
    let chunkSec: Double
    let chunkSecExplicit: Bool
    let streamMode: StreamDecodeMode?
    let language: String?
    let warmup: Bool
    let doBatch: Bool
    let doBatchRetranscribe: Bool
    let finalizationPass: FinalizationPass
    let emitJSON: Bool
    let compactJSON: Bool
    let jsonOutputPath: String?
    let speechTrace: Bool
}

private enum EditOp: UInt8 {
    case match = 0
    case substitute = 1
    case delete = 2
    case insert = 3
}

private func fmt(_ value: Double, _ pattern: String) -> String {
    String(format: pattern, value)
}

private func printUsage() {
    fputs(
        """
        Usage: asr-stream-test <wav-file> [model-dir] [options]

        Replays a WAV file through the streaming pipeline, then compares the
        final streaming result against a batch baseline and reports quality metrics.

        Positional args:
          <wav-file>              WAV file to replay
          [model-dir]             Model directory or repo id. Optional if Yuwp has a saved model.

        Options:
          --model <spec>          Model directory or repo id (overrides positional model-dir)
          --chunk-sec <sec>       Chunk duration in seconds (default: 2.25; 0.16 with --stream-mode stable-prefix)
          --stream-mode <mode>    rollback-batch or stable-prefix (default: auto from model path)
          --language <lang>       Language hint (e.g. English, Chinese)
          --warmup                Warm up Metal shaders before streaming
          --speech-trace          Include independent 36ms VAD/RMS frame annotations
          --no-batch              Skip batch baseline comparison
          --no-batch-retranscribe Disable the streaming segment batch-correction pass
          --full-session-retranscribe Opt into full-session batch on stop
          --json                  Emit pretty JSON to stdout
          --compact               Emit compact JSON to stdout
          --json-output <path>    Write JSON report to a file

        Examples:
          asr-stream-test sample.wav
          asr-stream-test sample.wav ~/models/Qwen3-ASR-1.7B-bf16 --warmup
          asr-stream-test sample.wav --json-output /tmp/stream-report.json

        Notes:
          - If model-dir is omitted, the tool tries Yuwp's saved transcription model.
          - Batch is used as a practical baseline, not human ground truth.

        """,
        stderr
    )
}

private func parseArgs() -> ParsedArgs {
    var args = Array(CommandLine.arguments.dropFirst())
    guard !args.isEmpty else {
        printUsage()
        exit(1)
    }

    var positional: [String] = []
    var explicitModel: String?
    var chunkSec = 2.25
    var chunkSecExplicit = false
    var streamMode: StreamDecodeMode?
    var language: String?
    var warmup = false
    var doBatch = true
    var doBatchRetranscribe = true
    var finalizationPass: FinalizationPass = .activeSegmentOnly
    var emitJSON = false
    var compactJSON = false
    var jsonOutputPath: String?
    var speechTrace = false

    while !args.isEmpty {
        let arg = args.removeFirst()
        switch arg {
        case "--model":
            guard !args.isEmpty else {
                fputs("--model requires a value\n", stderr)
                exit(1)
            }
            explicitModel = args.removeFirst()
        case "--chunk-sec":
            guard !args.isEmpty, let value = Double(args.removeFirst()) else {
                fputs("--chunk-sec requires a number\n", stderr)
                exit(1)
            }
            chunkSec = value
            chunkSecExplicit = true
        case "--stream-mode":
            guard !args.isEmpty else {
                fputs("--stream-mode requires rollback-batch or stable-prefix\n", stderr)
                exit(1)
            }
            switch args.removeFirst() {
            case "rollback-batch":
                streamMode = .rollbackBatch
            case "stable-prefix":
                streamMode = .stablePrefix
            default:
                fputs("--stream-mode must be rollback-batch or stable-prefix\n", stderr)
                exit(1)
            }
        case "--language":
            guard !args.isEmpty else {
                fputs("--language requires a value\n", stderr)
                exit(1)
            }
            language = args.removeFirst()
        case "--warmup":
            warmup = true
        case "--speech-trace":
            speechTrace = true
        case "--no-batch":
            doBatch = false
        case "--no-batch-retranscribe":
            doBatchRetranscribe = false
        case "--full-session-retranscribe":
            finalizationPass = .fullSessionRetranscribe
        case "--json":
            emitJSON = true
        case "--compact":
            compactJSON = true
        case "--json-output":
            guard !args.isEmpty else {
                fputs("--json-output requires a path\n", stderr)
                exit(1)
            }
            jsonOutputPath = args.removeFirst()
        case "-h", "--help":
            printUsage()
            exit(0)
        default:
            if arg.hasPrefix("--") {
                fputs("Unknown option: \(arg)\n", stderr)
                printUsage()
                exit(1)
            }
            positional.append(arg)
        }
    }

    guard !positional.isEmpty else {
        printUsage()
        exit(1)
    }
    guard !(emitJSON && compactJSON) else {
        fputs("Choose only one of --json or --compact\n", stderr)
        exit(1)
    }

    let wavPath = positional[0]
    let modelSpec = explicitModel ?? (positional.count >= 2 ? positional[1] : nil)
    if streamMode == .stablePrefix && !chunkSecExplicit {
        chunkSec = 0.16
    }
    return ParsedArgs(
        wavPath: wavPath,
        modelSpec: modelSpec,
        chunkSec: chunkSec,
        chunkSecExplicit: chunkSecExplicit,
        streamMode: streamMode,
        language: language,
        warmup: warmup,
        doBatch: doBatch,
        doBatchRetranscribe: doBatchRetranscribe,
        finalizationPass: finalizationPass,
        emitJSON: emitJSON,
        compactJSON: compactJSON,
        jsonOutputPath: jsonOutputPath,
        speechTrace: speechTrace
    )
}

private func isValidModelDirectory(_ url: URL) -> Bool {
    requiredModelFiles.allSatisfy { FileManager.default.fileExists(atPath: url.appendingPathComponent($0).path) }
}

private func looksLikePath(_ spec: String) -> Bool {
    spec.hasPrefix("/") || spec.hasPrefix("~") || spec.hasPrefix(".")
}

private func isRepoId(_ spec: String) -> Bool {
    let parts = spec.split(separator: "/")
    return parts.count == 2 && !parts[0].isEmpty && !parts[1].isEmpty
}

private func managedModelDirectory(for repoId: String) -> URL {
    FileManager.default.homeDirectoryForCurrentUser
        .appendingPathComponent("Library/Application Support/Yuwp/models", isDirectory: true)
        .appendingPathComponent(repoId.replacingOccurrences(of: "/", with: "--"), isDirectory: true)
}

private func huggingFaceSnapshot(for repoId: String) -> URL? {
    guard isRepoId(repoId) else { return nil }
    let parts = repoId.split(separator: "/", maxSplits: 1).map(String.init)
    guard parts.count == 2 else { return nil }

    let roots = [
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/huggingface/hub", isDirectory: true),
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Caches/huggingface/hub", isDirectory: true),
    ]

    for root in roots {
        let snapshotsDir = root
            .appendingPathComponent("models--\(parts[0])--\(parts[1])", isDirectory: true)
            .appendingPathComponent("snapshots", isDirectory: true)

        guard let snapshots = try? FileManager.default.contentsOfDirectory(
            at: snapshotsDir,
            includingPropertiesForKeys: [.contentModificationDateKey],
            options: [.skipsHiddenFiles]
        ) else { continue }

        let sorted = snapshots.sorted {
            let left = (try? $0.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            let right = (try? $1.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            return left > right
        }

        if let match = sorted.first(where: isValidModelDirectory(_:)) {
            return match
        }
    }

    return nil
}

private func resolveModelSpec(_ spec: String?) -> URL? {
    guard let spec else { return nil }
    let trimmed = spec.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !trimmed.isEmpty else { return nil }

    if looksLikePath(trimmed) {
        let expanded = NSString(string: trimmed).expandingTildeInPath
        let url = URL(fileURLWithPath: expanded).standardizedFileURL
        return isValidModelDirectory(url) ? url : nil
    }

    if isRepoId(trimmed) {
        let managed = managedModelDirectory(for: trimmed)
        if isValidModelDirectory(managed) { return managed }
        if let cached = huggingFaceSnapshot(for: trimmed) { return cached }
    }

    return nil
}

private func defaultYuwpModelSpec() -> String? {
    guard let domain = UserDefaults.standard.persistentDomain(forName: yuwpDefaultsDomain) else { return nil }
    let candidates = [
        domain["transcriptionModel"] as? String,
        domain["streamingModel"] as? String,
        domain["batchModel"] as? String,
    ]
    return candidates.compactMap { $0?.trimmingCharacters(in: .whitespacesAndNewlines) }.first(where: { !$0.isEmpty })
}

private func resolveModelURL(explicit spec: String?) -> URL? {
    if let resolved = resolveModelSpec(spec) { return resolved }
    return resolveModelSpec(defaultYuwpModelSpec())
}

private func normalize(_ text: String) -> String {
    var t = text.lowercased()
    t = t.unicodeScalars.map { scalar -> String in
        let v = scalar.value
        if v < 128 {
            let c = Character(scalar)
            if c.isPunctuation || "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~".contains(c) {
                return ""
            }
        }
        return String(scalar)
    }.joined()
    t = t.split(whereSeparator: \.isWhitespace).joined(separator: " ")
    return t.trimmingCharacters(in: .whitespacesAndNewlines)
}

private func normalizedWords(_ text: String) -> [String] {
    let normalized = normalize(text)
    guard !normalized.isEmpty else { return [] }
    return normalized.split(separator: " ").map(String.init)
}

private func normalizedChars(_ text: String) -> [Character] {
    Array(normalize(text).replacingOccurrences(of: " ", with: ""))
}

private func editSummary<T: Equatable>(reference: [T], hypothesis: [T]) -> EditSummary {
    let m = reference.count
    let n = hypothesis.count

    var cost = Array(repeating: Array(repeating: 0, count: n + 1), count: m + 1)
    var back = Array(repeating: Array(repeating: EditOp.match.rawValue, count: n + 1), count: m + 1)

    if m > 0 {
        for i in 1...m {
            cost[i][0] = i
            back[i][0] = EditOp.delete.rawValue
        }
    }
    if n > 0 {
        for j in 1...n {
            cost[0][j] = j
            back[0][j] = EditOp.insert.rawValue
        }
    }

    if m > 0 && n > 0 {
        for i in 1...m {
            for j in 1...n {
                if reference[i - 1] == hypothesis[j - 1] {
                    cost[i][j] = cost[i - 1][j - 1]
                    back[i][j] = EditOp.match.rawValue
                    continue
                }

                let sub = cost[i - 1][j - 1] + 1
                let del = cost[i - 1][j] + 1
                let ins = cost[i][j - 1] + 1
                let best = min(sub, del, ins)
                cost[i][j] = best

                if best == sub {
                    back[i][j] = EditOp.substitute.rawValue
                } else if best == del {
                    back[i][j] = EditOp.delete.rawValue
                } else {
                    back[i][j] = EditOp.insert.rawValue
                }
            }
        }
    }

    var i = m
    var j = n
    var substitutions = 0
    var deletions = 0
    var insertions = 0

    while i > 0 || j > 0 {
        let op = EditOp(rawValue: back[i][j]) ?? .match
        switch op {
        case .match:
            i -= 1
            j -= 1
        case .substitute:
            substitutions += 1
            i -= 1
            j -= 1
        case .delete:
            deletions += 1
            i -= 1
        case .insert:
            insertions += 1
            j -= 1
        }
    }

    return EditSummary(substitutions: substitutions, deletions: deletions, insertions: insertions)
}

private func computeRMS(_ audio: [Float]) -> Double {
    guard !audio.isEmpty else { return 0 }
    var sum: Double = 0
    for sample in audio {
        let value = Double(sample)
        sum += value * value
    }
    return sqrt(sum / Double(audio.count))
}

private func analyzeSpeechActivity(_ audio: [Float], vad: SileroVAD?) -> SpeechActivity {
    guard let vad else {
        let hasSpeech = computeRMS(audio) >= Double(speechRMS)
        let duration = hasSpeech ? Double(audio.count) / Double(ASRAudio.sampleRate) : 0
        return SpeechActivity(hasSpeech: hasSpeech, speechDurationSec: duration)
    }

    let frameSize = SileroVAD.chunkSize
    guard !audio.isEmpty else {
        return SpeechActivity(hasSpeech: false, speechDurationSec: 0)
    }

    var speechFrames = 0
    var offset = 0
    while offset < audio.count {
        let end = min(offset + frameSize, audio.count)
        var frame = Array(audio[offset ..< end])
        if frame.count < frameSize {
            frame.append(contentsOf: repeatElement(0, count: frameSize - frame.count))
        }
        if let probability = try? vad.process(frame), probability >= vadSpeechThreshold {
            speechFrames += 1
        }
        offset = end
    }

    let speechDurationSec = Double(speechFrames * frameSize) / Double(ASRAudio.sampleRate)
    return SpeechActivity(hasSpeech: speechFrames > 0, speechDurationSec: speechDurationSec)
}

// This annotation pass is independent of inference chunk boundaries. The replay
// harness freezes it at baseline so a candidate cannot improve its clock by
// changing what counts as speech. Do not use chunk-start as speech onset.
private func speechFrames(_ audio: [Float], vad: SileroVAD?) throws -> [SpeechFrame] {
    guard let vad else {
        throw NSError(domain: "stream-test", code: 1, userInfo: [
            NSLocalizedDescriptionKey: "--speech-trace requires Silero VAD"
        ])
    }
    vad.reset()
    defer { vad.reset() }
    var frames: [SpeechFrame] = []
    for start in stride(from: 0, to: audio.count, by: SileroVAD.chunkSize) {
        let end = min(start + SileroVAD.chunkSize, audio.count)
        var frame = Array(audio[start..<end])
        let rms = computeRMS(frame)
        frame.append(contentsOf: repeatElement(0, count: SileroVAD.chunkSize - frame.count))
        frames.append(SpeechFrame(
            startSec: Double(start) / Double(ASRAudio.sampleRate),
            endSec: Double(end) / Double(ASRAudio.sampleRate),
            rms: rms,
            probability: try vad.process(frame)
        ))
    }
    return frames
}

private func writeJSON<T: Encodable>(_ value: T, to path: String, compact: Bool) throws {
    let encoder = JSONEncoder()
    if compact {
        encoder.outputFormatting = []
    } else {
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    }
    let data = try encoder.encode(value)
    let url = URL(fileURLWithPath: path)
    try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
    try data.write(to: url)
}

private func runMain() throws {
    let parsed = parseArgs()
    let wavURL = URL(fileURLWithPath: NSString(string: parsed.wavPath).expandingTildeInPath)

    guard FileManager.default.fileExists(atPath: wavURL.path) else {
        fputs("Error: WAV file not found: \(wavURL.path)\n", stderr)
        exit(1)
    }

    guard let modelURL = resolveModelURL(explicit: parsed.modelSpec) else {
        fputs("Error: could not resolve a model directory. Pass [model-dir] or --model, or set Yuwp's transcription model first.\n", stderr)
        exit(1)
    }

    fputs("[stream-test] WAV:   \(wavURL.path)\n", stderr)
    fputs("[stream-test] Model: \(modelURL.path)\n", stderr)

    let transcriber = try Qwen3ASRTranscriber.load(from: modelURL)
    let vad = try? SileroVAD()
    if parsed.warmup {
        try transcriber.warmup()
    }

    let audio = try loadAudioFile(wavURL)
    let annotations = parsed.speechTrace ? try speechFrames(audio, vad: vad) : nil
    let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
    let audioDurationText = fmt(audioDuration, "%.2f")
    fputs("[stream-test] Audio: \(audioDurationText)s, \(audio.count) samples\n", stderr)

    let config: StreamConfig
    switch parsed.streamMode {
    case .stablePrefix:
        config = .stablePrefix(
            chunkSec: parsed.chunkSec,
            finalizationPass: parsed.finalizationPass
        )
    case .rollbackBatch:
        config = StreamConfig(
            chunkSec: parsed.chunkSec,
            batchRetranscribe: parsed.doBatchRetranscribe,
            finalizationPass: parsed.finalizationPass
        )
    case nil:
        if StreamConfig.isR2T2Model(at: modelURL) {
            config = .stablePrefix(
                chunkSec: parsed.chunkSecExplicit ? parsed.chunkSec : 0.16,
                finalizationPass: parsed.finalizationPass
            )
        } else {
            config = StreamConfig(
                chunkSec: parsed.chunkSec,
                batchRetranscribe: parsed.doBatchRetranscribe,
                finalizationPass: parsed.finalizationPass
            )
        }
    }
    let session = StreamingSession(
        transcriber: transcriber, config: config, language: parsed.language
    )
    let memBefore = MLX.Memory.activeMemory
    let steadyChunkSize = Int(config.chunkSec * Double(ASRAudio.sampleRate))
    let bootstrapChunkSec = min(config.chunkSec, 1.5)
    let bootstrapChunkSize = Int(bootstrapChunkSec * Double(ASRAudio.sampleRate))
    let chunkSecText = fmt(parsed.chunkSec, "%.2f")
    let bootstrapChunkSecText = fmt(bootstrapChunkSec, "%.2f")

    fputs("\n[stream-test] === Streaming (chunk=\(chunkSecText)s, bootstrap=\(bootstrapChunkSecText)s) ===\n", stderr)

    let streamT0 = Date()
    var offset = 0
    var chunkNum = 0
    var previousText = ""
    var chunkTraces: [ChunkTrace] = []
    var segmentCommitCount = 0
    var firstSpeechChunk: Int? = nil
    var firstSpeechAudioSec: Double? = nil
    var firstTextChunk: Int? = nil
    var firstTextAudioSec: Double? = nil
    var speechChunksWithoutGrowth = 0
    var speechSecondsWithoutGrowth = 0.0
    var currentSpeechNoGrowthRun = 0
    var currentSpeechNoGrowthSec = 0.0
    var pendingSpeechNoGrowthChunks = 0
    var pendingSpeechNoGrowthSec = 0.0
    var maxSpeechNoGrowthRun = 0
    var maxSpeechNoGrowthSec = 0.0
    var recoveredStallEvents = 0
    var recoveredSpeechNoGrowthChunks = 0
    var recoveredSpeechNoGrowthSec = 0.0
    var maxRecoveredSpeechNoGrowthSec = 0.0

    while offset < audio.count {
        let currentChunkSize = firstTextChunk == nil ? bootstrapChunkSize : steadyChunkSize
        let end = min(offset + currentChunkSize, audio.count)
        let chunk = Array(audio[offset ..< end])
        let speech = analyzeSpeechActivity(chunk, vad: vad)
        let result = session.processChunk(
            chunk,
            speechHint: SpeechActivityHint(hasSpeech: speech.hasSpeech, speechDurationSec: speech.speechDurationSec)
        )
        chunkNum += 1

        let audioStartSec = Double(offset) / Double(ASRAudio.sampleRate)
        let audioEndSec = Double(end) / Double(ASRAudio.sampleRate)
        let chunkDur = audioEndSec - audioStartSec
        let rms = computeRMS(chunk)
        let textChanged = result.text != previousText
        let charDelta = result.text.count - previousText.count
        let wordDelta = normalizedWords(result.text).count - normalizedWords(previousText).count
        let committedText = session.committedSegmentText()
        let activeText = session.activeSegmentText()

        if result.batchCorrected { segmentCommitCount += 1 }
        if firstSpeechChunk == nil && speech.hasSpeech {
            firstSpeechChunk = chunkNum
            firstSpeechAudioSec = audioStartSec
        }
        if firstTextChunk == nil && !result.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            firstTextChunk = chunkNum
            firstTextAudioSec = audioEndSec
        }

        if speech.hasSpeech && !textChanged {
            speechChunksWithoutGrowth += 1
            speechSecondsWithoutGrowth += speech.speechDurationSec
            currentSpeechNoGrowthRun += 1
            currentSpeechNoGrowthSec += speech.speechDurationSec
            maxSpeechNoGrowthRun = max(maxSpeechNoGrowthRun, currentSpeechNoGrowthRun)
            maxSpeechNoGrowthSec = max(maxSpeechNoGrowthSec, currentSpeechNoGrowthSec)
        } else if textChanged {
            let recoveredChunks = pendingSpeechNoGrowthChunks + currentSpeechNoGrowthRun
            let recoveredSec = pendingSpeechNoGrowthSec + currentSpeechNoGrowthSec
            if recoveredChunks > 0 && (speech.hasSpeech || !result.batchCorrected) {
                recoveredStallEvents += 1
                recoveredSpeechNoGrowthChunks += recoveredChunks
                recoveredSpeechNoGrowthSec += recoveredSec
                maxRecoveredSpeechNoGrowthSec = max(maxRecoveredSpeechNoGrowthSec, recoveredSec)
            }
            pendingSpeechNoGrowthChunks = 0
            pendingSpeechNoGrowthSec = 0
            currentSpeechNoGrowthRun = 0
            currentSpeechNoGrowthSec = 0
        } else {
            if currentSpeechNoGrowthRun > 0 {
                pendingSpeechNoGrowthChunks += currentSpeechNoGrowthRun
                pendingSpeechNoGrowthSec += currentSpeechNoGrowthSec
                currentSpeechNoGrowthRun = 0
                currentSpeechNoGrowthSec = 0
            }
            if result.batchCorrected {
                pendingSpeechNoGrowthChunks = 0
                pendingSpeechNoGrowthSec = 0
            }
        }

        let tag = result.batchCorrected ? " [BATCH]" : ""
        let timing = result.totalMs > 0
            ? String(
                format: "%.0fms (enc=%.0f pfx=%.0f dec=%.0f reuse=%.0f%%)",
                result.totalMs,
                result.encodeMs,
                result.prefillMs,
                result.decodeMs,
                result.reusePct
            )
            : String(format: "%.0fms", result.totalMs)

        fputs(String(format: "  chunk %2d (%.1fs): %@%@ rms=%.4f\n", chunkNum, chunkDur, timing, tag, rms), stderr)
        let preview = result.text.count > 100 ? String(result.text.prefix(97)) + "..." : result.text
        if !preview.isEmpty {
            fputs("    → \"\(preview)\"\n", stderr)
        }

        chunkTraces.append(
            ChunkTrace(
                index: chunkNum,
                audioStartSec: audioStartSec,
                audioEndSec: audioEndSec,
                durationSec: chunkDur,
                rms: rms,
                speechActive: speech.hasSpeech,
                speechDurationSec: speech.speechDurationSec,
                textChanged: textChanged,
                charDelta: charDelta,
                wordDelta: wordDelta,
                batchCorrected: result.batchCorrected,
                committedText: committedText,
                activeText: activeText,
                text: result.text,
                timing: TimingBreakdown(
                    encodeMs: result.encodeMs,
                    prefillMs: result.prefillMs,
                    decodeMs: result.decodeMs,
                    totalMs: result.totalMs,
                    reusePct: result.reusePct
                )
            )
        )

        previousText = result.text
        offset = end
    }

    let preFinalizeText = session.finalText()
    let finalizeT0 = Date()
    let finalText = session.finalize()
    let finalizationSec = Date().timeIntervalSince(finalizeT0)
    let streamTime = Date().timeIntervalSince(streamT0)
    let streamTimeText = fmt(streamTime, "%.2f")
    let memAfter = MLX.Memory.activeMemory
    let memPeak = MLX.Memory.peakMemory
    let finalizationAddedChars = max(0, finalText.count - preFinalizeText.count)
    let finalizationAddedWords = max(0, normalizedWords(finalText).count - normalizedWords(preFinalizeText).count)

    let speechToFirstTextSec: Double? = {
        guard let firstSpeechAudioSec, let firstTextAudioSec else { return nil }
        return max(0, firstTextAudioSec - firstSpeechAudioSec)
    }()

    let streaming = StreamingReport(
        elapsedSec: streamTime,
        finalizationSec: finalizationSec,
        chunkCount: chunkNum,
        segmentCommitCount: segmentCommitCount,
        firstSpeechChunk: firstSpeechChunk,
        firstSpeechAudioSec: firstSpeechAudioSec,
        firstTextChunk: firstTextChunk,
        firstTextAudioSec: firstTextAudioSec,
        speechToFirstTextSec: speechToFirstTextSec,
        speechChunksWithoutGrowth: speechChunksWithoutGrowth,
        speechSecondsWithoutGrowth: speechSecondsWithoutGrowth,
        maxConsecutiveSpeechChunksWithoutGrowth: maxSpeechNoGrowthRun,
        maxConsecutiveSpeechNoGrowthSec: maxSpeechNoGrowthSec,
        recoveredStallEvents: recoveredStallEvents,
        recoveredSpeechNoGrowthChunks: recoveredSpeechNoGrowthChunks,
        recoveredSpeechNoGrowthSec: recoveredSpeechNoGrowthSec,
        maxRecoveredSpeechNoGrowthSec: maxRecoveredSpeechNoGrowthSec,
        finalizationAddedChars: finalizationAddedChars,
        finalizationAddedWords: finalizationAddedWords,
        preFinalizeText: preFinalizeText,
        finalText: finalText,
        memory: MemoryReport(
            activeMB: memAfter / 1_048_576,
            peakMB: memPeak / 1_048_576,
            deltaMB: (memAfter - memBefore) / 1_048_576
        ),
        chunks: chunkTraces
    )

    let finalizationText = fmt(finalizationSec, "%.2f")
    fputs("\n[stream-test] Streaming done in \(streamTimeText)s\n", stderr)
    fputs("[stream-test] Memory: active=\(streaming.memory.activeMB)MB peak=\(streaming.memory.peakMB)MB delta=\(streaming.memory.deltaMB)MB\n", stderr)
    fputs("[stream-test] Pre-finalize: \"\(preFinalizeText)\"\n", stderr)
    fputs("[stream-test] Final:        \"\(finalText)\"\n", stderr)

    var batch: BatchReport? = nil
    var accuracy: AccuracyMetrics? = nil

    if parsed.doBatch {
        fputs("\n[stream-test] === Batch baseline ===\n", stderr)
        let batchResult = try transcriber.transcribe(audio: audio)
        batch = BatchReport(
            elapsedSec: batchResult.processingTime,
            audioDurationSec: batchResult.audioDuration,
            text: batchResult.text
        )
        let batchTimeText = fmt(batchResult.processingTime, "%.2f")
        fputs("[stream-test] Batch: \"\(batchResult.text)\"\n", stderr)
        fputs("[stream-test] Batch time: \(batchTimeText)s\n", stderr)

        let batchWords = normalizedWords(batchResult.text)
        let streamWords = normalizedWords(finalText)
        let batchChars = normalizedChars(batchResult.text)
        let streamChars = normalizedChars(finalText)
        let wordEdits = editSummary(reference: batchWords, hypothesis: streamWords)
        let charEdits = editSummary(reference: batchChars, hypothesis: streamChars)
        let wordDenom = max(batchWords.count, 1)
        let charDenom = max(batchChars.count, 1)
        let normalizedExactMatch = normalize(batchResult.text) == normalize(finalText)

        accuracy = AccuracyMetrics(
            exactMatch: batchResult.text == finalText,
            normalizedExactMatch: normalizedExactMatch,
            wordErrorRate: batchWords.isEmpty && streamWords.isEmpty ? 0 : Double(wordEdits.total) / Double(wordDenom),
            charErrorRate: batchChars.isEmpty && streamChars.isEmpty ? 0 : Double(charEdits.total) / Double(charDenom),
            wordEdits: EditCounts(
                substitutions: wordEdits.substitutions,
                deletions: wordEdits.deletions,
                insertions: wordEdits.insertions
            ),
            charEdits: EditCounts(
                substitutions: charEdits.substitutions,
                deletions: charEdits.deletions,
                insertions: charEdits.insertions
            ),
            batchWordCount: batchWords.count,
            streamWordCount: streamWords.count,
            batchCharCount: batchChars.count,
            streamCharCount: streamChars.count
        )

        let werText = fmt(accuracy?.wordErrorRate ?? 0, "%.3f")
        let cerText = fmt(accuracy?.charErrorRate ?? 0, "%.3f")
        fputs("\n[stream-test] === Quality summary ===\n", stderr)
        fputs("[stream-test] Normalized exact match: \(normalizedExactMatch ? "yes" : "no")\n", stderr)
        fputs(
            "[stream-test] WER: \(werText) "
                + "(S=\(wordEdits.substitutions) D=\(wordEdits.deletions) I=\(wordEdits.insertions), ref=\(batchWords.count))\n",
            stderr
        )
        fputs(
            "[stream-test] CER: \(cerText) "
                + "(S=\(charEdits.substitutions) D=\(charEdits.deletions) I=\(charEdits.insertions), ref=\(batchChars.count))\n",
            stderr
        )
    }

    if let firstSpeechAudioSec {
        let speechStartText = fmt(firstSpeechAudioSec, "%.2f")
        fputs("[stream-test] First speech at: \(speechStartText)s\n", stderr)
    }
    if let speechToFirstTextSec {
        let lagText = fmt(speechToFirstTextSec, "%.2f")
        fputs("[stream-test] Speech → first text: \(lagText)s\n", stderr)
    }
    fputs("[stream-test] Segment commits: \(segmentCommitCount)\n", stderr)
    let noGrowthSecText = fmt(speechSecondsWithoutGrowth, "%.2f")
    let maxNoGrowthSecText = fmt(maxSpeechNoGrowthSec, "%.2f")
    let recoveredNoGrowthSecText = fmt(recoveredSpeechNoGrowthSec, "%.2f")
    let maxRecoveredNoGrowthSecText = fmt(maxRecoveredSpeechNoGrowthSec, "%.2f")
    fputs(
        "[stream-test] Speech without growth: \(speechChunksWithoutGrowth) chunks / \(noGrowthSecText)s "
            + "(max run \(maxSpeechNoGrowthRun) chunks / \(maxNoGrowthSecText)s)\n",
        stderr
    )
    fputs(
        "[stream-test] Recovered stalls: \(recoveredStallEvents) events, \(recoveredSpeechNoGrowthChunks) chunks / \(recoveredNoGrowthSecText)s "
            + "(max recovered \(maxRecoveredNoGrowthSecText)s)\n",
        stderr
    )
    fputs(
        "[stream-test] Finalization added: \(finalizationAddedChars) chars, \(finalizationAddedWords) words in \(finalizationText)s\n",
        stderr
    )

    let report = EvalReport(
        speechFrames: annotations,
        config: ToolConfig(
            wavFile: wavURL.path,
            modelDir: modelURL.path,
            chunkSec: config.chunkSec,
            decodeMode: config.decodeMode.rawValue,
            warmup: parsed.warmup,
            batchBaselineEnabled: parsed.doBatch,
            batchRetranscribeEnabled: parsed.doBatchRetranscribe,
            finalizationPass: parsed.finalizationPass.rawValue
        ),
        streaming: streaming,
        batch: batch,
        accuracy: accuracy
    )

    if let outputPath = parsed.jsonOutputPath {
        try writeJSON(report, to: outputPath, compact: false)
        fputs("[stream-test] JSON report written to \(outputPath)\n", stderr)
    }

    if parsed.emitJSON || parsed.compactJSON {
        let encoder = JSONEncoder()
        encoder.outputFormatting = parsed.compactJSON ? [] : [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(report)
        FileHandle.standardOutput.write(data)
        FileHandle.standardOutput.write(Data("\n".utf8))
    } else {
        print(finalText)
    }
}

do {
    try runMain()
    exit(0)
} catch {
    fputs("Error: \(error.localizedDescription)\n", stderr)
    exit(1)
}
