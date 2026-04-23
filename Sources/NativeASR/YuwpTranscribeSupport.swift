import Foundation

public enum YuwpTranscribeSupport {
    public static func runCLI(arguments: [String], programName: String = "yuwp-asr transcribe") -> Int32 {
        do {
            let config = try parseCLI(arguments: arguments)
            try run(config: config)
            return 0
        } catch CLIError.help {
            printUsage(programName: programName)
            return 0
        } catch CLIError.usage {
            printUsage(programName: programName)
            return 1
        } catch let error as CLIError {
            fputs("Error: \(error.localizedDescription)\n\n", stderr)
            printUsage(programName: programName)
            return 1
        } catch {
            fputs("Error: \(error.localizedDescription)\n", stderr)
            return 1
        }
    }
}

private enum OutputFormat: String {
    case text
    case json
    case srt
    case vtt
}

private struct CLIConfig {
    let audioPath: String
    let modelSpec: String?
    let format: OutputFormat
    let outputPath: String?
    let language: String?
    let debug: Bool
}

private struct JSONSegment: Codable {
    let start: Double
    let end: Double
    let text: String
}

private struct JSONOutput: Codable {
    let text: String
    let language: String?
    let duration: Double
    let processingTime: Double
    let rtf: Double
    let speedMultiplier: Double
    let segments: [JSONSegment]?
    let debug: BatchSubtitleDebug?
}

private enum CLIError: LocalizedError {
    case help
    case usage
    case missingModel
    case missingValue(String)
    case invalidFormat
    case unknownOption(String)
    case alignerRequired(format: String)
    case debugRequiresJSON

    var errorDescription: String? {
        switch self {
        case .help, .usage:
            return nil
        case .missingModel:
            return "Could not resolve a model directory. Pass --model or set Yuwp's transcription model first."
        case .missingValue(let flag):
            return "\(flag) requires a value"
        case .invalidFormat:
            return "--format must be one of: text, json, srt, vtt"
        case .unknownOption(let flag):
            return "Unknown option: \(flag)"
        case .alignerRequired(let format):
            return "\(format) output requires the Qwen3 forced aligner model to be installed locally."
        case .debugRequiresJSON:
            return "--debug currently requires --format json"
        }
    }
}

private func printUsage(programName: String) {
    fputs("""
    Usage: \(programName) <audio-file> [options]

    Options:
      --model <path>                Model path or repo id (optional; defaults to Yuwp app config)
      --format <text|json|srt|vtt>  Output format (default: text)
      --output <path>               Write output to a file instead of stdout
      --language <lang>             Language hint
      --debug                       Include chunk/alignment debug metadata in JSON output

    Examples:
      \(programName) sample.m4a
      \(programName) sample.m4a --model ~/models/Qwen3-ASR-0.6B-4bit
      \(programName) sample.m4a --format json
      \(programName) sample.m4a --format srt --output sample.srt

    """, stderr)
}

private func parseCLI(arguments: [String]) throws -> CLIConfig {
    var args = arguments
    if args.contains("--help") || args.contains("-h") {
        throw CLIError.help
    }
    guard !args.isEmpty else {
        throw CLIError.usage
    }

    let audioPath = args.removeFirst()
    var modelSpec: String?
    var format: OutputFormat = .text
    var outputPath: String?
    var language: String?
    var debug = false

    while !args.isEmpty {
        let flag = args.removeFirst()
        switch flag {
        case "--model":
            guard !args.isEmpty else { throw CLIError.missingValue(flag) }
            modelSpec = args.removeFirst()
        case "--format":
            guard !args.isEmpty else { throw CLIError.missingValue(flag) }
            guard let parsed = OutputFormat(rawValue: args.removeFirst().lowercased()) else {
                throw CLIError.invalidFormat
            }
            format = parsed
        case "--output", "-o":
            guard !args.isEmpty else { throw CLIError.missingValue(flag) }
            outputPath = args.removeFirst()
        case "--language":
            guard !args.isEmpty else { throw CLIError.missingValue(flag) }
            language = args.removeFirst()
        case "--debug":
            debug = true
        default:
            throw CLIError.unknownOption(flag)
        }
    }

    if debug, format != .json {
        throw CLIError.debugRequiresJSON
    }

    return CLIConfig(
        audioPath: audioPath,
        modelSpec: modelSpec,
        format: format,
        outputPath: outputPath,
        language: language,
        debug: debug
    )
}

private func writeOutput(_ text: String, to outputPath: String?) throws {
    if let outputPath {
        let url = URL(fileURLWithPath: outputPath)
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try text.write(to: url, atomically: true, encoding: .utf8)
    } else {
        FileHandle.standardOutput.write(Data(text.utf8))
        if !text.hasSuffix("\n") {
            FileHandle.standardOutput.write(Data("\n".utf8))
        }
    }
}

private func encodeJSON(_ payload: JSONOutput) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(payload)
    return String(decoding: data, as: UTF8.self)
}

private func makeJSONOutput(
    transcript: String,
    language: String?,
    duration: Double,
    processingTime: Double,
    subtitles: [Subtitle]? = nil,
    debug: BatchSubtitleDebug? = nil
) -> JSONOutput {
    JSONOutput(
        text: transcript,
        language: normalizeLanguageCode(language),
        duration: duration,
        processingTime: processingTime,
        rtf: processingTime / max(duration, 1e-6),
        speedMultiplier: duration / max(processingTime, 1e-6),
        segments: subtitles?.map { JSONSegment(start: $0.start, end: $0.end, text: $0.text) },
        debug: debug
    )
}

private func validatePathExists(_ path: String, label: String) throws -> URL {
    let url = URL(fileURLWithPath: path).standardizedFileURL
    guard FileManager.default.fileExists(atPath: url.path) else {
        throw NSError(domain: "yuwp-asr", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(label) not found: \(path)"])
    }
    return url
}

private func loadDefaultAligner() throws -> ForcedAligner? {
    guard let alignerURL = YuwpModelSupport.defaultAlignerURL() else { return nil }
    return try ForcedAligner.load(from: alignerURL)
}

private func loadDefaultVAD() -> SileroVAD? {
    try? SileroVAD()
}

private final class LocalBatchTranscriptionService: BatchTranscriptionServing, @unchecked Sendable {
    private let transcriber: Qwen3ASRTranscriber
    private let inferenceLock = NSLock()

    init(transcriber: Qwen3ASRTranscriber) {
        self.transcriber = transcriber
    }

    func transcribeChunk(audio: [Float], language: String?, temperature: Float) throws -> TranscriptionResult {
        inferenceLock.lock()
        defer { inferenceLock.unlock() }
        return try transcriber.transcribe(audio: audio, language: language, temperature: temperature)
    }

    func subtitleItems(
        audio: [Float],
        transcript: String?,
        language: String?,
        temperature: Float,
        aligner: ForcedAligner
    ) throws -> (transcript: String, language: String, items: [ForcedAlignItem]) {
        let trimmedTranscript = transcript?.trimmingCharacters(in: .whitespacesAndNewlines)

        inferenceLock.lock()
        defer { inferenceLock.unlock() }

        let resolvedTranscript: String
        let resolvedLanguage: String
        if let trimmedTranscript, !trimmedTranscript.isEmpty {
            resolvedTranscript = trimmedTranscript
            resolvedLanguage = language ?? "English"
        } else {
            let result = try transcriber.transcribe(audio: audio, language: language, temperature: temperature)
            resolvedTranscript = result.text.trimmingCharacters(in: .whitespacesAndNewlines)
            resolvedLanguage = language ?? result.language ?? "English"
        }

        guard !resolvedTranscript.isEmpty else {
            return (resolvedTranscript, resolvedLanguage, [])
        }

        let items = aligner.align(audio: audio, text: resolvedTranscript, language: resolvedLanguage)
        return (resolvedTranscript, resolvedLanguage, items)
    }
}

private func run(config: CLIConfig) throws {
    let audioURL = try validatePathExists(config.audioPath, label: "Audio file")
    guard let modelURL = YuwpModelSupport.resolveConfiguredModelURL(explicitSpec: config.modelSpec) else {
        throw CLIError.missingModel
    }

    let transcriber = try Qwen3ASRTranscriber.load(from: modelURL)
    let aligner = try loadDefaultAligner()
    let vad = loadDefaultVAD()
    let service = LocalBatchTranscriptionService(transcriber: transcriber)

    let audio = try loadAudioFile(audioURL)
    let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)

    switch config.format {
    case .text:
        let result = try BatchTranscriptionPipeline.transcribe(
            using: service,
            audio: audio,
            language: config.language,
            temperature: 0.0,
            vad: vad
        )
        try writeOutput(result.text, to: config.outputPath)
    case .json:
        if let aligner {
            let result = try BatchTranscriptionPipeline.subtitle(
                using: service,
                audio: audio,
                transcript: nil,
                language: config.language,
                temperature: 0.0,
                aligner: aligner,
                vad: vad
            )
            let subtitles = groupSubtitles(result.items, language: config.language ?? result.language)
            let payload = makeJSONOutput(
                transcript: result.transcript,
                language: config.language ?? result.language,
                duration: audioDuration,
                processingTime: result.processingTime,
                subtitles: subtitles,
                debug: config.debug ? result.debug : nil
            )
            try writeOutput(try encodeJSON(payload), to: config.outputPath)
        } else {
            let result = try BatchTranscriptionPipeline.transcribe(
                using: service,
                audio: audio,
                language: config.language,
                temperature: 0.0,
                vad: vad
            )
            let payload = makeJSONOutput(
                transcript: result.text,
                language: config.language ?? result.language,
                duration: audioDuration,
                processingTime: result.processingTime,
                subtitles: nil,
                debug: nil
            )
            try writeOutput(try encodeJSON(payload), to: config.outputPath)
        }
    case .srt, .vtt:
        guard let aligner else {
            throw CLIError.alignerRequired(format: config.format.rawValue)
        }
        let result = try BatchTranscriptionPipeline.subtitle(
            using: service,
            audio: audio,
            transcript: nil,
            language: config.language,
            temperature: 0.0,
            aligner: aligner,
            vad: vad
        )
        let subtitles = groupSubtitles(result.items, language: config.language ?? result.language)
        switch config.format {
        case .srt:
            try writeOutput(formatSRT(subtitles), to: config.outputPath)
        case .vtt:
            try writeOutput(formatVTT(subtitles), to: config.outputPath)
        case .text, .json:
            break
        }
    }
}
