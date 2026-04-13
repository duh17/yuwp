import Foundation
import Darwin
import NativeASR

private enum OutputFormat: String {
    case text
    case json
    case srt
    case vtt
}

private enum StrategyKind: String {
    case auto
    case word
    case compactScript = "compact-script"
}

private struct Config {
    let inputPath: String
    let outputPath: String?
    let format: OutputFormat
    let language: String?
    let strategy: StrategyKind
    let maxUnits: Int?
    let maxDuration: Double?
    let pauseThreshold: Double?
}

private enum CLIError: LocalizedError {
    case usage
    case missingValue(String)
    case invalidFormat(String)
    case invalidStrategy(String)
    case invalidInteger(flag: String, value: String)
    case invalidDouble(flag: String, value: String)
    case unknownOption(String)
    case debugMissing
    case fileNotFound(String)

    var errorDescription: String? {
        switch self {
        case .usage:
            return nil
        case .missingValue(let flag):
            return "\(flag) requires a value"
        case .invalidFormat(let value):
            return "unsupported format '\(value)' (expected: text, json, srt, vtt)"
        case .invalidStrategy(let value):
            return "unsupported strategy '\(value)' (expected: auto, word, compact-script)"
        case .invalidInteger(let flag, let value):
            return "\(flag) requires an integer value (got '\(value)')"
        case .invalidDouble(let flag, let value):
            return "\(flag) requires a numeric value (got '\(value)')"
        case .unknownOption(let flag):
            return "unknown option: \(flag)"
        case .debugMissing:
            return "input JSON does not contain a debug payload"
        case .fileNotFound(let path):
            return "file not found: \(path)"
        }
    }
}

private func printUsage() {
    fputs("""
    Usage: asr-stitch-debug <debug-json> [options]

    Options:
      --format <text|json|srt|vtt>    Output format (default: json)
      --language <lang>               Override language used for stitching
      --strategy <auto|word|compact-script>
                                      Override stitching strategy (default: auto)
      --max-units <n>                 Override max units per subtitle
      --max-duration <seconds>        Override max subtitle duration
      --pause-threshold <seconds>     Override pause threshold for flushes
      --output <path>                 Write output to a file instead of stdout

    Examples:
      swift run asr-stitch-debug /tmp/xCd9ykretlg.debug.json --format srt
      swift run asr-stitch-debug /tmp/zTc0CensUA8.debug.json --language Chinese \
        --strategy compact-script --max-units 12 --pause-threshold 0.6 --format srt
      swift run asr-stitch-debug /tmp/xCd9ykretlg.debug.json --format json | jq '.segments[:5]'
    """, stderr)
}

private func parseCLI(arguments: [String]) throws -> Config {
    var args = arguments
    if args.isEmpty || args.contains("--help") || args.contains("-h") {
        throw CLIError.usage
    }

    let inputPath = args.removeFirst()
    var outputPath: String?
    var format: OutputFormat = .json
    var language: String?
    var strategy: StrategyKind = .auto
    var maxUnits: Int?
    var maxDuration: Double?
    var pauseThreshold: Double?

    while !args.isEmpty {
        let flag = args.removeFirst()
        switch flag {
        case "--format":
            guard !args.isEmpty else { throw CLIError.missingValue(flag) }
            let raw = args.removeFirst().lowercased()
            guard let parsed = OutputFormat(rawValue: raw) else { throw CLIError.invalidFormat(raw) }
            format = parsed
        case "--language":
            guard !args.isEmpty else { throw CLIError.missingValue(flag) }
            language = args.removeFirst()
        case "--strategy":
            guard !args.isEmpty else { throw CLIError.missingValue(flag) }
            let raw = args.removeFirst().lowercased()
            guard let parsed = StrategyKind(rawValue: raw) else { throw CLIError.invalidStrategy(raw) }
            strategy = parsed
        case "--max-units":
            guard !args.isEmpty else { throw CLIError.missingValue(flag) }
            let raw = args.removeFirst()
            guard let parsed = Int(raw) else { throw CLIError.invalidInteger(flag: flag, value: raw) }
            maxUnits = parsed
        case "--max-duration":
            guard !args.isEmpty else { throw CLIError.missingValue(flag) }
            let raw = args.removeFirst()
            guard let parsed = Double(raw) else { throw CLIError.invalidDouble(flag: flag, value: raw) }
            maxDuration = parsed
        case "--pause-threshold":
            guard !args.isEmpty else { throw CLIError.missingValue(flag) }
            let raw = args.removeFirst()
            guard let parsed = Double(raw) else { throw CLIError.invalidDouble(flag: flag, value: raw) }
            pauseThreshold = parsed
        case "--output", "-o":
            guard !args.isEmpty else { throw CLIError.missingValue(flag) }
            outputPath = args.removeFirst()
        default:
            throw CLIError.unknownOption(flag)
        }
    }

    return Config(
        inputPath: inputPath,
        outputPath: outputPath,
        format: format,
        language: language,
        strategy: strategy,
        maxUnits: maxUnits,
        maxDuration: maxDuration,
        pauseThreshold: pauseThreshold
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

private func prettyJSONString(_ data: Data) throws -> String {
    let object = try JSONSerialization.jsonObject(with: data)
    let pretty = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    return String(decoding: pretty, as: UTF8.self)
}

private func makeStrategy(config: Config, language: String?) -> any SubtitleStitchingStrategy {
    let defaultStrategy: any SubtitleStitchingStrategy = SubtitleStitchingRegistry.default.strategy(for: language)

    switch config.strategy {
    case .auto:
        if config.maxUnits == nil, config.maxDuration == nil, config.pauseThreshold == nil {
            return defaultStrategy
        }
        switch defaultStrategy.id {
        case "compact-script":
            return CompactScriptSubtitleStitchingStrategy(
                maxUnitsPerSubtitle: config.maxUnits ?? defaultStrategy.maxUnitsPerSubtitle,
                maxDuration: config.maxDuration ?? defaultStrategy.maxDuration,
                pauseThreshold: config.pauseThreshold ?? defaultStrategy.pauseThreshold,
                sentenceEndChars: defaultStrategy.sentenceEndChars
            )
        default:
            return WordSubtitleStitchingStrategy(
                maxUnitsPerSubtitle: config.maxUnits ?? defaultStrategy.maxUnitsPerSubtitle,
                maxDuration: config.maxDuration ?? defaultStrategy.maxDuration,
                pauseThreshold: config.pauseThreshold ?? defaultStrategy.pauseThreshold,
                sentenceEndChars: defaultStrategy.sentenceEndChars
            )
        }
    case .word:
        return WordSubtitleStitchingStrategy(
            maxUnitsPerSubtitle: config.maxUnits ?? SubtitleFormattingDefaults.maxWordsPerLine,
            maxDuration: config.maxDuration ?? SubtitleFormattingDefaults.maxDuration,
            pauseThreshold: config.pauseThreshold ?? SubtitleFormattingDefaults.pauseThreshold,
            sentenceEndChars: SubtitleFormattingDefaults.sentenceEndChars
        )
    case .compactScript:
        return CompactScriptSubtitleStitchingStrategy(
            maxUnitsPerSubtitle: config.maxUnits ?? SubtitleFormattingDefaults.maxWordsPerLine,
            maxDuration: config.maxDuration ?? SubtitleFormattingDefaults.maxDuration,
            pauseThreshold: config.pauseThreshold ?? SubtitleFormattingDefaults.pauseThreshold,
            sentenceEndChars: SubtitleFormattingDefaults.compactScriptSentenceEndChars
        )
    }
}

private func run(config: Config) throws {
    let inputURL = URL(fileURLWithPath: config.inputPath).standardizedFileURL
    guard FileManager.default.fileExists(atPath: inputURL.path) else {
        throw CLIError.fileNotFound(config.inputPath)
    }

    let data = try Data(contentsOf: inputURL)
    let payload = try JSONDecoder().decode(SubtitleDebugPayload.self, from: data)
    guard let debug = payload.debug else { throw CLIError.debugMissing }

    let language = config.language ?? payload.language ?? "English"
    let strategy = makeStrategy(config: config, language: language)
    let subtitles = groupSubtitles(subtitleItems(from: debug), strategy: strategy)

    switch config.format {
    case .text:
        try writeOutput(payload.text, to: config.outputPath)
    case .srt:
        try writeOutput(formatSRT(subtitles), to: config.outputPath)
    case .vtt:
        try writeOutput(formatVTT(subtitles), to: config.outputPath)
    case .json:
        let output = formatSubtitleJSON(
            transcript: payload.text,
            language: language,
            duration: payload.duration,
            subtitles: subtitles,
            debug: payload.debug
        )
        try writeOutput(try prettyJSONString(output), to: config.outputPath)
    }
}

do {
    let config = try parseCLI(arguments: Array(CommandLine.arguments.dropFirst()))
    try run(config: config)
} catch CLIError.usage {
    printUsage()
    exit(0)
} catch let error as CLIError {
    fputs("Error: \(error.localizedDescription)\n\n", stderr)
    printUsage()
    exit(1)
} catch {
    fputs("Error: \(error.localizedDescription)\n", stderr)
    exit(1)
}
