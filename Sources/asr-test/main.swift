// asr-test — CLI test for NativeASR batch transcription
// Usage: asr-test <wav-file> <model-dir> [--warmup] [--language <lang>]

import Foundation
import NativeASR

func printUsage() {
    fputs("""
    Usage: asr-test <wav-file> <model-dir> [options]

    Options:
      --warmup          Warm up Metal shaders before transcribing (eliminates JIT latency)
      --language <lang> Force language hint
      --no-language     Auto-detect language (default)
      --max-tokens <n>  Maximum tokens to generate (default: 4096)

    Example:
      asr-test recording.wav ~/workspace/qwen-asr/qwen3-asr-0.6b --warmup

    """, stderr)
}

func runMain() throws {
    var args = CommandLine.arguments.dropFirst()

    guard args.count >= 2 else {
        printUsage()
        exit(1)
    }

    let wavPath = args.removeFirst()
    let modelPath = args.removeFirst()
    var doWarmup = false
    var language: String? = nil
    var maxTokens = 4096

    while !args.isEmpty {
        let flag = args.removeFirst()
        switch flag {
        case "--warmup":
            doWarmup = true
        case "--language":
            guard !args.isEmpty else { fputs("--language requires an argument\n", stderr); exit(1) }
            language = args.removeFirst()
        case "--no-language":
            language = nil
        case "--max-tokens":
            guard !args.isEmpty, let n = Int(args.removeFirst()) else {
                fputs("--max-tokens requires an integer\n", stderr); exit(1)
            }
            maxTokens = n
        default:
            fputs("Unknown option: \(flag)\n", stderr)
            printUsage()
            exit(1)
        }
    }

    let wavURL = URL(fileURLWithPath: wavPath)
    let modelURL = URL(fileURLWithPath: modelPath)

    guard FileManager.default.fileExists(atPath: wavURL.path) else {
        fputs("Error: WAV file not found: \(wavPath)\n", stderr)
        exit(1)
    }
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
        fputs("Error: Model directory not found: \(modelPath)\n", stderr)
        exit(1)
    }

    // Load
    let transcriber = try Qwen3ASRTranscriber.load(from: modelURL)

    // Warmup
    if doWarmup {
        try transcriber.warmup()
    }

    // Transcribe
    fputs("[NativeASR] Transcribing \(wavURL.lastPathComponent)...\n", stderr)
    let result = try transcriber.transcribe(file: wavURL, language: language, maxTokens: maxTokens)

    // Output
    print(result.text)

    // Timing info to stderr
    fputs("""
    [NativeASR] Done
      Audio:    \(String(format: "%.2f", result.audioDuration))s
      Inference: \(String(format: "%.2f", result.processingTime))s
      RTF:      \(String(format: "%.3f", result.rtf)) (\(String(format: "%.1f", result.speedMultiplier))x real-time)
    \n
    """, stderr)
}

do {
    try runMain()
    exit(0)
} catch {
    fputs("Error: \(error.localizedDescription)\n", stderr)
    exit(1)
}
