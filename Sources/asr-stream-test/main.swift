// asr-stream-test — CLI streaming ASR validation tool
// Simulates streaming by feeding a WAV file in chunks to StreamingSession,
// then compares the final output against batch transcription.
//
// Usage: asr-stream-test <wav-file> <model-dir> [--chunk-sec 2.0] [--warmup]

import Foundation
import NativeASR

func printUsage() {
    fputs("""
    Usage: asr-stream-test <wav-file> <model-dir> [options]

    Options:
      --chunk-sec <sec>  Chunk duration in seconds (default: 2.0)
      --warmup           Warm up Metal shaders before streaming
      --no-batch         Skip batch comparison

    Example:
      asr-stream-test jfk.wav ~/models/Qwen3-ASR-0.6B-4bit --warmup


    """, stderr)
}

func runMain() throws {
    var args = Array(CommandLine.arguments.dropFirst())

    guard args.count >= 2 else {
        printUsage()
        exit(1)
    }

    let wavPath = args.removeFirst()
    let modelPath = args.removeFirst()
    var chunkSec = 2.0
    var doWarmup = false
    var doBatch = true

    while !args.isEmpty {
        let flag = args.removeFirst()
        switch flag {
        case "--chunk-sec":
            guard !args.isEmpty, let v = Double(args.removeFirst()) else {
                fputs("--chunk-sec requires a number\n", stderr); exit(1)
            }
            chunkSec = v
        case "--warmup":
            doWarmup = true
        case "--no-batch":
            doBatch = false
        default:
            fputs("Unknown option: \(flag)\n", stderr)
            printUsage()
            exit(1)
        }
    }

    let wavURL = URL(fileURLWithPath: wavPath)
    let modelURL = URL(fileURLWithPath: modelPath)

    guard FileManager.default.fileExists(atPath: wavURL.path) else {
        fputs("Error: WAV file not found: \(wavPath)\n", stderr); exit(1)
    }
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
        fputs("Error: Model directory not found: \(modelPath)\n", stderr); exit(1)
    }

    // --- Load model ---
    let transcriber = try Qwen3ASRTranscriber.load(from: modelURL)

    if doWarmup {
        try transcriber.warmup()
    }

    // --- Load audio ---
    fputs("[stream-test] Loading \(wavURL.lastPathComponent)...\n", stderr)
    let audio = try loadAudioFile(wavURL)
    let audioDuration = Double(audio.count) / Double(ASRAudio.sampleRate)
    fputs("[stream-test] Audio: \(String(format: "%.2f", audioDuration))s, \(audio.count) samples\n", stderr)

    // --- Streaming pass ---
    let config = StreamConfig(chunkSec: chunkSec)
    let session = StreamingSession(transcriber: transcriber, config: config)
    let chunkSize = Int(chunkSec * Double(ASRAudio.sampleRate))

    fputs("\n[stream-test] === Streaming (chunk=\(String(format: "%.1f", chunkSec))s) ===\n", stderr)
    let streamT0 = Date()
    var offset = 0
    var chunkNum = 0

    while offset < audio.count {
        let end = min(offset + chunkSize, audio.count)
        let chunk = Array(audio[offset ..< end])
        let result = session.processChunk(chunk)
        chunkNum += 1

        let chunkDur = Double(end - offset) / Double(ASRAudio.sampleRate)
        let tag = result.batchCorrected ? " [BATCH]" : ""
        let timing = result.totalMs > 0
            ? String(format: "%.0fms (enc=%.0f pfx=%.0f dec=%.0f reuse=%.0f%%)",
                     result.totalMs, result.encodeMs, result.prefillMs, result.decodeMs, result.reusePct)
            : String(format: "%.0fms", result.totalMs)

        fputs(String(format: "  chunk %2d (%.1fs): %@ %@\n",
                     chunkNum, chunkDur, timing, tag), stderr)

        let preview = result.text.count > 80
            ? String(result.text.prefix(77)) + "..."
            : result.text
        if !preview.isEmpty {
            fputs("    → \"\(preview)\"\n", stderr)
        }

        offset = end
    }

    let streamTime = Date().timeIntervalSince(streamT0)
    let streamText = session.finalText()

    fputs("\n[stream-test] Streaming done in \(String(format: "%.2f", streamTime))s\n", stderr)
    fputs("[stream-test] Final: \"\(streamText)\"\n", stderr)

    // --- Batch comparison ---
    if doBatch {
        fputs("\n[stream-test] === Batch transcription ===\n", stderr)
        let batchResult = try transcriber.transcribe(audio: audio)
        fputs("[stream-test] Batch: \"\(batchResult.text)\"\n", stderr)
        fputs("[stream-test] Batch time: \(String(format: "%.2f", batchResult.processingTime))s\n", stderr)

        if streamText == batchResult.text {
            fputs("\n[stream-test] ✅ MATCH — streaming and batch outputs are identical\n", stderr)
        } else {
            fputs("\n[stream-test] ⚠️  DIFF — outputs differ (expected for streaming)\n", stderr)
            fputs("[stream-test]   Stream: \(streamText.count) chars\n", stderr)
            fputs("[stream-test]   Batch:  \(batchResult.text.count) chars\n", stderr)
        }
    }

    // Print final text to stdout (machine-readable)
    print(streamText)
}

do {
    try runMain()
    exit(0)
} catch {
    fputs("Error: \(error.localizedDescription)\n", stderr)
    exit(1)
}
