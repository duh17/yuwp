import ASRIPC
import ASRServerSupport
import Foundation
import NativeASR

// swift-mlx-asr-server — native streaming ASR HTTP server.
// Keep this file as thin orchestration glue so the real logic lives in testable units.

do {
    let config = try parseASRServerCLI(arguments: Array(CommandLine.arguments.dropFirst()))

    guard let modelURL = YuwpModelSupport.resolveConfiguredModelURL(explicitSpec: config.modelSpec) else {
        fputs("Error: could not resolve a model directory. Pass --model or set Yuwp's transcription model first.\n", stderr)
        exit(1)
    }

    let transcriber = try Qwen3ASRTranscriber.load(from: modelURL)

    let batchTranscriber: Qwen3ASRTranscriber?
    if config.batchRetranscribeEnabled, let batchModelPath = config.batchModelPath {
        let batchURL = URL(fileURLWithPath: batchModelPath).standardizedFileURL
        guard FileManager.default.fileExists(atPath: batchURL.path) else {
            fputs("Batch model not found: \(batchModelPath)\n", stderr)
            exit(1)
        }
        batchTranscriber = batchURL == modelURL.standardizedFileURL
            ? transcriber
            : try Qwen3ASRTranscriber.load(from: batchURL)
    } else {
        batchTranscriber = nil
    }

    let aligner: ForcedAligner?
    if let alignerSpec = config.alignerModelPath {
        let alignerURL = URL(fileURLWithPath: alignerSpec).standardizedFileURL
        guard FileManager.default.fileExists(atPath: alignerURL.path) else {
            fputs("Aligner model not found: \(alignerSpec)\n", stderr)
            exit(1)
        }
        log("Loading aligner model from \(alignerURL.path)...")
        aligner = try ForcedAligner.load(from: alignerURL)
        if let aligner {
            log("Aligner loaded (classify_num=\(aligner.model.config.classifyNum))")
        }
    } else {
        aligner = nil
    }

    let vad: SileroVAD?
    if config.vadEnabled {
        do {
            vad = try SileroVAD()
            log("Silero VAD loaded")
        } catch {
            log("Silero VAD unavailable: \(error.localizedDescription)")
            vad = nil
        }
    } else {
        log("Silero VAD disabled")
        vad = nil
    }

    if config.warmup {
        try transcriber.warmup()
        if let batchTranscriber, batchTranscriber !== transcriber {
            try batchTranscriber.warmup()
        }
    }

    let manager = StreamingSessionManager(
        transcriber: transcriber,
        batchTranscriber: batchTranscriber,
        batchRetranscribeEnabled: config.batchRetranscribeEnabled,
        vad: vad
    )
    switch config.transport {
    case .http:
        startServer(
            host: config.host,
            port: config.port,
            mgr: manager,
            aligner: aligner,
            vad: vad,
            streamingModelName: modelURL.lastPathComponent,
            activeModelID: YuwpModelSupport.publicModelID(for: config.modelSpec ?? YuwpModelSupport.defaultYuwpModelSpec() ?? modelURL.path),
            batchModelName: batchTranscriber?.modelDirectory.lastPathComponent,
            batchRetranscribeEnabled: config.batchRetranscribeEnabled,
            parentPID: config.parentPID
        )
    case .stdio:
        startStdioServer(
            mgr: manager,
            streamingModelName: modelURL.lastPathComponent,
            activeModelID: YuwpModelSupport.publicModelID(for: config.modelSpec ?? YuwpModelSupport.defaultYuwpModelSpec() ?? modelURL.path),
            batchRetranscribeEnabled: config.batchRetranscribeEnabled
        )
    }
} catch let error as ASRServerCLIError {
    fputs("\(error.localizedDescription)\n", stderr)
    exit(1)
} catch {
    fputs("Error: \(error.localizedDescription)\n", stderr)
    exit(1)
}
