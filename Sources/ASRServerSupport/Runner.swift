import ASRIPC
import Foundation
import NativeASR

public func runASRServer(arguments: [String]) -> Int32 {
    do {
        let config = try parseASRServerCLI(arguments: arguments)

        guard let modelURL = YuwpModelSupport.resolveConfiguredModelURL(explicitSpec: config.modelSpec) else {
            fputs("Error: could not resolve a model directory. Pass --model or set Yuwp's transcription model first.\n", stderr)
            return 1
        }

        let transcriber = try Qwen3ASRTranscriber.load(from: modelURL)

        let batchTranscriber: Qwen3ASRTranscriber?
        if config.batchRetranscribeEnabled, let batchModelPath = config.batchModelPath {
            let batchURL = URL(fileURLWithPath: batchModelPath).standardizedFileURL
            guard FileManager.default.fileExists(atPath: batchURL.path) else {
                fputs("Batch model not found: \(batchModelPath)\n", stderr)
                return 1
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
                return 1
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
            log("Silero VAD disabled for live streaming")
            vad = nil
        }

        let batchVAD: SileroVAD?
        if config.transport == .stdio || config.batchChunking == .energy {
            batchVAD = nil
        } else {
            do {
                batchVAD = try SileroVAD()
                log("Batch Silero VAD loaded")
            } catch {
                log("Batch Silero VAD unavailable: \(error.localizedDescription)")
                batchVAD = nil
            }
        }

        if config.warmup {
            try transcriber.warmup()
            if let batchTranscriber, batchTranscriber !== transcriber {
                try batchTranscriber.warmup()
            }
        }

        let recordingConfiguration = ASRStreamRecordingConfiguration.fromEnvironment(
            transcriptionModel: modelURL.lastPathComponent
        )
        if recordingConfiguration.enabled {
            log("ASR stream recording enabled: \(recordingConfiguration.directory.path)")
        }

        let manager = StreamingSessionManager(
            transcriber: transcriber,
            batchTranscriber: batchTranscriber,
            batchRetranscribeEnabled: config.batchRetranscribeEnabled,
            vad: vad,
            batchVAD: batchVAD,
            batchChunking: config.batchChunking,
            recordingConfiguration: recordingConfiguration
        )
        switch config.transport {
        case .http:
            startServer(
                host: config.host,
                port: config.port,
                mgr: manager,
                aligner: aligner,
                vad: vad,
                batchVAD: batchVAD,
                batchChunking: config.batchChunking,
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
        return 0
    } catch let error as ASRServerCLIError {
        fputs("\(error.localizedDescription)\n", stderr)
        return 1
    } catch {
        fputs("Error: \(error.localizedDescription)\n", stderr)
        return 1
    }
}
