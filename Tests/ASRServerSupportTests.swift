import ASRIPC
import Foundation
import Testing
@testable import ASRServerSupport
import NativeASR

@Suite("ASR server support")
struct ASRServerSupportTests {
    @Test func cliParserHandlesPositionalModelAndFlags() throws {
        let config = try parseASRServerCLI(arguments: [
            "/models/qwen",
            "--port", "9999",
            "--host", "0.0.0.0",
            "--parent-pid", "123",
            "--warmup",
            "--batch-model", "/models/batch",
            "--aligner-model", "/models/aligner",
            "--disable-batch-retranscribe",
            "--disable-vad",
            "--transport", "stdio",
        ])

        #expect(config.modelSpec == "/models/qwen")
        #expect(config.port == 9999)
        #expect(config.host == "0.0.0.0")
        #expect(config.parentPID == 123)
        #expect(config.warmup)
        #expect(config.batchModelPath == "/models/batch")
        #expect(config.alignerModelPath == "/models/aligner")
        #expect(config.batchRetranscribeEnabled == false)
        #expect(config.vadEnabled == false)
        #expect(config.batchChunking == .energy)
        #expect(config.transport == .stdio)
    }

    @Test func cliParserSelectsBatchChunkingWithoutDisablingStreamingVAD() throws {
        let config = try parseASRServerCLI(arguments: [
            "--batch-chunking", "energy",
        ])

        #expect(config.batchChunking == .energy)
        #expect(config.vadEnabled)
    }

    @Test func cliParserRejectsInvalidBatchChunking() {
        #expect(throws: ASRServerCLIError.invalidBatchChunking("nope")) {
            try parseASRServerCLI(arguments: ["--batch-chunking", "nope"])
        }
    }

    @Test func automaticBatchChunkingKeepsShortVADAndUsesEnergyForLongAudio() {
        #expect(BatchChunkingMode.automatic.resolved(audioDuration: 120.0, hasVAD: true) == .vad)
        #expect(BatchChunkingMode.automatic.resolved(audioDuration: 120.001, hasVAD: true) == .energy)
        #expect(BatchChunkingMode.automatic.resolved(audioDuration: 120.0, hasVAD: false) == .energy)
        #expect(BatchChunkingMode.vad.resolved(audioDuration: 3.0, hasVAD: true) == .vad)
        #expect(BatchChunkingMode.vad.resolved(audioDuration: 3.0, hasVAD: false) == .energy)
    }

    @Test func explicitBatchModeWinsOverDisableVADRegardlessOfArgumentOrder() throws {
        let configurations = try [
            parseASRServerCLI(arguments: ["--disable-vad", "--batch-chunking", "vad"]),
            parseASRServerCLI(arguments: ["--batch-chunking", "vad", "--disable-vad"]),
            parseASRServerCLI(arguments: ["--disable-vad", "--batch-chunking", "automatic"]),
            parseASRServerCLI(arguments: ["--batch-chunking", "automatic", "--disable-vad"]),
        ]

        #expect(configurations.map(\.vadEnabled) == [false, false, false, false])
        #expect(configurations.map(\.batchChunking) == [.vad, .vad, .automatic, .automatic])
    }

    @Test func cliParserPrefersExplicitModelOverPositionalModel() throws {
        let config = try parseASRServerCLI(arguments: [
            "/models/from-position",
            "--model", "/models/explicit",
        ])

        #expect(config.modelSpec == "/models/explicit")
        #expect(config.transport == .stdio)
    }

    @Test func cliParserAllowsNoExplicitModelForFallbackResolution() throws {
        let config = try parseASRServerCLI(arguments: [
            "--port", "9999",
        ])

        #expect(config.modelSpec == nil)
        #expect(config.port == 9999)
        #expect(config.transport == .stdio)
    }

    @Test func cliParserUsesCanonicalModelFlagWithoutPositionalModel() throws {
        let config = try parseASRServerCLI(arguments: [
            "--model", "mlx-community/Qwen3-ASR-1.7B-bf16",
            "--host", "0.0.0.0",
        ])

        #expect(config.modelSpec == "mlx-community/Qwen3-ASR-1.7B-bf16")
        #expect(config.host == "0.0.0.0")
        #expect(config.transport == .stdio)
    }

    @Test func cliParserRejectsInvalidPort() {
        #expect(throws: ASRServerCLIError.invalidPort("wat")) {
            try parseASRServerCLI(arguments: ["--model", "/models/qwen", "--port", "wat"])
        }
    }

    @Test func cliParserRejectsInvalidTransport() {
        #expect(throws: ASRServerCLIError.invalidTransport("grpc")) {
            try parseASRServerCLI(arguments: ["--transport", "grpc"])
        }
    }

    @Test func streamRecordingConfigurationReadsEnvironment() {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("yuwp-asr-recordings-env", isDirectory: true)
        let config = ASRStreamRecordingConfiguration.fromEnvironment(
            [
                "YUWP_ASR_SAVE_RECORDINGS": "1",
                "YUWP_ASR_RECORDINGS_DIR": dir.path,
            ],
            transcriptionModel: "qwen"
        )

        #expect(config.enabled)
        #expect(config.directory == dir.standardizedFileURL)
        #expect(config.transcriptionModel == "qwen")
    }

    @Test func streamRecordingArtifactWriterWritesWavTranscriptAndMetadata() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("yuwp-asr-recording-artifact-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let pcm = Data(repeating: 0x7f, count: 3_200)
        let context = ASRStreamRecordingContext(
            sessionID: "session123",
            transcriptionModel: "qwen",
            languageHint: "English"
        )
        let date = Date(timeIntervalSince1970: 1_700_000_000)
        let handle = try ASRStreamRecordingArtifactWriter.writeRecording(
            pcmData: pcm,
            directory: dir,
            context: context,
            date: date
        )
        try ASRStreamRecordingArtifactWriter.writeTranscript("hello", for: handle, context: context, updatedAt: date)

        let wav = try Data(contentsOf: handle.audioURL)
        #expect(String(data: Data(wav[0..<4]), encoding: .ascii) == "RIFF")
        #expect(String(data: Data(wav[8..<12]), encoding: .ascii) == "WAVE")
        let dataLength = UInt32(wav[40]) | UInt32(wav[41]) << 8 | UInt32(wav[42]) << 16 | UInt32(wav[43]) << 24
        #expect(dataLength == UInt32(pcm.count))

        let transcript = try String(contentsOf: handle.transcriptURL, encoding: .utf8)
        #expect(transcript == "hello\n")
        let metadata = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: handle.metadataURL)) as? [String: Any])
        #expect(metadata["source"] as? String == "asr_stream")
        #expect(metadata["sessionID"] as? String == "session123")
        #expect(metadata["transcriptionModel"] as? String == "qwen")
        #expect(metadata["languageHint"] as? String == "English")
        #expect(metadata["transcript"] as? String == "hello")
    }

    @Test func multipartParserExtractsFieldsAndFiles() {
        let boundary = "Boundary-123"
        var body = Data()

        func append(_ string: String) {
            body.append(Data(string.utf8))
        }

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"language\"\r\n\r\n")
        append("English\r\n")
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(Data([0x01, 0x02, 0x03]))
        append("\r\n")
        append("--\(boundary)--\r\n")

        let parts = parseMultipartBody(
            body: body,
            contentTypeHeader: "multipart/form-data; boundary=\(boundary)"
        )

        #expect(parts?.count == 2)
        #expect(parts?.first?.name == "language")
        #expect(parts?.first?.textValue == "English")
        #expect(parts?.last?.name == "file")
        #expect(parts?.last?.filename == "clip.wav")
        #expect(parts?.last?.contentType == "audio/wav")
        #expect(parts?.last?.body == Data([0x01, 0x02, 0x03]))
    }

    @Test func subtitleFormattingAndLanguageNormalizationWork() throws {
        let items = [
            ForcedAlignItem(text: "hello", startTime: 0.0, endTime: 0.4),
            ForcedAlignItem(text: "world", startTime: 0.45, endTime: 0.8),
            ForcedAlignItem(text: "again.", startTime: 1.6, endTime: 2.0),
        ]

        let subtitles = groupSubtitles(items, maxWordsPerLine: 10, maxDuration: 5.0, pauseThreshold: 0.5)
        #expect(subtitles == [
            Subtitle(index: 1, start: 0.0, end: 0.8, text: "hello world"),
            Subtitle(index: 2, start: 1.6, end: 2.0, text: "again."),
        ])

        let stitchedBoundaryItems = [
            ForcedAlignItem(text: "technology.", startTime: 362.70, endTime: 363.70),
            ForcedAlignItem(text: "Ultimately,", startTime: 364.05, endTime: 364.40),
            ForcedAlignItem(text: "it's", startTime: 364.40, endTime: 364.60),
        ]
        #expect(groupSubtitles(stitchedBoundaryItems, maxWordsPerLine: 10, maxDuration: 5.0, pauseThreshold: 0.5) == [
            Subtitle(index: 1, start: 362.70, end: 363.70, text: "technology."),
            Subtitle(index: 2, start: 364.05, end: 364.60, text: "Ultimately, it's"),
        ])

        let chineseItems = [
            ForcedAlignItem(text: "交", startTime: 0.0, endTime: 0.1),
            ForcedAlignItem(text: "易；", startTime: 0.1, endTime: 0.2),
            ForcedAlignItem(text: "几", startTime: 0.21, endTime: 0.3),
            ForcedAlignItem(text: "乎", startTime: 0.3, endTime: 0.4),
        ]
        #expect(groupSubtitles(chineseItems, language: "Chinese") == [
            Subtitle(index: 1, start: 0.0, end: 0.2, text: "交易；"),
            Subtitle(index: 2, start: 0.21, end: 0.4, text: "几乎"),
        ])

        #expect(formatSRT(subtitles).contains("00:00:00,000 --> 00:00:00,800"))
        #expect(formatVTT(subtitles).hasPrefix("WEBVTT"))
        #expect(formatLRC(subtitles).contains("[00:00.00]hello world"))
        #expect(normalizeLanguageCode(" English ") == "en")
        #expect(normalizeLanguageCode("pt_br") == "pt-BR")

        let debug = BatchSubtitleDebug(
            chunkingMode: "VAD",
            chunkCount: 1,
            chunks: [
                BatchSubtitleChunkDebug(
                    index: 1,
                    start: 0.0,
                    end: 2.0,
                    duration: 2.0,
                    transcript: "hello world again",
                    items: [BatchAlignmentItemDebug(text: "hello,", alignText: "hello", start: 0.0, end: 0.4)]
                )
            ]
        )
        let payload = formatSubtitleJSON(
            transcript: "hello world again",
            language: "English",
            duration: 2.0,
            subtitles: subtitles,
            debug: debug
        )
        let json = try #require(try JSONSerialization.jsonObject(with: payload) as? [String: Any])
        #expect(json["language"] as? String == "en")
        #expect((json["segments"] as? [[String: Any]])?.count == 2)
        let debugJSON = try #require(json["debug"] as? [String: Any])
        #expect(debugJSON["chunkingMode"] as? String == "VAD")
        #expect(debugJSON["chunkCount"] as? Int == 1)
        let debugChunk = try #require((debugJSON["chunks"] as? [[String: Any]])?.first)
        let debugItem = try #require((debugChunk["items"] as? [[String: Any]])?.first)
        #expect(debugItem["text"] as? String == "hello,")
        #expect(debugItem["alignText"] as? String == "hello")
    }

    @Test func debugPayloadCanBeRestitchedWithoutRetranscribing() {
        let payload = SubtitleDebugPayload(
            text: "technology. Ultimately, it's the team.",
            language: "English",
            duration: 2.0,
            debug: BatchSubtitleDebug(
                chunkingMode: "VAD",
                chunkCount: 2,
                chunks: [
                    BatchSubtitleChunkDebug(
                        index: 1,
                        start: 0.0,
                        end: 1.0,
                        duration: 1.0,
                        transcript: "technology.",
                        items: [BatchAlignmentItemDebug(text: "technology.", start: 0.0, end: 0.8)]
                    ),
                    BatchSubtitleChunkDebug(
                        index: 2,
                        start: 1.0,
                        end: 2.0,
                        duration: 1.0,
                        transcript: "Ultimately, it's the team.",
                        items: [
                            BatchAlignmentItemDebug(text: "Ultimately,", alignText: "Ultimately", start: 1.05, end: 1.30),
                            BatchAlignmentItemDebug(text: "it's", start: 1.30, end: 1.45),
                            BatchAlignmentItemDebug(text: "the", start: 1.45, end: 1.60),
                            BatchAlignmentItemDebug(text: "team.", start: 1.60, end: 1.90),
                        ]
                    ),
                ]
            )
        )

        #expect(restitchSubtitles(from: payload) == [
            Subtitle(index: 1, start: 0.0, end: 0.8, text: "technology."),
            Subtitle(index: 2, start: 1.05, end: 1.90, text: "Ultimately, it's the team."),
        ])
    }

    @Test func subtitleRegistrySelectsCompactScriptStrategyForChinese() {
        let registry = SubtitleStitchingRegistry.default
        #expect(registry.strategy(for: "English").id == "word")
        #expect(registry.strategy(for: "Chinese").id == "compact-script")
        #expect(registry.strategy(for: "yue").id == "compact-script")
        #expect(registry.strategy(for: "ja").id == "compact-script")
    }

    @Test func routeInfoReportsCapabilities() throws {
        let manager = FakeManager()
        let context = ASRRouteContext(
            manager: manager,
            aligner: nil,
            vad: nil,
            streamingModelName: "streaming-model",
            batchModelName: "batch-model",
            batchRetranscribeEnabled: true,
            loadAudio: { _ in [] }
        )

        let response = routeRequest(HTTPRequest(method: "GET", path: "/v1/info", headers: [:], body: Data()), context: context)
        let json = try #require(try JSONSerialization.jsonObject(with: response.body) as? [String: Any])

        #expect(response.status == 200)
        #expect(json["model"] as? String == "streaming-model")
        #expect(json["final_accuracy_pass_model"] as? String == "batch-model")
        #expect(json["final_accuracy_pass_enabled"] as? Bool == true)
        #expect(json["status"] as? String == "ready")
        #expect(json["aligner"] as? Bool == false)
        #expect(json["vad"] as? Bool == false)
        #expect(json["batch_chunking"] as? String == "automatic")
        #expect(json["batch_chunking_requested"] as? String == "automatic")
        #expect(json["batch_chunking_resolution"] as? String == "energy")
        #expect(json["batch_chunking_vad_available"] as? Bool == false)
    }

    @Test func routeInfoDistinguishesExplicitVADFallbackFromRequestedMode() throws {
        let manager = FakeManager()
        let context = ASRRouteContext(
            manager: manager,
            aligner: nil,
            vad: nil,
            batchVAD: nil,
            batchChunking: .vad,
            streamingModelName: "streaming-model",
            batchModelName: nil,
            batchRetranscribeEnabled: false,
            loadAudio: { _ in [] }
        )

        let response = routeRequest(HTTPRequest(method: "GET", path: "/v1/info", headers: [:], body: Data()), context: context)
        let json = try #require(try JSONSerialization.jsonObject(with: response.body) as? [String: Any])

        #expect(json["batch_chunking"] as? String == "vad")
        #expect(json["batch_chunking_requested"] as? String == "vad")
        #expect(json["batch_chunking_resolution"] as? String == "energy")
        #expect(json["batch_chunking_vad_available"] as? Bool == false)
    }

    @Test func batchRouteHonorsExplicitVADFallbackWhenBatchVADIsUnavailable() throws {
        let manager = FakeManager()
        manager.transcribeResult = TranscriptionResult(
            text: "chunk",
            language: "English",
            audioDuration: ASRServerLimits.maxChunkSec,
            processingTime: 0.1
        )

        let multipart = makeMultipartRequest(
            fields: [:],
            fileName: "long.wav",
            fileContentType: "audio/wav",
            fileData: Data([0x00, 0x01])
        )
        let request = HTTPRequest(
            method: "POST",
            path: "/v1/audio/transcriptions",
            headers: ["content-type": multipart.contentType],
            body: multipart.body
        )
        let totalSamples = Int((ASRServerLimits.maxChunkSec * 2 + 1) * Double(ASRAudio.sampleRate))
        let context = ASRRouteContext(
            manager: manager,
            aligner: nil,
            vad: nil,
            batchVAD: nil,
            batchChunking: .vad,
            streamingModelName: "stream",
            batchModelName: nil,
            batchRetranscribeEnabled: true,
            loadAudio: { _ in Array(repeating: 0, count: totalSamples) }
        )

        let response = routeRequest(request, context: context)
        let json = try #require(try JSONSerialization.jsonObject(with: response.body) as? [String: Any])
        #expect(response.status == 200)
        #expect(json["text"] as? String == "chunk chunk chunk")
        #expect(manager.transcribeCallCount == 3)
    }

    @Test func batchRouteUsesInjectedAudioLoaderAndManager() throws {
        let manager = FakeManager()
        manager.transcribeResult = TranscriptionResult(
            text: "hello from batch",
            language: "English",
            audioDuration: 1.25,
            processingTime: 0.1
        )

        let multipart = makeMultipartRequest(
            fields: [
                "response_format": "json",
                "language": "English",
                "temperature": "0.25",
            ],
            fileName: "../clip.wav",
            fileContentType: "audio/wav",
            fileData: Data([0x00, 0x01])
        )
        let request = HTTPRequest(
            method: "POST",
            path: "/v1/audio/transcriptions",
            headers: ["content-type": multipart.contentType],
            body: multipart.body
        )

        let context = ASRRouteContext(
            manager: manager,
            aligner: nil,
            vad: nil,
            streamingModelName: "stream",
            activeModelID: "qwen3-asr-0.6b",
            batchModelName: nil,
            batchRetranscribeEnabled: true,
            loadAudio: { _ in Array(repeating: 0, count: 16_000) }
        )

        let response = routeRequest(request, context: context)
        let json = try #require(try JSONSerialization.jsonObject(with: response.body) as? [String: Any])

        #expect(response.status == 200)
        #expect(json["text"] as? String == "hello from batch")
        #expect(json["language"] as? String == "en")
        #expect(manager.transcribeCallCount == 1)
        #expect(manager.lastTemperature == 0.25)
        #expect(manager.lastLanguage == "English")
    }

    @Test func batchRouteRejectsUnsupportedModel() throws {
        let manager = FakeManager()
        let multipart = makeMultipartRequest(
            fields: [
                "model": "whisper-1",
            ],
            fileName: "clip.wav",
            fileContentType: "audio/wav",
            fileData: Data([0x00])
        )
        let request = HTTPRequest(
            method: "POST",
            path: "/v1/audio/transcriptions",
            headers: ["content-type": multipart.contentType],
            body: multipart.body
        )
        let context = ASRRouteContext(
            manager: manager,
            aligner: nil,
            vad: nil,
            streamingModelName: "stream",
            activeModelID: "qwen3-asr-0.6b",
            batchModelName: nil,
            batchRetranscribeEnabled: true,
            loadAudio: { _ in [] }
        )

        let response = routeRequest(request, context: context)
        let json = try #require(try JSONSerialization.jsonObject(with: response.body) as? [String: Any])
        let error = try #require(json["error"] as? [String: Any])
        #expect(response.status == 400)
        #expect(error["param"] as? String == "model")
    }

    @Test func streamRoutesDelegateToManager() throws {
        let manager = FakeManager()
        manager.feedResponse = ["text": "partial"]
        manager.stopResponse = ["text": "final"]
        let context = ASRRouteContext(
            manager: manager,
            aligner: nil,
            vad: nil,
            streamingModelName: "stream",
            batchModelName: nil,
            batchRetranscribeEnabled: true,
            loadAudio: { _ in [] }
        )

        let create = routeRequest(HTTPRequest(method: "POST", path: "/v1/audio/transcriptions/stream", headers: [:], body: Data()), context: context)
        let createJSON = try #require(try JSONSerialization.jsonObject(with: create.body) as? [String: Any])
        let sid = try #require(createJSON["session_id"] as? String)
        #expect(sid == manager.createdSessionID)
        #expect(manager.lastCreatedLanguage == nil)

        let feed = routeRequest(HTTPRequest(method: "POST", path: "/v1/audio/transcriptions/stream/abc123", headers: [:], body: Data([1, 2])), context: context)
        let feedJSON = try #require(try JSONSerialization.jsonObject(with: feed.body) as? [String: Any])
        #expect(feed.status == 200)
        #expect(feedJSON["text"] as? String == "partial")
        #expect(manager.lastFedSessionID == "abc123")

        let stop = routeRequest(HTTPRequest(method: "DELETE", path: "/v1/audio/transcriptions/stream/abc123", headers: [:], body: Data()), context: context)
        let stopJSON = try #require(try JSONSerialization.jsonObject(with: stop.body) as? [String: Any])
        #expect(stop.status == 200)
        #expect(stopJSON["text"] as? String == "final")
        #expect(manager.lastStoppedSessionID == "abc123")
    }

    @Test func streamCreatePassesLanguageQueryToManager() throws {
        let manager = FakeManager()
        let context = ASRRouteContext(
            manager: manager,
            aligner: nil,
            vad: nil,
            streamingModelName: "stream",
            batchModelName: nil,
            batchRetranscribeEnabled: true,
            loadAudio: { _ in [] }
        )

        let create = routeRequest(
            HTTPRequest(
                method: "POST",
                path: "/v1/audio/transcriptions/stream?language=Chinese",
                headers: [:],
                body: Data()
            ),
            context: context
        )

        #expect(create.status == 200)
        #expect(manager.lastCreatedLanguage == "Chinese")
    }

    @Test func batchRouteFallsBackToLowEnergyChunksWithoutVAD() throws {
        let manager = FakeManager()
        manager.transcribeResult = TranscriptionResult(
            text: "chunk",
            language: "English",
            audioDuration: ASRServerLimits.maxChunkSec,
            processingTime: 0.1
        )

        let multipart = makeMultipartRequest(
            fields: [:],
            fileName: "long.wav",
            fileContentType: "audio/wav",
            fileData: Data([0x00, 0x01])
        )
        let request = HTTPRequest(
            method: "POST",
            path: "/v1/audio/transcriptions",
            headers: ["content-type": multipart.contentType],
            body: multipart.body
        )

        let totalSamples = Int((ASRServerLimits.maxChunkSec * 2 + 1) * Double(ASRAudio.sampleRate))
        let context = ASRRouteContext(
            manager: manager,
            aligner: nil,
            vad: nil,
            streamingModelName: "stream",
            batchModelName: nil,
            batchRetranscribeEnabled: true,
            loadAudio: { _ in Array(repeating: 0, count: totalSamples) }
        )

        let response = routeRequest(request, context: context)
        let json = try #require(try JSONSerialization.jsonObject(with: response.body) as? [String: Any])
        #expect(response.status == 200)
        #expect(json["text"] as? String == "chunk chunk chunk")
        #expect(manager.transcribeCallCount == 3)
    }

    @Test func sharedBatchPipelineChunksLongAudioWithoutVAD() throws {
        let manager = FakeManager()
        manager.transcribeResult = TranscriptionResult(
            text: "chunk",
            language: "English",
            audioDuration: ASRServerLimits.maxChunkSec,
            processingTime: 0.1
        )

        let totalSamples = Int((ASRServerLimits.maxChunkSec * 2 + 1) * Double(ASRAudio.sampleRate))
        let audio = Array(repeating: Float(0), count: totalSamples)
        let result = try BatchTranscriptionPipeline.transcribe(
            using: manager,
            audio: audio,
            language: nil,
            temperature: 0.0,
            vad: nil
        )

        #expect(result.text == "chunk chunk chunk")
        #expect(result.language == "English")
        #expect(result.audioDuration == Double(totalSamples) / Double(ASRAudio.sampleRate))
        #expect(manager.transcribeCallCount == 3)
    }

    @Test func splitTextProportionallyProducesTrimmedBalancedChunks() {
        let parts = splitTextProportionally("hello world. nice to meet you.", chunkDurations: [1, 1])
        #expect(parts.count == 2)
        #expect(parts.allSatisfy { !$0.isEmpty && $0 == $0.trimmingCharacters(in: .whitespacesAndNewlines) })
        #expect(parts[0].contains("hello world"))
        #expect(parts[1].hasSuffix("you."))
    }
}

private final class FakeManager: ASRServing, @unchecked Sendable {
    var createdSessionID = "fake-session"
    var feedResponse: [String: Any]? = nil
    var stopResponse: [String: Any]? = nil
    var transcribeResult = TranscriptionResult(text: "", language: nil, audioDuration: 0, processingTime: 0)
    var subtitleResult: (transcript: String, language: String, items: [ForcedAlignItem]) = ("", "English", [])

    var transcribeCallCount = 0
    var lastLanguage: String?
    var lastTemperature: Float?
    var lastFedSessionID: String?
    var lastStoppedSessionID: String?
    var lastCreatedLanguage: String?

    func create(language: String?) -> String {
        lastCreatedLanguage = language
        return createdSessionID
    }

    func feed(_ sid: String, pcmData: Data) -> [String: Any]? {
        lastFedSessionID = sid
        return feedResponse
    }

    func stop(_ sid: String) -> [String: Any]? {
        lastStoppedSessionID = sid
        return stopResponse
    }

    func transcribeAudio(audio: [Float], language: String?, temperature: Float) throws -> TranscriptionResult {
        transcribeCallCount += 1
        lastLanguage = language
        lastTemperature = temperature
        return transcribeResult
    }

    func transcribeChunk(audio: [Float], language: String?, temperature: Float) throws -> TranscriptionResult {
        transcribeCallCount += 1
        lastLanguage = language
        lastTemperature = temperature
        return transcribeResult
    }

    func subtitleItems(
        audio: [Float],
        transcript: String?,
        language: String?,
        temperature: Float,
        aligner: ForcedAligner
    ) throws -> (transcript: String, language: String, items: [ForcedAlignItem]) {
        subtitleResult
    }
}

private func makeMultipartRequest(
    fields: [String: String],
    fileName: String,
    fileContentType: String,
    fileData: Data
) -> (body: Data, contentType: String) {
    let boundary = "Boundary-\(UUID().uuidString)"
    var body = Data()

    func append(_ string: String) {
        body.append(Data(string.utf8))
    }

    for (name, value) in fields.sorted(by: { $0.key < $1.key }) {
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n")
        append("\(value)\r\n")
    }

    append("--\(boundary)\r\n")
    append("Content-Disposition: form-data; name=\"file\"; filename=\"\(fileName)\"\r\n")
    append("Content-Type: \(fileContentType)\r\n\r\n")
    body.append(fileData)
    append("\r\n")
    append("--\(boundary)--\r\n")

    return (body, "multipart/form-data; boundary=\(boundary)")
}
