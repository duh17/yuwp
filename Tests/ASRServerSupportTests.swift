import Foundation
import Testing
@testable import ASRServerSupport
import NativeASR

@Suite("ASR server support")
struct ASRServerSupportTests {
    @Test func cliParserHandlesLegacyPositionalModelAndFlags() throws {
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
    }

    @Test func cliParserPrefersExplicitModelOverLegacyPositionalModel() throws {
        let config = try parseASRServerCLI(arguments: [
            "/models/legacy",
            "--model", "/models/explicit",
        ])

        #expect(config.modelSpec == "/models/explicit")
    }

    @Test func cliParserAllowsNoExplicitModelForFallbackResolution() throws {
        let config = try parseASRServerCLI(arguments: [
            "--port", "9999",
        ])

        #expect(config.modelSpec == nil)
        #expect(config.port == 9999)
    }

    @Test func cliParserUsesCanonicalModelFlagWithoutPositionalModel() throws {
        let config = try parseASRServerCLI(arguments: [
            "--model", "mlx-community/Qwen3-ASR-1.7B-bf16",
            "--host", "0.0.0.0",
        ])

        #expect(config.modelSpec == "mlx-community/Qwen3-ASR-1.7B-bf16")
        #expect(config.host == "0.0.0.0")
    }

    @Test func cliParserRejectsInvalidPort() {
        #expect(throws: ASRServerCLIError.invalidPort("wat")) {
            try parseASRServerCLI(arguments: ["--model", "/models/qwen", "--port", "wat"])
        }
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

        #expect(formatSRT(subtitles).contains("00:00:00,000 --> 00:00:00,800"))
        #expect(formatVTT(subtitles).hasPrefix("WEBVTT"))
        #expect(formatLRC(subtitles).contains("[00:00.00]hello world"))
        #expect(normalizeLanguageCode(" English ") == "en")
        #expect(normalizeLanguageCode("pt_br") == "pt-BR")

        let payload = formatSubtitleJSON(
            transcript: "hello world again",
            language: "English",
            duration: 2.0,
            subtitles: subtitles
        )
        let json = try #require(try JSONSerialization.jsonObject(with: payload) as? [String: Any])
        #expect(json["language"] as? String == "en")
        #expect((json["segments"] as? [[String: Any]])?.count == 2)
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
        #expect(json["streaming_model"] as? String == "streaming-model")
        #expect(json["batch_model"] as? String == "batch-model")
        #expect(json["status"] as? String == "ready")
        #expect(json["aligner"] as? Bool == false)
        #expect(json["vad"] as? Bool == false)
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

    func create() -> String {
        createdSessionID
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
