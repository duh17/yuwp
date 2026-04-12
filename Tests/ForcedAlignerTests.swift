import Foundation
import Testing
@testable import NativeASR

// MARK: - AlignmentProcessor unit tests (no model, no server)

@Suite("AlignmentProcessor")
struct AlignmentProcessorTests {

    // MARK: - Word Tokenization

    @Test func englishTokenization() {
        let words = AlignmentProcessor.tokenizeWords("hello world", language: "English")
        #expect(words == ["hello", "world"])
    }

    @Test func englishStripsNonKeptChars() {
        let words = AlignmentProcessor.tokenizeWords("Hello, World! It's fine.", language: "English")
        #expect(words == ["Hello", "World", "It's", "fine"])
    }

    @Test func chineseCharacterLevel() {
        let words = AlignmentProcessor.tokenizeWords("你好世界", language: "Chinese")
        #expect(words == ["你", "好", "世", "界"])
    }

    @Test func chineseMixedWithLatin() {
        let words = AlignmentProcessor.tokenizeWords("说hello世界", language: "Chinese")
        #expect(words == ["说", "hello", "世", "界"])
    }

    @Test func englishWithEmbeddedCJK() {
        // CJK characters embedded in English text get split out
        let words = AlignmentProcessor.tokenizeWords("test你好end", language: "English")
        #expect(words == ["test", "你", "好", "end"])
    }

    @Test func chineseDisplayJoinRemovesInsertedSpaces() {
        let text = AlignedTextRenderer.render(tokens: ["照", "片", "上", "这", "三", "位", "年", "轻"])
        #expect(text == "照片上这三位年轻")
    }

    @Test func chineseDisplayJoinKeepsLatinWordsReadable() {
        let text = AlignedTextRenderer.render(tokens: ["说", "hello", "world", "世", "界"])
        #expect(text == "说hello world世界")
    }

    @Test func englishDisplayJoinKeepsWordSpaces() {
        let text = AlignedTextRenderer.render(tokens: ["hello", "world"])
        #expect(text == "hello world")
    }

    @Test func chineseDisplayJoinCollapsesAsciiAcronyms() {
        let text = AlignedTextRenderer.render(tokens: ["中", "国", "A", "P", "P", "在", "中", "国"])
        #expect(text == "中国APP在中国")
    }

    @Test func emptyTextReturnsEmpty() {
        #expect(AlignmentProcessor.tokenizeWords("", language: "English").isEmpty)
        #expect(AlignmentProcessor.tokenizeWords("   ", language: "English").isEmpty)
    }

    @Test func punctuationOnlyReturnsEmpty() {
        #expect(AlignmentProcessor.tokenizeWords("...", language: "English").isEmpty)
        #expect(AlignmentProcessor.tokenizeWords("!@#$%", language: "English").isEmpty)
    }

    @Test func numbersAreKept() {
        let words = AlignmentProcessor.tokenizeWords("test123 456", language: "English")
        #expect(words == ["test123", "456"])
    }

    // MARK: - Timestamp Fixing (LIS)

    @Test func monotoneTimestampsUnchanged() {
        let input = [100, 200, 300, 400, 500]
        #expect(AlignmentProcessor.fixTimestamps(input) == input)
    }

    @Test func singleAnomalyFixed() {
        // 300 breaks monotonicity (between 400 and 500)
        let input = [100, 200, 400, 300, 500]
        let fixed = AlignmentProcessor.fixTimestamps(input)
        // Must be monotonically non-decreasing
        for i in 1 ..< fixed.count {
            #expect(fixed[i] >= fixed[i - 1],
                    Comment(rawValue: "Not monotone at \(i): \(fixed)"))
        }
    }

    @Test func multipleAnomaliesInterpolated() {
        // Large anomaly group (>2) triggers linear interpolation
        let input = [100, 500, 400, 300, 200, 600]
        let fixed = AlignmentProcessor.fixTimestamps(input)
        for i in 1 ..< fixed.count {
            #expect(fixed[i] >= fixed[i - 1],
                    Comment(rawValue: "Not monotone at \(i): \(fixed)"))
        }
        // First and last should be preserved (they're in the LIS)
        #expect(fixed.first == 100)
        #expect(fixed.last == 600)
    }

    @Test func emptyTimestamps() {
        #expect(AlignmentProcessor.fixTimestamps([]).isEmpty)
    }

    @Test func singleTimestamp() {
        #expect(AlignmentProcessor.fixTimestamps([42]) == [42])
    }

    @Test func allSameValues() {
        let input = [100, 100, 100, 100]
        let fixed = AlignmentProcessor.fixTimestamps(input)
        #expect(fixed == input)
    }

    // MARK: - CJK Detection

    @Test func detectsCJKCharacters() {
        #expect(AlignmentProcessor.isCJK(Unicode.Scalar(0x4E00)!))  // CJK Unified start
        #expect(AlignmentProcessor.isCJK(Unicode.Scalar(0x9FFF)!))  // CJK Unified end
        #expect(!AlignmentProcessor.isCJK(Unicode.Scalar("A")))
        #expect(!AlignmentProcessor.isCJK(Unicode.Scalar("0")))
    }

    @Test func keptCharacterClassification() {
        #expect(AlignmentProcessor.isKeptChar("a"))
        #expect(AlignmentProcessor.isKeptChar("Z"))
        #expect(AlignmentProcessor.isKeptChar("5"))
        #expect(AlignmentProcessor.isKeptChar("'"))
        #expect(!AlignmentProcessor.isKeptChar(","))
        #expect(!AlignmentProcessor.isKeptChar("!"))
        #expect(!AlignmentProcessor.isKeptChar(" "))
    }
}

// MARK: - ForcedAligner integration tests (require model)

@Suite("ForcedAligner", .tags(.integration),
       .enabled(if: ProcessInfo.processInfo.environment["ALIGNER_TEST"] != nil,
               "Set ALIGNER_TEST=<model-dir> to run"))
struct ForcedAlignerIntegrationTests {

    let aligner: ForcedAligner

    init() throws {
        guard let modelDir = ProcessInfo.processInfo.environment["ALIGNER_TEST"] else {
            throw XCTSkip("ALIGNER_TEST not set")
        }
        aligner = try ForcedAligner.load(from: URL(fileURLWithPath: modelDir))
    }

    private func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures")
            .appendingPathComponent(name)
    }

    private func loadFixtureAudio(_ name: String) throws -> [Float] {
        try loadAudioFile(fixtureURL(name))
    }

    @Test func alignEnglishProducesMonotonicTimestamps() throws {
        let audio = try loadFixtureAudio("asr_en.wav")
        let transcript = "mhm oh yeah yeah he wasnt even that big when i started listening to him"
        let items = aligner.align(audio: audio, text: transcript)

        #expect(!items.isEmpty)

        // All timestamps must be monotonically non-decreasing
        for i in 1 ..< items.count {
            #expect(items[i].startTime >= items[i - 1].startTime,
                    Comment(rawValue: "Start times not monotone at word \(i): \(items[i-1].text)→\(items[i].text)"))
        }

        // End times >= start times
        for item in items {
            #expect(item.endTime >= item.startTime,
                    Comment(rawValue: "End < start for '\(item.text)': \(item.startTime) → \(item.endTime)"))
        }
    }

    @Test func alignChineseProducesCharacterTimestamps() throws {
        let audio = try loadFixtureAudio("asr_zh.wav")
        let transcript = "甚至出现交易几乎停滞的情况"
        let items = aligner.align(audio: audio, text: transcript, language: "Chinese")

        // Chinese should produce one item per character
        #expect(items.count == 13)

        // Timestamps within audio duration
        let audioDuration = Double(audio.count) / 16000.0
        for item in items {
            #expect(item.endTime <= audioDuration + 1.0,
                    Comment(rawValue: "Timestamp beyond audio for '\(item.text)': \(item.endTime) > \(audioDuration)"))
        }
    }

    @Test func emptyTextReturnsNoItems() throws {
        let audio = try loadFixtureAudio("asr_en.wav")
        let items = aligner.align(audio: audio, text: "")
        #expect(items.isEmpty)
    }
}

// MARK: - Subtitle server endpoint integration tests

@Suite("Subtitle Endpoint", .tags(.integration),
       .enabled(if: ProcessInfo.processInfo.environment["ASR_TEST"] != nil,
               "Set ASR_TEST=1 with asr-server running (canonical: --model <dir>; optional --aligner-model) on :9748"))
struct SubtitleEndpointTests {
    let host: String
    let port: String

    var subtitleURL: String { "http://\(host):\(port)/v1/audio/subtitles" }

    init() {
        self.host = ProcessInfo.processInfo.environment["ASR_TEST_HOST"] ?? "127.0.0.1"
        self.port = ProcessInfo.processInfo.environment["ASR_TEST_PORT"] ?? "9748"
    }

    private func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures")
            .appendingPathComponent(name)
    }

    private func loadFixtureData(_ name: String) throws -> Data {
        try Data(contentsOf: fixtureURL(name))
    }

    private func http(
        _ method: String,
        _ url: String,
        body: Data? = nil,
        headers: [String: String] = [:]
    ) async throws -> (Data, Int) {
        var req = URLRequest(url: URL(string: url)!)
        req.httpMethod = method
        req.timeoutInterval = 30
        req.httpBody = body
        for (key, value) in headers {
            req.setValue(value, forHTTPHeaderField: key)
        }
        let (data, response) = try await URLSession.shared.data(for: req)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        return (data, status)
    }

    private func subtitleBody(
        fileName: String,
        fileData: Data,
        text: String,
        language: String = "English",
        responseFormat: String = "srt"
    ) -> (Data, String) {
        let boundary = "Boundary-\(UUID().uuidString)"
        var body = Data()
        func append(_ s: String) { body.append(Data(s.utf8)) }

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"file\"; filename=\"\(fileName)\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(fileData)
        append("\r\n")

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"text\"\r\n\r\n")
        append("\(text)\r\n")

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"language\"\r\n\r\n")
        append("\(language)\r\n")

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"response_format\"\r\n\r\n")
        append("\(responseFormat)\r\n")

        append("--\(boundary)--\r\n")
        return (body, "multipart/form-data; boundary=\(boundary)")
    }

    @Test func subtitleSRTFormat() async throws {
        let wav = try loadFixtureData("asr_en.wav")
        let transcript = "mhm oh yeah yeah he wasnt even that big"
        let (body, ct) = subtitleBody(fileName: "test.wav", fileData: wav, text: transcript, responseFormat: "srt")
        let (data, status) = try await http("POST", subtitleURL, body: body, headers: ["Content-Type": ct])

        #expect(status == 200)
        let srt = String(data: data, encoding: .utf8) ?? ""
        #expect(srt.contains("-->"), Comment(rawValue: "SRT should contain timestamps: \(srt)"))
        #expect(srt.contains("1\n"), Comment(rawValue: "SRT should have subtitle index"))
    }

    @Test func subtitleVTTFormat() async throws {
        let wav = try loadFixtureData("asr_en.wav")
        let transcript = "mhm oh yeah yeah he wasnt even that big"
        let (body, ct) = subtitleBody(fileName: "test.wav", fileData: wav, text: transcript, responseFormat: "vtt")
        let (data, status) = try await http("POST", subtitleURL, body: body, headers: ["Content-Type": ct])

        #expect(status == 200)
        let vtt = String(data: data, encoding: .utf8) ?? ""
        #expect(vtt.hasPrefix("WEBVTT"), Comment(rawValue: "VTT should start with WEBVTT header"))
        #expect(vtt.contains("-->"))
    }

    @Test func subtitleJSONFormat() async throws {
        let wav = try loadFixtureData("asr_en.wav")
        let transcript = "mhm oh yeah yeah he wasnt even that big"
        let (body, ct) = subtitleBody(fileName: "test.wav", fileData: wav, text: transcript, responseFormat: "json")
        let (data, status) = try await http("POST", subtitleURL, body: body, headers: ["Content-Type": ct])

        #expect(status == 200)
        let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(payload != nil, Comment(rawValue: "JSON response should be an object"))
        #expect(payload?["text"] as? String == transcript)
        #expect(payload?["language"] as? String == "en")
        #expect(payload?["duration"] is Double)
        let segments = payload?["segments"] as? [[String: Any]]
        #expect(segments != nil, Comment(rawValue: "JSON response should include segments"))
        #expect((segments ?? []).count > 0, Comment(rawValue: "Should have at least one subtitle segment"))
        if let first = segments?.first {
            #expect(first["start"] is Double)
            #expect(first["end"] is Double)
            #expect(first["text"] is String)
        }
    }

    @Test func subtitleWithoutTextAutoTranscribes() async throws {
        let wav = try loadFixtureData("asr_en.wav")
        let boundary = "Boundary-test"
        var body = Data()
        func append(_ s: String) { body.append(Data(s.utf8)) }
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"file\"; filename=\"test.wav\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(wav)
        append("\r\n")
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"language\"\r\n\r\n")
        append("English\r\n")
        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"response_format\"\r\n\r\n")
        append("json\r\n")
        append("--\(boundary)--\r\n")

        let (data, status) = try await http("POST", subtitleURL, body: body,
            headers: ["Content-Type": "multipart/form-data; boundary=\(boundary)"])
        #expect(status == 200)
        let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(payload != nil)
        #expect(payload?["text"] is String)
        #expect(payload?["language"] as? String == "en")
        #expect(payload?["duration"] is Double)
        let segments = payload?["segments"] as? [[String: Any]]
        #expect(segments != nil)
        #expect((segments ?? []).count > 0)
    }

    @Test func subtitleChineseAlignment() async throws {
        let wav = try loadFixtureData("asr_zh.wav")
        let transcript = "甚至出现交易几乎停滞的情况"
        let (body, ct) = subtitleBody(fileName: "test.wav", fileData: wav, text: transcript,
                                       language: "Chinese", responseFormat: "json")
        let (data, status) = try await http("POST", subtitleURL, body: body, headers: ["Content-Type": ct])

        #expect(status == 200)
        let payload = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        #expect(payload != nil)
        #expect(payload?["text"] as? String == transcript)
        #expect(payload?["language"] as? String == "zh")
        let segments = payload?["segments"] as? [[String: Any]]
        #expect(segments != nil)
        #expect((segments ?? []).count > 0)
        // Check that all subtitle texts contain Chinese characters without inserted ASCII spaces.
        for sub in segments ?? [] {
            let text = sub["text"] as? String ?? ""
            #expect(!text.isEmpty)
            #expect(!text.contains(" "))
        }
    }
}

private struct XCTSkip: Error {
    let message: String
    init(_ message: String) { self.message = message }
}
