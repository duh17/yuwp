import Foundation
import Testing
@testable import Yuwp
@testable import NativeASR

/// Integration tests for the native ASR server (asr-server).
/// Uses public-domain test fixtures — no personal voice recordings.
///
/// Requires asr-server running on localhost:9748 (or port set via ASR_TEST_PORT env).
/// Start the server before running:
///   .build/arm64-apple-macosx/release/asr-server <model-dir> --port 9748
///
/// Run with: ASR_TEST=1 swift test --filter "ASR Server"
@Suite("ASR Server", .tags(.integration),
       .enabled(if: ProcessInfo.processInfo.environment["ASR_TEST"] != nil,
               "Set ASR_TEST=1 with asr-server running on :9748"))
struct ASRServerTests {
    let host: String
    let port: String
    let chunkBytes = 64_000  // 2s of 16kHz s16le mono (32000 samples × 2 bytes)

    var baseURL: String { "http://\(host):\(port)/v1/audio/transcriptions/stream" }
    var batchURL: String { "http://\(host):\(port)/v1/audio/transcriptions" }
    var infoURL: String { "http://\(host):\(port)/v1/info" }

    init() {
        self.host = ProcessInfo.processInfo.environment["ASR_TEST_HOST"] ?? "127.0.0.1"
        self.port = ProcessInfo.processInfo.environment["ASR_TEST_PORT"] ?? "9748"
    }

    // MARK: - Helpers

    private func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures")
            .appendingPathComponent(name)
    }

    /// Load a WAV fixture and strip the 44-byte header to get raw s16le PCM.
    private func loadFixturePCM(_ name: String) throws -> Data {
        let wav = try Data(contentsOf: fixtureURL(name))
        return Data(wav.dropFirst(44))
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

    private func multipartBody(fileName: String, fileData: Data, responseFormat: String? = nil) -> (Data, String) {
        let boundary = "Boundary-\(UUID().uuidString)"
        var body = Data()

        func append(_ string: String) {
            body.append(Data(string.utf8))
        }

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"file\"; filename=\"\(fileName)\"\r\n")
        append("Content-Type: audio/wav\r\n\r\n")
        body.append(fileData)
        append("\r\n")

        append("--\(boundary)\r\n")
        append("Content-Disposition: form-data; name=\"model\"\r\n\r\n")
        append("gpt-4o-mini-transcribe\r\n")

        if let responseFormat {
            append("--\(boundary)\r\n")
            append("Content-Disposition: form-data; name=\"response_format\"\r\n\r\n")
            append("\(responseFormat)\r\n")
        }

        append("--\(boundary)--\r\n")
        return (body, "multipart/form-data; boundary=\(boundary)")
    }

    private func json(_ data: Data) -> [String: Any]? {
        try? JSONSerialization.jsonObject(with: data) as? [String: Any]
    }

    /// Create a session and stream a fixture through it, returning partials + final text.
    private func streamFixture(_ name: String) async throws -> StreamResult {
        let pcm = try loadFixturePCM(name)

        // Create session
        let (createData, createStatus) = try await http("POST", baseURL)
        #expect(createStatus == 200)
        let sid = json(createData)?["session_id"] as! String

        // Feed in 2s chunks
        var partials: [String] = []
        var offset = 0
        while offset < pcm.count {
            let end = min(offset + chunkBytes, pcm.count)
            let chunk = Data(pcm[offset..<end])
            offset = end

            let (feedData, feedStatus) = try await http("POST", "\(baseURL)/\(sid)", body: chunk)
            #expect(feedStatus == 200)
            if let text = json(feedData)?["text"] as? String {
                partials.append(text)
            }
        }

        // Stop session
        let (stopData, stopStatus) = try await http("DELETE", "\(baseURL)/\(sid)")
        #expect(stopStatus == 200)
        let finalText = json(stopData)?["text"] as? String ?? ""

        return StreamResult(partials: partials, final: finalText)
    }

    struct StreamResult {
        let partials: [String]
        let final: String
    }

    // MARK: - Server Info

    @Test func serverReportsReady() async throws {
        let (data, status) = try await http("GET", infoURL)
        #expect(status == 200)
        let info = json(data)
        #expect(info?["status"] as? String == "ready")
        #expect(info?["sample_rate"] as? Int == 16000)
    }

    // MARK: - Session Lifecycle

    @Test func createFeedStopWorks() async throws {
        // Create
        let (createData, createStatus) = try await http("POST", baseURL)
        #expect(createStatus == 200)
        let sid = json(createData)?["session_id"] as? String
        #expect(sid != nil)
        #expect(!sid!.isEmpty)

        // Feed 0.1s of silence
        let silence = Data(repeating: 0, count: 3200)
        let (_, feedStatus) = try await http("POST", "\(baseURL)/\(sid!)", body: silence)
        #expect(feedStatus == 200)

        // Stop
        let (stopData, stopStatus) = try await http("DELETE", "\(baseURL)/\(sid!)")
        #expect(stopStatus == 200)
        #expect(json(stopData)?["text"] is String)

        // Feed after stop → 404
        let (_, ghostStatus) = try await http("POST", "\(baseURL)/\(sid!)", body: silence)
        #expect(ghostStatus == 404)
    }

    // MARK: - Transcription Quality

    @Test func jfkProducesProgressivePartials() async throws {
        let result = try await streamFixture("jfk.wav")

        // 11s of speech should produce multiple non-empty partials
        let nonEmpty = result.partials.filter { !$0.isEmpty }
        #expect(nonEmpty.count >= 2,
                Comment(rawValue: "Expected 2+ non-empty partials for 11s, got \(nonEmpty.count)"))

        // Text should accumulate (later partials ≥ earlier ones)
        if nonEmpty.count >= 2 {
            #expect(nonEmpty.last!.count >= nonEmpty.first!.count,
                    Comment(rawValue: "Expected growing partials"))
        }

        // Final text should contain known JFK inaugural address content
        let lower = result.final.lowercased()
        #expect(lower.contains("country") || lower.contains("fellow") || lower.contains("ask"),
                Comment(rawValue: "Expected JFK content, got: \(result.final)"))
    }

    @Test func silenceProducesMinimalText() async throws {
        let result = try await streamFixture("silence.wav")
        #expect(result.final.count < 20,
                Comment(rawValue: "Silence produced too much text: '\(result.final)'"))
    }

    @Test func batchEndpointAcceptsMultipartUpload() async throws {
        let wav = try loadFixtureData("jfk.wav")
        let (body, contentType) = multipartBody(fileName: "jfk.wav", fileData: wav)
        let (data, status) = try await http("POST", batchURL, body: body, headers: ["Content-Type": contentType])
        #expect(status == 200)
        let text = json(data)?["text"] as? String ?? ""
        let lower = text.lowercased()
        #expect(lower.contains("country") || lower.contains("fellow") || lower.contains("ask"),
                Comment(rawValue: "Expected JFK content, got: \(text)"))
    }

    @Test func batchEndpointSupportsTextResponseFormat() async throws {
        let wav = try loadFixtureData("jfk.wav")
        let (body, contentType) = multipartBody(fileName: "jfk.wav", fileData: wav, responseFormat: "text")
        let (data, status) = try await http("POST", batchURL, body: body, headers: ["Content-Type": contentType])
        #expect(status == 200)
        let text = String(data: data, encoding: .utf8) ?? ""
        let lower = text.lowercased()
        #expect(lower.contains("country") || lower.contains("fellow") || lower.contains("ask"),
                Comment(rawValue: "Expected plain-text JFK content, got: \(text)"))
    }

    @Test func chineseFixtureProducesText() async throws {
        let result = try await streamFixture("asr_zh.wav")
        #expect(!result.final.isEmpty,
                Comment(rawValue: "Expected non-empty Chinese transcript"))
        let hasHan = result.final.unicodeScalars.contains { scalar in
            (0x4E00...0x9FFF).contains(scalar.value)
        }
        #expect(hasHan,
                Comment(rawValue: "Expected Han characters in transcript, got: \(result.final)"))
    }

    // MARK: - Concurrent Sessions

    @Test func concurrentSessionsAreIsolated() async throws {
        let jfk = try loadFixturePCM("jfk.wav")
        let silence = try loadFixturePCM("silence.wav")

        // Create two sessions
        let (createA, _) = try await http("POST", baseURL)
        let (createB, _) = try await http("POST", baseURL)
        let sidA = json(createA)?["session_id"] as! String
        let sidB = json(createB)?["session_id"] as! String

        // Interleave feeds — one chunk of each per round
        var offsetA = 0, offsetB = 0
        while offsetA < jfk.count || offsetB < silence.count {
            if offsetA < jfk.count {
                let end = min(offsetA + chunkBytes, jfk.count)
                let (_, s) = try await http("POST", "\(baseURL)/\(sidA)", body: Data(jfk[offsetA..<end]))
                #expect(s == 200)
                offsetA = end
            }
            if offsetB < silence.count {
                let end = min(offsetB + chunkBytes, silence.count)
                let (_, s) = try await http("POST", "\(baseURL)/\(sidB)", body: Data(silence[offsetB..<end]))
                #expect(s == 200)
                offsetB = end
            }
        }

        // Stop both
        let (stopA, _) = try await http("DELETE", "\(baseURL)/\(sidA)")
        let (stopB, _) = try await http("DELETE", "\(baseURL)/\(sidB)")
        let textA = json(stopA)?["text"] as? String ?? ""
        let textB = json(stopB)?["text"] as? String ?? ""

        // JFK should have real content, silence should be empty/minimal
        #expect(textA.count > textB.count,
                Comment(rawValue: "JFK (\(textA.count) chars) should have more text than silence (\(textB.count) chars)"))
        #expect(textA.lowercased().contains("country") || textA.lowercased().contains("fellow"),
                Comment(rawValue: "JFK session should contain expected keywords, got: \(textA)"))
    }

    // MARK: - Error Handling

    @Test func returnsProperErrorCodes() async throws {
        // Unknown session → 404
        let (_, notFound) = try await http("POST", "\(baseURL)/nonexistent", body: Data())
        #expect(notFound == 404)

        // Wrong method on /v1/info → 405
        let (_, methodInfo) = try await http("POST", infoURL)
        #expect(methodInfo == 405)

        // Wrong method on batch endpoint → 405
        let (_, methodBatch) = try await http("GET", batchURL)
        #expect(methodBatch == 405)

        // Wrong method on session → 405
        let (createData, _) = try await http("POST", baseURL)
        let sid = json(createData)?["session_id"] as! String
        let (_, methodSession) = try await http("PUT", "\(baseURL)/\(sid)")
        #expect(methodSession == 405)
        _ = try await http("DELETE", "\(baseURL)/\(sid)")  // cleanup
    }

    @Test func deleteNonexistentSession() async throws {
        let (_, status) = try await http("DELETE", "\(baseURL)/doesnotexist")
        #expect(status == 404)
    }

    // MARK: - Segment Commit
    //
    // The streaming session uses pause-triggered batch retranscribe to "commit"
    // a segment: after enough consecutive silent chunks, the active segment is
    // batch-retranscribed and frozen into a committed prefix that will never be
    // rewritten by later partials. These tests verify the contract:
    //
    //   1. Speech + ≥4s silence triggers a commit (response carries `batch_corrected: true`)
    //   2. Once committed, the prefix appears in every subsequent partial unchanged
    //   3. After a commit, new speech is appended (committedText + " " + activeText)
    //   4. Stop only finalizes the active segment — committed prefix is preserved verbatim

    /// 8s of zeroed s16le PCM. We use 4 chunks (not 2) so the test is robust
    /// against misalignment with the speech audio: even if the first silent
    /// chunk is "mixed" (last bit of speech + first bit of silence and
    /// classified as speech by RMS), we still get 3 fully silent chunks
    /// — well past the pauseChunks=2 commit threshold.
    private var silenceBufferBytes: Data { Data(repeating: 0, count: chunkBytes * 4) }

    /// Stream a sequence of audio buffers as 2s chunks. Returns every feed
    /// response (text + batch_corrected flag) plus the final text from stop.
    private func streamComposite(_ buffers: [Data]) async throws -> CompositeResult {
        var pcm = Data()
        for b in buffers { pcm.append(b) }

        let (createData, createStatus) = try await http("POST", baseURL)
        #expect(createStatus == 200)
        let sid = json(createData)?["session_id"] as! String

        var feeds: [(text: String, batchCorrected: Bool)] = []
        var offset = 0
        while offset < pcm.count {
            let end = min(offset + chunkBytes, pcm.count)
            let chunk = Data(pcm[offset..<end])
            offset = end
            let (feedData, feedStatus) = try await http("POST", "\(baseURL)/\(sid)", body: chunk)
            #expect(feedStatus == 200)
            let payload = json(feedData) ?? [:]
            let text = payload["text"] as? String ?? ""
            let bc = payload["batch_corrected"] as? Bool ?? false
            feeds.append((text, bc))
        }

        let (stopData, stopStatus) = try await http("DELETE", "\(baseURL)/\(sid)")
        #expect(stopStatus == 200)
        let finalText = json(stopData)?["text"] as? String ?? ""
        return CompositeResult(feeds: feeds, final: finalText)
    }

    struct CompositeResult {
        let feeds: [(text: String, batchCorrected: Bool)]
        let final: String

        var firstCommitFeedIndex: Int? {
            feeds.firstIndex(where: { $0.batchCorrected })
        }
    }

    /// Long silence after speech triggers a segment commit, signaled by
    /// `batch_corrected: true` in the feed response.
    @Test func pauseAfterSpeechTriggersSegmentCommit() async throws {
        let jfk = try loadFixturePCM("jfk.wav")
        let result = try await streamComposite([jfk, silenceBufferBytes])

        let commitIdx = result.firstCommitFeedIndex
        #expect(commitIdx != nil,
                Comment(rawValue: "Expected at least one feed with batch_corrected=true after silence, got: \(result.feeds.map { $0.batchCorrected })"))

        // Committed text should contain JFK content
        if let idx = commitIdx {
            let committedText = result.feeds[idx].text
            let lower = committedText.lowercased()
            #expect(lower.contains("country") || lower.contains("fellow") || lower.contains("ask"),
                    Comment(rawValue: "Committed segment should contain JFK content, got: \(committedText)"))
        }
    }

    /// Once committed, the prefix appears in every subsequent partial unchanged.
    @Test func committedPrefixIsImmutable() async throws {
        let jfk = try loadFixturePCM("jfk.wav")
        // jfk → 4s silence → jfk again → 4s silence
        let result = try await streamComposite([jfk, silenceBufferBytes, jfk, silenceBufferBytes])

        guard let firstCommit = result.firstCommitFeedIndex else {
            Issue.record("Expected at least one segment commit")
            return
        }
        let committedPrefix = result.feeds[firstCommit].text
        #expect(!committedPrefix.isEmpty,
                Comment(rawValue: "First committed segment should not be empty"))

        // Every feed AFTER the first commit must start with the committed prefix.
        // This is the immutability contract.
        for i in (firstCommit + 1)..<result.feeds.count {
            let later = result.feeds[i].text
            #expect(later.hasPrefix(committedPrefix),
                    Comment(rawValue: "Feed[\(i)] should start with committed prefix.\n  prefix: \(committedPrefix)\n  later:  \(later)"))
        }
        // Final text must also preserve the committed prefix
        #expect(result.final.hasPrefix(committedPrefix),
                Comment(rawValue: "Final text should start with committed prefix.\n  prefix: \(committedPrefix)\n  final:  \(result.final)"))
    }

    /// After a commit, new speech is appended to the committed text rather than
    /// replacing it. The final text should contain JFK content twice (once
    /// committed, once from the second active segment).
    @Test func newSpeechAppendsAfterCommit() async throws {
        let jfk = try loadFixturePCM("jfk.wav")
        let result = try await streamComposite([jfk, silenceBufferBytes, jfk])

        // Final text should be longer than a single JFK transcript — it
        // contains the committed first JFK + the second JFK (committed at stop)
        let singleResult = try await streamFixture("jfk.wav")
        let singleLen = singleResult.final.count

        #expect(result.final.count > singleLen,
                Comment(rawValue: "Final text after two JFK segments should be longer than one. single=\(singleLen) double=\(result.final.count)"))

        // Should contain JFK content (basic sanity)
        let lower = result.final.lowercased()
        #expect(lower.contains("country") || lower.contains("fellow"),
                Comment(rawValue: "Expected JFK content in composite result, got: \(result.final)"))
    }

    /// If the user stops with a very short trailing silence (no new speech
    /// after the last commit), the final text should equal the committed
    /// prefix — no hallucinated content from batching silence.
    @Test func stopAfterCommitDoesntRewriteCommittedText() async throws {
        let jfk = try loadFixturePCM("jfk.wav")
        // jfk → 8s silence (commits) → 2s more silence → stop
        let extraSilence = Data(repeating: 0, count: chunkBytes)  // 2s
        let result = try await streamComposite([jfk, silenceBufferBytes, extraSilence])

        guard let commitIdx = result.firstCommitFeedIndex else {
            Issue.record("Expected segment commit during silence")
            return
        }
        let committedText = result.feeds[commitIdx].text

        // Final text should equal the committed text — the trailing silence
        // adds nothing and must not corrupt the committed prefix
        #expect(result.final == committedText,
                Comment(rawValue: "Final text should equal committed text after pure-silence trailing segment.\n  committed: \(committedText)\n  final:     \(result.final)"))
    }

    /// Multiple commits chain correctly. Three JFK segments separated by
    /// silences should produce at least two `batch_corrected: true` feeds
    /// during streaming (one per inter-segment pause), and the final text
    /// should be ~3× a single JFK transcript length.
    @Test func multipleCommitsChainCorrectly() async throws {
        let jfk = try loadFixturePCM("jfk.wav")
        let single = try await streamFixture("jfk.wav")
        let singleLen = single.final.count

        // jfk → silence → jfk → silence → jfk → stop
        let result = try await streamComposite([
            jfk, silenceBufferBytes,
            jfk, silenceBufferBytes,
            jfk,
        ])

        // Count the number of commit events during streaming. We expect 2
        // (after the first and second silences). The third jfk is committed
        // implicitly on stop via finalize().
        let commitCount = result.feeds.filter { $0.batchCorrected }.count
        #expect(commitCount >= 2,
                Comment(rawValue: "Expected at least 2 streaming commits for three segments with two silences, got \(commitCount). feeds: \(result.feeds.map { $0.batchCorrected })"))

        // Final text should be substantially longer than a single JFK —
        // close to 3× (with some tolerance for segment boundary whitespace)
        #expect(result.final.count >= Int(Double(singleLen) * 2.5),
                Comment(rawValue: "Final text should be ~3× single JFK length. single=\(singleLen) triple=\(result.final.count)"))

        // Every commit event during streaming must produce a text that the
        // final still starts with — i.e., no commit was ever rewritten
        for (idx, feed) in result.feeds.enumerated() where feed.batchCorrected {
            #expect(result.final.hasPrefix(feed.text),
                    Comment(rawValue: "Commit[\(idx)] prefix not preserved in final.\n  commit: \(feed.text)\n  final:  \(result.final)"))
        }
    }

    /// After a commit, streaming is fully functional in the new segment —
    /// not just passively waiting for another pause. Feed jfk, let it commit,
    /// then feed jfk again and verify the SECOND jfk's content also appears
    /// in the final text. Guards against regressions where `resetActiveSegment`
    /// breaks the encoder/KV cache in a way that degrades subsequent segments.
    @Test func secondSegmentIsFullyTranscribedAfterCommit() async throws {
        let jfk = try loadFixturePCM("jfk.wav")
        let single = try await streamFixture("jfk.wav")
        let singleLen = single.final.count

        // jfk → silence (commits first jfk) → jfk → stop (commits second jfk)
        let result = try await streamComposite([jfk, silenceBufferBytes, jfk])

        // Final text should contain roughly double the single-jfk content.
        // If the second segment was broken (e.g., encoder cache regression),
        // final would be approximately singleLen.
        #expect(result.final.count >= Int(Double(singleLen) * 1.8),
                Comment(rawValue: "Second segment should contribute ~equal content to first. single=\(singleLen) final=\(result.final.count)\n  final: \(result.final)"))

        // Final should contain JFK keyword content
        let lower = result.final.lowercased()
        let keywordCount = ["country", "fellow", "ask"].reduce(0) { acc, kw in
            // Count occurrences of each keyword
            acc + lower.components(separatedBy: kw).count - 1
        }
        #expect(keywordCount >= 2,
                Comment(rawValue: "Expected at least 2 JFK keyword occurrences across two segments, got \(keywordCount). final: \(result.final)"))
    }

    /// If the active segment has audio but not enough to batch (<1s), the
    /// committed prefix must still be preserved exactly. Covers the branch
    /// where `finalize()` skips batch due to the audioBuffer length guard.
    @Test func shortActiveSegmentPreservesCommittedPrefix() async throws {
        let jfk = try loadFixturePCM("jfk.wav")
        // jfk → silence (commits) → 0.5s of jfk start (< 1s, can't batch) → stop
        // 0.5s = 16000 bytes of s16le PCM
        let shortAudio = Data(jfk.prefix(16000))
        let result = try await streamComposite([jfk, silenceBufferBytes, shortAudio])

        guard let commitIdx = result.firstCommitFeedIndex else {
            Issue.record("Expected segment commit during silence")
            return
        }
        let committedText = result.feeds[commitIdx].text

        // Final must start with the committed prefix verbatim. The short
        // active segment may or may not add a few characters via streaming
        // fallback — what matters is the committed text is not corrupted.
        #expect(result.final.hasPrefix(committedText),
                Comment(rawValue: "Committed prefix must be preserved when active segment is too short to batch.\n  committed: \(committedText)\n  final:     \(result.final)"))
    }
}

// MARK: - StreamingSession unit tests (no server)

@Suite("StreamingSession helpers")
struct StreamingSessionUnitTests {
    /// `appendSegment` is a static helper that joins committed and active text
    /// with a single space, handling empties without producing double-spaces.
    @Test func appendSegmentJoinsCleanly() {
        // Empty committed → returns trimmed segment
        #expect(StreamingSession.appendSegment("", "hello") == "hello")
        #expect(StreamingSession.appendSegment("", "  hello  ") == "hello")

        // Empty segment → returns committed unchanged
        #expect(StreamingSession.appendSegment("hello", "") == "hello")
        #expect(StreamingSession.appendSegment("hello", "   ") == "hello")

        // Both non-empty → single space join
        #expect(StreamingSession.appendSegment("hello", "world") == "hello world")
        #expect(StreamingSession.appendSegment("first segment.", "second one.") == "first segment. second one.")

        // Trims segment whitespace before joining
        #expect(StreamingSession.appendSegment("a", "  b  ") == "a b")

        // Both empty
        #expect(StreamingSession.appendSegment("", "") == "")
    }
}

extension Tag {
    @Tag static var integration: Self
}
