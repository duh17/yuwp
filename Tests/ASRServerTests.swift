import Foundation
import Testing
@testable import Yuwp

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
    var infoURL: String { "http://\(host):\(port)/v1/info" }

    init() {
        self.host = ProcessInfo.processInfo.environment["ASR_TEST_HOST"] ?? "127.0.0.1"
        self.port = ProcessInfo.processInfo.environment["ASR_TEST_PORT"] ?? "9748"
    }

    // MARK: - Helpers

    /// Load a WAV fixture and strip the 44-byte header to get raw s16le PCM.
    private func loadFixturePCM(_ name: String) throws -> Data {
        let fixtureDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures")
        let wav = try Data(contentsOf: fixtureDir.appendingPathComponent(name))
        return Data(wav.dropFirst(44))
    }

    private func http(_ method: String, _ url: String, body: Data? = nil) async throws -> (Data, Int) {
        var req = URLRequest(url: URL(string: url)!)
        req.httpMethod = method
        req.timeoutInterval = 30
        req.httpBody = body
        let (data, response) = try await URLSession.shared.data(for: req)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        return (data, status)
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
}

extension Tag {
    @Tag static var integration: Self
}
