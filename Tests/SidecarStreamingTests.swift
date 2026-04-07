import Foundation
import Testing
@testable import Yuwp

/// Integration tests that feed audio through the sidecar HTTP API.
/// Uses synthetic tone/silence fixtures — no real voice recordings.
///
/// Requires the sidecar running on localhost:9748.
/// Run with: swift test --filter "Sidecar Streaming"
@Suite("Sidecar Streaming", .tags(.integration), .disabled("Requires running sidecar — run explicitly"))
struct SidecarStreamingTests {
    let baseURL = "http://127.0.0.1:9748/v1/audio/transcriptions/stream"
    let chunkBytes = 64000  // 2s of 16kHz s16le mono

    // MARK: - Helpers

    private func loadFixturePCM(_ name: String) throws -> Data {
        let fixtureDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures")
        let wav = try Data(contentsOf: fixtureDir.appendingPathComponent(name))
        return wav.dropFirst(44)
    }

    private func streamFixture(_ name: String) async throws -> StreamResult {
        let pcm = try loadFixturePCM(name)

        // Create session
        var createReq = URLRequest(url: URL(string: baseURL)!)
        createReq.httpMethod = "POST"
        createReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        createReq.httpBody = Data("{}".utf8)
        let (createData, _) = try await URLSession.shared.data(for: createReq)
        let json = try JSONSerialization.jsonObject(with: createData) as! [String: Any]
        let sid = json["session_id"] as! String

        // Feed chunks
        var partials: [String] = []
        var offset = 0
        while offset < pcm.count {
            let end = min(offset + chunkBytes, pcm.count)
            let chunk = pcm[offset..<end]
            offset = end

            var feedReq = URLRequest(url: URL(string: "\(baseURL)/\(sid)")!)
            feedReq.httpMethod = "POST"
            feedReq.httpBody = Data(chunk)
            let (feedData, _) = try await URLSession.shared.data(for: feedReq)
            if let feedJson = try? JSONSerialization.jsonObject(with: feedData) as? [String: Any],
               let text = feedJson["text"] as? String {
                partials.append(text)
            }
        }

        // Stop
        var stopReq = URLRequest(url: URL(string: "\(baseURL)/\(sid)")!)
        stopReq.httpMethod = "DELETE"
        let (stopData, _) = try await URLSession.shared.data(for: stopReq)
        let stopJson = try JSONSerialization.jsonObject(with: stopData) as! [String: Any]
        let final = stopJson["text"] as? String ?? ""

        return StreamResult(partials: partials, final: final)
    }

    // MARK: - Tests

    @Test func toneProducesPartials() async throws {
        let result = try await streamFixture("tone-medium.wav")
        // 15s of audio should produce multiple feed responses
        #expect(result.partials.count >= 3,
                Comment(rawValue: "Expected 3+ partials for 15s, got \(result.partials.count)"))
    }

    @Test func silenceProducesEmptyOrMinimalText() async throws {
        let result = try await streamFixture("silence.wav")
        // Pure silence should not produce meaningful text
        #expect(result.final.count < 20,
                Comment(rawValue: "Silence produced too much text: \(result.final)"))
    }

    @Test func sessionLifecycleWorks() async throws {
        // Create
        var createReq = URLRequest(url: URL(string: baseURL)!)
        createReq.httpMethod = "POST"
        createReq.setValue("application/json", forHTTPHeaderField: "Content-Type")
        createReq.httpBody = Data("{}".utf8)
        let (createData, createResp) = try await URLSession.shared.data(for: createReq)
        let http = createResp as! HTTPURLResponse
        #expect(http.statusCode == 200)
        let json = try JSONSerialization.jsonObject(with: createData) as! [String: Any]
        let sid = json["session_id"] as! String
        #expect(!sid.isEmpty)

        // Feed
        var feedReq = URLRequest(url: URL(string: "\(baseURL)/\(sid)")!)
        feedReq.httpMethod = "POST"
        feedReq.httpBody = Data(repeating: 0, count: 3200) // 0.1s silence
        let (_, feedResp) = try await URLSession.shared.data(for: feedReq)
        #expect((feedResp as! HTTPURLResponse).statusCode == 200)

        // Stop
        var stopReq = URLRequest(url: URL(string: "\(baseURL)/\(sid)")!)
        stopReq.httpMethod = "DELETE"
        let (_, stopResp) = try await URLSession.shared.data(for: stopReq)
        #expect((stopResp as! HTTPURLResponse).statusCode == 200)

        // Feed after stop → 404
        var ghostReq = URLRequest(url: URL(string: "\(baseURL)/\(sid)")!)
        ghostReq.httpMethod = "POST"
        ghostReq.httpBody = Data(repeating: 0, count: 100)
        let (_, ghostResp) = try await URLSession.shared.data(for: ghostReq)
        #expect((ghostResp as! HTTPURLResponse).statusCode == 404)
    }

    struct StreamResult {
        let partials: [String]
        let final: String
    }
}

extension Tag {
    @Tag static var integration: Self
}
