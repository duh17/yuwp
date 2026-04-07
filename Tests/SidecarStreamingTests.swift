import Foundation
import Testing
@testable import Yuwp

/// Integration tests that feed real audio recordings through the sidecar HTTP API
/// and verify streaming behavior (partial sequence, rollback detection, final quality).
///
/// Requires the sidecar running on localhost:9748. Skipped if unreachable.
/// Run with: swift test --filter "Sidecar Streaming" (requires sidecar on :9748)
@Suite("Sidecar Streaming", .tags(.integration), .disabled("Requires running sidecar — run explicitly"))
struct SidecarStreamingTests {
    let baseURL = "http://127.0.0.1:9748/v1/audio/transcriptions/stream"
    let chunkBytes = 64000  // 2s of 16kHz s16le mono

    // MARK: - Helpers

    private func sidecarAvailable() async -> Bool {
        var req = URLRequest(url: URL(string: baseURL)!, timeoutInterval: 2)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = Data("{}".utf8)
        guard let (data, resp) = try? await URLSession.shared.data(for: req),
              let http = resp as? HTTPURLResponse, http.statusCode == 200 else { return false }
        // Clean up the session we just created
        if let sid = (try? JSONSerialization.jsonObject(with: data) as? [String: Any])?["session_id"] as? String {
            var del = URLRequest(url: URL(string: "\(baseURL)/\(sid)")!)
            del.httpMethod = "DELETE"
            _ = try? await URLSession.shared.data(for: del)
        }
        return true
    }

    private func loadFixturePCM(_ name: String) throws -> Data {
        let fixtureDir = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures")
        let wav = try Data(contentsOf: fixtureDir.appendingPathComponent(name))
        return wav.dropFirst(44) // Strip WAV header
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

        // Feed chunks, collect partials
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
               let text = feedJson["text"] as? String, !text.isEmpty {
                partials.append(text)
            }
        }

        // Stop session
        var stopReq = URLRequest(url: URL(string: "\(baseURL)/\(sid)")!)
        stopReq.httpMethod = "DELETE"
        let (stopData, _) = try await URLSession.shared.data(for: stopReq)
        let stopJson = try JSONSerialization.jsonObject(with: stopData) as! [String: Any]
        let final = stopJson["text"] as? String ?? ""

        return StreamResult(partials: partials, final: final)
    }

    private func findRollbacks(_ partials: [String]) -> Int {
        var count = 0
        for i in 1..<partials.count {
            if !partials[i].hasPrefix(partials[i - 1]) { count += 1 }
        }
        return count
    }

    // MARK: - Tests

    @Test func helloWorksWellProducesText() async throws {
        guard await sidecarAvailable() else { return }
        let result = try await streamFixture("hello-works-well.wav")

        #expect(!result.final.isEmpty, "Should produce non-empty final text")
        let lower = result.final.lowercased()
        #expect(lower.contains("hello") || lower.contains("works"),
                Comment(rawValue: "Expected 'hello' or 'works', got: \(result.final)"))
        #expect(result.partials.count >= 2,
                Comment(rawValue: "Expected 2+ partials for 15s audio, got \(result.partials.count)"))
    }

    @Test func doingAnyTestsProducesText() async throws {
        guard await sidecarAvailable() else { return }
        let result = try await streamFixture("doing-any-tests.wav")

        #expect(!result.final.isEmpty)
        #expect(result.final.lowercased().contains("test"),
                Comment(rawValue: "Expected 'test', got: \(result.final)"))
    }

    @Test func chineseSpeechProducesChineseText() async throws {
        guard await sidecarAvailable() else { return }
        let result = try await streamFixture("chinese-bye.wav")

        #expect(!result.final.isEmpty)
        let hasChinese = result.final.unicodeScalars.contains { $0.value >= 0x4E00 && $0.value <= 0x9FFF }
        #expect(hasChinese, Comment(rawValue: "Expected Chinese chars, got: \(result.final)"))
    }

    @Test func rollbackRateIsAcceptable() async throws {
        guard await sidecarAvailable() else { return }
        let result = try await streamFixture("hello-works-well.wav")
        let rollbacks = findRollbacks(result.partials)
        let rate = result.partials.isEmpty ? 0.0 : Double(rollbacks) / Double(result.partials.count)

        #expect(rate < 0.5,
                Comment(rawValue: "Rollback rate \(Int(rate * 100))%: \(rollbacks)/\(result.partials.count) partials"))
    }

    struct StreamResult {
        let partials: [String]
        let final: String
    }
}

extension Tag {
    @Tag static var integration: Self
}
