import Foundation
import Testing
@testable import Yuwp

@Suite("RecordingArtifactWriter")
struct RecordingArtifactWriterTests {
    @Test func writesAudioMetadataAndMixedLanguageTranscriptSidecars() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("yuwp-recording-artifact-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let context = RecordingArtifactContext(
            sessionID: "abc123",
            transcriptionModel: "mlx-community/Qwen3-ASR-1.7B-bf16",
            dictationLanguageMode: .mixed,
            languageHint: nil
        )
        let date = Date(timeIntervalSince1970: 1_776_895_200)

        let handle = try RecordingArtifactWriter.writeRecording(
            pcmData: Data([1, 2, 3, 4]),
            directory: dir,
            context: context,
            date: date
        )

        #expect(FileManager.default.fileExists(atPath: handle.audioURL.path))
        #expect(FileManager.default.fileExists(atPath: handle.metadataURL.path))
        #expect(!FileManager.default.fileExists(atPath: handle.transcriptURL.path))

        var metadata = try loadJSON(handle.metadataURL)
        #expect(metadata["audioFile"] as? String == handle.audioURL.lastPathComponent)
        #expect(metadata["transcriptFile"] as? String == handle.transcriptURL.lastPathComponent)
        #expect(metadata["sessionID"] as? String == "abc123")
        #expect(metadata["dictationLanguageMode"] as? String == "mixed")
        #expect(metadata["transcript"] is NSNull)

        let transcript = "hello 世界 this is mixed language"
        try RecordingArtifactWriter.writeTranscript(transcript, for: handle, updatedAt: date.addingTimeInterval(3))

        #expect(try String(contentsOf: handle.transcriptURL, encoding: .utf8) == transcript + "\n")
        metadata = try loadJSON(handle.metadataURL)
        #expect(metadata["transcript"] as? String == transcript)
    }

    @Test func avoidsOverwritingArtifactsInTheSameSecond() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("yuwp-recording-artifact-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let context = RecordingArtifactContext(
            sessionID: nil,
            transcriptionModel: "model",
            dictationLanguageMode: .fixed,
            languageHint: "English"
        )
        let date = Date(timeIntervalSince1970: 1_776_895_200)

        let first = try RecordingArtifactWriter.writeRecording(
            pcmData: Data([1, 2]),
            directory: dir,
            context: context,
            date: date
        )
        let second = try RecordingArtifactWriter.writeRecording(
            pcmData: Data([3, 4]),
            directory: dir,
            context: context,
            date: date
        )

        #expect(first.audioURL.lastPathComponent != second.audioURL.lastPathComponent)
        #expect(second.audioURL.deletingPathExtension().lastPathComponent.hasSuffix("-2"))
    }

    private func loadJSON(_ url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }
}
