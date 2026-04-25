import Foundation
import Testing
@testable import NativeASR

@Suite("SileroVAD")
struct SileroVADTests {
    private func fixtureURL(_ name: String) -> URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("fixtures")
            .appendingPathComponent(name)
    }

    private func loadFixtureAudio(_ name: String) throws -> [Float] {
        try loadAudioFile(fixtureURL(name))
    }

    @Test func initializesAndProcessesSilence() throws {
        let vad = try SileroVAD()
        let silence = [Float](repeating: 0, count: SileroVAD.chunkSize)
        let probability = try vad.process(silence)
        #expect(probability >= 0)
        #expect(probability <= 1)
    }

    @Test func rejectsWrongChunkSize() throws {
        let vad = try SileroVAD()
        #expect(throws: SileroVADError.self) {
            _ = try vad.process([Float](repeating: 0, count: 128))
        }
    }

    @Test func shortAudioStaysSingleChunk() throws {
        let vad = try SileroVAD()
        let audio = [Float](repeating: 0, count: SileroVAD.sampleRate * 20)
        let chunks = try vad.chunk(audio: audio)
        #expect(chunks.count == 1)
        #expect(abs(chunks[0].duration - 20.0) < 0.01)
    }

    @Test func resetKeepsModelUsable() throws {
        let vad = try SileroVAD()
        let noise = (0 ..< SileroVAD.chunkSize).map { _ in Float.random(in: -0.01 ... 0.01) }
        _ = try vad.process(noise)
        vad.reset()
        let probability = try vad.process(noise)
        #expect(probability >= 0)
        #expect(probability <= 1)
    }

    @Test func longAudioSplitsNearLongSilenceGap() throws {
        let vad = try SileroVAD()
        let speech = Array(try loadFixtureAudio("jfk.wav").prefix(SileroVAD.sampleRate * 4))
        let silence = [Float](repeating: 0, count: SileroVAD.sampleRate * 3)
        let audio = speech + silence + speech

        let chunks = try vad.chunk(audio: audio, config: VADChunkingConfig(
            threshold: 0.5,
            minSpeechDuration: 0.25,
            minSilenceDuration: 0.1,
            speechPad: 0.03,
            splitMinSilenceDuration: 1.0,
            maxChunkDuration: 8.0,
            minChunkDuration: 1.0
        ))

        #expect(chunks.count >= 2,
                Comment(rawValue: "Expected silence-guided chunking, got \(chunks.count) chunk(s)"))
        let boundaries = chunks.dropLast().map(\.endTime)
        #expect(boundaries.contains(where: { $0 > 4.5 && $0 < 6.5 }),
                Comment(rawValue: "Expected a boundary near the long silence gap, got boundaries: \(boundaries)"))
        for i in 1 ..< chunks.count {
            #expect(abs(chunks[i].startTime - chunks[i - 1].endTime) < 0.001)
        }
    }

    @Test func shortAudioAlsoSplitsNearLongSilenceGapWhenUnderMaxChunkDuration() throws {
        let vad = try SileroVAD()
        let speech = Array(try loadFixtureAudio("jfk.wav").prefix(SileroVAD.sampleRate * 4))
        let silence = [Float](repeating: 0, count: SileroVAD.sampleRate * 3)
        let audio = speech + silence + speech

        let chunks = try vad.chunk(audio: audio, config: VADChunkingConfig(
            threshold: 0.5,
            minSpeechDuration: 0.25,
            minSilenceDuration: 0.1,
            speechPad: 0.03,
            splitMinSilenceDuration: 1.0,
            maxChunkDuration: 120.0,
            minChunkDuration: 1.0
        ))

        #expect(chunks.count >= 2,
                Comment(rawValue: "Expected silence-guided chunking under the max duration, got \(chunks.count) chunk(s)"))
        let boundaries = chunks.dropLast().map(\.endTime)
        #expect(boundaries.contains(where: { $0 > 4.5 && $0 < 6.5 }),
                Comment(rawValue: "Expected a boundary near the long silence gap, got boundaries: \(boundaries)"))
    }

    @Test func energyChunkingSplitsNearSilenceAroundTargetBoundary() throws {
        let speech = Array(try loadFixtureAudio("jfk.wav").prefix(SileroVAD.sampleRate * 4))
        let speechBed = Array(repeating: speech, count: 29).flatMap { $0 }
        let silence = [Float](repeating: 0, count: SileroVAD.sampleRate * 4)
        let tail = Array(try loadFixtureAudio("jfk.wav").prefix(SileroVAD.sampleRate * 10))
        let audio = speechBed + silence + tail

        let chunks = chunkAudioByEnergy(audio, sampleRate: SileroVAD.sampleRate, config: EnergyChunkingConfig(
            maxChunkDuration: 120.0,
            minChunkDuration: 1.0,
            searchExpandDuration: 5.0,
            energyWindowDuration: 0.1,
            minProgressDuration: 1.0
        ))

        #expect(chunks.count == 2,
                Comment(rawValue: "Expected two chunks from low-energy split, got \(chunks.count)"))
        let boundary = try #require(chunks.first?.endTime)
        #expect(boundary > 114.0 && boundary < 123.0,
                Comment(rawValue: "Expected low-energy boundary near the silence gap, got \(boundary)"))
    }
}
