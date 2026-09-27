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

    @Test func streamLeftoverCarriesAcrossChunkBoundaries() {
        let chunkSamples = 2560
        let first = [Float](repeating: 0.01, count: chunkSamples)
        let (frames1, leftover1) = SileroVAD.takeFullFrames(leftover: [], samples: first)
        #expect(frames1.count == chunkSamples / SileroVAD.chunkSize)
        #expect(frames1.allSatisfy { $0.count == SileroVAD.chunkSize })
        #expect(leftover1.count == chunkSamples % SileroVAD.chunkSize)
        #expect(leftover1.count == 256)

        let (frames2, leftover2) = SileroVAD.takeFullFrames(leftover: leftover1, samples: first)
        let secondTotal = leftover1.count + chunkSamples
        #expect(frames2.count == secondTotal / SileroVAD.chunkSize)
        #expect(leftover2.count == secondTotal % SileroVAD.chunkSize)
        #expect(leftover2.count == 512)
        #expect(Array(frames2[0].prefix(leftover1.count)) == leftover1)

        let (straddle, remaining) = SileroVAD.takeFullFrames(
            leftover: [Float](repeating: 0.2, count: SileroVAD.chunkSize - 1),
            samples: [0.3]
        )
        #expect(straddle.count == 1)
        #expect(straddle[0].count == SileroVAD.chunkSize)
        #expect(straddle[0].last == 0.3)
        #expect(remaining.isEmpty)

        let (none, short) = SileroVAD.takeFullFrames(leftover: [], samples: [0.1, 0.2])
        #expect(none.isEmpty)
        #expect(short == [0.1, 0.2])
    }

    @Test func processStreamCarriesLeftoverWithoutPaddingPartialFrames() throws {
        let vad = try SileroVAD()
        let chunk = [Float](repeating: 0, count: 2560)
        let (first, mid) = vad.processStream(chunk, state: SileroVAD.StreamState())
        #expect(first.processedFrameCount == 4)
        #expect(mid.leftover.count == 256)

        let (second, end) = vad.processStream(chunk, state: mid)
        #expect(second.processedFrameCount == 4)
        #expect(end.leftover.count == 512)
    }

    @Test func processStreamMatchesChainedProcessAndCarriesLeftoverIntoNextFrame() throws {
        let vad = try SileroVAD()
        let audio = try loadFixtureAudio("jfk.wav")
        let first = Array(audio.prefix(2560))
        let second = Array(audio.dropFirst(2560).prefix(2560))
        #expect(first.count == 2560)
        #expect(second.count == 2560)

        func chained(_ samples: [Float], state: SileroVAD.StreamState) throws -> (
            probabilities: [Float], state: SileroVAD.StreamState
        ) {
            let (frames, leftover) = SileroVAD.takeFullFrames(leftover: state.leftover, samples: samples)
            var current = state
            var probabilities: [Float] = []
            for frame in frames {
                let (probability, next) = try vad.process(frame, state: current)
                probabilities.append(probability)
                current = next
            }
            current.leftover = leftover
            return (probabilities, current)
        }

        let (chainedFirstProbs, chainedFirst) = try chained(first, state: SileroVAD.StreamState())
        let (streamFirst, streamFirstState) = vad.processStream(first, state: SileroVAD.StreamState())
        #expect(streamFirstState == chainedFirst)
        #expect(streamFirst.speechFrameCount == chainedFirstProbs.filter { $0 >= 0.5 }.count)

        let (frames2, _) = SileroVAD.takeFullFrames(
            leftover: streamFirstState.leftover, samples: second
        )
        #expect(!frames2.isEmpty)
        #expect(Array(frames2[0].prefix(streamFirstState.leftover.count)) == streamFirstState.leftover)

        let (chainedSecondProbs, chainedSecond) = try chained(second, state: streamFirstState)
        let (streamSecond, streamSecondState) = vad.processStream(second, state: streamFirstState)
        #expect(streamSecondState == chainedSecond)
        #expect(streamSecond.speechFrameCount == chainedSecondProbs.filter { $0 >= 0.5 }.count)
    }

    @Test func processStreamPadsFinalFrameOnlyWhenEndingWithoutAFullFrame() throws {
        let vad = try SileroVAD()
        let short = [Float](repeating: 0, count: 100)
        let (live, liveState) = vad.processStream(short, state: SileroVAD.StreamState())
        #expect(live.processedFrameCount == 0)
        #expect(liveState.leftover.count == 100)

        let (stop, stopState) = vad.processStream(
            short, state: SileroVAD.StreamState(), padFinalFrame: true
        )
        #expect(stop.processedFrameCount == 1)
        #expect(stopState.leftover.isEmpty)
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
