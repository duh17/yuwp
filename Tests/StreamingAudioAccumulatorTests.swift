import Testing
@testable import ASRServerSupport

@Suite("Streaming audio accumulator")
struct StreamingAudioAccumulatorTests {
    @Test(arguments: [1, 2, 3, 7, 16, 31])
    func preservesSamplesAcrossFeedPartitions(feedSize: Int) {
        let samples = (0..<257).map(Float.init)
        var accumulator = StreamingAudioAccumulator(compactionThreshold: 32)
        var emitted: [Float] = []
        var offset = 0
        var chunkSize = 5

        while offset < samples.count {
            let end = min(offset + feedSize, samples.count)
            accumulator.append(contentsOf: samples[offset..<end])
            while let chunk = accumulator.takePrefix(chunkSize) {
                emitted.append(contentsOf: chunk)
                chunkSize = 7
            }
            offset = end
        }

        emitted.append(contentsOf: accumulator.drain())

        #expect(emitted == samples)
        #expect(accumulator.isEmpty)
    }

    @Test func emitsExactBoundaryWithoutLeavingATail() {
        var accumulator = StreamingAudioAccumulator()
        accumulator.append(contentsOf: [1, 2, 3, 4])

        #expect(accumulator.takePrefix(4) == [1, 2, 3, 4])
        #expect(accumulator.count == 0)
        #expect(accumulator.takePrefix(1) == nil)
    }

    @Test func keepsIncompleteTailForDrain() {
        var accumulator = StreamingAudioAccumulator()
        accumulator.append(contentsOf: [1, 2, 3])

        #expect(accumulator.takePrefix(4) == nil)
        #expect(accumulator.drain() == [1, 2, 3])
        #expect(accumulator.drain().isEmpty)
    }

    @Test func compactionCopiesLessThanLegacyFrontRemoval() {
        let feedSize = 1_600
        let steadyChunkSize = 28_000
        let totalSamples = 160_000
        var accumulator = StreamingAudioAccumulator(compactionThreshold: 32_000)
        var legacyPendingCount = 0
        var legacyCopiedSamples = 0

        for _ in stride(from: 0, to: totalSamples, by: feedSize) {
            accumulator.append(contentsOf: repeatElement(0, count: feedSize))
            legacyPendingCount += feedSize

            while accumulator.count >= steadyChunkSize {
                _ = accumulator.takePrefix(steadyChunkSize)
                legacyPendingCount -= steadyChunkSize
                legacyCopiedSamples += legacyPendingCount
            }
        }

        #expect(accumulator.count == legacyPendingCount)
        #expect(accumulator.compactedSampleCount * 4 <= legacyCopiedSamples * 3)
    }
}
