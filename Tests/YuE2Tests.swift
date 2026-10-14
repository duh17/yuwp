import Foundation
import Testing
@testable import NativeMusic

@Suite("YuE2 protocol")
struct YuE2ProtocolTests {
    @Test func acousticChunksUseHalfTheContextLeftAfterThePrefix() throws {
        // (24576 - 2950 - 3) / 2 = 10811 frames per chunk.
        #expect(try YuE2Protocol.chunkRanges(frames: 6683, prefixTokens: 2950) == [0 ..< 6683])
        #expect(try YuE2Protocol.chunkRanges(frames: 10812, prefixTokens: 2950) == [0 ..< 10811, 10811 ..< 10812])
        #expect(throws: YuE2Error.self) { try YuE2Protocol.chunkRanges(frames: 0, prefixTokens: 10) }
        #expect(throws: YuE2Error.self) { try YuE2Protocol.chunkRanges(frames: 10, prefixTokens: 24574) }
    }

    @Test func phaseLocalTokensMapBackToNativeIDs() {
        #expect(YuE2Protocol.nativeToken(local: 0, phase: .semantic) == YuE2Protocol.musicEnd)
        #expect(YuE2Protocol.nativeToken(local: 1, phase: .semantic) == YuE2Protocol.codecOffset)
        #expect(YuE2Protocol.nativeToken(local: YuE2Protocol.eod, phase: .abc) == YuE2Protocol.abcEnd)
        #expect(YuE2Protocol.nativeToken(local: 42, phase: .abc) == 42)
    }

    @Test func requestDefaultsGuidanceByModeAndRejectsScoresWithoutPlanning() throws {
        #expect(YuE2Request(style: "s", lyrics: "l", cot: .off).guidance == 1.01)
        #expect(YuE2Request(style: "s", lyrics: "l", cot: .full).guidance == 1)
        #expect(throws: YuE2Error.self) { try YuE2Request(style: "s", lyrics: "l", cot: .off, abc: "X:1").validate() }
        #expect(throws: YuE2Error.self) { try YuE2Request(style: "s", lyrics: "l", seed: UInt64(1) << 63).validate() }
    }
}

/// Reference ids from the official tiktoken-based YuE2TextTokenizer. Needs `qwen.tiktoken`:
/// `YUE2_MODEL=<mlx-Yue2-3B dir> swift test --filter YuE2Tokenizer`.
@Suite("YuE2 tokenizer", .enabled(if: ProcessInfo.processInfo.environment["YUE2_MODEL"] != nil, "set YUE2_MODEL to a model dir"))
struct YuE2TokenizerTests {
    let tokenizer: YuE2Tokenizer

    init() throws {
        let directory = URL(fileURLWithPath: ProcessInfo.processInfo.environment["YUE2_MODEL"] ?? "")
        tokenizer = try YuE2Tokenizer(url: directory.appendingPathComponent("qwen.tiktoken"))
    }

    @Test(arguments: [
        ("Hello world's  test\n\n[Verse]\n雁门关外，朔风如刀。",
         [9707, 1879, 594, 220, 1273, 271, 58, 68043, 921, 107181, 64689, 29256, 47815, 3837, 111838, 99208, 29524, 101953, 1773]),
        ("X:1\nT:\nM:4/4\nL:1/32\n|f'8z4f'4e'8c'8|",
         [55, 25, 16, 198, 51, 510, 44, 25, 19, 14, 19, 198, 43, 25, 16, 14, 18, 17, 198, 91, 69, 6, 23, 89, 19, 69, 6, 19, 68, 6, 23, 66, 6, 23, 91]),
        ("  spaces   and\ttabs 1234 ünïcödé\r\n",
         [220, 12621, 256, 323, 3244, 3435, 220, 16, 17, 18, 19, 10489, 77, 37572, 66, 2956, 128505, 319]),
    ])
    func encodesLikeTheReferenceAndRoundTrips(text: String, ids: [Int]) {
        #expect(tokenizer.encode(text) == ids)
        #expect(tokenizer.decode(ids) == text)
    }
}
