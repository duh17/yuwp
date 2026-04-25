import XCTest
@testable import NativeTTS

final class LongFormTTSChunkerTests: XCTestCase {
    func testShortTextStaysSingleChunk() {
        let text = "Short answer. Still one chunk."
        let chunks = LongFormTTSChunker.chunk(text)
        XCTAssertEqual(chunks, [text])
    }

    func testLongTextSplitsOnSentenceBoundaries() {
        let text = [
            "This is the first sentence and it is deliberately a little longer than usual.",
            "Here is the second sentence, which should stay attached to its own natural boundary.",
            "This third sentence pushes the total over the target chunk size so a split should happen here.",
            "Finally, the fourth sentence should land in a later chunk instead of being forced into the first one."
        ].joined(separator: " ")

        let chunks = LongFormTTSChunker.chunk(text, targetCharacters: 140, hardCharacterLimit: 180)
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertTrue(chunks.allSatisfy { !$0.isEmpty })
        XCTAssertTrue(chunks.dropLast().allSatisfy { ".!?".contains($0.last ?? " ") })
        XCTAssertTrue(chunks.allSatisfy { $0.count <= 180 })
        XCTAssertEqual(chunks.joined(separator: " "), text)
    }

    func testVeryLongSentenceFallsBackToWordSplits() {
        let words = Array(repeating: "stability", count: 40)
        let text = words.joined(separator: " ")
        let chunks = LongFormTTSChunker.chunk(text, targetCharacters: 60, hardCharacterLimit: 80)

        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertTrue(chunks.allSatisfy { $0.count <= 80 })
        XCTAssertEqual(chunks.joined(separator: " "), text)
    }

    func testOversizedWordStillRespectsHardLimit() {
        let text = String(repeating: "a", count: 205)
        let chunks = LongFormTTSChunker.chunk(text, targetCharacters: 60, hardCharacterLimit: 80)

        XCTAssertEqual(chunks.map(\.count), [80, 80, 45])
        XCTAssertEqual(chunks.joined(), text)
    }
}
