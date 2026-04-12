import Foundation
import Testing
@testable import Yuwp

@Suite("Transcript client semantics")
struct TranscriptUpdateTests {

    @Test func partialUsesFullTextWhenNoCommittedPrefixExists() {
        let state = TranscriptState.empty.applying(
            TranscriptUpdate(kind: .partial, text: "hello world")
        )

        #expect(state.committedText == "")
        #expect(state.activeText == "hello world")
        #expect(state.fullText == "hello world")
    }

    @Test func partialKeepsCommittedPrefixAndDerivesActiveTail() {
        let previous = TranscriptState(committedText: "Hello world.", activeText: "")
        let state = previous.applying(
            TranscriptUpdate(kind: .partial, text: "Hello world. testing now")
        )

        #expect(state.committedText == "Hello world.")
        #expect(state.activeText == "testing now")
        #expect(state.fullText == "Hello world. testing now")
    }

    @Test func segmentCommitPromotesFullTextToCommittedAndClearsActiveTail() {
        let previous = TranscriptState(committedText: "Hello world.", activeText: "testing")
        let state = previous.applying(
            TranscriptUpdate(kind: .segmentCommit, text: "Hello world. Testing.")
        )

        #expect(state.committedText == "Hello world. Testing.")
        #expect(state.activeText.isEmpty)
        #expect(state.fullText == "Hello world. Testing.")
    }

    @Test func finalPromotesEverythingToCommittedAndClearsActiveTail() {
        let previous = TranscriptState(committedText: "Hello world.", activeText: "testing")
        let state = previous.applying(
            TranscriptUpdate(kind: .final, text: "Hello world. Testing complete.")
        )

        #expect(state.committedText == "Hello world. Testing complete.")
        #expect(state.activeText.isEmpty)
    }

    @Test func explicitCommittedAndActiveFieldsOverrideDerivedSplit() {
        let previous = TranscriptState(committedText: "old", activeText: "tail")
        let state = previous.applying(
            TranscriptUpdate(
                kind: .partial,
                text: "committed active",
                committedText: "committed",
                activeText: "active"
            )
        )

        #expect(state.committedText == "committed")
        #expect(state.activeText == "active")
        #expect(state.fullText == "committed active")
    }

    @Test func finalIgnoresExplicitActiveTailToAvoidDuplication() {
        let state = TranscriptState.empty.applying(
            TranscriptUpdate(
                kind: .final,
                text: "more delightful",
                committedText: "more delightful",
                activeText: "more delightful"
            )
        )

        #expect(state.committedText == "more delightful")
        #expect(state.activeText.isEmpty)
        #expect(state.fullText == "more delightful")
    }

    @Test func kindDefaultsAreSane() {
        #expect(TranscriptUpdateKind.partial.settlesPreviewImmediately == false)
        #expect(TranscriptUpdateKind.segmentCommit.settlesPreviewImmediately)
        #expect(TranscriptUpdateKind.final.settlesPreviewImmediately)

        #expect(TranscriptUpdateKind.partial.commitsTargetText == false)
        #expect(TranscriptUpdateKind.segmentCommit.commitsTargetText == false)
        #expect(TranscriptUpdateKind.final.commitsTargetText)
    }
}
