import Foundation
import Testing
@testable import Yuwp

@Suite("NativeASRSession response parsing")
struct NativeASRProviderTests {
    @Test func explicitUpdateKindAndSplitFieldsArePreserved() {
        let update = NativeASRSession.parseTranscriptUpdate(
            [
                "text": "Hello world testing",
                "update_kind": "partial",
                "committed_text": "Hello world",
                "active_text": "testing",
            ],
            fallbackKind: .final
        )

        #expect(update == TranscriptUpdate(
            kind: .partial,
            text: "Hello world testing",
            committedText: "Hello world",
            activeText: "testing"
        ))
    }

    @Test func batchCorrectedFallbackMapsPartialToSegmentCommit() {
        let update = NativeASRSession.parseTranscriptUpdate(
            [
                "text": "Hello world.",
                "batch_corrected": true,
            ],
            fallbackKind: .partial
        )

        #expect(update?.kind == .segmentCommit)
        #expect(update?.text == "Hello world.")
    }

    @Test func invalidUpdateKindFallsBackGracefully() {
        let update = NativeASRSession.parseTranscriptUpdate(
            [
                "text": "final words",
                "update_kind": "chaos_mode",
            ],
            fallbackKind: .final
        )

        #expect(update?.kind == .final)
        #expect(update?.text == "final words")
    }

    @Test func missingTextReturnsNil() {
        let update = NativeASRSession.parseTranscriptUpdate(
            ["update_kind": "partial"],
            fallbackKind: .partial
        )

        #expect(update == nil)
    }
}
