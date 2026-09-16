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

@Suite("Yuwp build layout")
struct YuwpBuildLayoutTests {
    @Test func developmentASRBinarySearchIncludesSwiftBuildProducts() {
        let root = URL(fileURLWithPath: "/tmp/yuwp-repo", isDirectory: true)
        let paths = YuwpBuildLayout.developmentExecutableCandidates(
            named: "yuwp-asr",
            repositoryRoot: root
        ).map(\.path)

        #expect(paths.contains { $0.hasSuffix(".build/out/Products/Release/yuwp-asr") })
        #expect(paths.contains { $0.hasSuffix(".build/out/Products/Debug/yuwp-asr") })
        #expect(paths.contains { $0.hasSuffix(".build/arm64-apple-macosx/release/yuwp-asr") })
        #expect(paths.contains { $0.hasSuffix(".build/arm64-apple-macosx/debug/yuwp-asr") })
    }
}

@Suite("NativeASRProvider lifecycle")
struct NativeASRProviderLifecycleTests {
    @Test func offModeTransitionsToDisabledSynchronously() async {
        await MainActor.run {
            let provider = NativeASRProvider()
            var states: [ASRServerState] = []

            provider.serverMode = .off
            provider.onStateChange = { states.append($0) }

            provider.start()

            #expect(provider.state == .disabled)
            #expect(states == [.disabled])
        }
    }

    @Test func missingModelSurfacesImmediateErrorAndCallback() async {
        await MainActor.run {
            let provider = NativeASRProvider()
            let missingModel = "missing-model-\(UUID().uuidString)"
            var states: [ASRServerState] = []
            var errors: [String] = []

            provider.transcriptionModel = missingModel
            provider.onStateChange = { states.append($0) }
            provider.onError = { errors.append($0) }

            provider.start()

            #expect(provider.state == .error("Transcription model not found"))
            #expect(states.last == .error("Transcription model not found"))
            #expect(errors == ["Transcription model not found"])
        }
    }
}
