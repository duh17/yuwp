import NativeASR
import Testing
@testable import ASRServerSupport

@Suite("First partial preview isolation")
struct FirstPartialPreviewTests {
    @Test func inspectionNeedsAudioAndHeadroomBeforeCanonicalWork() {
        var preview = FirstPartialPreview()
        #expect(preview.reserveInspection(pendingSamples: 14_399, canonicalChunkSamples: 24_000) == false)
        #expect(preview.reserveInspection(pendingSamples: 22_400, canonicalChunkSamples: 24_000) == false)
        #expect(preview.reserveInspection(pendingSamples: 24_000, canonicalChunkSamples: 24_000) == false)
        #expect(preview.reserveInspection(pendingSamples: 14_400, canonicalChunkSamples: 24_000) == true)
        #expect(preview.reserveInspection(pendingSamples: 16_000, canonicalChunkSamples: 24_000) == false)
    }

    @Test func silenceAllowsANewInspectionOnlyAfterCanonicalConsumption() {
        var preview = FirstPartialPreview()
        #expect(preview.reserveInspection(pendingSamples: 14_400, canonicalChunkSamples: 24_000) == true)
        #expect(preview.reserveDecode(speechHint: SpeechActivityHint(hasSpeech: false, speechDurationSec: 0)) == false)
        #expect(preview.reserveInspection(pendingSamples: 17_600, canonicalChunkSamples: 24_000) == false)
        preview.canonicalChunkProcessed(text: "")
        #expect(preview.reserveInspection(pendingSamples: 14_400, canonicalChunkSamples: 24_000) == true)
    }

    @Test func oneSpeechConfirmedDecodeIsTheEntireSessionBudget() {
        var preview = FirstPartialPreview()
        #expect(preview.reserveDecode(speechHint: SpeechActivityHint(hasSpeech: true, speechDurationSec: 0.3)) == false)
        #expect(preview.reserveInspection(pendingSamples: 14_400, canonicalChunkSamples: 24_000) == true)
        #expect(preview.reserveDecode(speechHint: SpeechActivityHint(hasSpeech: true, speechDurationSec: 0.249)) == false)
        #expect(preview.reserveDecode(speechHint: SpeechActivityHint(hasSpeech: true, speechDurationSec: .nan)) == false)
        #expect(preview.reserveDecode(speechHint: SpeechActivityHint(hasSpeech: true, speechDurationSec: 0.252)) == true)
        preview.accept(text: "hello")
        #expect(preview.visibleText(canonicalText: "", isFinal: false) == "hello")
        preview.canonicalChunkProcessed(text: "")
        #expect(preview.reserveInspection(pendingSamples: 14_400, canonicalChunkSamples: 24_000) == false)
        #expect(preview.reserveDecode(speechHint: SpeechActivityHint(hasSpeech: true, speechDurationSec: 0.8)) == false)
    }

    @Test func canonicalTakeoverIsPermanentAndFinalNeverUsesPreview() {
        var preview = FirstPartialPreview()
        _ = preview.reserveInspection(pendingSamples: 14_400, canonicalChunkSamples: 24_000)
        _ = preview.reserveDecode(speechHint: SpeechActivityHint(hasSpeech: true, speechDurationSec: 0.3))
        preview.accept(text: "draft")
        #expect(preview.visibleText(canonicalText: "", isFinal: true).isEmpty)
        #expect(preview.visibleText(canonicalText: "final", isFinal: true) == "final")
        preview.canonicalChunkProcessed(text: "canonical")
        #expect(preview.visibleText(canonicalText: "canonical", isFinal: false) == "canonical")
        preview.canonicalChunkProcessed(text: "")
        preview.accept(text: "stale")
        #expect(preview.visibleText(canonicalText: "", isFinal: false).isEmpty)
    }

    @Test func punctuationCannotWinTheFirstTextClockAndSessionsAreIndependent() {
        var first = FirstPartialPreview()
        _ = first.reserveInspection(pendingSamples: 14_400, canonicalChunkSamples: 24_000)
        _ = first.reserveDecode(speechHint: SpeechActivityHint(hasSpeech: true, speechDurationSec: 0.3))
        first.accept(text: " ... ")
        #expect(first.visibleText(canonicalText: "", isFinal: false).isEmpty)
        first.canonicalChunkProcessed(text: "done")
        var second = FirstPartialPreview()
        #expect(second.reserveInspection(pendingSamples: 14_400, canonicalChunkSamples: 24_000) == true)
    }

    @Test(arguments: [-1, 0, 14_399, 19_201, 24_000, 100_000])
    func unsuitableBufferSizesNeverReservePreview(samples: Int) {
        var preview = FirstPartialPreview()
        #expect(preview.reserveInspection(pendingSamples: samples, canonicalChunkSamples: 24_000) == false)
    }

    @Test func exactHeadroomBoundaryIsInclusive() {
        var preview = FirstPartialPreview()
        #expect(preview.reserveInspection(pendingSamples: 19_200, canonicalChunkSamples: 24_000) == true)
    }

    @Test(arguments: [-0.1, 0, 0.249, Double.infinity, Double.nan])
    func invalidOrInsufficientSpeechNeverSpendsDecoderBudget(duration: Double) {
        var preview = FirstPartialPreview()
        _ = preview.reserveInspection(pendingSamples: 14_400, canonicalChunkSamples: 24_000)
        #expect(preview.reserveDecode(speechHint: SpeechActivityHint(hasSpeech: true, speechDurationSec: duration)) == false)
        preview.accept(text: "must not appear")
        #expect(preview.visibleText(canonicalText: "", isFinal: false).isEmpty)
    }

    @Test func negativeSpeechHintCannotBeOverriddenByItsDuration() {
        var preview = FirstPartialPreview()
        _ = preview.reserveInspection(pendingSamples: 14_400, canonicalChunkSamples: 24_000)
        #expect(preview.reserveDecode(speechHint: SpeechActivityHint(hasSpeech: false, speechDurationSec: 2)) == false)
    }
}
