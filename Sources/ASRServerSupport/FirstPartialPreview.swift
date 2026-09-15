import Foundation
import NativeASR

/// Display-only speculation. It cannot advance canonical audio or commit text.
/// Owned by one ManagedStreamingSession and accessed under its operation gate.
struct FirstPartialPreview: Sendable {
    static let sampleCount = ASRAudio.sampleRate * 9 / 10
    static let maxNewTokens = 12
    private static let minimumHeadroomSamples = ASRAudio.sampleRate * 3 / 10
    private static let minimumSpeechSec = 0.25

    private var inspectedWindow = false
    private var attemptedDecode = false
    private var canonicalHasText = false
    private var text = ""

    mutating func reserveInspection(pendingSamples: Int, canonicalChunkSamples: Int) -> Bool {
        guard !canonicalHasText, !attemptedDecode, !inspectedWindow,
              pendingSamples >= Self.sampleCount,
              pendingSamples <= canonicalChunkSamples - Self.minimumHeadroomSamples
        else { return false }
        inspectedWindow = true
        return true
    }

    mutating func reserveDecode(speechHint: SpeechActivityHint) -> Bool {
        guard inspectedWindow, !canonicalHasText, !attemptedDecode,
              speechHint.hasSpeech, speechHint.speechDurationSec.isFinite,
              speechHint.speechDurationSec >= Self.minimumSpeechSec
        else { return false }
        attemptedDecode = true
        return true
    }

    mutating func accept(text: String) {
        guard attemptedDecode, !canonicalHasText else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.contains(where: { $0.isLetter || $0.isNumber }) else { return }
        self.text = trimmed
    }

    mutating func canonicalChunkProcessed(text: String) {
        inspectedWindow = false
        if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            canonicalHasText = true
            self.text = ""
        }
    }

    func visibleText(canonicalText: String, isFinal: Bool) -> String {
        if isFinal || canonicalHasText || !canonicalText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return canonicalText
        }
        return text
    }
}
