import AppKit
import CoreGraphics

/// Injects text into any focused app by simulating keyboard events.
///
/// Works for terminals and other apps that don't expose AX text editing.
/// Streams via a character-diff: backspaces removed chars, types new suffix.
@MainActor
final class CGEventInjector: TextInjecting {

    // MARK: - TextInjecting

    private(set) var targetPosition: NSPoint
    var isLiveInjecting: Bool { true }

    func captureTarget() {
        // No-op — screen point captured at init time by TextInjectorFactory.
    }

    /// Stream partial text via diff: backspace removed chars, type new suffix.
    func inject(_ text: String) {
        let (backspaces, suffix) = CGEventInjector.diff(old: writtenText, new: text)
        if backspaces > 0 { sendBackspaces(count: backspaces) }
        if !suffix.isEmpty { postKeyboardEvents(suffix) }
        writtenText = text
    }

    /// Commit the final text: replace any streamed partial with the corrected version.
    func commit(_ text: String) {
        if writtenText != text {
            sendBackspaces(count: writtenText.count)
            postKeyboardEvents(text)
            yuwpLog("CGEvent commit: replaced \(writtenText.count) chars with \(text.count) chars")
        } else {
            yuwpLog("CGEvent commit: text unchanged (\(text.count) chars)")
        }
        writtenText = ""
    }

    func release() {
        writtenText = ""
    }

    // MARK: - Init

    init(screenPoint: NSPoint = .zero) {
        self.targetPosition = screenPoint
    }

    // MARK: - Diff (pure, testable)

    /// Compute the edit needed to replace `old` with `new`.
    /// Returns (backspaces to delete, new suffix to type).
    nonisolated static func diff(old: String, new: String) -> (backspaces: Int, suffix: String) {
        let commonLen = zip(old, new).prefix(while: { $0 == $1 }).count
        let backspaces = old.count - commonLen
        let suffix = String(new.dropFirst(commonLen))
        return (backspaces, suffix)
    }

    // MARK: - Private

    private var writtenText = ""

    /// Post text as synthetic keyboard events using Unicode strings.
    private func postKeyboardEvents(_ text: String) {
        guard !text.isEmpty else { return }

        // CGEvent Unicode string is capped per event — 20 UTF-16 units is reliable.
        let maxChunkUTF16 = 20
        let utf16 = Array(text.utf16)
        var offset = 0

        while offset < utf16.count {
            let end = min(offset + maxChunkUTF16, utf16.count)
            var chunk = Array(utf16[offset..<end])
            let len = chunk.count

            if let down = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: true) {
                down.keyboardSetUnicodeString(stringLength: len, unicodeString: &chunk)
                down.post(tap: .cgSessionEventTap)
            }
            if let up = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: false) {
                up.post(tap: .cgSessionEventTap)
            }
            offset = end
        }
    }

    /// Send backspace key events to erase characters from the target.
    private func sendBackspaces(count: Int) {
        let backspaceKeyCode: CGKeyCode = 51
        for _ in 0..<count {
            if let down = CGEvent(keyboardEventSource: nil, virtualKey: backspaceKeyCode, keyDown: true) {
                down.post(tap: .cgSessionEventTap)
            }
            if let up = CGEvent(keyboardEventSource: nil, virtualKey: backspaceKeyCode, keyDown: false) {
                up.post(tap: .cgSessionEventTap)
            }
        }
    }
}
