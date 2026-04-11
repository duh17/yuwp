import AppKit
import CoreGraphics
import Foundation

/// Injects text into any focused app by simulating keyboard events.
///
/// Works for terminals and other apps that don't expose AX text editing.
/// Streams via a character-diff: backspaces removed chars, types new suffix.
@MainActor
final class CGEventInjector: TextInjecting {

    // MARK: - TextInjecting

    let surfaceMode: DictationSurfaceMode = .terminal
    private(set) var targetPosition: NSPoint

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
    ///
    /// Terminals are noticeably less reliable with large multi-character Unicode
    /// payloads per event. Post one grapheme at a time with a tiny pacing delay
    /// so the focused app has a chance to consume each key event.
    private func postKeyboardEvents(_ text: String) {
        guard !text.isEmpty else { return }

        for character in text {
            var utf16 = Array(String(character).utf16)
            let len = utf16.count

            if let down = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: true) {
                down.keyboardSetUnicodeString(stringLength: len, unicodeString: &utf16)
                down.post(tap: .cgSessionEventTap)
            }
            if let up = CGEvent(keyboardEventSource: nil, virtualKey: 0, keyDown: false) {
                up.keyboardSetUnicodeString(stringLength: len, unicodeString: &utf16)
                up.post(tap: .cgSessionEventTap)
            }

            Thread.sleep(forTimeInterval: Self.interKeyDelay)
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
            Thread.sleep(forTimeInterval: Self.interKeyDelay)
        }
    }

    private static let interKeyDelay: TimeInterval = 0.0015
}
