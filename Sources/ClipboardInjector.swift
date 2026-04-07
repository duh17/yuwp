import AppKit
import CoreGraphics

/// Injects text via clipboard paste (Cmd+V).
///
/// Last-resort fallback when neither AX nor CGEvent can be used.
/// No live streaming — text only appears on commit.
@MainActor
final class ClipboardInjector: TextInjecting {

    // MARK: - TextInjecting

    private(set) var targetPosition: NSPoint = .zero
    var isLiveInjecting: Bool { false }

    func captureTarget() {
        // No-op — no element to probe for clipboard paste.
    }

    func inject(_ text: String) {
        // No-op — streaming not supported; text surfaces only on commit.
    }

    func commit(_ text: String) {
        pasteViaClipboard(text)
    }

    func release() {
        // Nothing to clean up.
    }

    // MARK: - Internal (also used by AXTextInjector for final commit)

    func pasteViaClipboard(_ text: String) {
        let pb = NSPasteboard.general

        // Snapshot current clipboard so we can restore it.
        let savedString = pb.string(forType: .string)
        let savedChangeCount = pb.changeCount

        pb.clearContents()
        pb.setString(text, forType: .string)

        let vKey: CGKeyCode = 9
        if let down = CGEvent(keyboardEventSource: nil, virtualKey: vKey, keyDown: true),
           let up = CGEvent(keyboardEventSource: nil, virtualKey: vKey, keyDown: false) {
            down.flags = .maskCommand
            up.flags = .maskCommand
            down.post(tap: .cgSessionEventTap)
            up.post(tap: .cgSessionEventTap)
        }

        // Restore clipboard after the paste event lands.
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            if pb.changeCount == savedChangeCount + 1 {
                pb.clearContents()
                if let saved = savedString {
                    pb.setString(saved, forType: .string)
                }
            }
        }

        yuwpLog("Pasted via clipboard")
    }
}
