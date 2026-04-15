import AppKit
import CoreGraphics

@MainActor
protocol ClipboardPasting {
    func pasteViaClipboard(_ text: String)
}

@MainActor
protocol ClipboardPasteboard {
    var changeCount: Int { get }
    func string(forType type: NSPasteboard.PasteboardType) -> String?
    func clearContents()
    func setString(_ text: String, forType type: NSPasteboard.PasteboardType)
}

@MainActor
final class SystemClipboardPasteboard: ClipboardPasteboard {
    private let pasteboard: NSPasteboard

    init(_ pasteboard: NSPasteboard = .general) {
        self.pasteboard = pasteboard
    }

    var changeCount: Int { pasteboard.changeCount }

    func string(forType type: NSPasteboard.PasteboardType) -> String? {
        pasteboard.string(forType: type)
    }

    func clearContents() {
        pasteboard.clearContents()
    }

    func setString(_ text: String, forType type: NSPasteboard.PasteboardType) {
        pasteboard.setString(text, forType: type)
    }
}

/// Injects text via clipboard (copy-only or Cmd+V paste).
///
/// Last-resort fallback when neither AX nor CGEvent can be used.
/// No live streaming — text only appears on commit.
@MainActor
final class ClipboardInjector: TextInjecting, ClipboardPasting {

    enum CommitMode {
        /// Write dictated text to clipboard and trigger Cmd+V, then restore prior clipboard.
        case pasteAndRestore
        /// Write dictated text to clipboard only. Used when no target was focused
        /// at capture time, so we never lose the dictated text.
        case copyOnly
    }

    // MARK: - TextInjecting

    let surfaceMode: DictationSurfaceMode = .bubbleClipboard
    private(set) var targetPosition: NSPoint

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

    // MARK: - Init

    init(screenPoint: NSPoint = .zero, commitMode: CommitMode = .pasteAndRestore) {
        self.targetPosition = screenPoint
        self.commitMode = commitMode
        self.pasteboard = SystemClipboardPasteboard()
        self.postPasteShortcut = Self.postCommandV
        self.scheduleRestore = { work in
            DispatchQueue.main.asyncAfter(deadline: .now() + Self.clipboardRestoreDelay) {
                work()
            }
        }
    }

    init(
        screenPoint: NSPoint = .zero,
        commitMode: CommitMode = .pasteAndRestore,
        pasteboard: any ClipboardPasteboard,
        postPasteShortcut: @escaping () -> Void,
        scheduleRestore: @escaping (@escaping () -> Void) -> Void
    ) {
        self.targetPosition = screenPoint
        self.commitMode = commitMode
        self.pasteboard = pasteboard
        self.postPasteShortcut = postPasteShortcut
        self.scheduleRestore = scheduleRestore
    }

    // MARK: - Internal (also used by AXTextInjector for final commit)

    func pasteViaClipboard(_ text: String) {
        let savedString = pasteboard.string(forType: .string)
        let savedChangeCount = pasteboard.changeCount

        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)

        guard commitMode == .pasteAndRestore else {
            yuwpLog("Copied dictated text to clipboard (no focused target)")
            return
        }

        postPasteShortcut()

        scheduleRestore { [pasteboard] in
            if pasteboard.changeCount == savedChangeCount + 1 {
                pasteboard.clearContents()
                if let saved = savedString {
                    pasteboard.setString(saved, forType: .string)
                }
            }
        }

        yuwpLog("Pasted via clipboard")
    }

    // MARK: - Private

    private let commitMode: CommitMode
    private let pasteboard: any ClipboardPasteboard
    private let postPasteShortcut: () -> Void
    private let scheduleRestore: (@escaping () -> Void) -> Void

    private static let clipboardRestoreDelay: TimeInterval = 0.45

    private static func postCommandV() {
        let commandKey: CGKeyCode = 55
        let vKey: CGKeyCode = 9

        guard
            let source = CGEventSource(stateID: .combinedSessionState),
            let commandDown = CGEvent(keyboardEventSource: source, virtualKey: commandKey, keyDown: true),
            let vDown = CGEvent(keyboardEventSource: source, virtualKey: vKey, keyDown: true),
            let vUp = CGEvent(keyboardEventSource: source, virtualKey: vKey, keyDown: false),
            let commandUp = CGEvent(keyboardEventSource: source, virtualKey: commandKey, keyDown: false)
        else {
            return
        }

        commandDown.flags = .maskCommand
        vDown.flags = .maskCommand
        vUp.flags = .maskCommand

        commandDown.post(tap: .cgSessionEventTap)
        vDown.post(tap: .cgSessionEventTap)
        vUp.post(tap: .cgSessionEventTap)
        commandUp.post(tap: .cgSessionEventTap)
    }
}
