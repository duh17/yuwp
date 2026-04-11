import AppKit
import Testing
@testable import Yuwp

@Suite("ClipboardInjector")
@MainActor
struct ClipboardInjectorTests {
    @Test func commitWritesPasteboardPostsPasteAndRestoresOriginalClipboard() {
        let pasteboard = FakePasteboard(string: "before", changeCount: 10)
        var pasteShortcutCount = 0
        var scheduledRestore: (() -> Void)?

        let injector = ClipboardInjector(
            screenPoint: NSPoint(x: 10, y: 20),
            pasteboard: pasteboard,
            postPasteShortcut: { pasteShortcutCount += 1 },
            scheduleRestore: { scheduledRestore = $0 }
        )

        #expect(injector.surfaceMode == .bubbleClipboard)
        #expect(injector.targetPosition == NSPoint(x: 10, y: 20))

        injector.commit("dictated text")

        #expect(pasteboard.currentString == "dictated text")
        #expect(pasteShortcutCount == 1)
        #expect(scheduledRestore != nil)

        scheduledRestore?()
        #expect(pasteboard.currentString == "before")
    }

    @Test func restoreDoesNotClobberClipboardIfUserChangedItAfterPaste() {
        let pasteboard = FakePasteboard(string: "before", changeCount: 3)
        var scheduledRestore: (() -> Void)?

        let injector = ClipboardInjector(
            pasteboard: pasteboard,
            postPasteShortcut: {},
            scheduleRestore: { scheduledRestore = $0 }
        )

        injector.commit("dictated text")
        pasteboard.simulateExternalChange(to: "user copied something else")

        scheduledRestore?()
        #expect(pasteboard.currentString == "user copied something else")
    }

    @Test func restoreToEmptyClipboardClearsTemporaryPasteText() {
        let pasteboard = FakePasteboard(string: nil, changeCount: 7)
        var scheduledRestore: (() -> Void)?

        let injector = ClipboardInjector(
            pasteboard: pasteboard,
            postPasteShortcut: {},
            scheduleRestore: { scheduledRestore = $0 }
        )

        injector.commit("dictated text")
        scheduledRestore?()

        #expect(pasteboard.currentString == nil)
    }
}

@MainActor
private final class FakePasteboard: ClipboardPasteboard {
    var currentString: String?
    var changeCount: Int

    init(string: String?, changeCount: Int = 0) {
        self.currentString = string
        self.changeCount = changeCount
    }

    func string(forType type: NSPasteboard.PasteboardType) -> String? {
        currentString
    }

    func clearContents() {
        currentString = nil
        changeCount += 1
    }

    func setString(_ text: String, forType type: NSPasteboard.PasteboardType) {
        currentString = text
    }

    func simulateExternalChange(to text: String) {
        currentString = text
        changeCount += 10
    }
}
