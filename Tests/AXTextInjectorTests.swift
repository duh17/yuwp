import AppKit
import Testing
@testable import Yuwp

@Suite("AXTextInjector")
@MainActor
struct AXTextInjectorTests {
    @Test func composeInjectedValueReplacesExistingPreviewRange() {
        let composed = AXTextInjector.composeInjectedValue(
            existing: "hello world",
            cursorPosition: 6,
            writtenLength: 5,
            injectedText: "friend"
        )

        #expect(composed == "hello friend")
    }

    @Test func removingInjectedPreviewRestoresOriginalText() {
        let restored = AXTextInjector.removingInjectedPreview(
            from: "hello friend",
            cursorPosition: 6,
            writtenLength: 6
        )

        #expect(restored == "hello ")
    }

    @Test func injectVerifiesAxWriteAndUpdatesCaretPosition() {
        let element = FakeAXTextElement(value: "")
        element.caretPoint = NSPoint(x: 42, y: 24)
        let clipboard = FakeClipboardPaster()

        let injector = AXTextInjector(
            element: element,
            cursorPosition: 0,
            screenPoint: .zero,
            clipboardHelper: clipboard,
            fallbackFactory: { SpyTextInjector(surfaceMode: .terminal, targetPosition: $0) }
        )

        injector.inject("Hello")
        injector.inject("Hello world")

        #expect(injector.surfaceMode == .nativeField)
        #expect(element.value == "Hello world")
        #expect(element.selectedRanges.count == 2)
        #expect(element.selectedRanges[0].location == 5)
        #expect(element.selectedRanges[0].length == 0)
        #expect(element.selectedRanges[1].location == 11)
        #expect(element.selectedRanges[1].length == 0)
        #expect(injector.targetPosition == NSPoint(x: 42, y: 24))
        #expect(clipboard.pastedTexts.isEmpty)
    }

    @Test func verificationFailureDegradesToFallbackAndRestoresOriginalValue() {
        let element = FakeAXTextElement(value: "orig")
        element.readValueOverride = { "orig" }
        let clipboard = FakeClipboardPaster()
        let fallback = SpyTextInjector(surfaceMode: .terminal, targetPosition: .zero)

        let injector = AXTextInjector(
            element: element,
            cursorPosition: 0,
            screenPoint: NSPoint(x: 9, y: 9),
            clipboardHelper: clipboard,
            fallbackFactory: { _ in fallback }
        )

        injector.inject("Hello")
        injector.inject("Hello again")

        #expect(injector.surfaceMode == .terminal)
        #expect(element.writeHistory == ["Helloorig", "orig"])
        #expect(fallback.injectedTexts == ["Hello again"])
    }

    @Test func writeFailureDegradesAndCommitForwardsToFallback() {
        let element = FakeAXTextElement(value: "orig")
        element.writeShouldSucceed = false
        let clipboard = FakeClipboardPaster()
        let fallback = SpyTextInjector(surfaceMode: .terminal, targetPosition: .zero)

        let injector = AXTextInjector(
            element: element,
            cursorPosition: 0,
            screenPoint: .zero,
            clipboardHelper: clipboard,
            fallbackFactory: { _ in fallback }
        )

        injector.inject("Hello")
        injector.commit("Final")

        #expect(injector.surfaceMode == .terminal)
        #expect(fallback.committedTexts == ["Final"])
        #expect(clipboard.pastedTexts.isEmpty)
    }

    @Test func commitRemovesPreviewAndPastesFinalText() {
        let element = FakeAXTextElement(value: "abc")
        let clipboard = FakeClipboardPaster()

        let injector = AXTextInjector(
            element: element,
            cursorPosition: 1,
            screenPoint: .zero,
            clipboardHelper: clipboard,
            fallbackFactory: { SpyTextInjector(surfaceMode: .terminal, targetPosition: $0) }
        )

        injector.inject("ZZ")
        injector.commit("FINAL")

        #expect(element.value == "abc")
        #expect(element.selectedRanges.last?.location == 1)
        #expect(element.selectedRanges.last?.length == 0)
        #expect(clipboard.pastedTexts == ["FINAL"])
    }

    @Test func releaseResetsFallbackBackToNativeField() {
        let element = FakeAXTextElement(value: "orig")
        element.readValueOverride = { "orig" }
        let fallback = SpyTextInjector(surfaceMode: .terminal, targetPosition: .zero)

        let injector = AXTextInjector(
            element: element,
            cursorPosition: 0,
            screenPoint: .zero,
            clipboardHelper: FakeClipboardPaster(),
            fallbackFactory: { _ in fallback }
        )

        injector.inject("Hello")
        #expect(injector.surfaceMode == .terminal)

        injector.release()

        #expect(injector.surfaceMode == .nativeField)
        #expect(fallback.releaseCallCount == 1)
    }
}

@MainActor
private final class FakeAXTextElement: AXTextElementAccessing {
    var value: String?
    var writeHistory: [String] = []
    var selectedRanges: [CFRange] = []
    var writeShouldSucceed = true
    var readValueOverride: (() -> String?)?
    var caretPoint: NSPoint?

    init(value: String?) {
        self.value = value
    }

    func readValue() -> String? {
        readValueOverride?() ?? value
    }

    func writeValue(_ value: String) -> Bool {
        writeHistory.append(value)
        guard writeShouldSucceed else { return false }
        self.value = value
        return true
    }

    func setSelectedRange(location: Int, length: Int) -> Bool {
        selectedRanges.append(CFRange(location: location, length: length))
        return true
    }

    func caretScreenPoint() -> NSPoint? {
        caretPoint
    }
}

@MainActor
private final class FakeClipboardPaster: ClipboardPasting {
    var pastedTexts: [String] = []

    func pasteViaClipboard(_ text: String) {
        pastedTexts.append(text)
    }
}

@MainActor
private final class SpyTextInjector: TextInjecting {
    let surfaceMode: DictationSurfaceMode
    var targetPosition: NSPoint
    var injectedTexts: [String] = []
    var committedTexts: [String] = []
    var releaseCallCount = 0

    init(surfaceMode: DictationSurfaceMode, targetPosition: NSPoint) {
        self.surfaceMode = surfaceMode
        self.targetPosition = targetPosition
    }

    func captureTarget() {}

    func inject(_ text: String) {
        injectedTexts.append(text)
    }

    func commit(_ text: String) {
        committedTexts.append(text)
    }

    func release() {
        releaseCallCount += 1
    }
}
