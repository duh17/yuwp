import AppKit
import ApplicationServices

@MainActor
protocol AXTextElementAccessing {
    func readValue() -> String?
    @discardableResult func writeValue(_ value: String) -> Bool
    @discardableResult func setSelectedRange(location: Int, length: Int) -> Bool
    func caretScreenPoint() -> NSPoint?
}

@MainActor
final class AXUIElementTextAccessor: AXTextElementAccessing {
    private let element: AXUIElement

    init(element: AXUIElement) {
        self.element = element
    }

    func readValue() -> String? {
        var ref: AnyObject?
        guard AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &ref) == .success
        else { return nil }
        return ref as? String
    }

    @discardableResult
    func writeValue(_ value: String) -> Bool {
        AXUIElementSetAttributeValue(element, kAXValueAttribute as CFString, value as CFTypeRef) == .success
    }

    @discardableResult
    func setSelectedRange(location: Int, length: Int) -> Bool {
        var range = CFRange(location: location, length: length)
        guard let val = AXValueCreate(.cfRange, &range) else { return false }
        return AXUIElementSetAttributeValue(
            element,
            kAXSelectedTextRangeAttribute as CFString,
            val
        ) == .success
    }

    func caretScreenPoint() -> NSPoint? {
        var rangeRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            element, kAXSelectedTextRangeAttribute as CFString, &rangeRef
        ) == .success else { return nil }
        var boundsRef: CFTypeRef?
        guard AXUIElementCopyParameterizedAttributeValue(
            element,
            kAXBoundsForRangeParameterizedAttribute as CFString,
            rangeRef!,
            &boundsRef
        ) == .success else { return nil }
        var rect = CGRect.zero
        guard AXValueGetValue(boundsRef as! AXValue, .cgRect, &rect) else { return nil }
        return NSPoint(x: rect.origin.x, y: rect.origin.y + rect.height)
    }
}

/// Injects text directly into an AX-accessible text field.
///
/// Writes the full value to avoid selection-flash. Verifies the first inject
/// by reading back the written text — some apps (e.g. terminal emulators) accept
/// AX writes silently but don't surface them. On verification failure, degrades
/// transparently to a `CGEventInjector` fallback.
///
/// Final commit uses clipboard paste (most reliable for AX fields; avoids
/// edge cases with undo history and text attributes).
@MainActor
final class AXTextInjector: TextInjecting {

    // MARK: - TextInjecting

    var surfaceMode: DictationSurfaceMode {
        fallback == nil ? .nativeField : .terminal
    }

    private(set) var targetPosition: NSPoint

    func captureTarget() {
        // No-op — element captured at init time by TextInjectorFactory.
    }

    func inject(_ text: String) {
        if let fallback {
            fallback.inject(text)
            return
        }

        let existing = element.readValue() ?? ""
        let newValue = Self.composeInjectedValue(
            existing: existing,
            cursorPosition: cursorPosition,
            writtenLength: writtenLength,
            injectedText: text
        )

        guard element.writeValue(newValue) else {
            yuwpLog("AX value write failed — degrading to CGEvent")
            fallback = fallbackFactory(targetPosition)
            return
        }

        if !axVerified {
            if let readBack = element.readValue(), readBack.contains(text) {
                axVerified = true
                yuwpLog("AX injection verified")
            } else {
                yuwpLog("AX write accepted but verification failed — degrading to CGEvent")
                _ = element.writeValue(existing)
                fallback = fallbackFactory(targetPosition)
                return
            }
        }

        writtenLength = text.count
        _ = element.setSelectedRange(location: cursorPosition + text.count, length: 0)
        if let pt = element.caretScreenPoint() {
            targetPosition = pt
        }
    }

    func commit(_ text: String) {
        if let fallback {
            fallback.commit(text)
            return
        }

        if writtenLength > 0 {
            if let existing = element.readValue() {
                let restored = Self.removingInjectedPreview(
                    from: existing,
                    cursorPosition: cursorPosition,
                    writtenLength: writtenLength
                )
                _ = element.writeValue(restored)
                _ = element.setSelectedRange(location: cursorPosition, length: 0)
            }
            writtenLength = 0
        }

        clipboardHelper.pasteViaClipboard(text)
    }

    func release() {
        writtenLength = 0
        axVerified = false
        fallback?.release()
        fallback = nil
    }

    // MARK: - Init

    init(element: AXUIElement, cursorPosition: Int, screenPoint: NSPoint) {
        self.element = AXUIElementTextAccessor(element: element)
        self.cursorPosition = cursorPosition
        self.targetPosition = screenPoint
        self.clipboardHelper = ClipboardInjector()
        self.fallbackFactory = { CGEventInjector(screenPoint: $0) }
    }

    init(
        element: any AXTextElementAccessing,
        cursorPosition: Int,
        screenPoint: NSPoint,
        clipboardHelper: any ClipboardPasting,
        fallbackFactory: @escaping (NSPoint) -> any TextInjecting
    ) {
        self.element = element
        self.cursorPosition = cursorPosition
        self.targetPosition = screenPoint
        self.clipboardHelper = clipboardHelper
        self.fallbackFactory = fallbackFactory
    }

    // MARK: - Pure helpers

    static func composeInjectedValue(
        existing: String,
        cursorPosition: Int,
        writtenLength: Int,
        injectedText: String
    ) -> String {
        let before = String(existing.prefix(cursorPosition))
        let after = String(existing.dropFirst(cursorPosition + writtenLength))
        return before + injectedText + after
    }

    static func removingInjectedPreview(
        from existing: String,
        cursorPosition: Int,
        writtenLength: Int
    ) -> String {
        let before = String(existing.prefix(cursorPosition))
        let after = String(existing.dropFirst(cursorPosition + writtenLength))
        return before + after
    }

    // MARK: - Private

    private let element: any AXTextElementAccessing
    private let cursorPosition: Int
    private var writtenLength = 0
    private var axVerified = false
    private var fallback: (any TextInjecting)?
    private let clipboardHelper: any ClipboardPasting
    private let fallbackFactory: (NSPoint) -> any TextInjecting
}
