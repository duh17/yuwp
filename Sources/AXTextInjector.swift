import AppKit
import ApplicationServices

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
        if let fb = fallback {
            fb.inject(text)
            return
        }

        let existing = readValue(from: element) ?? ""
        let before = String(existing.prefix(cursorPosition))
        let after = String(existing.dropFirst(cursorPosition + writtenLength))
        let newValue = before + text + after

        let writeOK = AXUIElementSetAttributeValue(
            element, kAXValueAttribute as CFString, newValue as CFTypeRef
        ) == .success

        if writeOK {
            if !axVerified {
                if let readBack = readValue(from: element), readBack.contains(text) {
                    axVerified = true
                    yuwpLog("AX injection verified")
                } else {
                    yuwpLog("AX write accepted but verification failed — degrading to CGEvent")
                    // Undo the bogus write before handing off.
                    AXUIElementSetAttributeValue(
                        element, kAXValueAttribute as CFString, existing as CFTypeRef
                    )
                    fallback = CGEventInjector(screenPoint: targetPosition)
                    return
                }
            }

            writtenLength = text.count
            let cursorEnd = cursorPosition + text.count
            var range = CFRange(location: cursorEnd, length: 0)
            if let val = AXValueCreate(.cfRange, &range) {
                AXUIElementSetAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, val)
            }
            if let pt = caretScreenPoint(from: element) {
                targetPosition = pt
            }
        } else {
            yuwpLog("AX value write failed — degrading to CGEvent")
            fallback = CGEventInjector(screenPoint: targetPosition)
        }
    }

    func commit(_ text: String) {
        if let fb = fallback {
            fb.commit(text)
            return
        }

        // Strip out any AX-injected preview before committing via clipboard.
        if writtenLength > 0 {
            if let existing = readValue(from: element) {
                let before = String(existing.prefix(cursorPosition))
                let after = String(existing.dropFirst(cursorPosition + writtenLength))
                AXUIElementSetAttributeValue(
                    element, kAXValueAttribute as CFString, (before + after) as CFTypeRef
                )
                var range = CFRange(location: cursorPosition, length: 0)
                if let val = AXValueCreate(.cfRange, &range) {
                    AXUIElementSetAttributeValue(
                        element, kAXSelectedTextRangeAttribute as CFString, val
                    )
                }
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
        self.element = element
        self.cursorPosition = cursorPosition
        self.targetPosition = screenPoint
    }

    // MARK: - Private

    private let element: AXUIElement
    private let cursorPosition: Int
    private var writtenLength = 0
    private var axVerified = false
    private var fallback: CGEventInjector?
    private let clipboardHelper = ClipboardInjector()

    private func readValue(from element: AXUIElement) -> String? {
        var ref: AnyObject?
        guard AXUIElementCopyAttributeValue(element, kAXValueAttribute as CFString, &ref) == .success
        else { return nil }
        return ref as? String
    }

    private func caretScreenPoint(from element: AXUIElement) -> NSPoint? {
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
