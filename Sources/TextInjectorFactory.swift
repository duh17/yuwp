import AppKit
import ApplicationServices

/// Creates the best `TextInjecting` implementation for the currently focused element.
///
/// Decision tree:
///   1. No focused element → `ClipboardInjector`
///   2. Non-editable AX role (terminals, etc.) → `CGEventInjector`
///   3. Editable role but AX write probe fails → `ClipboardInjector`
///   4. Editable role, AX write probe OK → `AXTextInjector`
///      (degrades to `CGEventInjector` on first inject if verification fails)
@MainActor
enum TextInjectorFactory {

    // Roles known to support AX text editing.
    private static let editableRoles: Set<String> = [
        "AXTextField", "AXTextArea", "AXComboBox", "AXSearchField",
    ]

    /// Probe the focused element and return an appropriate injector.
    /// Must be called before any UI panel appears — focus changes after that.
    static func capture() -> any TextInjecting {
        guard let focused = focusedElement() else {
            yuwpLog("No focused element — will use clipboard fallback")
            return ClipboardInjector()
        }

        let point = readTargetScreenPoint(from: focused) ?? NSEvent.mouseLocation

        // Read role.
        var roleRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            focused, kAXRoleAttribute as CFString, &roleRef
        ) == .success, let role = roleRef as? String else {
            yuwpLog("No AX role — using CGEvent")
            return CGEventInjector(screenPoint: point)
        }

        guard editableRoles.contains(role) else {
            yuwpLog("Non-editable role (\(role)) — using CGEvent")
            return CGEventInjector(screenPoint: point)
        }

        // Verify AX selection range is readable.
        var rangeRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            focused, kAXSelectedTextRangeAttribute as CFString, &rangeRef
        ) == .success else {
            yuwpLog("No AX selection range — using clipboard")
            return ClipboardInjector()
        }

        // Verify we can write selected text — the op used by AXTextInjector.
        let writeOK = AXUIElementSetAttributeValue(
            focused, kAXSelectedTextAttribute as CFString, "" as CFTypeRef
        )
        guard writeOK == .success else {
            yuwpLog("AX write probe failed — using clipboard")
            return ClipboardInjector()
        }

        let cursor = readCursorOffset(from: focused) ?? 0
        yuwpLog("Target captured (AX, cursor: \(cursor))")
        return AXTextInjector(element: focused, cursorPosition: cursor, screenPoint: point)
    }

    // MARK: - Private AX helpers

    private static func focusedElement() -> AXUIElement? {
        let sys = AXUIElementCreateSystemWide()
        var ref: AnyObject?
        guard AXUIElementCopyAttributeValue(
            sys, kAXFocusedUIElementAttribute as CFString, &ref
        ) == .success else { return nil }
        // swiftlint:disable:next force_cast
        return (ref as! AXUIElement)
    }

    private static func readCursorOffset(from element: AXUIElement) -> Int? {
        var ref: AnyObject?
        guard AXUIElementCopyAttributeValue(
            element, kAXSelectedTextRangeAttribute as CFString, &ref
        ) == .success else { return nil }
        var range = CFRange(location: 0, length: 0)
        // swiftlint:disable:next force_cast
        guard AXValueGetValue(ref as! AXValue, .cfRange, &range) else { return nil }
        return range.location
    }

    private static func readTargetScreenPoint(from element: AXUIElement) -> NSPoint? {
        if let caretPoint = caretScreenPoint(from: element) { return caretPoint }
        var ref: AnyObject?
        guard AXUIElementCopyAttributeValue(
            element, kAXPositionAttribute as CFString, &ref
        ) == .success else { return nil }
        var point = CGPoint.zero
        // swiftlint:disable:next force_cast
        AXValueGetValue(ref as! AXValue, .cgPoint, &point)
        return NSPoint(x: point.x, y: point.y)
    }

    private static func caretScreenPoint(from element: AXUIElement) -> NSPoint? {
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
        // swiftlint:disable:next force_cast
        guard AXValueGetValue(boundsRef as! AXValue, .cgRect, &rect) else { return nil }
        return NSPoint(x: rect.origin.x, y: rect.origin.y + rect.height)
    }
}
