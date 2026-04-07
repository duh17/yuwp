import AppKit
import ApplicationServices

/// Injects transcribed text into the currently focused text field.
///
/// Uses macOS Accessibility APIs (AXUIElement) to directly manipulate
/// the focused text element. Falls back to clipboard paste (Cmd+V)
/// when AX text manipulation isn't supported by the target.
///
/// Flow:
///   1. captureTarget() — snapshot the focused element before UI appears
///   2. inject(_:) — stream partial results into the field (AX only)
///   3. commit(_:) — finalize text (AX or clipboard fallback)
///   4. release() — cleanup
@MainActor
final class TextInjector {

    // MARK: - Public

    /// Current screen position of the caret (updated on each inject)
    private(set) var targetPosition: NSPoint = .zero

    /// Whether live AX injection is active (text streams directly into the target field).
    /// When true, callers can skip showing the transcript overlay.
    var isLiveInjecting: Bool {
        anchor?.method == .accessibility
    }

    /// Snapshot the currently focused text element.
    /// Must be called before showing any panel that could steal focus.
    func captureTarget() {
        release()

        guard let focused = focusedElement() else {
            yuwpLog("No focused element — will use clipboard fallback")
            return
        }

        let method = probeMethod(for: focused)
        let cursor = readCursorOffset(from: focused) ?? 0
        let point = readTargetScreenPoint(from: focused) ?? NSEvent.mouseLocation

        anchor = Anchor(
            element: focused,
            method: method,
            cursorPosition: cursor,
            screenPoint: point
        )
        targetPosition = point

        yuwpLog("Target captured (\(method), cursor: \(cursor))")
    }

    /// Stream a partial transcription into the field.
    /// Writes the full value directly to avoid selection flashing.
    /// No-op when using clipboard method (text only committed on finish).
    func inject(_ text: String) {
        guard let anchor, anchor.method == .accessibility else { return }

        // Read the existing text, splice our portion in, write the whole value back.
        // This avoids the visible select-then-replace flash that select+replace causes.
        let existing = readValue(from: anchor.element) ?? ""
        let before = String(existing.prefix(anchor.cursorPosition))
        let after = String(existing.dropFirst(anchor.cursorPosition + writtenLength))
        let newValue = before + text + after

        let writeOK = AXUIElementSetAttributeValue(
            anchor.element, kAXValueAttribute as CFString, newValue as CFTypeRef
        ) == .success

        if writeOK {
            writtenLength = text.count
            // Move cursor to end of injected text
            let cursorEnd = anchor.cursorPosition + text.count
            var range = CFRange(location: cursorEnd, length: 0)
            if let val = AXValueCreate(.cfRange, &range) {
                AXUIElementSetAttributeValue(
                    anchor.element, kAXSelectedTextRangeAttribute as CFString, val
                )
            }
            // Update caret screen position so the indicator can follow
            if let pt = caretScreenPoint(from: anchor.element) {
                targetPosition = pt
            }
        } else {
            // Value write failed — degrade to clipboard for commit
            self.anchor = Anchor(
                element: anchor.element,
                method: .paste,
                cursorPosition: anchor.cursorPosition,
                screenPoint: anchor.screenPoint
            )
            yuwpLog("AX value write failed — degrading to clipboard")
        }
    }

    /// Commit the final transcription.
    /// Uses clipboard paste if AX isn't available.
    func commit(_ text: String) {
        guard let anchor else {
            yuwpLog("commit: anchor is nil, text lost (\(text.count) chars)")
            return
        }

        // Always use clipboard paste for the final commit — it's the most
        // reliable path across all apps. AX inject is only for live preview.
        // First, undo any AX-injected preview text by writing the value without it.
        if anchor.method == .accessibility && writtenLength > 0 {
            if let existing = readValue(from: anchor.element) {
                let before = String(existing.prefix(anchor.cursorPosition))
                let after = String(existing.dropFirst(anchor.cursorPosition + writtenLength))
                AXUIElementSetAttributeValue(
                    anchor.element, kAXValueAttribute as CFString, (before + after) as CFTypeRef
                )
                // Restore cursor to injection point
                var range = CFRange(location: anchor.cursorPosition, length: 0)
                if let val = AXValueCreate(.cfRange, &range) {
                    AXUIElementSetAttributeValue(
                        anchor.element, kAXSelectedTextRangeAttribute as CFString, val
                    )
                }
            }
            writtenLength = 0
        }

        pasteViaClipboard(text)
    }

    /// Cleanup after dictation ends.
    func release() {
        anchor = nil
        writtenLength = 0
        targetPosition = .zero
    }

    // MARK: - Types

    private enum Method: CustomStringConvertible {
        /// Direct AX text manipulation (read/write cursor + selection)
        case accessibility
        /// Clipboard save → paste → restore
        case paste

        var description: String {
            switch self {
            case .accessibility: "AX"
            case .paste: "clipboard"
            }
        }
    }

    private struct Anchor {
        let element: AXUIElement
        let method: Method
        let cursorPosition: Int
        let screenPoint: NSPoint
    }

    // MARK: - State

    private var anchor: Anchor?
    private var writtenLength = 0

    // Roles that are known to support AX text editing
    private static let editableRoles: Set<String> = [
        "AXTextField", "AXTextArea", "AXComboBox", "AXSearchField",
    ]

    // MARK: - AX Helpers

    /// Read the full text value from an element.
    private func readValue(from element: AXUIElement) -> String? {
        var ref: AnyObject?
        guard AXUIElementCopyAttributeValue(
            element, kAXValueAttribute as CFString, &ref
        ) == .success else { return nil }
        return ref as? String
    }

    /// Get the system-wide focused UI element.
    private func focusedElement() -> AXUIElement? {
        let sys = AXUIElementCreateSystemWide()
        var ref: AnyObject?
        guard AXUIElementCopyAttributeValue(
            sys, kAXFocusedUIElementAttribute as CFString, &ref
        ) == .success else { return nil }
        return (ref as! AXUIElement)
    }

    /// Determine whether we can do AX text manipulation on this element.
    private func probeMethod(for element: AXUIElement) -> Method {
        // Check the element's role
        var roleRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            element, kAXRoleAttribute as CFString, &roleRef
        ) == .success, let role = roleRef as? String else {
            return .paste
        }

        // AXWebArea can sometimes work, but is unreliable — prefer clipboard
        guard Self.editableRoles.contains(role) else { return .paste }

        // Verify we can actually read and write the selection range
        var rangeRef: AnyObject?
        guard AXUIElementCopyAttributeValue(
            element, kAXSelectedTextRangeAttribute as CFString, &rangeRef
        ) == .success else { return .paste }

        // Verify we can write selected text (the operation we'll use for injection)
        let writeOK = AXUIElementSetAttributeValue(
            element, kAXSelectedTextAttribute as CFString, "" as CFTypeRef
        )
        return writeOK == .success ? .accessibility : .paste
    }

    /// Read the current cursor offset in the text field.
    private func readCursorOffset(from element: AXUIElement) -> Int? {
        var ref: AnyObject?
        guard AXUIElementCopyAttributeValue(
            element, kAXSelectedTextRangeAttribute as CFString, &ref
        ) == .success else { return nil }
        var range = CFRange(location: 0, length: 0)
        guard AXValueGetValue(ref as! AXValue, .cfRange, &range) else { return nil }
        return range.location
    }

    /// Read the screen position of the target, preferring caret bounds.
    private func readTargetScreenPoint(from element: AXUIElement) -> NSPoint? {
        // Try caret bounds first (most precise)
        if let caretPoint = caretScreenPoint(from: element) {
            return caretPoint
        }
        // Fallback: element origin
        var ref: AnyObject?
        guard AXUIElementCopyAttributeValue(
            element, kAXPositionAttribute as CFString, &ref
        ) == .success else { return nil }
        var point = CGPoint.zero
        AXValueGetValue(ref as! AXValue, .cgPoint, &point)
        return NSPoint(x: point.x, y: point.y)
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

    /// Select a range of text in the element.
    private func selectRange(in element: AXUIElement, location: Int, length: Int) -> Bool {
        var range = CFRange(location: location, length: length)
        guard let value = AXValueCreate(.cfRange, &range) else { return false }
        return AXUIElementSetAttributeValue(
            element, kAXSelectedTextRangeAttribute as CFString, value
        ) == .success
    }

    /// Replace the current selection with new text.
    private func replaceSelection(in element: AXUIElement, with text: String) -> Bool {
        AXUIElementSetAttributeValue(
            element, kAXSelectedTextAttribute as CFString, text as CFTypeRef
        ) == .success
    }

    // MARK: - Clipboard Fallback

    private func pasteViaClipboard(_ text: String) {
        let pb = NSPasteboard.general

        // Snapshot current clipboard content
        let savedString = pb.string(forType: .string)
        let savedChangeCount = pb.changeCount

        // Stage our text
        pb.clearContents()
        pb.setString(text, forType: .string)

        // Simulate Cmd+V
        let vKey: CGKeyCode = 9
        if let down = CGEvent(keyboardEventSource: nil, virtualKey: vKey, keyDown: true),
           let up = CGEvent(keyboardEventSource: nil, virtualKey: vKey, keyDown: false) {
            down.flags = .maskCommand
            up.flags = .maskCommand
            down.post(tap: .cgSessionEventTap)
            up.post(tap: .cgSessionEventTap)
        }

        // Restore clipboard after the paste event lands
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.15) {
            // Only restore if nobody else touched the clipboard
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
