import AppKit
import ApplicationServices

/// Creates the `TextInjecting` implementation for the currently focused element.
///
/// Default behavior is safety-first: bubble preview grows live and commit pastes
/// via clipboard. Direct insertion paths are experimental and opt-in.
///
/// Capability tree (before policy gating):
///   1. No focused element → `ClipboardInjector`
///   2. Non-editable AX role (terminals, etc.) → `CGEventInjector`
///   3. Editable role but AX value probe fails → `CGEventInjector`
///   4. Editable role, AX value probe OK → `AXTextInjector`
///      (degrades to `CGEventInjector` on first inject if verification fails)
@MainActor
enum TextInjectorFactory {
    enum Strategy: Equatable {
        case ax
        case cgEvent
        case clipboard
    }

    struct InjectionPolicy: Equatable, Sendable {
        var allowTextFieldDirectInjection: Bool
        var allowTerminalDirectInjection: Bool

        static let safeDefault = InjectionPolicy(
            allowTextFieldDirectInjection: false,
            allowTerminalDirectInjection: false
        )

        @MainActor
        static var fromConfig: InjectionPolicy {
            InjectionPolicy(
                allowTextFieldDirectInjection: Config.shared.experimentalDirectTextFieldInsertionEnabled,
                allowTerminalDirectInjection: Config.shared.experimentalDirectTerminalInsertionEnabled
            )
        }
    }

    // Roles known to support AX text editing.
    nonisolated private static let editableRoles: Set<String> = [
        "AXTextField", "AXTextArea", "AXComboBox", "AXSearchField",
    ]

    static func decideStrategy(
        hasFocusedElement: Bool,
        role: String?,
        hasReadableSelectionRange: Bool,
        canWriteSelectedText: Bool
    ) -> Strategy {
        guard hasFocusedElement else { return .clipboard }
        guard let role else { return .cgEvent }
        guard editableRoles.contains(role) else { return .cgEvent }
        guard hasReadableSelectionRange else { return .clipboard }
        guard canWriteSelectedText else { return .cgEvent }
        return .ax
    }

    nonisolated static func allowsDirectInjection(
        strategy: Strategy,
        role: String?,
        policy: InjectionPolicy
    ) -> Bool {
        switch strategy {
        case .clipboard:
            return false
        case .ax:
            return policy.allowTextFieldDirectInjection
        case .cgEvent:
            guard let role else { return policy.allowTerminalDirectInjection }
            return editableRoles.contains(role)
                ? policy.allowTextFieldDirectInjection
                : policy.allowTerminalDirectInjection
        }
    }

    /// Probe the focused element and return an appropriate injector.
    /// Must be called before any UI panel appears — focus changes after that.
    static func capture(policy: InjectionPolicy = .fromConfig) -> any TextInjecting {
        guard let focused = focusedElement() else {
            yuwpLog("No focused element — clipboard paste fallback; dictated text remains on clipboard")
            return ClipboardInjector(screenPoint: NSEvent.mouseLocation, commitMode: .pasteAndKeep)
        }

        let point = readTargetScreenPoint(from: focused) ?? NSEvent.mouseLocation

        // Read role.
        var roleRef: AnyObject?
        let roleStatus = AXUIElementCopyAttributeValue(
            focused, kAXRoleAttribute as CFString, &roleRef
        )
        let role = roleStatus == .success ? roleRef as? String : nil

        // Verify AX selection range is readable.
        var rangeRef: AnyObject?
        let hasSelectionRange = AXUIElementCopyAttributeValue(
            focused, kAXSelectedTextRangeAttribute as CFString, &rangeRef
        ) == .success

        let canWriteSelectedText = supportsAXTextInjection(
            focused: focused,
            role: role,
            hasReadableSelectionRange: hasSelectionRange
        )

        let strategy = decideStrategy(
            hasFocusedElement: true,
            role: role,
            hasReadableSelectionRange: hasSelectionRange,
            canWriteSelectedText: canWriteSelectedText
        )

        if !allowsDirectInjection(strategy: strategy, role: role, policy: policy) {
            switch strategy {
            case .ax, .cgEvent:
                if let role, editableRoles.contains(role) {
                    yuwpLog("Experimental direct text-field insertion disabled — using clipboard preview + paste")
                } else {
                    yuwpLog("Experimental direct terminal insertion disabled — using clipboard preview + paste")
                }
            case .clipboard:
                if role == nil {
                    yuwpLog("No AX role — using clipboard fallback")
                } else if !hasSelectionRange {
                    yuwpLog("No AX selection range — using clipboard")
                } else {
                    yuwpLog("AX value attribute not settable — using clipboard")
                }
            }
            return ClipboardInjector(screenPoint: point)
        }

        switch strategy {
        case .clipboard:
            if role == nil {
                yuwpLog("No AX role — using clipboard fallback")
            } else if !hasSelectionRange {
                yuwpLog("No AX selection range — using clipboard")
            } else {
                yuwpLog("AX value attribute not settable — using clipboard")
            }
            return ClipboardInjector(screenPoint: point)

        case .cgEvent:
            if let role, editableRoles.contains(role), hasSelectionRange, !canWriteSelectedText {
                yuwpLog("AX value attribute not settable — using CGEvent")
            } else if let role {
                yuwpLog("Non-editable role (\(role)) — using CGEvent")
            } else {
                yuwpLog("No AX role — using CGEvent")
            }
            return CGEventInjector(screenPoint: point)

        case .ax:
            let cursor = readCursorOffset(from: focused) ?? 0
            yuwpLog("Target captured (AX, cursor: \(cursor))")
            return AXTextInjector(element: focused, cursorPosition: cursor, screenPoint: point)
        }
    }

    // MARK: - Private AX helpers

    nonisolated static func supportsAXTextInjection(
        focused: AXUIElement,
        role: String?,
        hasReadableSelectionRange: Bool,
        attributeIsSettable: (AXUIElement, CFString) -> Bool = Self.systemAttributeIsSettable
    ) -> Bool {
        guard let role, editableRoles.contains(role), hasReadableSelectionRange else {
            return false
        }
        return attributeIsSettable(focused, kAXValueAttribute as CFString)
    }

    nonisolated private static func systemAttributeIsSettable(_ element: AXUIElement, _ attribute: CFString) -> Bool {
        var settable = DarwinBoolean(false)
        return AXUIElementIsAttributeSettable(element, attribute, &settable) == .success && settable.boolValue
    }

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
