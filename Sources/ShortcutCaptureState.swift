import Foundation

enum ShortcutCaptureResult: Equatable {
    case none
    case captured(KeyBinding)
    case cancelled
    case invalid
}

struct ShortcutCaptureState: Equatable {
    private(set) var pendingModifierKeyCode: UInt16?
    private(set) var pendingModifierCaptureDeadline: TimeInterval?

    private var pendingSingleModifierBinding: KeyBinding?

    var pendingSingleModifierKeyCode: UInt16? {
        pendingSingleModifierBinding?.keyCode
    }

    mutating func handleKeyDown(
        keyCode: UInt16,
        modifiers: UInt64,
        timestamp: TimeInterval
    ) -> ShortcutCaptureResult {
        if keyCode == 53 { // Esc
            reset()
            return .cancelled
        }

        if let expired = resolveTimeout(at: timestamp) {
            return expired
        }

        guard modifiers != 0 else {
            reset()
            return .invalid
        }

        reset()
        return .captured(KeyBinding(keyCode: keyCode, modifiers: modifiers))
    }

    mutating func handleFlagsChanged(
        keyCode: UInt16,
        timestamp: TimeInterval
    ) -> ShortcutCaptureResult {
        guard KeyBinding.isModifierKeyCode(keyCode) else { return .none }

        if let expired = resolveTimeout(at: timestamp) {
            return expired
        }

        if pendingModifierKeyCode == keyCode {
            pendingModifierKeyCode = nil

            if let pendingSingleModifierBinding,
               pendingSingleModifierBinding.keyCode == keyCode,
               let deadline = pendingModifierCaptureDeadline,
               timestamp <= deadline {
                clearPendingSingleModifierCapture()
                return .captured(KeyBinding(keyCode: keyCode, modifiers: 0, activation: .doubleTap))
            }

            pendingSingleModifierBinding = KeyBinding(keyCode: keyCode, modifiers: 0)
            pendingModifierCaptureDeadline = timestamp + KeyBindingTiming.doubleTapTimeout
            return .none
        }

        pendingModifierKeyCode = keyCode
        return .none
    }

    mutating func resolveTimeout(at timestamp: TimeInterval) -> ShortcutCaptureResult? {
        guard let pendingSingleModifierBinding,
              let deadline = pendingModifierCaptureDeadline,
              timestamp >= deadline else {
            return nil
        }

        clearPendingSingleModifierCapture()
        pendingModifierKeyCode = nil
        return .captured(pendingSingleModifierBinding)
    }

    mutating func reset() {
        pendingModifierKeyCode = nil
        clearPendingSingleModifierCapture()
    }

    private mutating func clearPendingSingleModifierCapture() {
        pendingSingleModifierBinding = nil
        pendingModifierCaptureDeadline = nil
    }
}

struct ModifierHotkeyTracker: Equatable {
    let keyCode: UInt16
    let activation: KeyBindingActivation

    private(set) var isPressed = false
    private var pendingFirstTapReleaseAt: TimeInterval?

    init?(binding: KeyBinding) {
        guard binding.isModifierOnly else { return nil }
        self.keyCode = binding.keyCode
        self.activation = binding.activation
    }

    mutating func handleFlagsChanged(
        keyCode: UInt16,
        timestamp: TimeInterval = CFAbsoluteTimeGetCurrent()
    ) -> ShortcutPhase? {
        guard keyCode == self.keyCode else { return nil }

        switch activation {
        case .singlePress:
            defer { isPressed.toggle() }
            return isPressed ? .released : .pressed

        case .doubleTap:
            return handleDoubleTap(timestamp: timestamp)
        }
    }

    mutating func reset() {
        isPressed = false
        pendingFirstTapReleaseAt = nil
    }

    private mutating func handleDoubleTap(timestamp: TimeInterval) -> ShortcutPhase? {
        if isPressed {
            isPressed = false
            if let pendingFirstTapReleaseAt,
               timestamp - pendingFirstTapReleaseAt <= KeyBindingTiming.doubleTapTimeout {
                self.pendingFirstTapReleaseAt = nil
                return .pressed
            }

            pendingFirstTapReleaseAt = timestamp
            return nil
        }

        if let pendingFirstTapReleaseAt,
           timestamp - pendingFirstTapReleaseAt > KeyBindingTiming.doubleTapTimeout {
            self.pendingFirstTapReleaseAt = nil
        }

        isPressed = true
        return nil
    }
}
