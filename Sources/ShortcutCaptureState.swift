import Foundation

enum ShortcutCaptureResult: Equatable {
    case none
    case captured(KeyBinding)
    case cancelled
    case invalid
}

struct ShortcutCaptureState: Equatable {
    private(set) var pendingModifierKeyCode: UInt16?

    mutating func handleKeyDown(keyCode: UInt16, modifiers: UInt64) -> ShortcutCaptureResult {
        if keyCode == 53 { // Esc
            pendingModifierKeyCode = nil
            return .cancelled
        }

        guard modifiers != 0 else {
            pendingModifierKeyCode = nil
            return .invalid
        }

        pendingModifierKeyCode = nil
        return .captured(KeyBinding(keyCode: keyCode, modifiers: modifiers))
    }

    mutating func handleFlagsChanged(keyCode: UInt16) -> ShortcutCaptureResult {
        guard KeyBinding.isModifierKeyCode(keyCode) else { return .none }

        if pendingModifierKeyCode == keyCode {
            pendingModifierKeyCode = nil
            return .captured(KeyBinding(keyCode: keyCode, modifiers: 0))
        }

        pendingModifierKeyCode = keyCode
        return .none
    }
}

struct ModifierHotkeyTracker: Equatable {
    let keyCode: UInt16
    private(set) var isPressed = false

    init?(binding: KeyBinding) {
        guard binding.isModifierOnly else { return nil }
        self.keyCode = binding.keyCode
    }

    mutating func handleFlagsChanged(keyCode: UInt16) -> ShortcutPhase? {
        guard keyCode == self.keyCode else { return nil }
        defer { isPressed.toggle() }
        return isPressed ? .released : .pressed
    }

    mutating func reset() {
        isPressed = false
    }
}
