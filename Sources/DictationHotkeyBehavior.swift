import Foundation

enum ShortcutCommand: String, Sendable {
    case dictation
}

enum ShortcutPhase: Sendable, Equatable {
    case pressed
    case released
}

struct ShortcutEvent: Sendable, Equatable {
    let command: ShortcutCommand
    let phase: ShortcutPhase
}

enum DictationHotkeyAction: Equatable {
    case start
    case stop
    case none
}

/// Small reducer that maps shortcut press events to dictation actions.
/// Keeps the hotkey interaction policy testable and out of AppDelegate branches.
struct DictationHotkeyBehavior: Equatable {
    mutating func handle(
        phase: ShortcutPhase,
        isSessionActive: Bool
    ) -> DictationHotkeyAction {
        guard phase == .pressed else { return .none }
        return isSessionActive ? .stop : .start
    }

    mutating func sessionDidEnd() {}
}
