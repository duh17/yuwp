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

/// Small reducer that maps shortcut press/release events to dictation actions.
/// Keeps the hotkey interaction policy testable and out of AppDelegate branches.
struct DictationHotkeyBehavior: Equatable {
    private(set) var sessionStartedByPushToTalk = false

    mutating func handle(
        phase: ShortcutPhase,
        mode: DictationInteractionMode,
        isSessionActive: Bool
    ) -> DictationHotkeyAction {
        switch mode {
        case .toggle:
            guard phase == .pressed else { return .none }
            sessionStartedByPushToTalk = false
            return isSessionActive ? .stop : .start

        case .pushToTalk:
            switch phase {
            case .pressed:
                guard !isSessionActive else { return .none }
                sessionStartedByPushToTalk = true
                return .start

            case .released:
                guard isSessionActive, sessionStartedByPushToTalk else { return .none }
                sessionStartedByPushToTalk = false
                return .stop
            }
        }
    }

    mutating func sessionDidEnd() {
        sessionStartedByPushToTalk = false
    }
}
