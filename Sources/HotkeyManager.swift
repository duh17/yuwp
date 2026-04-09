import AppKit
import Carbon.HIToolbox
import CoreGraphics

/// Detects the configured dictation key combo globally and emits press/release events.
///
/// Combo shortcuts use Carbon global hotkeys. Enter interception during active
/// dictation still uses an event tap because it needs to swallow Return in the
/// focused app until final commit completes.
@MainActor
final class HotkeyManager {
    var onShortcutEvent: (@Sendable (ShortcutEvent) -> Void)?

    /// Called when Enter/Return is pressed during an active dictation session.
    /// Set by AppDelegate. The event is swallowed; caller is responsible for
    /// stopping dictation and replaying Enter after final commit.
    var onEnterDuringSession: (@Sendable () -> Void)?

    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?
    private var carbonHotKey: EventHotKeyRef?
    private var carbonHandler: EventHandlerRef?

    // Static state for the C callbacks (no captures allowed)
    nonisolated(unsafe) private static var instance: HotkeyManager?
    nonisolated(unsafe) private static var activeBinding = KeyBinding.ctrlBacktick
    /// When true, Enter/Return keys are intercepted during dictation.
    nonisolated(unsafe) static var sessionActive = false

    private static let hotKeySignature: OSType = 0x59555750 // 'YUWP'
    private static let hotKeyID: UInt32 = 1

    /// Start global hotkey registration and the Enter interception tap.
    /// Returns false if Accessibility permission is missing.
    func start() -> Bool {
        stop()

        let binding = Config.shared.dictationBinding
        HotkeyManager.instance = self
        HotkeyManager.activeBinding = binding

        guard installCarbonHotkey(for: binding) else {
            yuwpLog("Failed to register Carbon hotkey: \(binding.description)")
            stop()
            return false
        }

        guard installEnterInterceptionTap() else {
            yuwpLog("Failed to create event tap — Accessibility permission required")
            stop()
            return false
        }

        yuwpLog("Hotkey registered: \(binding.description)")
        return true
    }

    func stop() {
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: false)
        }
        if let source = runLoopSource {
            CFRunLoopRemoveSource(CFRunLoopGetCurrent(), source, .commonModes)
        }
        eventTap = nil
        runLoopSource = nil

        if let hotKey = carbonHotKey {
            UnregisterEventHotKey(hotKey)
        }
        if let handler = carbonHandler {
            RemoveEventHandler(handler)
        }
        carbonHotKey = nil
        carbonHandler = nil
        HotkeyManager.instance = nil
    }

    func restart() -> Bool {
        stop()
        return start()
    }

    // MARK: - Carbon Hotkey

    private func installCarbonHotkey(for binding: KeyBinding) -> Bool {
        var eventTypes = [
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyPressed)),
            EventTypeSpec(eventClass: OSType(kEventClassKeyboard), eventKind: UInt32(kEventHotKeyReleased)),
        ]

        let installStatus = InstallEventHandler(
            GetApplicationEventTarget(),
            Self.carbonEventHandler,
            2,
            &eventTypes,
            nil,
            &carbonHandler
        )
        guard installStatus == noErr else { return false }

        let hotKeyID = EventHotKeyID(signature: Self.hotKeySignature, id: Self.hotKeyID)
        let registerStatus = RegisterEventHotKey(
            UInt32(binding.keyCode),
            binding.carbonModifiers,
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &carbonHotKey
        )
        return registerStatus == noErr
    }

    private static let carbonEventHandler: EventHandlerUPP = { _, eventRef, _ in
        guard let eventRef else { return noErr }

        var eventHotKeyID = EventHotKeyID()
        let status = GetEventParameter(
            eventRef,
            EventParamName(kEventParamDirectObject),
            EventParamType(typeEventHotKeyID),
            nil,
            MemoryLayout<EventHotKeyID>.size,
            nil,
            &eventHotKeyID
        )
        guard status == noErr,
              eventHotKeyID.signature == hotKeySignature,
              eventHotKeyID.id == HotkeyManager.hotKeyID else {
            return noErr
        }

        let phase: ShortcutPhase
        switch GetEventKind(eventRef) {
        case UInt32(kEventHotKeyPressed):
            phase = .pressed
        case UInt32(kEventHotKeyReleased):
            phase = .released
        default:
            return noErr
        }

        Task { @MainActor in
            HotkeyManager.instance?.onShortcutEvent?(
                ShortcutEvent(command: .dictation, phase: phase)
            )
        }
        return noErr
    }

    // MARK: - Enter Interception Tap

    private func installEnterInterceptionTap() -> Bool {
        let eventMask: CGEventMask = 1 << CGEventType.keyDown.rawValue

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: eventMask,
            callback: Self.eventTapCallback,
            userInfo: nil
        ) else {
            return false
        }

        eventTap = tap
        runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)
        return true
    }

    private static let eventTapCallback: CGEventTapCallBack = { _, type, event, _ in
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            Task { @MainActor in HotkeyManager.instance?.reenable() }
            return Unmanaged.passRetained(event)
        }

        guard type == .keyDown else {
            return Unmanaged.passRetained(event)
        }

        if sessionActive {
            let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))
            if keyCode == 36 || keyCode == 76 { // Return or numpad Enter
                Task { @MainActor in HotkeyManager.instance?.onEnterDuringSession?() }
                return nil
            }
        }

        return Unmanaged.passRetained(event)
    }

    // MARK: - Helpers

    private func reenable() {
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: true)
            yuwpLog("Event tap re-enabled")
        }
    }
}

private extension KeyBinding {
    var carbonModifiers: UInt32 {
        var result: UInt32 = 0
        if modifiers & 0x100000 != 0 { result |= UInt32(cmdKey) }
        if modifiers & 0x80000 != 0 { result |= UInt32(optionKey) }
        if modifiers & 0x40000 != 0 { result |= UInt32(controlKey) }
        if modifiers & 0x20000 != 0 { result |= UInt32(shiftKey) }
        return result
    }
}
