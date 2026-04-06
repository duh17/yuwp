import AppKit
import CoreGraphics

/// Detects global hotkeys via CGEvent tap.
///
/// Two modes:
///   - **combo**: modifier + key (e.g., Ctrl + `)
///   - **doubleTap**: tap a modifier key twice quickly (e.g., double-tap Right Option)
///
/// Double-tap detection tracks clean taps only — if any other key is pressed
/// while the modifier is held (e.g., using Option as an actual modifier),
/// that tap is ignored.
@MainActor
final class HotkeyManager {
    var onToggle: (@Sendable () -> Void)?

    private var eventTap: CFMachPort?
    private var runLoopSource: CFRunLoopSource?

    // Static state for the C callback (no captures allowed)
    nonisolated(unsafe) private static var instance: HotkeyManager?
    nonisolated(unsafe) private static var activeMode: HotkeyMode = .doubleTapRightOption

    // Double-tap state machine
    nonisolated(unsafe) private static var lastCleanTapTime: CFAbsoluteTime = 0
    nonisolated(unsafe) private static var modifierIsDown = false
    nonisolated(unsafe) private static var tapIsDirty = false

    /// Start the event tap. Returns false if Accessibility permission is missing.
    func start() -> Bool {
        stop()

        let mode = Config.shared.hotkeyMode
        HotkeyManager.instance = self
        HotkeyManager.activeMode = mode
        HotkeyManager.lastCleanTapTime = 0
        HotkeyManager.modifierIsDown = false
        HotkeyManager.tapIsDirty = false

        let eventMask: CGEventMask
        switch mode {
        case .combo:
            eventMask = (1 << CGEventType.keyDown.rawValue)
        case .doubleTap:
            // flagsChanged for modifier press/release + keyDown to detect dirty taps
            eventMask = (1 << CGEventType.flagsChanged.rawValue)
                      | (1 << CGEventType.keyDown.rawValue)
        }

        guard let tap = CGEvent.tapCreate(
            tap: .cgSessionEventTap,
            place: .headInsertEventTap,
            options: .defaultTap,
            eventsOfInterest: eventMask,
            callback: Self._eventCallback,
            userInfo: nil
        ) else {
            yuwpLog("Failed to create event tap — Accessibility permission required")
            return false
        }

        eventTap = tap
        runLoopSource = CFMachPortCreateRunLoopSource(kCFAllocatorDefault, tap, 0)
        CFRunLoopAddSource(CFRunLoopGetCurrent(), runLoopSource, .commonModes)
        CGEvent.tapEnable(tap: tap, enable: true)

        yuwpLog("Hotkey registered: \(mode.description)")
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
        HotkeyManager.instance = nil
    }

    /// Restart with new configuration.
    func restart() -> Bool {
        stop()
        return start()
    }

    // MARK: - Event Callback

    private static let _eventCallback: CGEventTapCallBack = {
        _, type, event, _ -> Unmanaged<CGEvent>? in

        // Re-enable if system disabled the tap
        if type == .tapDisabledByTimeout || type == .tapDisabledByUserInput {
            Task { @MainActor in HotkeyManager.instance?.reenable() }
            return Unmanaged.passRetained(event)
        }

        switch activeMode {
        case .combo(let targetKey, let targetMods):
            return handleCombo(type: type, event: event,
                               targetKey: targetKey, targetMods: targetMods)

        case .doubleTap(let targetKey, let interval):
            return handleDoubleTap(type: type, event: event,
                                   targetKey: targetKey, interval: interval)
        }
    }

    // MARK: - Combo Mode

    private static func handleCombo(
        type: CGEventType, event: CGEvent,
        targetKey: UInt16, targetMods: UInt64
    ) -> Unmanaged<CGEvent>? {
        guard type == .keyDown else { return Unmanaged.passRetained(event) }

        let keyCode = event.getIntegerValueField(.keyboardEventKeycode)
        guard keyCode == Int64(targetKey) else { return Unmanaged.passRetained(event) }

        // Check modifiers (mask to device-independent bits)
        let eventMods = event.flags.rawValue & 0x1F0000
        guard (eventMods & targetMods) == targetMods else { return Unmanaged.passRetained(event) }

        Task { @MainActor in HotkeyManager.instance?.onToggle?() }
        return nil // Swallow the event
    }

    // MARK: - Double-Tap Mode

    private static func handleDoubleTap(
        type: CGEventType, event: CGEvent,
        targetKey: UInt16, interval: TimeInterval
    ) -> Unmanaged<CGEvent>? {
        let keyCode = UInt16(event.getIntegerValueField(.keyboardEventKeycode))

        // Any non-modifier keyDown while our modifier is held → dirty tap
        if type == .keyDown && keyCode != targetKey {
            if modifierIsDown { tapIsDirty = true }
            return Unmanaged.passRetained(event)
        }

        // Only handle flagsChanged for our target modifier key
        guard type == .flagsChanged && keyCode == targetKey else {
            return Unmanaged.passRetained(event)
        }

        // Determine press/release using device-specific flag bits
        let mask = deviceMask(for: targetKey)
        let keyIsDown = (event.flags.rawValue & mask) != 0

        if keyIsDown && !modifierIsDown {
            // Modifier pressed
            modifierIsDown = true
            tapIsDirty = false

        } else if !keyIsDown && modifierIsDown {
            // Modifier released
            modifierIsDown = false

            if !tapIsDirty {
                // Clean tap completed — check for double-tap
                let now = CFAbsoluteTimeGetCurrent()
                let elapsed = now - lastCleanTapTime

                if elapsed < interval && lastCleanTapTime > 0 {
                    // Double-tap detected!
                    lastCleanTapTime = 0
                    Task { @MainActor in HotkeyManager.instance?.onToggle?() }
                } else {
                    lastCleanTapTime = now
                }
            }
        }

        // Never swallow modifier events — let them pass through normally
        return Unmanaged.passRetained(event)
    }

    // MARK: - Helpers

    /// Device-specific modifier masks to distinguish left/right keys.
    /// Based on IOLLEvent.h NX_DEVICE*KEYMASK constants.
    private static func deviceMask(for keyCode: UInt16) -> UInt64 {
        switch keyCode {
        case 58: return 0x20       // Left Option   (NX_DEVICELALTKEYMASK)
        case 61: return 0x40       // Right Option  (NX_DEVICERALTKEYMASK)
        case 59: return 0x01       // Left Control  (NX_DEVICELCTLKEYMASK)
        case 62: return 0x2000     // Right Control (NX_DEVICERCTLKEYMASK)
        case 55: return 0x08       // Left Command  (NX_DEVICELCMDKEYMASK)
        case 54: return 0x10       // Right Command (NX_DEVICERCMDKEYMASK)
        case 56: return 0x02       // Left Shift    (NX_DEVICELSHIFTKEYMASK)
        case 60: return 0x04       // Right Shift   (NX_DEVICERSHIFTKEYMASK)
        case 63: return CGEventFlags.maskSecondaryFn.rawValue  // Fn/Globe
        default: return 0
        }
    }

    private func reenable() {
        if let tap = eventTap {
            CGEvent.tapEnable(tap: tap, enable: true)
            yuwpLog("Event tap re-enabled")
        }
    }
}
