import Foundation

/// How the global hotkey triggers dictation.
enum HotkeyMode: Sendable {
    /// Modifier + key combination (e.g., Ctrl + `)
    case combo(keyCode: UInt16, modifiers: UInt64)
    /// Tap a modifier key twice quickly (e.g., double-tap Right Option)
    case doubleTap(keyCode: UInt16, interval: TimeInterval)

    var description: String {
        switch self {
        case .combo(let keyCode, let modifiers):
            var parts: [String] = []
            if modifiers & 0x40000 != 0 { parts.append("Ctrl") }
            if modifiers & 0x80000 != 0 { parts.append("⌥") }
            if modifiers & 0x100000 != 0 { parts.append("⌘") }
            if modifiers & 0x20000 != 0 { parts.append("⇧") }
            parts.append(Self.keyName(for: keyCode))
            return parts.joined(separator: "+")
        case .doubleTap(let keyCode, _):
            return "Double-tap \(Self.keyName(for: keyCode))"
        }
    }

    static func keyName(for keyCode: UInt16) -> String {
        switch keyCode {
        case 50: return "`"
        case 54: return "Right ⌘"
        case 55: return "Left ⌘"
        case 56: return "Left ⇧"
        case 58: return "Left ⌥"
        case 59: return "Left Ctrl"
        case 60: return "Right ⇧"
        case 61: return "Right ⌥"
        case 62: return "Right Ctrl"
        case 63: return "Fn"
        default: return "Key \(keyCode)"
        }
    }
}

// Preset hotkey configurations
extension HotkeyMode {
    static let doubleTapRightControl = HotkeyMode.doubleTap(keyCode: 62, interval: 0.4)
    static let doubleTapRightOption = HotkeyMode.doubleTap(keyCode: 61, interval: 0.4)
    static let doubleTapFn = HotkeyMode.doubleTap(keyCode: 63, interval: 0.4)
    static let ctrlBacktick = HotkeyMode.combo(keyCode: 50, modifiers: 0x40000)

    static let presets: [(label: String, mode: HotkeyMode)] = [
        ("Double-tap Right Ctrl", .doubleTapRightControl),
        ("Double-tap Right ⌥", .doubleTapRightOption),
        ("Double-tap Fn", .doubleTapFn),
        ("Ctrl + `", .ctrlBacktick),
    ]
}

/// Central configuration for Yuwp.
@MainActor
final class Config {
    static let shared = Config()

    private let defaults = UserDefaults.standard

    // MARK: - Hotkey

    var hotkeyMode: HotkeyMode {
        get {
            let mode = defaults.string(forKey: "hotkeyMode") ?? "doubleTap"
            switch mode {
            case "combo":
                let keyCode = UInt16(defaults.integer(forKey: "hotkeyComboKeyCode")).nonZero ?? 50
                let mods = defaults.object(forKey: "hotkeyComboModifiers") != nil
                    ? UInt64(defaults.integer(forKey: "hotkeyComboModifiers"))
                    : 0x40000
                return .combo(keyCode: keyCode, modifiers: mods)
            default:
                let keyCode = UInt16(defaults.integer(forKey: "hotkeyDoubleTapKeyCode")).nonZero ?? 62
                let interval = defaults.object(forKey: "hotkeyDoubleTapInterval") != nil
                    ? defaults.double(forKey: "hotkeyDoubleTapInterval")
                    : 0.4
                return .doubleTap(keyCode: keyCode, interval: interval)
            }
        }
        set {
            switch newValue {
            case .combo(let keyCode, let modifiers):
                defaults.set("combo", forKey: "hotkeyMode")
                defaults.set(Int(keyCode), forKey: "hotkeyComboKeyCode")
                defaults.set(Int(modifiers), forKey: "hotkeyComboModifiers")
            case .doubleTap(let keyCode, let interval):
                defaults.set("doubleTap", forKey: "hotkeyMode")
                defaults.set(Int(keyCode), forKey: "hotkeyDoubleTapKeyCode")
                defaults.set(interval, forKey: "hotkeyDoubleTapInterval")
            }
        }
    }

    // MARK: - Model

    /// ASR model for streaming partials (low-latency, runs on every audio chunk)
    var streamingModel: String {
        get { defaults.string(forKey: "streamingModel") ?? "mlx-community/Qwen3-ASR-0.6B-4bit" }
        set { defaults.set(newValue, forKey: "streamingModel") }
    }

    /// ASR model for batch retranscription (high-quality correction on pause/stop)
    var batchModel: String {
        get { defaults.string(forKey: "batchModel") ?? "mlx-community/Qwen3-ASR-1.7B-bf16" }
        set { defaults.set(newValue, forKey: "batchModel") }
    }

    /// Whether to run batch retranscription with a second model for quality correction
    var batchRetranscribeEnabled: Bool {
        get {
            defaults.object(forKey: "batchRetranscribeEnabled") != nil
                ? defaults.bool(forKey: "batchRetranscribeEnabled")
                : true
        }
        set { defaults.set(newValue, forKey: "batchRetranscribeEnabled") }
    }

    // MARK: - Recordings

    var saveRecordings: Bool {
        get { defaults.bool(forKey: "saveRecordings") }
        set { defaults.set(newValue, forKey: "saveRecordings") }
    }

    var recordingsDir: URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Yuwp/recordings", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}

private extension UInt16 {
    var nonZero: UInt16? { self == 0 ? nil : self }
}
