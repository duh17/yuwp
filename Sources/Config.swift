import Foundation

/// How the dictation shortcut behaves.
enum DictationInteractionMode: String, Sendable, CaseIterable {
    case toggle
    case pushToTalk

    var description: String {
        switch self {
        case .toggle: "Toggle"
        case .pushToTalk: "Push to Talk"
        }
    }
}

/// Whether the bundled ASR server is disabled, local-only, or LAN-visible.
enum ServerMode: String, Sendable, CaseIterable {
    case off
    case localhost
    case allInterfaces

    var description: String {
        switch self {
        case .off: "Off"
        case .localhost: "Localhost"
        case .allInterfaces: "0.0.0.0"
        }
    }

    var bindHost: String? {
        switch self {
        case .off: nil
        case .localhost: "127.0.0.1"
        case .allInterfaces: "0.0.0.0"
        }
    }

    /// The app itself should always talk to the local server through loopback,
    /// even when the server is bound to all interfaces.
    var clientHost: String { "127.0.0.1" }
}

/// Concrete global key combo binding.
struct KeyBinding: Sendable, Codable, Equatable {
    let keyCode: UInt16
    let modifiers: UInt64

    var description: String {
        var parts: [String] = []
        if modifiers & 0x40000 != 0 { parts.append("Ctrl") }
        if modifiers & 0x80000 != 0 { parts.append("⌥") }
        if modifiers & 0x100000 != 0 { parts.append("⌘") }
        if modifiers & 0x20000 != 0 { parts.append("⇧") }
        parts.append(Self.keyName(for: keyCode))
        return parts.joined(separator: "+")
    }

    static func keyName(for keyCode: UInt16) -> String {
        switch keyCode {
        case 2: return "D"
        case 3: return "F"
        case 49: return "Space"
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

extension KeyBinding {
    static let ctrlBacktick = KeyBinding(keyCode: 50, modifiers: 0x40000)
    static let optionSpace = KeyBinding(keyCode: 49, modifiers: 0x80000)
    static let commandShiftD = KeyBinding(keyCode: 2, modifiers: 0x100000 | 0x20000)

    static let presets: [(label: String, binding: KeyBinding)] = [
        ("Ctrl + `", .ctrlBacktick),
        ("⌥ + Space", .optionSpace),
        ("⌘ + ⇧ + D", .commandShiftD),
    ]
}

/// Central configuration for Yuwp.
@MainActor
final class Config {
    static let shared = Config()

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    // MARK: - Dictation

    var dictationInteractionMode: DictationInteractionMode {
        get {
            let raw = defaults.string(forKey: "dictationInteractionMode") ?? DictationInteractionMode.toggle.rawValue
            return DictationInteractionMode(rawValue: raw) ?? .toggle
        }
        set {
            defaults.set(newValue.rawValue, forKey: "dictationInteractionMode")
        }
    }

    var dictationBinding: KeyBinding {
        get {
            let keyCode = UInt16(defaults.integer(forKey: "dictationBindingKeyCode")).nonZero ?? KeyBinding.ctrlBacktick.keyCode
            let modifiers = defaults.object(forKey: "dictationBindingModifiers") != nil
                ? UInt64(defaults.integer(forKey: "dictationBindingModifiers"))
                : KeyBinding.ctrlBacktick.modifiers
            return KeyBinding(keyCode: keyCode, modifiers: modifiers)
        }
        set {
            defaults.set(Int(newValue.keyCode), forKey: "dictationBindingKeyCode")
            defaults.set(Int(newValue.modifiers), forKey: "dictationBindingModifiers")
        }
    }

    var serverMode: ServerMode {
        get {
            let raw = defaults.string(forKey: "serverMode") ?? ServerMode.localhost.rawValue
            return ServerMode(rawValue: raw) ?? .localhost
        }
        set {
            defaults.set(newValue.rawValue, forKey: "serverMode")
        }
    }

    var serverPort: UInt16 {
        get {
            let value = defaults.object(forKey: "serverPort") != nil
                ? defaults.integer(forKey: "serverPort")
                : 9748
            return UInt16(clamping: max(1, min(value, 65_535)))
        }
        set {
            defaults.set(Int(newValue), forKey: "serverPort")
        }
    }

    // MARK: - Model

    /// ASR model for streaming partials (low-latency, runs on every audio chunk)
    var streamingModel: String {
        get { defaults.string(forKey: "streamingModel") ?? "mlx-community/Qwen3-ASR-0.6B-4bit" }
        set { defaults.set(newValue, forKey: "streamingModel") }
    }

    /// ASR model for batch retranscription (final pass on pause/stop)
    var batchModel: String {
        get { defaults.string(forKey: "batchModel") ?? "mlx-community/Qwen3-ASR-0.6B-4bit" }
        set { defaults.set(newValue, forKey: "batchModel") }
    }

    /// Whether to run batch retranscription for a final pass on pause/stop
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

// MARK: - Model Presets

struct ModelPreset {
    let label: String
    let streamingModel: String
    let batchModel: String
    let batchEnabled: Bool

    /// Short description for status display in the menu.
    var summary: String {
        let streaming = Self.shortName(streamingModel)
        if batchEnabled {
            let batch = Self.shortName(batchModel)
            if batch == streaming {
                return "\(streaming) stream+final"
            }
            return "\(streaming) + \(batch) final"
        }
        return "\(streaming) only"
    }

    /// Extract short label: "mlx-community/Qwen3-ASR-1.7B-bf16" -> "1.7B-bf16"
    static func shortName(_ model: String) -> String {
        if let last = model.split(separator: "/").last {
            let parts = last.split(separator: "-").dropFirst(2)
            if !parts.isEmpty {
                return parts.joined(separator: "-")
            }
        }
        return model
    }

    static let presets: [ModelPreset] = [
        ModelPreset(
            label: "Small",
            streamingModel: "mlx-community/Qwen3-ASR-0.6B-4bit",
            batchModel: "mlx-community/Qwen3-ASR-0.6B-4bit",
            batchEnabled: true
        ),
        ModelPreset(
            label: "Large",
            streamingModel: "mlx-community/Qwen3-ASR-1.7B-bf16",
            batchModel: "mlx-community/Qwen3-ASR-1.7B-bf16",
            batchEnabled: true
        ),
    ]

    /// Find which preset matches the current config, if any.
    @MainActor
    static func current() -> ModelPreset? {
        let cfg = Config.shared
        return presets.first { p in
            p.streamingModel == cfg.streamingModel
                && p.batchEnabled == cfg.batchRetranscribeEnabled
                && (!p.batchEnabled || p.batchModel == cfg.batchModel)
        }
    }
}

private extension UInt16 {
    var nonZero: UInt16? { self == 0 ? nil : self }
}
