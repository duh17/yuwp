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
        case .localhost: "This Mac only"
        case .allInterfaces: "Local network"
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

    var isModifierOnly: Bool {
        modifiers == 0 && Self.isModifierKeyCode(keyCode)
    }

    var description: String {
        var parts: [String] = []
        if modifiers & 0x40000 != 0 { parts.append("Ctrl") }
        if modifiers & 0x80000 != 0 { parts.append("⌥") }
        if modifiers & 0x100000 != 0 { parts.append("⌘") }
        if modifiers & 0x20000 != 0 { parts.append("⇧") }
        parts.append(Self.keyName(for: keyCode))
        return parts.joined(separator: "+")
    }

    static func isModifierKeyCode(_ keyCode: UInt16) -> Bool {
        switch keyCode {
        case 54, 55, 56, 58, 59, 60, 61, 62, 63:
            true
        default:
            false
        }
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

    var audioInputSelection: AudioInputSelection {
        get {
            AudioInputSelection(persistenceString: defaults.string(forKey: "audioInputSelection"))
        }
        set {
            defaults.set(newValue.persistenceString, forKey: "audioInputSelection")
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

    /// ASR model used for both live decoding and batch segment commits.
    var transcriptionModel: String {
        get {
            defaults.string(forKey: "transcriptionModel")
                ?? defaults.string(forKey: "streamingModel")
                ?? defaults.string(forKey: "batchModel")
                ?? "mlx-community/Qwen3-ASR-0.6B-4bit"
        }
        set { defaults.set(newValue, forKey: "transcriptionModel") }
    }

    /// Whether Yuwp runs a batch pass whenever it commits a segment.
    var batchCommitEnabled: Bool {
        get {
            if defaults.object(forKey: "batchCommitEnabled") != nil {
                return defaults.bool(forKey: "batchCommitEnabled")
            }
            if defaults.object(forKey: "batchRetranscribeEnabled") != nil {
                return defaults.bool(forKey: "batchRetranscribeEnabled")
            }
            return true
        }
        set { defaults.set(newValue, forKey: "batchCommitEnabled") }
    }

    // MARK: - Recordings

    var saveRecordings: Bool {
        get {
            defaults.object(forKey: "saveRecordings") != nil
                ? defaults.bool(forKey: "saveRecordings")
                : false
        }
        set { defaults.set(newValue, forKey: "saveRecordings") }
    }

    var defaultRecordingsDir: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Yuwp/recordings", isDirectory: true)
    }

    var usesDefaultRecordingsDir: Bool {
        let stored = defaults.string(forKey: "recordingsDirPath")?.trimmingCharacters(in: .whitespacesAndNewlines)
        return stored?.isEmpty != false
    }

    var recordingsDir: URL {
        let dir: URL
        if usesDefaultRecordingsDir {
            dir = defaultRecordingsDir
        } else {
            let path = defaults.string(forKey: "recordingsDirPath") ?? defaultRecordingsDir.path
            let expanded = NSString(string: path).expandingTildeInPath
            dir = URL(fileURLWithPath: expanded, isDirectory: true).standardizedFileURL
        }
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func setRecordingsDir(_ url: URL) {
        defaults.set(url.standardizedFileURL.path, forKey: "recordingsDirPath")
    }

    func resetRecordingsDir() {
        defaults.removeObject(forKey: "recordingsDirPath")
    }
}

// MARK: - Model Presets

struct ModelPreset {
    let label: String
    let transcriptionModel: String
    let batchCommitEnabled: Bool

    /// Short description for status display in the menu.
    var summary: String {
        let model = Self.shortName(transcriptionModel)
        return batchCommitEnabled ? "\(model) live+commit" : "\(model) live only"
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
            label: "Fast",
            transcriptionModel: "mlx-community/Qwen3-ASR-0.6B-4bit",
            batchCommitEnabled: true
        ),
        ModelPreset(
            label: "Best Accuracy",
            transcriptionModel: "mlx-community/Qwen3-ASR-1.7B-bf16",
            batchCommitEnabled: true
        ),
    ]

    /// Find which preset matches the current config, if any.
    @MainActor
    static func current() -> ModelPreset? {
        let cfg = Config.shared
        return presets.first { p in
            p.transcriptionModel == cfg.transcriptionModel
                && p.batchCommitEnabled == cfg.batchCommitEnabled
        }
    }
}

private extension UInt16 {
    var nonZero: UInt16? { self == 0 ? nil : self }
}
