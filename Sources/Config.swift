import Foundation

enum KeyBindingActivation: String, Sendable, Codable {
    case singlePress
    case doubleTap

    var descriptionSuffix: String {
        switch self {
        case .singlePress: ""
        case .doubleTap: " (double tap)"
        }
    }
}

enum KeyBindingTiming {
    static let doubleTapTimeout: TimeInterval = 0.4
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
    let activation: KeyBindingActivation

    init(keyCode: UInt16, modifiers: UInt64, activation: KeyBindingActivation = .singlePress) {
        self.keyCode = keyCode
        self.modifiers = modifiers
        self.activation = activation
    }

    var isModifierOnly: Bool {
        modifiers == 0 && Self.isModifierKeyCode(keyCode)
    }

    var normalized: KeyBinding {
        guard activation == .doubleTap, !isModifierOnly else { return self }
        return KeyBinding(keyCode: keyCode, modifiers: modifiers)
    }

    var description: String {
        var parts: [String] = []
        if modifiers & 0x40000 != 0 { parts.append("Ctrl") }
        if modifiers & 0x80000 != 0 { parts.append("⌥") }
        if modifiers & 0x100000 != 0 { parts.append("⌘") }
        if modifiers & 0x20000 != 0 { parts.append("⇧") }
        parts.append(Self.keyName(for: keyCode))
        return parts.joined(separator: "+") + activation.descriptionSuffix
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

enum MicPanelAnimationSelection: String, Codable, Sendable, CaseIterable {
    case system
    case standard
    case calm
    case lively
    case still
    case custom

    var title: String {
        switch self {
        case .system: "System Default"
        case .standard: "Standard"
        case .calm: "Calm"
        case .lively: "Lively"
        case .still: "Still"
        case .custom: "Custom"
        }
    }

    var summary: String {
        switch self {
        case .system: "Follows macOS Reduce Motion and uses Yuwp’s recommended default."
        case .standard: "Matches the current default mic panel behavior."
        case .calm: "Softer motion and gentler glow for lower visual noise."
        case .lively: "Faster response and stronger motion for a more animated feel."
        case .still: "Minimal movement with subdued glow."
        case .custom: "Use custom tuning values for motion and glow."
        }
    }
}

struct MicPanelAnimationCustom: Codable, Equatable, Sendable {
    var smoothingAttack: Double
    var smoothingDecay: Double
    var phaseStep: Double
    var idleBarAmplitude: Double
    var levelBarScale: Double
    var glowWidthBase: Double
    var glowWidthScale: Double
    var glowAlphaBase: Double
    var glowAlphaScale: Double

    static let `default` = MicPanelAnimationCustom(
        smoothingAttack: 0.4,
        smoothingDecay: 0.15,
        phaseStep: 0.08,
        idleBarAmplitude: 2.0,
        levelBarScale: 1.0,
        glowWidthBase: 0.5,
        glowWidthScale: 1.5,
        glowAlphaBase: 0.08,
        glowAlphaScale: 0.35
    )

    var clamped: MicPanelAnimationCustom {
        MicPanelAnimationCustom(
            smoothingAttack: smoothingAttack.clamped(to: 0.05...1.0),
            smoothingDecay: smoothingDecay.clamped(to: 0.02...1.0),
            phaseStep: phaseStep.clamped(to: 0.0...0.3),
            idleBarAmplitude: idleBarAmplitude.clamped(to: 0.0...6.0),
            levelBarScale: levelBarScale.clamped(to: 0.2...1.8),
            glowWidthBase: glowWidthBase.clamped(to: 0.0...3.0),
            glowWidthScale: glowWidthScale.clamped(to: 0.0...4.0),
            glowAlphaBase: glowAlphaBase.clamped(to: 0.0...0.4),
            glowAlphaScale: glowAlphaScale.clamped(to: 0.0...0.8)
        )
    }
}

struct MicPanelAnimationTuning: Equatable, Sendable {
    let smoothingAttack: Double
    let smoothingDecay: Double
    let phaseStep: Double
    let idleBarAmplitude: Double
    let levelBarScale: Double
    let glowWidthBase: Double
    let glowWidthScale: Double
    let glowAlphaBase: Double
    let glowAlphaScale: Double

    init(_ custom: MicPanelAnimationCustom) {
        let value = custom.clamped
        self.smoothingAttack = value.smoothingAttack
        self.smoothingDecay = value.smoothingDecay
        self.phaseStep = value.phaseStep
        self.idleBarAmplitude = value.idleBarAmplitude
        self.levelBarScale = value.levelBarScale
        self.glowWidthBase = value.glowWidthBase
        self.glowWidthScale = value.glowWidthScale
        self.glowAlphaBase = value.glowAlphaBase
        self.glowAlphaScale = value.glowAlphaScale
    }
}

struct MicPanelAnimationConfig: Codable, Equatable, Sendable {
    var selection: MicPanelAnimationSelection
    var custom: MicPanelAnimationCustom?

    static let `default` = MicPanelAnimationConfig(selection: .system, custom: nil)

    var customOrDefault: MicPanelAnimationCustom {
        (custom ?? .default).clamped
    }

    func resolvedSelection(reduceMotion: Bool) -> MicPanelAnimationSelection {
        switch selection {
        case .system:
            reduceMotion ? .calm : .standard
        default:
            selection
        }
    }

    func resolvedTuning(reduceMotion: Bool) -> MicPanelAnimationTuning {
        switch resolvedSelection(reduceMotion: reduceMotion) {
        case .system, .standard:
            MicPanelAnimationTuning(.default)
        case .calm:
            MicPanelAnimationTuning(MicPanelAnimationCustom(
                smoothingAttack: 0.28,
                smoothingDecay: 0.10,
                phaseStep: 0.04,
                idleBarAmplitude: 0.8,
                levelBarScale: 0.9,
                glowWidthBase: 0.4,
                glowWidthScale: 0.9,
                glowAlphaBase: 0.05,
                glowAlphaScale: 0.18
            ))
        case .lively:
            MicPanelAnimationTuning(MicPanelAnimationCustom(
                smoothingAttack: 0.55,
                smoothingDecay: 0.22,
                phaseStep: 0.12,
                idleBarAmplitude: 3.0,
                levelBarScale: 1.15,
                glowWidthBase: 0.65,
                glowWidthScale: 2.0,
                glowAlphaBase: 0.10,
                glowAlphaScale: 0.5
            ))
        case .still:
            MicPanelAnimationTuning(MicPanelAnimationCustom(
                smoothingAttack: 0.20,
                smoothingDecay: 0.08,
                phaseStep: 0.0,
                idleBarAmplitude: 0.0,
                levelBarScale: 0.65,
                glowWidthBase: 0.4,
                glowWidthScale: 0.5,
                glowAlphaBase: 0.04,
                glowAlphaScale: 0.12
            ))
        case .custom:
            MicPanelAnimationTuning(customOrDefault)
        }
    }
}

enum DictationChimeSelection: String, Codable, Sendable, CaseIterable {
    case systemDefault
    case soft
    case mechanical
    case none
    case custom

    var title: String {
        switch self {
        case .systemDefault: "System Default"
        case .soft: "Soft"
        case .mechanical: "Mechanical"
        case .none: "Muted"
        case .custom: "Custom"
        }
    }

    var summary: String {
        switch self {
        case .systemDefault: "Use Yuwp’s built-in start and stop sounds."
        case .soft: "Quieter built-in sounds with a gentler feel."
        case .mechanical: "Sharper built-in sounds with a more tactile feel."
        case .none: "Disable this sound."
        case .custom: "Use an imported audio file from Application Support."
        }
    }
}

struct ImportedSoundAsset: Codable, Equatable, Sendable {
    var relativePath: String
    var displayName: String
}

struct DictationChimeConfig: Codable, Equatable, Sendable {
    var selection: DictationChimeSelection
    var customAsset: ImportedSoundAsset?

    static let `default` = DictationChimeConfig(selection: .systemDefault, customAsset: nil)
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

    var dictationBinding: KeyBinding {
        get {
            let keyCode = UInt16(defaults.integer(forKey: "dictationBindingKeyCode")).nonZero ?? KeyBinding.ctrlBacktick.keyCode
            let modifiers = defaults.object(forKey: "dictationBindingModifiers") != nil
                ? UInt64(defaults.integer(forKey: "dictationBindingModifiers"))
                : KeyBinding.ctrlBacktick.modifiers
            let activationRaw = defaults.string(forKey: "dictationBindingActivation") ?? KeyBindingActivation.singlePress.rawValue
            let activation = KeyBindingActivation(rawValue: activationRaw) ?? .singlePress
            return KeyBinding(keyCode: keyCode, modifiers: modifiers, activation: activation).normalized
        }
        set {
            let binding = newValue.normalized
            defaults.set(Int(binding.keyCode), forKey: "dictationBindingKeyCode")
            defaults.set(Int(binding.modifiers), forKey: "dictationBindingModifiers")
            defaults.set(binding.activation.rawValue, forKey: "dictationBindingActivation")
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

    /// Experimental: when enabled, Yuwp may inject directly into editable AX fields.
    /// Default is off — bubble preview + paste commit is safer and more predictable.
    var experimentalDirectTextFieldInsertionEnabled: Bool {
        get {
            defaults.object(forKey: "experimentalDirectTextFieldInsertionEnabled") != nil
                ? defaults.bool(forKey: "experimentalDirectTextFieldInsertionEnabled")
                : false
        }
        set {
            defaults.set(newValue, forKey: "experimentalDirectTextFieldInsertionEnabled")
        }
    }

    /// Experimental: when enabled, Yuwp may inject directly in terminal/AX-hostile apps via CGEvent.
    /// Default is off — bubble preview + paste commit is safer and more predictable.
    var experimentalDirectTerminalInsertionEnabled: Bool {
        get {
            defaults.object(forKey: "experimentalDirectTerminalInsertionEnabled") != nil
                ? defaults.bool(forKey: "experimentalDirectTerminalInsertionEnabled")
                : false
        }
        set {
            defaults.set(newValue, forKey: "experimentalDirectTerminalInsertionEnabled")
        }
    }

    var micPanelAnimation: MicPanelAnimationConfig {
        get {
            let selection = MicPanelAnimationSelection(rawValue: defaults.string(forKey: "micPanelAnimationSelection") ?? "") ?? .system
            let custom: MicPanelAnimationCustom? = decode(MicPanelAnimationCustom.self, forKey: "micPanelAnimationCustom")
            return MicPanelAnimationConfig(selection: selection, custom: custom)
        }
        set {
            defaults.set(newValue.selection.rawValue, forKey: "micPanelAnimationSelection")
            encode(newValue.custom?.clamped, forKey: "micPanelAnimationCustom")
        }
    }

    var startChime: DictationChimeConfig {
        get {
            let selection = DictationChimeSelection(rawValue: defaults.string(forKey: "startChimeSelection") ?? "") ?? .systemDefault
            let customAsset: ImportedSoundAsset? = decode(ImportedSoundAsset.self, forKey: "startChimeCustomAsset")
            return DictationChimeConfig(selection: selection, customAsset: customAsset)
        }
        set {
            defaults.set(newValue.selection.rawValue, forKey: "startChimeSelection")
            encode(newValue.customAsset, forKey: "startChimeCustomAsset")
        }
    }

    var stopChime: DictationChimeConfig {
        get {
            let selection = DictationChimeSelection(rawValue: defaults.string(forKey: "stopChimeSelection") ?? "") ?? .systemDefault
            let customAsset: ImportedSoundAsset? = decode(ImportedSoundAsset.self, forKey: "stopChimeCustomAsset")
            return DictationChimeConfig(selection: selection, customAsset: customAsset)
        }
        set {
            defaults.set(newValue.selection.rawValue, forKey: "stopChimeSelection")
            encode(newValue.customAsset, forKey: "stopChimeCustomAsset")
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

    // MARK: - Privacy

    /// Diagnostic stderr logging for troubleshooting. Default is off.
    var diagnosticLoggingEnabled: Bool {
        get {
            defaults.object(forKey: "diagnosticLoggingEnabled") != nil
                ? defaults.bool(forKey: "diagnosticLoggingEnabled")
                : false
        }
        set { defaults.set(newValue, forKey: "diagnosticLoggingEnabled") }
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

    private func decode<T: Decodable>(_ type: T.Type, forKey key: String) -> T? {
        guard let data = defaults.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode(type, from: data)
    }

    private func encode<T: Encodable>(_ value: T?, forKey key: String) {
        guard let value else {
            defaults.removeObject(forKey: key)
            return
        }
        defaults.set(try? JSONEncoder().encode(value), forKey: key)
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

private extension Double {
    func clamped(to range: ClosedRange<Double>) -> Double {
        min(max(self, range.lowerBound), range.upperBound)
    }
}
