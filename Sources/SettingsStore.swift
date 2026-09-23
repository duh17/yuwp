import ASRIPC
import Combine
import Foundation

struct SettingsSnapshot: Sendable, Equatable {
    var dictationBinding: KeyBinding
    var audioInputSelection: AudioInputSelection
    var availableAudioInputs: [AudioInputDeviceDescriptor]
    var dictationLanguageMode: DictationLanguageMode
    var fixedDictationLanguage: String
    var supportedDictationLanguages: [String]
    var experimentalDirectTextFieldInsertionEnabled: Bool = false
    var experimentalDirectTerminalInsertionEnabled: Bool = false

    var serverMode: ServerMode
    var serverPort: UInt16
    var asrTransport: ASRIPCTransport

    var transcriptionModel: String
    var batchCommitEnabled: Bool
    var transcriptionDownloadStatus: String?
    var alignerDownloadStatus: String?
    var isModelDownloadInProgress: Bool
    var alignerModelRepoId: String
    var alignerInstalled: Bool

    var saveRecordings: Bool
    var diagnosticLoggingEnabled: Bool
    var recordingsDir: URL
    var usingDefaultRecordingsDir: Bool

    var micPanelAnimation: MicPanelAnimationConfig
    var startChime: DictationChimeConfig
    var stopChime: DictationChimeConfig
    var vocabularyHints: [String] = []
}

@MainActor
final class SettingsStore: ObservableObject {
    enum Alert: String, Identifiable, Equatable {
        case invalidServerPort

        var id: String { rawValue }
    }

    @Published private(set) var snapshot: SettingsSnapshot
    @Published var serverPortDraft: String
    @Published var transcriptionModelDraft: String
    @Published var selectedDownloadModelRepoId: String
    @Published var alert: Alert?

    var onDictationBindingChange: ((KeyBinding) -> Void)?
    var onDictationBindingRecordingChange: ((Bool) -> Void)?
    var onAudioInputSelectionChange: ((AudioInputSelection) -> Void)?
    var onDictationLanguageModeChange: ((DictationLanguageMode) -> Void)?
    var onFixedDictationLanguageChange: ((String) -> Void)?
    var onExperimentalDirectTextFieldInsertionChange: ((Bool) -> Void)?
    var onExperimentalDirectTerminalInsertionChange: ((Bool) -> Void)?
    var onServerModeChange: ((ServerMode) -> Void)?
    var onASRTransportChange: ((ASRIPCTransport) -> Void)?
    var onServerPortChange: ((UInt16) -> Void)?
    var onSaveRecordingsChange: ((Bool) -> Void)?
    var onDiagnosticLoggingChange: ((Bool) -> Void)?
    var onChooseRecordingsDirectory: (() -> Void)?
    var onResetRecordingsDirectory: (() -> Void)?
    var onRevealRecordingsDirectory: (() -> Void)?
    var onChooseModelDirectory: (() -> Void)?
    var onBatchCommitChange: ((Bool) -> Void)?
    var onApplyModelSpec: ((String) -> Void)?
    var onDownloadModel: ((String) -> Void)?
    var onMicPanelAnimationChange: ((MicPanelAnimationConfig) -> Void)?
    var onStartChimeChange: ((DictationChimeConfig) -> Void)?
    var onStopChimeChange: ((DictationChimeConfig) -> Void)?
    var onChooseCustomChime: ((DictationChimeRole) -> Void)?
    var onPreviewChime: ((DictationChimeRole, DictationChimeConfig) -> Void)?

    init(snapshot: SettingsSnapshot) {
        self.snapshot = snapshot
        self.serverPortDraft = "\(snapshot.serverPort)"
        self.transcriptionModelDraft = snapshot.transcriptionModel
        if DownloadableASRModel.supported.contains(where: { $0.repoId == snapshot.transcriptionModel }) {
            self.selectedDownloadModelRepoId = snapshot.transcriptionModel
        } else {
            self.selectedDownloadModelRepoId = DownloadableASRModel.supported.first?.repoId ?? ""
        }
    }

    func sync(_ snapshot: SettingsSnapshot) {
        let previous = self.snapshot
        self.snapshot = snapshot

        if serverPortDraft == "\(previous.serverPort)" {
            serverPortDraft = "\(snapshot.serverPort)"
        }
        if transcriptionModelDraft == previous.transcriptionModel {
            transcriptionModelDraft = snapshot.transcriptionModel
        }
    }

    var audioInputSelections: [AudioInputSelection] {
        var selections: [AudioInputSelection] = [.systemDefault] + snapshot.availableAudioInputs.map(\.selection)
        if case .device = snapshot.audioInputSelection,
           !snapshot.availableAudioInputs.contains(where: { $0.selection == snapshot.audioInputSelection }) {
            selections.append(snapshot.audioInputSelection)
        }
        return selections
    }


    var audioInputDescriptionText: String {
        switch snapshot.audioInputSelection {
        case .systemDefault:
            if let defaultDevice = snapshot.availableAudioInputs.first(where: \.isDefault) {
                return "Follows the current macOS default input: \(defaultDevice.detailText). Best when you switch microphones often."
            }
            return "Follows the current macOS default input device."
        case .device(let uid):
            if let device = snapshot.availableAudioInputs.first(where: { $0.uid == uid }) {
                return "Pinned to \(device.detailText). Yuwp will capture this device’s native format and convert it to 16 kHz mono for transcription."
            }
            return "The selected device is currently unavailable. Yuwp will fall back to the system default input until it reconnects."
        }
    }

    var dictationLanguageModeSummaryText: String {
        snapshot.dictationLanguageMode.summary
    }

    var fixedDictationLanguageDescriptionText: String {
        "Choose a supported language from the active model."
    }

    var dictationLanguageModes: [DictationLanguageMode] {
        DictationLanguageMode.allCases
    }

    var supportedDictationLanguages: [String] {
        snapshot.supportedDictationLanguages
    }

    var serverModeDescriptionText: String {
        switch snapshot.serverMode {
        case .off:
            return "Turns off Yuwp’s bundled transcription server. Dictation won’t work until you turn it back on."
        case .localhost:
            return "Only Yuwp and other apps on this Mac can connect to the server."
        case .allInterfaces:
            return "Makes the server available on your local network so other devices can connect to this Mac. This API is unauthenticated and unencrypted — only use on trusted networks."
        }
    }

    var asrTransportDescriptionText: String {
        switch snapshot.serverMode {
        case .allInterfaces:
            return "Local network mode requires HTTP transport."
        case .localhost:
            return snapshot.asrTransport.settingsDescription
        case .off:
            return "Choose how Yuwp talks to its ASR process when the server is enabled."
        }
    }

    var serverPortDescriptionText: String {
        if snapshot.serverMode == .localhost, snapshot.asrTransport == .stdio {
            return "Standard I/O transport does not use a local TCP port. This value is remembered for HTTP mode and local-network serving."
        }
        return "Use a custom port if you need Yuwp to avoid another local service."
    }

    var recordingsPathText: String {
        let home = NSHomeDirectory()
        let path = snapshot.recordingsDir.path.hasPrefix(home)
            ? "~" + String(snapshot.recordingsDir.path.dropFirst(home.count))
            : snapshot.recordingsDir.path
        return snapshot.usingDefaultRecordingsDir ? "Default location: \(path)" : path
    }

    var modelStatusText: String {
        let name = ModelLocator.displayName(for: snapshot.transcriptionModel)
        let installed = ModelLocator.resolve(snapshot.transcriptionModel) != nil
        return installed ? "Installed: \(name)" : "Missing: \(name)"
    }

    var alignerModelDisplayName: String {
        let shortName = ModelLocator.shortRepoName(snapshot.alignerModelRepoId)
        let withoutForcedAligner = shortName.replacingOccurrences(of: "ForcedAligner-", with: "")
        return withoutForcedAligner.replacingOccurrences(of: "-", with: " ")
    }

    var alignerStatusText: String {
        snapshot.alignerInstalled
            ? "Installed: \(alignerModelDisplayName)"
            : "Missing: \(alignerModelDisplayName)"
    }

    var vocabularyHintsSummaryText: String {
        let phrases = snapshot.vocabularyHints
        if phrases.isEmpty {
            return "None loaded. Yuwp still has built-in hints when the app starts a take."
        }
        return phrases.joined(separator: ", ")
    }

    var vocabularyHintsStatusText: String {
        let count = snapshot.vocabularyHints.count
        if count == 0 {
            return "Phrases sent with each dictation as ASR hints, not replacements."
        }
        return "\(count) phrases sent with each dictation. Built-in Yuwp/Oppi plus Oppi's dictionary. Editing in-app is next."
    }

    var downloadableModels: [DownloadableASRModel] {
        DownloadableASRModel.supported
    }

    var modelDownloadStatusText: String? {
        snapshot.transcriptionDownloadStatus
    }

    var alignerModelDownloadStatusText: String? {
        snapshot.alignerDownloadStatus
    }

    var isModelDownloadInProgress: Bool {
        snapshot.isModelDownloadInProgress
    }

    var selectedDownloadModelIsManagedInstalled: Bool {
        guard !selectedDownloadModelRepoId.isEmpty else { return false }
        return ModelLocator.managedDirectoryIfExists(forRepoId: selectedDownloadModelRepoId) != nil
    }

    var selectedDownloadModelIsCurrent: Bool {
        snapshot.transcriptionModel == selectedDownloadModelRepoId
    }

    var downloadRowStatusText: String {
        if let status = modelDownloadStatusText {
            return status
        }
        if selectedDownloadModelIsCurrent && selectedDownloadModelIsManagedInstalled {
            return "Installed in Application Support and active now."
        }
        if selectedDownloadModelIsManagedInstalled {
            return "Already downloaded in Application Support. Click to switch to this model without downloading again."
        }
        return "Downloads the selected model into Application Support so Yuwp can manage it directly."
    }

    var downloadButtonTitle: String {
        if modelDownloadStatusText != nil {
            return "Downloading…"
        }
        if selectedDownloadModelIsCurrent && selectedDownloadModelIsManagedInstalled {
            return "Current"
        }
        if selectedDownloadModelIsManagedInstalled {
            return "Use Downloaded"
        }
        return "Download"
    }

    var canDownloadSelectedModel: Bool {
        !selectedDownloadModelRepoId.isEmpty
            && !snapshot.isModelDownloadInProgress
            && !(selectedDownloadModelIsCurrent && selectedDownloadModelIsManagedInstalled)
    }

    var alignerDownloadRowStatusText: String {
        if let status = alignerModelDownloadStatusText {
            return status
        }
        if snapshot.alignerInstalled {
            return "Installed locally and ready for timestamped transcription and subtitles."
        }
        return "Not bundled. Download to enable timestamped transcription and subtitles (SRT/VTT/verbose JSON)."
    }

    var alignerDownloadButtonTitle: String {
        if alignerModelDownloadStatusText != nil {
            return "Downloading…"
        }
        return snapshot.alignerInstalled ? "Installed" : "Download"
    }

    var canDownloadAligner: Bool {
        !snapshot.alignerInstalled && !snapshot.isModelDownloadInProgress
    }

    var micPanelAnimationSummaryText: String {
        snapshot.micPanelAnimation.selection.summary
    }

    var micPanelAnimationSelection: MicPanelAnimationSelection {
        snapshot.micPanelAnimation.selection
    }

    var micPanelAnimationCustom: MicPanelAnimationCustom {
        snapshot.micPanelAnimation.customOrDefault
    }

    func chimeConfig(for role: DictationChimeRole) -> DictationChimeConfig {
        switch role {
        case .start: snapshot.startChime
        case .stop: snapshot.stopChime
        }
    }

    func chimeSummaryText(for role: DictationChimeRole) -> String {
        chimeConfig(for: role).selection.summary
    }

    func customChimeDisplayName(for role: DictationChimeRole) -> String {
        chimeConfig(for: role).customAsset?.displayName ?? "No file selected"
    }

    func setDictationBinding(_ binding: KeyBinding) {
        snapshot.dictationBinding = binding
        onDictationBindingChange?(binding)
    }

    func setDictationBindingRecording(_ isRecording: Bool) {
        onDictationBindingRecordingChange?(isRecording)
    }

    func setAudioInputSelection(_ selection: AudioInputSelection) {
        snapshot.audioInputSelection = selection
        onAudioInputSelectionChange?(selection)
    }

    func setDictationLanguageMode(_ mode: DictationLanguageMode) {
        snapshot.dictationLanguageMode = mode
        onDictationLanguageModeChange?(mode)
    }

    func setFixedDictationLanguage(_ language: String) {
        guard snapshot.fixedDictationLanguage != language else { return }
        snapshot.fixedDictationLanguage = language
        onFixedDictationLanguageChange?(language)
    }

    func setExperimentalDirectTextFieldInsertionEnabled(_ enabled: Bool) {
        snapshot.experimentalDirectTextFieldInsertionEnabled = enabled
        onExperimentalDirectTextFieldInsertionChange?(enabled)
    }

    func setExperimentalDirectTerminalInsertionEnabled(_ enabled: Bool) {
        snapshot.experimentalDirectTerminalInsertionEnabled = enabled
        onExperimentalDirectTerminalInsertionChange?(enabled)
    }

    func setServerMode(_ mode: ServerMode) {
        snapshot.serverMode = mode
        onServerModeChange?(mode)
    }

    func setASRTransport(_ transport: ASRIPCTransport) {
        snapshot.asrTransport = transport
        onASRTransportChange?(transport)
    }

    func applyServerPortDraft() {
        let trimmed = serverPortDraft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = Int(trimmed), (1...65_535).contains(value), let port = UInt16(exactly: value) else {
            alert = .invalidServerPort
            return
        }
        alert = nil
        onServerPortChange?(port)
        serverPortDraft = "\(port)"
    }

    func dismissAlert() {
        alert = nil
    }

    func setSaveRecordings(_ enabled: Bool) {
        snapshot.saveRecordings = enabled
        onSaveRecordingsChange?(enabled)
    }

    func setDiagnosticLoggingEnabled(_ enabled: Bool) {
        snapshot.diagnosticLoggingEnabled = enabled
        onDiagnosticLoggingChange?(enabled)
    }

    func chooseRecordingsDirectory() {
        onChooseRecordingsDirectory?()
    }

    func resetRecordingsDirectory() {
        onResetRecordingsDirectory?()
    }

    func revealRecordingsDirectory() {
        onRevealRecordingsDirectory?()
    }

    func setBatchCommitEnabled(_ enabled: Bool) {
        snapshot.batchCommitEnabled = enabled
        onBatchCommitChange?(enabled)
    }

    func applyTranscriptionModelDraft() {
        onApplyModelSpec?(transcriptionModelDraft)
    }

    func chooseModelDirectory() {
        onChooseModelDirectory?()
    }

    func downloadModel(repoId: String) {
        guard !repoId.isEmpty else { return }
        onDownloadModel?(repoId)
    }

    func downloadSelectedModel() {
        downloadModel(repoId: selectedDownloadModelRepoId)
    }

    func setMicPanelAnimationSelection(_ selection: MicPanelAnimationSelection) {
        var config = snapshot.micPanelAnimation
        config.selection = selection
        if selection == .custom, config.custom == nil {
            config.custom = .default
        }
        snapshot.micPanelAnimation = config
        onMicPanelAnimationChange?(config)
    }

    func updateMicPanelAnimationCustom(_ update: (inout MicPanelAnimationCustom) -> Void) {
        var config = snapshot.micPanelAnimation
        config.selection = .custom
        var custom = config.customOrDefault
        update(&custom)
        config.custom = custom.clamped
        snapshot.micPanelAnimation = config
        onMicPanelAnimationChange?(config)
    }

    func setChimeSelection(_ selection: DictationChimeSelection, for role: DictationChimeRole) {
        var config = chimeConfig(for: role)
        config.selection = selection
        switch role {
        case .start:
            snapshot.startChime = config
            onStartChimeChange?(config)
        case .stop:
            snapshot.stopChime = config
            onStopChimeChange?(config)
        }
    }

    func chooseCustomChime(for role: DictationChimeRole) {
        onChooseCustomChime?(role)
    }

    func previewChime(_ role: DictationChimeRole) {
        onPreviewChime?(role, chimeConfig(for: role))
    }
}
