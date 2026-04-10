import AppKit
import AVFoundation
import Sparkle

// Yuwp — system-wide voice dictation for macOS
// Press hotkey → speak → text streams into any focused text field
// Powered by Qwen3-ASR via native asr-server

/// Log to stderr (unbuffered, visible even when stdout is piped)
func yuwpLog(_ msg: String) {
    FileHandle.standardError.write(Data("[yuwp] \(msg)\n".utf8))
}

@main
struct YuwpApp {
    static func main() {
        let app = NSApplication.shared
        let delegate = AppDelegate()
        app.delegate = delegate
        app.setActivationPolicy(.accessory) // Menu bar only, no dock icon
        app.run()
    }
}

@MainActor
final class AppDelegate: NSObject, NSApplicationDelegate {
    // Infrastructure — live for the app's lifetime
    private let updaterController = AppDelegate.makeUpdaterController()
    private let hotkeyManager = HotkeyManager()
    private let asrProvider: NativeASRProvider = {
        let p = NativeASRProvider(port: Config.shared.serverPort)
        p.serverMode = Config.shared.serverMode
        p.transcriptionModel = Config.shared.transcriptionModel
        p.batchCommitEnabled = Config.shared.batchCommitEnabled
        return p
    }()
    private let audioCapture = AudioCapture()
    private let micPanel = MicPanel()

    // Per-dictation session (created on start, torn down on stop)
    private var session: DictationSession?
    private var dictationHotkeyBehavior = DictationHotkeyBehavior()
    /// Set when Enter was intercepted mid-session; triggers Enter replay after final commit.
    private var pendingEnter = false

    // Menu bar state
    private var statusItem: NSStatusItem!
    private var statusMenuItem: NSMenuItem!
    private var saveRecordingsMenuItem: NSMenuItem!
    private var settingsWindowController: SettingsWindowController?
    private var providerReady = false
    private var hasPermission = false
    private var permissionTimer: Timer?
    private var modelDownloadStatus: String?

    // MARK: - Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        normalizeModelSelection()
        setupMenuBar()
        setupMicPanelDismiss()
        startSttProvider()
        requestMicPermission()
        checkPermission()

        let env = ProcessInfo.processInfo.environment
        let shouldOpenSettings = env["YUWP_OPEN_SETTINGS_ON_LAUNCH"] == "1" || env["YUWP_SETTINGS_SNAPSHOT_PATH"] != nil
        if shouldOpenSettings {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
                self?.openSettings()
                self?.captureSettingsSnapshotIfRequested()
            }
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        asrProvider.shutdown()
    }

    private func captureSettingsSnapshotIfRequested() {
        guard let path = ProcessInfo.processInfo.environment["YUWP_SETTINGS_SNAPSHOT_PATH"], !path.isEmpty else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.5) { [weak self] in
            self?.captureSettingsSnapshot(to: path)
        }
    }

    private func captureSettingsSnapshot(to path: String) {
        defer { NSApp.terminate(nil) }

        guard let view = settingsWindowController?.window?.contentView else {
            yuwpLog("Failed to capture settings snapshot — no settings content view")
            return
        }

        view.layoutSubtreeIfNeeded()
        let bounds = view.bounds
        guard let rep = view.bitmapImageRepForCachingDisplay(in: bounds) else {
            yuwpLog("Failed to capture settings snapshot — bitmap rep unavailable")
            return
        }

        view.cacheDisplay(in: bounds, to: rep)
        guard let data = rep.representation(using: .png, properties: [:]) else {
            yuwpLog("Failed to capture settings snapshot — PNG encoding failed")
            return
        }

        do {
            try data.write(to: URL(fileURLWithPath: path))
            yuwpLog("Saved settings snapshot to: \(path)")
        } catch {
            yuwpLog("Failed to write settings snapshot: \(error.localizedDescription)")
        }
    }

    private static func makeUpdaterController() -> SPUStandardUpdaterController? {
        let info = Bundle.main.infoDictionary ?? [:]
        let feedURL = (info["SUFeedURL"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let publicKey = (info["SUPublicEDKey"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !feedURL.isEmpty, !publicKey.isEmpty else {
            yuwpLog("Sparkle disabled — missing SUFeedURL or SUPublicEDKey")
            return nil
        }
        return SPUStandardUpdaterController(
            startingUpdater: true, updaterDelegate: nil, userDriverDelegate: nil
        )
    }

    // MARK: - Hotkey Toggle

    private func normalizeModelSelection() {
        let shared = Config.shared
        shared.transcriptionModel = shared.transcriptionModel
        shared.batchCommitEnabled = shared.batchCommitEnabled
        asrProvider.transcriptionModel = shared.transcriptionModel
        asrProvider.batchCommitEnabled = shared.batchCommitEnabled
    }

    private func setupMicPanelDismiss() {
        micPanel.onDismiss = { [weak self] in
            Task { @MainActor in
                guard let self, self.session?.isActive == true else { return }
                yuwpLog("Escape pressed — stopping dictation")
                self.stopDictation()
            }
        }

        hotkeyManager.onEnterDuringSession = { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                yuwpLog("Enter intercepted — stopping dictation, will replay Enter after commit")
                self.pendingEnter = true
                self.stopDictation()
            }
        }
    }

    private func handleShortcutEvent(_ event: ShortcutEvent) {
        guard event.command == .dictation else { return }

        let action = dictationHotkeyBehavior.handle(
            phase: event.phase,
            mode: Config.shared.dictationInteractionMode,
            isSessionActive: session?.isActive == true
        )

        switch action {
        case .start:
            startDictation()
        case .stop:
            stopDictation()
        case .none:
            break
        }
    }

    private func startDictation() {
        guard session == nil else { return }
        guard Config.shared.serverMode != .off else {
            yuwpLog("Server mode is off — enable This Mac only or Local network to dictate")
            return
        }
        guard providerReady else {
            let missingModels = missingConfiguredModelLabels()
            if missingModels.isEmpty {
                yuwpLog("Model still loading, please wait...")
            } else {
                yuwpLog("\(missingModels.joined(separator: " + ")) model missing — open Settings → Transcription to fix it")
            }
            return
        }

        HotkeyManager.sessionActive = true
        let injector = TextInjectorFactory.capture()
        let s = DictationSession(
            sttSession: asrProvider.makeSession(),
            textInjector: injector,
            audioCapture: audioCapture
        )
        s.onEvent = { [weak self] event in self?.handleSessionEvent(event) }
        s.onRequestStop = { [weak self] in
            Task { @MainActor in self?.stopDictation() }
        }
        session = s

        // Update menu bar icon
        statusItem.button?.image = NSImage(
            systemSymbolName: "waveform.circle.fill",
            accessibilityDescription: "Yuwp — Listening"
        )

        // Start with minimal waveform pill. Upgrades to full text pill
        // only if we enter clipboard-fallback mode (first .partialTranscript event).
        micPanel.show(near: injector.targetPosition, minimal: true)

        s.start()
    }

    private func stopDictation() {
        guard let s = session else { return }
        HotkeyManager.sessionActive = false
        dictationHotkeyBehavior.sessionDidEnd()
        let pcmData = s.stop()

        // Hide panel and reset icon immediately — don't wait for server's final
        micPanel.hide()
        statusItem.button?.image = NSImage(
            systemSymbolName: "waveform",
            accessibilityDescription: "Yuwp"
        )

        if Config.shared.saveRecordings, let pcmData, !pcmData.isEmpty {
            saveRecording(pcmData)
        }
    }

    // MARK: - Session Events → UI

    private func handleSessionEvent(_ event: DictationEvent) {
        switch event {
        case .liveInjectionVerified:
            break // already showing minimal pill

        case .partialTranscript(let text):
            // Non-live mode — upgrade from compact dot to full pill to show text
            micPanel.show(near: session?.textInjector.targetPosition ?? .zero)
            micPanel.updateTranscript(text)

        case .caretMoved:
            break // pill stays in pinned position

        case .audioLevel(let level):
            micPanel.updateAudioLevel(level)

        case .finished:
            micPanel.hide()
            HotkeyManager.sessionActive = false
            dictationHotkeyBehavior.sessionDidEnd()
            statusItem.button?.image = NSImage(
                systemSymbolName: "waveform",
                accessibilityDescription: "Yuwp"
            )
            session = nil
            if pendingEnter {
                pendingEnter = false
                replayEnterKey()
            }
        }
    }

    // MARK: - STT Provider

    private func startSttProvider() {
        asrProvider.onStateChange = { [weak self] state in
            Task { @MainActor in
                guard let self else { return }
                self.providerReady = state == .ready
                self.updateStatus()
                if state == .ready {
                    yuwpLog("STT provider ready")
                }
            }
        }
        asrProvider.onError = { error in
            Task { @MainActor in yuwpLog("STT error: \(error)") }
        }
        asrProvider.start()
    }

    // MARK: - Microphone Permission

    private func requestMicPermission() {
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            Task { @MainActor in
                yuwpLog(granted ? "Microphone permission granted" : "Microphone permission denied")
            }
        }
    }

    // MARK: - Accessibility Permission

    private func checkPermission() {
        hotkeyManager.onShortcutEvent = { [weak self] event in
            Task { @MainActor in self?.handleShortcutEvent(event) }
        }

        if hotkeyManager.start() {
            onPermissionGranted()
        } else {
            let opts = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
            _ = AXIsProcessTrustedWithOptions(opts)

            updateStatus()
            yuwpLog("Waiting for Accessibility permission...")

            permissionTimer = Timer.scheduledTimer(withTimeInterval: 2.0, repeats: true) {
                [weak self] _ in
                Task { @MainActor in
                    guard let self else { return }
                    if self.hotkeyManager.start() {
                        self.permissionTimer?.invalidate()
                        self.permissionTimer = nil
                        self.onPermissionGranted()
                    }
                }
            }
        }
    }

    private func onPermissionGranted() {
        hasPermission = true
        updateStatus()
        yuwpLog("Ready. \(Config.shared.dictationBinding.description) to dictate.")
    }

    @objc private func openAccessibilitySettings() {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!
        NSWorkspace.shared.open(url)
    }

    // MARK: - Menu Bar

    private func setupMenuBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        guard let button = statusItem.button else { return }
        button.image = NSImage(systemSymbolName: "waveform", accessibilityDescription: "Yuwp")

        let menu = NSMenu()

        statusMenuItem = NSMenuItem(title: "Loading model...", action: nil, keyEquivalent: "")
        statusMenuItem.isEnabled = false
        menu.addItem(statusMenuItem)
        menu.addItem(.separator())

        let settingsItem = NSMenuItem(title: "Settings…", action: #selector(openSettings), keyEquivalent: ",")
        settingsItem.target = self
        menu.addItem(settingsItem)

        menu.addItem(.separator())

        saveRecordingsMenuItem = NSMenuItem(title: "Save Recordings", action: #selector(toggleSaveRecordings(_:)), keyEquivalent: "")
        saveRecordingsMenuItem.target = self
        saveRecordingsMenuItem.state = Config.shared.saveRecordings ? .on : .off
        menu.addItem(saveRecordingsMenuItem)

        if let updaterController {
            menu.addItem(.separator())
            let updateItem = NSMenuItem(
                title: "Check for Updates...",
                action: #selector(SPUStandardUpdaterController.checkForUpdates(_:)),
                keyEquivalent: ""
            )
            updateItem.target = updaterController
            menu.addItem(updateItem)
        }

        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit Yuwp", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

        statusItem.menu = menu
    }

    @objc private func openSettings() {
        let controller = settingsWindowController ?? SettingsWindowController()
        if settingsWindowController == nil {
            controller.onDictationModeChange = { [weak self] mode in
                self?.applyDictationMode(mode)
            }
            controller.onDictationBindingChange = { [weak self] binding in
                self?.applyDictationBinding(binding)
            }
            controller.onServerModeChange = { [weak self] mode in
                self?.applyServerMode(mode)
            }
            controller.onServerPortChange = { [weak self] port in
                self?.applyServerPort(port)
            }
            controller.onSaveRecordingsChange = { [weak self] enabled in
                self?.applySaveRecordings(enabled)
            }
            controller.onChooseRecordingsDirectory = { [weak self] in
                self?.chooseRecordingsDirectory()
            }
            controller.onResetRecordingsDirectory = { [weak self] in
                self?.resetRecordingsDirectory()
            }
            controller.onRevealRecordingsDirectory = { [weak self] in
                self?.revealRecordingsDirectory()
            }
            controller.onModelPresetChange = { [weak self] index in
                self?.applyModelPreset(index: index)
            }
            controller.onBatchCommitChange = { [weak self] enabled in
                self?.setBatchCommitEnabled(enabled)
            }
            controller.onApplyModelSpec = { [weak self] spec in
                self?.applyModelSpec(spec)
            }
            controller.onDownloadModel = { [weak self] repoId in
                guard let self else { return }
                Task { await self.downloadModel(repoId: repoId) }
            }
            settingsWindowController = controller
        }

        syncSettingsWindow(controller)
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func applyDictationMode(_ mode: DictationInteractionMode) {
        guard Config.shared.dictationInteractionMode != mode else { return }
        Config.shared.dictationInteractionMode = mode
        yuwpLog("Dictation mode changed to: \(mode.description)")
    }

    private func applyDictationBinding(_ binding: KeyBinding) {
        guard Config.shared.dictationBinding != binding else { return }
        Config.shared.dictationBinding = binding

        if hasPermission, hotkeyManager.restart() {
            yuwpLog("Dictation shortcut changed to: \(binding.description)")
        }
    }

    private func applyServerMode(_ mode: ServerMode) {
        guard mode != Config.shared.serverMode else { return }
        Config.shared.serverMode = mode
        asrProvider.serverMode = mode
        restartProviderForSettingsChange()
        yuwpLog("Server mode changed to: \(mode.description)")
    }

    private func applyServerPort(_ port: UInt16) {
        guard port != Config.shared.serverPort else { return }
        Config.shared.serverPort = port
        asrProvider.port = port
        restartProviderForSettingsChange()
        yuwpLog("Server port changed to: \(port)")
    }

    private func restartProviderForSettingsChange() {
        if session?.isActive == true { stopDictation() }
        asrProvider.shutdown()
        asrProvider.start()
        updateStatus()
        syncSettingsWindow()
    }

    private func syncSettingsWindow(_ controller: SettingsWindowController? = nil) {
        let target = controller ?? settingsWindowController
        target?.sync(
            dictationMode: Config.shared.dictationInteractionMode,
            dictationBinding: Config.shared.dictationBinding,
            serverMode: Config.shared.serverMode,
            serverPort: Config.shared.serverPort,
            transcriptionModel: Config.shared.transcriptionModel,
            batchCommitEnabled: Config.shared.batchCommitEnabled,
            saveRecordings: Config.shared.saveRecordings,
            recordingsDir: Config.shared.recordingsDir,
            usingDefaultRecordingsDir: Config.shared.usesDefaultRecordingsDir
        )
        saveRecordingsMenuItem?.state = Config.shared.saveRecordings ? .on : .off
    }

    // MARK: - Status

    private func updateStatus() {
        if !hasPermission {
            statusMenuItem.title = "⚠ Grant Accessibility Permission"
            statusMenuItem.action = #selector(openAccessibilitySettings)
            statusMenuItem.target = self
            statusMenuItem.isEnabled = true
            return
        }

        if Config.shared.serverMode == .off {
            statusMenuItem.title = "Server mode is off"
            statusMenuItem.action = nil
            statusMenuItem.isEnabled = false
            return
        }

        if let modelDownloadStatus {
            statusMenuItem.title = "⬇︎ \(modelDownloadStatus)"
            statusMenuItem.action = nil
            statusMenuItem.isEnabled = false
            return
        }

        let missingModels = missingConfiguredModelLabels()
        if !missingModels.isEmpty {
            statusMenuItem.title = "⚠ \(missingModels.joined(separator: " + ")) model missing"
            statusMenuItem.action = nil
            statusMenuItem.isEnabled = false
            return
        }

        switch asrProvider.state {
        case .disabled:
            statusMenuItem.title = "Server mode is off"
        case .stopped:
            statusMenuItem.title = "Stopped"
        case .starting:
            statusMenuItem.title = "Loading model..."
        case .ready:
            let endpoint = Config.shared.serverMode == .allInterfaces
                ? "0.0.0.0:\(asrProvider.port)"
                : "127.0.0.1:\(asrProvider.port)"
            statusMenuItem.title = "✓ Ready (\(endpoint))"
        case .error(let msg):
            statusMenuItem.title = "⚠ \(msg)"
        }
        statusMenuItem.action = nil
        statusMenuItem.isEnabled = false
    }

    private func missingConfiguredModelLabels() -> [String] {
        ModelLocator.resolve(Config.shared.transcriptionModel) == nil ? ["Transcription"] : []
    }

    // MARK: - Models

    private func applyModelPreset(index: Int) {
        guard index >= 0, index < ModelPreset.presets.count else { return }
        let preset = ModelPreset.presets[index]
        if ModelPreset.current()?.label == preset.label { return }

        applyModelConfig(
            transcriptionModel: preset.transcriptionModel,
            batchCommitEnabled: preset.batchCommitEnabled
        )
        yuwpLog("Model changed to: \(preset.label) (\(preset.summary))")
    }

    private func setBatchCommitEnabled(_ newValue: Bool) {
        if newValue, ModelLocator.resolve(Config.shared.transcriptionModel) == nil {
            showAlert(
                title: "Model missing",
                message: "Pick or download a valid transcription model before enabling the batch commit pass."
            )
            syncSettingsWindow()
            return
        }
        guard Config.shared.batchCommitEnabled != newValue else { return }
        applyModelConfig(batchCommitEnabled: newValue)
        yuwpLog("Batch commit \(newValue ? "enabled" : "disabled")")
    }

    private func applyModelSpec(_ spec: String) {
        let trimmed = spec.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            syncSettingsWindow()
            return
        }

        if ModelLocator.resolve(trimmed) != nil {
            applyModelConfig(transcriptionModel: trimmed)
            yuwpLog("Transcription model changed to: \(trimmed)")
            return
        }

        if ModelLocator.isRepoId(trimmed) {
            let alert = NSAlert()
            alert.messageText = "Download model from Hugging Face?"
            alert.informativeText = "Yuwp couldn't find `\(trimmed)` locally. Download it now into Application Support so the app can manage it directly? This same model is used for live decoding and batch segment commits."
            alert.addButton(withTitle: "Download")
            alert.addButton(withTitle: "Save Anyway")
            alert.addButton(withTitle: "Cancel")
            switch alert.runModal() {
            case .alertFirstButtonReturn:
                Task { await downloadModel(repoId: trimmed) }
            case .alertSecondButtonReturn:
                applyModelConfig(transcriptionModel: trimmed)
            default:
                syncSettingsWindow()
            }
            return
        }

        showAlert(
            title: "Model folder not found",
            message: "Yuwp couldn't find a valid model directory at `\(trimmed)`. Pick a folder with config.json, model.safetensors, vocab.json, and merges.txt."
        )
        syncSettingsWindow()
    }

    private func applyModelConfig(
        transcriptionModel: String? = nil,
        batchCommitEnabled: Bool? = nil
    ) {
        if session?.isActive == true { stopDictation() }

        if let transcriptionModel {
            Config.shared.transcriptionModel = transcriptionModel
            asrProvider.transcriptionModel = transcriptionModel
        }
        if let batchCommitEnabled {
            Config.shared.batchCommitEnabled = batchCommitEnabled
            asrProvider.batchCommitEnabled = batchCommitEnabled
        }

        asrProvider.shutdown()
        asrProvider.start()
        updateStatus()
        syncSettingsWindow()
    }

    private func showAlert(title: String, message: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }

    private func downloadModel(repoId: String) async {
        modelDownloadStatus = "Preparing \(ModelLocator.shortRepoName(repoId))…"
        updateStatus()

        do {
            let _ = try await ModelDownloadManager.shared.download(repoId: repoId) { [weak self] progress, status in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    let pct = Int(progress * 100)
                    self.modelDownloadStatus = "\(ModelLocator.shortRepoName(repoId)) \(pct)% — \(status)"
                    self.updateStatus()
                }
            }

            modelDownloadStatus = nil
            applyModelConfig(transcriptionModel: repoId)
            yuwpLog("Downloaded model: \(repoId)")
        } catch {
            modelDownloadStatus = nil
            updateStatus()
            syncSettingsWindow()
            showAlert(title: "Model download failed", message: error.localizedDescription)
            yuwpLog("Model download failed: \(repoId) — \(error.localizedDescription)")
        }
    }

    // MARK: - Recording Settings

    @objc private func toggleSaveRecordings(_ sender: NSMenuItem) {
        applySaveRecordings(sender.state != .on)
    }

    private func applySaveRecordings(_ enabled: Bool) {
        guard Config.shared.saveRecordings != enabled else {
            syncSettingsWindow()
            return
        }
        Config.shared.saveRecordings = enabled
        syncSettingsWindow()
        yuwpLog("Save recordings \(enabled ? "enabled" : "disabled")")
    }

    private func chooseRecordingsDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = Config.shared.recordingsDir
        panel.message = "Choose where Yuwp should save recorded audio files."
        if panel.runModal() == .OK, let url = panel.url {
            Config.shared.setRecordingsDir(url)
            syncSettingsWindow()
            yuwpLog("Recordings location changed to: \(Config.shared.recordingsDir.path)")
        }
    }

    private func resetRecordingsDirectory() {
        Config.shared.resetRecordingsDir()
        syncSettingsWindow()
        yuwpLog("Recordings location reset to default: \(Config.shared.defaultRecordingsDir.path)")
    }

    private func revealRecordingsDirectory() {
        NSWorkspace.shared.activateFileViewerSelecting([Config.shared.recordingsDir])
    }

    // MARK: - Enter Replay

    /// Post a Return key event to the focused app after dictation commits.
    private func replayEnterKey() {
        let returnKeyCode: CGKeyCode = 36
        if let down = CGEvent(keyboardEventSource: nil, virtualKey: returnKeyCode, keyDown: true) {
            down.post(tap: .cgSessionEventTap)
        }
        if let up = CGEvent(keyboardEventSource: nil, virtualKey: returnKeyCode, keyDown: false) {
            up.post(tap: .cgSessionEventTap)
        }
        yuwpLog("Enter replayed after commit")
    }

    // MARK: - Recording

    private func saveRecording(_ pcmData: Data) {
        let dir = Config.shared.recordingsDir
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let filename = "yuwp-\(formatter.string(from: Date())).wav"
        let url = dir.appendingPathComponent(filename)

        do {
            try WAVWriter.write(pcmData, to: url)
            yuwpLog("Recording saved: \(url.path) (\(String(format: "%.1f", Double(pcmData.count) / 32000))s)")
        } catch {
            yuwpLog("Failed to save recording: \(error)")
        }
    }
}
