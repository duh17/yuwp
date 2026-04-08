import AppKit
import AVFoundation

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
    private let hotkeyManager = HotkeyManager()
    private let asrProvider: NativeASRProvider = {
        let p = NativeASRProvider()
        p.streamingModel = Config.shared.streamingModel
        p.batchModel = Config.shared.batchModel
        p.batchRetranscribeEnabled = Config.shared.batchRetranscribeEnabled
        return p
    }()
    private let audioCapture = AudioCapture()
    private let micPanel = MicPanel()

    // Per-dictation session (created on start, torn down on stop)
    private var session: DictationSession?
    /// Set when Enter was intercepted mid-session; triggers Enter replay after final commit.
    private var pendingEnter = false

    // Menu bar state
    private var statusItem: NSStatusItem!
    private var statusMenuItem: NSMenuItem!
    private var hotkeyMenuItem: NSMenuItem!
    private var hotkeySubmenu: NSMenu!
    private var modelMenuItem: NSMenuItem!
    private var modelSubmenu: NSMenu!
    private var providerReady = false
    private var hasPermission = false
    private var permissionTimer: Timer?
    private var modelDownloadStatus: String?

    // MARK: - Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupMenuBar()
        setupMicPanelDismiss()
        startSttProvider()
        requestMicPermission()
        checkPermission()
    }

    // MARK: - Hotkey Toggle

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

    private func toggleListening() {
        if session?.isActive == true {
            stopDictation()
        } else {
            startDictation()
        }
    }

    private func startDictation() {
        guard session == nil, providerReady else {
            if !providerReady { yuwpLog("Model still loading, please wait...") }
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
        let pcmData = s.stop()

        // Hide panel and reset icon immediately — don't wait for server's final
        micPanel.hide()
        statusItem.button?.image = NSImage(
            systemSymbolName: "waveform",
            accessibilityDescription: "Yuwp"
        )

        if let pcmData, !pcmData.isEmpty {
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
        hotkeyManager.onToggle = { [weak self] in
            Task { @MainActor in self?.toggleListening() }
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
        yuwpLog("Ready. \(Config.shared.hotkeyMode.description) to dictate.")
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

        hotkeyMenuItem = NSMenuItem(title: "Hotkey: \(Config.shared.hotkeyMode.description)", action: nil, keyEquivalent: "")
        hotkeySubmenu = NSMenu()
        rebuildHotkeySubmenu()
        hotkeyMenuItem.submenu = hotkeySubmenu
        menu.addItem(hotkeyMenuItem)

        modelMenuItem = NSMenuItem(title: modelMenuTitle(), action: nil, keyEquivalent: "")
        modelSubmenu = NSMenu()
        rebuildModelSubmenu()
        modelMenuItem.submenu = modelSubmenu
        menu.addItem(modelMenuItem)

        menu.addItem(.separator())
        menu.addItem(withTitle: "Quit Yuwp", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")

        statusItem.menu = menu
    }

    private func rebuildHotkeySubmenu() {
        hotkeySubmenu.removeAllItems()
        let currentMode = Config.shared.hotkeyMode
        for preset in HotkeyMode.presets {
            let item = NSMenuItem(title: preset.label, action: #selector(changeHotkey(_:)),
                                  keyEquivalent: "")
            item.target = self
            item.representedObject = HotkeyMode.presets.firstIndex(where: { $0.label == preset.label })
            item.state = modesMatch(currentMode, preset.mode) ? .on : .off
            hotkeySubmenu.addItem(item)
        }
    }

    @objc private func changeHotkey(_ sender: NSMenuItem) {
        guard let idx = sender.representedObject as? Int,
              idx < HotkeyMode.presets.count else { return }
        let mode = HotkeyMode.presets[idx].mode
        Config.shared.hotkeyMode = mode
        if hasPermission {
            if hotkeyManager.restart() {
                hotkeyMenuItem.title = "Hotkey: \(mode.description)"
                rebuildHotkeySubmenu()
                yuwpLog("Hotkey changed to: \(mode.description)")
            }
        }
    }

    private func updateStatus() {
        if !hasPermission {
            statusMenuItem.title = "⚠ Grant Accessibility Permission"
            statusMenuItem.action = #selector(openAccessibilitySettings)
            statusMenuItem.target = self
            statusMenuItem.isEnabled = true
            return
        }

        if let modelDownloadStatus {
            statusMenuItem.title = "⬇︎ \(modelDownloadStatus)"
            statusMenuItem.action = nil
            statusMenuItem.isEnabled = false
            modelMenuItem?.title = modelMenuTitle()
            return
        }

        switch asrProvider.state {
        case .stopped:
            statusMenuItem.title = "Stopped"
        case .starting:
            statusMenuItem.title = "Loading model..."
        case .ready:
            statusMenuItem.title = "✓ Ready"
        case .error(let msg):
            statusMenuItem.title = "⚠ \(msg)"
        }
        statusMenuItem.action = nil
        statusMenuItem.isEnabled = false
        modelMenuItem?.title = modelMenuTitle()
    }

    // MARK: - Model Menu

    private enum ModelRole {
        case streaming
        case batch

        var label: String {
            switch self {
            case .streaming: "Streaming"
            case .batch: "Batch"
            }
        }
    }

    private func modelMenuTitle() -> String {
        let label = ModelPreset.current()?.label ?? "Custom"
        return "Model: \(label)"
    }

    private func rebuildModelSubmenu() {
        modelSubmenu.removeAllItems()
        let currentPreset = ModelPreset.current()

        for (idx, preset) in ModelPreset.presets.enumerated() {
            let item = NSMenuItem(
                title: "\(preset.label)  (\(preset.summary))",
                action: #selector(changeModel(_:)),
                keyEquivalent: ""
            )
            item.target = self
            item.representedObject = idx
            item.state = currentPreset?.label == preset.label ? .on : .off
            modelSubmenu.addItem(item)
        }

        modelSubmenu.addItem(.separator())

        let batchToggle = NSMenuItem(
            title: "Enable batch retranscribe",
            action: #selector(toggleBatchRetranscribe(_:)),
            keyEquivalent: ""
        )
        batchToggle.target = self
        batchToggle.state = Config.shared.batchRetranscribeEnabled ? .on : .off
        modelSubmenu.addItem(batchToggle)

        modelSubmenu.addItem(.separator())

        let streamingInfo = NSMenuItem(title: currentModelStatusText(for: .streaming), action: nil, keyEquivalent: "")
        streamingInfo.isEnabled = false
        modelSubmenu.addItem(streamingInfo)
        let setStreaming = modelSubmenu.addItem(withTitle: "Set Streaming Model ID or Path…", action: #selector(setStreamingModelSpec), keyEquivalent: "")
        setStreaming.target = self
        let chooseStreaming = modelSubmenu.addItem(withTitle: "Choose Streaming Model Folder…", action: #selector(chooseStreamingModelFolder), keyEquivalent: "")
        chooseStreaming.target = self
        let streamingDownloads = NSMenuItem(title: "Download Streaming Model", action: nil, keyEquivalent: "")
        streamingDownloads.submenu = makeDownloadSubmenu(role: .streaming)
        modelSubmenu.addItem(streamingDownloads)

        modelSubmenu.addItem(.separator())

        let batchInfo = NSMenuItem(title: currentModelStatusText(for: .batch), action: nil, keyEquivalent: "")
        batchInfo.isEnabled = false
        modelSubmenu.addItem(batchInfo)
        let setBatch = modelSubmenu.addItem(withTitle: "Set Batch Model ID or Path…", action: #selector(setBatchModelSpec), keyEquivalent: "")
        setBatch.target = self
        let chooseBatch = modelSubmenu.addItem(withTitle: "Choose Batch Model Folder…", action: #selector(chooseBatchModelFolder), keyEquivalent: "")
        chooseBatch.target = self
        let batchDownloads = NSMenuItem(title: "Download Batch Model", action: nil, keyEquivalent: "")
        batchDownloads.submenu = makeDownloadSubmenu(role: .batch)
        modelSubmenu.addItem(batchDownloads)
    }

    private func currentModelStatusText(for role: ModelRole) -> String {
        let spec = currentModelSpec(for: role)
        let name = ModelLocator.displayName(for: spec)
        if role == .batch && !Config.shared.batchRetranscribeEnabled {
            return "\(role.label): \(name) (disabled)"
        }
        let installed = ModelLocator.resolve(spec) != nil
        return "\(role.label): \(name) \(installed ? "✓" : "⚠ missing")"
    }

    private func currentModelSpec(for role: ModelRole) -> String {
        switch role {
        case .streaming: Config.shared.streamingModel
        case .batch: Config.shared.batchModel
        }
    }

    private func makeDownloadSubmenu(role: ModelRole) -> NSMenu {
        let menu = NSMenu()
        for model in DownloadableASRModel.supported {
            let item = NSMenuItem(
                title: model.label,
                action: role == .streaming ? #selector(downloadStreamingModel(_:)) : #selector(downloadBatchModel(_:)),
                keyEquivalent: ""
            )
            item.target = self
            item.representedObject = model.repoId
            item.state = currentModelSpec(for: role) == model.repoId ? .on : .off
            menu.addItem(item)
        }
        return menu
    }

    @objc private func changeModel(_ sender: NSMenuItem) {
        guard let idx = sender.representedObject as? Int,
              idx < ModelPreset.presets.count else { return }
        let preset = ModelPreset.presets[idx]
        if ModelPreset.current()?.label == preset.label { return }

        applyModelConfig(
            streamingModel: preset.streamingModel,
            batchModel: preset.batchModel,
            batchEnabled: preset.batchEnabled
        )
        yuwpLog("Model changed to: \(preset.label) (\(preset.summary))")
    }

    @objc private func toggleBatchRetranscribe(_ sender: NSMenuItem) {
        let newValue = !Config.shared.batchRetranscribeEnabled
        if newValue, ModelLocator.resolve(Config.shared.batchModel) == nil {
            showAlert(
                title: "Batch model missing",
                message: "Pick or download a valid batch model before enabling retranscription."
            )
            return
        }
        applyModelConfig(batchEnabled: newValue)
        yuwpLog("Batch retranscribe \(newValue ? "enabled" : "disabled")")
    }

    @objc private func setStreamingModelSpec() {
        promptForModelSpec(role: .streaming)
    }

    @objc private func setBatchModelSpec() {
        promptForModelSpec(role: .batch)
    }

    @objc private func chooseStreamingModelFolder() {
        chooseModelFolder(role: .streaming)
    }

    @objc private func chooseBatchModelFolder() {
        chooseModelFolder(role: .batch)
    }

    @objc private func downloadStreamingModel(_ sender: NSMenuItem) {
        guard let repoId = sender.representedObject as? String else { return }
        Task { await downloadModel(repoId: repoId, applyTo: .streaming) }
    }

    @objc private func downloadBatchModel(_ sender: NSMenuItem) {
        guard let repoId = sender.representedObject as? String else { return }
        Task { await downloadModel(repoId: repoId, applyTo: .batch) }
    }

    private func promptForModelSpec(role: ModelRole) {
        let title = "Set \(role.label) Model"
        let message = "Enter a Hugging Face repo id (for example `mlx-community/Qwen3-ASR-0.6B-4bit`) or a local model folder path."
        guard let spec = promptForText(title: title, message: message, initialValue: currentModelSpec(for: role)) else {
            return
        }
        applyModelSpec(spec, for: role)
    }

    private func chooseModelFolder(role: ModelRole) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = URL(fileURLWithPath: NSHomeDirectory())
        panel.message = "Choose a folder containing config.json, model.safetensors, vocab.json, and merges.txt."
        if panel.runModal() == .OK, let url = panel.url {
            applyModelSpec(url.path, for: role)
        }
    }

    private func applyModelSpec(_ spec: String, for role: ModelRole) {
        let trimmed = spec.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        if ModelLocator.resolve(trimmed) != nil {
            switch role {
            case .streaming:
                applyModelConfig(streamingModel: trimmed)
            case .batch:
                applyModelConfig(batchModel: trimmed)
            }
            yuwpLog("\(role.label) model changed to: \(trimmed)")
            return
        }

        if ModelLocator.isRepoId(trimmed) {
            let alert = NSAlert()
            alert.messageText = "Download model from Hugging Face?"
            alert.informativeText = "Yuwp couldn't find `\(trimmed)` locally. Download it now into Application Support so the app can manage it directly?"
            alert.addButton(withTitle: "Download")
            alert.addButton(withTitle: "Save Anyway")
            alert.addButton(withTitle: "Cancel")
            switch alert.runModal() {
            case .alertFirstButtonReturn:
                Task { await downloadModel(repoId: trimmed, applyTo: role) }
            case .alertSecondButtonReturn:
                switch role {
                case .streaming:
                    applyModelConfig(streamingModel: trimmed)
                case .batch:
                    applyModelConfig(batchModel: trimmed)
                }
            default:
                break
            }
            return
        }

        showAlert(
            title: "Model folder not found",
            message: "Yuwp couldn't find a valid model directory at `\(trimmed)`. Pick a folder with config.json, model.safetensors, vocab.json, and merges.txt."
        )
    }

    private func applyModelConfig(
        streamingModel: String? = nil,
        batchModel: String? = nil,
        batchEnabled: Bool? = nil
    ) {
        if session?.isActive == true { stopDictation() }

        if let streamingModel {
            Config.shared.streamingModel = streamingModel
            asrProvider.streamingModel = streamingModel
        }
        if let batchModel {
            Config.shared.batchModel = batchModel
            asrProvider.batchModel = batchModel
        }
        if let batchEnabled {
            Config.shared.batchRetranscribeEnabled = batchEnabled
            asrProvider.batchRetranscribeEnabled = batchEnabled
        }

        asrProvider.shutdown()
        asrProvider.start()
        rebuildModelSubmenu()
        updateStatus()
    }

    private func promptForText(title: String, message: String, initialValue: String) -> String? {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: "Save")
        alert.addButton(withTitle: "Cancel")

        let field = NSTextField(frame: NSRect(x: 0, y: 0, width: 360, height: 24))
        field.stringValue = initialValue
        alert.accessoryView = field

        return alert.runModal() == .alertFirstButtonReturn
            ? field.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
            : nil
    }

    private func showAlert(title: String, message: String) {
        let alert = NSAlert()
        alert.messageText = title
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }

    private func downloadModel(repoId: String, applyTo role: ModelRole?) async {
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
            switch role {
            case .streaming:
                applyModelConfig(streamingModel: repoId)
            case .batch:
                applyModelConfig(batchModel: repoId)
            case nil:
                rebuildModelSubmenu()
                updateStatus()
            }
            yuwpLog("Downloaded model: \(repoId)")
        } catch {
            modelDownloadStatus = nil
            rebuildModelSubmenu()
            updateStatus()
            showAlert(title: "Model download failed", message: error.localizedDescription)
            yuwpLog("Model download failed: \(repoId) — \(error.localizedDescription)")
        }
    }

    private func modesMatch(_ a: HotkeyMode, _ b: HotkeyMode) -> Bool {
        switch (a, b) {
        case (.combo(let ak, let am), .combo(let bk, let bm)):
            return ak == bk && am == bm
        case (.doubleTap(let ak, _), .doubleTap(let bk, _)):
            return ak == bk
        default: return false
        }
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
