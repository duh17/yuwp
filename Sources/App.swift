import ASRIPC
import AppKit
import AVFoundation
import Carbon.HIToolbox
import Sparkle
import UniformTypeIdentifiers

// Yuwp — system-wide voice dictation for macOS
// Press hotkey → speak → text streams into any focused text field
// Powered by Qwen3-ASR via native yuwp-asr serve

private final class DiagnosticLogState: @unchecked Sendable {
    private let lock = NSLock()
    private var enabled = false

    func setEnabled(_ value: Bool) {
        lock.lock()
        enabled = value
        lock.unlock()
    }

    func isEnabled() -> Bool {
        lock.lock()
        let value = enabled
        lock.unlock()
        return value
    }
}

private let yuwpDiagnosticLogState = DiagnosticLogState()

func setYuwpDiagnosticLoggingEnabled(_ enabled: Bool) {
    yuwpDiagnosticLogState.setEnabled(enabled)
}

private func isYuwpDiagnosticLoggingEnabled() -> Bool {
    yuwpDiagnosticLogState.isEnabled()
}

/// Log to stderr when diagnostic logging is enabled.
func yuwpLog(_ msg: String) {
    guard isYuwpDiagnosticLoggingEnabled() else { return }
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
final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate {
    // Infrastructure — live for the app's lifetime
    private lazy var updaterController: SPUStandardUpdaterController? = makeUpdaterController()
    private let hotkeyManager = HotkeyManager()
    private let audioInputCatalog = SystemAudioInputCatalog()
    private let asrProvider: NativeASRProvider = {
        let env = ProcessInfo.processInfo.environment
        let snapshotRequested = env["YUWP_SETTINGS_SNAPSHOT_PATH"]?.isEmpty == false
        let snapshotPort = env["YUWP_SETTINGS_SNAPSHOT_PORT"].flatMap(UInt16.init)
        let runtimePort = snapshotRequested ? (snapshotPort ?? 29_748) : Config.shared.serverPort

        let p = NativeASRProvider(port: runtimePort)
        p.serverMode = Config.shared.serverMode
        p.asrTransport = Config.shared.asrTransport
        p.transcriptionModel = Config.shared.transcriptionModel
        p.batchCommitEnabled = Config.shared.batchCommitEnabled
        p.diagnosticLoggingEnabled = Config.shared.diagnosticLoggingEnabled
        return p
    }()
    private lazy var audioCapture = AudioCapture(inputCatalog: audioInputCatalog)
    private let micPanel = MicPanel()
    private let chimePlayer = DictationChimePlayer()

    // Per-dictation session (created on start, torn down on stop)
    private var session: DictationSession?
    private var activeDictationLanguageMode: DictationLanguageMode?
    private var activeDictationLanguageHint: String?
    private var activeTranscriptionModel: String?
    private var pendingRecordingArtifact: RecordingArtifactHandle?
    private var pendingRecordingTranscript: String?
    private var appState = AppState()

    // Menu bar state
    private var statusItem: NSStatusItem!
    private var statusMenuItem: NSMenuItem!
    private var audioInputMenuItem: NSMenuItem!
    private var audioInputSubmenu: NSMenu!
    private var saveRecordingsMenuItem: NSMenuItem!
    private var updateMenuItem: NSMenuItem?
    private var sparkleUpdateAvailable = false
    private var settingsWindowController: SettingsWindowController?
    private var permissionTimer: Timer?
    private var hotkeyRecordingActive = false
    private let onboardingModelPromptKey = "didShowOnboardingModelPrompt"

    // MARK: - Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        setYuwpDiagnosticLoggingEnabled(Config.shared.diagnosticLoggingEnabled)
        normalizeStartupConfiguration()
        syncAppStateFromConfig()
        audioCapture.inputSelection = Config.shared.audioInputSelection
        micPanel.animationConfig = Config.shared.micPanelAnimation
        setupMainMenuShortcuts()
        setupMenuBar()
        syncRuntimeUI()
        updateStatus()
        setupMicPanelDismiss()
        startSttProvider()
        syncMicrophonePermissionStatus()
        checkPermission()

        let env = ProcessInfo.processInfo.environment
        let shouldOpenSettings = env["YUWP_OPEN_SETTINGS_ON_LAUNCH"] == "1" || env["YUWP_SETTINGS_SNAPSHOT_PATH"] != nil
        if shouldOpenSettings {
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { [weak self] in
                self?.openSettings()
                self?.triggerAutoDownloadIfRequested()
                self?.captureSettingsSnapshotIfRequested()
            }
        } else {
            maybeOpenSettingsForFirstRunModelOnboarding()
            triggerAutoDownloadIfRequested()
        }
    }

    func applicationWillTerminate(_ notification: Notification) {
        asrProvider.shutdown()
    }

    private func captureSettingsSnapshotIfRequested() {
        let env = ProcessInfo.processInfo.environment
        guard let path = env["YUWP_SETTINGS_SNAPSHOT_PATH"], !path.isEmpty else { return }
        let delay = env["YUWP_SETTINGS_SNAPSHOT_DELAY"].flatMap(Double.init) ?? 0.5
        DispatchQueue.main.asyncAfter(deadline: .now() + delay) { [weak self] in
            self?.captureSettingsSnapshot(to: path)
        }
    }

    private func maybeOpenSettingsForFirstRunModelOnboarding() {
        guard !UserDefaults.standard.bool(forKey: onboardingModelPromptKey) else { return }
        guard !appState.missingConfiguredModelLabels.isEmpty else { return }

        UserDefaults.standard.set(true, forKey: onboardingModelPromptKey)
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.45) { [weak self] in
            self?.openSettings()
        }
    }

    private func triggerAutoDownloadIfRequested() {
        let env = ProcessInfo.processInfo.environment
        guard let repoId = env["YUWP_AUTO_DOWNLOAD_MODEL_REPO_ID"]?.trimmingCharacters(in: .whitespacesAndNewlines), !repoId.isEmpty else {
            return
        }
        Task { await downloadModel(repoId: repoId) }
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

    private func makeUpdaterController() -> SPUStandardUpdaterController? {
        let info = Bundle.main.infoDictionary ?? [:]
        let feedURL = (info["SUFeedURL"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let publicKey = (info["SUPublicEDKey"] as? String)?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !feedURL.isEmpty, !publicKey.isEmpty else {
            yuwpLog("Sparkle disabled — missing SUFeedURL or SUPublicEDKey")
            return nil
        }
        return SPUStandardUpdaterController(
            startingUpdater: true, updaterDelegate: nil, userDriverDelegate: self
        )
    }

    private func setSparkleUpdateAvailable(_ available: Bool) {
        sparkleUpdateAvailable = available
        updateMenuItem?.title = available ? "Update Available..." : "Check for Updates..."
    }

    private func syncAppStateFromConfig() {
        appState.settings = AppSettingsState(
            serverMode: Config.shared.serverMode,
            asrTransport: Config.shared.asrTransport
        )
        appState.missingConfiguredModelLabels = Self.requiredDictationModelLabels(
            transcriptionModel: Config.shared.transcriptionModel
        )
    }

    private func send(_ action: AppAction) {
        let effects = appState.send(action, dictationBindingDescription: Config.shared.dictationBinding.description)
        syncRuntimeUI()
        updateStatus()
        syncSettingsWindow()
        run(effects)
    }

    private func syncRuntimeUI() {
        HotkeyManager.sessionActive = appState.sessionPhase.isCapturingAudio
        statusItem?.button?.image = statusBarIconImage(
            listening: appState.sessionPhase == .listening,
            accessibilityDescription: appState.sessionPhase == .listening ? "Yuwp — Listening" : "Yuwp"
        )
    }

    private func statusBarIconImage(listening: Bool, accessibilityDescription: String) -> NSImage? {
        guard let symbol = NSImage(
            systemSymbolName: listening ? "waveform.circle.fill" : "waveform",
            accessibilityDescription: accessibilityDescription
        ) else {
            return nil
        }
        let config = NSImage.SymbolConfiguration(pointSize: 14, weight: .regular)
        let configured = symbol.withSymbolConfiguration(config) ?? symbol
        configured.isTemplate = true
        return configured
    }

    private func run(_ effects: [AppEffect]) {
        for effect in effects {
            switch effect {
            case .startDictation:
                performStartDictation()
            case .stopDictation:
                performStopDictation()
            case .replayEnter:
                replayEnterKey()
            case .requestMicrophonePermission:
                requestMicPermission()
            case .presentMicPanel(let state):
                micPanel.present(state)
            case .updateMicLevel(let level):
                micPanel.updateAudioLevel(level)
            case .hideMicPanel:
                micPanel.hide()
            case .log(let message):
                yuwpLog(message)
            }
        }
    }

    // MARK: - Hotkey Toggle

    private func normalizeStartupConfiguration() {
        let shared = Config.shared
        shared.transcriptionModel = shared.transcriptionModel
        shared.batchCommitEnabled = shared.batchCommitEnabled
        if shared.serverMode == .allInterfaces, shared.asrTransport == .stdio {
            shared.asrTransport = .http
        } else {
            shared.asrTransport = shared.asrTransport
        }
        asrProvider.transcriptionModel = shared.transcriptionModel
        asrProvider.batchCommitEnabled = shared.batchCommitEnabled
        asrProvider.asrTransport = shared.asrTransport
        asrProvider.serverMode = shared.serverMode
        asrProvider.port = shared.serverPort
        asrProvider.diagnosticLoggingEnabled = shared.diagnosticLoggingEnabled
        asrProvider.saveRecordings = shared.saveRecordings
        asrProvider.recordingsDir = shared.recordingsDir
    }

    private func setupMicPanelDismiss() {
        micPanel.onDismiss = { [weak self] in
            Task { @MainActor in
                self?.send(.micPanelDismissed)
            }
        }

        hotkeyManager.onEnterDuringSession = { [weak self] in
            Task { @MainActor in
                self?.send(.enterIntercepted)
            }
        }
    }

    private func handleShortcutEvent(_ event: ShortcutEvent) {
        send(.shortcutReceived(event))
    }

    private func performStartDictation() {
        guard session == nil else { return }

        audioCapture.inputSelection = Config.shared.audioInputSelection
        let injectorPolicy = TextInjectorFactory.InjectionPolicy(
            allowTextFieldDirectInjection: Config.shared.experimentalDirectTextFieldInsertionEnabled,
            allowTerminalDirectInjection: Config.shared.experimentalDirectTerminalInsertionEnabled
        )
        let injector = TextInjectorFactory.capture(policy: injectorPolicy)
        let languageHint = resolveDictationLanguageHint()
        activeDictationLanguageMode = Config.shared.dictationLanguageMode
        activeDictationLanguageHint = languageHint
        activeTranscriptionModel = Config.shared.transcriptionModel
        pendingRecordingArtifact = nil
        pendingRecordingTranscript = nil
        let s = DictationSession(
            sttSession: asrProvider.makeSession(),
            textInjector: injector,
            audioCapture: audioCapture,
            languageHint: languageHint
        )
        s.onEvent = { [weak self] event in self?.handleSessionEvent(event) }
        s.onFinalTranscript = { [weak self] transcript in
            self?.handleFinalTranscript(transcript)
        }
        s.onRequestStop = { [weak self] in
            Task { @MainActor in self?.send(.sessionStopRequested) }
        }
        session = s
        let startChime = Config.shared.startChime
        chimePlayer.play(.start)

        // Give the start cue a brief head start before audio capture spins up.
        // Capture startup can mask/cut the cue tail if we start immediately.
        if startChime.selection == .none {
            s.start()
        } else {
            let leadInMs: UInt64 = switch startChime.selection {
            case .mechanical: 260
            case .soft: 160
            case .systemDefault: 180
            case .custom: 200
            case .none: 0
            }
            Task { @MainActor [weak self, weak s] in
                try? await Task.sleep(for: .milliseconds(leadInMs))
                guard let self, let s, self.session === s else { return }
                s.start()
            }
        }
    }

    private func performStopDictation() {
        guard let s = session else { return }
        let sessionID = s.debugSessionID
        let pcmData = s.stop()
        chimePlayer.play(.stop)

        if Config.shared.saveRecordings, let pcmData, !pcmData.isEmpty {
            pendingRecordingArtifact = saveRecording(pcmData, sessionID: sessionID)
            flushPendingRecordingTranscriptIfPossible()
        }
    }

    // MARK: - Session Events → UI

    private func handleSessionEvent(_ event: DictationEvent) {
        if case .finished = event {
            session = nil
            activeDictationLanguageMode = nil
            activeDictationLanguageHint = nil
            activeTranscriptionModel = nil
        }
        send(.sessionEvent(event))
    }

    private func handleFinalTranscript(_ transcript: String) {
        pendingRecordingTranscript = transcript
        flushPendingRecordingTranscriptIfPossible()
    }

    private func flushPendingRecordingTranscriptIfPossible() {
        guard let artifact = pendingRecordingArtifact,
              let transcript = pendingRecordingTranscript else { return }

        do {
            try RecordingArtifactWriter.writeTranscript(transcript, for: artifact)
            pendingRecordingTranscript = nil
            yuwpLog("Transcript saved: path=\(artifact.transcriptURL.path)")
        } catch {
            yuwpLog("Failed to save transcript artifact: \(error)")
        }
    }

    // MARK: - STT Provider

    private func startSttProvider() {
        asrProvider.onStateChange = { [weak self] state in
            Task { @MainActor in
                self?.send(.providerStateChanged(state))
            }
        }
        asrProvider.onError = { error in
            yuwpLog("STT error: \(error)")
        }
        asrProvider.start()
    }

    // MARK: - Microphone Permission

    private func syncMicrophonePermissionStatus() {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        send(.microphonePermissionChanged(microphonePermissionState(for: status)))
    }

    private func requestMicPermission() {
        let status = AVCaptureDevice.authorizationStatus(for: .audio)
        switch status {
        case .authorized:
            send(.microphonePermissionChanged(.granted))
            yuwpLog("Microphone permission already granted")

        case .denied, .restricted:
            send(.microphonePermissionChanged(.denied))
            yuwpLog("Microphone permission denied")

        case .notDetermined:
            send(.microphonePermissionChanged(.notDetermined))
            AVCaptureDevice.requestAccess(for: .audio) { granted in
                Task { @MainActor in
                    self.send(.microphonePermissionChanged(granted ? .granted : .denied))
                    if granted {
                        yuwpLog("Microphone permission granted — press your shortcut again to start dictation")
                    } else {
                        yuwpLog("Microphone permission denied")
                    }
                }
            }

        @unknown default:
            send(.microphonePermissionChanged(.denied))
            yuwpLog("Microphone permission status unknown — treating as denied")
        }
    }

    private func microphonePermissionState(for status: AVAuthorizationStatus) -> MicrophonePermissionState {
        switch status {
        case .authorized:
            return .granted
        case .denied, .restricted:
            return .denied
        case .notDetermined:
            return .notDetermined
        @unknown default:
            return .denied
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
            send(.accessibilityPermissionChanged(false))
            updateStatus()
            yuwpLog("Accessibility permission required — choose 'Grant Accessibility Permission' from the menu")

            permissionTimer?.invalidate()
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
        send(.accessibilityPermissionChanged(true))
    }

    @objc private func openAccessibilitySettings() {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility")!
        NSWorkspace.shared.open(url)
    }

    @objc private func openMicrophoneSettings() {
        let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Microphone")!
        NSWorkspace.shared.open(url)
    }

    // MARK: - Menu Bar

    private func setupMainMenuShortcuts() {
        let mainMenu = NSMenu(title: "MainMenu")

        let appMenuItem = NSMenuItem(title: "Yuwp", action: nil, keyEquivalent: "")
        let appMenu = NSMenu(title: "Yuwp")

        let settingsShortcutItem = NSMenuItem(title: "Settings…", action: #selector(openSettings), keyEquivalent: ",")
        settingsShortcutItem.keyEquivalentModifierMask = [.command]
        settingsShortcutItem.target = self
        appMenu.addItem(settingsShortcutItem)
        appMenu.addItem(.separator())

        let quitShortcutItem = NSMenuItem(title: "Quit Yuwp", action: #selector(NSApplication.terminate(_:)), keyEquivalent: "q")
        quitShortcutItem.keyEquivalentModifierMask = [.command]
        quitShortcutItem.target = NSApp
        appMenu.addItem(quitShortcutItem)

        appMenuItem.submenu = appMenu
        mainMenu.addItem(appMenuItem)

        let windowMenuItem = NSMenuItem(title: "Window", action: nil, keyEquivalent: "")
        let windowMenu = NSMenu(title: "Window")

        let closeWindowItem = NSMenuItem(title: "Close Window", action: #selector(NSWindow.performClose(_:)), keyEquivalent: "w")
        closeWindowItem.keyEquivalentModifierMask = [.command]
        windowMenu.addItem(closeWindowItem)

        windowMenuItem.submenu = windowMenu
        mainMenu.addItem(windowMenuItem)

        NSApp.windowsMenu = windowMenu
        NSApp.mainMenu = mainMenu
    }

    private func setupMenuBar() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        guard let button = statusItem.button else { return }
        button.image = statusBarIconImage(listening: false, accessibilityDescription: "Yuwp")

        let menu = NSMenu()
        menu.delegate = self

        statusMenuItem = NSMenuItem(title: "Loading model...", action: nil, keyEquivalent: "")
        statusMenuItem.isEnabled = false
        menu.addItem(statusMenuItem)
        menu.addItem(.separator())

        let settingsItem = makeMenuItem(
            title: "Settings…",
            symbolName: "gearshape",
            action: #selector(openSettings),
            keyEquivalent: ",",
            target: self
        )
        menu.addItem(settingsItem)

        audioInputMenuItem = makeMenuItem(title: "Input Device", symbolName: "mic", action: nil, keyEquivalent: "")
        audioInputSubmenu = NSMenu(title: "Input Device")
        audioInputMenuItem.submenu = audioInputSubmenu
        menu.addItem(audioInputMenuItem)
        rebuildAudioInputMenu()

        menu.addItem(.separator())

        saveRecordingsMenuItem = makeMenuItem(
            title: "Save Recordings",
            symbolName: "record.circle",
            action: #selector(toggleSaveRecordings(_:)),
            keyEquivalent: "",
            target: self
        )
        updateSaveRecordingsMenuItem()
        menu.addItem(saveRecordingsMenuItem)

        if let updaterController {
            menu.addItem(.separator())
            let updateItem = makeMenuItem(
                title: sparkleUpdateAvailable ? "Update Available..." : "Check for Updates...",
                symbolName: "arrow.trianglehead.clockwise",
                action: #selector(SPUStandardUpdaterController.checkForUpdates(_:)),
                keyEquivalent: "",
                target: updaterController
            )
            updateMenuItem = updateItem
            menu.addItem(updateItem)
        }

        menu.addItem(.separator())
        let quitItem = makeMenuItem(
            title: "Quit Yuwp",
            symbolName: "xmark.square",
            action: #selector(NSApplication.terminate(_:)),
            keyEquivalent: "q",
            target: NSApp
        )
        menu.addItem(quitItem)

        statusItem.menu = menu
    }

    private func makeMenuItem(
        title: String,
        symbolName: String,
        action: Selector?,
        keyEquivalent: String,
        target: AnyObject? = nil
    ) -> NSMenuItem {
        let item = NSMenuItem(title: title, action: action, keyEquivalent: keyEquivalent)
        item.target = target
        item.image = menuSymbol(named: symbolName, accessibilityDescription: title)
        return item
    }

    private func menuSymbol(named symbolName: String, accessibilityDescription: String) -> NSImage? {
        guard let symbol = NSImage(systemSymbolName: symbolName, accessibilityDescription: accessibilityDescription) else {
            return nil
        }
        let configuration = NSImage.SymbolConfiguration(pointSize: 14, weight: .regular)
        let configured = symbol.withSymbolConfiguration(configuration) ?? symbol
        configured.isTemplate = true
        return configured
    }

    private func updateSaveRecordingsMenuItem() {
        guard let saveRecordingsMenuItem else { return }
        let enabled = Config.shared.saveRecordings
        saveRecordingsMenuItem.state = .off
        saveRecordingsMenuItem.image = menuSymbol(
            named: enabled ? "record.circle.fill" : "record.circle",
            accessibilityDescription: saveRecordingsMenuItem.title
        )
    }

    @objc private func openSettings() {
        let controller = settingsWindowController ?? SettingsWindowController(store: SettingsStore(snapshot: makeSettingsSnapshot()))
        if settingsWindowController == nil {
            let store = controller.store
            store.onDictationBindingChange = { [weak self] binding in
                self?.applyDictationBinding(binding)
            }
            store.onDictationBindingRecordingChange = { [weak self] isRecording in
                self?.setHotkeyRecordingActive(isRecording)
            }
            store.onAudioInputSelectionChange = { [weak self] selection in
                self?.applyAudioInputSelection(selection)
            }
            store.onDictationLanguageModeChange = { [weak self] mode in
                self?.applyDictationLanguageMode(mode)
            }
            store.onFixedDictationLanguageChange = { [weak self] language in
                self?.applyFixedDictationLanguage(language)
            }
            store.onExperimentalDirectTextFieldInsertionChange = { [weak self] enabled in
                self?.applyExperimentalDirectTextFieldInsertion(enabled)
            }
            store.onExperimentalDirectTerminalInsertionChange = { [weak self] enabled in
                self?.applyExperimentalDirectTerminalInsertion(enabled)
            }
            store.onServerModeChange = { [weak self] mode in
                self?.applyServerMode(mode)
            }
            store.onASRTransportChange = { [weak self] transport in
                self?.applyASRTransport(transport)
            }
            store.onServerPortChange = { [weak self] port in
                self?.applyServerPort(port)
            }
            store.onSaveRecordingsChange = { [weak self] enabled in
                self?.applySaveRecordings(enabled)
            }
            store.onDiagnosticLoggingChange = { [weak self] enabled in
                self?.applyDiagnosticLogging(enabled)
            }
            store.onChooseRecordingsDirectory = { [weak self] in
                self?.chooseRecordingsDirectory()
            }
            store.onResetRecordingsDirectory = { [weak self] in
                self?.resetRecordingsDirectory()
            }
            store.onRevealRecordingsDirectory = { [weak self] in
                self?.revealRecordingsDirectory()
            }
            store.onChooseModelDirectory = { [weak self] in
                self?.chooseTranscriptionModelDirectory()
            }
            store.onBatchCommitChange = { [weak self] enabled in
                self?.setBatchCommitEnabled(enabled)
            }
            store.onApplyModelSpec = { [weak self] spec in
                self?.applyModelSpec(spec)
            }
            store.onDownloadModel = { [weak self] repoId in
                guard let self else { return }
                let activateForTranscription = repoId != NativeASRProvider.defaultAlignerModel
                Task { await self.downloadModel(repoId: repoId, activateForTranscription: activateForTranscription) }
            }
            store.onMicPanelAnimationChange = { [weak self] config in
                self?.applyMicPanelAnimation(config)
            }
            store.onStartChimeChange = { [weak self] config in
                self?.applyChimeConfig(config, role: .start)
            }
            store.onStopChimeChange = { [weak self] config in
                self?.applyChimeConfig(config, role: .stop)
            }
            store.onChooseCustomChime = { [weak self] role in
                self?.chooseCustomChime(for: role)
            }
            store.onPreviewChime = { [weak self] role, config in
                self?.previewChime(role: role, config: config)
            }
            settingsWindowController = controller
        }

        syncSettingsWindow(controller)
        controller.showWindow(nil)
        controller.window?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    private func applyDictationBinding(_ binding: KeyBinding) {
        guard Config.shared.dictationBinding != binding else { return }
        Config.shared.dictationBinding = binding
        defer { syncSettingsWindow() }

        if hotkeyRecordingActive {
            yuwpLog("Dictation shortcut changed to: \(binding.description)")
            return
        }

        if appState.hasAccessibilityPermission, hotkeyManager.restart() {
            yuwpLog("Dictation shortcut changed to: \(binding.description)")
        }
    }

    private func setHotkeyRecordingActive(_ isRecording: Bool) {
        guard hotkeyRecordingActive != isRecording else { return }
        hotkeyRecordingActive = isRecording

        guard appState.hasAccessibilityPermission else { return }

        if isRecording {
            hotkeyManager.stop()
            return
        }

        if !hotkeyManager.start() {
            yuwpLog("Failed to restore dictation shortcut after recording")
        }
    }

    private func applyAudioInputSelection(_ selection: AudioInputSelection) {
        guard Config.shared.audioInputSelection != selection else {
            syncSettingsWindow()
            return
        }

        let devices = audioInputCatalog.availableInputDevices()
        Config.shared.audioInputSelection = selection
        audioCapture.inputSelection = selection

        if session?.isActive == true {
            yuwpLog("Input device changed during dictation — stopping current session")
            send(.sessionStopRequested)
        }

        syncSettingsWindow()
        yuwpLog("Input device changed to: \(selection.summary(using: devices))")
    }

    private func applyDictationLanguageMode(_ mode: DictationLanguageMode) {
        guard Config.shared.dictationLanguageMode != mode else { return }
        Config.shared.dictationLanguageMode = mode
        syncSettingsWindow()
        yuwpLog("Dictation language mode changed to: \(mode.title)")
    }

    private func applyFixedDictationLanguage(_ language: String) {
        guard Config.shared.fixedDictationLanguage != language else { return }
        Config.shared.fixedDictationLanguage = language
        syncSettingsWindow()
        yuwpLog("Fixed dictation language changed to: \(Config.shared.fixedDictationLanguage)")
    }

    private func resolveDictationLanguageHint() -> String? {
        let supported = Config.shared.supportedDictationLanguages

        let hint: String? = switch Config.shared.dictationLanguageMode {
        case .mixed:
            nil
        case .fixed:
            Config.shared.fixedDictationLanguage
        case .followInputSource:
            currentInputSourceLanguage(supportedLanguages: supported)
        }

        if let hint {
            yuwpLog("Dictation language hint: \(hint)")
        } else {
            yuwpLog("Dictation language hint: auto")
        }

        return hint
    }

    private func currentInputSourceLanguage(supportedLanguages: [String]) -> String? {
        guard let source = TISCopyCurrentKeyboardInputSource()?.takeRetainedValue() else {
            return nil
        }

        let sourceID = Self.tisStringProperty(source, key: kTISPropertyInputSourceID)
        let modeID = Self.tisStringProperty(source, key: kTISPropertyInputModeID)
        let tags = Self.tisStringArrayProperty(source, key: kTISPropertyInputSourceLanguages) ?? []

        return DictationLanguageCatalog.languageForInputSource(
            languageTags: tags,
            sourceID: sourceID,
            inputModeID: modeID,
            supportedLanguages: supportedLanguages
        )
    }

    private static func tisStringProperty(_ source: TISInputSource, key: CFString) -> String? {
        guard let raw = TISGetInputSourceProperty(source, key) else { return nil }
        return Unmanaged<CFTypeRef>.fromOpaque(raw).takeUnretainedValue() as? String
    }

    private static func tisStringArrayProperty(_ source: TISInputSource, key: CFString) -> [String]? {
        guard let raw = TISGetInputSourceProperty(source, key) else { return nil }
        return Unmanaged<CFTypeRef>.fromOpaque(raw).takeUnretainedValue() as? [String]
    }

    private func applyServerMode(_ mode: ServerMode) {
        guard mode != Config.shared.serverMode else { return }

        if mode == .allInterfaces,
           Config.shared.serverMode != .allInterfaces,
           !confirmLocalNetworkServerMode() {
            syncSettingsWindow()
            yuwpLog("Server mode change cancelled: local network mode not confirmed")
            return
        }

        let previousTransport = Config.shared.asrTransport
        let effectiveTransport = ServerRuntimePolicy.effectiveTransport(
            serverMode: mode,
            requestedTransport: previousTransport
        )

        Config.shared.serverMode = mode
        asrProvider.serverMode = mode
        Config.shared.asrTransport = effectiveTransport
        asrProvider.asrTransport = effectiveTransport

        if effectiveTransport != previousTransport {
            yuwpLog("ASR transport switched to HTTP because local network mode requires HTTP")
        }

        syncAppStateFromConfig()
        restartProviderForSettingsChange()
        yuwpLog("Server mode changed to: \(mode.description)")
    }

    private func applyASRTransport(_ transport: ASRIPCTransport) {
        guard transport != Config.shared.asrTransport else { return }

        if !ServerRuntimePolicy.canUseTransport(transport, in: Config.shared.serverMode) {
            showAlert(
                title: "Standard I/O unavailable in Local network mode",
                message: "Local network mode requires HTTP transport so other devices can connect. Switch Availability to This Mac only to use Standard I/O."
            )
            syncSettingsWindow()
            return
        }

        Config.shared.asrTransport = transport
        asrProvider.asrTransport = transport
        syncAppStateFromConfig()
        restartProviderForSettingsChange()
        yuwpLog("ASR transport changed to: \(transport.settingsTitle)")
    }

    private func confirmLocalNetworkServerMode() -> Bool {
        let alert = NSAlert()
        alert.alertStyle = .warning
        alert.messageText = "Expose transcription server to your local network?"
        alert.informativeText = "Local network mode binds Yuwp to 0.0.0.0:\(Config.shared.serverPort). Any device on your local network can send audio and receive transcripts from this Mac. The API is currently unauthenticated and unencrypted. Only enable this on trusted networks."
        alert.addButton(withTitle: "Enable Local Network")
        alert.addButton(withTitle: "Cancel")
        return alert.runModal() == .alertFirstButtonReturn
    }

    private func applyExperimentalDirectTextFieldInsertion(_ enabled: Bool) {
        guard enabled != Config.shared.experimentalDirectTextFieldInsertionEnabled else { return }
        Config.shared.experimentalDirectTextFieldInsertionEnabled = enabled
        syncSettingsWindow()
        yuwpLog("Experimental direct text-field insertion \(enabled ? "enabled" : "disabled") (applies next dictation session)")
    }

    private func applyExperimentalDirectTerminalInsertion(_ enabled: Bool) {
        guard enabled != Config.shared.experimentalDirectTerminalInsertionEnabled else { return }
        Config.shared.experimentalDirectTerminalInsertionEnabled = enabled
        syncSettingsWindow()
        yuwpLog("Experimental direct terminal insertion \(enabled ? "enabled" : "disabled") (applies next dictation session)")
    }

    private func applyServerPort(_ port: UInt16) {
        guard port != Config.shared.serverPort else { return }
        Config.shared.serverPort = port
        asrProvider.port = port

        if !ServerRuntimePolicy.shouldRestartProviderForPortChange(
            serverMode: Config.shared.serverMode,
            transport: Config.shared.asrTransport
        ) {
            syncSettingsWindow()
            yuwpLog("Server port set to \(port) (applies when HTTP transport is active)")
            return
        }

        restartProviderForSettingsChange()
        yuwpLog("Server port changed to: \(port)")
    }

    private func restartProviderForSettingsChange() {
        if session?.isActive == true { send(.sessionStopRequested) }
        asrProvider.shutdown()
        asrProvider.start()
        updateStatus()
        syncSettingsWindow()
    }

    private func applyMicPanelAnimation(_ config: MicPanelAnimationConfig) {
        guard Config.shared.micPanelAnimation != config else {
            syncSettingsWindow()
            return
        }
        Config.shared.micPanelAnimation = config
        micPanel.animationConfig = config
        syncSettingsWindow()
        yuwpLog("Mic panel animation changed to: \(config.selection.title)")
    }

    private func applyChimeConfig(_ config: DictationChimeConfig, role: DictationChimeRole) {
        let current = switch role {
        case .start: Config.shared.startChime
        case .stop: Config.shared.stopChime
        }
        guard current != config else {
            syncSettingsWindow()
            return
        }

        switch role {
        case .start:
            Config.shared.startChime = config
        case .stop:
            Config.shared.stopChime = config
        }

        syncSettingsWindow()
        yuwpLog("\(role.title) changed to: \(config.selection.title)")
    }

    private func chooseCustomChime(for role: DictationChimeRole) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.audio]
        panel.message = "Choose an audio file for \(role.title.lowercased())."

        guard panel.runModal() == .OK, let url = panel.url else {
            syncSettingsWindow()
            return
        }

        do {
            let asset = try DictationChimeAssetManager.importSound(from: url)
            var config = switch role {
            case .start: Config.shared.startChime
            case .stop: Config.shared.stopChime
            }
            config.selection = .custom
            config.customAsset = asset
            applyChimeConfig(config, role: role)
        } catch {
            showAlert(title: "Sound import failed", message: error.localizedDescription)
            syncSettingsWindow()
        }
    }

    private func previewChime(role: DictationChimeRole, config: DictationChimeConfig) {
        chimePlayer.play(role, config: config)
    }

    private func syncSettingsWindow(_ controller: SettingsWindowController? = nil) {
        let availableAudioInputs = audioInputCatalog.availableInputDevices()
        let target = controller ?? settingsWindowController
        target?.sync(makeSettingsSnapshot(availableAudioInputs: availableAudioInputs))
        updateSaveRecordingsMenuItem()
        rebuildAudioInputMenu(with: availableAudioInputs)
    }

    private func makeSettingsSnapshot(availableAudioInputs: [AudioInputDeviceDescriptor]? = nil) -> SettingsSnapshot {
        let inputs = availableAudioInputs ?? audioInputCatalog.availableInputDevices()
        let alignerRepoId = NativeASRProvider.defaultAlignerModel
        let activeDownload = appState.activeModelDownload
        let transcriptionDownloadStatus = activeDownload?.repoId == alignerRepoId ? nil : activeDownload?.status
        let alignerDownloadStatus = activeDownload?.repoId == alignerRepoId ? activeDownload?.status : nil

        return SettingsSnapshot(
            dictationBinding: Config.shared.dictationBinding,
            audioInputSelection: Config.shared.audioInputSelection,
            availableAudioInputs: inputs,
            dictationLanguageMode: Config.shared.dictationLanguageMode,
            fixedDictationLanguage: Config.shared.fixedDictationLanguage,
            supportedDictationLanguages: Config.shared.supportedDictationLanguages,
            experimentalDirectTextFieldInsertionEnabled: Config.shared.experimentalDirectTextFieldInsertionEnabled,
            experimentalDirectTerminalInsertionEnabled: Config.shared.experimentalDirectTerminalInsertionEnabled,
            serverMode: Config.shared.serverMode,
            serverPort: Config.shared.serverPort,
            asrTransport: Config.shared.asrTransport,
            transcriptionModel: Config.shared.transcriptionModel,
            batchCommitEnabled: Config.shared.batchCommitEnabled,
            transcriptionDownloadStatus: transcriptionDownloadStatus,
            alignerDownloadStatus: alignerDownloadStatus,
            isModelDownloadInProgress: activeDownload != nil,
            alignerModelRepoId: alignerRepoId,
            alignerInstalled: ModelLocator.resolve(alignerRepoId) != nil,
            saveRecordings: Config.shared.saveRecordings,
            diagnosticLoggingEnabled: Config.shared.diagnosticLoggingEnabled,
            recordingsDir: Config.shared.recordingsDir,
            usingDefaultRecordingsDir: Config.shared.usesDefaultRecordingsDir,
            micPanelAnimation: Config.shared.micPanelAnimation,
            startChime: Config.shared.startChime,
            stopChime: Config.shared.stopChime
        )
    }

    private func rebuildAudioInputMenu(with devices: [AudioInputDeviceDescriptor]? = nil) {
        guard let audioInputSubmenu else { return }
        let availableInputs = devices ?? audioInputCatalog.availableInputDevices()
        let selection = Config.shared.audioInputSelection
        audioInputSubmenu.removeAllItems()

        let systemDefaultTitle: String
        if let defaultDevice = availableInputs.first(where: \.isDefault) {
            systemDefaultTitle = "System Default — \(defaultDevice.name)"
        } else {
            systemDefaultTitle = "System Default"
        }
        let systemDefaultItem = NSMenuItem(
            title: systemDefaultTitle,
            action: #selector(selectAudioInputFromMenu(_:)),
            keyEquivalent: ""
        )
        systemDefaultItem.target = self
        systemDefaultItem.state = selection == .systemDefault ? .on : .off
        systemDefaultItem.representedObject = AudioInputSelection.systemDefault.persistenceString as NSString
        audioInputSubmenu.addItem(systemDefaultItem)

        if !availableInputs.isEmpty {
            audioInputSubmenu.addItem(.separator())
        }

        for device in availableInputs {
            let item = NSMenuItem(title: device.menuTitle, action: #selector(selectAudioInputFromMenu(_:)), keyEquivalent: "")
            item.target = self
            item.state = selection == device.selection ? .on : .off
            item.representedObject = device.selection.persistenceString as NSString
            audioInputSubmenu.addItem(item)
        }

        if case .device(let uid) = selection, !availableInputs.contains(where: { $0.uid == uid }) {
            audioInputSubmenu.addItem(.separator())
            let unavailable = NSMenuItem(title: "Unavailable device — using system default for now", action: nil, keyEquivalent: "")
            unavailable.isEnabled = false
            audioInputSubmenu.addItem(unavailable)
        }
    }

    @objc private func selectAudioInputFromMenu(_ sender: NSMenuItem) {
        let raw = sender.representedObject as? String
        applyAudioInputSelection(AudioInputSelection(persistenceString: raw))
    }

    func menuWillOpen(_ menu: NSMenu) {
        guard menu == statusItem?.menu else { return }
        rebuildAudioInputMenu()
    }

    // MARK: - Status

    private func updateStatus() {
        let status = appState.statusDisplay(port: asrProvider.port)
        statusMenuItem.title = status.title
        statusMenuItem.image = menuSymbol(named: status.symbolName, accessibilityDescription: status.title)
        statusMenuItem.isEnabled = status.isEnabled
        switch status.behavior {
        case .none:
            statusMenuItem.action = nil
            statusMenuItem.target = nil
        case .openAccessibilitySettings:
            statusMenuItem.action = #selector(openAccessibilitySettings)
            statusMenuItem.target = self
        case .openMicrophoneSettings:
            statusMenuItem.action = #selector(openMicrophoneSettings)
            statusMenuItem.target = self
        case .openSettings:
            statusMenuItem.action = #selector(openSettings)
            statusMenuItem.target = self
        }
    }

    nonisolated static func requiredDictationModelLabels(transcriptionModel: String) -> [String] {
        ModelLocator.resolve(transcriptionModel) == nil ? ["Model"] : []
    }

    // MARK: - Models

    private func setBatchCommitEnabled(_ newValue: Bool) {
        if newValue, ModelLocator.resolve(Config.shared.transcriptionModel) == nil {
            showAlert(
                title: "Model missing",
                message: "Pick or download a valid model before enabling the final accuracy pass."
            )
            syncSettingsWindow()
            return
        }
        guard Config.shared.batchCommitEnabled != newValue else { return }
        applyModelConfig(batchCommitEnabled: newValue)
        yuwpLog("Final accuracy pass \(newValue ? "enabled" : "disabled")")
    }

    private func applyModelSpec(_ spec: String) {
        let trimmed = spec.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            syncSettingsWindow()
            return
        }

        if ModelLocator.resolve(trimmed) != nil {
            applyModelConfig(transcriptionModel: trimmed)
            yuwpLog("Model changed to: \(trimmed)")
            return
        }

        if ModelLocator.isRepoId(trimmed) {
            let alert = NSAlert()
            alert.messageText = "Download model from Hugging Face?"
            alert.informativeText = "Yuwp couldn't find `\(trimmed)` locally. Download it now into Application Support so the app can manage it directly? This same model is used for live decoding and the optional final accuracy pass."
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

    private func chooseTranscriptionModelDirectory() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.directoryURL = ModelLocator.localDirectory(for: Config.shared.transcriptionModel)
            ?? ModelLocator.managedRoot()
        panel.message = "Choose a model folder containing config.json, model.safetensors, vocab.json, and merges.txt."

        guard panel.runModal() == .OK, let url = panel.url else {
            syncSettingsWindow()
            return
        }

        let path = url.standardizedFileURL.path
        applyModelSpec(path)
    }

    private func applyModelConfig(
        transcriptionModel: String? = nil,
        batchCommitEnabled: Bool? = nil
    ) {
        if session?.isActive == true { send(.sessionStopRequested) }

        if let transcriptionModel {
            Config.shared.transcriptionModel = transcriptionModel
            asrProvider.transcriptionModel = transcriptionModel
        }
        if let batchCommitEnabled {
            Config.shared.batchCommitEnabled = batchCommitEnabled
            asrProvider.batchCommitEnabled = batchCommitEnabled
        }

        syncAppStateFromConfig()
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

    private func downloadStatusLabel(for repoId: String) -> String {
        if repoId == NativeASRProvider.defaultAlignerModel {
            return "Word-level alignment model"
        }
        return ModelLocator.shortRepoName(repoId)
    }

    private func downloadModel(repoId: String, activateForTranscription: Bool = true) async {
        let statusLabel = downloadStatusLabel(for: repoId)
        send(.modelDownloadStatusChanged(repoId: repoId, status: "Preparing \(statusLabel)…"))

        do {
            let _ = try await ModelDownloadManager.shared.download(repoId: repoId) { [weak self] progress, status in
                Task { @MainActor [weak self] in
                    guard let self else { return }
                    let pct = Int(progress * 100)
                    let message = "\(statusLabel) \(pct)% — \(status)"
                    yuwpLog("Model download progress: \(message)")
                    self.send(.modelDownloadStatusChanged(repoId: repoId, status: message))
                }
            }

            send(.modelDownloadStatusChanged(repoId: repoId, status: nil))
            if activateForTranscription {
                applyModelConfig(transcriptionModel: repoId)
            } else {
                syncAppStateFromConfig()
                restartProviderForSettingsChange()
            }
            yuwpLog("Downloaded model: \(repoId)")
        } catch {
            send(.modelDownloadStatusChanged(repoId: repoId, status: nil))
            syncSettingsWindow()
            showAlert(title: "Model download failed", message: error.localizedDescription)
            yuwpLog("Model download failed: \(repoId) — \(error.localizedDescription)")
        }
    }

    // MARK: - Recording Settings

    @objc private func toggleSaveRecordings(_ sender: NSMenuItem) {
        applySaveRecordings(!Config.shared.saveRecordings)
    }

    private func applySaveRecordings(_ enabled: Bool) {
        guard Config.shared.saveRecordings != enabled else {
            syncSettingsWindow()
            return
        }
        Config.shared.saveRecordings = enabled
        asrProvider.saveRecordings = enabled
        restartProviderForSettingsChange()
        syncSettingsWindow()
        yuwpLog("Save recordings \(enabled ? "enabled" : "disabled")")
    }

    private func applyDiagnosticLogging(_ enabled: Bool) {
        guard Config.shared.diagnosticLoggingEnabled != enabled else {
            syncSettingsWindow()
            return
        }

        Config.shared.diagnosticLoggingEnabled = enabled
        setYuwpDiagnosticLoggingEnabled(enabled)
        asrProvider.diagnosticLoggingEnabled = enabled
        restartProviderForSettingsChange()
        syncSettingsWindow()

        if enabled {
            yuwpLog("Diagnostic logging enabled")
        }
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
            asrProvider.recordingsDir = Config.shared.recordingsDir
            restartProviderForSettingsChange()
            syncSettingsWindow()
            yuwpLog("Recordings location changed to: \(Config.shared.recordingsDir.path)")
        }
    }

    private func resetRecordingsDirectory() {
        Config.shared.resetRecordingsDir()
        asrProvider.recordingsDir = Config.shared.recordingsDir
        restartProviderForSettingsChange()
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

    private func saveRecording(_ pcmData: Data, sessionID: String?) -> RecordingArtifactHandle? {
        let dir = Config.shared.recordingsDir
        let context = RecordingArtifactContext(
            sessionID: sessionID,
            transcriptionModel: activeTranscriptionModel ?? Config.shared.transcriptionModel,
            dictationLanguageMode: activeDictationLanguageMode ?? Config.shared.dictationLanguageMode,
            languageHint: activeDictationLanguageHint
        )

        do {
            let artifact = try RecordingArtifactWriter.writeRecording(
                pcmData: pcmData,
                directory: dir,
                context: context
            )
            let sid = sessionID ?? "unknown"
            yuwpLog(
                "Recording saved: sid=\(sid) path=\(artifact.audioURL.path) "
                    + "(\(String(format: "%.1f", Double(pcmData.count) / 32000))s)"
            )
            return artifact
        } catch {
            yuwpLog("Failed to save recording: \(error)")
            return nil
        }
    }
}

extension AppDelegate: SPUStandardUserDriverDelegate {
    nonisolated var supportsGentleScheduledUpdateReminders: Bool { true }

    nonisolated func standardUserDriverShouldHandleShowingScheduledUpdate(
        _ update: SUAppcastItem,
        andInImmediateFocus immediateFocus: Bool
    ) -> Bool {
        immediateFocus
    }

    nonisolated func standardUserDriverWillHandleShowingUpdate(
        _ handleShowingUpdate: Bool,
        forUpdate update: SUAppcastItem,
        state: SPUUserUpdateState
    ) {
        guard !handleShowingUpdate else { return }
        Task { @MainActor in
            self.setSparkleUpdateAvailable(true)
        }
    }

    nonisolated func standardUserDriverDidReceiveUserAttention(forUpdate update: SUAppcastItem) {
        Task { @MainActor in
            self.setSparkleUpdateAvailable(false)
        }
    }

    nonisolated func standardUserDriverWillFinishUpdateSession() {
        Task { @MainActor in
            self.setSparkleUpdateAvailable(false)
        }
    }
}
