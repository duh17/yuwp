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
    }

    @objc private func changeModel(_ sender: NSMenuItem) {
        guard let idx = sender.representedObject as? Int,
              idx < ModelPreset.presets.count else { return }
        let preset = ModelPreset.presets[idx]

        // Skip if already active
        if ModelPreset.current()?.label == preset.label { return }

        // Stop any active dictation
        if session?.isActive == true { stopDictation() }

        // Persist
        Config.shared.streamingModel = preset.streamingModel
        Config.shared.batchModel = preset.batchModel
        Config.shared.batchRetranscribeEnabled = preset.batchEnabled

        // Update server and restart — onStateChange handles menu updates
        asrProvider.streamingModel = preset.streamingModel
        asrProvider.batchModel = preset.batchModel
        asrProvider.batchRetranscribeEnabled = preset.batchEnabled
        asrProvider.shutdown()
        asrProvider.start()
        rebuildModelSubmenu()

        yuwpLog("Model changed to: \(preset.label) (\(preset.summary))")
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
