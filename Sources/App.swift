import AppKit
import AVFoundation

// Yuwp — system-wide voice dictation for macOS
// Press hotkey → speak → text streams into any focused text field
// Powered by Qwen3-ASR via local Python sidecar (mlx-audio)

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
    private let asrSidecar: ASRSidecar = {
        let s = ASRSidecar()
        s.streamingModel = Config.shared.streamingModel
        s.batchModel = Config.shared.batchModel
        s.batchRetranscribeEnabled = Config.shared.batchRetranscribeEnabled
        return s
    }()
    private let audioCapture = AudioCapture()
    private let textInjector = TextInjector()
    private let micPanel = MicPanel()

    // Per-dictation session (created on start, torn down on stop)
    private var session: DictationSession?

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
        startSttProvider()
        requestMicPermission()
        checkPermission()
    }

    // MARK: - Hotkey Toggle

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

        let s = DictationSession(
            sttSession: asrSidecar.makeSession(),
            textInjector: textInjector,
            audioCapture: audioCapture
        )
        s.onEvent = { [weak self] event in self?.handleSessionEvent(event) }
        session = s

        // Update menu bar icon
        statusItem.button?.image = NSImage(
            systemSymbolName: "waveform.circle.fill",
            accessibilityDescription: "Yuwp — Listening"
        )

        // Show full panel initially; switches to compact dot if AX verifies
        micPanel.show(near: textInjector.targetPosition)

        s.start()
    }

    private func stopDictation() {
        guard let s = session else { return }
        let pcmData = s.stop()

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
        case .liveInjectionVerified(let caret):
            micPanel.showCompact(near: caret)

        case .partialTranscript(let text):
            micPanel.updateTranscript(text)

        case .caretMoved(let point):
            micPanel.showCompact(near: point)

        case .audioLevel(let level):
            micPanel.updateAudioLevel(level)

        case .finished:
            micPanel.hide()
            session = nil
        }
    }

    // MARK: - STT Provider

    private func startSttProvider() {
        asrSidecar.onReady = { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                self.providerReady = true
                if self.hasPermission { self.updateMenuForReady() }
                yuwpLog("STT provider ready")
            }
        }
        asrSidecar.onError = { error in
            Task { @MainActor in yuwpLog("STT error: \(error)") }
        }
        asrSidecar.start()
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

            statusMenuItem.title = "⚠ Grant Accessibility Permission"
            statusMenuItem.action = #selector(openAccessibilitySettings)
            statusMenuItem.target = self
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
        updateMenuForReady()
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

    private func updateMenuForReady() {
        if providerReady {
            statusMenuItem.title = "✓ Ready"
            statusMenuItem.action = nil
            statusMenuItem.isEnabled = false
        } else {
            statusMenuItem.title = "Loading model..."
            statusMenuItem.action = nil
            statusMenuItem.isEnabled = false
        }
        hotkeyMenuItem.title = "Hotkey: \(Config.shared.hotkeyMode.description)"
        modelMenuItem.title = modelMenuTitle()
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

        // Update sidecar and restart
        asrSidecar.streamingModel = preset.streamingModel
        asrSidecar.batchModel = preset.batchModel
        asrSidecar.batchRetranscribeEnabled = preset.batchEnabled

        providerReady = false
        updateMenuForReady()
        rebuildModelSubmenu()
        asrSidecar.shutdown()
        asrSidecar.start()

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
