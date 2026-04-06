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
    private var statusItem: NSStatusItem!
    private let hotkeyManager = HotkeyManager()
    private let audioCapture = AudioCapture()
    private let textInjector = TextInjector()
    private let micPanel = MicPanel()
    private let sidecar = ASRSidecar()
    private let typewriter = TypewriterAnimator()
    private var isListening = false
    private var sidecarReady = false
    private var permissionTimer: Timer?
    private var hasPermission = false

    // Menu items that need dynamic updates
    private var statusMenuItem: NSMenuItem!
    private var hotkeyMenuItem: NSMenuItem!
    private var hotkeySubmenu: NSMenu!

    func applicationDidFinishLaunching(_ notification: Notification) {
        setupMenuBar()
        startSidecar()
        requestMicPermission()
        checkPermission()
    }

    // MARK: - Microphone Permission

    private func requestMicPermission() {
        // Force-request mic access at launch. On macOS, authorizationStatus
        // returns .authorized for non-sandboxed apps even without TCC access,
        // so we must call requestAccess unconditionally to trigger the prompt.
        AVCaptureDevice.requestAccess(for: .audio) { granted in
            Task { @MainActor in
                if granted {
                    yuwpLog("Microphone permission granted")
                } else {
                    yuwpLog("Microphone permission denied")
                }
            }
        }
    }

    // MARK: - Permission Onboarding

    private func checkPermission() {
        // Try to create the event tap directly — this is the ground truth
        // for whether we have Accessibility permission.
        // AXIsProcessTrusted() can return stale/wrong results.
        hotkeyManager.onToggle = { [weak self] in
            Task { @MainActor in self?.toggleListening() }
        }

        if hotkeyManager.start() {
            onPermissionGranted()
        } else {
            // Event tap failed — need Accessibility permission
            let opts = ["AXTrustedCheckOptionPrompt": true] as CFDictionary
            _ = AXIsProcessTrustedWithOptions(opts)

            statusMenuItem.title = "⚠ Grant Accessibility Permission"
            statusMenuItem.action = #selector(openAccessibilitySettings)
            statusMenuItem.target = self
            yuwpLog("Waiting for Accessibility permission...")

            // Poll by trying to create the tap
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

        // Status line
        statusMenuItem = NSMenuItem(title: "Loading model...", action: nil, keyEquivalent: "")
        statusMenuItem.isEnabled = false
        menu.addItem(statusMenuItem)
        menu.addItem(.separator())

        // Hotkey display + submenu
        hotkeyMenuItem = NSMenuItem(title: "Hotkey: \(Config.shared.hotkeyMode.description)", action: nil, keyEquivalent: "")
        hotkeySubmenu = NSMenu()
        rebuildHotkeySubmenu()
        hotkeyMenuItem.submenu = hotkeySubmenu
        menu.addItem(hotkeyMenuItem)

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

        // Restart event tap with new config
        if hasPermission {
            let ok = hotkeyManager.restart()
            if ok {
                hotkeyMenuItem.title = "Hotkey: \(mode.description)"
                rebuildHotkeySubmenu()
                yuwpLog("Hotkey changed to: \(mode.description)")
            }
        }
    }

    private func updateMenuForReady() {
        let mode = Config.shared.hotkeyMode
        if sidecarReady {
            statusMenuItem.title = "✓ Ready"
            statusMenuItem.action = nil
            statusMenuItem.isEnabled = false
        }
        hotkeyMenuItem.title = "Hotkey: \(mode.description)"
    }

    /// Compare two HotkeyMode values for equality (for menu checkmarks).
    private func modesMatch(_ a: HotkeyMode, _ b: HotkeyMode) -> Bool {
        switch (a, b) {
        case (.combo(let ak, let am), .combo(let bk, let bm)):
            return ak == bk && am == bm
        case (.doubleTap(let ak, _), .doubleTap(let bk, _)):
            return ak == bk
        default:
            return false
        }
    }

    // MARK: - Hotkey Toggle

    private func toggleListening() {
        if isListening {
            stopListening()
        } else {
            startListening()
        }
    }

    // MARK: - ASR Sidecar

    private func startSidecar() {
        sidecar.onReady = { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                self.sidecarReady = true
                if self.hasPermission { self.updateMenuForReady() }
                yuwpLog("ASR model loaded and ready")
            }
        }
        sidecar.onPartialResult = { [weak self] text in
            Task { @MainActor in
                guard let self else { return }
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                guard !trimmed.isEmpty, trimmed.lowercased() != "none" else { return }
                self.typewriter.update(fullText: text)
                self.textInjector.inject(self.typewriter.displayText)
                self.micPanel.updateTranscript(self.typewriter.displayText)
                self.driveTypewriterDisplay()
            }
        }
        sidecar.onFinalResult = { [weak self] text in
            Task { @MainActor in
                guard let self else { return }
                self.typewriter.commitCurrentAnimation()
                self.textInjector.commit(text)
                self.micPanel.updateTranscript(text)
                self.stopListening()
            }
        }
        sidecar.onError = { error in
            Task { @MainActor in
                yuwpLog("ASR error: \(error)")
            }
        }

        sidecar.start()
    }

    // MARK: - Listening

    private func startListening() {
        guard !isListening, sidecarReady else {
            if !sidecarReady {
                yuwpLog("Model still loading, please wait...")
            }
            return
        }

        isListening = true

        // Capture the focused element before showing any UI
        textInjector.captureTarget()

        // Update menu bar icon
        statusItem.button?.image = NSImage(
            systemSymbolName: "waveform.circle.fill",
            accessibilityDescription: "Yuwp — Listening"
        )

        // Show floating mic indicator
        micPanel.show(near: textInjector.targetPosition)

        // Tell sidecar to start a new session
        sidecar.beginSession()

        // Wire audio level to waveform visualization
        audioCapture.onAudioLevel = { [weak self] level in
            Task { @MainActor in
                self?.micPanel.updateAudioLevel(level)
            }
        }

        // Start audio capture and pipe PCM to sidecar
        audioCapture.start { [weak self] buffer in
            self?.sidecar.sendAudio(buffer)
        }

        yuwpLog("Listening...")
    }

    private func driveTypewriterDisplay() {
        guard typewriter.isAnimating else { return }
        Task { @MainActor in
            while typewriter.isAnimating {
                try? await Task.sleep(nanoseconds: 16_000_000)
                micPanel.updateTranscript(typewriter.displayText)
                textInjector.inject(typewriter.displayText)
            }
        }
    }

    private func stopListening() {
        guard isListening else { return }
        isListening = false

        let pcmData = audioCapture.stop()
        sidecar.endSession()
        typewriter.reset()
        textInjector.release()
        micPanel.hide()

        if let pcmData, !pcmData.isEmpty {
            saveRecording(pcmData)
        }

        statusItem.button?.image = NSImage(
            systemSymbolName: "waveform",
            accessibilityDescription: "Yuwp"
        )

        yuwpLog("Stopped.")
    }

    // MARK: - Recording

    private func saveRecording(_ pcmData: Data) {
        let dir = Config.shared.recordingsDir
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let filename = "yuwp-\(formatter.string(from: Date())).wav"
        let url = dir.appendingPathComponent(filename)

        let sampleRate: UInt32 = 16000
        let channels: UInt16 = 1
        let bitsPerSample: UInt16 = 16
        let byteRate = sampleRate * UInt32(channels) * UInt32(bitsPerSample / 8)
        let blockAlign = channels * (bitsPerSample / 8)
        let dataSize = UInt32(pcmData.count)

        var wav = Data()
        wav.append(contentsOf: "RIFF".utf8)
        wav.append(withUnsafeBytes(of: (36 + dataSize).littleEndian) { Data($0) })
        wav.append(contentsOf: "WAVE".utf8)
        wav.append(contentsOf: "fmt ".utf8)
        wav.append(withUnsafeBytes(of: UInt32(16).littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: UInt16(1).littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: channels.littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: sampleRate.littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: byteRate.littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: blockAlign.littleEndian) { Data($0) })
        wav.append(withUnsafeBytes(of: bitsPerSample.littleEndian) { Data($0) })
        wav.append(contentsOf: "data".utf8)
        wav.append(withUnsafeBytes(of: dataSize.littleEndian) { Data($0) })
        wav.append(pcmData)

        do {
            try wav.write(to: url)
            yuwpLog("Recording saved: \(url.path) (\(String(format: "%.1f", Double(pcmData.count) / 32000))s)")
        } catch {
            yuwpLog("Failed to save recording: \(error)")
        }
    }
}
