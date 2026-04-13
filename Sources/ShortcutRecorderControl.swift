import AppKit
import SwiftUI

@MainActor
final class ShortcutRecorderView: NSStackView {
    var onChange: ((KeyBinding) -> Void)?
    var onRecordingChange: ((Bool) -> Void)?
    var binding: KeyBinding {
        didSet {
            guard !isRecording else { return }
            bindingLabel.stringValue = binding.description
        }
    }

    private let defaultBinding: KeyBinding
    private let bindingLabel = NSTextField(labelWithString: "")
    private let recordButton = NSButton(title: "Record…", target: nil, action: nil)
    private let resetButton = NSButton(title: "Reset Default", target: nil, action: nil)
    private var recordingMonitor: Any?
    private var captureTimer: Timer?
    private var isRecording = false
    private var captureState = ShortcutCaptureState()

    init(defaultBinding: KeyBinding) {
        self.defaultBinding = defaultBinding
        self.binding = defaultBinding
        super.init(frame: .zero)
        orientation = .horizontal
        alignment = .centerY
        spacing = 8
        setup()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }


    private func setup() {
        bindingLabel.font = .monospacedSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)
        bindingLabel.stringValue = binding.description
        bindingLabel.setContentHuggingPriority(.defaultLow, for: .horizontal)
        bindingLabel.widthAnchor.constraint(greaterThanOrEqualToConstant: 120).isActive = true

        recordButton.target = self
        recordButton.action = #selector(toggleRecording(_:))
        recordButton.bezelStyle = .rounded

        resetButton.target = self
        resetButton.action = #selector(resetToDefault(_:))
        resetButton.bezelStyle = .rounded

        addArrangedSubview(bindingLabel)
        addArrangedSubview(recordButton)
        addArrangedSubview(resetButton)
    }

    @objc private func toggleRecording(_ sender: NSButton) {
        isRecording ? stopRecording() : startRecording()
    }

    @objc private func resetToDefault(_ sender: NSButton) {
        binding = defaultBinding
        onChange?(defaultBinding)
    }

    private func startRecording() {
        guard !isRecording else { return }
        captureState = ShortcutCaptureState()
        isRecording = true
        bindingLabel.stringValue = "Press shortcut… Double-tap a modifier to record it that way."
        recordButton.title = "Cancel"
        onRecordingChange?(true)

        recordingMonitor = NSEvent.addLocalMonitorForEvents(matching: [.keyDown, .flagsChanged]) { [weak self] event in
            guard let self else { return event }

            let result: ShortcutCaptureResult
            let now = CFAbsoluteTimeGetCurrent()
            switch event.type {
            case .keyDown:
                result = self.captureState.handleKeyDown(
                    keyCode: UInt16(event.keyCode),
                    modifiers: Self.modifierMask(from: event.modifierFlags),
                    timestamp: now
                )
            case .flagsChanged:
                result = self.captureState.handleFlagsChanged(
                    keyCode: UInt16(event.keyCode),
                    timestamp: now
                )
                self.updateRecordingHint()
            default:
                return event
            }

            self.scheduleCaptureTimerIfNeeded()
            return self.handleCaptureResult(result)
        }
    }

    private func stopRecording() {
        let wasRecording = isRecording
        if let monitor = recordingMonitor {
            NSEvent.removeMonitor(monitor)
            recordingMonitor = nil
        }
        captureTimer?.invalidate()
        captureTimer = nil
        captureState = ShortcutCaptureState()
        isRecording = false
        bindingLabel.stringValue = binding.description
        recordButton.title = "Record…"
        if wasRecording {
            onRecordingChange?(false)
        }
    }

    private func handleCaptureResult(_ result: ShortcutCaptureResult) -> NSEvent? {
        switch result {
        case .none:
            return nil
        case .captured(let binding):
            self.binding = binding
            onChange?(binding)
            stopRecording()
            return nil
        case .cancelled:
            stopRecording()
            return nil
        case .invalid:
            NSSound.beep()
            bindingLabel.stringValue = "Use a modifier combo, tap one modifier once, or double-tap one modifier."
            return nil
        }
    }

    private func updateRecordingHint() {
        if let pendingModifierKeyCode = captureState.pendingModifierKeyCode {
            bindingLabel.stringValue = "Release \(KeyBinding.keyName(for: pendingModifierKeyCode)) to use it, or press another key for a combo…"
            return
        }

        if let modifierKeyCode = captureState.pendingSingleModifierKeyCode {
            bindingLabel.stringValue = "Double-tap \(KeyBinding.keyName(for: modifierKeyCode)) for a double-tap hotkey, or wait to keep it as a single modifier."
        }
    }

    private func scheduleCaptureTimerIfNeeded() {
        captureTimer?.invalidate()
        captureTimer = nil

        guard let deadline = captureState.pendingModifierCaptureDeadline else { return }
        let interval = max(0, deadline - CFAbsoluteTimeGetCurrent())
        captureTimer = Timer.scheduledTimer(withTimeInterval: interval, repeats: false) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self else { return }
                let result = self.captureState.resolveTimeout(at: CFAbsoluteTimeGetCurrent()) ?? .none
                _ = self.handleCaptureResult(result)
            }
        }
    }

    private static func modifierMask(from flags: NSEvent.ModifierFlags) -> UInt64 {
        let filtered = flags.intersection(.deviceIndependentFlagsMask)
        var result: UInt64 = 0
        if filtered.contains(.control) { result |= 0x40000 }
        if filtered.contains(.option) { result |= 0x80000 }
        if filtered.contains(.command) { result |= 0x100000 }
        if filtered.contains(.shift) { result |= 0x20000 }
        return result
    }
}

struct ShortcutRecorderRepresentable: NSViewRepresentable {
    @Binding var binding: KeyBinding
    let defaultBinding: KeyBinding
    var onRecordingChange: ((Bool) -> Void)?

    func makeNSView(context: Context) -> ShortcutRecorderView {
        let view = ShortcutRecorderView(defaultBinding: defaultBinding)
        view.binding = binding
        view.onChange = { newBinding in
            self.binding = newBinding
        }
        view.onRecordingChange = { isRecording in
            onRecordingChange?(isRecording)
        }
        return view
    }

    func updateNSView(_ nsView: ShortcutRecorderView, context: Context) {
        if nsView.binding != binding {
            nsView.binding = binding
        }
        nsView.onChange = { newBinding in
            self.binding = newBinding
        }
        nsView.onRecordingChange = { isRecording in
            onRecordingChange?(isRecording)
        }
    }
}
