import AppKit

@MainActor
final class SettingsWindowController: NSWindowController {
    var onDictationModeChange: ((DictationInteractionMode) -> Void)?
    var onDictationBindingChange: ((KeyBinding) -> Void)?
    var onServerModeChange: ((ServerMode) -> Void)?
    var onServerPortChange: ((UInt16) -> Void)?

    private let dictationModeControl = NSSegmentedControl(labels: ["Toggle", "Push to Talk"], trackingMode: .selectOne, target: nil, action: nil)
    private let shortcutRecorder = ShortcutRecorderView(defaultBinding: .ctrlBacktick)
    private let serverModePopup = NSPopUpButton(frame: .zero, pullsDown: false)
    private let serverPortField = NSTextField(frame: .zero)
    private let applyPortButton = NSButton(title: "Apply", target: nil, action: nil)

    init() {
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 420, height: 240),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Yuwp Settings"
        window.center()
        super.init(window: window)
        buildUI()
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func sync(
        dictationMode: DictationInteractionMode,
        dictationBinding: KeyBinding,
        serverMode: ServerMode,
        serverPort: UInt16
    ) {
        dictationModeControl.selectedSegment = dictationMode == .toggle ? 0 : 1
        shortcutRecorder.binding = dictationBinding
        if let index = ServerMode.allCases.firstIndex(of: serverMode) {
            serverModePopup.selectItem(at: index)
        }
        serverPortField.stringValue = "\(serverPort)"
    }

    private func buildUI() {
        guard let contentView = window?.contentView else { return }

        dictationModeControl.target = self
        dictationModeControl.action = #selector(dictationModeChanged(_:))

        serverModePopup.addItems(withTitles: ServerMode.allCases.map(\.description))
        serverModePopup.target = self
        serverModePopup.action = #selector(serverModeChanged(_:))

        serverPortField.placeholderString = "9748"
        serverPortField.alignment = .right
        serverPortField.font = .monospacedDigitSystemFont(ofSize: NSFont.systemFontSize, weight: .regular)

        applyPortButton.target = self
        applyPortButton.action = #selector(applyPort(_:))
        applyPortButton.bezelStyle = .rounded

        shortcutRecorder.onChange = { [weak self] binding in
            self?.onDictationBindingChange?(binding)
        }

        let stack = NSStackView()
        stack.orientation = .vertical
        stack.alignment = .leading
        stack.spacing = 16
        stack.translatesAutoresizingMaskIntoConstraints = false

        let intro = label("Keep this page tiny. Configure the one dictation shortcut, one mode, and server exposure.")
        intro.textColor = .secondaryLabelColor
        intro.maximumNumberOfLines = 2
        stack.addArrangedSubview(intro)

        stack.addArrangedSubview(makeRow(label: "Dictation Mode", control: dictationModeControl))
        stack.addArrangedSubview(makeRow(label: "Shortcut", control: shortcutRecorder))
        stack.addArrangedSubview(makeRow(label: "Server Mode", control: serverModePopup))
        stack.addArrangedSubview(makePortRow())

        contentView.addSubview(stack)
        NSLayoutConstraint.activate([
            stack.leadingAnchor.constraint(equalTo: contentView.leadingAnchor, constant: 20),
            stack.trailingAnchor.constraint(equalTo: contentView.trailingAnchor, constant: -20),
            stack.topAnchor.constraint(equalTo: contentView.topAnchor, constant: 20),
        ])
    }

    private func makeRow(label title: String, control: NSView) -> NSView {
        let row = NSGridView(views: [[label(title), control]])
        row.translatesAutoresizingMaskIntoConstraints = false
        row.rowSpacing = 8
        row.columnSpacing = 16
        row.xPlacement = .leading
        row.column(at: 0).width = 110
        return row
    }

    private func makePortRow() -> NSView {
        let stack = NSStackView()
        stack.orientation = .horizontal
        stack.alignment = .centerY
        stack.spacing = 8

        serverPortField.widthAnchor.constraint(equalToConstant: 90).isActive = true
        stack.addArrangedSubview(serverPortField)
        stack.addArrangedSubview(applyPortButton)

        let hint = label("1–65535")
        hint.textColor = .secondaryLabelColor
        stack.addArrangedSubview(hint)

        return makeRow(label: "Server Port", control: stack)
    }

    private func label(_ string: String) -> NSTextField {
        let field = NSTextField(labelWithString: string)
        field.lineBreakMode = .byWordWrapping
        return field
    }

    @objc private func dictationModeChanged(_ sender: NSSegmentedControl) {
        let mode: DictationInteractionMode = sender.selectedSegment == 0 ? .toggle : .pushToTalk
        onDictationModeChange?(mode)
    }

    @objc private func serverModeChanged(_ sender: NSPopUpButton) {
        guard sender.indexOfSelectedItem >= 0,
              sender.indexOfSelectedItem < ServerMode.allCases.count else { return }
        onServerModeChange?(ServerMode.allCases[sender.indexOfSelectedItem])
    }

    @objc private func applyPort(_ sender: NSButton) {
        let trimmed = serverPortField.stringValue.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = Int(trimmed), (1...65_535).contains(value), let port = UInt16(exactly: value) else {
            NSSound.beep()
            return
        }
        onServerPortChange?(port)
        serverPortField.stringValue = "\(port)"
    }
}

@MainActor
private final class ShortcutRecorderView: NSStackView {
    var onChange: ((KeyBinding) -> Void)?
    var binding: KeyBinding {
        didSet { bindingLabel.stringValue = binding.description }
    }

    private let defaultBinding: KeyBinding
    private let bindingLabel = NSTextField(labelWithString: "")
    private let recordButton = NSButton(title: "Record…", target: nil, action: nil)
    private let resetButton = NSButton(title: "Reset Default", target: nil, action: nil)
    private var recordingMonitor: Any?
    private var isRecording = false

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
        isRecording = true
        bindingLabel.stringValue = "Press shortcut…"
        recordButton.title = "Cancel"

        recordingMonitor = NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            guard let self else { return event }
            if event.keyCode == 53 { // Esc
                self.stopRecording()
                return nil
            }

            let modifiers = Self.modifierMask(from: event.modifierFlags)
            guard modifiers != 0 else {
                NSSound.beep()
                return nil
            }

            let binding = KeyBinding(keyCode: UInt16(event.keyCode), modifiers: modifiers)
            self.binding = binding
            self.stopRecording()
            self.onChange?(binding)
            return nil
        }
    }

    private func stopRecording() {
        if let monitor = recordingMonitor {
            NSEvent.removeMonitor(monitor)
            recordingMonitor = nil
        }
        isRecording = false
        bindingLabel.stringValue = binding.description
        recordButton.title = "Record…"
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
