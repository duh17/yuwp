import AppKit

/// Floating dictation indicator with reactive audio waveform.
/// Translucent dark pill with animated bars that respond to mic input,
/// plus multiline transcript text.
@MainActor
final class MicPanel {
    private var panel: NSPanel?
    private var contentView: NSView?
    private var textView: NSTextView?
    private var waveformBars: [CALayer] = []
    private var animationTimer: Timer?

    // Audio level state
    private var targetLevel: Float = 0
    private var smoothLevel: Float = 0
    private var barPhase: Float = 0

    // Layout constants
    private let panelWidth: CGFloat = 340
    private let minHeight: CGFloat = 40
    private let maxHeight: CGFloat = 260
    private let cornerRadius: CGFloat = 12
    private let padding: CGFloat = 10

    // Waveform constants
    private let barCount = 5
    private let barWidth: CGFloat = 3
    private let barSpacing: CGFloat = 3
    private let barMinHeight: CGFloat = 4
    private let barMaxHeight: CGFloat = 20
    private let waveformLeft: CGFloat = 12

    // Per-bar "personality" — center bar is tallest, edges are shorter
    private let barScale: [CGFloat] = [0.5, 0.8, 1.0, 0.8, 0.5]
    // Phase offsets so bars don't all move identically
    private let barPhaseOffset: [Float] = [0, 0.7, 1.4, 2.1, 2.8]

    func show(near position: NSPoint) {
        if panel == nil { createPanel() }
        guard let panel else { return }

        let mouse = NSEvent.mouseLocation
        panel.setFrameOrigin(NSPoint(x: mouse.x + 16, y: mouse.y - minHeight - 16))

        textView?.string = ""
        resizeToFit()
        panel.orderFront(nil)
        startAnimation()
    }

    func updateTranscript(_ text: String) {
        textView?.string = text
        resizeToFit()
    }

    /// Feed normalized audio level (0.0–1.0).
    func updateAudioLevel(_ level: Float) {
        targetLevel = level
    }

    func hide() {
        stopAnimation()
        panel?.orderOut(nil)
        targetLevel = 0
        smoothLevel = 0
    }

    // MARK: - Animation (main thread timer at ~60fps)

    private func startAnimation() {
        guard animationTimer == nil else { return }
        animationTimer = Timer.scheduledTimer(withTimeInterval: 1.0 / 60.0, repeats: true) {
            [weak self] _ in
            Task { @MainActor in self?.tick() }
        }
    }

    private func stopAnimation() {
        animationTimer?.invalidate()
        animationTimer = nil
    }

    /// Called every frame (~60fps) to update bar heights.
    private func tick() {
        // Smooth toward target (fast attack, slower decay)
        let attack: Float = 0.4
        let decay: Float = 0.15
        let factor = targetLevel > smoothLevel ? attack : decay
        smoothLevel += (targetLevel - smoothLevel) * factor
        barPhase += 0.08

        guard let contentView else { return }
        let panelH = contentView.bounds.height

        CATransaction.begin()
        CATransaction.setDisableActions(true)

        for i in 0..<barCount {
            guard i < waveformBars.count else { break }
            let bar = waveformBars[i]

            // Base height from audio level, scaled per bar
            let levelContribution = CGFloat(smoothLevel) * barScale[i] * barMaxHeight

            // Subtle idle oscillation (visible even during silence)
            let phase = barPhase + barPhaseOffset[i]
            let idle = (sin(phase) * 0.5 + 0.5) * 2.0 // 0–2pt oscillation

            let height = max(barMinHeight, levelContribution + CGFloat(idle))
            let x = waveformLeft + CGFloat(i) * (barWidth + barSpacing)
            let y = (panelH - height) / 2

            bar.frame = CGRect(x: x, y: y, width: barWidth, height: height)

            // Brightness follows level
            let brightness = 0.5 + CGFloat(smoothLevel) * 0.5
            bar.backgroundColor = NSColor(white: 1.0, alpha: brightness).cgColor
        }

        CATransaction.commit()
    }

    // MARK: - Layout

    private func resizeToFit() {
        guard let panel, let textView, let contentView else { return }

        let waveformWidth = waveformLeft + CGFloat(barCount) * (barWidth + barSpacing) + 8
        let textWidth = panelWidth - waveformWidth - padding
        let text = textView.string

        var neededHeight = minHeight
        if !text.isEmpty {
            let storage = NSTextStorage(string: text, attributes: [
                .font: NSFont.systemFont(ofSize: 15, weight: .regular),
            ])
            let container = NSTextContainer(
                size: NSSize(width: textWidth, height: .greatestFiniteMagnitude)
            )
            let lm = NSLayoutManager()
            lm.addTextContainer(container)
            storage.addLayoutManager(lm)
            lm.ensureLayout(for: container)
            let textH = lm.usedRect(for: container).height
            neededHeight = min(max(textH + 20, minHeight), maxHeight)
        }

        var frame = panel.frame
        let bottom = frame.origin.y + frame.height - neededHeight
        frame.size.height = neededHeight
        frame.origin.y = bottom
        panel.setFrame(frame, display: true, animate: false)

        contentView.frame = NSRect(x: 0, y: 0, width: panelWidth, height: neededHeight)

        let waveformAreaWidth = waveformLeft + CGFloat(barCount) * (barWidth + barSpacing) + 8
        textView.frame = NSRect(
            x: waveformAreaWidth,
            y: 6,
            width: textWidth,
            height: neededHeight - 12
        )
    }

    // MARK: - Create

    private func createPanel() {
        let p = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: panelWidth, height: minHeight),
            styleMask: [.nonactivatingPanel, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )
        p.isFloatingPanel = true
        p.becomesKeyOnlyIfNeeded = true
        p.level = .statusBar
        p.isOpaque = false
        p.backgroundColor = .clear
        p.hasShadow = true
        p.hidesOnDeactivate = false
        p.titleVisibility = .hidden
        p.titlebarAppearsTransparent = true
        p.isMovableByWindowBackground = true

        // Dark translucent pill
        let cv = NSView(frame: NSRect(x: 0, y: 0, width: panelWidth, height: minHeight))
        cv.wantsLayer = true
        cv.layer?.cornerRadius = cornerRadius
        cv.layer?.masksToBounds = true
        cv.layer?.backgroundColor = NSColor(white: 0.12, alpha: 0.92).cgColor
        cv.layer?.borderWidth = 0.5
        cv.layer?.borderColor = NSColor(white: 1, alpha: 0.08).cgColor

        // Waveform bars
        for i in 0..<barCount {
            let bar = CALayer()
            let x = waveformLeft + CGFloat(i) * (barWidth + barSpacing)
            bar.frame = CGRect(x: x, y: (minHeight - barMinHeight) / 2,
                               width: barWidth, height: barMinHeight)
            bar.cornerRadius = barWidth / 2
            bar.backgroundColor = NSColor(white: 1.0, alpha: 0.5).cgColor
            cv.layer?.addSublayer(bar)
            waveformBars.append(bar)
        }

        // Text view
        let waveformAreaWidth = waveformLeft + CGFloat(barCount) * (barWidth + barSpacing) + 8
        let textWidth = panelWidth - waveformAreaWidth - padding
        let tv = NSTextView(frame: NSRect(x: waveformAreaWidth, y: 6,
                                          width: textWidth, height: minHeight - 12))
        tv.isEditable = false
        tv.isSelectable = false
        tv.drawsBackground = false
        tv.font = .systemFont(ofSize: 15, weight: .regular)
        tv.textColor = NSColor(white: 1, alpha: 0.9)
        tv.isVerticallyResizable = false
        tv.isHorizontallyResizable = false
        tv.textContainer?.widthTracksTextView = true
        tv.textContainer?.lineFragmentPadding = 0
        cv.addSubview(tv)
        textView = tv

        p.contentView = cv
        contentView = cv
        panel = p
    }
}
