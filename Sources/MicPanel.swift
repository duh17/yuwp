import AppKit

/// Floating dictation indicator.
/// Text-only pill with audio-reactive border glow that follows the mouse cursor.
/// In AX mode, shows a compact waveform dot near the text caret instead.
@MainActor
final class MicPanel {
    // Main pill
    private var panel: NSPanel?
    private var contentView: NSView?
    private var textView: NSTextView?

    // Compact dot (AX mode — waveform bars near caret)
    private var compactMode = false
    private var compactPanel: NSPanel?
    private var compactView: NSView?
    private var compactBars: [CALayer] = []

    // Animation state
    private var animationTimer: Timer?
    private var targetLevel: Float = 0
    private var smoothLevel: Float = 0
    private var barPhase: Float = 0

    // Layout constants
    private let panelWidth: CGFloat = 320
    private let minHeight: CGFloat = 36
    private let cornerRadius: CGFloat = 10
    private let textPadding: CGFloat = 14
    private let verticalPadding: CGFloat = 8
    private let compactSize: CGFloat = 28
    private let cursorGap: CGFloat = 20 // gap between mouse and panel bottom

    // Glow
    private let glowColor = NSColor(calibratedRed: 0.5, green: 0.7, blue: 1.0, alpha: 1.0)

    // Compact waveform
    private let barCount = 5
    private let barWidth: CGFloat = 3
    private let barScale: [CGFloat] = [0.5, 0.8, 1.0, 0.8, 0.5]
    private let barPhaseOffset: [Float] = [0, 0.7, 1.4, 2.1, 2.8]

    // MARK: - Public API

    /// Show the text pill near a position (clipboard-fallback mode).
    /// The pill follows the mouse cursor while visible.
    func show(near position: NSPoint) {
        compactMode = false
        if panel == nil { createPanel() }
        guard let panel else { return }

        let anchor = (position == .zero) ? NSEvent.mouseLocation : position
        let size = panel.frame.size
        let origin = clampToScreen(
            NSPoint(x: anchor.x - size.width / 2, y: anchor.y + cursorGap),
            panelSize: size
        )
        panel.setFrameOrigin(origin)

        textView?.string = ""
        resizeToFit()
        compactPanel?.orderOut(nil)
        panel.alphaValue = 1
        panel.orderFront(nil)
        startAnimation()
    }

    /// Show a tiny waveform-only dot near the caret.
    /// Used when AX injection is live and the full pill is redundant.
    func showCompact(near position: NSPoint) {
        compactMode = true
        if compactPanel == nil { createCompactPanel() }
        guard let cp = compactPanel else { return }

        let anchor = (position == .zero) ? NSEvent.mouseLocation : position
        let origin = clampToScreen(
            NSPoint(x: anchor.x + 4, y: anchor.y + 4),
            panelSize: cp.frame.size
        )
        cp.setFrameOrigin(origin)

        panel?.orderOut(nil)
        cp.alphaValue = 1
        cp.orderFront(nil)
        startAnimation()
    }

    func updateTranscript(_ text: String) {
        textView?.string = text
        resizeToFit()
    }

    /// Feed normalized audio level (0.0-1.0).
    func updateAudioLevel(_ level: Float) {
        targetLevel = level
    }

    func hide() {
        stopAnimation()
        panel?.orderOut(nil)
        compactPanel?.orderOut(nil)
        compactMode = false
        targetLevel = 0
        smoothLevel = 0
    }

    // MARK: - Screen Clamping

    private func clampToScreen(_ point: NSPoint, panelSize: NSSize) -> NSPoint {
        guard let screen = NSScreen.main?.visibleFrame else { return point }
        var x = point.x
        var y = point.y

        if x + panelSize.width > screen.maxX { x = screen.maxX - panelSize.width - 8 }
        if x < screen.minX { x = screen.minX + 8 }
        if y + panelSize.height > screen.maxY { y = screen.maxY - panelSize.height - 8 }
        if y < screen.minY { y = screen.minY + 8 }
        return NSPoint(x: x, y: y)
    }

    // MARK: - Animation Loop

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

    private func tick() {
        // Smooth toward target (fast attack, slower decay)
        let attack: Float = 0.4
        let decay: Float = 0.15
        let factor = targetLevel > smoothLevel ? attack : decay
        smoothLevel += (targetLevel - smoothLevel) * factor
        barPhase += 0.08

        CATransaction.begin()
        CATransaction.setDisableActions(true)

        if compactMode {
            tickCompact()
        } else {
            tickPill()
        }

        CATransaction.commit()
    }

    /// Animate the pill's border glow and follow the mouse cursor.
    private func tickPill() {
        guard let panel, let contentView else { return }

        // Border glow: brightens with audio level
        let level = CGFloat(smoothLevel)
        let borderWidth = 0.5 + level * 1.5
        let borderAlpha = 0.08 + level * 0.35
        contentView.layer?.borderWidth = borderWidth
        contentView.layer?.borderColor = glowColor.withAlphaComponent(borderAlpha).cgColor

        // Follow mouse cursor (smooth lerp — panel gently trails the cursor)
        let mouse = NSEvent.mouseLocation
        let targetX = mouse.x - panelWidth / 2
        let targetY = mouse.y + cursorGap
        let current = panel.frame.origin
        let lerp: CGFloat = 0.12
        let newX = current.x + (targetX - current.x) * lerp
        let newY = current.y + (targetY - current.y) * lerp

        let clamped = clampToScreen(
            NSPoint(x: newX, y: newY),
            panelSize: panel.frame.size
        )
        panel.setFrameOrigin(clamped)
    }

    /// Animate compact waveform bars.
    private func tickCompact() {
        let ch = compactSize
        for i in 0..<compactBars.count {
            let bar = compactBars[i]
            let levelContribution = CGFloat(smoothLevel) * barScale[i] * (ch * 0.6)
            let phase = barPhase + barPhaseOffset[i]
            let idle = (sin(phase) * 0.5 + 0.5) * 1.5
            let height = max(2, levelContribution + CGFloat(idle))
            let x: CGFloat = 4 + CGFloat(i) * (barWidth + 2)
            let y = (ch - height) / 2
            bar.frame = CGRect(x: x, y: y, width: barWidth, height: height)
            let brightness = 0.5 + CGFloat(smoothLevel) * 0.5
            bar.backgroundColor = NSColor(
                calibratedRed: 1.0, green: 0.3, blue: 0.3, alpha: brightness
            ).cgColor
        }
    }

    // MARK: - Layout

    private func resizeToFit() {
        guard let panel, let textView, let contentView else { return }

        let textWidth = panelWidth - textPadding * 2
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

            // No fixed cap — grow up to screen height minus margin
            let screenH = NSScreen.main?.visibleFrame.height ?? 800
            let maxHeight = screenH - 100
            neededHeight = min(max(textH + verticalPadding * 2, minHeight), maxHeight)
        }

        // Keep bottom edge fixed, grow upward
        var frame = panel.frame
        frame.size.height = neededHeight
        panel.setFrame(frame, display: true, animate: false)

        contentView.frame = NSRect(x: 0, y: 0, width: panelWidth, height: neededHeight)
        textView.frame = NSRect(
            x: textPadding,
            y: verticalPadding,
            width: textWidth,
            height: neededHeight - verticalPadding * 2
        )
    }

    // MARK: - Panel Creation

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
        p.isMovableByWindowBackground = false // mouse-follow handles positioning

        // Dark translucent pill — text only, no waveform bars
        let cv = NSView(frame: NSRect(x: 0, y: 0, width: panelWidth, height: minHeight))
        cv.wantsLayer = true
        cv.layer?.cornerRadius = cornerRadius
        cv.layer?.masksToBounds = true
        cv.layer?.backgroundColor = NSColor(white: 0.10, alpha: 0.92).cgColor
        cv.layer?.borderWidth = 0.5
        cv.layer?.borderColor = NSColor(white: 1, alpha: 0.08).cgColor

        // Text view — full-width, no waveform bars eating space
        let textWidth = panelWidth - textPadding * 2
        let tv = NSTextView(frame: NSRect(
            x: textPadding, y: verticalPadding,
            width: textWidth, height: minHeight - verticalPadding * 2
        ))
        tv.isEditable = false
        tv.isSelectable = false
        tv.drawsBackground = false
        tv.font = .systemFont(ofSize: 15, weight: .regular)
        tv.textColor = NSColor(white: 1, alpha: 0.92)
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

    private func createCompactPanel() {
        let size = compactSize
        let p = NSPanel(
            contentRect: NSRect(x: 0, y: 0, width: size, height: size),
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

        let cv = NSView(frame: NSRect(x: 0, y: 0, width: size, height: size))
        cv.wantsLayer = true
        cv.layer?.cornerRadius = size / 2
        cv.layer?.masksToBounds = true
        cv.layer?.backgroundColor = NSColor(white: 0.12, alpha: 0.85).cgColor
        cv.layer?.borderWidth = 0.5
        cv.layer?.borderColor = NSColor(
            calibratedRed: 1.0, green: 0.3, blue: 0.3, alpha: 0.3
        ).cgColor

        compactBars = []
        for i in 0..<barCount {
            let bar = CALayer()
            let x: CGFloat = 4 + CGFloat(i) * (barWidth + 2)
            bar.frame = CGRect(x: x, y: (size - 2) / 2, width: barWidth, height: 2)
            bar.cornerRadius = barWidth / 2
            bar.backgroundColor = NSColor(
                calibratedRed: 1.0, green: 0.3, blue: 0.3, alpha: 0.5
            ).cgColor
            cv.layer?.addSublayer(bar)
            compactBars.append(bar)
        }

        p.contentView = cv
        compactView = cv
        compactPanel = p
    }
}
