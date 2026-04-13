import AppKit
import SwiftUI

@MainActor
final class SettingsWindowController: NSWindowController {
    let store: SettingsStore
    private let hostingController: NSHostingController<SettingsView>

    init(store: SettingsStore) {
        self.store = store
        self.hostingController = NSHostingController(rootView: SettingsView(store: store))

        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 980, height: 760),
            styleMask: [.titled, .closable],
            backing: .buffered,
            defer: false
        )
        window.title = "Yuwp Settings"
        window.center()
        window.minSize = NSSize(width: 900, height: 620)
        window.contentViewController = hostingController

        super.init(window: window)
    }

    @available(*, unavailable)
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }

    func sync(_ snapshot: SettingsSnapshot) {
        store.sync(snapshot)
    }
}
