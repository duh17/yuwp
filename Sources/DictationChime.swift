import AppKit
import Foundation

enum DictationChimeRole: String, Sendable {
    case start
    case stop

    var title: String {
        switch self {
        case .start: "Start Sound"
        case .stop: "Stop Sound"
        }
    }

    var defaultSoundName: NSSound.Name {
        switch self {
        case .start: NSSound.Name("Glass")
        case .stop: NSSound.Name("Pop")
        }
    }

    var softSoundName: NSSound.Name {
        switch self {
        case .start: NSSound.Name("Ping")
        case .stop: NSSound.Name("Tink")
        }
    }

    var mechanicalSoundName: NSSound.Name {
        switch self {
        // Morse is too subtle/inconsistent as a start cue on some systems.
        case .start: NSSound.Name("Funk")
        case .stop: NSSound.Name("Pop")
        }
    }
}

enum DictationChimeAssetManager {
    static func soundsDirectory() -> URL {
        let dir = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("Yuwp/Sounds", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static func importSound(from sourceURL: URL) throws -> ImportedSoundAsset {
        let ext = sourceURL.pathExtension.isEmpty ? "aiff" : sourceURL.pathExtension
        let fileName = UUID().uuidString + "." + ext
        let destination = soundsDirectory().appendingPathComponent(fileName)

        if FileManager.default.fileExists(atPath: destination.path) {
            try? FileManager.default.removeItem(at: destination)
        }
        try FileManager.default.copyItem(at: sourceURL, to: destination)

        return ImportedSoundAsset(
            relativePath: fileName,
            displayName: sourceURL.lastPathComponent
        )
    }

    static func url(for asset: ImportedSoundAsset) -> URL {
        soundsDirectory().appendingPathComponent(asset.relativePath)
    }
}

@MainActor
final class DictationChimePlayer {
    typealias NamedSoundFactory = (NSSound.Name) -> NSSound?
    typealias CustomSoundFactory = (URL) -> NSSound?

    // Keep strong refs while playing; short-lived local NSSound instances can
    // be deallocated before audible output begins on some systems.
    private var activeSounds: [DictationChimeRole: NSSound] = [:]
    private let namedSoundFactory: NamedSoundFactory
    private let customSoundFactory: CustomSoundFactory
    private let beep: () -> Void

    init(
        namedSoundFactory: @escaping NamedSoundFactory = { NSSound(named: $0) },
        customSoundFactory: @escaping CustomSoundFactory = { NSSound(contentsOf: $0, byReference: false) },
        beep: @escaping () -> Void = { NSSound.beep() }
    ) {
        self.namedSoundFactory = namedSoundFactory
        self.customSoundFactory = customSoundFactory
        self.beep = beep
    }

    func play(_ role: DictationChimeRole, config: DictationChimeConfig? = nil) {
        let resolvedConfig = config ?? configForRole(role)
        let sound = sound(for: role, config: resolvedConfig)

        guard let sound else {
            activeSounds.removeValue(forKey: role)
            if resolvedConfig.selection != .none {
                yuwpLog("Chime \(role.rawValue): selection=\(resolvedConfig.selection.rawValue) resolved nil -> beep fallback")
                beep()
            }
            return
        }

        if let prior = activeSounds[role], prior.isPlaying {
            prior.stop()
        }
        activeSounds[role] = sound

        let started = sound.play()
        yuwpLog("Chime \(role.rawValue): selection=\(resolvedConfig.selection.rawValue) sound=\(sound.name ?? "custom") started=\(started)")

        if !started, let fallback = namedSoundFactory(role.defaultSoundName) {
            activeSounds[role] = fallback
            let fallbackStarted = fallback.play()
            yuwpLog("Chime \(role.rawValue): fallback sound=\(role.defaultSoundName) started=\(fallbackStarted)")
            if !fallbackStarted {
                beep()
            }
        }
    }

    private func configForRole(_ role: DictationChimeRole) -> DictationChimeConfig {
        switch role {
        case .start:
            Config.shared.startChime
        case .stop:
            Config.shared.stopChime
        }
    }

    private func sound(for role: DictationChimeRole, config: DictationChimeConfig) -> NSSound? {
        switch config.selection {
        case .systemDefault:
            return namedSoundFactory(role.defaultSoundName)
        case .soft:
            return namedSoundFactory(role.softSoundName)
        case .mechanical:
            return namedSoundFactory(role.mechanicalSoundName)
        case .none:
            return nil
        case .custom:
            guard let asset = config.customAsset else {
                yuwpLog("Custom \(role.rawValue) chime selected but no file is configured — falling back to default")
                return namedSoundFactory(role.defaultSoundName)
            }
            let url = DictationChimeAssetManager.url(for: asset)
            guard FileManager.default.fileExists(atPath: url.path), let sound = customSoundFactory(url) else {
                yuwpLog("Custom \(role.rawValue) chime missing or unreadable at: \(url.path) — falling back to default")
                return namedSoundFactory(role.defaultSoundName)
            }
            return sound
        }
    }
}
