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
        case .start: NSSound.Name("Morse")
        case .stop: NSSound.Name("Funk")
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
    func play(_ role: DictationChimeRole, config: DictationChimeConfig? = nil) {
        let resolvedConfig = config ?? configForRole(role)
        let sound = sound(for: role, config: resolvedConfig)

        if let sound {
            if sound.isPlaying {
                sound.stop()
            }
            sound.play()
        } else if resolvedConfig.selection != .none {
            NSSound.beep()
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
            return NSSound(named: role.defaultSoundName)
        case .soft:
            return NSSound(named: role.softSoundName)
        case .mechanical:
            return NSSound(named: role.mechanicalSoundName)
        case .none:
            return nil
        case .custom:
            guard let asset = config.customAsset else {
                yuwpLog("Custom \(role.rawValue) chime selected but no file is configured — falling back to default")
                return NSSound(named: role.defaultSoundName)
            }
            let url = DictationChimeAssetManager.url(for: asset)
            guard FileManager.default.fileExists(atPath: url.path), let sound = NSSound(contentsOf: url, byReference: false) else {
                yuwpLog("Custom \(role.rawValue) chime missing or unreadable at: \(url.path) — falling back to default")
                return NSSound(named: role.defaultSoundName)
            }
            return sound
        }
    }
}
