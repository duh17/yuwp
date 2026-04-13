import AudioToolbox
import CoreAudio
import Foundation

enum AudioInputSelection: Sendable, Equatable {
    case systemDefault
    case device(uid: String)

    init(persistenceString: String?) {
        guard let persistenceString, !persistenceString.isEmpty else {
            self = .systemDefault
            return
        }
        if persistenceString == "system-default" {
            self = .systemDefault
        } else if let uid = persistenceString.split(separator: ":", maxSplits: 1).dropFirst().first,
                  persistenceString.hasPrefix("device:") {
            self = .device(uid: String(uid))
        } else {
            self = .systemDefault
        }
    }

    var persistenceString: String {
        switch self {
        case .systemDefault:
            "system-default"
        case .device(let uid):
            "device:\(uid)"
        }
    }

    func summary(using devices: [AudioInputDeviceDescriptor]) -> String {
        switch self {
        case .systemDefault:
            if let defaultDevice = devices.first(where: \.isDefault) {
                return "System Default (\(defaultDevice.name))"
            }
            return "System Default"
        case .device(let uid):
            if let device = devices.first(where: { $0.uid == uid }) {
                return device.name
            }
            return "Unavailable device"
        }
    }
}

enum AudioInputTransport: Sendable, Equatable {
    case builtIn
    case aggregate
    case virtual
    case pci
    case usb
    case fireWire
    case bluetooth
    case bluetoothLE
    case hdmi
    case displayPort
    case airPlay
    case avb
    case thunderbolt
    case continuityWired
    case continuityWireless
    case continuity
    case unknown

    var description: String {
        switch self {
        case .builtIn: "Built-in"
        case .aggregate: "Aggregate"
        case .virtual: "Virtual"
        case .pci: "PCI"
        case .usb: "USB"
        case .fireWire: "FireWire"
        case .bluetooth: "Bluetooth"
        case .bluetoothLE: "Bluetooth LE"
        case .hdmi: "HDMI"
        case .displayPort: "DisplayPort"
        case .airPlay: "AirPlay"
        case .avb: "AVB"
        case .thunderbolt: "Thunderbolt"
        case .continuityWired: "Continuity (Wired)"
        case .continuityWireless: "Continuity (Wireless)"
        case .continuity: "Continuity"
        case .unknown: "Unknown"
        }
    }

    init(coreAudioTransportType: UInt32) {
        switch coreAudioTransportType {
        case kAudioDeviceTransportTypeBuiltIn:
            self = .builtIn
        case kAudioDeviceTransportTypeAggregate, kAudioDeviceTransportTypeAutoAggregate:
            self = .aggregate
        case kAudioDeviceTransportTypeVirtual:
            self = .virtual
        case kAudioDeviceTransportTypePCI:
            self = .pci
        case kAudioDeviceTransportTypeUSB:
            self = .usb
        case kAudioDeviceTransportTypeFireWire:
            self = .fireWire
        case kAudioDeviceTransportTypeBluetooth:
            self = .bluetooth
        case kAudioDeviceTransportTypeBluetoothLE:
            self = .bluetoothLE
        case kAudioDeviceTransportTypeHDMI:
            self = .hdmi
        case kAudioDeviceTransportTypeDisplayPort:
            self = .displayPort
        case kAudioDeviceTransportTypeAirPlay:
            self = .airPlay
        case kAudioDeviceTransportTypeAVB:
            self = .avb
        case kAudioDeviceTransportTypeThunderbolt:
            self = .thunderbolt
        case kAudioDeviceTransportTypeContinuityCaptureWired:
            self = .continuityWired
        case kAudioDeviceTransportTypeContinuityCaptureWireless:
            self = .continuityWireless
        case 0x63636170: // 'ccap' — older Continuity Capture transport
            self = .continuity
        default:
            self = .unknown
        }
    }
}

struct AudioInputDeviceDescriptor: Sendable, Equatable {
    let audioObjectID: AudioObjectID
    let uid: String
    let name: String
    let transport: AudioInputTransport
    let isDefault: Bool

    var selection: AudioInputSelection { .device(uid: uid) }

    var menuTitle: String {
        isDefault ? "\(name) (Default)" : name
    }

    var detailText: String {
        if transport == .unknown {
            return name
        }
        return "\(name) — \(transport.description)"
    }
}

protocol AudioInputCatalog: Sendable {
    func availableInputDevices() -> [AudioInputDeviceDescriptor]
    func resolve(_ selection: AudioInputSelection) -> AudioInputDeviceDescriptor?
}

struct SystemAudioInputCatalog: AudioInputCatalog {
    func availableInputDevices() -> [AudioInputDeviceDescriptor] {
        let defaultDeviceID = defaultInputDeviceID()

        return allDeviceIDs().compactMap { deviceID in
            guard isDeviceAlive(deviceID), hasInputChannels(deviceID) else { return nil }
            let isDefault = deviceID == defaultDeviceID
            guard let name = cfStringProperty(deviceID, selector: kAudioObjectPropertyName), !name.isEmpty,
                  let uid = cfStringProperty(deviceID, selector: kAudioDevicePropertyDeviceUID), !uid.isEmpty else {
                return nil
            }
            if !isDefault,
               (name.hasPrefix("CADefaultDeviceAggregate") || uid.hasPrefix("CADefaultDeviceAggregate")) {
                return nil
            }

            let transport = AudioInputTransport(
                coreAudioTransportType: uint32Property(deviceID, selector: kAudioDevicePropertyTransportType) ?? 0
            )
            return AudioInputDeviceDescriptor(
                audioObjectID: deviceID,
                uid: uid,
                name: name,
                transport: transport,
                isDefault: isDefault
            )
        }
        .sorted { lhs, rhs in
            if lhs.isDefault != rhs.isDefault { return lhs.isDefault && !rhs.isDefault }
            return lhs.name.localizedCaseInsensitiveCompare(rhs.name) == .orderedAscending
        }
    }

    func resolve(_ selection: AudioInputSelection) -> AudioInputDeviceDescriptor? {
        let devices = availableInputDevices()
        switch selection {
        case .systemDefault:
            return devices.first(where: \.isDefault) ?? devices.first
        case .device(let uid):
            return devices.first(where: { $0.uid == uid })
        }
    }

    private func allDeviceIDs() -> [AudioObjectID] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDevices,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size) == noErr else {
            return []
        }

        var devices = Array(repeating: AudioObjectID(0), count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &devices) == noErr else {
            return []
        }
        return devices
    }

    private func defaultInputDeviceID() -> AudioObjectID? {
        objectIDProperty(
            AudioObjectID(kAudioObjectSystemObject),
            selector: kAudioHardwarePropertyDefaultInputDevice,
            scope: kAudioObjectPropertyScopeGlobal
        )
    }

    private func isDeviceAlive(_ deviceID: AudioObjectID) -> Bool {
        uint32Property(deviceID, selector: kAudioDevicePropertyDeviceIsAlive) != 0
    }

    private func hasInputChannels(_ deviceID: AudioObjectID) -> Bool {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyStreamConfiguration,
            mScope: kAudioDevicePropertyScopeInput,
            mElement: kAudioObjectPropertyElementMain
        )
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(deviceID, &address, 0, nil, &size) == noErr else {
            return false
        }

        let raw = UnsafeMutableRawPointer.allocate(byteCount: Int(size), alignment: MemoryLayout<AudioBufferList>.alignment)
        defer { raw.deallocate() }

        guard AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, raw) == noErr else {
            return false
        }

        let audioBufferList = raw.bindMemory(to: AudioBufferList.self, capacity: 1)
        let buffers = UnsafeMutableAudioBufferListPointer(audioBufferList)
        return buffers.contains(where: { $0.mNumberChannels > 0 })
    }

    private func objectIDProperty(
        _ objectID: AudioObjectID,
        selector: AudioObjectPropertySelector,
        scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal,
        element: AudioObjectPropertyElement = kAudioObjectPropertyElementMain
    ) -> AudioObjectID? {
        var address = AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: element)
        var value = AudioObjectID(0)
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, &value) == noErr else {
            return nil
        }
        return value
    }

    private func uint32Property(
        _ objectID: AudioObjectID,
        selector: AudioObjectPropertySelector,
        scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal,
        element: AudioObjectPropertyElement = kAudioObjectPropertyElementMain
    ) -> UInt32? {
        var address = AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: element)
        var value: UInt32 = 0
        var size = UInt32(MemoryLayout<UInt32>.size)
        guard AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, &value) == noErr else {
            return nil
        }
        return value
    }

    private func cfStringProperty(
        _ objectID: AudioObjectID,
        selector: AudioObjectPropertySelector,
        scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal,
        element: AudioObjectPropertyElement = kAudioObjectPropertyElementMain
    ) -> String? {
        var address = AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: element)
        var value: CFString = "" as CFString
        var size = UInt32(MemoryLayout<CFString?>.size)
        let status = withUnsafeMutablePointer(to: &value) { ptr in
            AudioObjectGetPropertyData(objectID, &address, 0, nil, &size, ptr)
        }
        guard status == noErr else { return nil }
        return value as String
    }
}
