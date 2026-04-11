import Testing
@testable import Yuwp

@Suite("Audio input selection")
struct AudioInputSelectionTests {
    private let devices: [AudioInputDeviceDescriptor] = [
        AudioInputDeviceDescriptor(
            audioObjectID: 1,
            uid: "default-mic",
            name: "Studio Display Microphone",
            transport: .builtIn,
            isDefault: true
        ),
        AudioInputDeviceDescriptor(
            audioObjectID: 2,
            uid: "usb-mic",
            name: "USB Audio",
            transport: .usb,
            isDefault: false
        ),
    ]

    @Test func persistenceStringRoundTrips() {
        #expect(AudioInputSelection(persistenceString: nil) == .systemDefault)
        #expect(AudioInputSelection(persistenceString: "") == .systemDefault)
        #expect(AudioInputSelection(persistenceString: "system-default") == .systemDefault)
        #expect(AudioInputSelection(persistenceString: "device:usb-mic") == .device(uid: "usb-mic"))
        #expect(AudioInputSelection.device(uid: "usb-mic").persistenceString == "device:usb-mic")
        #expect(AudioInputSelection.systemDefault.persistenceString == "system-default")
    }

    @Test func summaryUsesResolvedDeviceName() {
        #expect(AudioInputSelection.systemDefault.summary(using: devices) == "System Default (Studio Display Microphone)")
        #expect(AudioInputSelection.device(uid: "usb-mic").summary(using: devices) == "USB Audio")
        #expect(AudioInputSelection.device(uid: "missing").summary(using: devices) == "Unavailable device")
    }

    @Test func deviceDetailTextIncludesTransportWhenKnown() {
        #expect(devices[0].detailText == "Studio Display Microphone — Built-in")
        #expect(devices[1].detailText == "USB Audio — USB")
    }
}
