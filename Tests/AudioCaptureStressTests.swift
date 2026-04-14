import Foundation
import Testing
@testable import Yuwp

@Suite(
    "Audio Capture Stress",
    .tags(.integration),
    .enabled(
        if: ProcessInfo.processInfo.environment["AUDIO_STRESS_TEST"] != nil,
        "Set AUDIO_STRESS_TEST=1 (requires microphone permission)."
    )
)
struct AudioCaptureStressTests {
    private enum StressFailure: Error, CustomStringConvertible {
        case noInputDevices
        case startTimedOut(iteration: Int, selection: String, timeoutMs: Int)
        case stopTimedOut(iteration: Int, selection: String, timeoutMs: Int)
        case startReturnedFalse(iteration: Int, selection: String)

        var description: String {
            switch self {
            case .noInputDevices:
                "No audio input devices were discovered"
            case .startTimedOut(let iteration, let selection, let timeoutMs):
                "start() timed out (iteration=\(iteration), selection=\(selection), timeoutMs=\(timeoutMs))"
            case .stopTimedOut(let iteration, let selection, let timeoutMs):
                "stop() timed out (iteration=\(iteration), selection=\(selection), timeoutMs=\(timeoutMs))"
            case .startReturnedFalse(let iteration, let selection):
                "start() returned false (iteration=\(iteration), selection=\(selection))"
            }
        }
    }

    private final class LockedBox<T>: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: T?

        func set(_ value: T) {
            lock.lock()
            storage = value
            lock.unlock()
        }

        func get() -> T? {
            lock.lock()
            defer { lock.unlock() }
            return storage
        }
    }

    private let catalog = SystemAudioInputCatalog()
    private let baseIterations: Int
    private let holdSeconds: TimeInterval
    private let settleSeconds: TimeInterval
    private let startTimeoutMs: Int
    private let stopTimeoutMs: Int

    init() {
        let env = ProcessInfo.processInfo.environment

        baseIterations = max(1, Self.intEnv(env, "AUDIO_STRESS_ITERATIONS", default: 60))
        holdSeconds = TimeInterval(max(20, Self.intEnv(env, "AUDIO_STRESS_HOLD_MS", default: 250))) / 1000.0
        settleSeconds = TimeInterval(max(0, Self.intEnv(env, "AUDIO_STRESS_SETTLE_MS", default: 50))) / 1000.0
        startTimeoutMs = max(250, Self.intEnv(env, "AUDIO_STRESS_START_TIMEOUT_MS", default: 4000))
        stopTimeoutMs = max(250, Self.intEnv(env, "AUDIO_STRESS_STOP_TIMEOUT_MS", default: 4000))
    }

    @Test func rapidStartStopOnSystemDefault() throws {
        try runLoop(
            selections: [.systemDefault],
            iterations: baseIterations
        )
    }

    @Test func startStopWhileSwitchingInputSelections() throws {
        let devices = catalog.availableInputDevices()
        guard !devices.isEmpty else {
            throw StressFailure.noInputDevices
        }

        var selections: [AudioInputSelection] = [.systemDefault]
        selections.append(contentsOf: devices.map(\.selection))

        // Ensure each discovered selection is exercised multiple times.
        let iterations = max(baseIterations, selections.count * 20)
        try runLoop(
            selections: selections,
            iterations: iterations
        )
    }

    private func runLoop(
        selections: [AudioInputSelection],
        iterations: Int
    ) throws {
        guard !selections.isEmpty else { throw StressFailure.noInputDevices }

        for iteration in 0..<iterations {
            let selection = selections[iteration % selections.count]
            let selectionLabel = selection.persistenceString
            let capture = AudioCapture(inputCatalog: catalog)
            capture.inputSelection = selection

            let started = try startWithTimeout(
                capture,
                iteration: iteration,
                selectionLabel: selectionLabel
            )
            guard started else {
                throw StressFailure.startReturnedFalse(
                    iteration: iteration,
                    selection: selectionLabel
                )
            }

            Thread.sleep(forTimeInterval: holdSeconds)
            try stopWithTimeout(
                capture,
                iteration: iteration,
                selectionLabel: selectionLabel
            )
            if settleSeconds > 0 {
                Thread.sleep(forTimeInterval: settleSeconds)
            }
        }
    }

    private func startWithTimeout(
        _ capture: AudioCapture,
        iteration: Int,
        selectionLabel: String
    ) throws -> Bool {
        let result = LockedBox<Bool>()
        let sema = DispatchSemaphore(value: 0)

        DispatchQueue.global(qos: .userInitiated).async {
            let started = capture.start { _ in }
            result.set(started)
            sema.signal()
        }

        let timeout = DispatchTime.now() + .milliseconds(startTimeoutMs)
        guard sema.wait(timeout: timeout) == .success else {
            throw StressFailure.startTimedOut(
                iteration: iteration,
                selection: selectionLabel,
                timeoutMs: startTimeoutMs
            )
        }

        return result.get() ?? false
    }

    private func stopWithTimeout(
        _ capture: AudioCapture,
        iteration: Int,
        selectionLabel: String
    ) throws {
        let sema = DispatchSemaphore(value: 0)

        DispatchQueue.global(qos: .userInitiated).async {
            _ = capture.stop()
            sema.signal()
        }

        let timeout = DispatchTime.now() + .milliseconds(stopTimeoutMs)
        guard sema.wait(timeout: timeout) == .success else {
            throw StressFailure.stopTimedOut(
                iteration: iteration,
                selection: selectionLabel,
                timeoutMs: stopTimeoutMs
            )
        }
    }

    private static func intEnv(
        _ env: [String: String],
        _ key: String,
        default defaultValue: Int
    ) -> Int {
        guard let raw = env[key], let value = Int(raw) else { return defaultValue }
        return value
    }
}
