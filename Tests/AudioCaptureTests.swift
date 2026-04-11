import Testing
@testable import Yuwp

@Suite("AudioCapture")
struct AudioCaptureTests {
    private struct ReferenceWarningTracker {
        var speechPauseTimeout: Double?
        var consecutiveSilentBuffers = 0
        var silenceWarningFired = false
        var hasDetectedSpeech = false
        var consecutivePauseBuffers = 0
        var speechPauseFired = false

        mutating func reset() {
            consecutiveSilentBuffers = 0
            silenceWarningFired = false
            hasDetectedSpeech = false
            consecutivePauseBuffers = 0
            speechPauseFired = false
        }

        mutating func ingest(rms: Float) -> [AudioCaptureWarning] {
            if rms >= AudioCapture.WarningTracker.speechDetectRmsThreshold {
                hasDetectedSpeech = true
                consecutiveSilentBuffers = 0
                consecutivePauseBuffers = 0
                return []
            }

            var warnings: [AudioCaptureWarning] = []

            if !hasDetectedSpeech {
                if rms < AudioCapture.WarningTracker.deadMicRmsThreshold {
                    consecutiveSilentBuffers += 1
                    let threshold = AudioCapture.WarningTracker.bufferThreshold(
                        for: AudioCapture.WarningTracker.silenceWarningSeconds
                    )
                    if consecutiveSilentBuffers >= threshold, !silenceWarningFired {
                        silenceWarningFired = true
                        warnings.append(.silentInput(seconds: AudioCapture.WarningTracker.silenceWarningSeconds))
                    }
                } else {
                    consecutiveSilentBuffers = 0
                }
            } else {
                consecutiveSilentBuffers = 0
            }

            if let timeout = speechPauseTimeout, !speechPauseFired, hasDetectedSpeech {
                if rms < AudioCapture.WarningTracker.speechPauseRmsThreshold {
                    consecutivePauseBuffers += 1
                    let threshold = AudioCapture.WarningTracker.bufferThreshold(for: timeout)
                    if consecutivePauseBuffers >= threshold {
                        speechPauseFired = true
                        warnings.append(.speechPause(seconds: timeout))
                    }
                } else {
                    consecutivePauseBuffers = 0
                }
            }

            return warnings
        }
    }

    private struct SeededGenerator {
        private var state: UInt64

        init(seed: Int) {
            self.state = UInt64(truncatingIfNeeded: seed) &+ 0x9E3779B97F4A7C15
        }

        mutating func next() -> UInt64 {
            state &+= 0x9E3779B97F4A7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
            z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
            return z ^ (z >> 31)
        }

        mutating func nextInt(upperBound: Int) -> Int {
            Int(next() % UInt64(upperBound))
        }

        mutating func nextFloat(in range: ClosedRange<Float>) -> Float {
            let unit = Float(next() & 0xFFFF) / Float(0xFFFF)
            return range.lowerBound + (range.upperBound - range.lowerBound) * unit
        }
    }

    private var silenceThresholdBuffers: Int {
        AudioCapture.WarningTracker.bufferThreshold(for: AudioCapture.WarningTracker.silenceWarningSeconds)
    }

    private func collectWarnings(
        tracker: inout AudioCapture.WarningTracker,
        rms: Float,
        count: Int
    ) -> [AudioCaptureWarning] {
        var warnings: [AudioCaptureWarning] = []
        for _ in 0..<count {
            warnings.append(contentsOf: tracker.ingest(rms: rms))
        }
        return warnings
    }

    private func nextRms(_ generator: inout SeededGenerator) -> Float {
        switch generator.nextInt(upperBound: 9) {
        case 0:
            return 0
        case 1:
            return AudioCapture.WarningTracker.deadMicRmsThreshold * 0.5
        case 2:
            return AudioCapture.WarningTracker.deadMicRmsThreshold * 1.5
        case 3:
            return AudioCapture.WarningTracker.speechPauseRmsThreshold * 0.5
        case 4:
            return AudioCapture.WarningTracker.speechPauseRmsThreshold * 1.2
        case 5:
            return (AudioCapture.WarningTracker.speechPauseRmsThreshold + AudioCapture.WarningTracker.speechDetectRmsThreshold) / 2
        case 6:
            return AudioCapture.WarningTracker.speechDetectRmsThreshold
        case 7:
            return AudioCapture.WarningTracker.speechDetectRmsThreshold * 1.5
        default:
            return generator.nextFloat(in: 0 ... 0.03)
        }
    }

    private func nextTimeout(_ generator: inout SeededGenerator) -> Double? {
        let options: [Double?] = [nil, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
        return options[generator.nextInt(upperBound: options.count)]
    }

    @Test func silentInputWarningFiresOnceAfterThreshold() {
        var tracker = AudioCapture.WarningTracker()

        var warnings = collectWarnings(tracker: &tracker, rms: 0, count: silenceThresholdBuffers)
        #expect(warnings == [.silentInput(seconds: AudioCapture.WarningTracker.silenceWarningSeconds)])
        #expect(tracker.silenceWarningFired)

        warnings.append(contentsOf: collectWarnings(tracker: &tracker, rms: 0, count: silenceThresholdBuffers))
        #expect(warnings.count == 1)
    }

    @Test func audibleBufferResetsSilentCounterBeforeWarning() {
        var tracker = AudioCapture.WarningTracker()

        _ = collectWarnings(tracker: &tracker, rms: 0, count: silenceThresholdBuffers - 1)
        #expect(tracker.consecutiveSilentBuffers == silenceThresholdBuffers - 1)
        #expect(!tracker.silenceWarningFired)

        _ = tracker.ingest(rms: AudioCapture.WarningTracker.deadMicRmsThreshold * 4)
        #expect(tracker.consecutiveSilentBuffers == 0)
        #expect(!tracker.silenceWarningFired)

        let warnings = collectWarnings(tracker: &tracker, rms: 0, count: silenceThresholdBuffers)
        #expect(warnings == [.silentInput(seconds: AudioCapture.WarningTracker.silenceWarningSeconds)])
    }

    @Test func deadMicWarningDoesNotFireAfterSpeechWasDetected() {
        var tracker = AudioCapture.WarningTracker()
        tracker.speechPauseTimeout = 2.0

        _ = tracker.ingest(rms: AudioCapture.WarningTracker.speechDetectRmsThreshold)
        #expect(tracker.hasDetectedSpeech)

        let warnings = collectWarnings(tracker: &tracker, rms: 0, count: silenceThresholdBuffers + 5)
        #expect(!warnings.contains(.silentInput(seconds: AudioCapture.WarningTracker.silenceWarningSeconds)))
        #expect(warnings.contains(.speechPause(seconds: 2.0)))
    }

    @Test func speechPauseRequiresSpeechAndFiresOnce() {
        var tracker = AudioCapture.WarningTracker()
        tracker.speechPauseTimeout = 2.0
        let pauseThresholdBuffers = AudioCapture.WarningTracker.bufferThreshold(for: 2.0)

        let beforeSpeech = collectWarnings(tracker: &tracker, rms: 0.001, count: pauseThresholdBuffers + 5)
        #expect(beforeSpeech.isEmpty)
        #expect(!tracker.hasDetectedSpeech)

        _ = tracker.ingest(rms: AudioCapture.WarningTracker.speechDetectRmsThreshold)
        #expect(tracker.hasDetectedSpeech)

        let almostPaused = collectWarnings(tracker: &tracker, rms: 0.001, count: pauseThresholdBuffers - 1)
        #expect(almostPaused.isEmpty)
        #expect(!tracker.speechPauseFired)

        let pauseWarning = tracker.ingest(rms: 0.001)
        #expect(pauseWarning == [.speechPause(seconds: 2.0)])
        #expect(tracker.speechPauseFired)

        let laterWarnings = collectWarnings(tracker: &tracker, rms: 0.001, count: pauseThresholdBuffers)
        #expect(laterWarnings.isEmpty)
    }

    @Test func speechPauseUsesFullFractionalTimeout() {
        var tracker = AudioCapture.WarningTracker()
        tracker.speechPauseTimeout = 0.25
        let pauseThresholdBuffers = AudioCapture.WarningTracker.bufferThreshold(for: 0.25)

        _ = tracker.ingest(rms: AudioCapture.WarningTracker.speechDetectRmsThreshold)

        let earlyWarnings = collectWarnings(tracker: &tracker, rms: 0.001, count: pauseThresholdBuffers - 1)
        #expect(earlyWarnings.isEmpty)
        #expect(!tracker.speechPauseFired)

        let finalWarning = tracker.ingest(rms: 0.001)
        #expect(finalWarning == [.speechPause(seconds: 0.25)])
    }

    @Test func postSpeechNoiseResetsPauseCountdown() {
        var tracker = AudioCapture.WarningTracker()
        tracker.speechPauseTimeout = 0.5
        let pauseThresholdBuffers = AudioCapture.WarningTracker.bufferThreshold(for: 0.5)

        _ = tracker.ingest(rms: AudioCapture.WarningTracker.speechDetectRmsThreshold)
        _ = collectWarnings(tracker: &tracker, rms: 0.001, count: pauseThresholdBuffers - 1)
        #expect(tracker.consecutivePauseBuffers == pauseThresholdBuffers - 1)

        _ = tracker.ingest(rms: AudioCapture.WarningTracker.speechPauseRmsThreshold * 1.2)
        #expect(tracker.consecutivePauseBuffers == 0)

        let warnings = collectWarnings(tracker: &tracker, rms: 0.001, count: pauseThresholdBuffers)
        #expect(warnings == [.speechPause(seconds: 0.5)])
    }

    @Test func routeChangeGracePeriodSuppressesStartupNoise() {
        #expect(AudioCapture.WarningTracker.warningForRouteChange(elapsedSinceStart: 0.2) == nil)
        #expect(
            AudioCapture.WarningTracker.warningForRouteChange(
                elapsedSinceStart: AudioCapture.WarningTracker.routeChangeGracePeriod
            ) == .routeChanged
        )
        #expect(AudioCapture.WarningTracker.warningForRouteChange(elapsedSinceStart: 3.0) == .routeChanged)
    }

    @Test func resetClearsWarningStateForNextSession() {
        var tracker = AudioCapture.WarningTracker()
        tracker.speechPauseTimeout = 1.0

        _ = collectWarnings(tracker: &tracker, rms: 0, count: silenceThresholdBuffers)
        _ = tracker.ingest(rms: AudioCapture.WarningTracker.speechDetectRmsThreshold)
        _ = collectWarnings(
            tracker: &tracker,
            rms: 0.001,
            count: AudioCapture.WarningTracker.bufferThreshold(for: 1.0)
        )

        #expect(tracker.silenceWarningFired)
        #expect(tracker.speechPauseFired)
        #expect(tracker.hasDetectedSpeech)

        tracker.reset()

        #expect(!tracker.silenceWarningFired)
        #expect(!tracker.speechPauseFired)
        #expect(!tracker.hasDetectedSpeech)
        #expect(tracker.consecutiveSilentBuffers == 0)
        #expect(tracker.consecutivePauseBuffers == 0)
        #expect(tracker.speechPauseTimeout == 1.0)

        let warnings = collectWarnings(tracker: &tracker, rms: 0, count: silenceThresholdBuffers)
        #expect(warnings == [.silentInput(seconds: AudioCapture.WarningTracker.silenceWarningSeconds)])
    }

    @Test(arguments: Array(0..<24))
    func warningTrackerFuzzMatchesReferenceModel(seed: Int) {
        var actual = AudioCapture.WarningTracker()
        var expected = ReferenceWarningTracker()
        var generator = SeededGenerator(seed: seed)

        for step in 0..<400 {
            let action = generator.nextInt(upperBound: 10)
            let actualWarnings: [AudioCaptureWarning]
            let expectedWarnings: [AudioCaptureWarning]

            switch action {
            case 0:
                let timeout = nextTimeout(&generator)
                actual.speechPauseTimeout = timeout
                expected.speechPauseTimeout = timeout
                actualWarnings = []
                expectedWarnings = []
            case 1:
                actual.reset()
                expected.reset()
                actualWarnings = []
                expectedWarnings = []
            default:
                let rms = nextRms(&generator)
                actualWarnings = actual.ingest(rms: rms)
                expectedWarnings = expected.ingest(rms: rms)
            }

            #expect(
                actualWarnings == expectedWarnings,
                Comment(rawValue: "seed=\(seed) step=\(step) warnings actual=\(actualWarnings) expected=\(expectedWarnings)")
            )
            #expect(
                actual.speechPauseTimeout == expected.speechPauseTimeout,
                Comment(rawValue: "seed=\(seed) step=\(step) timeout actual=\(String(describing: actual.speechPauseTimeout)) expected=\(String(describing: expected.speechPauseTimeout))")
            )
            #expect(
                actual.consecutiveSilentBuffers == expected.consecutiveSilentBuffers,
                Comment(rawValue: "seed=\(seed) step=\(step) consecutiveSilentBuffers actual=\(actual.consecutiveSilentBuffers) expected=\(expected.consecutiveSilentBuffers)")
            )
            #expect(
                actual.silenceWarningFired == expected.silenceWarningFired,
                Comment(rawValue: "seed=\(seed) step=\(step) silenceWarningFired mismatch")
            )
            #expect(
                actual.hasDetectedSpeech == expected.hasDetectedSpeech,
                Comment(rawValue: "seed=\(seed) step=\(step) hasDetectedSpeech mismatch")
            )
            #expect(
                actual.consecutivePauseBuffers == expected.consecutivePauseBuffers,
                Comment(rawValue: "seed=\(seed) step=\(step) consecutivePauseBuffers actual=\(actual.consecutivePauseBuffers) expected=\(expected.consecutivePauseBuffers)")
            )
            #expect(
                actual.speechPauseFired == expected.speechPauseFired,
                Comment(rawValue: "seed=\(seed) step=\(step) speechPauseFired mismatch")
            )
        }
    }

    @Test func stopReturnsNilWhenNotRunning() {
        let capture = AudioCapture()
        #expect(capture.stop() == nil)
    }
}
