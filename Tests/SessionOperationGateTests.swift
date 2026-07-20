import Dispatch
import Foundation
import Testing
@testable import ASRServerSupport

@Suite("Session operation gate")
struct SessionOperationGateTests {
    @Test func closePreventsLaterOperationsAndRunsOnce() {
        let gate = SessionOperationGate()

        #expect(gate.withActiveOperation { "active" } == "active")
        #expect(gate.close { "closed" } == "closed")
        #expect(gate.withActiveOperation { "late" } == nil)
        #expect(gate.close { "closed-again" } == nil)
    }

    @Test func conditionalCloseLeavesGateActiveWhenConditionChanges() {
        let gate = SessionOperationGate()

        #expect(gate.closeIf({ false }) { "closed" } == nil)
        #expect(gate.withActiveOperation { "still-active" } == "still-active")
        #expect(gate.closeIf({ true }) { "closed" } == "closed")
        #expect(gate.withActiveOperation { "late" } == nil)
    }

    @Test func activeOperationsAreSerialized() {
        let gate = SessionOperationGate()
        let events = LockedEvents()
        let firstEntered = DispatchSemaphore(value: 0)
        let allowFirstToFinish = DispatchSemaphore(value: 0)
        let secondStarted = DispatchSemaphore(value: 0)
        let completed = DispatchGroup()

        completed.enter()
        DispatchQueue.global().async {
            _ = gate.withActiveOperation {
                events.append("first-start")
                firstEntered.signal()
                allowFirstToFinish.wait()
                events.append("first-end")
            }
            completed.leave()
        }

        firstEntered.wait()
        completed.enter()
        DispatchQueue.global().async {
            secondStarted.signal()
            _ = gate.withActiveOperation {
                events.append("second")
            }
            completed.leave()
        }

        secondStarted.wait()
        allowFirstToFinish.signal()
        #expect(completed.wait(timeout: .now() + 1) == .success)
        #expect(events.values == ["first-start", "first-end", "second"])
    }

    @Test func closeWaitsForInFlightOperationBeforeClosing() {
        let gate = SessionOperationGate()
        let events = LockedEvents()
        let activeEntered = DispatchSemaphore(value: 0)
        let allowActiveToFinish = DispatchSemaphore(value: 0)
        let closeStarted = DispatchSemaphore(value: 0)
        let completed = DispatchGroup()

        completed.enter()
        DispatchQueue.global().async {
            _ = gate.withActiveOperation {
                events.append("active-start")
                activeEntered.signal()
                allowActiveToFinish.wait()
                events.append("active-end")
            }
            completed.leave()
        }

        activeEntered.wait()
        completed.enter()
        DispatchQueue.global().async {
            closeStarted.signal()
            _ = gate.close {
                events.append("close")
            }
            completed.leave()
        }

        closeStarted.wait()
        allowActiveToFinish.signal()
        #expect(completed.wait(timeout: .now() + 1) == .success)
        #expect(events.values == ["active-start", "active-end", "close"])
        #expect(gate.withActiveOperation { events.append("late") } == nil)
    }
}

private final class LockedEvents: @unchecked Sendable {
    private let lock = NSLock()
    private var storage: [String] = []

    var values: [String] {
        lock.lock()
        defer { lock.unlock() }
        return storage
    }

    func append(_ value: String) {
        lock.lock()
        storage.append(value)
        lock.unlock()
    }
}
