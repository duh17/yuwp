import Foundation

/// Serializes work for one streaming session and provides a one-way close.
///
/// `close` waits for an in-flight operation, marks the gate closed, and runs its
/// body while still holding exclusive access. Later operations fail instead of
/// touching finalized session state.
final class SessionOperationGate: @unchecked Sendable {
    private let lock = NSLock()
    private var isClosed = false

    func withActiveOperation<Result>(_ body: () throws -> Result) rethrows -> Result? {
        lock.lock()
        guard !isClosed else {
            lock.unlock()
            return nil
        }
        defer { lock.unlock() }
        return try body()
    }

    func close<Result>(_ body: () throws -> Result) rethrows -> Result? {
        try closeIf({ true }, body)
    }

    func closeIf<Result>(
        _ shouldClose: () -> Bool,
        _ body: () throws -> Result
    ) rethrows -> Result? {
        lock.lock()
        guard !isClosed, shouldClose() else {
            lock.unlock()
            return nil
        }
        isClosed = true
        defer { lock.unlock() }
        return try body()
    }
}
