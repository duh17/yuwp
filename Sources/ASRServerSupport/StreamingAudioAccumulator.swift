/// FIFO storage for streaming PCM samples.
///
/// Consuming a chunk advances an index instead of shifting the unread suffix on
/// every inference boundary. Storage is compacted only after enough consumed
/// samples have accumulated and the consumed prefix is at least half the array.
struct StreamingAudioAccumulator: Sendable {
    private var storage: [Float] = []
    private var readIndex = 0
    private let compactionThreshold: Int

    private(set) var compactedSampleCount = 0

    init(compactionThreshold: Int = 32_000) {
        precondition(compactionThreshold > 0)
        self.compactionThreshold = compactionThreshold
    }

    var count: Int { storage.count - readIndex }
    var isEmpty: Bool { count == 0 }

    mutating func append<C: Collection>(contentsOf samples: C) where C.Element == Float {
        storage.append(contentsOf: samples)
    }

    /// A preview must never move the canonical inference boundary.
    func peekPrefix(_ requestedCount: Int) -> [Float]? {
        precondition(requestedCount > 0)
        guard count >= requestedCount else { return nil }
        return Array(storage[readIndex..<(readIndex + requestedCount)])
    }

    mutating func takePrefix(_ requestedCount: Int) -> [Float]? {
        precondition(requestedCount > 0)
        guard count >= requestedCount else { return nil }

        let endIndex = readIndex + requestedCount
        let result = Array(storage[readIndex..<endIndex])
        readIndex = endIndex
        compactIfNeeded()
        return result
    }

    mutating func drain() -> [Float] {
        guard !isEmpty else {
            resetStorage()
            return []
        }

        let result = Array(storage[readIndex...])
        resetStorage()
        return result
    }

    private mutating func compactIfNeeded() {
        if readIndex == storage.count {
            resetStorage()
            return
        }
        guard readIndex >= compactionThreshold,
              readIndex >= storage.count / 2
        else { return }

        let unread = Array(storage[readIndex...])
        compactedSampleCount += unread.count
        storage = unread
        readIndex = 0
    }

    private mutating func resetStorage() {
        storage.removeAll(keepingCapacity: true)
        readIndex = 0
    }
}
