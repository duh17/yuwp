// Reuse an existing transcript as a proposal, never as conditioning evidence.
// Every emitted token is selected by the same greedy target model.

struct TranscriptDraft {
    private let tokens: [Int]
    private let positions: [Int: [Int]]

    init(tokens: [Int]) {
        self.tokens = tokens
        var positions: [Int: [Int]] = [:]
        for (index, token) in tokens.enumerated() {
            positions[token, default: []].append(index)
        }
        self.positions = positions
    }

    func continuation(after verified: [Int], limit: Int) -> [Int] {
        guard limit > 0, let last = verified.last else { return [] }
        if last == Qwen3ASRTokenizer.asrText {
            return Array(tokens.prefix(limit))
        }
        guard let candidates = positions[last] else { return [] }
        var bestEnd: Int?
        var bestLength = 0
        for end in candidates {
            var length = 1
            let maximum = min(4, min(verified.count, end + 1))
            while length < maximum,
                  verified[verified.count - 1 - length] == tokens[end - length] {
                length += 1
            }
            if length > bestLength {
                bestLength = length
                bestEnd = end
            }
        }
        guard let end = bestEnd else { return [] }
        let start = end + 1
        return Array(tokens[start..<min(tokens.count, start + limit)])
    }

    /// `verify` processes all inputs with a causal mask and returns one greedy
    /// next-token choice per input. `rewind` removes rejected draft KV positions.
    /// The current token is already verified, but is not yet in the KV cache.
    static func greedyDecode(
        firstToken: Int,
        maxTokens: Int,
        eosTokens: Set<Int>,
        draft: TranscriptDraft,
        verify: ([Int]) -> [Int],
        rewind: (Int) -> Void
    ) -> [Int] {
        guard maxTokens > 0 else { return [] }
        var generated: [Int] = []
        var lastToken = -1
        var repetitionCount = 0

        func accept(_ token: Int) -> Bool {
            guard !eosTokens.contains(token) else { return false }
            if token == lastToken {
                repetitionCount += 1
                if repetitionCount >= 10 { return false }
            } else {
                repetitionCount = 0
                lastToken = token
            }
            generated.append(token)
            return generated.count < maxTokens
        }

        var current = firstToken
        while accept(current) {
            // At most eight input positions: one verified token plus seven drafts.
            let proposed = draft.continuation(
                after: generated, limit: min(7, maxTokens - generated.count)
            )
            let choices = verify([current] + proposed)
            precondition(choices.count == proposed.count + 1)
            var matched = 0
            while matched < proposed.count, choices[matched] == proposed[matched] {
                guard accept(proposed[matched]) else { return generated }
                matched += 1
            }
            // Preserve only current + matching drafts. The mismatch/bonus token
            // is verified, but must be supplied as the next forward pass's input.
            rewind(proposed.count - matched)
            current = choices[matched]
        }
        return generated
    }
}
