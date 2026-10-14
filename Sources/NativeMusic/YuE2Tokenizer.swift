import Foundation

/// YuE2's frozen text/ABC tokenizer: tiktoken-style byte-level BPE over `qwen.tiktoken`
/// (151643 ordinary tokens, Qwen2 pre-tokenizer). Specials are never produced by `encode`.
public final class YuE2Tokenizer: @unchecked Sendable {
    private static let pattern =
        #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#

    private let ranks: [[UInt8]: Int]
    private let tokens: [[UInt8]]
    private let regex: NSRegularExpression

    public init(url: URL) throws {
        let data = try Data(contentsOf: url)
        var ranks: [[UInt8]: Int] = [:]
        ranks.reserveCapacity(151_643)
        var tokens = [[UInt8]](repeating: [], count: YuE2Protocol.eod)
        for line in data.split(separator: UInt8(ascii: "\n")) where !line.isEmpty {
            let parts = line.split(separator: UInt8(ascii: " "))
            guard parts.count == 2, let bytes = Data(base64Encoded: Data(parts[0])),
                  let rank = Int(String(decoding: parts[1], as: UTF8.self)), rank >= 0, rank < tokens.count
            else { throw YuE2Error.invalidConfig("malformed qwen.tiktoken line") }
            ranks[[UInt8](bytes)] = rank
            tokens[rank] = [UInt8](bytes)
        }
        guard ranks.count == YuE2Protocol.eod else {
            throw YuE2Error.invalidConfig("expected 151643 ordinary tokens in qwen.tiktoken, found \(ranks.count)")
        }
        self.ranks = ranks
        self.tokens = tokens
        regex = try NSRegularExpression(pattern: Self.pattern)
    }

    public func encode(_ text: String) -> [Int] {
        let normalized = text.precomposedStringWithCanonicalMapping
        let ns = normalized as NSString
        var ids: [Int] = []
        for match in regex.matches(in: normalized, range: NSRange(location: 0, length: ns.length)) {
            let piece = [UInt8](ns.substring(with: match.range).utf8)
            if let rank = ranks[piece] {
                ids.append(rank)
            } else {
                ids.append(contentsOf: bytePairMerge(piece))
            }
        }
        return ids
    }

    /// Lossy for invalid UTF-8 (replacement characters), like tiktoken's `errors="replace"`.
    public func decode(_ ids: [Int]) -> String {
        var bytes: [UInt8] = []
        for id in ids where id >= 0 && id < tokens.count {
            bytes.append(contentsOf: tokens[id])
        }
        return String(decoding: bytes, as: UTF8.self)
    }

    /// Repeatedly merge the adjacent pair whose concatenation has the lowest rank.
    private func bytePairMerge(_ piece: [UInt8]) -> [Int] {
        var bounds = Array(0 ... piece.count)
        func rank(_ i: Int) -> Int? {
            guard i + 2 < bounds.count else { return nil }
            return ranks[Array(piece[bounds[i] ..< bounds[i + 2]])]
        }
        var pairRanks = (0 ..< bounds.count - 1).map { rank($0) }
        while true {
            var best: (index: Int, rank: Int)?
            for (i, r) in pairRanks.enumerated() {
                if let r, r < (best?.rank ?? Int.max) { best = (i, r) }
            }
            guard let index = best?.index else { break }
            bounds.remove(at: index + 1)
            pairRanks.remove(at: index + 1)
            pairRanks[index] = rank(index)
            if index > 0 { pairRanks[index - 1] = rank(index - 1) }
        }
        return (0 ..< bounds.count - 1).map { i in
            // Every single byte is a token, so a merged span always resolves.
            ranks[Array(piece[bounds[i] ..< bounds[i + 1]])] ?? 0
        }
    }
}
