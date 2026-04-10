import Foundation

/// Script-aware rendering for aligned output tokens.
///
/// This is intentionally a display-layer joiner, not semantic normalization.
/// It decides whether adjacent token fragments should be concatenated or
/// separated by a space based on script and punctuation boundaries.
enum ScriptClassifier {
    static func isCJK(_ scalar: Unicode.Scalar) -> Bool {
        let c = scalar.value
        return (0x4E00 ... 0x9FFF).contains(c)
            || (0x3400 ... 0x4DBF).contains(c)
            || (0x20000 ... 0x2A6DF).contains(c)
            || (0x2A700 ... 0x2B73F).contains(c)
            || (0x2B740 ... 0x2B81F).contains(c)
            || (0x2B820 ... 0x2CEAF).contains(c)
            || (0xF900 ... 0xFAFF).contains(c)
    }

    static func isCompactScript(_ scalar: Unicode.Scalar) -> Bool {
        if isCJK(scalar) { return true }
        let c = scalar.value
        return (0x3040 ... 0x309F).contains(c)   // Hiragana
            || (0x30A0 ... 0x30FF).contains(c)   // Katakana
            || (0x31F0 ... 0x31FF).contains(c)   // Katakana Phonetic Extensions
            || (0xFF66 ... 0xFF9D).contains(c)   // Half-width Katakana
    }

    static func isCompactScript(_ character: Character) -> Bool {
        character.unicodeScalars.contains(where: isCompactScript)
    }
}

public enum AlignedTextRenderer {
    public static func render(tokens: [String]) -> String {
        join(tokens)
    }

    public static func render(segments: [String]) -> String {
        join(segments)
    }

    private static func join<S: Sequence>(_ parts: S) -> String where S.Element == String {
        var result = ""
        var previousPart: String?

        for rawPart in parts {
            let part = rawPart.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !part.isEmpty else { continue }
            guard let lastPart = previousPart else {
                result = part
                previousPart = part
                continue
            }

            if shouldInsertSpace(between: lastPart, and: part) {
                result.append(" ")
            }
            result.append(part)
            previousPart = part
        }

        return result
    }

    private static func shouldInsertSpace(between lhsPart: String, and rhsPart: String) -> Bool {
        guard let lhs = lhsPart.last, let rhs = rhsPart.first else { return false }

        if isSingleASCIIAcronymFragment(lhsPart), isSingleASCIIAcronymFragment(rhsPart) {
            return false
        }
        if isOpeningPunctuation(lhs) || isClosingPunctuation(rhs) {
            return false
        }
        if ScriptClassifier.isCompactScript(lhs) || ScriptClassifier.isCompactScript(rhs) {
            return false
        }
        if lhs.isWhitespace || rhs.isWhitespace {
            return false
        }
        if lhs.isLetter || lhs.isNumber || rhs.isLetter || rhs.isNumber {
            return true
        }
        return false
    }

    private static func isSingleASCIIAcronymFragment(_ text: String) -> Bool {
        guard text.count == 1, let scalar = text.unicodeScalars.first else { return false }
        return (65 ... 90).contains(scalar.value) || (48 ... 57).contains(scalar.value)
    }

    private static func isOpeningPunctuation(_ ch: Character) -> Bool {
        "([{".contains(ch) || "‘“《〈【『「（".contains(ch)
    }

    private static func isClosingPunctuation(_ ch: Character) -> Bool {
        ",.!?;:)]}".contains(ch) || "，。！？、；：）》〉】』」）".contains(ch)
    }
}
