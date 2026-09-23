import Foundation
import Testing
@testable import Yuwp

@Suite("OppiDictationDictionary")
struct OppiDictationDictionaryTests {
    @Test func readsGlobalPhrasesAndClipsCaps() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("yuwp-oppi-dict-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let url = dir.appendingPathComponent("dictation-dictionary.json")
        let payload = """
        {
          "version": 1,
          "global": {
            "revision": 1,
            "phrases": ["Yuwp", "Oppi", "Yuwp", "  ", "Astra"]
          },
          "workspaces": {}
        }
        """
        try Data(payload.utf8).write(to: url)

        let phrases = OppiDictationDictionary.phrases(from: url)
        #expect(phrases == ["Yuwp", "Oppi", "Astra"])
    }

    @Test func missingFileYieldsNoHints() {
        let url = URL(fileURLWithPath: "/tmp/yuwp-missing-dictation-dictionary.json")
        #expect(OppiDictationDictionary.phrases(from: url).isEmpty)
    }

    @Test func globalPhrasesAlwaysIncludeBuiltInYuwpEvenWithoutOppiFile() {
        let phrases = OppiDictationDictionary.globalPhrases(
            environment: ["OPPI_DATA_DIR": "/tmp/yuwp-missing-oppi-data"]
        )
        #expect(phrases.contains("Yuwp"))
        #expect(phrases.contains("Oppi"))
        #expect(phrases.firstIndex(of: "Yuwp") == 0)
    }

    @Test func respectsOppiDataDirOverride() {
        let url = OppiDictationDictionary.dictionaryURL(
            environment: ["OPPI_DATA_DIR": "/tmp/oppi-data"]
        )
        #expect(url.path.hasSuffix("/tmp/oppi-data/settings/dictation-dictionary.json"))
    }
}
