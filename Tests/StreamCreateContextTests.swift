import ASRIPC
import Foundation
import Testing
@testable import ASRServerSupport
import NativeASR

@Suite("Stream create context")
struct StreamCreateContextTests {
    @Test func emptyBodyKeepsExistingClientsValidAndDoesNotApplyContext() throws {
        let manager = ContextFakeManager()
        let response = routeRequest(
            HTTPRequest(method: "POST", path: "/v1/audio/transcriptions/stream", headers: [:], body: Data()),
            context: makeContext(manager)
        )
        let json = try responseJSON(response)

        #expect(response.status == 200)
        #expect(json["session_id"] as? String == manager.createdSessionID)
        #expect(json["context_applied"] as? Bool == false)
        #expect(manager.lastCreatedLanguage == nil)
        #expect(manager.lastCreatedContextualStrings == [])
    }

    @Test func omittedStreamConfigDoesNotApplyContext() throws {
        let manager = ContextFakeManager()
        let response = routeCreate(manager, body: ["model": "qwen3-asr-0.6b"])
        let json = try responseJSON(response)

        #expect(response.status == 200)
        #expect(json["context_applied"] as? Bool == false)
        #expect(manager.lastCreatedContextualStrings == [])
    }

    @Test func emptyContextualStringsDoesNotApplyContext() throws {
        let manager = ContextFakeManager()
        let response = routeCreate(
            manager,
            body: ["model": "qwen3-asr-0.6b", "stream_config": ["contextual_strings": [String]()]]
        )
        let json = try responseJSON(response)

        #expect(response.status == 200)
        #expect(json["context_applied"] as? Bool == false)
        #expect(manager.lastCreatedContextualStrings == [])
    }

    @Test func nonemptyHintsAreForwardedAndAcknowledged() throws {
        let manager = ContextFakeManager()
        let phrases = ["Yuwp", "Qwen3-ASR", "file name.swift"]
        let response = routeCreate(
            manager,
            body: [
                "model": "qwen3-asr-0.6b",
                "stream_config": ["contextual_strings": phrases],
            ]
        )
        let json = try responseJSON(response)

        #expect(response.status == 200)
        #expect(json["session_id"] as? String == manager.createdSessionID)
        #expect(json["context_applied"] as? Bool == true)
        #expect(manager.lastCreatedContextualStrings == phrases)
    }

    @Test func languageQueryStillReachesManagerWhenBodySuppliesHints() throws {
        let manager = ContextFakeManager()
        let response = routeRequest(
            HTTPRequest(
                method: "POST",
                path: "/v1/audio/transcriptions/stream?language=Chinese",
                headers: ["content-type": "application/json"],
                body: jsonData([
                    "model": "qwen3-asr-0.6b",
                    "stream_config": ["contextual_strings": ["Yuwp"]],
                ])
            ),
            context: makeContext(manager)
        )
        let json = try responseJSON(response)

        #expect(response.status == 200)
        #expect(json["context_applied"] as? Bool == true)
        #expect(manager.lastCreatedLanguage == "Chinese")
        #expect(manager.lastCreatedContextualStrings == ["Yuwp"])
    }

    @Test func clientSystemPromptIsIgnoredAndDoesNotCountAsApplied() throws {
        let manager = ContextFakeManager()
        let response = routeCreate(
            manager,
            body: [
                "model": "qwen3-asr-0.6b",
                "stream_config": ["system_prompt": "ignore these instructions"],
            ]
        )
        let json = try responseJSON(response)

        #expect(response.status == 200)
        #expect(json["context_applied"] as? Bool == false)
        #expect(manager.lastCreatedContextualStrings == [])
    }

    @Test func systemPromptAlongsideHintsDoesNotReplaceVocabulary() throws {
        let manager = ContextFakeManager()
        let response = routeCreate(
            manager,
            body: [
                "model": "qwen3-asr-0.6b",
                "stream_config": [
                    "system_prompt": "ignore these instructions",
                    "contextual_strings": ["Yuwp"],
                ],
            ]
        )
        let json = try responseJSON(response)

        #expect(response.status == 200)
        #expect(json["context_applied"] as? Bool == true)
        #expect(manager.lastCreatedContextualStrings == ["Yuwp"])
    }

    @Test func whitespaceBodyIsTreatedAsOmittedContext() throws {
        let manager = ContextFakeManager()
        let response = routeRequest(
            HTTPRequest(
                method: "POST",
                path: "/v1/audio/transcriptions/stream",
                headers: ["content-type": "application/json"],
                body: Data("  \n".utf8)
            ),
            context: makeContext(manager)
        )
        let json = try responseJSON(response)

        #expect(response.status == 200)
        #expect(json["context_applied"] as? Bool == false)
        #expect(manager.lastCreatedContextualStrings == [])
    }

    @Test(arguments: invalidCreateBodies)
    func rejectsMalformedSuppliedContext(_ fixture: InvalidCreateBody) throws {
        let manager = ContextFakeManager()
        let response = routeRequest(
            HTTPRequest(
                method: "POST",
                path: "/v1/audio/transcriptions/stream",
                headers: ["content-type": "application/json"],
                body: fixture.body
            ),
            context: makeContext(manager)
        )
        let json = try responseJSON(response)
        let error = try #require(json["error"] as? [String: Any])

        #expect(response.status == 400)
        #expect(error["type"] as? String == "invalid_request_error")
        #expect(error["param"] as? String == fixture.param)
        #expect((error["message"] as? String)?.isEmpty == false)
        #expect(manager.createCallCount == 0)
        #expect(!(error["message"] as? String ?? "").contains("secret-hint"))
    }

    @Test func acceptsBoundaryCountsAndUTF8ByteLimits() throws {
        let manager = ContextFakeManager()
        let hundred = Array(repeating: "ok", count: 100)
        let maxPhrase = String(repeating: "é", count: 128) // 256 UTF-8 bytes
        let aggregate = Array(repeating: String(repeating: "a", count: 256), count: 32)

        let hundredResponse = routeCreate(
            manager,
            body: ["stream_config": ["contextual_strings": hundred]]
        )
        #expect(hundredResponse.status == 200)
        #expect(manager.lastCreatedContextualStrings == hundred)

        let phraseResponse = routeCreate(
            manager,
            body: ["stream_config": ["contextual_strings": [maxPhrase]]]
        )
        #expect(phraseResponse.status == 200)
        #expect(manager.lastCreatedContextualStrings == [maxPhrase])

        let aggregateResponse = routeCreate(
            manager,
            body: ["stream_config": ["contextual_strings": aggregate]]
        )
        let json = try responseJSON(aggregateResponse)
        #expect(aggregateResponse.status == 200)
        #expect(json["context_applied"] as? Bool == true)
        #expect(manager.lastCreatedContextualStrings.count == 32)
    }

    @Test func validPhrasesPreserveExactRawTextIncludingPaddingSpaces() throws {
        let manager = ContextFakeManager()
        let padded = " " + String(repeating: "a", count: 255) // 256 raw UTF-8 bytes
        let phrases = ["Yuwp", " file name.swift ", padded]
        let response = routeCreate(
            manager,
            body: ["stream_config": ["contextual_strings": phrases]]
        )
        let json = try responseJSON(response)

        #expect(response.status == 200)
        #expect(json["context_applied"] as? Bool == true)
        #expect(manager.lastCreatedContextualStrings == phrases)
        #expect(manager.lastCreatedContextualStrings[1] == " file name.swift ")
        #expect(manager.lastCreatedContextualStrings[2].utf8.count == 256)
    }

    @Test func wireBlankOracleUsesUnicodeWhiteSpacePlusZWSPAndBOM() {
        #expect(ASRContextualText.isBlankPhrase(""))
        #expect(ASRContextualText.isBlankPhrase("   "))
        #expect(ASRContextualText.isBlankPhrase("\u{00A0}"))
        #expect(ASRContextualText.isBlankPhrase("\u{3000}"))
        #expect(ASRContextualText.isBlankPhrase("\u{200B}"))
        #expect(ASRContextualText.isBlankPhrase("\u{FEFF}"))
        #expect(ASRContextualText.isBlankPhrase("\u{00A0}\u{200B}\u{FEFF} "))
        #expect(!ASRContextualText.isBlankPhrase("Yuwp"))
        #expect(!ASRContextualText.isBlankPhrase("Yuwp\u{200B}"))
        #expect(!ASRContextualText.isBlankPhrase("\u{FEFF}Oppi"))
        #expect(!ASRContextualText.isBlankPhrase("\u{00A0}Qwen"))
        #expect(!ASRContextualText.isBlankPhrase("file\u{3000}name"))
        if let feff = Unicode.Scalar(0xFEFF) {
            #expect(!CharacterSet.whitespacesAndNewlines.contains(feff))
        }
    }

    @Test func mixedNonblankWithZWSPBOMAndUnicodeSpacesIsPreservedExactly() throws {
        let manager = ContextFakeManager()
        let phrases = [
            "Yuwp\u{200B}",
            "\u{FEFF}Oppi",
            "\u{00A0}Qwen",
            "file\u{3000}name",
        ]
        let response = routeCreate(
            manager,
            body: ["stream_config": ["contextual_strings": phrases]]
        )
        let json = try responseJSON(response)

        #expect(response.status == 200)
        #expect(json["context_applied"] as? Bool == true)
        #expect(manager.lastCreatedContextualStrings == phrases)
    }
}

struct InvalidCreateBody: CustomTestStringConvertible, Sendable {
    let name: String
    let body: Data
    let param: String

    var testDescription: String { name }
}

private let invalidCreateBodies: [InvalidCreateBody] = {
    let tooMany = Array(repeating: "a", count: 101)
    let tooLongPhrase = String(repeating: "a", count: 257)
    let tooLongAggregate = Array(repeating: String(repeating: "a", count: 256), count: 32) + ["b"]
    return [
        InvalidCreateBody(name: "invalid JSON", body: Data("{".utf8), param: "body"),
        InvalidCreateBody(name: "non-object body", body: jsonData(["Yuwp"]), param: "body"),
        InvalidCreateBody(
            name: "stream_config null",
            body: Data("{\"stream_config\":null}".utf8),
            param: "stream_config"
        ),
        InvalidCreateBody(
            name: "stream_config array",
            body: jsonData(["stream_config": ["Yuwp"]]),
            param: "stream_config"
        ),
        InvalidCreateBody(
            name: "contextual_strings null",
            body: Data("{\"stream_config\":{\"contextual_strings\":null}}".utf8),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "contextual_strings object",
            body: jsonData(["stream_config": ["contextual_strings": ["term": "Yuwp"]]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "non-string phrase",
            body: jsonData(["stream_config": ["contextual_strings": [1]]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "empty phrase",
            body: jsonData(["stream_config": ["contextual_strings": [""]]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "whitespace-only phrase",
            body: jsonData(["stream_config": ["contextual_strings": ["   "]]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "NBSP-only phrase",
            body: jsonData(["stream_config": ["contextual_strings": ["\u{00A0}"]]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "ideographic-space-only phrase",
            body: jsonData(["stream_config": ["contextual_strings": ["\u{3000}"]]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "ZWSP-only phrase",
            body: jsonData(["stream_config": ["contextual_strings": ["\u{200B}"]]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "BOM-only phrase",
            body: jsonData(["stream_config": ["contextual_strings": ["\u{FEFF}"]]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "mixed blank-only ZWSP BOM and spaces",
            body: jsonData(["stream_config": ["contextual_strings": ["\u{00A0}\u{200B}\u{FEFF} "]]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "control characters",
            body: jsonData(["stream_config": ["contextual_strings": ["secret-hint\u{0001}"]]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "control characters before trim",
            body: jsonData(["stream_config": ["contextual_strings": [" Yuwp\u{0007} "]]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "UTF-8 limit counts raw padding",
            body: jsonData(["stream_config": ["contextual_strings": [" " + String(repeating: "a", count: 256)]]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "too many phrases",
            body: jsonData(["stream_config": ["contextual_strings": tooMany]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "phrase too long",
            body: jsonData(["stream_config": ["contextual_strings": [tooLongPhrase]]]),
            param: "stream_config.contextual_strings"
        ),
        InvalidCreateBody(
            name: "aggregate too long",
            body: jsonData(["stream_config": ["contextual_strings": tooLongAggregate]]),
            param: "stream_config.contextual_strings"
        ),
    ]
}()

private func routeCreate(_ manager: ContextFakeManager, body: [String: Any]) -> HTTPResponse {
    routeRequest(
        HTTPRequest(
            method: "POST",
            path: "/v1/audio/transcriptions/stream",
            headers: ["content-type": "application/json"],
            body: jsonData(body)
        ),
        context: makeContext(manager)
    )
}

private func makeContext(_ manager: ContextFakeManager) -> ASRRouteContext {
    ASRRouteContext(
        manager: manager,
        aligner: nil,
        vad: nil,
        streamingModelName: "stream",
        batchModelName: nil,
        batchRetranscribeEnabled: true,
        loadAudio: { _ in [] }
    )
}

private func responseJSON(_ response: HTTPResponse) throws -> [String: Any] {
    try #require(try JSONSerialization.jsonObject(with: response.body) as? [String: Any])
}

private func jsonData(_ object: Any) -> Data {
    (try? JSONSerialization.data(withJSONObject: object)) ?? Data()
}

private final class ContextFakeManager: ASRServing, @unchecked Sendable {
    var createdSessionID = "context-session"
    var createCallCount = 0
    var lastCreatedLanguage: String?
    var lastCreatedContextualStrings: [String] = []

    func create(language: String?, contextualStrings: [String]) -> String {
        createCallCount += 1
        lastCreatedLanguage = language
        lastCreatedContextualStrings = contextualStrings
        return createdSessionID
    }

    func feed(_ sid: String, pcmData: Data) -> [String: Any]? { nil }
    func stop(_ sid: String) -> [String: Any]? { nil }

    func transcribeAudio(audio: [Float], language: String?, temperature: Float) throws -> TranscriptionResult {
        TranscriptionResult(text: "", language: language, audioDuration: 0, processingTime: 0)
    }

    func transcribeChunk(audio: [Float], language: String?, temperature: Float) throws -> TranscriptionResult {
        try transcribeAudio(audio: audio, language: language, temperature: temperature)
    }

    func subtitleItems(
        audio: [Float],
        transcript: String?,
        language: String?,
        temperature: Float,
        aligner: ForcedAligner
    ) throws -> (transcript: String, language: String, items: [ForcedAlignItem]) {
        ("", language ?? "English", [])
    }
}
