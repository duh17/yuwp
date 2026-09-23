import ASRIPC
import Foundation
import Testing

@Suite("ASRIPC framing")
struct ASRIPCTests {
    @Test func frameCodecEncodesAndDecodesHeaderLengths() {
        let metadata = Data("{\"id\":1}".utf8)
        let binary = Data([0x01, 0x02, 0x03, 0x04])

        let frame = ASRIPCFrameCodec.encodeFrame(metadata: metadata, binary: binary)
        let header = frame.prefix(ASRIPCFrameCodec.headerSize)

        let lengths = ASRIPCFrameCodec.decodeHeader(Data(header))
        #expect(lengths?.metadataLength == metadata.count)
        #expect(lengths?.binaryLength == binary.count)
    }

    @Test func createRequestDecodesMissingContextualStringsAsEmpty() throws {
        let json = Data("""
        {"id":1,"command":"create","language":"English"}
        """.utf8)
        let request = try ASRIPCCodec.decodeRequest(metadata: json)
        #expect(request.contextualStrings.isEmpty)
        #expect(request.language == "English")
    }

    @Test func createRequestRoundTripsContextualStrings() throws {
        let request = ASRIPCRequest(
            id: 7,
            command: .create,
            language: "English",
            contextualStrings: ["Yuwp", "Oppi"]
        )
        let encoded = try ASRIPCCodec.encode(request, binary: Data())
        let header = try #require(
            ASRIPCFrameCodec.decodeHeader(Data(encoded.prefix(ASRIPCFrameCodec.headerSize)))
        )
        let metadata = encoded.subdata(
            in: ASRIPCFrameCodec.headerSize ..< (ASRIPCFrameCodec.headerSize + header.metadataLength)
        )
        let decoded = try ASRIPCCodec.decodeRequest(metadata: metadata)
        #expect(decoded.contextualStrings == ["Yuwp", "Oppi"])
    }

    @Test func requestAndResponseCodecRoundTrip() throws {
        let request = ASRIPCRequest(id: 42, command: .feed, sessionID: "abc123", language: "Chinese")
        let binary = Data([0x10, 0x20])
        let encodedRequest = try ASRIPCCodec.encode(request, binary: binary)
        let requestHeader = try #require(
            ASRIPCFrameCodec.decodeHeader(Data(encodedRequest.prefix(ASRIPCFrameCodec.headerSize)))
        )
        let requestMetadata = encodedRequest.subdata(
            in: ASRIPCFrameCodec.headerSize ..< (ASRIPCFrameCodec.headerSize + requestHeader.metadataLength)
        )
        let decodedRequest = try ASRIPCCodec.decodeRequest(metadata: requestMetadata)

        #expect(decodedRequest == request)
        #expect(decodedRequest.contextualStrings.isEmpty)
        #expect(requestHeader.binaryLength == binary.count)

        let response = ASRIPCResponse(
            id: 42,
            ok: true,
            sessionID: "abc123",
            text: "hello",
            committedText: "hello",
            activeText: "",
            updateKind: "final",
            isFinal: true
        )
        let encodedResponse = try ASRIPCCodec.encode(response)
        let responseHeader = try #require(
            ASRIPCFrameCodec.decodeHeader(Data(encodedResponse.prefix(ASRIPCFrameCodec.headerSize)))
        )
        let responseMetadata = encodedResponse.subdata(
            in: ASRIPCFrameCodec.headerSize ..< (ASRIPCFrameCodec.headerSize + responseHeader.metadataLength)
        )
        let decodedResponse = try ASRIPCCodec.decodeResponse(metadata: responseMetadata)

        #expect(decodedResponse == response)
    }
}
