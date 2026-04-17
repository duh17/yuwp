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

    @Test func requestAndResponseCodecRoundTrip() throws {
        let request = ASRIPCRequest(id: 42, command: .feed, sessionID: "abc123")
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
