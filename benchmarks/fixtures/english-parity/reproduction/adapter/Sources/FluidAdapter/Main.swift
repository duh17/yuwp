import Foundation
import Darwin
import ASRIPC
import AdapterCore

@main struct Main {
    @MainActor static func main() async {
        // Preserve a dedicated protocol FD, then route any SDK print/C stdout
        // to stderr before touching FluidAudio. Only framed bytes reach the client.
        let protocolFD = dup(STDOUT_FILENO)
        guard protocolFD >= 0, dup2(STDERR_FILENO, STDOUT_FILENO) >= 0 else { exit(2) }
        signal(SIGPIPE, SIG_IGN)
        let output = FileHandle(fileDescriptor: protocolFD, closeOnDealloc: true)
        func reply(_ response: ASRIPCResponse) throws {
            try output.write(contentsOf: ASRIPCCodec.encode(response))
        }
        do {
            let args = Array(CommandLine.arguments.dropFirst())
            guard args.count == 4, args[0] == "--candidate",
                  let candidate = FluidBackend.Candidate(rawValue: args[1]),
                  args[2] == "--model-dir", args[3].hasPrefix("/") else {
                throw AdapterError.invalid("usage: fluid-parity-adapter --candidate n1|p1 --model-dir /absolute/local/directory; launching loads models (N1 also runs health-probe inference)")
            }
            let backend = FluidBackend(candidate: candidate, directory: URL(fileURLWithPath: args[3], isDirectory: true))
            let engine = SessionEngine(backend: backend)
            var startupError: String?
            do { try await engine.prepare() }
            catch { startupError = error.localizedDescription; log("load failed: \(error.localizedDescription)") }

            // Deliberately sequential: blocking full-frame read, awaited operation,
            // full reply write. No read-ahead task, no concurrent feed/finish.
            while true {
                let frame: ASRIPCFrame
                do {
                    guard let next = try FrameReader.next(read: {
                        try FileHandle.standardInput.read(upToCount: $0) ?? Data()
                    }) else { break }
                    frame = next
                } catch {
                    try reply(ASRIPCResponse(id: 0, ok: false, error: error.localizedDescription))
                    log("fatal framing error; id unavailable")
                    exit(2)
                }
                let request: ASRIPCRequest
                do { request = try ASRIPCCodec.decodeRequest(metadata: frame.metadata) }
                catch {
                    // An invalid command can still have a recoverable request ID.
                    struct IDOnly: Decodable { let id: UInt64 }
                    let id = (try? JSONDecoder().decode(IDOnly.self, from: frame.metadata).id) ?? 0
                    try reply(ASRIPCResponse(id: id, ok: false, error: "invalid request JSON: \(error.localizedDescription)"))
                    // Fail closed: no ambiguous missing packet can later finalize.
                    exit(2)
                }
                if let startupError {
                    try reply(ASRIPCResponse(id: request.id, ok: false, error: "models not ready: \(startupError)", status: "failed"))
                } else {
                    try reply(await engine.handle(request, binary: frame.binary))
                }
            }
        } catch {
            log(error.localizedDescription)
            exit(2)
        }
    }
}
