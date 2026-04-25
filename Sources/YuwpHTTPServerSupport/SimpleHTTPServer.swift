#if canImport(Darwin)
import Darwin
#else
import Glibc
#endif
import Foundation

public struct HTTPServerConfig: Sendable {
    public let host: String
    public let port: UInt16
    public let parentPID: Int32?
    public let maxBodySize: Int
    public let log: @Sendable (String) -> Void

    public init(
        host: String,
        port: UInt16,
        parentPID: Int32? = nil,
        maxBodySize: Int = HTTPServerLimits.defaultMaxBodySize,
        log: @escaping @Sendable (String) -> Void = { _ in }
    ) {
        self.host = host
        self.port = port
        self.parentPID = parentPID
        self.maxBodySize = maxBodySize
        self.log = log
    }
}

public final class HTTPShutdownCoordinator: @unchecked Sendable {
    private let lock = NSLock()
    private var serverSocket: Int32 = -1
    private var shuttingDown = false
    private var parentWatchTimer: DispatchSourceTimer?
    private var signalSources: [DispatchSourceSignal] = []
    private let log: @Sendable (String) -> Void

    public init(log: @escaping @Sendable (String) -> Void = { _ in }) {
        self.log = log
    }

    public func setServerSocket(_ fd: Int32) {
        lock.lock()
        serverSocket = fd
        lock.unlock()
    }

    public func isShutdownRequested() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return shuttingDown
    }

    public func installSignalHandlers() {
        let signals = [SIGINT, SIGTERM]
        for signalNumber in signals {
            signal(signalNumber, SIG_IGN)
            let source = DispatchSource.makeSignalSource(
                signal: signalNumber,
                queue: DispatchQueue.global(qos: .userInitiated)
            )
            source.setEventHandler { [weak self] in
                _ = self?.requestShutdown(reason: "Received signal \(signalNumber) — shutting down")
            }
            source.resume()
            signalSources.append(source)
        }
    }

    public func startParentWatch(expectedParentPID: Int32?) {
        guard let expectedParentPID else { return }
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .background))
        timer.schedule(deadline: .now() + .seconds(1), repeating: .seconds(1))
        timer.setEventHandler { [weak self] in
            let currentParentPID = getppid()
            guard currentParentPID == expectedParentPID else {
                _ = self?.requestShutdown(
                    reason: "Parent process \(expectedParentPID) disappeared (current ppid: \(currentParentPID)) — shutting down"
                )
                return
            }
        }

        lock.lock()
        parentWatchTimer?.cancel()
        parentWatchTimer = timer
        lock.unlock()
        timer.resume()
    }

    @discardableResult
    public func requestShutdown(reason: String? = nil) -> Bool {
        lock.lock()
        guard !shuttingDown else {
            lock.unlock()
            return false
        }
        shuttingDown = true
        let timer = parentWatchTimer
        parentWatchTimer = nil
        let fd = serverSocket
        serverSocket = -1
        lock.unlock()

        if let reason { log(reason) }
        timer?.cancel()
        if fd >= 0 {
            shutdown(fd, SHUT_RDWR)
            close(fd)
        }
        return true
    }

    public func cleanup() {
        lock.lock()
        let timer = parentWatchTimer
        parentWatchTimer = nil
        let fd = serverSocket
        serverSocket = -1
        let sources = signalSources
        signalSources = []
        lock.unlock()

        timer?.cancel()
        for source in sources { source.cancel() }
        if fd >= 0 { close(fd) }
    }
}

public func startHTTPServer(
    config: HTTPServerConfig,
    handler: @escaping @Sendable (HTTPRequest) -> HTTPResponse
) -> Never {
    let shutdown = HTTPShutdownCoordinator(log: config.log)
    let serverFd = socket(AF_INET, SOCK_STREAM, 0)
    guard serverFd >= 0 else { fputs("socket() failed\n", stderr); exit(1) }

    var opt: Int32 = 1
    setsockopt(serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, socklen_t(MemoryLayout<Int32>.size))
    signal(SIGPIPE, SIG_IGN)

    var addr = sockaddr_in()
    addr.sin_family = sa_family_t(AF_INET)
    addr.sin_port = config.port.bigEndian
    addr.sin_addr.s_addr = inet_addr(config.host)

    let bindResult = withUnsafePointer(to: &addr) { pointer in
        pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) {
            bind(serverFd, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
        }
    }
    guard bindResult == 0 else {
        fputs("bind() failed on \(config.host):\(config.port) — errno \(errno)\n", stderr)
        exit(1)
    }
    guard listen(serverFd, 32) == 0 else { fputs("listen() failed\n", stderr); exit(1) }

    shutdown.setServerSocket(serverFd)
    shutdown.installSignalHandlers()
    shutdown.startParentWatch(expectedParentPID: config.parentPID)
    config.log("Listening on http://\(config.host):\(config.port)")

    let inFlight = DispatchGroup()
    while !shutdown.isShutdownRequested() {
        var clientAddr = sockaddr_in()
        var addrLen = socklen_t(MemoryLayout<sockaddr_in>.size)
        let clientFd = withUnsafeMutablePointer(to: &clientAddr) { pointer in
            pointer.withMemoryRebound(to: sockaddr.self, capacity: 1) { accept(serverFd, $0, &addrLen) }
        }
        if clientFd < 0 {
            if errno == EINTR { continue }
            break
        }

        inFlight.enter()
        DispatchQueue.global(qos: .userInitiated).async {
            defer {
                close(clientFd)
                inFlight.leave()
            }
            if let request = readHTTPRequest(fd: clientFd, maxBodySize: config.maxBodySize) {
                writeHTTPResponse(fd: clientFd, response: handler(request))
            }
        }
    }

    config.log("Shutting down...")
    if inFlight.wait(timeout: .now() + 5) == .timedOut {
        config.log("Timed out waiting for in-flight requests")
    }
    shutdown.cleanup()
    config.log("Server stopped")
    exit(0)
}

public func readHTTPRequest(fd: Int32, maxBodySize: Int = HTTPServerLimits.defaultMaxBodySize) -> HTTPRequest? {
    var data = Data()
    var buf = [UInt8](repeating: 0, count: 4096)
    while true {
        let n = recv(fd, &buf, buf.count, 0)
        if n <= 0 { return nil }
        data.append(buf, count: n)
        if let headerRange = data.range(of: Data("\r\n\r\n".utf8)) {
            let headerData = data[..<headerRange.lowerBound]
            guard let headerText = String(data: headerData, encoding: .utf8) else { return nil }
            let lines = headerText.components(separatedBy: "\r\n")
            guard let requestLine = lines.first else { return nil }
            let parts = requestLine.split(separator: " ")
            guard parts.count >= 2 else { return nil }
            var headers: [String: String] = [:]
            for line in lines.dropFirst() {
                guard let colon = line.firstIndex(of: ":") else { continue }
                let key = line[..<colon].trimmingCharacters(in: .whitespaces).lowercased()
                let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
                headers[key] = value
            }
            let contentLength = Int(headers["content-length"] ?? "0") ?? 0
            if contentLength > maxBodySize {
                writeHTTPResponse(
                    fd: fd,
                    response: jsonResponse(
                        status: 413,
                        ["error": "request body too large: \(contentLength) bytes (max \(maxBodySize))"]
                    )
                )
                return nil
            }
            let bodyStart = headerRange.upperBound
            while data.count - bodyStart < contentLength {
                let m = recv(fd, &buf, buf.count, 0)
                if m <= 0 { return nil }
                data.append(buf, count: m)
            }
            let body = Data(data[bodyStart ..< bodyStart + contentLength])
            return HTTPRequest(method: String(parts[0]), path: String(parts[1]), headers: headers, body: body)
        }
        if data.count > 64 * 1024 { return nil }
    }
}

public func writeHTTPResponse(fd: Int32, response: HTTPResponse) {
    let reason: String
    switch response.status {
    case 200: reason = "OK"
    case 201: reason = "Created"
    case 400: reason = "Bad Request"
    case 404: reason = "Not Found"
    case 405: reason = "Method Not Allowed"
    case 413: reason = "Payload Too Large"
    case 415: reason = "Unsupported Media Type"
    case 422: reason = "Unprocessable Entity"
    case 500: reason = "Internal Server Error"
    case 501: reason = "Not Implemented"
    default: reason = "OK"
    }
    var header = "HTTP/1.1 \(response.status) \(reason)\r\n"
    header += "Content-Type: \(response.contentType)\r\n"
    if response.stream != nil {
        header += "Transfer-Encoding: chunked\r\n"
    } else {
        header += "Content-Length: \(response.body.count)\r\n"
    }
    header += "Connection: close\r\n\r\n"
    _ = header.withCString { send(fd, $0, strlen($0), 0) }

    if let stream = response.stream {
        let writer = HTTPStreamWriter(
            writeChunk: { chunk in writeChunkedData(fd: fd, chunk) },
            writeFinalChunk: { _ = "0\r\n\r\n".withCString { send(fd, $0, strlen($0), 0) } }
        )
        stream(writer)
        writer.finish()
        return
    }

    sendAll(fd: fd, response.body)
}

private func sendAll(fd: Int32, _ data: Data) {
    data.withUnsafeBytes { raw in
        guard let base = raw.baseAddress else { return }
        var sentBytes = 0
        while sentBytes < data.count {
            let n = send(fd, base.advanced(by: sentBytes), data.count - sentBytes, 0)
            if n <= 0 { break }
            sentBytes += n
        }
    }
}

private func writeChunkedData(fd: Int32, _ data: Data) {
    guard !data.isEmpty else { return }
    let prefix = String(data.count, radix: 16) + "\r\n"
    _ = prefix.withCString { send(fd, $0, strlen($0), 0) }
    sendAll(fd: fd, data)
    _ = "\r\n".withCString { send(fd, $0, strlen($0), 0) }
}
