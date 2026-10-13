import Foundation
@preconcurrency import MLX

/// Reads a PyTorch zip checkpoint (`torch.save` of a flat state dict) into MLX arrays without Python.
///
/// Supports the subset `torch.save` emits for plain tensor state dicts: protocol-2 pickles of
/// `OrderedDict` → `torch._utils._rebuild_tensor_v2(storage, offset, size, stride, ...)` with
/// persistent storage ids, over an uncompressed (stored) zip archive.
public enum TorchCheckpointReader {
    public struct ReadError: Error, LocalizedError {
        public let message: String
        public var errorDescription: String? { "PyTorch checkpoint: \(message)" }
    }

    public static func load(url: URL) throws -> [String: MLXArray] {
        let data = try Data(contentsOf: url, options: .alwaysMapped)
        let entries = try zipEntries(data)
        guard let pickleEntry = entries.first(where: { $0.key.hasSuffix("/data.pkl") }) else {
            throw ReadError(message: "data.pkl not found in \(url.lastPathComponent)")
        }
        let prefix = String(pickleEntry.key.dropLast("data.pkl".count))
        var unpickler = Unpickler(bytes: try entryBytes(data, pickleEntry.value))
        let root = try unpickler.run()
        guard case .dict(let items) = root else {
            throw ReadError(message: "root object is not a state dict")
        }
        var out: [String: MLXArray] = [:]
        for (keyValue, tensorValue) in items {
            guard case .string(let key) = keyValue, case .tensor(let tensor) = tensorValue else {
                throw ReadError(message: "state dict entry is not a named tensor")
            }
            guard let entry = entries[prefix + "data/" + tensor.storageKey] else {
                throw ReadError(message: "storage \(tensor.storageKey) for \(key) is missing")
            }
            out[key] = try materialize(tensor, bytes: try entryBytes(data, entry), name: key)
        }
        return out
    }

    // MARK: Zip

    struct ZipEntry {
        var localHeaderOffset: Int
        var size: Int
    }

    static func zipEntries(_ data: Data) throws -> [String: ZipEntry] {
        let eocdSignature: UInt32 = 0x0605_4b50
        let minEOCD = 22
        guard data.count >= minEOCD else { throw ReadError(message: "file too small for zip") }
        var eocd = -1
        var index = data.count - minEOCD
        let lowerBound = max(0, data.count - minEOCD - 65_535)
        while index >= lowerBound {
            if data.readUInt32(at: index) == eocdSignature { eocd = index; break }
            index -= 1
        }
        guard eocd >= 0 else { throw ReadError(message: "zip end of central directory not found") }
        var entryCount = Int(data.readUInt16(at: eocd + 10))
        var directoryOffset = Int(data.readUInt32(at: eocd + 16))
        if directoryOffset == 0xFFFF_FFFF || entryCount == 0xFFFF {
            // ZIP64 end-of-central-directory locator precedes the classic record.
            let locator = eocd - 20
            guard locator >= 0, data.readUInt32(at: locator) == 0x0706_4b50 else {
                throw ReadError(message: "zip64 locator missing")
            }
            let zip64EOCD = Int(data.readUInt64(at: locator + 8))
            guard data.readUInt32(at: zip64EOCD) == 0x0606_4b50 else {
                throw ReadError(message: "zip64 end of central directory missing")
            }
            entryCount = Int(data.readUInt64(at: zip64EOCD + 32))
            directoryOffset = Int(data.readUInt64(at: zip64EOCD + 48))
        }
        var entries: [String: ZipEntry] = [:]
        var cursor = directoryOffset
        for _ in 0 ..< entryCount {
            guard data.readUInt32(at: cursor) == 0x0201_4b50 else {
                throw ReadError(message: "bad zip central directory entry")
            }
            let method = data.readUInt16(at: cursor + 10)
            var compressedSize = Int(data.readUInt32(at: cursor + 20))
            var uncompressedSize = Int(data.readUInt32(at: cursor + 24))
            let nameLength = Int(data.readUInt16(at: cursor + 28))
            let extraLength = Int(data.readUInt16(at: cursor + 30))
            let commentLength = Int(data.readUInt16(at: cursor + 32))
            var localOffset = Int(data.readUInt32(at: cursor + 42))
            let nameStart = cursor + 46
            let name = String(decoding: data[nameStart ..< nameStart + nameLength], as: UTF8.self)
            var extra = nameStart + nameLength
            let extraEnd = extra + extraLength
            while extra + 4 <= extraEnd {
                let tag = data.readUInt16(at: extra)
                let size = Int(data.readUInt16(at: extra + 2))
                if tag == 0x0001 {
                    var field = extra + 4
                    if uncompressedSize == 0xFFFF_FFFF { uncompressedSize = Int(data.readUInt64(at: field)); field += 8 }
                    if compressedSize == 0xFFFF_FFFF { compressedSize = Int(data.readUInt64(at: field)); field += 8 }
                    if localOffset == 0xFFFF_FFFF { localOffset = Int(data.readUInt64(at: field)) }
                }
                extra += 4 + size
            }
            guard method == 0, compressedSize == uncompressedSize else {
                throw ReadError(message: "\(name) is compressed; only stored entries are supported")
            }
            entries[name] = ZipEntry(localHeaderOffset: localOffset, size: uncompressedSize)
            cursor = extraEnd + commentLength
        }
        return entries
    }

    static func entryBytes(_ data: Data, _ entry: ZipEntry) throws -> Data {
        let header = entry.localHeaderOffset
        guard data.readUInt32(at: header) == 0x0403_4b50 else {
            throw ReadError(message: "bad zip local header")
        }
        let nameLength = Int(data.readUInt16(at: header + 26))
        let extraLength = Int(data.readUInt16(at: header + 28))
        let start = header + 30 + nameLength + extraLength
        guard start + entry.size <= data.count else { throw ReadError(message: "zip entry out of range") }
        return data[start ..< start + entry.size]
    }

    // MARK: Tensors

    struct TensorRef {
        var storageKey: String
        var dtype: DType
        var offset: Int
        var size: [Int]
        var stride: [Int]
    }

    static func materialize(_ tensor: TensorRef, bytes: Data, name: String) throws -> MLXArray {
        let itemSize = tensor.dtype.size
        let storageCount = bytes.count / itemSize
        let flat = MLXArray(bytes.prefix(storageCount * itemSize), [storageCount], dtype: tensor.dtype)
        if tensor.size.isEmpty {
            return flat[tensor.offset]
        }
        var contiguousStride = 1
        var isContiguous = true
        for (dim, stride) in zip(tensor.size, tensor.stride).reversed() {
            if dim != 1 && stride != contiguousStride { isContiguous = false }
            contiguousStride *= dim
        }
        let count = tensor.size.reduce(1, *)
        guard tensor.offset + count <= storageCount || !isContiguous else {
            throw ReadError(message: "\(name) exceeds its storage")
        }
        if isContiguous {
            return flat[tensor.offset ..< tensor.offset + count].reshaped(tensor.size)
        }
        return asStrided(flat, tensor.size, strides: tensor.stride, offset: tensor.offset)
    }

    static func dtype(forStorage name: String) throws -> DType {
        switch name {
        case "FloatStorage": .float32
        case "HalfStorage": .float16
        case "BFloat16Storage": .bfloat16
        case "LongStorage": .int64
        case "IntStorage": .int32
        case "ShortStorage": .int16
        case "CharStorage": .int8
        case "ByteStorage": .uint8
        case "BoolStorage": .bool
        case "DoubleStorage": .float64
        default: throw ReadError(message: "unsupported storage type \(name)")
        }
    }

    // MARK: Pickle

    indirect enum Value {
        case none
        case bool(Bool)
        case int(Int)
        case float(Double)
        case string(String)
        case bytes(Data)
        case tuple([Value])
        case list([Value])
        case dict([(Value, Value)])
        case global(module: String, name: String)
        case storage(key: String, dtype: DType)
        case tensor(TensorRef)
        case mark
        case opaque
    }

    struct Unpickler {
        let bytes: Data
        var position: Int
        var stack: [Value] = []
        var memo: [Int: Value] = [:]

        init(bytes: Data) {
            self.bytes = bytes
            self.position = bytes.startIndex
        }

        mutating func byte() throws -> UInt8 {
            guard position < bytes.endIndex else { throw ReadError(message: "pickle truncated") }
            defer { position += 1 }
            return bytes[position]
        }

        mutating func take(_ count: Int) throws -> Data {
            guard position + count <= bytes.endIndex else { throw ReadError(message: "pickle truncated") }
            defer { position += count }
            return bytes[position ..< position + count]
        }

        mutating func line() throws -> String {
            var out: [UInt8] = []
            while true {
                let next = try byte()
                if next == 0x0A { break }
                out.append(next)
            }
            return String(decoding: out, as: UTF8.self)
        }

        mutating func uint(_ count: Int) throws -> Int {
            let chunk = try take(count)
            var value = 0
            for (shift, byte) in chunk.enumerated() { value |= Int(byte) << (8 * shift) }
            return value
        }

        mutating func popMark() throws -> [Value] {
            guard let markIndex = stack.lastIndex(where: { if case .mark = $0 { true } else { false } }) else {
                throw ReadError(message: "pickle mark missing")
            }
            let items = Array(stack[(markIndex + 1)...])
            stack.removeSubrange(markIndex...)
            return items
        }

        mutating func pop() throws -> Value {
            guard let value = stack.popLast() else { throw ReadError(message: "pickle stack underflow") }
            return value
        }

        mutating func run() throws -> Value {
            while true {
                let op = try byte()
                switch op {
                case 0x80: _ = try byte() // PROTO
                case 0x95: _ = try take(8) // FRAME
                case 0x2E: return try pop() // STOP
                case 0x28: stack.append(.mark) // MARK
                case 0x4E: stack.append(.none)
                case 0x88: stack.append(.bool(true))
                case 0x89: stack.append(.bool(false))
                case 0x4B: stack.append(.int(try uint(1))) // BININT1
                case 0x4D: stack.append(.int(try uint(2))) // BININT2
                case 0x4A: stack.append(.int(Int(Int32(truncatingIfNeeded: try uint(4))))) // BININT
                case 0x8A: // LONG1
                    let count = Int(try byte())
                    let chunk = try take(count)
                    var value = 0
                    for (shift, byte) in chunk.enumerated() { value |= Int(byte) << (8 * shift) }
                    if count > 0, count < 8, (chunk.last ?? 0) & 0x80 != 0 { value -= 1 << (8 * count) }
                    stack.append(.int(value))
                case 0x47: // BINFLOAT (big-endian)
                    let chunk = try take(8)
                    let bits = chunk.reduce(UInt64(0)) { ($0 << 8) | UInt64($1) }
                    stack.append(.float(Double(bitPattern: bits)))
                case 0x58: // BINUNICODE
                    let count = try uint(4)
                    stack.append(.string(String(decoding: try take(count), as: UTF8.self)))
                case 0x8C: // SHORT_BINUNICODE
                    let count = Int(try byte())
                    stack.append(.string(String(decoding: try take(count), as: UTF8.self)))
                case 0x55: // SHORT_BINSTRING
                    let count = Int(try byte())
                    stack.append(.string(String(decoding: try take(count), as: UTF8.self)))
                case 0x43: // SHORT_BINBYTES
                    let count = Int(try byte())
                    stack.append(.bytes(try take(count)))
                case 0x42: // BINBYTES
                    let count = try uint(4)
                    stack.append(.bytes(try take(count)))
                case 0x29: stack.append(.tuple([])) // EMPTY_TUPLE
                case 0x85: stack.append(.tuple([try pop()])) // TUPLE1
                case 0x86: // TUPLE2
                    let second = try pop()
                    let first = try pop()
                    stack.append(.tuple([first, second]))
                case 0x87: // TUPLE3
                    let third = try pop()
                    let second = try pop()
                    let first = try pop()
                    stack.append(.tuple([first, second, third]))
                case 0x74: stack.append(.tuple(try popMark())) // TUPLE
                case 0x5D: stack.append(.list([])) // EMPTY_LIST
                case 0x6C: stack.append(.list(try popMark())) // LIST
                case 0x7D: stack.append(.dict([])) // EMPTY_DICT
                case 0x65: // APPENDS
                    let items = try popMark()
                    guard case .list(var list) = try pop() else { throw ReadError(message: "APPENDS target") }
                    list.append(contentsOf: items)
                    stack.append(.list(list))
                case 0x61: // APPEND
                    let item = try pop()
                    guard case .list(var list) = try pop() else { throw ReadError(message: "APPEND target") }
                    list.append(item)
                    stack.append(.list(list))
                case 0x73: // SETITEM
                    let value = try pop()
                    let key = try pop()
                    try setItems([key, value])
                case 0x75: try setItems(try popMark()) // SETITEMS
                case 0x63: // GLOBAL
                    let module = try line()
                    let name = try line()
                    stack.append(.global(module: module, name: name))
                case 0x71: memo[try uint(1)] = stack.last // BINPUT
                case 0x72: memo[try uint(4)] = stack.last // LONG_BINPUT
                case 0x94: memo[memo.count] = stack.last // MEMOIZE
                case 0x68: // BINGET
                    let key = try uint(1)
                    guard let value = memo[key] else { throw ReadError(message: "memo \(key) missing") }
                    stack.append(value)
                case 0x6A: // LONG_BINGET
                    let key = try uint(4)
                    guard let value = memo[key] else { throw ReadError(message: "memo \(key) missing") }
                    stack.append(value)
                case 0x51: // BINPERSID
                    let pid = try pop()
                    guard case .tuple(let parts) = pid, parts.count >= 3,
                          case .global(_, let storageName) = parts[1],
                          case .string(let key) = parts[2]
                    else { throw ReadError(message: "unsupported persistent id") }
                    stack.append(.storage(key: key, dtype: try TorchCheckpointReader.dtype(forStorage: storageName)))
                case 0x52: // REDUCE
                    let arguments = try pop()
                    let callable = try pop()
                    stack.append(try reduce(callable, arguments))
                case 0x62: // BUILD (state for OrderedDict / tensors is irrelevant)
                    _ = try pop()
                case 0x81: // NEWOBJ
                    _ = try pop()
                    _ = try pop()
                    stack.append(.opaque)
                default:
                    throw ReadError(message: String(format: "unsupported pickle opcode 0x%02X", op))
                }
            }
        }

        mutating func setItems(_ flat: [Value]) throws {
            guard case .dict(var items) = try pop() else { throw ReadError(message: "SETITEMS target is not a dict") }
            var index = 0
            while index + 1 < flat.count {
                items.append((flat[index], flat[index + 1]))
                index += 2
            }
            stack.append(.dict(items))
        }

        func reduce(_ callable: Value, _ arguments: Value) throws -> Value {
            guard case .global(let module, let name) = callable, case .tuple(let args) = arguments else {
                return .opaque
            }
            switch (module, name) {
            case ("collections", "OrderedDict"):
                return .dict([])
            case ("torch._utils", "_rebuild_tensor_v2"), ("torch._utils", "_rebuild_tensor"):
                guard args.count >= 4,
                      case .storage(let key, let dtype) = args[0],
                      case .int(let offset) = args[1],
                      case .tuple(let sizeValues) = args[2],
                      case .tuple(let strideValues) = args[3]
                else { throw ReadError(message: "malformed _rebuild_tensor_v2") }
                let size = try sizeValues.map { value -> Int in
                    guard case .int(let number) = value else { throw ReadError(message: "tensor size") }
                    return number
                }
                let stride = try strideValues.map { value -> Int in
                    guard case .int(let number) = value else { throw ReadError(message: "tensor stride") }
                    return number
                }
                return .tensor(TensorRef(storageKey: key, dtype: dtype, offset: offset, size: size, stride: stride))
            default:
                return .opaque
            }
        }
    }
}

extension Data {
    func readUInt16(at offset: Int) -> UInt16 {
        let base = startIndex + offset
        return UInt16(self[base]) | UInt16(self[base + 1]) << 8
    }

    func readUInt32(at offset: Int) -> UInt32 {
        let base = startIndex + offset
        var value: UInt32 = 0
        for index in 0 ..< 4 { value |= UInt32(self[base + index]) << (8 * UInt32(index)) }
        return value
    }

    func readUInt64(at offset: Int) -> UInt64 {
        let base = startIndex + offset
        var value: UInt64 = 0
        for index in 0 ..< 8 { value |= UInt64(self[base + index]) << (8 * UInt64(index)) }
        return value
    }
}
