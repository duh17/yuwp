import Foundation
@preconcurrency import MLX

public enum YuE2Audio {
    /// Write `[samples, channels]` audio as a 32-bit float WAV.
    public static func writeWav(_ audio: MLXArray, sampleRate: Int, to url: URL) throws {
        guard audio.ndim == 2 else { throw YuE2Error.invalidInput("expected [samples, channels] audio") }
        let channels = audio.dim(1)
        let samples = audio.asType(.float32).asArray(Float.self)
        var data = Data()
        func put<T: FixedWidthInteger>(_ value: T) { withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) } }
        let payload = samples.count * 4
        data.append(contentsOf: Array("RIFF".utf8))
        put(UInt32(36 + payload))
        data.append(contentsOf: Array("WAVEfmt ".utf8))
        put(UInt32(16))
        put(UInt16(3)) // IEEE float
        put(UInt16(channels))
        put(UInt32(sampleRate))
        put(UInt32(sampleRate * channels * 4))
        put(UInt16(channels * 4))
        put(UInt16(32))
        data.append(contentsOf: Array("data".utf8))
        put(UInt32(payload))
        samples.withUnsafeBufferPointer { data.append(UnsafeBufferPointer(start: UnsafeRawPointer($0.baseAddress)?.assumingMemoryBound(to: UInt8.self), count: payload)) }
        try data.write(to: url)
    }
}
