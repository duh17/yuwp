// NativeASR — Mel Spectrogram
// Computes log-mel spectrogram from 16kHz float32 PCM.
//
// DESIGN NOTES:
// - Matches WhisperFeatureExtractor: 400-sample FFT, 160-sample hop, 128 mel bins
// - Pre-computed Hann window and mel filterbank are cached as nonisolated(unsafe) statics
// - PITFALL: Drop last STFT frame to match PyTorch torch.stft(center=True) behavior
// - Output shape: (nFrames, nMels) — caller transposes to (nMels, nFrames) for model

import Foundation
import MLX

// MARK: - Constants

/// Audio processing constants for Qwen3-ASR
/// These match WhisperFeatureExtractor — do not change without retraining.
public enum ASRAudio {
    public static let sampleRate = 16000
    public static let nFft = 400          // 25ms window at 16kHz
    public static let hopLength = 160     // 10ms hop at 16kHz
    public static let nMels = 128
    public static let chunkLength = 30    // seconds
    public static let nSamples = chunkLength * sampleRate   // 480,000
    public static let nFrames = nSamples / hopLength        // 3000

    /// Pre-computed Hann window, shape (nFft,) = (400,)
    nonisolated(unsafe) public static let hannWindow: MLXArray = {
        let n = MLXArray((0 ..< nFft).map { Float($0) })
        let window = 0.5 * (1.0 - MLX.cos(n * (2.0 * Float.pi / Float(nFft - 1))))
        eval(window)
        return window
    }()

    /// Pre-computed mel filterbank, shape (nMels, nFft/2+1) = (128, 201)
    nonisolated(unsafe) public static let melFilterbank: MLXArray = {
        let filters = computeMelFilters(
            sampleRate: sampleRate, nFft: nFft, nMels: nMels,
            fMin: 0.0, fMax: Float(sampleRate) / 2.0
        )
        eval(filters)
        return filters
    }()
}

// MARK: - Log-Mel Spectrogram

/// Compute log-mel spectrogram from raw audio waveform.
///
/// - Parameter audio: 1D float array at 16kHz
/// - Returns: Log-mel spectrogram, shape (nFrames, 128)
public func logMelSpectrogram(audio: MLXArray) -> MLXArray {
    let window = ASRAudio.hannWindow
    let stftResult = computeSTFT(audio, window: window, nFft: ASRAudio.nFft, hopLength: ASRAudio.hopLength)

    // CRITICAL: Drop last frame to match PyTorch torch.stft behavior
    let freqs: MLXArray
    if stftResult.shape[0] > 1 {
        freqs = stftResult[0 ..< (stftResult.shape[0] - 1), 0...]
    } else {
        freqs = stftResult
    }

    // Power spectrum: |STFT|^2
    let magnitudes = MLX.pow(MLX.abs(freqs), 2)

    // Apply mel filterbank: (frames, freq) @ (freq, mels) → (frames, mels)
    let melSpec = MLX.matmul(magnitudes, ASRAudio.melFilterbank.T)

    // Log scale with floor clipping (prevents -inf on silent audio)
    var logSpec = MLX.log10(MLX.maximum(melSpec, MLXArray(Float(1e-10))))

    // Dynamic range compression (80dB range)
    logSpec = MLX.maximum(logSpec, logSpec.max() - 8.0)

    // Normalize to ~[-1, 1] range (WhisperFeatureExtractor constants)
    logSpec = (logSpec + 4.0) / 4.0

    return logSpec
}

// MARK: - STFT

private func computeSTFT(_ x: MLXArray, window: MLXArray, nFft: Int, hopLength: Int) -> MLXArray {
    // Reflection padding (center=True equivalent)
    let padded = reflectPad1D(x, padding: nFft / 2)

    let numFrames = 1 + (padded.shape[0] - nFft) / hopLength
    precondition(numFrames > 0, "Audio too short for STFT (need at least \(nFft) samples)")

    // Overlapping frames via strided view (no copy)
    let frames = MLX.asStrided(padded, [numFrames, nFft], strides: [hopLength, 1])

    // Apply window and compute real FFT
    return MLX.rfft(frames * window)
}

private func reflectPad1D(_ x: MLXArray, padding: Int) -> MLXArray {
    guard padding > 0 else { return x }
    let n = x.shape[0]
    guard n > 1 else { return x }

    var prefix = flip1D(x[1 ..< min(padding + 1, n)])
    var suffix = flip1D(x[max(0, n - padding - 1) ..< (n - 1)])

    while prefix.shape[0] < padding {
        let need = padding - prefix.shape[0]
        let extra = flip1D(x[1 ..< (min(need, n - 1) + 1)])
        prefix = MLX.concatenated([extra, prefix])
    }
    while suffix.shape[0] < padding {
        let need = padding - suffix.shape[0]
        let extra = flip1D(x[(max(0, n - need - 1)) ..< (n - 1)])
        suffix = MLX.concatenated([suffix, extra])
    }

    return MLX.concatenated([prefix[0 ..< padding], x, suffix[0 ..< padding]])
}

private func flip1D(_ x: MLXArray) -> MLXArray {
    let indices = MLXArray((0 ..< x.shape[0]).reversed().map { Int32($0) })
    return x[indices]
}

// MARK: - Mel Filterbank

/// Compute Slaney-style mel filterbank matrix.
/// - Returns: shape (nMels, nFft/2+1)
func computeMelFilters(sampleRate: Int, nFft: Int, nMels: Int, fMin: Float, fMax: Float) -> MLXArray {
    func hzToMel(_ hz: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel = minLogHz / fSp
        let logstep: Float = log(Float(6.4)) / 27.0
        return hz >= minLogHz ? minLogMel + log(hz / minLogHz) / logstep : hz / fSp
    }

    func melToHz(_ mel: Float) -> Float {
        let fSp: Float = 200.0 / 3.0
        let minLogHz: Float = 1000.0
        let minLogMel = minLogHz / fSp
        let logstep: Float = log(Float(6.4)) / 27.0
        return mel >= minLogMel ? minLogHz * exp(logstep * (mel - minLogMel)) : fSp * mel
    }

    let melMin = hzToMel(fMin)
    let melMax = hzToMel(fMax)
    let melPoints = (0 ... nMels + 1).map { i in
        melToHz(melMin + Float(i) * (melMax - melMin) / Float(nMels + 1))
    }

    let fftFreqs = (0 ..< (nFft / 2 + 1)).map { Float($0) * Float(sampleRate) / Float(nFft) }
    var filterbank = [[Float]](repeating: [Float](repeating: 0, count: nFft / 2 + 1), count: nMels)

    for m in 0 ..< nMels {
        let fLeft = melPoints[m]
        let fCenter = melPoints[m + 1]
        let fRight = melPoints[m + 2]
        let enorm = 2.0 / (melPoints[m + 2] - melPoints[m])
        for k in 0 ..< (nFft / 2 + 1) {
            let freq = fftFreqs[k]
            if freq >= fLeft, freq <= fCenter {
                filterbank[m][k] = (freq - fLeft) / (fCenter - fLeft) * enorm
            } else if freq > fCenter, freq <= fRight {
                filterbank[m][k] = (fRight - freq) / (fRight - fCenter) * enorm
            }
        }
    }

    return MLXArray(filterbank.flatMap { $0 }).reshaped([nMels, nFft / 2 + 1])
}
