import AVFoundation
import Foundation
@preconcurrency import MLX
import Tokenizers

struct AuKEncodedPrompt {
    var inputIds: MLXArray
    var audioFeatures: MLXArray?
    var audioTokenMask: MLXArray?
    var audioFeatureLen: Int?
}

enum AuKProcessor {
    static func loadMonoWaveform(from url: URL, targetSampleRate: Int) throws -> [Float] {
        let audioFile = try AVAudioFile(forReading: url)
        let format = audioFile.processingFormat
        let frameCount = AVAudioFrameCount(audioFile.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AuKError.invalidInput("Failed to allocate audio buffer for \(url.path)")
        }
        try audioFile.read(into: buffer)
        guard let channelData = buffer.floatChannelData else {
            throw AuKError.invalidInput("Failed to read float audio for \(url.path)")
        }
        let frames = Int(buffer.frameLength)
        let channelCount = Int(format.channelCount)
        var channels: [[Float]] = []
        channels.reserveCapacity(max(channelCount, 1))
        for c in 0 ..< max(channelCount, 1) {
            channels.append(Array(UnsafeBufferPointer(start: channelData[c], count: frames)))
        }
        let mono = aukDownmixToMono(channels: channels)
        let sourceRate = Int(format.sampleRate)
        if sourceRate == targetSampleRate {
            return mono
        }
        return try resampleAudio(mono, from: sourceRate, to: targetSampleRate)
    }

    static func logMelSpectrogram(_ samples: [Float]) -> MLXArray {
        let audio = MLXArray(samples)
        let window = MLXArray(aukPeriodicHann(size: AuKFlashConfig.nFft))
        let freqs = stft(
            audio: audio,
            window: window,
            nFft: AuKFlashConfig.nFft,
            hopLength: AuKFlashConfig.hopLength,
            padMode: .reflect,
            removeDC: true
        )
        let usable: MLXArray
        if freqs.dim(0) > 1 {
            usable = freqs[0 ..< (freqs.dim(0) - 1), 0...]
        } else {
            usable = freqs
        }
        let magnitudes = abs(usable).square()
        let filters = melFilters(
            sampleRate: AuKFlashConfig.thinkerSampleRate,
            nFft: AuKFlashConfig.nFft,
            nMels: AuKFlashConfig.audioMelBins,
            norm: "slaney",
            melScale: .slaney
        )
        var mel = matmul(magnitudes, filters)
        mel = maximum(mel, MLXArray(Float(1e-10)))
        mel = log10(mel)
        mel = maximum(mel, mel.max() - 8)
        mel = (mel + 4) / 4
        return mel
    }

    static func encode(
        instruction: String,
        tokenizer: Tokenizer,
        referenceAudioURL: URL?
    ) throws -> AuKEncodedPrompt {
        let hasAudio = referenceAudioURL != nil
        let text = aukApplyChatTemplate(instruction: instruction, hasAudio: hasAudio)
        var ids = tokenizer.encode(text: text, addSpecialTokens: false)
        var audioFeatures: MLXArray?
        var audioMask: MLXArray?
        var featureLen: Int?
        if let referenceAudioURL {
            let samples = try loadMonoWaveform(from: referenceAudioURL, targetSampleRate: AuKFlashConfig.thinkerSampleRate)
            let mel = logMelSpectrogram(samples)
            eval(mel)
            featureLen = mel.dim(0)
            let tokenCount = aukAudioTowerTokenCount(melFrames: mel.dim(0))
            ids = aukExpandAudioTokens(ids, count: max(tokenCount, 1))
            audioFeatures = mel.reshaped([1, mel.dim(0), mel.dim(1)])
            let mask = ids.map { $0 == AuKFlashConfig.audioTokenId }
            audioMask = MLXArray(mask.map { $0 ? Int32(1) : Int32(0) }).reshaped([1, ids.count])
        }
        let inputIds = MLXArray(ids.map(Int32.init)).reshaped([1, ids.count])
        return AuKEncodedPrompt(
            inputIds: inputIds,
            audioFeatures: audioFeatures,
            audioTokenMask: audioMask,
            audioFeatureLen: featureLen
        )
    }
}
