import Foundation
import AVFoundation
@preconcurrency import CoreML
import FluidAudio
import AdapterCore

@MainActor final class FluidBackend: BenchmarkBackend {
    enum Candidate: String { case n1, p1 }
    let candidate: Candidate
    let directory: URL
    var modelName: String { candidate == .n1 ? "Nemotron-0.6B-en-560ms-N1-fused" : "Parakeet-TDT-0.6B-v2-en-P1" }
    var chunkSeconds: Double { candidate == .n1 ? 0.56 : 1.0 }
    var finalAccuracyPassEnabled: Bool { candidate == .p1 }
    private var nemotron: StreamingNemotronAsrManager?
    private var models: AsrModels?
    private var parakeet: AsrManager?
    private var prefix: [Float] = []
    private var schedule = PreviewSchedule()
    private var visible = ""

    init(candidate: Candidate, directory: URL) {
        self.candidate = candidate
        self.directory = directory
        // Must precede every loader, including AsrModels.load's ModelHub calls.
        ModelHub.offlineMode = true
    }

    private func require(_ relative: String) throws {
        let path = directory.appendingPathComponent(relative).path
        guard FileManager.default.fileExists(atPath: path) else {
            throw AdapterError.invalid("required local asset missing: \(path)")
        }
    }

    private func validateNemotron() throws {
        for file in ["metadata.json", "tokenizer.json", "encoder/encoder_int8.mlmodelc", "decoder.mlmodelc", "joint.mlmodelc", "decoder_joint.mlmodelc"] {
            try require(file)
        }
        let metadata = directory.appendingPathComponent("metadata.json")
        let config = try NemotronStreamingConfig(from: metadata)
        guard config.chunkMs == 560, config.chunkMelFrames == 56, config.chunkSamples == 8960,
              config.sampleRate == 16000, config.totalMelFrames == 65, config.preEncodeCache == 9 else {
            throw AdapterError.invalid("N1 requires explicit 560ms / 8960 samples / 65 mel frames / 16kHz metadata")
        }
    }

    private func newNemotron() async throws -> StreamingNemotronAsrManager {
        try validateNemotron()
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .cpuAndNeuralEngine
        let manager = StreamingNemotronAsrManager(configuration: configuration, requestedChunkSize: .ms560)
        // Local-only overload; contains an encoder inference health probe.
        try await manager.loadModels(from: directory)
        let loaded = await manager.config
        guard loaded.chunkMs == 560, loaded.chunkSamples == 8960 else {
            throw AdapterError.invalid("loaded Nemotron tier does not match N1")
        }
        log("N1 resolved encoder/decoder/joint/decoder_joint=cpuAndNeuralEngine; native Swift mel; fused required; encoder load probe executed")
        return manager
    }

    func prepare() async throws {
        guard ModelHub.offlineMode else { throw AdapterError.invalid("offline mode must be enabled") }
        switch candidate {
        case .n1:
            // Readiness is a real local load/probe, not file-existence success.
            // Discard this manager; every create must initialize clean throwing state.
            _ = try await newNemotron()
        case .p1:
            guard directory.lastPathComponent == "parakeet-tdt-0.6b-v2-coreml" else {
                throw AdapterError.invalid("P1 directory must retain SDK repo basename parakeet-tdt-0.6b-v2-coreml")
            }
            for file in ["Preprocessor.mlmodelc", "Encoder.mlmodelc", "Decoder.mlmodelc", "JointDecision.mlmodelc", "parakeet_vocab.json"] {
                try require(file)
            }
            let configuration = MLModelConfiguration()
            // SDK uses configuration.computeUnits for decoder/joint; override
            // encoder separately. Do not inherit the SDK CPU+ANE decoder default.
            configuration.computeUnits = .cpuOnly
            let loaded = try await AsrModels.load(from: directory, configuration: configuration,
                version: .v2, encoderComputeUnits: .cpuAndNeuralEngine)
            guard loaded.preprocessor.configuration.computeUnits == .cpuOnly,
                  loaded.decoder.configuration.computeUnits == .cpuOnly,
                  loaded.joint.configuration.computeUnits == .cpuOnly,
                  loaded.encoder?.configuration.computeUnits == .cpuAndNeuralEngine else {
                throw AdapterError.invalid("P1 component compute-unit configuration mismatch")
            }
            models = loaded
            log("P1 resolved Preprocessor/Decoder/JointDecision=cpuOnly; Encoder=cpuAndNeuralEngine; version=v2; SDK internal chunk concurrency=1")
        }
    }

    func create() async throws {
        discard()
        switch candidate {
        case .n1: nemotron = try await newNemotron()
        case .p1:
            guard let models else { throw AdapterError.invalid("Parakeet models not ready") }
            parakeet = AsrManager(config: ASRConfig(parallelChunkConcurrency: 1), models: models)
        }
    }

    func feed(_ samples: [Float]) async throws -> String {
        switch candidate {
        case .n1:
            guard let nemotron else { throw AdapterError.invalid("Nemotron stream missing") }
            let buffer = try Self.buffer(samples)
            // SDK process returns "" unconditionally. Getter decodes the same
            // accumulated IDs as +Pipeline's cumulative partial callback.
            _ = try await nemotron.process(audioBuffer: buffer)
            visible = await nemotron.getPartialTranscript()
        case .p1:
            let previewDue = try schedule.receive(samples.count)
            prefix.append(contentsOf: samples)
            if previewDue { visible = try await transcribePrefix() }
        }
        return visible
    }

    private func transcribePrefix() async throws -> String {
        guard let parakeet else { throw AdapterError.invalid("Parakeet stream missing") }
        // Fresh decoder state on every full-prefix pass, not a rolling window.
        var state = try TdtDecoderState(decoderLayers: 2)
        return try await parakeet.transcribe(prefix, decoderState: &state).text
    }

    func finish() async throws -> String {
        switch candidate {
        case .n1:
            guard let nemotron else { throw AdapterError.invalid("Nemotron stream missing") }
            return try await nemotron.finish()
        case .p1: return try await transcribePrefix()
        }
    }

    func discard() {
        // No best-effort SDK reset, no asynchronous cleanup racing the next stream.
        nemotron = nil
        parakeet = nil
        prefix = []
        schedule = PreviewSchedule()
        visible = ""
    }

    nonisolated private static func buffer(_ samples: [Float]) throws -> sending AVAudioPCMBuffer {
        guard let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000,
                                         channels: 1, interleaved: false),
              let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)),
              let channel = buffer.floatChannelData?[0] else {
            throw AdapterError.invalid("unable to allocate mono 16kHz audio buffer")
        }
        buffer.frameLength = AVAudioFrameCount(samples.count)
        for (i, value) in samples.enumerated() { channel[i] = value }
        return buffer
    }
}

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}
