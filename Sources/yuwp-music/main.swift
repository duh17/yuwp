import Foundation
@preconcurrency import MLX
import NativeMusic

let usage = """
Usage:
  yuwp-music generate --model <mlx-Yue2-3B dir> --style <text> (--lyrics <text> | --lyrics-file <file>)
                      [--abc-file <score.abc>] [--mode full|melody|off] [--seed N] [--steps 32]
                      [--precision 8bit|4bit|bf16] [--cfg <scale>] [--plan-only] --out <dir>
  yuwp-music render   --model <dir> --from <song dir with prefix.npy + semantic.npy [+ noise.npy]>
                      [--steps 32] [--seed N] --out <dir>
  yuwp-music decode   --model <dir> --latent <latent.npy> --out <audio.wav>

Common: --vae <dir> (default <model>/vae), --vae-core-frames 256.

generate writes score.abc, prefix.npy, semantic.npy, noise.npy, latent.npy, audio.wav and result.json.
Sections-only lyrics ([Intro], [Verse], …) plus a score with a resting Vocal line give instrumentals.
YuE2 weights are CC BY-NC 4.0.
"""

struct Options {
    var values: [String: String] = [:]
    var flags: Set<String> = []

    init(_ args: ArraySlice<String>) throws {
        var it = args.makeIterator()
        while let arg = it.next() {
            guard arg.hasPrefix("--") else { throw YuE2Error.invalidInput("unexpected argument \(arg)") }
            let name = String(arg.dropFirst(2))
            if ["plan-only", "help"].contains(name) {
                flags.insert(name)
            } else if let value = it.next() {
                values[name] = value
            } else {
                throw YuE2Error.invalidInput("--\(name) needs a value")
            }
        }
    }

    func url(_ name: String) -> URL? { values[name].map { URL(fileURLWithPath: ($0 as NSString).expandingTildeInPath).standardizedFileURL } }
    func require(_ name: String) throws -> String {
        guard let value = values[name] else { throw YuE2Error.invalidInput("--\(name) is required") }
        return value
    }
    func requireURL(_ name: String) throws -> URL {
        guard let url = url(name) else { throw YuE2Error.invalidInput("--\(name) is required") }
        return url
    }
    func int(_ name: String, default value: Int) throws -> Int {
        guard let raw = values[name] else { return value }
        guard let parsed = Int(raw) else { throw YuE2Error.invalidInput("--\(name) must be an integer") }
        return parsed
    }
}

func log(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

func save(_ ints: [Int], _ url: URL) throws {
    try MLX.save(array: MLXArray(ints.map(Int32.init)), url: url)
}

func loadInts(_ url: URL) throws -> [Int] {
    try MLX.loadArray(url: url).asType(.int32).asArray(Int32.self).map(Int.init)
}

func writeJSON(_ object: [String: Any], _ url: URL) throws {
    let data = try JSONSerialization.data(withJSONObject: object, options: [.prettyPrinted, .sortedKeys])
    try data.write(to: url)
}

func timingJSON(_ t: YuE2StageTiming?) -> Any {
    guard let t else { return NSNull() }
    return ["seconds": t.seconds, "prefill_seconds": t.prefillSeconds, "prefix_tokens": t.prefixTokens,
            "output_tokens": t.outputTokens, "tokens_per_second": t.tokensPerSecond]
}

func makeEngine(_ options: Options) throws -> YuE2Engine {
    let precisionName = options.values["precision"] ?? "8bit"
    guard let precision = YuE2Precision(rawValue: precisionName) else { throw YuE2Error.invalidInput("--precision must be 8bit, 4bit or bf16") }
    return try YuE2Engine(directory: try options.requireURL("model"), precision: precision, vaeDirectory: options.url("vae"))
}

/// Solve and decode, writing latent.npy and audio.wav; returns (nar seconds, vae seconds, audio seconds).
func render(_ engine: YuE2Engine, prefix: [Int], codes: [Int], noise: MLXArray, steps: Int, coreFrames: Int, out: URL) throws -> [String: Any] {
    var start = Date()
    var lastReport = Date.distantPast
    let latents = try engine.synthesize(prefix: prefix, codes: codes, noise: noise, steps: steps) { done, total in
        if Date().timeIntervalSince(lastReport) > 5 || done == total {
            lastReport = Date()
            log("  flow matching \(done)/\(total) steps (\(String(format: "%.0f", Date().timeIntervalSince(start)))s)")
        }
    }
    let narSeconds = Date().timeIntervalSince(start)
    try MLX.save(array: latents, url: out.appendingPathComponent("latent.npy"))
    start = Date()
    let audio = try engine.vaeDecoder().decode(latents, coreFrames: coreFrames)
    let vaeSeconds = Date().timeIntervalSince(start)
    try YuE2Audio.writeWav(audio, sampleRate: YuE2Protocol.sampleRate, to: out.appendingPathComponent("audio.wav"))
    let audioSeconds = Double(audio.dim(0)) / Double(YuE2Protocol.sampleRate)
    log(String(format: "rendered %.1fs of audio: NAR %.1fs (%d steps), VAE %.1fs", audioSeconds, narSeconds, steps, vaeSeconds))
    return ["nar_seconds": narSeconds, "vae_seconds": vaeSeconds, "audio_seconds": audioSeconds, "ode_steps": steps]
}

func runGenerate(_ options: Options) throws {
    let out = try options.requireURL("out")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let lyrics: String = if let file = options.url("lyrics-file") { try String(contentsOf: file, encoding: .utf8) } else { try options.require("lyrics") }
    guard let cot = YuE2Cot(rawValue: options.values["mode"] ?? "full") else { throw YuE2Error.invalidInput("--mode must be full, melody or off") }
    let abc = try options.url("abc-file").map { try String(contentsOf: $0, encoding: .utf8) }
    let cfg = try options.values["cfg"].map { raw -> Float in
        guard let value = Float(raw) else { throw YuE2Error.invalidInput("--cfg must be a number") }
        return value
    }
    let request = YuE2Request(
        style: try options.require("style"), lyrics: lyrics, cot: cot,
        seed: UInt64(try options.int("seed", default: 831_001)), abc: abc, cfgScale: cfg
    )
    let steps = try options.int("steps", default: 32)
    let started = Date()
    let engine = try makeEngine(options)

    log(abc == nil && cot != .off ? "planning the score…" : "using \(cot == .off ? "no" : "the supplied") score")
    let plan = try engine.plan(request)
    if let score = plan.abc {
        try score.write(to: out.appendingPathComponent("score.abc"), atomically: true, encoding: .utf8)
    }
    if let t = plan.timing {
        log(String(format: "score: %d tokens in %.1fs (%.1f tok/s)%@", t.outputTokens, t.seconds, t.tokensPerSecond, plan.truncated ? " — truncated" : ""))
    }
    var result: [String: Any] = [
        "request": ["style": request.style, "lyrics": request.lyrics, "mode": cot.rawValue, "seed": request.seed,
                    "cfg_scale": request.guidance, "supplied_abc": abc != nil],
        "precision": engine.precision.rawValue,
        "abc_timing": timingJSON(plan.timing), "abc_truncated": plan.truncated,
    ]
    try save(plan.prefix, out.appendingPathComponent("prefix.npy"))
    if options.flags.contains("plan-only") {
        try writeJSON(result, out.appendingPathComponent("result.json"))
        log("wrote \(out.path)")
        return
    }

    log("generating semantic tokens…")
    var count = 0
    let semantic = try engine.semantic(plan) { _ in
        count += 1
        if count % 1000 == 0 { log("  \(count) tokens (\(count / YuE2Protocol.frameRate)s of audio)") }
    }
    let t = semantic.timing
    log(String(format: "semantic: %d tokens in %.1fs (%.1f tok/s, prefill %.2fs)%@", t.outputTokens, t.seconds, t.tokensPerSecond, t.prefillSeconds, semantic.truncated ? " — truncated" : ""))
    try save(semantic.codes, out.appendingPathComponent("semantic.npy"))
    result["semantic_timing"] = timingJSON(t)
    result["semantic_truncated"] = semantic.truncated
    engine.releaseDecoder()

    let noise = YuE2Engine.noise(frames: semantic.codes.count, seed: request.seed)
    try MLX.save(array: noise, url: out.appendingPathComponent("noise.npy"))
    let rendered = try render(engine, prefix: plan.prefix, codes: semantic.codes, noise: noise, steps: steps,
                              coreFrames: try options.int("vae-core-frames", default: 256), out: out)
    result.merge(rendered) { $1 }
    result["e2e_seconds"] = Date().timeIntervalSince(started)
    try writeJSON(result, out.appendingPathComponent("result.json"))
    log("wrote \(out.path)")
}

func runRender(_ options: Options) throws {
    let source = try options.requireURL("from")
    let out = try options.requireURL("out")
    try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
    let prefix = try loadInts(source.appendingPathComponent("prefix.npy"))
    let codes = try loadInts(source.appendingPathComponent("semantic.npy"))
    let noiseURL = source.appendingPathComponent("noise.npy")
    let noise = FileManager.default.fileExists(atPath: noiseURL.path) && options.values["seed"] == nil
        ? try MLX.loadArray(url: noiseURL).asType(.float32)
        : YuE2Engine.noise(frames: codes.count, seed: UInt64(try options.int("seed", default: 831_001)))
    let engine = try makeEngine(options)
    let rendered = try render(engine, prefix: prefix, codes: codes, noise: noise, steps: try options.int("steps", default: 32),
                              coreFrames: try options.int("vae-core-frames", default: 256), out: out)
    try writeJSON(rendered.merging(["source": source.path]) { $1 }, out.appendingPathComponent("result.json"))
    log("wrote \(out.path)")
}

func runDecode(_ options: Options) throws {
    let latents = try MLX.loadArray(url: try options.requireURL("latent")).asType(.float32)
    let engine = try makeEngine(options)
    let start = Date()
    let audio = try engine.vaeDecoder().decode(latents, coreFrames: try options.int("vae-core-frames", default: 256))
    try YuE2Audio.writeWav(audio, sampleRate: YuE2Protocol.sampleRate, to: try options.requireURL("out"))
    log(String(format: "decoded %.1fs of audio in %.1fs", Double(audio.dim(0)) / Double(YuE2Protocol.sampleRate), Date().timeIntervalSince(start)))
}

let arguments = CommandLine.arguments.dropFirst()
do {
    guard let command = arguments.first else {
        print(usage)
        exit(2)
    }
    let options = try Options(arguments.dropFirst())
    if options.flags.contains("help") || command == "--help" || command == "help" {
        print(usage)
        exit(0)
    }
    switch command {
    case "generate": try runGenerate(options)
    case "render": try runRender(options)
    case "decode": try runDecode(options)
    default:
        print(usage)
        exit(2)
    }
} catch {
    log("error: \(error)")
    exit(1)
}
