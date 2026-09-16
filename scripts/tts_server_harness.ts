#!/usr/bin/env bun

import { spawn, spawnSync } from "node:child_process";
import { createInterface } from "node:readline";
import { existsSync, mkdirSync } from "node:fs";
import { join, resolve } from "node:path";

const repo = resolve(import.meta.dir, "..");
const outDir = process.env.TTS_HARNESS_OUT_DIR || "/tmp/yuwp-tts-correctness-harness";
mkdirSync(outDir, { recursive: true });

const model = expandHome(process.env.TTS_VOICE_MODEL || "~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign/snapshots/385f44a7d86fa76e503b8633f22a5196b999e53b");
const refAudio = expandHome(process.env.TTS_REF_AUDIO || join(outDir, "lighthearted-male-reference.wav"));
const refText = process.env.TTS_REF_TEXT || "All right. Here is the new voice test. I am calm, but not too serious. I can explain the plan, notice the weird bugs, and occasionally admit that the machine has decided to say only done, which is honestly pretty funny.";
const text = process.env.TTS_TEST_TEXT || "The resident server correctness harness is checking the cloned voice path. Audio should begin quickly, remain intelligible, and keep a stable speaker identity while generation continues in the background.";
const requiredWords = ["resident", "server", "correctness", "harness", "cloned", "voice", "audio", "quickly", "intelligible", "stable", "speaker", "generation", "background"];
const ttsBin = binaryPath("yuwp-tts", process.env.TTS_BUILD_CONFIGURATION || "release");
const asrBin = binaryPath("yuwp-asr", process.env.ASR_BUILD_CONFIGURATION || "debug");

ensureArtifacts();
if (!existsSync(refAudio)) throw new Error(`Missing reference audio fixture: ${refAudio}. Run scripts/tts_correctness_harness.ts once first.`);

const output = join(outDir, "server-clone-stream.wav");
const child = spawn(ttsBin, [
  "serve", "--model", model, "--ref-audio", refAudio, "--ref-text", refText,
  "--language", "English", "--temperature", process.env.TTS_TEMPERATURE || "0.8",
  "--streaming-interval", process.env.TTS_STREAMING_INTERVAL || "0.2",
], { cwd: repo, stdio: ["pipe", "pipe", "pipe"] });
child.stderr.on("data", (data) => process.stderr.write(data));
const rl = createInterface({ input: child.stdout });

let ready: any;
let done: any;
await new Promise<void>((resolvePromise, reject) => {
  const timer = setTimeout(() => reject(new Error("server harness timed out")), 180_000);
  rl.on("line", (line) => {
    const event = JSON.parse(line);
    if (event.event === "ready") {
      ready = event;
      child.stdin.write(JSON.stringify({ id: "server", text, out: output, emitChunks: false }) + "\n");
    } else if (event.event === "done") {
      done = event;
      child.stdin.write("shutdown\n");
      clearTimeout(timer);
      resolvePromise();
    } else if (event.event === "error") {
      clearTimeout(timer);
      reject(new Error(event.error));
    }
  });
  child.on("error", reject);
});
child.kill();

const transcript = transcribe(output);
const score = scoreTranscript(transcript, requiredWords);
metric("server_load_s", Number(ready.loadSeconds));
metric("first_audio_s", Number(done.firstAudioSeconds));
metric("audio_duration_s", Number(done.audioDurationSeconds));
metric("tts_wall_s", Number(done.wallSeconds));
metric("chunks", Number(done.chunks));
metric("tokens", Number(done.tokens));
metric("tokens_per_second", Number(done.tokensPerSecond));
metric("peak_memory_gb", Number(done.peakMemoryGB));
metric("asr_coverage", score.coverage);
metric("missing_required_words", score.missing.length);
console.log(`TRANSCRIPT ${JSON.stringify(transcript)}`);
if (score.missing.length) console.log(`MISSING ${score.missing.join(",")}`);

const failures: string[] = [];
if (score.coverage < Number(process.env.TTS_MIN_COVERAGE || 0.8)) failures.push("low ASR coverage");
if (Number(done.firstAudioSeconds) > Number(process.env.TTS_MAX_FIRST_AUDIO_S || 1.0)) failures.push("first audio too slow");
if (Number(done.peakMemoryGB) > Number(process.env.TTS_MAX_PEAK_MEMORY_GB || 8.0)) failures.push("peak memory too high");
if (failures.length) {
  console.error(failures.map((f) => `FAIL ${f}`).join("\n"));
  process.exit(1);
}

function ensureArtifacts() {
  if (!existsSync(ttsBin)) run("swift", ["build", "-c", "release", "--product", "yuwp-tts"], "build yuwp-tts");
  if (!existsSync(asrBin)) run("swift", ["build", "--product", "yuwp-asr"], "build yuwp-asr");
}

function transcribe(audio: string): string {
  const result = run(asrBin, ["transcribe", audio, "--language", "en"], "yuwp-asr transcribe");
  return result.stdout.trim().replace(/\s+/g, " ");
}

function run(command: string, args: string[], label: string) {
  const result = spawnSync(command, args, { cwd: repo, encoding: "utf8", timeout: 900_000, maxBuffer: 20 * 1024 * 1024 });
  if (result.error) throw result.error;
  if ((result.status ?? 0) !== 0) throw new Error(`${label} failed: ${result.stderr || result.stdout}`);
  return { stdout: result.stdout || "", stderr: result.stderr || "" };
}

function scoreTranscript(transcript: string, expected: string[]) {
  const words = new Set(transcript.toLowerCase().replace(/[^a-z0-9]+/g, " ").split(/\s+/));
  const missing = expected.filter((word) => !words.has(word));
  return { coverage: (expected.length - missing.length) / expected.length, missing };
}

function binaryPath(product: string, configuration: string): string {
  const swiftBuild = configuration === "release" ? "Release" : "Debug";
  const candidates = [
    join(repo, ".build", "out", "Products", swiftBuild, product),
    join(repo, ".build", "arm64-apple-macosx", configuration, product),
    join(repo, ".build", configuration, product),
  ];
  return candidates.find((candidate) => existsSync(candidate)) || candidates[0];
}

function metric(name: string, value: number) { console.log(`METRIC ${name}=${value.toFixed(6)}`); }
function expandHome(value: string) { return value.startsWith("~/") ? join(process.env.HOME || "", value.slice(2)) : value; }
