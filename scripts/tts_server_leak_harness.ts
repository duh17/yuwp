#!/usr/bin/env bun

import { spawn, spawnSync } from "node:child_process";
import { createInterface } from "node:readline";
import { existsSync, mkdirSync } from "node:fs";
import { join, resolve } from "node:path";

const repo = resolve(import.meta.dir, "..");
const outDir = process.env.TTS_HARNESS_OUT_DIR || "/tmp/yuwp-tts-correctness-harness";
mkdirSync(outDir, { recursive: true });

const iterations = Number(process.env.TTS_LEAK_ITERATIONS || 5);
const model = expandHome(process.env.TTS_VOICE_MODEL || "~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign/snapshots/385f44a7d86fa76e503b8633f22a5196b999e53b");
const refAudio = expandHome(process.env.TTS_REF_AUDIO || join(outDir, "lighthearted-male-reference.wav"));
const refText = process.env.TTS_REF_TEXT || "All right. Here is the new voice test. I am calm, but not too serious. I can explain the plan, notice the weird bugs, and occasionally admit that the machine has decided to say only done, which is honestly pretty funny.";
const text = process.env.TTS_TEST_TEXT || "The resident leak harness is checking repeated cloned voice streaming requests. Audio should stay intelligible while memory remains bounded across the run.";
const requiredWords = ["resident", "leak", "harness", "repeated", "cloned", "voice", "streaming", "requests", "audio", "intelligible", "memory", "bounded"];
const ttsBin = binaryPath("yuwp-tts", process.env.TTS_BUILD_CONFIGURATION || "release");
const asrBin = binaryPath("yuwp-asr", process.env.ASR_BUILD_CONFIGURATION || "debug");
if (!existsSync(refAudio)) throw new Error(`Missing reference audio fixture: ${refAudio}. Run scripts/tts_correctness_harness.ts once first.`);

const child = spawn(ttsBin, [
  "serve", "--model", model, "--ref-audio", refAudio, "--ref-text", refText,
  "--language", "English", "--temperature", process.env.TTS_TEMPERATURE || "0.8",
  "--streaming-interval", process.env.TTS_STREAMING_INTERVAL || "0.2",
], { cwd: repo, stdio: ["pipe", "pipe", "pipe"] });
child.stderr.on("data", (data) => process.stderr.write(data));
const rl = createInterface({ input: child.stdout });

let ready = false;
let current = 0;
const doneEvents: any[] = [];
const doneRSSMB: number[] = [];
const rssSamplesMB: number[] = [];
const outputPaths: string[] = [];
const sampleTimer = setInterval(() => {
  if (child.pid) rssSamplesMB.push(readRSSMB(child.pid));
}, 250);

await new Promise<void>((resolvePromise, reject) => {
  const timer = setTimeout(() => reject(new Error("leak harness timed out")), Math.max(180_000, iterations * 90_000));
  rl.on("line", (line) => {
    const event = JSON.parse(line);
    if (event.event === "ready") {
      ready = true;
      sendNext();
    } else if (event.event === "done") {
      doneEvents.push(event);
      if (child.pid) doneRSSMB.push(readRSSMB(child.pid));
      if (doneEvents.length >= iterations) {
        child.stdin.write("shutdown\n");
        clearTimeout(timer);
        resolvePromise();
      } else {
        sendNext();
      }
    } else if (event.event === "error") {
      clearTimeout(timer);
      reject(new Error(event.error));
    }
  });
  child.on("error", reject);
});
clearInterval(sampleTimer);
child.kill();
if (!ready || doneEvents.length !== iterations) throw new Error("server did not complete requested iterations");

const firstTranscript = transcribe(outputPaths[0]);
const lastTranscript = transcribe(outputPaths[outputPaths.length - 1]);
const firstScore = scoreTranscript(firstTranscript, requiredWords);
const lastScore = scoreTranscript(lastTranscript, requiredWords);
const firstAudio = doneEvents.map((event) => Number(event.firstAudioSeconds));
const wall = doneEvents.map((event) => Number(event.wallSeconds));
const peakMem = doneEvents.map((event) => Number(event.peakMemoryGB));
const rssStart = rssSamplesMB[0] ?? 0;
const rssEnd = rssSamplesMB[rssSamplesMB.length - 1] ?? rssStart;
const rssMax = Math.max(...rssSamplesMB, rssStart);
const doneRSSFirst = doneRSSMB[0] ?? rssStart;
const doneRSSEnd = doneRSSMB[doneRSSMB.length - 1] ?? doneRSSFirst;
const doneRSSMax = Math.max(...doneRSSMB, doneRSSFirst);

metric("iterations", iterations);
metric("first_audio_mean_s", mean(firstAudio));
metric("first_audio_max_s", Math.max(...firstAudio));
metric("tts_wall_mean_s", mean(wall));
metric("peak_memory_max_gb", Math.max(...peakMem));
metric("rss_start_mb", rssStart);
metric("rss_end_mb", rssEnd);
metric("rss_max_mb", rssMax);
metric("rss_delta_mb", rssEnd - rssStart);
metric("done_rss_first_mb", doneRSSFirst);
metric("done_rss_end_mb", doneRSSEnd);
metric("done_rss_max_mb", doneRSSMax);
metric("done_rss_delta_mb", doneRSSEnd - doneRSSFirst);
metric("asr_coverage_first", firstScore.coverage);
metric("asr_coverage_last", lastScore.coverage);
metric("missing_required_words_last", lastScore.missing.length);
console.log(`FIRST_TRANSCRIPT ${JSON.stringify(firstTranscript)}`);
console.log(`LAST_TRANSCRIPT ${JSON.stringify(lastTranscript)}`);
if (lastScore.missing.length) console.log(`MISSING_LAST ${lastScore.missing.join(",")}`);

const failures: string[] = [];
if (firstScore.coverage < 0.8 || lastScore.coverage < 0.8) failures.push("ASR coverage below 0.8");
if (doneRSSEnd - doneRSSFirst > Number(process.env.TTS_MAX_DONE_RSS_DELTA_MB || 250)) failures.push(`post-first RSS delta too high: ${(doneRSSEnd - doneRSSFirst).toFixed(1)}MB`);
if (Math.max(...peakMem) > Number(process.env.TTS_MAX_PEAK_MEMORY_GB || 8.0)) failures.push("peak MLX memory too high");
if (failures.length) {
  console.error(failures.map((failure) => `FAIL ${failure}`).join("\n"));
  process.exit(1);
}

function sendNext() {
  const id = `iter-${current}`;
  const out = join(outDir, `leak-${id}.wav`);
  outputPaths.push(out);
  child.stdin.write(JSON.stringify({ id, text, out, emitChunks: false }) + "\n");
  current += 1;
}

function readRSSMB(pid: number): number {
  const result = spawnSync("ps", ["-o", "rss=", "-p", String(pid)], { encoding: "utf8" });
  const kb = Number((result.stdout || "0").trim());
  return Number.isFinite(kb) ? kb / 1024 : 0;
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
function mean(values: number[]) { return values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1); }
function metric(name: string, value: number) { console.log(`METRIC ${name}=${value.toFixed(6)}`); }
function expandHome(value: string) { return value.startsWith("~/") ? join(process.env.HOME || "", value.slice(2)) : value; }
