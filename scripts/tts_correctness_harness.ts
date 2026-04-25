#!/usr/bin/env bun

import { existsSync, mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";
import { performance } from "node:perf_hooks";

const repo = resolve(import.meta.dir, "..");
const outDir = process.env.TTS_HARNESS_OUT_DIR || "/tmp/yuwp-tts-correctness-harness";
const timeoutMs = Number(process.env.TTS_HARNESS_TIMEOUT_MS || 900_000);
const sampleRate = 24_000;
const buildConfiguration = process.env.TTS_BUILD_CONFIGURATION || "release";
const ttsBin = binaryPath("yuwp-tts", buildConfiguration);
const asrBin = binaryPath("yuwp-asr", process.env.ASR_BUILD_CONFIGURATION || "debug");

const voiceDesignModel = expandHome(
	process.env.TTS_VOICE_MODEL ||
		"~/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-VoiceDesign/snapshots/385f44a7d86fa76e503b8633f22a5196b999e53b",
);

const referenceText =
	process.env.TTS_REF_TEXT ||
	"All right. Here is the new voice test. I am calm, but not too serious. I can explain the plan, notice the weird bugs, and occasionally admit that the machine has decided to say only done, which is honestly pretty funny.";

const referenceVoicePrompt =
	process.env.TTS_REF_VOICE ||
	"A natural middle-aged Asian man voice with a light-hearted calm tone. Warm, thoughtful, and a little dryly funny, like a relaxed technical teammate explaining progress late at night. Medium-low pitch, clear conversational English, gentle pacing, occasional tiny smile in the voice, not announcer-like, not dramatic, not overly polished, not robotic.";

const referenceAudio = expandHome(
	process.env.TTS_REF_AUDIO || join(outDir, "lighthearted-male-reference.wav"),
);

const cloneText =
	process.env.TTS_TEST_TEXT ||
	"The streaming correctness harness is checking the cloned voice path. Audio should begin quickly, remain intelligible, and keep a stable speaker identity while generation continues in the background.";

const requiredWords = [
	"streaming",
	"correctness",
	"harness",
	"cloned",
	"voice",
	"audio",
	"quickly",
	"intelligible",
	"stable",
	"speaker",
	"generation",
	"background",
];

const thresholds = {
	coverage: Number(process.env.TTS_MIN_COVERAGE || 0.8),
	firstAudioS: Number(process.env.TTS_MAX_FIRST_AUDIO_S || 2.5),
	peakMemoryGB: Number(process.env.TTS_MAX_PEAK_MEMORY_GB || 8.0),
	minDurationS: Number(process.env.TTS_MIN_AUDIO_DURATION_S || 5.0),
};

type CommandResult = {
	code: number;
	stdout: string;
	stderr: string;
	seconds: number;
};

type TTSMetrics = {
	firstAudioS: number;
	audioDurationS: number;
	wallS: number;
	chunks: number;
	tokens: number;
	tokensPerSecond: number;
	peakMemoryGB: number;
};

function main() {
	mkdirSync(outDir, { recursive: true });
	ensureBuildArtifacts();
	ensureReferenceAudio();

	const output = join(outDir, "clone-stream.wav");
	const tts = runTTS([
		"--model",
		voiceDesignModel,
		"--text",
		cloneText,
		"--ref-audio",
		referenceAudio,
		"--ref-text",
		referenceText,
		"--language",
		"English",
		"--temperature",
		process.env.TTS_TEMPERATURE || "0.8",
		"--stream",
		"--streaming-interval",
		process.env.TTS_STREAMING_INTERVAL || "1.0",
		"--out",
		output,
	]);

	const metrics = parseTTSMetrics(tts.stderr);
	const transcript = transcribe(output);
	const score = scoreTranscript(transcript, requiredWords);

	metric("first_audio_s", metrics.firstAudioS);
	metric("audio_duration_s", metrics.audioDurationS);
	metric("tts_wall_s", metrics.wallS);
	metric("chunks", metrics.chunks);
	metric("tokens", metrics.tokens);
	metric("tokens_per_second", metrics.tokensPerSecond);
	metric("peak_memory_gb", metrics.peakMemoryGB);
	metric("asr_coverage", score.coverage);
	metric("missing_required_words", score.missing.length);

	console.log(`TRANSCRIPT ${JSON.stringify(transcript)}`);
	if (score.missing.length > 0) console.log(`MISSING ${score.missing.join(",")}`);

	const failures: string[] = [];
	if (score.coverage < thresholds.coverage) failures.push(`ASR coverage ${score.coverage.toFixed(3)} < ${thresholds.coverage}`);
	if (metrics.firstAudioS > thresholds.firstAudioS) failures.push(`first audio ${metrics.firstAudioS.toFixed(3)}s > ${thresholds.firstAudioS}s`);
	if (metrics.peakMemoryGB > thresholds.peakMemoryGB) failures.push(`peak memory ${metrics.peakMemoryGB.toFixed(3)}GB > ${thresholds.peakMemoryGB}GB`);
	if (metrics.audioDurationS < thresholds.minDurationS) failures.push(`audio duration ${metrics.audioDurationS.toFixed(3)}s < ${thresholds.minDurationS}s`);

	if (failures.length > 0) {
		console.error(failures.map((failure) => `FAIL ${failure}`).join("\n"));
		process.exit(1);
	}
}

function ensureBuildArtifacts() {
	if (!existsSync(ttsBin)) {
		const args = buildConfiguration === "release"
			? ["build", "-c", "release", "--product", "yuwp-tts"]
			: ["build", "--product", "yuwp-tts"];
		run("swift", args, { label: "build yuwp-tts" });
	}
	if (!existsSync(asrBin)) {
		const args = (process.env.ASR_BUILD_CONFIGURATION || "debug") === "release"
			? ["build", "-c", "release", "--product", "yuwp-asr"]
			: ["build", "--product", "yuwp-asr"];
		run("swift", args, { label: "build yuwp-asr" });
	}
	const metallib = buildConfiguration === "release"
		? join(repo, ".build", "arm64-apple-macosx", "release", "mlx.metallib")
		: join(repo, ".build", "debug", "mlx.metallib");
	if (!existsSync(metallib)) {
		run("bash", ["scripts/build_mlx_metallib.sh", buildConfiguration === "release" ? "release" : "debug"], { label: "build mlx.metallib" });
	}
}

function ensureReferenceAudio() {
	if (existsSync(referenceAudio)) return;
	mkdirSync(dirname(referenceAudio), { recursive: true });
	console.error(`Generating reference voice fixture: ${referenceAudio}`);
	runTTS([
		"--model",
		voiceDesignModel,
		"--text",
		referenceText,
		"--voice",
		referenceVoicePrompt,
		"--language",
		"English",
		"--temperature",
		"0.9",
		"--stream",
		"--streaming-interval",
		"1.0",
		"--out",
		referenceAudio,
	]);
}

function runTTS(args: string[]): CommandResult {
	return run(ttsBin, args, { label: "yuwp-tts" });
}

function transcribe(audioPath: string): string {
	const result = run(asrBin, ["transcribe", audioPath, "--language", "en"], {
		label: "yuwp-asr transcribe",
	});
	return result.stdout.trim().replace(/\s+/g, " ");
}

function run(command: string, args: string[], options: { label: string }): CommandResult {
	const start = performance.now();
	const result = spawnSync(command, args, {
		cwd: repo,
		encoding: "utf8",
		timeout: timeoutMs,
		maxBuffer: 20 * 1024 * 1024,
	});
	const seconds = (performance.now() - start) / 1000;
	if (result.error) {
		throw new Error(`${options.label} failed: ${result.error.message}`);
	}
	const code = result.status ?? 0;
	const stdout = result.stdout || "";
	const stderr = result.stderr || "";
	if (code !== 0) {
		console.error(stderr || stdout);
		throw new Error(`${options.label} exited ${code}`);
	}
	return { code, stdout, stderr, seconds };
}

function parseTTSMetrics(stderr: string): TTSMetrics {
	const summary = stderr.match(/streamed ([0-9.]+)s audio in ([0-9.]+)s; first audio \+([0-9.]+)s; chunks=(\d+); tokens=(\d+)/);
	const generation = stderr.match(/Generation:\s+\d+ tokens,\s+([0-9.]+) tokens\/s/);
	const memory = stderr.match(/Peak Memory Usage:\s+([0-9.]+) GB/);
	if (!summary || !generation || !memory) {
		throw new Error(`Could not parse yuwp-tts metrics from stderr:\n${stderr}`);
	}
	return {
		audioDurationS: Number(summary[1]),
		wallS: Number(summary[2]),
		firstAudioS: Number(summary[3]),
		chunks: Number(summary[4]),
		tokens: Number(summary[5]),
		tokensPerSecond: Number(generation[1]),
		peakMemoryGB: Number(memory[1]),
	};
}

function scoreTranscript(transcript: string, expectedWords: string[]) {
	const words = new Set(contentWords(transcript));
	const missing = expectedWords.filter((word) => !words.has(normalizeWord(word)));
	const coverage = (expectedWords.length - missing.length) / expectedWords.length;
	return { coverage, missing };
}

function contentWords(text: string): string[] {
	return text
		.toLowerCase()
		.replace(/[^a-z0-9]+/g, " ")
		.split(/\s+/)
		.map(normalizeWord)
		.filter((word) => word.length >= 3);
}

function normalizeWord(word: string): string {
	if (word === "tts") return "speech";
	if (word === "backgrounds") return "background";
	if (word === "speakers") return "speaker";
	return word;
}

function metric(name: string, value: number) {
	console.log(`METRIC ${name}=${Number.isFinite(value) ? value.toFixed(6) : value}`);
}

function binaryPath(product: string, configuration: string): string {
	const candidates = configuration === "release"
		? [
			join(repo, ".build", "arm64-apple-macosx", "release", product),
			join(repo, ".build", "release", product),
		]
		: [
			join(repo, ".build", "debug", product),
			join(repo, ".build", "arm64-apple-macosx", "debug", product),
		];
	return candidates.find((candidate) => existsSync(candidate)) || candidates[0];
}

function expandHome(value: string): string {
	if (value === "~") return process.env.HOME || value;
	if (value.startsWith("~/")) return join(process.env.HOME || "", value.slice(2));
	return value;
}

main();
