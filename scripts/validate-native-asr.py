#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "mlx-audio>=0.2.2",
# ]
# ///
"""Validation harness for the native Swift Qwen3-ASR experiment.

This script does two jobs:
1. Generate golden transcriptions from the current Python/MLX reference path.
2. Compare a candidate native CLI against those goldens.

Why this exists:
- Phase 1 of the native Swift port is batch transcription first.
- We want a tiny, deterministic smoke suite with immediate pass/fail signal.
- Goldens come from the current Python implementation so we can move fast
  without arguing about expected transcripts every run.

The tracked sample sets intentionally use public, non-personal audio.

Examples:

  # Generate / refresh the default smoke goldens (committed public fixtures)
  uv run scripts/validate-native-asr.py golden

  # Compare a built native CLI against the saved goldens
  uv run scripts/validate-native-asr.py compare \
    --candidate-cmd 'swift run --package-path . asr-test {audio} {model_dir} {language_arg}' \
    --model-dir '~/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0'

Notes:
- The candidate command runs through `bash -lc`.
- Placeholders available in --candidate-cmd:
    {audio}        absolute path to the audio file (shell-quoted)
    {model_dir}    absolute model directory path (shell-quoted)
    {language}     raw language value (shell-quoted; empty when nil)
    {language_arg} either `--language <lang>` or `--no-language`
    {id}           sample case id (not quoted)
- Goldens use the current Python batch path (`mlx_audio.stt.generate_transcription`).
- The script chdirs to a temp dir during reference transcription because mlx-audio
  writes `transcript.txt` to the current working directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import string
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
TRACKED_FIXTURES_DIR = REPO_ROOT / "Tests" / "fixtures"
DEFAULT_GOLDEN_PATH = TRACKED_FIXTURES_DIR / "native-asr-golden-smoke.json"
DEFAULT_REFERENCE_MODEL = "mlx-community/Qwen3-ASR-0.6B-bf16"
QWEN_ASR_ROOT = Path.home() / "workspace" / "qwen-asr"  # only needed for "extended" sample set


@dataclass(frozen=True)
class SampleCase:
    id: str
    audio: Path
    language: str | None = None
    notes: str | None = None
    duration_s: float | None = None


BUILTIN_SAMPLE_SETS: dict[str, list[SampleCase]] = {
    "smoke": [
        SampleCase(
            id="jfk",
            audio=TRACKED_FIXTURES_DIR / "jfk.wav",
            notes="Public English speech sample.",
        ),
        SampleCase(
            id="official_en",
            audio=TRACKED_FIXTURES_DIR / "asr_en.wav",
            notes="Official Qwen3-ASR English sample.",
        ),
        SampleCase(
            id="official_zh",
            audio=TRACKED_FIXTURES_DIR / "asr_zh.wav",
            notes="Official Qwen3-ASR Chinese sample.",
        ),
    ],
    "extended": [
        SampleCase(
            id="jfk",
            audio=TRACKED_FIXTURES_DIR / "jfk.wav",
            notes="Public English speech sample.",
        ),
        SampleCase(
            id="official_en",
            audio=TRACKED_FIXTURES_DIR / "asr_en.wav",
            notes="Official Qwen3-ASR English sample.",
        ),
        SampleCase(
            id="official_zh",
            audio=TRACKED_FIXTURES_DIR / "asr_zh.wav",
            notes="Official Qwen3-ASR Chinese sample.",
        ),
        SampleCase(
            id="test_speech",
            audio=QWEN_ASR_ROOT / "samples" / "test_speech.wav",
            notes="Very short public speech sample.",
        ),
        SampleCase(
            id="back_down_the_road",
            audio=QWEN_ASR_ROOT / "samples" / "night_of_the_living_dead_1968" / "10s_back_down_the_road.wav",
            notes="Public-domain movie clip.",
        ),
    ],
}


def available_sample_sets() -> dict[str, list[SampleCase]]:
    return dict(BUILTIN_SAMPLE_SETS)


@dataclass
class GoldenCase:
    id: str
    audio: str
    language: str | None
    notes: str | None
    duration_s: float
    processing_time_s: float
    text: str
    normalized_text: str


@dataclass
class CompareCase:
    id: str
    audio: str
    duration_s: float
    golden_text: str
    candidate_text: str
    golden_normalized: str
    candidate_normalized: str
    raw_exact: bool
    normalized_exact: bool
    cer: float
    wer: float
    candidate_time_s: float
    load_time_s: float | None
    inference_time_s: float | None
    stderr_tail: str


def iso_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def audio_duration_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-i",
            str(path),
            "-show_entries",
            "format=duration",
            "-v",
            "quiet",
            "-of",
            "csv=p=0",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()[:200]}")
    return float(result.stdout.strip())


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(rf"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein(seq_a: list[str] | str, seq_b: list[str] | str) -> int:
    if seq_a == seq_b:
        return 0
    if len(seq_a) == 0:
        return len(seq_b)
    if len(seq_b) == 0:
        return len(seq_a)
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a

    previous = list(range(len(seq_b) + 1))
    for i, item_a in enumerate(seq_a, start=1):
        current = [i]
        for j, item_b in enumerate(seq_b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (0 if item_a == item_b else 1)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def char_error_rate(expected: str, actual: str) -> float:
    if not expected and not actual:
        return 0.0
    return levenshtein(expected, actual) / max(len(expected), 1)


def word_error_rate(expected: str, actual: str) -> float:
    expected_words = expected.split()
    actual_words = actual.split()
    if not expected_words and not actual_words:
        return 0.0
    return levenshtein(expected_words, actual_words) / max(len(expected_words), 1)


def require_cases(sample_set: str) -> list[SampleCase]:
    sample_sets = available_sample_sets()
    if sample_set not in sample_sets:
        raise SystemExit(f"Unknown sample set: {sample_set}. Choose from: {', '.join(sorted(sample_sets))}")
    cases = sample_sets[sample_set]
    missing = [str(case.audio) for case in cases if not case.audio.exists()]
    if missing:
        raise SystemExit(
            "Missing sample files:\n- " + "\n- ".join(missing) +
            "\n\nMake sure the tracked fixtures are present under Tests/fixtures/ (and optional extended samples under ~/workspace/qwen-asr)."
        )
    return cases


def transcribe_reference(model_name: str, cases: list[SampleCase]) -> list[GoldenCase]:
    from mlx_audio.stt import load_model
    from mlx_audio.stt.generate import generate_transcription

    print(f"[validate-native-asr] Loading reference model: {model_name}", file=sys.stderr)
    load_t0 = time.perf_counter()
    wrapper = load_model(model_name)
    print(
        f"[validate-native-asr] Reference model loaded in {time.perf_counter() - load_t0:.1f}s",
        file=sys.stderr,
    )

    out: list[GoldenCase] = []
    for case in cases:
        print(f"[validate-native-asr] Reference → {case.id} ({case.audio.name})", file=sys.stderr)
        prev_cwd = Path.cwd()
        duration_s = case.duration_s if case.duration_s is not None else audio_duration_seconds(case.audio)
        t0 = time.perf_counter()
        try:
            os.chdir(tempfile.gettempdir())
            result = generate_transcription(wrapper, str(case.audio))
            text = result.text.strip()
        finally:
            os.chdir(prev_cwd)
        elapsed = time.perf_counter() - t0
        out.append(
            GoldenCase(
                id=case.id,
                audio=str(case.audio),
                language=case.language,
                notes=case.notes,
                duration_s=duration_s,
                processing_time_s=elapsed,
                text=text,
                normalized_text=normalize_text(text),
            )
        )
    return out


def write_golden_file(path: Path, sample_set: str, model_name: str, cases: list[GoldenCase]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "created_at": iso_now(),
        "sample_set": sample_set,
        "reference": {
            "type": "python-batch",
            "model": model_name,
            "implementation": "mlx_audio.stt.generate_transcription",
        },
        "normalization": {
            "lowercase": True,
            "collapse_whitespace": True,
            "strip_ascii_punctuation": True,
        },
        "cases": [asdict(case) for case in cases],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def load_golden_file(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def extract_candidate_text(stdout: str) -> str:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""


def prepare_candidate_audio(audio: Path) -> tuple[Path, Path | None]:
    if audio.suffix.lower() == ".wav":
        return audio, None

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = Path(tmp.name)
    tmp.close()
    result = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(audio),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed converting {audio}: {result.stderr[-500:]}")
    return tmp_path, tmp_path


def run_candidate_command(
    command_template: str,
    *,
    audio: Path,
    model_dir: str | None,
    language: str | None,
    case_id: str,
) -> tuple[str, float, str]:
    candidate_audio, cleanup_path = prepare_candidate_audio(audio)
    try:
        placeholders = {
            "audio": shlex.quote(str(candidate_audio)),
            "model_dir": shlex.quote(model_dir or ""),
            "language": shlex.quote(language or ""),
            "language_arg": f"--language {shlex.quote(language)}" if language else "--no-language",
            "id": case_id,
        }
        command = command_template.format(**placeholders)

        t0 = time.perf_counter()
        result = subprocess.run(
            ["bash", "-lc", command],
            capture_output=True,
            text=True,
        )
        elapsed = time.perf_counter() - t0

        if result.returncode != 0:
            raise RuntimeError(
                f"Candidate command failed for {case_id} (exit {result.returncode})\n"
                f"COMMAND: {command}\n"
                f"STDOUT:\n{result.stdout[-1000:]}\n"
                f"STDERR:\n{result.stderr[-2000:]}"
            )

        return extract_candidate_text(result.stdout), elapsed, result.stderr.strip()
    finally:
        if cleanup_path is not None:
            cleanup_path.unlink(missing_ok=True)


def extract_timing(stderr_text: str, label: str) -> float | None:
    match = re.search(rf"{re.escape(label)}\s*([0-9.]+)s", stderr_text)
    return float(match.group(1)) if match else None


def compare_against_goldens(
    golden: dict[str, Any],
    *,
    command_template: str,
    model_dir: str | None,
) -> list[CompareCase]:
    cases: list[CompareCase] = []
    for case in golden["cases"]:
        audio = Path(case["audio"])
        if not audio.exists():
            raise SystemExit(f"Golden audio file missing: {audio}")

        print(f"[validate-native-asr] Candidate → {case['id']} ({audio.name})", file=sys.stderr)
        candidate_text, candidate_time_s, stderr_text = run_candidate_command(
            command_template,
            audio=audio,
            model_dir=model_dir,
            language=case.get("language"),
            case_id=case["id"],
        )
        golden_normalized = case["normalized_text"]
        candidate_normalized = normalize_text(candidate_text)
        cases.append(
            CompareCase(
                id=case["id"],
                audio=case["audio"],
                duration_s=float(case["duration_s"]),
                golden_text=case["text"],
                candidate_text=candidate_text,
                golden_normalized=golden_normalized,
                candidate_normalized=candidate_normalized,
                raw_exact=case["text"] == candidate_text,
                normalized_exact=golden_normalized == candidate_normalized,
                cer=char_error_rate(golden_normalized, candidate_normalized),
                wer=word_error_rate(golden_normalized, candidate_normalized),
                candidate_time_s=candidate_time_s,
                load_time_s=extract_timing(stderr_text, "Model loaded in"),
                inference_time_s=extract_timing(stderr_text, "Inference:"),
                stderr_tail=stderr_text[-400:],
            )
        )
    return cases


def print_golden_summary(path: Path, payload: dict[str, Any]) -> None:
    print(f"Saved goldens → {path}")
    print(f"Reference model: {payload['reference']['model']}")
    print(f"Sample set: {payload['sample_set']}")
    print()
    for case in payload["cases"]:
        preview = case["text"].replace("\n", " ")
        if len(preview) > 96:
            preview = preview[:93] + "..."
        print(
            f"- {case['id']:<24} {case['duration_s']:>6.1f}s  "
            f"{case['processing_time_s']:>6.2f}s  {preview}"
        )


def print_compare_summary(cases: list[CompareCase]) -> bool:
    print(
        f"{'case':<24} {'dur':>6} {'cand':>7} {'infer':>7} {'raw':>5} {'norm':>6} {'cer':>7} {'wer':>7}"
    )
    print("-" * 82)
    all_ok = True
    for case in cases:
        raw = "yes" if case.raw_exact else "no"
        norm = "yes" if case.normalized_exact else "no"
        infer = f"{case.inference_time_s:.2f}s" if case.inference_time_s is not None else "n/a"
        all_ok &= case.normalized_exact
        print(
            f"{case.id:<24} {case.duration_s:>6.1f}s {case.candidate_time_s:>6.2f}s {infer:>7} "
            f"{raw:>5} {norm:>6} {case.cer:>6.3f} {case.wer:>6.3f}"
        )
        if not case.normalized_exact:
            print("    golden   :", case.golden_text)
            print("    candidate:", case.candidate_text)

    avg_total = sum(case.candidate_time_s for case in cases) / max(len(cases), 1)
    infer_values = [case.inference_time_s for case in cases if case.inference_time_s is not None]
    avg_infer = sum(infer_values) / len(infer_values) if infer_values else None
    print()
    print(f"avg total wall: {avg_total:.2f}s")
    if avg_infer is not None:
        print(f"avg native inference: {avg_infer:.2f}s")
    print()
    return all_ok


def command_golden(args: argparse.Namespace) -> int:
    cases = require_cases(args.sample_set)
    goldens = transcribe_reference(args.reference_model, cases)
    write_golden_file(args.output, args.sample_set, args.reference_model, goldens)
    payload = load_golden_file(args.output)
    print_golden_summary(args.output, payload)
    return 0


def command_compare(args: argparse.Namespace) -> int:
    golden = load_golden_file(args.golden)
    cases = compare_against_goldens(
        golden,
        command_template=args.candidate_cmd,
        model_dir=args.model_dir,
    )
    ok = print_compare_summary(cases)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "created_at": iso_now(),
            "golden_file": str(args.golden),
            "candidate_cmd": args.candidate_cmd,
            "model_dir": args.model_dir,
            "all_normalized_exact": ok,
            "cases": [asdict(case) for case in cases],
        }
        args.json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"Saved comparison → {args.json_out}")

    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validation harness for native Swift Qwen3-ASR")
    subparsers = parser.add_subparsers(dest="command", required=True)

    golden = subparsers.add_parser("golden", help="Generate / refresh golden outputs from Python batch reference")
    golden.add_argument("--sample-set", default="smoke", choices=sorted(available_sample_sets()))
    golden.add_argument("--reference-model", default=DEFAULT_REFERENCE_MODEL)
    golden.add_argument("--output", type=Path, default=DEFAULT_GOLDEN_PATH)
    golden.set_defaults(func=command_golden)

    compare = subparsers.add_parser("compare", help="Compare a candidate CLI against saved goldens")
    compare.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    compare.add_argument("--candidate-cmd", required=True,
                         help="Shell command template. Uses {audio}, {model_dir}, {language}, {language_arg}, {id} placeholders.")
    compare.add_argument("--model-dir", default=None,
                         help="Model directory substituted into {model_dir}. Optional if the command does not need it.")
    compare.add_argument("--json-out", type=Path, default=None)
    compare.set_defaults(func=command_compare)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
