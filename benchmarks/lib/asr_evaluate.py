#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = [
#   "mlx-audio>=0.2.2",
# ]
# ///
"""Ground-truth multilingual ASR evaluation for Yuwp and MLX-Audio.

The JSONL manifest has one object per line:
  {"id":"en_0001","audio":"clips/en_0001.wav","language":"en",
   "metric":"wer","reference":"the reference transcript"}

Audio paths are resolved relative to the manifest. English uses WER; Chinese and
Japanese use whitespace-independent CER. Results report corpus (micro) error
rates and standard RTF (wall time / audio duration; lower is better).
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from .batch_compare import (
    DEFAULT_MLX_MODEL,
    DEFAULT_YUWP_MODEL,
    YUWP_METALLIB,
    audio_duration_seconds,
    find_free_port,
    iso_now,
    load_mlx_audio_wrapper,
    model_label,
    multipart_request,
    require_path,
    resolve_yuwp_server_bin,
    run_yuwp_server,
    start_yuwp_server,
    stop_process,
)
from .transcript_metrics import ErrorRateAccumulator

LANGUAGE_NAMES = {"en": "English", "zh": "Chinese", "ja": "Japanese"}
EXPECTED_METRICS = {"en": "wer", "zh": "cer", "ja": "cer"}


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    audio: Path
    language: str
    metric: str
    reference: str


@dataclass
class EvaluationMeasurement:
    variant_id: str
    case_id: str
    language: str
    metric: str
    duration_s: float
    wall_s: float
    reference: str
    hypothesis: str
    errors: int
    reference_units: int

    @property
    def rtf(self) -> float:
        return self.wall_s / max(self.duration_s, 1e-9)


def load_manifest(path: Path) -> list[EvaluationCase]:
    path = path.expanduser().resolve()
    cases: list[EvaluationCase] = []
    seen_ids: set[str] = set()
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            item = json.loads(raw_line)
            case_id = str(item["id"])
            language = str(item["language"]).lower()
            metric = str(item["metric"]).lower()
            reference = str(item["reference"])
            audio = Path(item["audio"]).expanduser()
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid manifest line {line_number}: {error}") from error
        if language not in LANGUAGE_NAMES:
            raise ValueError(f"line {line_number}: unsupported language {language!r}")
        expected_metric = EXPECTED_METRICS[language]
        if metric != expected_metric:
            label = LANGUAGE_NAMES[language]
            raise ValueError(f"line {line_number}: {label} must use {expected_metric}, not {metric}")
        if case_id in seen_ids:
            raise ValueError(f"line {line_number}: duplicate id {case_id!r}")
        seen_ids.add(case_id)
        if not audio.is_absolute():
            audio = path.parent / audio
        audio = audio.resolve()
        if not audio.is_file():
            raise ValueError(f"line {line_number}: audio file not found: {audio}")
        cases.append(EvaluationCase(case_id, audio, language, metric, reference))
    if not cases:
        raise ValueError("manifest contains no cases")
    return cases


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize(measurements: list[EvaluationMeasurement]) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[EvaluationMeasurement]] = {}
    for row in measurements:
        grouped.setdefault((row.variant_id, row.language), []).append(row)

    result: dict[str, dict[str, dict[str, Any]]] = {}
    for (variant_id, language), rows in sorted(grouped.items()):
        errors = sum(row.errors for row in rows)
        reference_units = sum(row.reference_units for row in rows)
        total_audio = sum(row.duration_s for row in rows)
        total_wall = sum(row.wall_s for row in rows)
        rtfs = [row.rtf for row in rows]
        result.setdefault(variant_id, {})[language] = {
            "metric": rows[0].metric,
            "cases": len(rows),
            "errors": errors,
            "reference_units": reference_units,
            "error_rate": errors / max(reference_units, 1),
            "total_audio_s": total_audio,
            "total_wall_s": total_wall,
            "corpus_rtf": total_wall / max(total_audio, 1e-9),
            "median_utterance_rtf": statistics.median(rtfs),
            "p95_utterance_rtf": percentile(rtfs, 0.95),
            "median_wall_s": statistics.median(row.wall_s for row in rows),
            "p95_wall_s": percentile([row.wall_s for row in rows], 0.95),
        }
    return result


def score(metric: str, reference: str, hypothesis: str) -> tuple[int, int]:
    accumulator = ErrorRateAccumulator(metric)
    accumulator.add(reference, hypothesis)
    result = accumulator.summary()
    return int(result["errors"]), int(result["reference_units"])


def save_hypothesis(directory: Path | None, variant_id: str, case: EvaluationCase, text: str) -> None:
    if directory is None:
        return
    variant_directory = directory / variant_id.replace("/", "--")
    variant_directory.mkdir(parents=True, exist_ok=True)
    (variant_directory / f"{case.id}.txt").write_text(text + "\n")


def run_yuwp(
    variant_id: str,
    model_dir: Path,
    cases: list[EvaluationCase],
    repeat: int,
    save_text_dir: Path | None,
) -> tuple[list[EvaluationMeasurement], dict[str, Any]]:
    server = start_yuwp_server(model_dir, disable_vad=False)
    rows: list[EvaluationMeasurement] = []
    try:
        for repetition in range(1, repeat + 1):
            for index, case in enumerate(cases, start=1):
                print(
                    f"[evaluate] {variant_id} {index}/{len(cases)} r{repetition}/{repeat} {case.id}",
                    file=sys.stderr,
                )
                hypothesis, wall_s, _ = run_yuwp_server(
                    case.audio,
                    server,
                    language=LANGUAGE_NAMES[case.language],
                )
                duration_s = audio_duration_seconds(case.audio)
                errors, reference_units = score(case.metric, case.reference, hypothesis)
                save_hypothesis(save_text_dir, variant_id, case, hypothesis)
                rows.append(EvaluationMeasurement(
                    variant_id, case.id, case.language, case.metric, duration_s, wall_s,
                    case.reference, hypothesis, errors, reference_units,
                ))
    finally:
        stop_process(server.process)
    return rows, {"startup_s": server.startup_s, "log_path": server.log_path}


def run_whisper_cpp(
    variant_id: str,
    model_path: Path,
    server_bin: Path,
    cases: list[EvaluationCase],
    repeat: int,
    save_text_dir: Path | None,
) -> tuple[list[EvaluationMeasurement], dict[str, Any]]:
    port = find_free_port()
    log_file = tempfile.NamedTemporaryFile(prefix="whisper-cpp-evaluate-", suffix=".log", delete=False)
    log_file.close()
    started = time.perf_counter()
    log_handle = open(log_file.name, "w")
    process = subprocess.Popen(
        [str(server_bin), "--model", str(model_path), "--host", "127.0.0.1", "--port", str(port)],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        for _ in range(600):
            if process.poll() is not None:
                raise RuntimeError(f"whisper-server exited during startup; log: {log_file.name}")
            try:
                with urlopen(f"http://127.0.0.1:{port}/", timeout=1):
                    break
            except Exception:
                time.sleep(0.1)
        else:
            raise RuntimeError(f"whisper-server did not become ready; log: {log_file.name}")
        startup_s = time.perf_counter() - started
        rows: list[EvaluationMeasurement] = []
        for repetition in range(1, repeat + 1):
            for index, case in enumerate(cases, start=1):
                print(
                    f"[evaluate] {variant_id} {index}/{len(cases)} r{repetition}/{repeat} {case.id}",
                    file=sys.stderr,
                )
                request_started = time.perf_counter()
                response = multipart_request(
                    f"http://127.0.0.1:{port}/inference",
                    case.audio,
                    language=case.language,
                )
                wall_s = time.perf_counter() - request_started
                hypothesis = json.loads(response.decode())["text"].strip()
                duration_s = audio_duration_seconds(case.audio)
                errors, reference_units = score(case.metric, case.reference, hypothesis)
                save_hypothesis(save_text_dir, variant_id, case, hypothesis)
                rows.append(EvaluationMeasurement(
                    variant_id, case.id, case.language, case.metric, duration_s, wall_s,
                    case.reference, hypothesis, errors, reference_units,
                ))
        return rows, {"startup_s": startup_s, "log_path": log_file.name}
    finally:
        stop_process(process)
        log_handle.close()


def run_mlx_audio(
    variant_id: str,
    model_name: str,
    cases: list[EvaluationCase],
    repeat: int,
    save_text_dir: Path | None,
) -> tuple[list[EvaluationMeasurement], dict[str, Any]]:
    from mlx_audio.stt.generate import generate_transcription

    wrapper, startup_s = load_mlx_audio_wrapper(model_name)
    rows: list[EvaluationMeasurement] = []
    for repetition in range(1, repeat + 1):
        for index, case in enumerate(cases, start=1):
            print(
                f"[evaluate] {variant_id} {index}/{len(cases)} r{repetition}/{repeat} {case.id}",
                file=sys.stderr,
            )
            started = time.perf_counter()
            result = generate_transcription(
                wrapper,
                str(case.audio),
                output_path="/tmp/yuwp-asr-evaluate",
                language=case.language,
                verbose=False,
            )
            wall_s = time.perf_counter() - started
            hypothesis = result.text.strip()
            duration_s = audio_duration_seconds(case.audio)
            errors, reference_units = score(case.metric, case.reference, hypothesis)
            save_hypothesis(save_text_dir, variant_id, case, hypothesis)
            rows.append(EvaluationMeasurement(
                variant_id, case.id, case.language, case.metric, duration_s, wall_s,
                case.reference, hypothesis, errors, reference_units,
            ))
    return rows, {"startup_s": startup_s}


def print_report(summary: dict[str, dict[str, dict[str, Any]]], metadata: dict[str, Any]) -> None:
    print("\nGround-truth ASR evaluation (error/RTF: lower is better)\n")
    print("| Variant | Language | Cases | Metric | Error | Corpus RTF | Median RTF | P95 RTF | Startup s |")
    print("|---|---|---:|:---:|---:|---:|---:|---:|---:|")
    for variant_id, languages in summary.items():
        for language, row in languages.items():
            startup = metadata.get(variant_id, {}).get("startup_s")
            startup_text = "" if startup is None else f"{startup:.2f}"
            print(
                f"| {variant_id} | {language} | {row['cases']} | {row['metric'].upper()} | "
                f"{100 * row['error_rate']:.2f}% | {row['corpus_rtf']:.4f} | "
                f"{row['median_utterance_rtf']:.4f} | {row['p95_utterance_rtf']:.4f} | {startup_text} |"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground-truth multilingual ASR evaluation")
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL evaluation manifest")
    parser.add_argument("--tool", action="append", choices=["yuwp", "mlx-audio", "whisper-cpp"], required=True)
    parser.add_argument("--yuwp-model", type=Path, action="append")
    parser.add_argument("--mlx-model", action="append")
    parser.add_argument("--whisper-model", type=Path, action="append", help="whisper.cpp GGML model")
    parser.add_argument("--whisper-bin", type=Path, default=Path("/opt/homebrew/bin/whisper-server"))
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--save-text-dir", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.repeat < 1:
        raise SystemExit("--repeat must be at least 1")
    cases = load_manifest(args.manifest)
    variants: list[tuple[str, str, Path | str]] = []
    if "yuwp" in args.tool:
        require_path(resolve_yuwp_server_bin(), "Yuwp server binary")
        require_path(YUWP_METALLIB, "mlx.metallib")
        for model in args.yuwp_model or [DEFAULT_YUWP_MODEL]:
            model = require_path(model.expanduser().resolve(), "Yuwp model")
            variants.append((f"yuwp/{model_label(model)}", "yuwp", model))
    if "mlx-audio" in args.tool:
        for model in args.mlx_model or [DEFAULT_MLX_MODEL]:
            variants.append((f"mlx-audio/{model_label(model)}", "mlx-audio", model))
    if "whisper-cpp" in args.tool:
        if not args.whisper_model:
            raise SystemExit("--whisper-model is required with --tool whisper-cpp")
        require_path(args.whisper_bin.expanduser().resolve(), "whisper-server binary")
        for model in args.whisper_model:
            model = require_path(model.expanduser().resolve(), "whisper.cpp model")
            variants.append((f"whisper-cpp/{model_label(model)}", "whisper-cpp", model))

    print(f"Cases: {len(cases)}; variants: {len(variants)}; runs: {len(cases) * len(variants) * args.repeat}", file=sys.stderr)
    if args.dry_run:
        for variant_id, tool, model in variants:
            print(f"- {variant_id}: {tool} {model}")
        return 0

    measurements: list[EvaluationMeasurement] = []
    metadata: dict[str, Any] = {}
    for variant_id, tool, model in variants:
        if tool == "yuwp":
            rows, details = run_yuwp(variant_id, Path(model), cases, args.repeat, args.save_text_dir)
        elif tool == "whisper-cpp":
            rows, details = run_whisper_cpp(
                variant_id,
                Path(model),
                args.whisper_bin.expanduser().resolve(),
                cases,
                args.repeat,
                args.save_text_dir,
            )
        else:
            rows, details = run_mlx_audio(variant_id, str(model), cases, args.repeat, args.save_text_dir)
        measurements.extend(rows)
        metadata[variant_id] = details

    summary = summarize(measurements)
    print_report(summary, metadata)
    if args.json:
        payload = {
            "created_at": iso_now(),
            "manifest": str(args.manifest.expanduser().resolve()),
            "repeat": args.repeat,
            "metadata": metadata,
            "measurements": [asdict(row) | {"rtf": row.rtf} for row in measurements],
            "summary": summary,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n")
        print(f"Saved JSON → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
