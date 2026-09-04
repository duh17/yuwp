#!/usr/bin/env -S uv run --python 3.14 --script
"""Materialize local-only long-form benchmark data from a pinned source plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = REPO_ROOT / "benchmarks" / "fixtures" / "subtitle-external" / "benchmark-plan.json"
EXPECTED_NLP_HEADER = ["token", "speaker", "ts", "endTs", "punctuation", "case", "tags", "wer_tags"]
DOWNLOAD_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class ParsedTranscript:
    text: str
    spoken_tokens: int
    excluded_tokens: dict[str, int]


def load_benchmark_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text())
    if plan.get("schema_version") != 1:
        raise ValueError(f"unsupported benchmark plan schema: {plan.get('schema_version')!r}")
    earnings = plan.get("earnings21")
    if not isinstance(earnings, dict) or not isinstance(earnings.get("cases"), list):
        raise ValueError("benchmark plan is missing earnings21 cases")
    return plan


def parse_earnings_nlp(path: Path) -> ParsedTranscript:
    spoken: list[str] = []
    excluded: Counter[str] = Counter()
    with path.open(newline="") as handle:
        reader = csv.reader(handle, delimiter="|")
        try:
            header = next(reader)
        except StopIteration as error:
            raise ValueError(f"empty Earnings-21 reference: {path}") from error
        if header != EXPECTED_NLP_HEADER:
            raise ValueError(f"unexpected Earnings-21 NLP header in {path}: {header}")
        for line_number, row in enumerate(reader, start=2):
            if len(row) != len(EXPECTED_NLP_HEADER):
                raise ValueError(f"invalid Earnings-21 NLP row {path}:{line_number}: {len(row)} fields")
            token = row[0].strip()
            if not token:
                continue
            if token.startswith("<") and token.endswith(">"):
                excluded[token] += 1
                continue
            spoken.append(token)
    return ParsedTranscript(
        text=" ".join(spoken),
        spoken_tokens=len(spoken),
        excluded_tokens=dict(sorted(excluded.items())),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(DOWNLOAD_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_asset(path: Path, *, expected_size: int, expected_sha256: str) -> None:
    if not path.is_file():
        raise ValueError(f"missing asset: {path}")
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(f"size mismatch for {path}: expected {expected_size}, got {actual_size}")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise ValueError(f"sha256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}")


def download_asset(url: str, destination: Path, *, expected_size: int, expected_sha256: str) -> None:
    if destination.is_file():
        try:
            verify_asset(destination, expected_size=expected_size, expected_sha256=expected_sha256)
            return
        except ValueError:
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    request = urllib.request.Request(url, headers={"User-Agent": "Yuwp benchmark materializer"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response, partial.open("wb") as output:
            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                output.write(chunk)
        verify_asset(partial, expected_size=expected_size, expected_sha256=expected_sha256)
        os.replace(partial, destination)
    except Exception:
        partial.unlink(missing_ok=True)
        raise


def _local_paths(root: Path, case: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    case_id = str(case["id"])
    return (
        root / "audio" / f"{case_id}.mp3",
        root / "source-nlp" / f"{case_id}.nlp",
        root / "references" / f"{case_id}.txt",
        root / "references" / f"{case_id}.metadata.json",
    )


def _raw_url(repository: str, revision: str, source_path: str) -> str:
    owner_and_repo = repository.removeprefix("https://github.com/").rstrip("/")
    return f"https://raw.githubusercontent.com/{owner_and_repo}/{revision}/earnings21/{source_path}"


def build_earnings_manifest(plan: dict[str, Any], root: Path) -> dict[str, Any]:
    split_rows: dict[str, list[dict[str, Any]]] = {"dev": [], "heldout": []}
    split_summary: dict[str, dict[str, Any]] = {
        "dev": {"case_ids": [], "duration_seconds": 0.0, "source_tokens": 0},
        "heldout": {"case_ids": [], "duration_seconds": 0.0, "source_tokens": 0},
    }

    for case in plan["cases"]:
        split = str(case["split"])
        if split not in split_rows:
            raise ValueError(f"unsupported Earnings-21 split: {split}")
        case_id = str(case["id"])
        audio_path, _, reference_path, metadata_path = _local_paths(root, case)
        if not audio_path.is_file() or not reference_path.is_file():
            raise ValueError(f"case {case_id} is not materialized under {root}")
        reference = reference_path.read_text().strip()
        if not reference:
            raise ValueError(f"case {case_id} has an empty reference")
        row = {
            "id": f"earnings21-{case_id}",
            "audio": str(audio_path.relative_to(root)),
            "language": "en",
            "metric": "wer",
            "reference": reference,
        }
        split_rows[split].append(row)
        details = split_summary[split]
        details["case_ids"].append(case_id)
        details["duration_seconds"] += float(case["duration_seconds"])
        if metadata_path.is_file():
            details["source_tokens"] += int(json.loads(metadata_path.read_text())["spoken_tokens"])
        else:
            details["source_tokens"] += len(reference.split())

    dev_ids = set(split_summary["dev"]["case_ids"])
    heldout_ids = set(split_summary["heldout"]["case_ids"])
    if not dev_ids or not heldout_ids or not dev_ids.isdisjoint(heldout_ids):
        raise ValueError("Earnings-21 dev and heldout cases must be non-empty and disjoint")

    for split, rows in split_rows.items():
        manifest = root / f"asr-{split}.jsonl"
        manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    summary = {
        "schema_version": 1,
        "source": plan["source"],
        "normalization": {
            "profile": "spoken-v1",
            "excluded_reference_tokens": "angle-bracket placeholders such as <inaudible> and <crosstalk>",
        },
        "splits": split_summary,
    }
    (root / "dataset.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary


def prepare_earnings21(plan: dict[str, Any], root: Path, *, download: bool) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    source = plan["source"]
    repository = str(source["repository"])
    revision = str(source["revision"])

    for case in plan["cases"]:
        case_id = str(case["id"])
        audio_path, nlp_path, reference_path, metadata_path = _local_paths(root, case)
        for kind, destination in (("audio", audio_path), ("reference", nlp_path)):
            asset = case[kind]
            if download:
                download_asset(
                    _raw_url(repository, revision, str(asset["path"])),
                    destination,
                    expected_size=int(asset["size"]),
                    expected_sha256=str(asset["sha256"]),
                )
            else:
                verify_asset(
                    destination,
                    expected_size=int(asset["size"]),
                    expected_sha256=str(asset["sha256"]),
                )
        parsed = parse_earnings_nlp(nlp_path)
        reference_path.parent.mkdir(parents=True, exist_ok=True)
        reference_path.write_text(parsed.text + "\n")
        metadata_path.write_text(json.dumps({
            "case_id": case_id,
            "spoken_tokens": parsed.spoken_tokens,
            "excluded_tokens": parsed.excluded_tokens,
        }, indent=2) + "\n")

    return build_earnings_manifest(plan, root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    subparsers = parser.add_subparsers(dest="dataset", required=True)

    earnings = subparsers.add_parser("earnings21", help="materialize the pinned Earnings-21 dev/heldout cases")
    earnings.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT / "benchmarks" / "data" / "long-subtitle" / "earnings21",
    )
    earnings.add_argument("--check-only", action="store_true", help="verify local assets without network access")

    subparsers.add_parser("must-cinema-status", help="print the unresolved MuST-Cinema access requirements")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plan = load_benchmark_plan(args.plan.expanduser().resolve())
    if args.dataset == "must-cinema-status":
        print(json.dumps(plan["must_cinema"], indent=2))
        return 0

    root = args.root.expanduser().resolve()
    summary = prepare_earnings21(plan["earnings21"], root, download=not args.check_only)
    print(json.dumps({"root": str(root), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
