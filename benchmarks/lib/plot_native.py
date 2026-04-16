#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = ["matplotlib"]
# ///
"""
Render comparison plots from native server benchmark JSON output.

Examples:
  uv run benchmarks/cli.py plot-native /tmp/yuwp-bench.json
  uv run benchmarks/cli.py plot-native /tmp/yuwp-bench.json --out-dir /tmp/yuwp-bench-plots
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_OUT_DIR = Path("/tmp/yuwp-native-bench-plots")
MODEL_COLORS = {
    "Qwen3-ASR-0.6B-4bit": "#2E86DE",
    "Qwen3-ASR-1.7B-bf16": "#E74C3C",
}
MODEL_LABEL_MAP = {
    "Qwen3-ASR-0.6B-4bit": "Small (0.6B-4bit)",
    "Qwen3-ASR-1.7B-bf16": "Large (1.7B-bf16)",
}
CONCURRENCY_MARKERS = {1: "o", 2: "s", 4: "^", 8: "D"}


def load_results(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if "results" not in payload:
        raise ValueError("expected JSON from the native server benchmark with top-level 'results'")
    return payload


def model_display_name(label: str) -> str:
    return MODEL_LABEL_MAP.get(label, label)


def model_color(label: str) -> str:
    return MODEL_COLORS.get(label, None) or f"C{abs(hash(label)) % 10}"


def ensure_out_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_fig(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def sorted_concurrency(results: list[dict[str, Any]]) -> list[int]:
    values = {run["concurrency"] for model in results for run in model["runs"]}
    return sorted(values)


def plot_concurrency_metrics(results: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    metrics = [
        ("avg_chunk_ms", "Avg chunk latency (ms)"),
        ("p95_chunk_ms", "P95 chunk latency (ms)"),
        ("avg_first_nonempty_ms", "First non-empty partial (ms)"),
        ("avg_realtime_factor", "Realtime factor"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    conc = sorted_concurrency(results)

    for ax, (metric, title) in zip(axes.flat, metrics):
        for model in results:
            xs: list[int] = []
            ys: list[float] = []
            for run in sorted(model["runs"], key=lambda r: r["concurrency"]):
                value = run["summary"].get(metric)
                if value is None:
                    continue
                xs.append(run["concurrency"])
                ys.append(value)
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=model_color(model["label"]),
                label=model_display_name(model["label"]),
            )
        ax.set_title(title)
        ax.set_xlabel("Concurrent clients")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(conc)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Native ASR latency vs concurrent clients", fontsize=14)
    out = out_dir / "concurrency-latency.png"
    save_fig(fig, out)
    return [out]


def plot_concurrency_resource_metrics(results: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    metrics = [
        ("rss_before_mb", "RSS before workload (MB)"),
        ("peak_rss_mb", "Peak RSS during workload (MB)"),
        ("peak_delta_mb", "Peak RSS increase (MB)"),
        ("throughput_x", "Throughput (audio seconds / wall second)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    conc = sorted_concurrency(results)

    for ax, (metric, title) in zip(axes.flat, metrics):
        for model in results:
            xs: list[int] = []
            ys: list[float] = []
            for run in sorted(model["runs"], key=lambda r: r["concurrency"]):
                value = run["summary"].get(metric)
                if value is None:
                    continue
                xs.append(run["concurrency"])
                ys.append(value)
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=model_color(model["label"]),
                label=model_display_name(model["label"]),
            )
        ax.set_title(title)
        ax.set_xlabel("Concurrent clients")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(conc)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Native ASR resource usage vs concurrent clients", fontsize=14)
    out = out_dir / "concurrency-resource.png"
    save_fig(fig, out)
    return [out]


def plot_quality_metrics(results: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    metrics = [
        ("stream_similarity_mean", "Mean stream similarity"),
        ("final_similarity_mean", "Mean final similarity"),
        ("stream_exact_norm", "Exact normalized stream matches"),
        ("final_exact_norm", "Exact normalized final matches"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    conc = sorted_concurrency(results)

    for ax, (metric, title) in zip(axes.flat, metrics):
        for model in results:
            xs: list[int] = []
            ys: list[float] = []
            for run in sorted(model["runs"], key=lambda r: r["concurrency"]):
                value = run["summary"].get(metric)
                if value is None:
                    continue
                xs.append(run["concurrency"])
                ys.append(value)
            ax.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                color=model_color(model["label"]),
                label=model_display_name(model["label"]),
            )
        ax.set_title(title)
        ax.set_xlabel("Concurrent clients")
        ax.grid(True, alpha=0.25)
        ax.set_xticks(conc)
    axes[0, 0].legend(loc="best")
    fig.suptitle("Transcript quality vs concurrent clients", fontsize=14)
    out = out_dir / "concurrency-quality.png"
    save_fig(fig, out)
    return [out]


def plot_duration_scatter(results: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    conc_values = sorted_concurrency(results)
    cols = len(conc_values)
    fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 8), squeeze=False)

    for col, conc in enumerate(conc_values):
        ax_top = axes[0, col]
        ax_bottom = axes[1, col]
        for model in results:
            run = next((r for r in model["runs"] if r["concurrency"] == conc), None)
            if run is None:
                continue
            sessions = run["sessions"]
            xs = [s["duration_s"] for s in sessions]
            avg_chunk = [s["avg_chunk_ms"] for s in sessions]
            total_time = [s["total_time_s"] for s in sessions]
            ax_top.scatter(
                xs,
                avg_chunk,
                alpha=0.75,
                s=40,
                label=model_display_name(model["label"]),
                color=model_color(model["label"]),
            )
            ax_bottom.scatter(
                xs,
                total_time,
                alpha=0.75,
                s=40,
                label=model_display_name(model["label"]),
                color=model_color(model["label"]),
            )

        ax_top.set_title(f"Concurrency {conc}: latency by audio length")
        ax_top.set_xlabel("Audio duration (s)")
        ax_top.set_ylabel("Avg chunk latency (ms)")
        ax_top.grid(True, alpha=0.25)

        ax_bottom.set_title(f"Concurrency {conc}: wall time by audio length")
        ax_bottom.set_xlabel("Audio duration (s)")
        ax_bottom.set_ylabel("Session wall time (s)")
        ax_bottom.grid(True, alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
    fig.suptitle("Per-file impact of audio length and concurrency", fontsize=14, y=1.02)
    out = out_dir / "duration-scatter.png"
    save_fig(fig, out)
    return [out]


def plot_duration_line(results: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    conc_values = sorted_concurrency(results)
    fig, axes = plt.subplots(1, len(conc_values), figsize=(5 * len(conc_values), 4), squeeze=False)

    for idx, conc in enumerate(conc_values):
        ax = axes[0, idx]
        for model in results:
            run = next((r for r in model["runs"] if r["concurrency"] == conc), None)
            if run is None:
                continue
            sessions = sorted(run["sessions"], key=lambda s: (s["duration_s"], s["file"]))
            ax.plot(
                [s["duration_s"] for s in sessions],
                [s["realtime_factor"] for s in sessions],
                marker=CONCURRENCY_MARKERS.get(conc, "o"),
                linewidth=1.8,
                label=model_display_name(model["label"]),
                color=model_color(model["label"]),
            )
        ax.set_title(f"Concurrency {conc}")
        ax.set_xlabel("Audio duration (s)")
        ax.set_ylabel("Realtime factor")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4))
    fig.suptitle("Per-file speed vs audio length", fontsize=14, y=1.05)
    out = out_dir / "duration-realtime-factor.png"
    save_fig(fig, out)
    return [out]


def write_index(results_path: Path, images: list[Path], out_dir: Path, payload: dict[str, Any]) -> Path:
    cases = payload.get("cases", [])
    html = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<title>Yuwp Native ASR Benchmark Plots</title>",
        "<style>",
        "body{font:14px -apple-system,BlinkMacSystemFont,sans-serif;margin:24px;line-height:1.4}",
        "img{max-width:100%;border:1px solid #ddd;border-radius:8px;margin:12px 0 32px}",
        "code{background:#f6f8fa;padding:2px 4px;border-radius:4px}",
        "</style>",
        "<h1>Yuwp Native ASR Benchmark Plots</h1>",
        f"<p>Source JSON: <code>{results_path}</code></p>",
        f"<p>Cases: {len(cases)}</p>",
    ]
    if cases:
        html.append("<ul>")
        for case in cases:
            html.append(f"<li><code>{case}</code></li>")
        html.append("</ul>")
    for image in images:
        html.append(f"<h2>{image.stem}</h2>")
        html.append(f"<img src='{image.name}' alt='{image.stem}'>")
    out = out_dir / "index.html"
    out.write_text("\n".join(html))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot results from the native server benchmark")
    parser.add_argument("json", help="Path to benchmark JSON from `benchmarks/cli.py server-load`")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help=f"Output directory (default: {DEFAULT_OUT_DIR})")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.json).expanduser().resolve()
    out_dir = ensure_out_dir(Path(args.out_dir).expanduser())
    payload = load_results(results_path)
    results = payload["results"]

    images: list[Path] = []
    images += plot_concurrency_metrics(results, out_dir)
    images += plot_concurrency_resource_metrics(results, out_dir)
    images += plot_quality_metrics(results, out_dir)
    images += plot_duration_scatter(results, out_dir)
    images += plot_duration_line(results, out_dir)
    index = write_index(results_path, images, out_dir, payload)

    print(f"Wrote plots to {out_dir}")
    print(f"Open {index}")
    for image in images:
        print(f"  - {image}")


if __name__ == "__main__":
    main()
