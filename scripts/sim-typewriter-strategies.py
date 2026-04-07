#!/usr/bin/env -S uv run --python 3.14 --script
# /// script
# requires-python = ">=3.14"
# dependencies = []
# ///
"""
Simulate different TypewriterAnimator strategies against real partial data.

Replays recorded streaming partials (from log-streaming-partials.py) through
multiple animation strategies and compares UX quality metrics:
  - snap_events: how many times text visually jumps (bad)
  - snap_chars: total characters that snap instead of animate (bad)
  - animate_chars: total characters smoothly animated (good)
  - stale_chars: characters hidden from display (latency cost)
  - final_accuracy: does the strategy produce the correct final text?

Strategies:
  A. BASELINE — current TypewriterAnimator behavior
  B. PERIOD_MERGE — treat "remove trailing punct + append" as pure append
  C. HOLD_BACK — hold back N trailing chars until confirmed by next chunk
  D. BATCH_DAMPEN — on large corrections (>20 chars), word-diff and only
     snap the actual changed words
  E. COMBINED — period_merge + hold_back(3) + batch_dampen

Usage:
  uv run scripts/sim-typewriter-strategies.py
  uv run scripts/sim-typewriter-strategies.py --partials results/partials-analysis.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path


TRAILING_PUNCT = set(".!?。！？")


def common_prefix_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def word_diff_snap_chars(old: str, new: str) -> int:
    """Count chars that actually changed between old and new using word-level diff.

    Returns the number of characters the user would see 'jump' if we do a
    word-level snap instead of full-string snap.
    """
    old_words = old.split()
    new_words = new.split()

    # Find common prefix words
    prefix_words = 0
    for i in range(min(len(old_words), len(new_words))):
        if old_words[i] != new_words[i]:
            break
        prefix_words += 1

    # Find common suffix words
    suffix_words = 0
    for i in range(1, min(len(old_words), len(new_words)) - prefix_words + 1):
        if old_words[-i] != new_words[-i]:
            break
        suffix_words += 1

    # Changed words in old text
    changed_old = old_words[prefix_words:len(old_words) - suffix_words if suffix_words else len(old_words)]
    changed_new = new_words[prefix_words:len(new_words) - suffix_words if suffix_words else len(new_words)]

    # Snap chars = chars in the changed region of the OLD text that get replaced
    snap = sum(len(w) + 1 for w in changed_old)  # +1 for space
    return max(0, snap - 1)  # remove trailing space


@dataclass
class AnimatorState:
    """Tracks what the user sees at each step."""
    display: str = ""
    target: str = ""


@dataclass
class StepResult:
    chunk_idx: int
    new_text: str
    display_before: str
    display_after: str
    snap_chars: int       # chars that visually jump
    animate_chars: int    # chars smoothly animated
    is_disruption: bool   # True if user sees text change non-monotonically


@dataclass
class StrategyResult:
    name: str
    steps: list[StepResult] = field(default_factory=list)

    @property
    def snap_events(self) -> int:
        return sum(1 for s in self.steps if s.snap_chars > 0)

    @property
    def total_snap_chars(self) -> int:
        return sum(s.snap_chars for s in self.steps)

    @property
    def total_animate_chars(self) -> int:
        return sum(s.animate_chars for s in self.steps)

    @property
    def disruptions(self) -> int:
        return sum(1 for s in self.steps if s.is_disruption)

    @property
    def max_snap(self) -> int:
        return max((s.snap_chars for s in self.steps), default=0)

    @property
    def large_snaps(self) -> int:
        return sum(1 for s in self.steps if s.snap_chars > 10)


# ---------------------------------------------------------------------------
# Strategy A: BASELINE (current TypewriterAnimator)
# ---------------------------------------------------------------------------

def strategy_baseline(partials: list[dict]) -> StrategyResult:
    result = StrategyResult(name="A. BASELINE (current)")
    state = AnimatorState()

    for p in partials:
        new_text = p["text"]
        old_display = state.display
        old_target = state.target

        cp = common_prefix_len(old_target, new_text)
        removed = len(old_target) - cp
        is_correction = removed > 0

        if is_correction:
            # Snap corrected portion, animate only truly new chars beyond old length
            snap_to = min(len(old_target), len(new_text))
            snap_chars = removed
            animate_chars = max(0, len(new_text) - len(old_target))
        else:
            snap_chars = 0
            animate_chars = len(new_text) - cp

        state.target = new_text
        state.display = new_text  # after animation completes

        result.steps.append(StepResult(
            chunk_idx=p["chunk"],
            new_text=new_text,
            display_before=old_display,
            display_after=new_text,
            snap_chars=snap_chars,
            animate_chars=animate_chars,
            is_disruption=is_correction,
        ))

    return result


# ---------------------------------------------------------------------------
# Strategy B: PERIOD_MERGE
# Treat "remove trailing punctuation + append" as a pure append.
# The period was speculative; removing it and adding more text is natural.
# ---------------------------------------------------------------------------

def strategy_period_merge(partials: list[dict]) -> StrategyResult:
    result = StrategyResult(name="B. PERIOD_MERGE")
    state = AnimatorState()

    for p in partials:
        new_text = p["text"]
        old_target = state.target

        cp = common_prefix_len(old_target, new_text)
        removed = len(old_target) - cp
        removed_text = old_target[cp:] if removed > 0 else ""

        # Check if removal is just trailing punctuation
        is_punct_only = removed > 0 and all(c in TRAILING_PUNCT for c in removed_text.strip())

        if is_punct_only and len(new_text) > len(old_target):
            # Treat as pure append from the common prefix point
            snap_chars = 0
            animate_chars = len(new_text) - cp
            is_correction = False
        elif removed > 0:
            snap_chars = removed
            animate_chars = max(0, len(new_text) - len(old_target))
            is_correction = True
        else:
            snap_chars = 0
            animate_chars = len(new_text) - cp
            is_correction = False

        state.target = new_text
        state.display = new_text

        result.steps.append(StepResult(
            chunk_idx=p["chunk"],
            new_text=new_text,
            display_before=state.display,
            display_after=new_text,
            snap_chars=snap_chars,
            animate_chars=animate_chars,
            is_disruption=is_correction,
        ))

    return result


# ---------------------------------------------------------------------------
# Strategy C: HOLD_BACK
# Don't display the last N characters until confirmed by the next chunk.
# This absorbs small rollbacks (<= N chars) completely.
# ---------------------------------------------------------------------------

def strategy_hold_back(partials: list[dict], hold: int = 3) -> StrategyResult:
    result = StrategyResult(name=f"C. HOLD_BACK({hold})")
    confirmed = ""   # text confirmed by seeing next chunk extend it
    pending = ""     # last `hold` chars, not yet displayed

    for p in partials:
        new_text = p["text"]

        # How much of the old confirmed+pending text survives?
        old_full = confirmed + pending
        cp = common_prefix_len(old_full, new_text)

        # Everything up to cp is confirmed by this chunk
        # New pending = last `hold` chars of new_text
        if len(new_text) > hold:
            new_confirmed = new_text[:-hold]
            new_pending = new_text[-hold:]
        else:
            new_confirmed = ""
            new_pending = new_text

        # Compute display change
        old_display = confirmed  # what user was seeing
        new_display = new_confirmed  # what user will see

        display_cp = common_prefix_len(old_display, new_display)
        display_removed = len(old_display) - display_cp

        if display_removed > 0:
            snap_chars = display_removed
            animate_chars = max(0, len(new_display) - len(old_display))
            is_correction = True
        else:
            snap_chars = 0
            animate_chars = len(new_display) - display_cp
            is_correction = False

        result.steps.append(StepResult(
            chunk_idx=p["chunk"],
            new_text=new_text,
            display_before=old_display,
            display_after=new_display,
            snap_chars=snap_chars,
            animate_chars=animate_chars,
            is_disruption=is_correction,
        ))

        confirmed = new_confirmed
        pending = new_pending

    return result


# ---------------------------------------------------------------------------
# Strategy D: BATCH_DAMPEN
# On large corrections (>20 chars changed), do word-level diff and only
# snap the actually-changed words. The rest appears stable.
# ---------------------------------------------------------------------------

def strategy_batch_dampen(partials: list[dict]) -> StrategyResult:
    result = StrategyResult(name="D. BATCH_DAMPEN")
    state = AnimatorState()

    for p in partials:
        new_text = p["text"]
        old_target = state.target

        cp = common_prefix_len(old_target, new_text)
        removed = len(old_target) - cp

        if removed > 20:
            # Large correction — use word-level diff
            snap_chars = word_diff_snap_chars(old_target, new_text)
            animate_chars = max(0, len(new_text) - len(old_target))
            is_correction = snap_chars > 0
        elif removed > 0:
            snap_chars = removed
            animate_chars = max(0, len(new_text) - len(old_target))
            is_correction = True
        else:
            snap_chars = 0
            animate_chars = len(new_text) - cp
            is_correction = False

        state.target = new_text
        state.display = new_text

        result.steps.append(StepResult(
            chunk_idx=p["chunk"],
            new_text=new_text,
            display_before=state.display,
            display_after=new_text,
            snap_chars=snap_chars,
            animate_chars=animate_chars,
            is_disruption=is_correction,
        ))

    return result


# ---------------------------------------------------------------------------
# Strategy E: COMBINED (period_merge + hold_back(3) + batch_dampen)
# ---------------------------------------------------------------------------

def strategy_combined(partials: list[dict], hold: int = 3) -> StrategyResult:
    result = StrategyResult(name=f"E. COMBINED (merge+hold({hold})+dampen)")
    confirmed = ""
    pending = ""

    for p in partials:
        new_text = p["text"]

        # Hold-back: compute confirmed display
        if len(new_text) > hold:
            new_confirmed = new_text[:-hold]
            new_pending = new_text[-hold:]
        else:
            new_confirmed = ""
            new_pending = new_text

        old_display = confirmed
        new_display = new_confirmed

        display_cp = common_prefix_len(old_display, new_display)
        display_removed = len(old_display) - display_cp
        display_removed_text = old_display[display_cp:] if display_removed > 0 else ""

        # Period merge: trailing punct removal is not a disruption
        is_punct_only = (display_removed > 0 and
                         all(c in TRAILING_PUNCT for c in display_removed_text.strip()) and
                         len(new_display) >= len(old_display))

        if is_punct_only:
            snap_chars = 0
            animate_chars = len(new_display) - display_cp
            is_correction = False
        elif display_removed > 20:
            # Batch dampen: word-level diff for large corrections
            snap_chars = word_diff_snap_chars(old_display, new_display)
            animate_chars = max(0, len(new_display) - len(old_display))
            is_correction = snap_chars > 0
        elif display_removed > 0:
            snap_chars = display_removed
            animate_chars = max(0, len(new_display) - len(old_display))
            is_correction = True
        else:
            snap_chars = 0
            animate_chars = len(new_display) - display_cp
            is_correction = False

        result.steps.append(StepResult(
            chunk_idx=p["chunk"],
            new_text=new_text,
            display_before=old_display,
            display_after=new_display,
            snap_chars=snap_chars,
            animate_chars=animate_chars,
            is_disruption=is_correction,
        ))

        confirmed = new_confirmed
        pending = new_pending

    return result


# ---------------------------------------------------------------------------
# Run simulation
# ---------------------------------------------------------------------------

def run_strategies(partials: list[dict]) -> list[StrategyResult]:
    return [
        strategy_baseline(partials),
        strategy_period_merge(partials),
        strategy_hold_back(partials, hold=3),
        strategy_hold_back(partials, hold=6),
        strategy_batch_dampen(partials),
        strategy_combined(partials, hold=3),
        strategy_combined(partials, hold=6),
    ]


def print_file_results(fname: str, duration: float, strategies: list[StrategyResult]) -> None:
    print(f"\n{'='*90}")
    print(f"  {fname} ({duration:.1f}s)")
    print(f"{'='*90}")

    # Header
    label_w = 40
    col_w = 10
    header = f"{'':>{label_w}}"
    for s in strategies:
        short = s.name.split(".")[0] + "." + s.name.split(".")[-1][:8]
        header += f"  {short:>{col_w}}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Snap events (disruptions)", [str(s.snap_events) for s in strategies]),
        ("Total snap chars", [str(s.total_snap_chars) for s in strategies]),
        ("Max single snap", [str(s.max_snap) for s in strategies]),
        ("Large snaps (>10 chars)", [str(s.large_snaps) for s in strategies]),
        ("Total animate chars", [str(s.total_animate_chars) for s in strategies]),
        ("Disruption rate", [f"{s.disruptions/max(len(s.steps),1)*100:.0f}%" for s in strategies]),
    ]

    for label, vals in rows:
        line = f"{label:>{label_w}}"
        for v in vals:
            line += f"  {v:>{col_w}}"
        print(line)


def print_detailed_comparison(fname: str, strategies: list[StrategyResult]) -> None:
    """Show chunk-by-chunk comparison for files with interesting differences."""
    baseline = strategies[0]
    # Only show chunks where strategies differ
    interesting = []
    for i, step in enumerate(baseline.steps):
        if step.snap_chars > 0:
            interesting.append(i)

    if not interesting:
        return

    print(f"\n  Correction details for {fname}:")
    for idx in interesting:
        b = baseline.steps[idx]
        print(f"\n    chunk {b.chunk_idx}: \"{b.new_text[:60]}{'...' if len(b.new_text) > 60 else ''}\"")
        for s in strategies:
            step = s.steps[idx]
            marker = " ***" if step.snap_chars != b.snap_chars else ""
            short = s.name.split(". ")[1][:20] if ". " in s.name else s.name[:20]
            print(f"      {short:>22}: snap={step.snap_chars:>3}  animate={step.animate_chars:>3}  disruption={'Y' if step.is_disruption else 'N'}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate typewriter strategies")
    parser.add_argument("--partials", default=None, help="Path to partials JSON")
    parser.add_argument("--verbose", action="store_true", help="Show per-chunk details")
    args = parser.parse_args()

    # Load partials
    if args.partials:
        path = args.partials
    else:
        results_dir = Path(__file__).resolve().parent.parent / "results"
        path = str(results_dir / "partials-analysis.json")

    if not os.path.exists(path):
        print(f"Partials file not found: {path}")
        print("Run log-streaming-partials.py first")
        sys.exit(1)

    with open(path) as f:
        data = json.load(f)

    print("Simulating 7 TypewriterAnimator strategies against recorded partials\n")
    print("Strategies:")
    print("  A. BASELINE     — current behavior (snap corrections, animate appends)")
    print("  B. PERIOD_MERGE — trailing punct removal treated as append, not correction")
    print("  C3. HOLD_BACK(3) — hide last 3 chars until confirmed by next chunk")
    print("  C6. HOLD_BACK(6) — hide last 6 chars until confirmed")
    print("  D. BATCH_DAMPEN — word-level diff on large (>20 char) corrections")
    print("  E3. COMBINED(3) — period_merge + hold_back(3) + batch_dampen")
    print("  E6. COMBINED(6) — period_merge + hold_back(6) + batch_dampen")

    # Run per file
    all_strategies: list[list[StrategyResult]] = []
    for entry in data:
        partials = entry["partials"]
        strategies = run_strategies(partials)
        all_strategies.append(strategies)
        print_file_results(entry["file"], entry["duration_s"], strategies)
        if args.verbose:
            print_detailed_comparison(entry["file"], strategies)

    # Global summary
    print(f"\n{'='*90}")
    print("GLOBAL TOTALS")
    print(f"{'='*90}")

    n_strategies = len(all_strategies[0]) if all_strategies else 0
    totals: list[dict] = []
    for si in range(n_strategies):
        total = {
            "name": all_strategies[0][si].name,
            "snap_events": 0,
            "snap_chars": 0,
            "max_snap": 0,
            "large_snaps": 0,
            "animate_chars": 0,
            "disruptions": 0,
            "steps": 0,
        }
        for file_strats in all_strategies:
            s = file_strats[si]
            total["snap_events"] += s.snap_events
            total["snap_chars"] += s.total_snap_chars
            total["max_snap"] = max(total["max_snap"], s.max_snap)
            total["large_snaps"] += s.large_snaps
            total["animate_chars"] += s.total_animate_chars
            total["disruptions"] += s.disruptions
            total["steps"] += len(s.steps)
        totals.append(total)

    label_w = 40
    col_w = 10
    header = f"{'':>{label_w}}"
    for t in totals:
        short = t["name"].split(".")[0] + "." + t["name"].split(".")[-1][:8]
        header += f"  {short:>{col_w}}"
    print(header)
    print("-" * len(header))

    rows = [
        ("Snap events", [str(t["snap_events"]) for t in totals]),
        ("Total snap chars", [str(t["snap_chars"]) for t in totals]),
        ("Max single snap", [str(t["max_snap"]) for t in totals]),
        ("Large snaps (>10 chars)", [str(t["large_snaps"]) for t in totals]),
        ("Total animate chars", [str(t["animate_chars"]) for t in totals]),
        ("Disruption rate", [f"{t['disruptions']/max(t['steps'],1)*100:.0f}%" for t in totals]),
    ]

    baseline_snaps = totals[0]["snap_events"]
    baseline_chars = totals[0]["snap_chars"]

    for label, vals in rows:
        line = f"{label:>{label_w}}"
        for v in vals:
            line += f"  {v:>{col_w}}"
        print(line)

    # Improvement summary
    print(f"\n  Improvement vs BASELINE:")
    for t in totals[1:]:
        snap_reduction = baseline_snaps - t["snap_events"]
        char_reduction = baseline_chars - t["snap_chars"]
        pct = 100 * char_reduction / max(baseline_chars, 1)
        print(f"    {t['name']:45s}  snap events: {snap_reduction:+d}  snap chars: {char_reduction:+d} ({pct:+.0f}%)")

    # Recommendation
    print(f"\n{'='*90}")
    print("RECOMMENDATION")
    print(f"{'='*90}")
    best = min(totals[1:], key=lambda t: (t["disruptions"], t["snap_chars"]))
    print(f"  Best strategy: {best['name']}")
    print(f"  Disruptions: {best['disruptions']} (vs {totals[0]['disruptions']} baseline)")
    print(f"  Snap chars: {best['snap_chars']} (vs {totals[0]['snap_chars']} baseline)")


if __name__ == "__main__":
    main()
