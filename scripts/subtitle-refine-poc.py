#!/usr/bin/env -S uv run --python 3.14 --script
import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from urllib.request import Request, urlopen

STRONG_END = set('。？！；…!?')
WEAK_END = set('，、：,')
ORPHAN_START = set('的了啊呢吗吧着过把被给跟和与及并而且但却就也还又都才仍让向从对于在为将所使令来去上下降进出里外前后中内左右部位年月日天点分秒种项个')
ORPHAN_END = set('的地得了着过把被给跟和与及并而且但却就也还又都才仍让向从对于在为将所使令来去上下降进出里外前后中内左右部位年月日天点分秒种项个张雪峰全办后启正无可常疗识')
EXACT_RESIDUES = {
    '的医院。', '了讣告。', '来了。', '常高。', '题。', '治疗，', '识啊。', '部变成了灰白色。', '峰死了”，', '理解。', '吧？', '性', '面。'
}
ZERO_STUTTER_CHARS = set('在那我你这哪就还又才再嗯呃啊哦哎')


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


@dataclass
class Segment:
    index: int
    start: float
    end: float
    text: str


@dataclass
class BoundaryDecision:
    left_index: int
    right_index: int
    merge: bool
    confidence: float
    reason: str
    score: int
    latency_sec: float
    before_left: str
    before_right: str
    after_text: str | None = None
    local_penalty_before: int | None = None
    local_penalty_after: int | None = None


def parse_timecode(value: str) -> float:
    hh, mm, rest = value.split(':')
    ss, ms = rest.split(',')
    return int(hh) * 3600 + int(mm) * 60 + int(ss) + int(ms) / 1000


def format_timecode(value: float) -> str:
    total_ms = max(0, int(round(value * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    ss = total_s % 60
    total_m = total_s // 60
    mm = total_m % 60
    hh = total_m // 60
    return f'{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}'


def parse_srt(path: Path) -> list[Segment]:
    blocks = re.split(r'\n\s*\n', path.read_text().strip())
    out: list[Segment] = []
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        idx = int(lines[0])
        start_s, end_s = lines[1].split(' --> ')
        text = ''.join(lines[2:])
        out.append(Segment(index=idx, start=parse_timecode(start_s), end=parse_timecode(end_s), text=text))
    return out


def write_srt(path: Path, segments: list[Segment]) -> None:
    parts = []
    for i, seg in enumerate(segments, start=1):
        parts.append(f'{i}\n{format_timecode(seg.start)} --> {format_timecode(seg.end)}\n{seg.text}')
    path.write_text('\n\n'.join(parts) + '\n')


def load_debug_items(debug_path: Path) -> list[dict]:
    data = json.loads(debug_path.read_text())
    return [item for chunk in data['debug']['chunks'] for item in chunk['items']]


def collect_zero_stutter_chars(items: list[dict]) -> list[str]:
    chars: list[str] = []
    for i, item in enumerate(items[:-1]):
        nxt = items[i + 1]
        text = item['text']
        if (
            len(text) == 1
            and text == nxt['text']
            and text in ZERO_STUTTER_CHARS
            and ((nxt['end'] - nxt['start']) <= 0.001 or (item['end'] - item['start']) <= 0.001)
        ):
            chars.append(text)
    return chars


def apply_safe_stutter_cleanup(segments: list[Segment], chars: list[str]) -> tuple[list[Segment], list[dict]]:
    if not chars:
        return segments[:], []
    out: list[Segment] = []
    changes: list[dict] = []
    patterns = {char: char + char for char in sorted(set(chars))}
    for seg in segments:
        new_text = seg.text
        dropped: list[dict] = []
        for char, doubled in patterns.items():
            if doubled in new_text:
                new_text = new_text.replace(doubled, char, 1)
                dropped.append({'char': char, 'pattern': doubled})
        if new_text != seg.text:
            changes.append({
                'index': seg.index,
                'start': seg.start,
                'end': seg.end,
                'before': seg.text,
                'after': new_text,
                'dropped': dropped,
            })
        out.append(Segment(index=seg.index, start=seg.start, end=seg.end, text=new_text))
    return out, changes


def segment_penalty(seg: Segment) -> int:
    text = seg.text
    if not text:
        return 10
    score = 0
    if len(text) <= 2:
        score += 6
    elif len(text) <= 4:
        score += 3
    if text in EXACT_RESIDUES:
        score += 5
    if text[0] in ORPHAN_START and len(text) <= 8:
        score += 3
    if text[-1] in ORPHAN_END and text[-1] not in STRONG_END and len(text) <= 12:
        score += 3
    if text[-1] in WEAK_END and len(text) <= 8:
        score += 2
    if (seg.end - seg.start) < 0.22:
        score += 6
    return score


def boundary_score(left: Segment, right: Segment, max_chars: int, max_duration: float) -> int:
    combined = left.text + right.text
    duration = right.end - left.start
    score = 0
    if left.text and left.text[-1] not in STRONG_END:
        score += 1
    else:
        return 0
    if right.text and right.text[0] not in STRONG_END:
        score += 1
    if len(right.text) <= 3:
        score += 4
    elif len(right.text) <= 6:
        score += 1
    if len(left.text) <= 3:
        score += 2
    if right.text in EXACT_RESIDUES:
        score += 4
    if right.text and right.text[0] in ORPHAN_START:
        score += 4
    if left.text and left.text[-1] in ORPHAN_END:
        score += 2
    if left.text and right.text and left.text[-1] not in STRONG_END | WEAK_END and right.text[0] not in STRONG_END | WEAK_END:
        score += 2
    if len(combined) <= max_chars:
        score += 1
    if duration <= max_duration:
        score += 1
    if re.search(r'[\u4e00-\u9fff]$', left.text) and re.search(r'^[\u4e00-\u9fff]', right.text):
        score += 1
    return score


def local_penalty(segments: list[Segment], idx: int, max_chars: int, max_duration: float) -> int:
    score = 0
    start = max(0, idx - 1)
    end = min(len(segments), idx + 2)
    for j in range(start, end):
        score += segment_penalty(segments[j])
    if idx - 1 >= 0:
        score += boundary_score(segments[idx - 1], segments[idx], max_chars, max_duration) // 2
    if idx + 1 < len(segments):
        score += boundary_score(segments[idx], segments[idx + 1], max_chars, max_duration) // 2
    return score


def merge_pair(left: Segment, right: Segment) -> Segment:
    return Segment(index=left.index, start=left.start, end=right.end, text=left.text + right.text)


def should_consider(left: Segment, right: Segment, score: int, max_chars: int, max_duration: float, score_threshold: int) -> bool:
    if score < score_threshold:
        return False
    if left.text.endswith(tuple(STRONG_END)):
        return False
    if (right.end - left.start) > max_duration:
        return False
    if len(left.text + right.text) > max_chars:
        return False
    return True


def model_decision(base_url: str, model: str, prev_text: str, left: Segment, right: Segment, next_text: str, max_chars: int, max_duration: float) -> tuple[bool, float, str, float]:
    system = (
        'You judge one Chinese subtitle boundary. Do not rewrite text. '
        'Return strict JSON only: {"merge": true|false, "confidence": 0-1, "reason": "..."}. '
        'Merge only if joining the two adjacent subtitles clearly fixes a phrase split or tiny residue. '
        'Do not merge merely because the sentence continues. Preserve exact text if merged.'
    )
    user = (
        f'Previous: {prev_text or "<START>"}\n'
        f'Left: {left.text}\n'
        f'Right: {right.text}\n'
        f'Next: {next_text or "<END>"}\n\n'
        f'Should Left and Right be merged? Reject if each line is already a reasonable subtitle. '
        f'Prefer keeping natural phrases together. Respect a practical merged limit of about {max_chars} Chinese characters '
        f'and {max_duration:.1f} seconds.'
    )
    payload = {
        'model': model,
        'temperature': 0,
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
    }
    start = time.perf_counter()
    req = Request(
        base_url,
        method='POST',
        headers={'content-type': 'application/json'},
        data=json.dumps(payload).encode('utf-8'),
    )
    with urlopen(req, timeout=120) as resp:
        body = json.loads(resp.read().decode('utf-8'))
    latency = time.perf_counter() - start
    content = body['choices'][0]['message']['content']
    match = re.search(r'\{.*\}', content, re.S)
    if not match:
        raise RuntimeError(f'No JSON found in model response: {content[:300]}')
    data = json.loads(match.group(0))
    return bool(data.get('merge')), float(data.get('confidence', 0.0)), str(data.get('reason', '')), latency


def qc_report(segments: list[Segment], max_chars: int, max_duration: float) -> dict:
    durations = [seg.end - seg.start for seg in segments]
    char_counts = [len(seg.text) for seg in segments]
    cps = [len(seg.text) / max(0.001, seg.end - seg.start) for seg in segments]
    orphan_start = sum(1 for seg in segments if seg.text and seg.text[0] in ORPHAN_START and len(seg.text) <= 8)
    orphan_end = sum(1 for seg in segments if seg.text and seg.text[-1] in ORPHAN_END and seg.text[-1] not in STRONG_END and len(seg.text) <= 12)
    tiny = sum(1 for seg in segments if len(seg.text) <= 3)
    exact_residue = sum(1 for seg in segments if seg.text in EXACT_RESIDUES)
    suspicious_boundaries = []
    for i in range(len(segments) - 1):
        score = boundary_score(segments[i], segments[i + 1], max_chars, max_duration)
        if score >= 8:
            suspicious_boundaries.append({
                'left_index': i + 1,
                'right_index': i + 2,
                'score': score,
                'left': segments[i].text,
                'right': segments[i + 1].text,
            })
    return {
        'segment_count': len(segments),
        'duration_avg': round(sum(durations) / max(1, len(durations)), 3),
        'duration_max': round(max(durations) if durations else 0, 3),
        'chars_avg': round(sum(char_counts) / max(1, len(char_counts)), 2),
        'chars_max': max(char_counts) if char_counts else 0,
        'cps_avg': round(sum(cps) / max(1, len(cps)), 2),
        'cps_p95': round(sorted(cps)[max(0, math.ceil(len(cps) * 0.95) - 1)] if cps else 0, 2),
        'orphan_start_count': orphan_start,
        'orphan_end_count': orphan_end,
        'tiny_segment_count': tiny,
        'exact_residue_count': exact_residue,
        'suspicious_boundary_count': len(suspicious_boundaries),
        'top_suspicious_boundaries': suspicious_boundaries[:40],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-srt', required=True)
    ap.add_argument('--debug-json', required=True)
    ap.add_argument('--output-srt', required=True)
    ap.add_argument('--report-json', required=True)
    ap.add_argument('--audit-json', required=True)
    ap.add_argument('--model-url', default='http://127.0.0.1:8400/v1/chat/completions')
    ap.add_argument('--model', default='Qwen3.5-27B-8bit')
    ap.add_argument('--max-chars', type=int, default=24)
    ap.add_argument('--max-duration', type=float, default=5.0)
    ap.add_argument('--max-calls', type=int, default=80)
    ap.add_argument('--passes', type=int, default=2)
    ap.add_argument('--score-threshold', type=int, default=8)
    args = ap.parse_args()

    start_wall = time.perf_counter()
    segments = parse_srt(Path(args.input_srt))
    before_qc = qc_report(segments, args.max_chars, args.max_duration)

    debug_items = load_debug_items(Path(args.debug_json))
    zero_stutter_chars = collect_zero_stutter_chars(debug_items)
    segments, stutter_changes = apply_safe_stutter_cleanup(segments, zero_stutter_chars)

    decisions: list[BoundaryDecision] = []
    llm_calls = 0
    llm_latency = 0.0

    for _ in range(args.passes):
        i = 0
        changed = False
        while i < len(segments) - 1 and llm_calls < args.max_calls:
            left, right = segments[i], segments[i + 1]
            score = boundary_score(left, right, args.max_chars, args.max_duration)
            if not should_consider(left, right, score, args.max_chars, args.max_duration, args.score_threshold):
                i += 1
                continue
            prev_text = segments[i - 1].text if i - 1 >= 0 else ''
            next_text = segments[i + 2].text if i + 2 < len(segments) else ''
            before_local = local_penalty(segments, i, args.max_chars, args.max_duration)
            merge, conf, reason, latency = model_decision(
                args.model_url,
                args.model,
                prev_text,
                left,
                right,
                next_text,
                args.max_chars,
                args.max_duration,
            )
            llm_calls += 1
            llm_latency += latency
            accepted = False
            after_text = None
            after_local = None
            if merge and conf >= 0.55:
                merged = merge_pair(left, right)
                candidate_segments = segments[:i] + [merged] + segments[i + 2:]
                after_local = local_penalty(candidate_segments, max(0, i - 1), args.max_chars, args.max_duration)
                if after_local + 1 < before_local:
                    segments = candidate_segments
                    accepted = True
                    changed = True
                    after_text = merged.text
            decisions.append(BoundaryDecision(
                left_index=i + 1,
                right_index=i + 2,
                merge=accepted,
                confidence=conf,
                reason=reason,
                score=score,
                latency_sec=latency,
                before_left=left.text,
                before_right=right.text,
                after_text=after_text,
                local_penalty_before=before_local,
                local_penalty_after=after_local,
            ))
            if not accepted:
                i += 1
        if not changed:
            break

    after_qc = qc_report(segments, args.max_chars, args.max_duration)
    wall = time.perf_counter() - start_wall

    write_srt(Path(args.output_srt), segments)
    Path(args.audit_json).write_text(json.dumps({
        'stutter_changes': stutter_changes,
        'decisions': [asdict(x) for x in decisions],
    }, ensure_ascii=False, indent=2))
    Path(args.report_json).write_text(json.dumps({
        'input_srt': args.input_srt,
        'output_srt': args.output_srt,
        'debug_json': args.debug_json,
        'model': args.model,
        'max_chars': args.max_chars,
        'max_duration': args.max_duration,
        'passes': args.passes,
        'score_threshold': args.score_threshold,
        'max_calls': args.max_calls,
        'llm_calls': llm_calls,
        'llm_latency_total_sec': round(llm_latency, 3),
        'avg_llm_latency_sec': round(llm_latency / max(1, llm_calls), 3),
        'wall_time_sec': round(wall, 3),
        'before_qc': before_qc,
        'after_qc': after_qc,
        'segment_delta': after_qc['segment_count'] - before_qc['segment_count'],
        'suspicious_boundary_delta': after_qc['suspicious_boundary_count'] - before_qc['suspicious_boundary_count'],
        'stutter_change_count': len(stutter_changes),
        'accepted_merge_count': sum(1 for x in decisions if x.merge),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
