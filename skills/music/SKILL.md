---
name: music
description: "Generate songs or instrumentals locally with YuE2 (yuwp-music, Swift/MLX): style + lyrics to 48 kHz stereo, ABC score edits, covers. Not speech (tts) or transcription."
container: false
---

# Music

Local song generation with `yuwp-music`, a native Swift/MLX port of YuE2-3B (HKUST / M-A-P). Style text and section-tagged lyrics go in; a 48 kHz stereo WAV comes out, plus the ABC score YuE2 planned. Up to ~360 s per song. No Python at runtime.

Before writing a style or lyrics, read [prompting](references/prompting.md). It holds the English and Chinese community practice, failure recovery, and the YuE v1 advice to ignore.

## Choose the path

| Ask | Path |
|---|---|
| New song from a brief or lyrics | `generate`, `--mode full` (default) |
| Instrumental, no vocals | plan → `scripts/instrumentalize.ts` → `generate --abc-file` (below) |
| Change key, tempo, melody, or harmony of a song we made | edit a copy of its `score.abc`, regenerate with `--abc-file` |
| Same melody, new arrangement | chord-free score + `--mode melody` |
| Cover a recording | transcribe to ABC with mlx-Yue (below), then `--mode melody` |
| Same tokens, more or fewer flow steps | `render --from <song dir> --steps N` |
| Voice cloning, continue/extend a clip, inpaint a bar | not supported by YuE2; say so |
| Speech, narration, sound effects | stop — use `tts` |

## Setup

```bash
YUWP_MUSIC=~/workspace/yuwp/.build/out/Products/Release/yuwp-music
YUE2=~/workspace/mlx-Yue/models/mlx-Yue2-3B   # or wherever vanch007/mlx-Yue2-3B was downloaded
```

Build if missing: `cd ~/workspace/yuwp && swift build -c release --product yuwp-music`. Weights if missing: `hf download vanch007/mlx-Yue2-3B --local-dir ~/models/mlx-Yue2-3B` (~10 GB: `ar-8bit`, `ar-bf16`, `nar-bf16`, `qwen.tiktoken`, `vae/`).

One GPU job at a time. A 97 s song peaks near 28 GB unified memory; leave other large MLX jobs alone rather than killing them.

## Generate

```bash
OUT=.pi/music/<slug>            # workspace .pi/ is gitignored
mkdir -p "$OUT"
cat > "$OUT/lyrics.txt" <<'EOF'
[Verse]
...
[Chorus]
...
EOF
"$YUWP_MUSIC" generate --model "$YUE2" \
  --style "English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, 88 BPM" \
  --lyrics-file "$OUT/lyrics.txt" --seed 41 --out "$OUT/take1"
```

| Flag | Default | Use |
|---|---|---|
| `--mode full\|melody\|off` | `full` | `full` plans melody + chords (better songs in the paper's listening test); `melody` plans or takes a chord-free melody; `off` skips the score |
| `--seed N` | 831001 | Record it. Different seeds are the main variation lever |
| `--steps N` | 32 | Flow-matching steps. 8 is ~4× faster on that stage; use for auditions, re-render the pick at 32 |
| `--precision 8bit\|4bit\|bf16` | `8bit` | Token-generation weights only. 4-bit is faster, quality unmeasured |
| `--cfg X` | 1.0 (1.01 for `off`) | Leave alone. Higher costs a second branch and is not a proven fix |
| `--abc-file F` | — | Render this score instead of planning one; needs `full` or `melody` |
| `--plan-only` | — | Write `score.abc` and stop (seconds) |

The CLI does not expose temperature, top-p, top-k, or repetition penalty; the defaults match upstream YuE2.

Wall time on the M3 Ultra: a 97 s song took 51 s end to end; a 4.4 min song about 160 s. Run songs longer than a minute or two with `background_job`.

## Outputs and checks

`<out>/` holds `audio.wav`, `score.abc`, `prefix.npy`, `semantic.npy`, `noise.npy`, `latent.npy`, and `result.json`.

After every take:

1. `result.json`: `abc_truncated` and `semantic_truncated` must be false. A truncated song still plays but ends mid-phrase. Fix by shortening the lyrics, not by raising limits.
2. `audio_seconds` is plausible for the lyric length.
3. When the words matter, run the take through Yuwp ASR and diff against the lyrics. This is a pronunciation check, not a quality score:

   ```bash
   curl -sf http://127.0.0.1:7936/v1/info >/dev/null && \
   curl -sf http://127.0.0.1:7936/v1/audio/transcriptions -F file=@"$OUT/take1/audio.wav" | jq -r .text
   ```

   For instrumentals, near-empty ASR text means no lexical vocals; hums can still slip through.

You cannot hear the result. Report what was checked (flags, duration, ASR) and leave the listening verdict to Chen. Embed the audio:

```text
![[.pi/music/<slug>/take1/audio.wav]]
```

Do not play audio locally unless Chen is at the desk and asked.

## Instrumentals

Style words alone ("no vocals") leak vocals often. The upstream recipe works on the score: keep the chords, silence the Vocal line, move its melody to Ins.

```bash
printf '[Intro]\n\n[Verse]\n\n[Chorus]\n\n[Outro]\n' > "$OUT/sections.txt"
STYLE="Instrumental, warm solo piano, lyrical and gentle, soft strings, no vocals, 80 BPM"
"$YUWP_MUSIC" generate --model "$YUE2" --style "$STYLE" --lyrics-file "$OUT/sections.txt" \
  --seed 3 --plan-only --out "$OUT/plan"
bun ~/.pi/agent/skills/music/scripts/instrumentalize.ts "$OUT/plan/score.abc" "$OUT/inst.abc"
"$YUWP_MUSIC" generate --model "$YUE2" --style "$STYLE" --lyrics-file "$OUT/sections.txt" \
  --abc-file "$OUT/inst.abc" --seed 3 --out "$OUT/take1"
```

If vocals remain, try another seed before anything else.

## Editing a score

Copy `score.abc`, edit the copy, regenerate with `--abc-file` and the same style and lyrics. Editing regenerates the whole song; it is not inpainting. Key (`K:`), tempo (`Q:`), meter, and notes live in the score; "A minor" or "120 BPM" in the style is only a hint. The score is a bounded two-voice dialect (`V: Vocal`, `V: Ins`, quoted chords on Vocal, `L:` grid, no tuplets or grace notes). Rules and the chord vocabulary: `~/workspace/mlx-Yue/skills/yue2-music/references/abc-editing.md`. Validate an edit with

```bash
~/workspace/mlx-Yue/.venv/bin/mlx-yue abc inspect edited.abc --output /tmp/inspect.json
~/workspace/mlx-Yue/.venv/bin/mlx-yue abc compare score.abc edited.abc   # melody invariants
```

## Covers

YuE2 has no reference-audio input. A cover is a transcribed score sung by a new voice; it does not clone the singer or keep the mix.

```bash
~/workspace/mlx-Yue/.venv/bin/mlx-yue transcribe source.wav --task melody-full --output "$OUT/source-score"
~/workspace/mlx-Yue/.venv/bin/mlx-yue abc strip-chords "$OUT/source-score/score.abc" "$OUT/cover.abc"
"$YUWP_MUSIC" generate --model "$YUE2" --mode melody --abc-file "$OUT/cover.abc" \
  --style "<target style>" --lyrics-file "$OUT/lyrics.txt" --out "$OUT/cover"
```

The first `transcribe` downloads SheetSage2 and MERT2 weights (not cached on mac-studio yet). Keep chords and use `--mode full` to preserve the original harmony. Supply the real lyrics; transcription does not recover words.

## License

YuE2 weights are CC BY-NC 4.0 with an added permission for individual creators to monetize generated songs. Companies need a separate weight license. Do not tell Chen outputs are unusable, and do not call them cleared for business use.

## Background

Research behind this skill (English and Chinese community, Reddit, papers, local takes): `~/workspace/deep-research/reports/2026-10-15-yue-community/report.md`.
