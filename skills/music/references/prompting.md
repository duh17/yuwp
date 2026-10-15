# Prompting YuE2

Distilled from the 2026-10-15 research run (English + Chinese community, Reddit, the YuE2 paper, official examples) plus local takes on mac-studio. Full evidence: `~/workspace/deep-research/reports/2026-10-15-yue-community/report.md`.

## YuE v1 advice is a trap

Most tutorials, tag lists, and Reddit tips are for YuE v1 (January 2025). YuE2 (September 2026) is a different model. Do not use:

| v1 habit | Why it is wrong for YuE2 |
|---|---|
| Space-separated tags from `top_200_tags.json`, `[Genre]` line, "bright vocal vocal" | YuE2 takes one free-text style line |
| `run_n_segments`, ~30 s per section, 4–6 lines "because a segment is 30 s" | YuE2 plans the whole song in one 24,576-token context |
| CFG 1.5/1.2, repetition penalty 1.1 | YuE2 defaults are CFG 1.0 and the upstream sampler |
| Dual-track ICL, "reference audio", 30 s chorus prompt | YuE2 has no audio-prompt input |
| "CoT" meaning segment-wise lyrics | YuE2 `--mode` is the ABC score plan |
| Separate Mandarin/Cantonese checkpoints | YuE2 is one checkpoint |

## Style line

One comma-separated line, most important first: language, genre, vocal (gender, timbre, delivery), 2–5 concrete instruments, mood or arc, tempo phrase.

```text
English, warm piano pop, expressive female voice, acoustic piano, rounded bass and light drums, lyrical memorable melody, unhurried phrasing, 88 BPM
Jazz ballad, brushed drums, upright bass, warm piano, muted trumpet, intimate Mandarin male vocal, 60 BPM
City Pop, upbeat, danceable, groovy bass, electric guitar, synth, energetic, joyful, neon city night
```

- Name the language. Official Mandarin examples use English style words with Chinese lyrics; Chinese-prose style lines are also accepted.
- Prefer a real genre name plus a few instruments over long production prose. Long "horror metal" paragraphs did not produce metal; prompts eat context the song needs.
- Tempo and key words are hints. "A minor" did not set the key in one port's test. Exact key and tempo live in the score (`K:`, `Q:`).
- "No X" is soft. "no guitar" appears in an official edit example; "no instruments" did nothing in a community test.
- No system instructions or "sing clearly" requests in the style; the runtime already adds the mode instruction.

## Lyrics

- Section tags on their own lines, title case as in the official files: `[Intro]`, `[Verse]`, `[Pre-Chorus]`, `[Chorus]`, `[Bridge]`, `[Interlude]`, `[Outro]`. Blank line between sections. Empty `[Intro]` / `[Interlude]` / `[Outro]` give instrumental space.
- Only words to sing. No titles, stage directions, `(repeat chorus)`, phoneme spellings, or pinyin substitutions. Write every repeated chorus out in full, keeping line lengths stable.
- Short lines. Official Mandarin lines are two short phrases (`路灯眨着眼睛 偷看谁的身影`); roughly 7–12 characters per Chinese line, 6–10 syllables per English line is a starting point, not a rule.
- Length is set by the lyrics and the plan, not a duration flag. A 2-verse + 2-chorus song lands around 3 minutes; 8 lines gave ~97 s with a long intro. Keep the full song under ~360 s; overlong lyrics truncate.

## Chinese songs

- Mandarin is the only Chinese variety the YuE2 authors evaluated (94 of 192 benchmark prompts). Local take: 8 Mandarin lines with a TVB-wuxia style came back from ASR with most lines exact and a few characters substituted (`灯影斜` → `灯影前`).
- Cantonese is untested upstream, and our two local takes point to Mandarin pronunciation: with either `Cantonese, …` or `粤语, …` in the style, Yuwp ASR (which reports Cantonese as `yue`) detected both as `zh`. ASR language ID on singing is a weak signal, so have Chen listen, but do not promise Cantonese diction. YuE v1's `Cantonese` tag and checkpoints do not carry over.
- 古风, 戏腔, 武侠, 粤语流行: no YuE2 practitioner reports. Describe audible traits (era, ensemble, vocal) rather than relying on genre labels, and say the recipe is untested.
- No Chinese user reported a cfg or sampler fix for 吐字不清, 漏词, or 重复. Shorter lines, naming the language, and rerolling the seed are the documented moves.

## Settings

| Situation | Do |
|---|---|
| New song | `--mode full`, default CFG, 32 steps, record the seed |
| Lyrics accuracy matters | 2+ seeds, keep the take with the best ASR diff (the paper's own pick is the lower-error of two) |
| Quick auditions | `--steps 8` per seed, then `render --from <pick> --steps 32` |
| Melody fixed, new arrangement | chord-free score + `--mode melody` |
| Free-form, no plan | `--mode off` (experts preferred planned songs in the paper) |

Do not raise CFG, steps, or (in other runtimes) temperature/repetition penalty as a fix; community sweeps did not cure skipped or doubled lines.

## Failure recovery

| Symptom | First move | Then |
|---|---|---|
| Ending cut mid-phrase | check `*_truncated` in `result.json` | shorten lyrics/style; do not raise limits |
| Skipped, mumbled, doubled lines late in a long song | new seed | fewer sections; write repeats in full |
| Vocals on an instrumental | score recipe in SKILL.md (Vocal → Ins) | new seed; stem separation only as a last resort |
| Wrong genre or instruments | shorter style: genre + 3–4 instruments | new seed before CFG |
| Wrong key or tempo | edit `K:` / `Q:` in a score copy, `--abc-file` | — |
| Robotic or over-aligned vocal | expected weakness, stronger in R&B/jazz | try a different genre framing or seed |
| Metal, classical, opera, industrial | warn up front: listeners keep rejecting these | — |
| Voice clone, extend a clip, inpaint | not supported | score cover with a new voice |

## Where YuE2 stands

- Good at: full sung songs in pop, rock, jazz/funk, city pop, Mandarin and English; editable plans; score covers; 48 kHz local quality.
- Weak at: exact duration/key from prose, guaranteed instrumentals, niche heavy or classical genres, voice cloning, continuation.
- Versus others (paper + practitioners): community prefers it to ACE-Step 1.5 and MiniMax Music 3 for local sung songs; experts still preferred Suno v6 overall (59% vs 32% against YuE2 best-of-8). Do not call it "better than Suno".

## Local findings

Takes on mac-studio with `yuwp-music` (8-bit tokens, 32 steps, seed 7 unless noted):

| Take | Result |
|---|---|
| Mandarin, TVB-wuxia style, 8 lines | 97.5 s in 51 s wall; ASR matched most lines; ~38 s intro before the vocal |
| Instrumental, piano, Vocal → Ins score (seed 3) | 90 s; ASR text empty |
| Cantonese lyrics, `Cantonese, 1990s Hong Kong TVB wuxia theme, …` | 68 s; ASR language `zh`; one verse line replaced by invented words |
| Same lyrics, Chinese style `粤语, 1990年代香港TVB武侠剧主题曲, …` | 67 s; ASR language `zh`; all lines present, several characters substituted |
| English jazz-funk, horns + choir, 11 sections | 229 s in 149 s; nearly every line intelligible; the bridge lyrics were not heard; plan merged `[Pre-Chorus]` and `[Interlude]` into neighbors |
| Portuguese bossa nova | 175 s; lyrics clean; plan chose `M:2/4`, `Q:1/4=67` (a proper bossa feel) |
| Japanese 80s city pop | 205 s; lyrics largely intact, katakana words came back as English (`highway`, `melody`) |
| English Afrobeat, call-and-response bridge | 158 s; all sections sung including the call-and-response |

Yuwp ASR sometimes returns an English translation of later sung lines in other languages (`The sky of Panama is so blue`). That is the ASR, not the song; check the language of the early lines before blaming the take.
