# Yuwp Public Launch Readiness

Date: 2026-04-23

## Current Verdict

Yuwp is close enough for a focused pre-launch push, but not ready for a broad marketing push until the public repo has a credible landing surface and at least one short demo video.

The app/release mechanics are stronger than the public story. The repo already has a notarized DMG release path, Sparkle appcast generation, privacy-first local ASR positioning, and a passing local test suite. The current public surface does not yet explain the product quickly enough for someone arriving from a link.

## Current State

- Public repo: `https://github.com/duh17/yuwp`
- Latest public release: `v0.1.1`, published 2026-04-17
- Latest release assets: `Yuwp-0.1.1.dmg`, `appcast.xml`
- Local branch: `master`
- Local state: one commit ahead of `origin/master`, plus uncommitted app/settings changes
- Local media assets found: only `Resources/Yuwp.icns`
- Existing landing page: none
- Existing screenshots or demo videos in repo: none
- Local validation run: `swift build` and `swift test` passed on 2026-04-23, with ASR/server/audio stress tests skipped because they require a running model server or microphone access
- Local server readiness check: `swift-mlx-asr-server` responded to `/v1/info` with `"status":"ready"` on `127.0.0.1:7936`
- Manual app test still outstanding: hotkey -> record -> transcribe -> inject text

## Blocking Gaps

1. No visual proof.
   A visitor cannot see Yuwp working before downloading a macOS app that needs Accessibility, Microphone, and model setup.

2. No landing page.
   The README is useful for developers but not enough for launch traffic. It lacks a hero message, screenshots, demo, target-user framing, comparison, FAQ, and strong download CTA.

3. No demo-video publishing path.
   Demo videos should not be committed directly into normal Git history. GitHub recommends keeping repositories small, warns on regular files over 50 MiB, and supports release assets up to 2 GiB each.

4. Release is behind local work.
   `v0.1.1` points at `origin/master`; local `master` is ahead and has uncommitted changes. A launch should ship from a clean, tagged release after final manual end-to-end testing.

5. GitHub repo metadata is underused.
   The repo has no homepage URL and no topics. The current GitHub description mentions `mlx-audio`, while the README positions Yuwp as Swift + MLX/Qwen3-ASR. Align this before outreach.

## Demo Video Plan

Publish demo videos as GitHub Release assets first. Link them from the README and landing page. Do not commit large `.mov` or `.mp4` files into the repository unless there is a deliberate Git LFS policy.

Recommended initial set:

- `yuwp-demo-30s.mp4`: one-take dictation into Notes/TextEdit, showing hotkey, mic bubble, final injection.
- `yuwp-terminal-demo-20s.mp4`: dictation into a terminal or code editor to prove non-standard text surfaces.
- `yuwp-privacy-settings-15s.mp4`: local model/settings view, no cloud API, recording off by default.
- `yuwp-server-api-20s.mp4`: optional CLI/HTTP transcription for technical users.

Encoding targets:

- Format: MP4, H.264 video, AAC audio.
- Resolution: 1920x1080 or 1440x900.
- Size: keep each demo under 100 MiB if practical.
- Captions: add burned-in captions or a short transcript nearby for silent autoplay contexts.
- Privacy: crop menu bar/user data, disable notifications, use throwaway documents and terminal paths.

Suggested release-asset workflow:

```bash
gh release upload v0.1.2 \
  path/to/yuwp-demo-30s.mp4 \
  path/to/yuwp-terminal-demo-20s.mp4 \
  --clobber
```

After upload, link assets from README and landing page using stable release URLs:

```text
https://github.com/duh17/yuwp/releases/download/v0.1.2/yuwp-demo-30s.mp4
```

## Landing Page Brief

Best first build: a static GitHub Pages page in `docs/` or a generated site deployed to GitHub Pages. GitHub Pages can publish static files from a repository, and `/docs` is a supported source folder.

Recommended page structure:

- Hero: "Fast local dictation for macOS. No cloud API after model download."
- Primary CTA: "Download for macOS"
- Secondary CTA: "View source on GitHub"
- Proof: autoplay muted 30-second demo video
- Benefits: local ASR, global hotkey, works across apps, terminal fallback, optional HTTP API
- Setup: macOS 14+, Apple Silicon, grant Accessibility/Microphone, download/select model
- Privacy: local processing, recordings off by default, diagnostics off by default
- Developer angle: `yuwp-asr`, `swift-mlx-asr-server`, OpenAI-compatible HTTP endpoint
- FAQ: model size/download, permissions, unsupported apps, Intel Mac support, accuracy/latency expectations
- Release trust: notarized DMG, Sparkle updates, MIT license, third-party notices

Minimum assets needed for a credible first page:

- App icon exported as PNG or SVG-friendly image.
- One hero demo video.
- One settings screenshot.
- One menubar/mic-panel screenshot or GIF.
- A 1-paragraph privacy statement.
- A short troubleshooting/permissions section.

## README Changes Before Launch

Update README in this order:

1. Add screenshot/demo immediately after the opening paragraph.
2. Add a "Why Yuwp" section for non-developers.
3. Add direct release DMG link for the latest launch tag, not only `/releases/latest`.
4. Add "Demo videos" with GitHub Release asset links.
5. Add "Permissions and privacy" near first-launch instructions.
6. Add a small FAQ before CLI/server docs.
7. Move deep CLI details lower so the app story comes first.

## Marketing Push Readiness

Do a limited awareness push once these are true:

- Clean worktree, tagged release, notarized DMG uploaded.
- README has video/screenshot proof above the fold.
- Landing page is live and linked from GitHub repo homepage.
- Demo videos are public release assets.
- End-to-end manual test passes: hotkey -> record -> transcribe -> inject text.
- `/v1/info` readiness test passes with the release server binary and model.
- Manual hotkey-to-injection test passes on the packaged app.
- Known limitations are explicit: macOS 14+, Apple Silicon, initial model download, required permissions.

Initial outreach sequence:

1. Soft launch to personal network and a small macOS/dev audience.
2. Fix install/onboarding issues from first users.
3. Post a short demo-led announcement to developer/macOS communities.
4. Publish a longer technical write-up about local ASR, Swift/MLX, and text injection tradeoffs.
5. Use release download counts, GitHub stars, and issue volume to decide whether to do a larger Product Hunt/Hacker News-style push.

## Immediate Next Tasks

- Resolve or commit the current app/settings changes.
- Cut `v0.1.2` only after manual hotkey-to-injection validation.
- Record the 30-second hero demo.
- Upload demo videos as release assets.
- Build `docs/index.html` landing page once at least one demo video URL exists.
- Update README and GitHub repo metadata.

## References

- GitHub release assets: https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases
- GitHub large file guidance: https://docs.github.com/repositories/working-with-files/managing-large-files/about-large-files-on-github
- GitHub Pages publishing source: https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site
