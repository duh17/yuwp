{
  "id": "c5da8aee",
  "title": "Bubble design polish — themes, layout, remove max height cap",
  "tags": [
    "ui",
    "polish"
  ],
  "status": "open",
  "created_at": "2026-04-07T07:42:13.999Z"
}

## Issues
1. **Max height cap** — long dictations get cut off (maxHeight: 260pt).
   Need to keep expanding, with the panel scrolling or growing up to
   screen height minus some margin.
2. **Empty space** — the bubble has a lot of wasted area, especially with
   short text. The waveform bars take fixed space on the left.
3. **Theme/animation variety** — could offer different visual styles.

## Fix: Remove max height cap
- Remove `maxHeight` constant or increase to screen height - 100
- Panel should grow as text grows, always keeping text visible
- `clampToScreen` already handles edge cases

## Design ideas to brainstorm
- **Compact**: text-only pill, no waveform bars (waveform moves to border glow)
- **Minimal**: just a thin underline below the text with pulsing color
- **Floating card**: wider, shorter, text wraps less aggressively
- **Side panel**: docked to screen edge, like Spotlight
- **Inline ghost text**: for AX mode, show upcoming text as gray "ghost" characters

## Configurable themes
- Add theme selection to the menu bar submenu
- Store in Config (UserDefaults)
- Start with 2-3 themes: Default, Compact, Minimal
