{
  "id": "7afaef02",
  "title": "Voice Clipboard / Scratchpad — dictation history panel",
  "tags": [
    "feature",
    "ui"
  ],
  "status": "open",
  "created_at": "2026-04-07T07:42:00.752Z"
}

## Concept
Like Apple's Voice Memos but for dictation. A persistent panel showing all
dictation sessions with:
- Transcribed text
- Which app it was injected into (captured from AX at dictation time)
- Timestamp
- Play button to replay the WAV recording
- Copy button to re-paste the text

## Activation
Different hotkey from dictation (e.g., a combo key, or long-press the dictation key).
Could also be accessible from the menu bar dropdown.

## Data
Already have everything needed:
- WAV recordings saved to `~/Library/Application Support/Yuwp/recordings/`
- Text from final transcription
- Target app info from TextInjector.captureTarget()

Just need to persist a metadata index (JSON or SQLite) linking:
`{timestamp, wav_path, text, target_app, target_element_role}`

## UI
- NSPanel (like the current mic panel, but larger and persistent)
- List of dictation entries, newest first
- Waveform visualization for each entry
- Click to copy text, play button for audio
- Search/filter
