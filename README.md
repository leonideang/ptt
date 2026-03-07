# PTT — Push-to-Talk Transcription

Local, on-device speech-to-text for macOS. Hold a key, speak, release — your words appear where your cursor is.

Runs entirely on your Mac using [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper). No API keys, no cloud, no data leaves your machine.

## Requirements

- macOS 13+ with Apple Silicon (M1/M2/M3/M4)
- [uv](https://docs.astral.sh/uv/) — `brew install uv`

## Install

```bash
git clone https://github.com/leonideang/ptt.git
cd ptt
uv run ptt.py --install   # Sets up auto-start on login
```

First launch downloads the Whisper model (~1.5 GB). Subsequent starts take ~3 seconds.

### Grant Accessibility permission

PTT needs Accessibility access to read hotkeys and paste text:

**System Settings → Privacy & Security → Accessibility** → add your terminal app.

## Usage

PTT lives in your menu bar as **◉**. Hold **Right Option** to record, release to transcribe and paste.

| Icon | State |
|------|-------|
| ◉ | Idle — ready |
| ● | Recording |
| ⏳ | Transcribing |
| ⚠ | Error — check log |

### Menu bar options

- **Språk** — toggle Swedish / English
- **Modell** — switch between Turbo (fast) and KB Swedish (better Swedish accuracy)
- **Kalibrera mikrofon** — re-measure ambient noise (do this if you change rooms)
- **Visa logg** — open log in Console.app

### CLI flags

```bash
ptt                          # Swedish, Right Option, Turbo model
ptt --lang en                # English
ptt --model kb-sv            # Swedish-optimized model (slower, better Swedish)
ptt --key alt                # Use Left Option instead
ptt --device 2               # Specific microphone
ptt --list-devices           # Show available microphones
ptt --install                # Auto-start on login
ptt --uninstall              # Remove auto-start
```

## Models

| Key | Model | Speed | Best for |
|-----|-------|-------|----------|
| `turbo` | whisper-large-v3-turbo (809M) | ~1s/chunk | General use, all languages |
| `kb-sv` | KB Whisper Large (1550M) | ~2s/chunk | Swedish accuracy |

Switch models live from the menu bar — no restart needed.

## How it works

1. **Hold hotkey** → recording starts with pre-roll buffer (captures word onsets)
2. **Speak** → audio is monitored for natural pauses
3. **Pause detected** → chunk is transcribed and pasted immediately
4. **Continue speaking** → more chunks transcribed and pasted
5. **Release hotkey** → final chunk transcribed, done

The mic auto-calibrates at startup by measuring 2 seconds of ambient noise. This means PTT works even when whispering.

## Troubleshooting

**Text not pasting?** Check Accessibility permissions. The text is always copied to clipboard as fallback — you can Cmd+V manually.

**Wrong microphone?** Run `ptt --list-devices` and use `ptt --device N`.

**Too sensitive / not sensitive enough?** Click "Kalibrera mikrofon" in the menu bar, or restart PTT in a quieter environment.

**Log location:** `~/Library/Logs/ptt.log` (or click "Visa logg" in the menu bar).

## License

MIT
