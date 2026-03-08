# PTT — Push-to-Talk Transcription

Local, on-device speech-to-text for macOS. Hold a key, speak, release — your words appear where your cursor is.

Runs entirely on your Mac using [MLX Whisper](https://github.com/ml-explore/mlx-examples/tree/main/whisper). No cloud, no data leaves your machine.

## Requirements

- macOS 13+ with Apple Silicon (M1/M2/M3/M4)
- [uv](https://docs.astral.sh/uv/) — `brew install uv`

## Install

### Quick install (recommended)

```bash
curl -fsSL https://raw.githubusercontent.com/leonideang/ptt/main/install.sh | bash
ptt --install    # Auto-start on login
```

### macOS app

```bash
git clone https://github.com/leonideang/ptt.git
cd ptt
./build-app.sh
cp -r dist/PTT.app /Applications/
```

Then open PTT from Applications (or Spotlight). Requires `uv` — install with `brew install uv`.

### With Homebrew

```bash
brew tap leonideang/tap
brew install ptt
ptt --install
```

### From source

```bash
git clone https://github.com/leonideang/ptt.git
cd ptt
uv run ptt.py              # Start PTT
uv run ptt.py --install    # Auto-start on login
```

First launch downloads the Whisper model (~1.5 GB). Subsequent starts take ~3 seconds.

### Accessibility permission

PTT needs Accessibility access to read hotkeys and paste text. It will prompt you on first run, or grant it manually:

**System Settings → Privacy & Security → Accessibility** → add Terminal (or your terminal app, or PTT.app).

## Usage

PTT lives in your menu bar as a small waveform icon.

| Action | What happens |
|--------|-------------|
| Hold **Right Option (⌥)** | Start recording |
| Release | Transcribe and paste |
| Hold **⌘ + Right Option** | Record in *polish mode* (see below) |

| Icon | State |
|------|-------|
| Waveform (calm) | Idle — ready |
| Waveform (animated) | Recording |
| Waveform (pulsing) | Transcribing |

### Text polish (optional)

Hold **⌘ + your hotkey** instead of just the hotkey. PTT transcribes as usual, then sends the text to [Groq](https://groq.com/) to clean up filler words, repetitions, and stutters — turning spoken language into polished written text.

**Setup:**

1. Get a free API key at [console.groq.com](https://console.groq.com)
2. Add it via **Settings** in the menu bar, or from the terminal:
   ```bash
   security add-generic-password -a groq -s ptt -w "gsk_your_key_here"
   ```

Polish mode falls back to raw transcription if no API key is set or if the API is unavailable.

> **Privacy:** In polish mode, your transcribed text is sent to Groq's servers for processing. Normal transcription is always 100% local.

### Settings

Click the waveform icon → **Settings** to configure:

- **Language** — English, Swedish (if system has Swedish)
- **Model** — Turbo (fast, all languages) or KB Swedish (best for Swedish)
- **Hotkey** — Right/Left Option or Control
- **Microphone** — shows detected mic, calibrate button
- **Vocabulary** — custom words to improve recognition accuracy
- **Text polish** — Groq API key, editable polish prompt
- **Start at login** — auto-start via LaunchAgent
- **Logging** — toggle log file, view/clear log

### CLI

```bash
ptt                          # Start with defaults
ptt --lang en                # English
ptt --lang sv                # Swedish
ptt --model turbo            # Turbo model (default)
ptt --model kb-sv            # Swedish-optimized model
ptt --key alt                # Use Left Option instead
ptt --device 2               # Specific microphone
ptt --list-devices           # Show available microphones
ptt --install                # Auto-start on login
ptt --uninstall              # Remove auto-start
ptt --reset                  # Reset all settings
```

## Models

| Key | Model | Size | Speed | Best for |
|-----|-------|------|-------|----------|
| `kb-sm` | KB Whisper Small (default) | 459 MB | ~0.5s/chunk | Swedish — fast and accurate |
| `small` | Whisper Small | 461 MB | ~0.5s/chunk | All languages |
| `turbo` | Whisper Large v3 Turbo | 809 MB | ~1s/chunk | Best quality, all languages |
| `kb-lg` | KB Whisper Large | 1.5 GB | ~2s/chunk | Best Swedish accuracy |

Switch models from Settings — no restart needed.

## Word hints

If Whisper consistently misses certain words (names, technical terms), add them via **Settings → Vocabulary → Edit**, or edit `~/.config/ptt/prompt.txt` directly:

```
# One per line or comma-separated:
Claude Code, Kajabi, PTT
```

## How it works

1. **Hold hotkey** → recording starts (with 400ms pre-roll buffer)
2. **Speak** → audio monitored for natural pauses
3. **Pause detected** → chunk transcribed and pasted immediately
4. **Continue speaking** → more chunks transcribed and pasted
5. **Release hotkey** → final chunk transcribed, done

The mic auto-calibrates at startup by measuring 2 seconds of ambient noise.

## Files

| Path | Purpose |
|------|---------|
| `~/.config/ptt/settings.json` | All settings |
| `~/.config/ptt/prompt.txt` | Word hints for recognition |
| `~/.config/ptt/polish_prompt.txt` | Custom instructions for text polish AI |
| `~/Library/Logs/ptt.log` | Log file |
| `~/Library/LaunchAgents/com.ptt.transcription.plist` | Auto-start config |

## Troubleshooting

**Text not pasting?** Check Accessibility permissions. Text is always copied to clipboard as fallback — ⌘V manually.

**Wrong microphone?** Run `ptt --list-devices` and use `ptt --device N`. PTT auto-detects the built-in MacBook mic.

**Too sensitive / not sensitive enough?** Click **Calibrate** in Settings, or restart PTT in a quieter/louder environment.

**Polish not working?** Check that your Groq API key is set (Settings → Text polish). The key should start with `gsk_`. Also verify your Groq account is active at [console.groq.com](https://console.groq.com).

**PTT doesn't start?** Check `~/Library/Logs/ptt.log` for errors. Reset settings with `ptt --reset`.

## Uninstall

```bash
ptt --uninstall                              # Remove auto-start
rm -rf ~/.config/ptt                         # Remove settings
rm -f ~/Library/Logs/ptt.log                 # Remove log
rm -rf /Applications/PTT.app                 # If installed as app
rm -f ~/.local/bin/ptt                       # If installed via script
rm -rf ~/.local/share/ptt                    # If installed via script
```

## License

MIT
