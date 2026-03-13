#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "mlx-whisper>=0.4",
#     "sounddevice>=0.5",
#     "numpy>=1.26",
#     "rumps>=0.4",
#     "pyobjc-framework-Quartz>=10.0",
#     "pyobjc-framework-ApplicationServices>=10.0",
# ]
# ///
"""PTT -- Push-to-Talk Transcription for macOS (Apple Silicon).

Hold a key, speak, release -- text appears where your cursor is.
Runs on-device with MLX Whisper. No cloud, no API keys.

Usage:
  uv run ptt.py                     Start with defaults
  uv run ptt.py --list-devices      Show audio input devices
  uv run ptt.py --install           Auto-start on login
  uv run ptt.py --uninstall         Remove auto-start
  uv run ptt.py --reset             Reset settings to defaults
"""

import argparse
import collections
import ctypes
import json
import logging
import os
import queue
import signal
import shutil
import subprocess
import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CONFIG_DIR = os.path.expanduser("~/.config/ptt")
SETTINGS_PATH = os.path.join(CONFIG_DIR, "settings.json")
PROMPT_PATH = os.path.join(CONFIG_DIR, "prompt.txt")
POLISH_PROMPT_PATH = os.path.join(CONFIG_DIR, "polish_prompt.txt")
LOG_DIR = os.path.expanduser("~/Library/Logs")
LOG_PATH = os.path.join(LOG_DIR, "ptt.log")
PID_PATH = os.path.join(CONFIG_DIR, "ptt.pid")
LAUNCHAGENT_LABEL = "com.ptt.transcription"
PLIST_PATH = os.path.expanduser(
    f"~/Library/LaunchAgents/{LAUNCHAGENT_LABEL}.plist"
)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

# Truncate log if older than 1 hour or larger than 256 KB
try:
    if os.path.isfile(LOG_PATH):
        age = time.time() - os.path.getmtime(LOG_PATH)
        size = os.path.getsize(LOG_PATH)
        if age > 3600 or size > 256_000:
            open(LOG_PATH, "w").close()
except Exception:
    pass

_file_handler = logging.FileHandler(LOG_PATH)
_log_handlers = [_file_handler]
if os.isatty(2):  # Only add stderr handler when running in a terminal
    _log_handlers.append(logging.StreamHandler())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=_log_handlers,
)
log = logging.getLogger("ptt")


def set_logging_enabled(enabled: bool):
    """Enable or disable file logging."""
    if enabled:
        _file_handler.setLevel(logging.INFO)
    else:
        _file_handler.setLevel(logging.CRITICAL + 1)

# Suppress noisy third-party loggers
for _name in ("httpx", "huggingface_hub", "tqdm"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = {
    "kb-sm": {
        "repo": "Leonidng/kb-whisper-small-mlx",
        "label_sv": "KB Small — snabb svenska",
        "label_en": "KB Small — fast Swedish",
        "swedish_only": True,
    },
    "small": {
        "repo": "mlx-community/whisper-small-mlx",
        "label_sv": "Small — alla språk",
        "label_en": "Small — all languages",
        "swedish_only": False,
    },
    "turbo": {
        "repo": "mlx-community/whisper-large-v3-turbo",
        "label_sv": "Turbo — bäst kvalitet",
        "label_en": "Turbo — best quality",
        "swedish_only": False,
    },
    "kb-lg": {
        "repo": "bratland/kb-whisper-large-mlx",
        "label_sv": "KB Large — bäst på svenska",
        "label_en": "KB Large — best Swedish",
        "swedish_only": True,
    },
}


def model_label(key: str) -> str:
    info = MODELS[key]
    return info["label_sv"] if system_has_swedish() else info["label_en"]

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.1
MIN_SPEECH_SECONDS = 0.3
PAUSE_SECONDS = 1.0
PRE_ROLL_SECONDS = 0.5
CALIBRATION_SECONDS = 2.0
THRESHOLD_MULTIPLIER = 3.0
MIN_THRESHOLD = 0.002
ANIM_INTERVAL = 3

# ---------------------------------------------------------------------------
# Hotkeys
# ---------------------------------------------------------------------------

HOTKEYS = {
    "alt_r":  {"code": 61, "flag": 1 << 19, "label_sv": "Höger ⌥ Option", "label_en": "Right ⌥ Option"},
    "alt":    {"code": 58, "flag": 1 << 19, "label_sv": "Vänster ⌥ Option", "label_en": "Left ⌥ Option"},
    "ctrl_r": {"code": 62, "flag": 1 << 18, "label_sv": "Höger ⌃ Control", "label_en": "Right ⌃ Control"},
    "ctrl":   {"code": 59, "flag": 1 << 18, "label_sv": "Vänster ⌃ Control", "label_en": "Left ⌃ Control"},
}


def hotkey_label(key: str) -> str:
    info = HOTKEYS[key]
    return info["label_sv"] if system_has_swedish() else info["label_en"]

# ---------------------------------------------------------------------------
# Waveform icon definitions
# ---------------------------------------------------------------------------

WAVE_IDLE = [0.15, 0.30, 0.50, 0.30, 0.15]
WAVE_REC = [
    [0.35, 0.75, 0.45, 0.90, 0.50],
    [0.50, 0.40, 0.85, 0.35, 0.70],
    [0.70, 0.90, 0.35, 0.65, 0.40],
    [0.45, 0.55, 0.70, 0.45, 0.85],
]
WAVE_BUSY = [0.25, 0.50, 0.25, 0.50, 0.25]

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

_DEFAULT_SETTINGS = {
    "language": None,  # auto-detect
    "model": "kb-sm",
    "hotkey": "alt_r",
    "device": None,
    "autostart": False,
    "intro_shown": False,
    "logging": True,
    "sounds": False,
    "restore_clipboard": True,
    "history": True,
}


_SYS_SWEDISH: bool | None = None


def system_has_swedish() -> bool:
    global _SYS_SWEDISH
    if _SYS_SWEDISH is None:
        try:
            out = subprocess.run(
                ["defaults", "read", "-g", "AppleLanguages"],
                capture_output=True, text=True,
            ).stdout.lower()
            _SYS_SWEDISH = "sv" in out
        except Exception:
            _SYS_SWEDISH = False
    return _SYS_SWEDISH


def detect_system_language() -> str:
    return "sv" if system_has_swedish() else "en"


def _t(sv: str, en: str) -> str:
    """Return Swedish or English string based on system language."""
    return sv if system_has_swedish() else en


def load_settings() -> dict:
    settings = dict(_DEFAULT_SETTINGS)
    if os.path.isfile(SETTINGS_PATH):
        try:
            with open(SETTINGS_PATH) as f:
                saved = json.load(f)
            settings.update(saved)
        except Exception:
            pass
    if settings["language"] is None:
        settings["language"] = detect_system_language()
    # Migrate old model keys
    if settings["model"] == "kb-sv":
        settings["model"] = "kb-sm"
    if settings["model"] not in MODELS:
        settings["model"] = _DEFAULT_SETTINGS["model"]
    return settings


def save_settings(settings: dict):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


# ---------------------------------------------------------------------------
# Prompt / word hints
# ---------------------------------------------------------------------------


def _read_commented_file(path: str, sep: str = ", ") -> str | None:
    """Read a config file, stripping comments (#) and blank lines."""
    if not os.path.isfile(path):
        return None
    lines = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    text = sep.join(lines).strip()
    return text if text else None


def read_prompt() -> str | None:
    return _read_commented_file(PROMPT_PATH, sep=", ")


def _ensure_config_file(path: str, sv_template: str, en_template: str):
    """Create a config file with localized template if it doesn't exist."""
    if not os.path.isfile(path):
        with open(path, "w") as f:
            f.write(_t(sv_template, en_template))


def ensure_prompt_file():
    _ensure_config_file(
        PROMPT_PATH,
        "# PTT — Ordlista\n"
        "#\n"
        "# Skriv ord och namn som ofta hörs fel.\n"
        "# En post per rad, eller kommaseparerat.\n"
        "# Rader som börjar med # ignoreras.\n"
        "#\n"
        "# Exempel:\n"
        "# Claude Code, Kajabi\n",
        "# PTT — Word hints\n"
        "#\n"
        "# Add words and names that are often misheard.\n"
        "# One per line, or comma-separated.\n"
        "# Lines starting with # are ignored.\n"
        "#\n"
        "# Example:\n"
        "# Claude Code, Kajabi\n",
    )


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------

_ARTIFACTS = frozenset({
    "thank you", "thanks for watching", "subscribe", "you",
    "tack för att ni tittade", "tack för att ni tittar",
    "undertextning", "musik", "textning", "textning.nu",
    "untertitelung", "sous-titrage", "amara.org",
    "text", ".", "..", "...",
})


def is_hallucination(text: str) -> bool:
    t = text.strip().lower()
    if not t or len(t) <= 1:
        return True
    if t[0] in "([♪♫":
        return True
    if t in _ARTIFACTS:
        return True
    words = t.split()
    if len(words) >= 4:
        half = len(words) // 2
        if words[:half] == words[half : half * 2]:
            return True
    return False


# ---------------------------------------------------------------------------
# macOS helpers
# ---------------------------------------------------------------------------


def check_accessibility() -> bool:
    try:
        lib = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework"
            "/ApplicationServices"
        )
        lib.AXIsProcessTrusted.restype = ctypes.c_bool
        return lib.AXIsProcessTrusted()
    except Exception:
        return True


def find_builtin_mic() -> tuple[int | None, str]:
    import sounddevice as sd

    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] <= 0:
            continue
        name = d["name"].lower()
        # Skip speakers / output devices that happen to match
        if any(w in name for w in ("högtalare", "speaker", "output", "hdmi")):
            continue
        if any(w in name for w in ("macbook", "built-in", "mikrofon", "internal")):
            return i, d["name"]
    idx = sd.default.device[0]
    if idx is not None:
        return idx, sd.query_devices(idx)["name"]
    return None, "Unknown"


_paste_generation = 0


def paste_text(text: str, restore_clipboard: bool = True):
    global _paste_generation
    from AppKit import NSPasteboard, NSPasteboardTypeString
    from Quartz import (
        CGEventCreateKeyboardEvent, CGEventPost,
        CGEventSetFlags, kCGEventFlagMaskCommand, kCGHIDEventTap,
    )

    _paste_generation += 1
    gen = _paste_generation

    pb = NSPasteboard.generalPasteboard()

    # Save current clipboard before overwriting
    old_contents = None
    if restore_clipboard:
        old_contents = pb.stringForType_(NSPasteboardTypeString)

    pb.clearContents()
    pb.setString_forType_(text, NSPasteboardTypeString)
    time.sleep(0.03)
    for pressed in (True, False):
        ev = CGEventCreateKeyboardEvent(None, 9, pressed)
        CGEventSetFlags(ev, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, ev)
        if pressed:
            time.sleep(0.03)

    # Restore clipboard after paste lands (skipped if a newer paste arrived)
    if restore_clipboard and old_contents is not None:
        def _restore():
            time.sleep(0.3)
            if _paste_generation != gen:
                return  # a newer paste superseded this one
            pb.clearContents()
            pb.setString_forType_(old_contents, NSPasteboardTypeString)
        threading.Thread(target=_restore, daemon=True).start()


# ---------------------------------------------------------------------------
# Icon generation
# ---------------------------------------------------------------------------


def make_waveform(bars: list[float], size: int = 18):
    from AppKit import NSBezierPath, NSColor, NSImage
    from Foundation import NSMakeRect, NSMakeSize

    img = NSImage.alloc().initWithSize_(NSMakeSize(size, size))
    img.lockFocus()

    n = len(bars)
    bar_w, gap = 2.0, 1.5
    total_w = n * bar_w + (n - 1) * gap
    x0 = (size - total_w) / 2.0

    NSColor.blackColor().setFill()
    for i, h in enumerate(bars):
        x = x0 + i * (bar_w + gap)
        bh = max(2.0, h * size * 0.78)
        y = (size - bh) / 2.0
        NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
            NSMakeRect(x, y, bar_w, bh), bar_w / 2, bar_w / 2
        ).fill()

    img.unlockFocus()
    img.setTemplate_(True)
    return img


# ---------------------------------------------------------------------------
# LaunchAgent
# ---------------------------------------------------------------------------


def _find_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    for p in ("~/.local/bin/uv", "~/.cargo/bin/uv"):
        expanded = os.path.expanduser(p)
        if os.path.isfile(expanded):
            return expanded
    return "uv"


def _write_launchagent():
    uv_path = _find_uv()
    script_path = os.path.abspath(__file__)

    plist = f"""\
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" \
"http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHAGENT_LABEL}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{uv_path}</string>
        <string>run</string>
        <string>--script</string>
        <string>{script_path}</string>
    </array>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>
    <key>StandardOutPath</key>
    <string>{LOG_PATH}</string>
    <key>StandardErrorPath</key>
    <string>{LOG_PATH}</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin:{os.path.dirname(uv_path)}</string>
        <key>UV_HTTP_TIMEOUT</key>
        <string>300</string>
    </dict>
</dict>
</plist>"""

    os.makedirs(os.path.dirname(PLIST_PATH), exist_ok=True)
    if os.path.exists(PLIST_PATH):
        subprocess.run(["launchctl", "unload", PLIST_PATH], capture_output=True)
    with open(PLIST_PATH, "w") as f:
        f.write(plist)
    subprocess.run(["launchctl", "load", PLIST_PATH], capture_output=True)


def _remove_launchagent():
    if os.path.exists(PLIST_PATH):
        subprocess.run(["launchctl", "unload", PLIST_PATH], capture_output=True)
        os.remove(PLIST_PATH)


def set_autostart(enabled: bool):
    if enabled:
        _write_launchagent()
        log.info("LaunchAgent installed")
    else:
        _remove_launchagent()
        log.info("LaunchAgent removed")


# ---------------------------------------------------------------------------
# Instance management
# ---------------------------------------------------------------------------


def _kill_other_instances():
    """Kill any previous PTT instance using a PID file."""
    if os.path.isfile(PID_PATH):
        try:
            old_pid = int(open(PID_PATH).read().strip())
            if old_pid != os.getpid():
                os.kill(old_pid, signal.SIGTERM)
                log.info("Killed previous PTT process %d", old_pid)
                time.sleep(0.5)
        except (ValueError, ProcessLookupError, PermissionError):
            pass
        except Exception:
            pass

    with open(PID_PATH, "w") as f:
        f.write(str(os.getpid()))

    # Clean up PID file on exit
    import atexit

    def _cleanup_pid():
        try:
            os.remove(PID_PATH)
        except OSError:
            pass

    atexit.register(_cleanup_pid)


# ---------------------------------------------------------------------------
# Groq polish (clean up spoken text)
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"
POLISH_PROMPT = (
    "Du får talspråkig text från en rösttranskribering. "
    "Gör om den till ren, tydlig skriven text. "
    "Behåll innebörden exakt. Ta bort upprepningar, fyllnadsord och stakningar. "
    "Svara BARA med den rensade texten, inget annat."
)


_groq_key_cache: str | None | bool = False  # False = not yet looked up


def _get_groq_key() -> str | None:
    """Read Groq API key from macOS Keychain (cached, tries account then service)."""
    global _groq_key_cache
    if _groq_key_cache is not False:
        return _groq_key_cache
    for flag in ("-a", "-s"):
        try:
            result = subprocess.run(
                ["security", "find-generic-password", flag, "groq", "-w"],
                capture_output=True, text=True,
            )
            key = result.stdout.strip()
            if key and not result.returncode:
                _groq_key_cache = key
                return key
        except Exception:
            pass
    _groq_key_cache = None
    return None


def _set_groq_key(key: str) -> bool:
    """Store Groq API key in macOS Keychain."""
    global _groq_key_cache
    subprocess.run(
        ["security", "delete-generic-password", "-a", "groq"],
        capture_output=True,
    )
    result = subprocess.run(
        ["security", "add-generic-password", "-a", "groq", "-s", "ptt", "-w", key],
        capture_output=True,
    )
    ok = result.returncode == 0
    _groq_key_cache = key if ok else False
    return ok


def _delete_groq_key() -> bool:
    """Remove Groq API key from macOS Keychain."""
    global _groq_key_cache
    result = subprocess.run(
        ["security", "delete-generic-password", "-a", "groq"],
        capture_output=True,
    )
    _groq_key_cache = False
    return result.returncode == 0


def _read_polish_prompt(language: str) -> str:
    """Read custom polish prompt or return default."""
    custom = _read_commented_file(POLISH_PROMPT_PATH, sep=" ")
    if custom:
        return custom
    if language == "en":
        return (
            "You receive spoken text from a voice transcription. "
            "Clean it into clear, polished written text. "
            "Keep the meaning exactly. Remove repetitions, filler words, and stutters. "
            "Respond with ONLY the cleaned text, nothing else."
        )
    return POLISH_PROMPT


def _ensure_polish_prompt_file():
    """Create default polish prompt file if it doesn't exist."""
    _ensure_config_file(
        POLISH_PROMPT_PATH,
        "# PTT — Polerings-prompt\n"
        "#\n"
        "# Instruktioner till AI:n som polerar din text.\n"
        "# Rader som börjar med # ignoreras.\n"
        "# Lämna tom för standardprompt.\n"
        "#\n"
        "# Standardprompt (svenska):\n"
        "# Du får talspråkig text från en rösttranskribering.\n"
        "# Gör om den till ren, tydlig skriven text.\n"
        "# Behåll innebörden exakt.\n"
        "# Ta bort upprepningar, fyllnadsord och stakningar.\n"
        "# Svara BARA med den rensade texten, inget annat.\n",
        "# PTT — Polish prompt\n"
        "#\n"
        "# Instructions for the AI that polishes your text.\n"
        "# Lines starting with # are ignored.\n"
        "# Leave empty for default prompt.\n"
        "#\n"
        "# Default prompt (English):\n"
        "# You receive spoken text from a voice transcription.\n"
        "# Clean it into clear, polished written text.\n"
        "# Keep the meaning exactly.\n"
        "# Remove repetitions, filler words, and stutters.\n"
        "# Respond with ONLY the cleaned text, nothing else.\n",
    )


def polish_text(text: str, language: str) -> str | None:
    """Send text to Groq to clean up spoken language."""
    import urllib.request
    import urllib.error

    api_key = _get_groq_key()
    if not api_key:
        log.warning("No Groq API key — add via Settings or: security add-generic-password -a groq -s ptt -w KEY")
        return None

    system = _read_polish_prompt(language)

    body = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "temperature": 0.3,
    }).encode()

    req = urllib.request.Request(
        GROQ_ENDPOINT,
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        log.error("Groq API error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Settings window (native NSWindow via PyObjC)
# ---------------------------------------------------------------------------

_ptt_ref = None  # Module-level ref for ObjC delegate callbacks
_SettingsDelegate = None


def _get_delegate_class():
    """Lazily create the NSObject subclass for settings window callbacks."""
    global _SettingsDelegate
    if _SettingsDelegate is not None:
        return _SettingsDelegate

    from Foundation import NSObject

    class _SD(NSObject):
        def langChanged_(self, sender):
            if _ptt_ref:
                idx = sender.indexOfSelectedItem()
                codes = _ptt_ref._win_lang_codes
                if 0 <= idx < len(codes):
                    _ptt_ref._set_language(codes[idx])

        def modelChanged_(self, sender):
            if _ptt_ref:
                idx = sender.indexOfSelectedItem()
                keys = _ptt_ref._win_model_keys
                if 0 <= idx < len(keys):
                    _ptt_ref._set_model(keys[idx])

        def hotkeyChanged_(self, sender):
            if _ptt_ref:
                idx = sender.indexOfSelectedItem()
                keys = _ptt_ref._win_hotkey_keys
                if 0 <= idx < len(keys):
                    _ptt_ref._set_hotkey(keys[idx])

        def autostartChanged_(self, sender):
            if _ptt_ref:
                _ptt_ref._set_autostart(sender.state() == 1)

        def calibrateClicked_(self, sender):
            if _ptt_ref:
                _ptt_ref._on_recalibrate()

        def editPromptClicked_(self, sender):
            if _ptt_ref:
                _ptt_ref._on_edit_prompt()

        def saveGroqKey_(self, sender):
            if _ptt_ref:
                _ptt_ref._on_save_groq_key()

        def removeGroqKey_(self, sender):
            if _ptt_ref:
                _ptt_ref._on_remove_groq_key()

        def loggingChanged_(self, sender):
            if _ptt_ref:
                _ptt_ref._set_logging(sender.state() == 1)

        def editPolishPrompt_(self, sender):
            if _ptt_ref:
                _ptt_ref._on_edit_polish_prompt()

        def savePolishPrompt_(self, sender):
            if _ptt_ref:
                _ptt_ref._on_save_polish_prompt()

        def clearLog_(self, sender):
            if _ptt_ref:
                _ptt_ref._on_clear_log()

        def openLog_(self, sender):
            if _ptt_ref:
                _ptt_ref._on_open_log()

        def clipboardChanged_(self, sender):
            if _ptt_ref:
                _ptt_ref._set_setting("restore_clipboard", sender.state() == 1)

        def soundsChanged_(self, sender):
            if _ptt_ref:
                _ptt_ref._set_setting("sounds", sender.state() == 1)

        def historyChanged_(self, sender):
            if _ptt_ref:
                _ptt_ref._set_setting("history", sender.state() == 1)

    _SettingsDelegate = _SD
    return _SettingsDelegate


# ---------------------------------------------------------------------------
# Transcription subprocess
# ---------------------------------------------------------------------------


def _transcription_worker_main(model_repo: str, language: str, conn):
    """Worker process: loads the Whisper model and handles transcription requests.
    Runs in a separate subprocess so the model memory can be released when idle.
    """
    import mlx_whisper

    mlx_whisper.transcribe(
        np.zeros(SAMPLE_RATE, dtype=np.float32),
        path_or_hf_repo=model_repo,
        language=language,
    )
    conn.send({"status": "ready"})

    while True:
        try:
            req = conn.recv()
        except (EOFError, OSError):
            break
        if req is None:
            break
        audio, kwargs = req
        try:
            result = mlx_whisper.transcribe(audio, path_or_hf_repo=model_repo, **kwargs)
            conn.send({"status": "ok", "result": result})
        except Exception as e:
            conn.send({"status": "error", "error": str(e)})


class TranscriptionProcess:
    """Manages a Whisper model in a subprocess for memory isolation.

    The subprocess starts on first use and is killed automatically after
    IDLE_TIMEOUT seconds of inactivity, freeing model memory. It restarts
    automatically on next use. Model switches kill the old process first,
    so only one model is ever in memory.
    """

    IDLE_TIMEOUT = 5 * 60  # seconds

    def __init__(self):
        self._lock = threading.Lock()
        self._proc = None
        self._conn = None
        self._model_repo = None
        self._idle_timer = None

    def warm_up(self, model_repo: str, language: str):
        """Pre-load the model (blocks until ready). Restarts if model changed."""
        with self._lock:
            if self._proc is not None and self._proc.is_alive() and self._model_repo == model_repo:
                return
            self._start(model_repo, language)

    def transcribe(self, audio, model_repo: str, **kwargs):
        """Send audio to worker and return result dict, or None on error."""
        self._cancel_idle_timer()
        with self._lock:
            if self._proc is None or not self._proc.is_alive() or self._model_repo != model_repo:
                language = kwargs.get("language", "en")
                try:
                    self._start(model_repo, language)
                except Exception as e:
                    log.error("Failed to start transcription worker: %s", e)
                    return None
            try:
                self._conn.send((audio, kwargs))
                msg = self._conn.recv()
            except Exception as e:
                log.error("Transcription IPC error: %s", e)
                self._kill()
                return None
        self._schedule_idle_timer()
        if msg and msg.get("status") == "ok":
            return msg["result"]
        log.error("Worker error: %s", msg.get("error") if msg else "no response")
        return None

    def stop(self):
        """Stop the worker subprocess and release model memory."""
        self._cancel_idle_timer()
        with self._lock:
            self._kill()

    def _start(self, model_repo: str, language: str):
        """Launch a new worker subprocess. Must be called with _lock held."""
        self._kill()
        import multiprocessing
        ctx = multiprocessing.get_context("spawn")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        proc = ctx.Process(
            target=_transcription_worker_main,
            args=(model_repo, language, child_conn),
            daemon=True,
        )
        proc.start()
        child_conn.close()
        try:
            msg = parent_conn.recv()
        except Exception as e:
            proc.terminate()
            raise RuntimeError(f"Worker startup error: {e}") from e
        if not msg or msg.get("status") != "ready":
            proc.terminate()
            raise RuntimeError(f"Worker not ready: {msg}")
        self._proc = proc
        self._conn = parent_conn
        self._model_repo = model_repo
        log.info("Transcription worker ready (%s)", model_repo)

    def _kill(self):
        """Terminate the subprocess. Must be called with _lock held."""
        if self._proc and self._proc.is_alive():
            try:
                self._conn.send(None)
            except Exception:
                pass
            self._proc.join(timeout=3)
            if self._proc.is_alive():
                self._proc.terminate()
                self._proc.join(timeout=2)
            log.info("Transcription worker stopped — model memory freed")
        self._proc = None
        self._conn = None
        self._model_repo = None

    def _schedule_idle_timer(self):
        self._cancel_idle_timer()
        t = threading.Timer(self.IDLE_TIMEOUT, self._on_idle)
        t.daemon = True
        t.start()
        self._idle_timer = t

    def _cancel_idle_timer(self):
        t, self._idle_timer = self._idle_timer, None
        if t:
            t.cancel()

    def _on_idle(self):
        log.info("No activity for %d min — unloading model to free memory", self.IDLE_TIMEOUT // 60)
        with self._lock:
            self._kill()


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


class PTTApp:
    def __init__(self, settings: dict):
        self.settings = settings
        self.model_key = settings["model"]
        self.model_repo = MODELS[self.model_key]["repo"]
        self.language = settings["language"]
        self.hotkey = settings["hotkey"]
        self.device_idx = settings["device"]
        self.device_name = ""

        hk = HOTKEYS[self.hotkey]
        self.hotkey_code = hk["code"]
        self.hotkey_flag = hk["flag"]

        self.recording = False
        self.ready = False
        self.polish_mode = False
        self.silence_threshold = MIN_THRESHOLD

        self.audio_queue: queue.Queue = queue.Queue()
        self.block_size = int(SAMPLE_RATE * BLOCK_DURATION)
        self.pre_roll: collections.deque = collections.deque(
            maxlen=int(PRE_ROLL_SECONDS / BLOCK_DURATION)
        )

        self._rec_thread: threading.Thread | None = None
        self._stream = None
        self._app = None

        # Icons
        self._icon_idle = None
        self._icon_rec: list = []
        self._icon_busy = None

        # Settings window
        self._settings_win = None
        self._settings_delegate = None
        self._win_lang_codes: list = []
        self._win_model_keys: list = []
        self._win_hotkey_keys: list = []
        self._groq_key_field = None
        self._groq_status_label = None
        self._polish_prompt_tv = None
        self._history: collections.deque = collections.deque(maxlen=10)
        self._worker = TranscriptionProcess()

    # ---- Persistence -----------------------------------------------------

    def _save(self):
        self.settings.update({
            "language": self.language,
            "model": self.model_key,
            "hotkey": self.hotkey,
        })
        # Don't persist auto-detected device — keep it as None for auto-detect
        save_settings(self.settings)

    # ---- Icons -----------------------------------------------------------

    def _create_icons(self):
        self._icon_idle = make_waveform(WAVE_IDLE)
        self._icon_rec = [make_waveform(f) for f in WAVE_REC]
        self._icon_busy = make_waveform(WAVE_BUSY)

    def _paste(self, text: str):
        """Paste text, respecting clipboard restore setting."""
        restore = self.settings.get("restore_clipboard", True)
        paste_text(text, restore_clipboard=restore)
        self._add_history(text.strip())

    def _play_sound(self, name: str):
        """Play a system sound if sounds are enabled."""
        if not self.settings.get("sounds", False):
            return
        try:
            from AppKit import NSSound
            sound = NSSound.soundNamed_(name)
            if sound:
                sound.play()
        except Exception:
            pass

    def _add_history(self, text: str):
        if not text or not self.settings.get("history", True):
            return
        self._history.appendleft(text)
        self._update_history_menu()

    def _update_history_menu(self):
        from PyObjCTools import AppHelper

        def _do_update():
            import rumps
            try:
                menu = self._app.menu
                hist_key = _t("Senaste", "Recent")
                if hist_key in menu:
                    del menu[hist_key]
                if not self._history:
                    return
                sub = rumps.MenuItem(hist_key)
                for text in list(self._history):
                    display = (text[:55] + "…") if len(text) > 55 else text
                    item = rumps.MenuItem(display, callback=self._make_copy_cb(text))
                    sub[display] = item
                menu.insert_after(_t("Inställningar…", "Settings…"), sub)
            except Exception:
                log.debug("History menu update failed", exc_info=True)

        AppHelper.callAfter(_do_update)

    def _make_copy_cb(self, text: str):
        def cb(_):
            from AppKit import NSPasteboard, NSPasteboardTypeString
            pb = NSPasteboard.generalPasteboard()
            pb.clearContents()
            pb.setString_forType_(text, NSPasteboardTypeString)
            self._notify(_t("Kopierat", "Copied"), (text[:55] + "…") if len(text) > 55 else text)
        return cb

    def _show_icon(self, nsimage):
        try:
            si = self._app._nsapp.nsstatusitem
            if si:
                b = si.button()
                if b:
                    b.setImage_(nsimage)
                    b.setTitle_("")
        except Exception:
            pass

    # ---- Menu ------------------------------------------------------------

    def run(self):
        import rumps
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

        global _ptt_ref
        _ptt_ref = self

        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory
        )

        self._create_icons()
        ensure_prompt_file()

        self._app = rumps.App("PTT", title="PTT", quit_button=None)

        self._app.menu = [
            rumps.MenuItem(_t("Inställningar…", "Settings…"), callback=self._on_open_settings),
            None,
            rumps.MenuItem(_t("Avsluta", "Quit"), callback=self._on_quit),
        ]

        # Accessibility check
        if not check_accessibility():
            log.warning("Accessibility permission missing")
            rumps.alert(
                title=_t("PTT behöver Accessibility", "PTT needs Accessibility"),
                message=_t(
                    "PTT behöver behörighet för att läsa tangenter "
                    "och klistra in text.\n\n"
                    "Lägg till appen i:\n"
                    "Inställningar → Integritet och säkerhet → Hjälpmedel",
                    "PTT needs permission to read hotkeys "
                    "and paste text.\n\n"
                    "Add the app in:\n"
                    "Settings → Privacy & Security → Accessibility",
                ),
                ok=_t("Öppna inställningar", "Open Settings"),
            )
            subprocess.Popen([
                "open",
                "x-apple.systempreferences:"
                "com.apple.preference.security?Privacy_Accessibility",
            ])

        threading.Thread(target=self._init, daemon=True).start()
        self._app.run()

    def _notify(self, title: str, message: str):
        try:
            import rumps
            rumps.notification("PTT", title, message, sound=False)
        except Exception:
            pass

    # ---- Init ------------------------------------------------------------

    def _wait_for_statusitem(self, timeout: float = 10.0):
        """Poll until rumps has created the NSStatusItem."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                si = self._app._nsapp.nsstatusitem
                if si is not None:
                    # Give the main thread a moment to finish layout
                    time.sleep(0.3)
                    return True
            except AttributeError:
                pass
            time.sleep(0.1)
        log.warning("Timed out waiting for NSStatusItem")
        return False

    def _init(self):
        import sounddevice as sd
        from AppKit import NSEvent

        self._wait_for_statusitem()

        try:
            self._show_icon(self._icon_busy)

            log.info("Loading model: %s", self.model_repo)
            self._worker.warm_up(self.model_repo, self.language)
            log.info("Model loaded")

            if self.device_idx is not None:
                self.device_name = sd.query_devices(self.device_idx)["name"]
            else:
                self.device_idx, self.device_name = find_builtin_mic()
            log.info("Mic: %s (device %s)", self.device_name, self.device_idx)

            self._calibrate()

            self._stream = self._create_stream()
            self._stream.start()

            mask = 1 << 12  # NSEventMaskFlagsChanged

            def on_global(event):
                self._on_modifier(event)

            def on_local(event):
                self._on_modifier(event)
                return event

            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, on_global)
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, on_local)

            prompt = read_prompt()
            if prompt:
                log.info("Prompt: %s", prompt[:80])

            # Start mic device monitor
            threading.Thread(target=self._monitor_mic, daemon=True).start()

            self.ready = True
            self._show_icon(self._icon_idle)

            label = hotkey_label(self.hotkey)
            log.info("PTT ready")

            # First-run intro
            if not self.settings.get("intro_shown"):
                self._notify(
                    _t("Redo!", "Ready!"),
                    _t(
                        f"Håll {label} och prata.\n"
                        "Släpp för att transkribera och klistra in.\n"
                        "Klicka ikonen för inställningar.",
                        f"Hold {label} and speak.\n"
                        "Release to transcribe and paste.\n"
                        "Click the icon for settings.",
                    ),
                )
                self.settings["intro_shown"] = True
                save_settings(self.settings)
            else:
                self._notify(
                    _t("Redo!", "Ready!"),
                    _t(f"Håll {label} och prata", f"Hold {label} and speak"),
                )

        except Exception as e:
            log.exception("Init failed")
            self._notify(_t("Kunde inte starta", "Failed to start"), str(e))

    def _create_stream(self):
        import sounddevice as sd
        return sd.InputStream(
            samplerate=SAMPLE_RATE, channels=1,
            blocksize=self.block_size, device=self.device_idx,
            callback=self._audio_cb,
        )

    def _calibrate(self):
        import sounddevice as sd

        log.info("Calibrating…")
        rec = sd.rec(
            int(SAMPLE_RATE * CALIBRATION_SECONDS),
            samplerate=SAMPLE_RATE, channels=1, device=self.device_idx,
        )
        sd.wait()
        ambient = float(np.sqrt(np.mean(rec ** 2)))
        self.silence_threshold = max(ambient * THRESHOLD_MULTIPLIER, MIN_THRESHOLD)
        log.info("Ambient=%.5f -> threshold=%.5f", ambient, self.silence_threshold)

    def _monitor_mic(self):
        """Poll for default input device changes every 5s."""
        import sounddevice as sd
        last_default = sd.default.device[0]
        while True:
            time.sleep(5)
            try:
                cur = sd.default.device[0]
                if cur != last_default and not self.recording:
                    last_default = cur
                    new_name = sd.query_devices(cur)["name"]
                    log.info("Mic changed: %s (device %d)", new_name, cur)
                    self.device_idx = cur
                    self.device_name = new_name
                    try:
                        self._stream.stop()
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = self._create_stream()
                    self._stream.start()
                    self._calibrate()
                    self._notify(
                        _t("Mikrofon bytt", "Microphone switched"),
                        new_name,
                    )
            except (sd.PortAudioError, OSError) as e:
                log.debug("Mic monitor error: %s", e)
            except Exception:
                log.exception("Unexpected mic monitor error")

    # ---- Hotkey ----------------------------------------------------------

    def _on_modifier(self, event):
        if not self.ready or event.keyCode() != self.hotkey_code:
            return
        flags = event.modifierFlags()
        pressed = bool(flags & self.hotkey_flag)
        cmd_held = bool(flags & (1 << 20))  # NSEventModifierFlagCommand

        if pressed and not self.recording:
            self.polish_mode = cmd_held
            if self.polish_mode:
                log.info("Polish mode ON")
            self._start_rec()
        elif not pressed and self.recording:
            self._stop_rec()

    # ---- Recording -------------------------------------------------------

    def _audio_cb(self, indata, frames, time_info, status):
        if status:
            log.warning("Audio: %s", status)
        block = indata.copy()
        if self.recording:
            self.audio_queue.put(block)
        else:
            self.pre_roll.append(block)

    def _start_rec(self):
        if self._rec_thread and self._rec_thread.is_alive():
            return

        # Restart audio stream if it died
        if self._stream and not self._stream.active:
            log.warning("Audio stream inactive, restarting")
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = self._create_stream()
            self._stream.start()

        self.recording = True
        self._show_icon(self._icon_rec[0])
        self._play_sound("Tink")
        log.info("REC start")

        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
        for block in self.pre_roll:
            self.audio_queue.put(block)
        self.pre_roll.clear()

        self._rec_thread = threading.Thread(target=self._rec_loop, daemon=True)
        self._rec_thread.start()

    def _stop_rec(self):
        self.recording = False
        self._play_sound("Pop")
        log.info("REC stop")

    def _rec_loop(self):
        speech_buf: list = []
        silent_blocks = 0
        has_speech = False
        block_n = 0
        frame = 0
        polish_parts: list[str] = []
        empty_polls = 0
        prev_text = ""  # context for cross-chunk continuity

        while self.recording:
            try:
                block = self.audio_queue.get(timeout=0.05)
                empty_polls = 0
            except queue.Empty:
                empty_polls += 1
                # If no audio for 3+ seconds, stream may be dead
                if empty_polls > 60:
                    log.warning("No audio data for 3s — stream may be dead")
                    empty_polls = 0
                continue

            block_n += 1
            if block_n % ANIM_INTERVAL == 0:
                frame = (frame + 1) % len(self._icon_rec)
                self._show_icon(self._icon_rec[frame])

            rms = float(np.sqrt(np.mean(block ** 2)))

            if rms > self.silence_threshold:
                speech_buf.append(block)
                silent_blocks = 0
                has_speech = True
            elif has_speech:
                speech_buf.append(block)
                silent_blocks += 1
                if silent_blocks * BLOCK_DURATION >= PAUSE_SECONDS:
                    text = self._transcribe(list(speech_buf), prev_text)
                    if text:
                        prev_text = text
                    if self.polish_mode and text:
                        polish_parts.append(text)
                    speech_buf.clear()
                    silent_blocks = 0
                    has_speech = False

        # Drain any audio blocks that arrived while _transcribe was blocking
        while not self.audio_queue.empty():
            try:
                block = self.audio_queue.get_nowait()
                rms = float(np.sqrt(np.mean(block ** 2)))
                if rms > self.silence_threshold:
                    speech_buf.append(block)
                    has_speech = True
                elif has_speech:
                    speech_buf.append(block)
            except queue.Empty:
                break

        if speech_buf and has_speech:
            text = self._transcribe(speech_buf, prev_text)
            if self.polish_mode and text:
                polish_parts.append(text)

        if self.polish_mode and polish_parts:
            raw = " ".join(polish_parts)
            log.info("Polish mode: sending %d chars to Groq", len(raw))
            self._show_icon(self._icon_busy)
            polished = polish_text(raw, self.language)
            if polished:
                try:
                    self._paste(polished + " ")
                    log.info("Polished and pasted %d chars (was %d)", len(polished), len(raw))
                except Exception:
                    log.exception("Paste failed after polish")
            else:
                log.warning("Polish failed, pasting raw text")
                try:
                    self._paste(raw + " ")
                except Exception:
                    log.exception("Paste failed")

        self.polish_mode = False
        self._show_icon(self._icon_idle)

    # ---- Transcription ---------------------------------------------------

    def _transcribe(self, blocks, prev_text=""):
        audio = np.concatenate(blocks).flatten().astype(np.float32)
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_SPEECH_SECONDS:
            return None

        self._show_icon(self._icon_busy)
        t0 = time.monotonic()

        kwargs = {"language": self.language}
        prompt_parts = []
        prompt = read_prompt()
        if prompt:
            prompt_parts.append(prompt)
        if prev_text:
            prompt_parts.append(prev_text[-200:])
        if prompt_parts:
            kwargs["initial_prompt"] = " ".join(prompt_parts)

        result = self._worker.transcribe(audio, self.model_repo, **kwargs)
        if result is None:
            if self.recording:
                self._show_icon(self._icon_rec[0])
            return None

        elapsed = time.monotonic() - t0
        text = result.get("text", "").strip()

        if text and not is_hallucination(text):
            speed = duration / elapsed if elapsed > 0 else 0
            if self.polish_mode:
                log.info("Buffered %d chars for polish  (%.1fs -> %.1fs, %.0fx)", len(text), duration, elapsed, speed)
                if self.recording:
                    self._show_icon(self._icon_rec[0])
                return text
            try:
                self._paste(text + " ")
                log.info("Pasted %d chars  (%.1fs -> %.1fs, %.0fx)", len(text), duration, elapsed, speed)
            except Exception:
                log.exception("Paste failed -- text in clipboard")
        else:
            log.info("Filtered hallucination")

        if self.recording:
            self._show_icon(self._icon_rec[0])
        return None

    # ---- Settings: internal setters --------------------------------------

    def _set_language(self, code: str):
        if code == self.language:
            return
        self.language = code
        self._save()
        log.info("Language -> %s", code)

    def _set_model(self, key: str):
        if key == self.model_key:
            return
        self.model_key = key
        self.model_repo = MODELS[key]["repo"]
        self._save()
        log.info("Model -> %s", key)
        self._notify(_t("Byter modell…", "Switching model…"), model_label(key))

        def preload():
            self._show_icon(self._icon_busy)

            try:
                self._worker.warm_up(self.model_repo, self.language)
                self._notify(_t("Modell laddad", "Model loaded"), model_label(key))
            except Exception as e:
                log.exception("Model switch failed")
                self._notify(_t("Fel vid modellbyte", "Model switch failed"), str(e))
            self._show_icon(self._icon_idle)

        threading.Thread(target=preload, daemon=True).start()

    def _set_hotkey(self, key: str):
        if key == self.hotkey:
            return
        self.hotkey = key
        hk = HOTKEYS[key]
        self.hotkey_code = hk["code"]
        self.hotkey_flag = hk["flag"]
        self._save()
        log.info("Hotkey -> %s", key)
        self._notify(_t("Tangent ändrad", "Hotkey changed"), hotkey_label(key))

    def _set_setting(self, key: str, value):
        self.settings[key] = value
        save_settings(self.settings)

    def _set_logging(self, enabled: bool):
        self._set_setting("logging", enabled)
        set_logging_enabled(enabled)
        log.info("Logging -> %s", "on" if enabled else "off")

    def _set_autostart(self, enabled: bool):
        self._set_setting("autostart", enabled)
        set_autostart(enabled)
        msg = _t(
            "Startar automatiskt vid inloggning" if enabled else "Autostart avstängd",
            "Starts automatically at login" if enabled else "Autostart disabled",
        )
        self._notify("Autostart", msg)

    # ---- Settings window -------------------------------------------------

    def _on_open_settings(self, _sender=None):
        from AppKit import (
            NSWindow, NSTextField, NSSecureTextField, NSPopUpButton, NSButton,
            NSFont, NSApp, NSBackingStoreBuffered, NSColor, NSBox,
        )
        from Foundation import NSMakeRect

        # Reuse existing window if visible
        if self._settings_win and self._settings_win.isVisible():
            self._settings_win.makeKeyAndOrderFront_(None)
            NSApp.activateIgnoringOtherApps_(True)
            return

        DelegateCls = _get_delegate_class()
        delegate = DelegateCls.alloc().init()
        self._settings_delegate = delegate  # prevent GC

        W, H = 420, 840
        style = 1 | 2  # NSWindowStyleMaskTitled | NSWindowStyleMaskClosable

        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, H), style, NSBackingStoreBuffered, False
        )
        win.setTitle_(_t("PTT — Inställningar", "PTT — Settings"))
        win.center()

        content = win.contentView()
        y = H - 40
        LX = 20       # label x
        PX = 170      # popup x
        PW = 220      # popup width
        BX = 295      # button x
        BW = 100      # button width
        sec = NSColor.secondaryLabelColor()
        ter = NSColor.tertiaryLabelColor()

        def add_label(text, x, yy, w=370, bold=False, size=12, color=None):
            tf = NSTextField.alloc().initWithFrame_(NSMakeRect(x, yy, w, 18))
            tf.setStringValue_(text)
            tf.setBezeled_(False)
            tf.setDrawsBackground_(False)
            tf.setEditable_(False)
            tf.setSelectable_(False)
            if bold:
                tf.setFont_(NSFont.boldSystemFontOfSize_(size))
            else:
                tf.setFont_(NSFont.systemFontOfSize_(size))
            if color:
                tf.setTextColor_(color)
            content.addSubview_(tf)
            return tf

        def add_popup(items, selected_idx, action_sel, yy):
            popup = NSPopUpButton.alloc().initWithFrame_pullsDown_(
                NSMakeRect(PX, yy - 3, PW, 24), False
            )
            for title in items:
                popup.addItemWithTitle_(title)
            if 0 <= selected_idx < len(items):
                popup.selectItemAtIndex_(selected_idx)
            popup.setTarget_(delegate)
            popup.setAction_(action_sel)
            content.addSubview_(popup)
            return popup

        def add_btn(title, yy, action_sel, x=BX, w=BW):
            btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, yy - 4, w, 28))
            btn.setTitle_(title)
            btn.setBezelStyle_(1)
            btn.setTarget_(delegate)
            btn.setAction_(action_sel)
            content.addSubview_(btn)
            return btn

        def add_checkbox(title, yy, checked, action_sel):
            chk = NSButton.alloc().initWithFrame_(NSMakeRect(LX, yy, 350, 22))
            chk.setButtonType_(3)
            chk.setTitle_(title)
            chk.setState_(1 if checked else 0)
            chk.setTarget_(delegate)
            chk.setAction_(action_sel)
            content.addSubview_(chk)
            return chk

        def add_separator(yy):
            sep = NSBox.alloc().initWithFrame_(NSMakeRect(LX, yy, W - 40, 1))
            sep.setBoxType_(2)  # NSBoxSeparator
            content.addSubview_(sep)
            return sep

        # ── TRANSCRIPTION ──────────────────────────────────────
        add_label(_t("Transkribering", "Transcription"), LX, y, bold=True, size=13)
        y -= 20
        add_label(
            _t("Håll tangenten, prata, släpp — texten klistras in",
               "Hold hotkey, speak, release — text is pasted"),
            LX, y, size=11, color=sec,
        )
        y -= 28

        # Language
        add_label(_t("Språk", "Language"), LX, y)
        self._win_lang_codes = []
        lang_labels = []
        available_langs = [("en", "English")]
        if system_has_swedish():
            available_langs.append(("sv", "Svenska"))
        for code, lbl in available_langs:
            self._win_lang_codes.append(code)
            lang_labels.append(lbl)
        try:
            lang_idx = self._win_lang_codes.index(self.language)
        except ValueError:
            lang_idx = 0
        add_popup(lang_labels, lang_idx, "langChanged:", y)
        y -= 32

        # Model
        add_label(_t("Modell", "Model"), LX, y)
        self._win_model_keys = []
        model_labels_list = []
        has_sv = system_has_swedish()
        for key, info in MODELS.items():
            if info["swedish_only"] and not has_sv:
                continue
            self._win_model_keys.append(key)
            model_labels_list.append(model_label(key))
        try:
            model_idx = self._win_model_keys.index(self.model_key)
        except ValueError:
            model_idx = 0
        add_popup(model_labels_list, model_idx, "modelChanged:", y)
        y -= 32

        # Hotkey
        add_label(_t("Tangent", "Hotkey"), LX, y)
        self._win_hotkey_keys = []
        hk_labels = []
        for key, info in HOTKEYS.items():
            self._win_hotkey_keys.append(key)
            hk_labels.append(hotkey_label(key))
        try:
            hotkey_idx = self._win_hotkey_keys.index(self.hotkey)
        except ValueError:
            hotkey_idx = 0
        add_popup(hk_labels, hotkey_idx, "hotkeyChanged:", y)
        y -= 32

        # Microphone
        add_label(_t("Mikrofon", "Microphone"), LX, y)
        if self.device_name:
            mic_text = self.device_name
        elif not self.ready:
            mic_text = _t("Laddar…", "Loading…")
        else:
            mic_text = _t("Ej identifierad", "Not detected")
        add_label(mic_text, PX, y, w=120, size=11, color=sec)
        add_btn(_t("Kalibrera", "Calibrate"), y, "calibrateClicked:")
        y -= 32

        # Word hints
        add_label(
            _t("Ordlista", "Vocabulary"),
            LX, y,
        )
        add_label(
            _t("Förbättra igenkänning av namn och termer",
               "Improve recognition of names and terms"),
            PX, y, w=140, size=11, color=sec,
        )
        add_btn(_t("Redigera…", "Edit…"), y, "editPromptClicked:")

        y -= 20
        add_separator(y)
        y -= 16

        # ── TEXT POLISH ────────────────────────────────────────
        add_label(_t("Textpolering", "Text polish"), LX, y, bold=True, size=13)
        y -= 20
        add_label(
            _t("Håll ⌘ + tangent för att rensa talspråk",
               "Hold ⌘ + hotkey to clean up spoken text"),
            LX, y, size=11, color=sec,
        )
        y -= 16
        add_label(
            _t("Texten skickas till Groq (groq.com) för bearbetning",
               "Text is sent to Groq (groq.com) for processing"),
            LX, y, size=10, color=ter,
        )
        y -= 26

        has_key = _get_groq_key() is not None
        add_label("Groq API", LX, y)
        if has_key:
            add_label(
                _t("Nyckel sparad ✓", "Key saved ✓"),
                PX, y, w=140, size=11,
                color=NSColor.systemGreenColor(),
            )
            add_btn(_t("Ta bort", "Remove"), y, "removeGroqKey:")
        else:
            key_field = NSSecureTextField.alloc().initWithFrame_(NSMakeRect(PX, y - 2, 110, 24))
            key_field.setPlaceholderString_("gsk_...")
            key_field.setFont_(NSFont.systemFontOfSize_(11))
            content.addSubview_(key_field)
            self._groq_key_field = key_field
            add_btn(_t("Spara", "Save"), y, "saveGroqKey:")
        y -= 30

        # Polish prompt — collapsible inline editor
        add_label(_t("Prompt", "Prompt"), LX, y)
        add_label(
            _t("Instruktioner till AI:n", "Instructions for the AI"),
            PX, y, w=140, size=11, color=sec,
        )
        y -= 24

        from AppKit import NSTextView, NSScrollView
        prompt_scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(LX, y - 70, W - 40, 74))
        prompt_scroll.setHasVerticalScroller_(True)
        prompt_scroll.setBorderType_(3)  # NSBezelBorder
        prompt_tv = NSTextView.alloc().initWithFrame_(NSMakeRect(0, 0, W - 58, 70))
        prompt_tv.setFont_(NSFont.systemFontOfSize_(11))
        prompt_tv.setEditable_(True)
        prompt_tv.setRichText_(False)
        # Load current prompt
        current_prompt = _read_commented_file(POLISH_PROMPT_PATH, sep=" ") or ""
        prompt_tv.setString_(current_prompt)
        prompt_scroll.setDocumentView_(prompt_tv)
        content.addSubview_(prompt_scroll)
        self._polish_prompt_tv = prompt_tv
        y -= 80

        add_btn(_t("Spara prompt", "Save prompt"), y, "savePolishPrompt:", x=LX, w=120)

        y -= 20
        add_separator(y)
        y -= 16

        # ── GENERAL ───────────────────────────────────────────
        add_label(_t("Övrigt", "General"), LX, y, bold=True, size=13)
        y -= 28

        add_checkbox(_t("Starta PTT vid inloggning", "Start PTT at login"),
                    y, self.settings.get("autostart"), "autostartChanged:")
        y -= 26

        add_checkbox(_t("Logga till fil", "Log to file"),
                    y, self.settings.get("logging", True), "loggingChanged:")
        add_btn(_t("Rensa", "Clear"), y, "clearLog:", x=BX - 60, w=70)
        add_btn(_t("Visa", "View"), y, "openLog:", x=BX + 20, w=70)
        y -= 26

        add_checkbox(_t("Återställ urklipp efter inklistring", "Restore clipboard after paste"),
                    y, self.settings.get("restore_clipboard", True), "clipboardChanged:")
        y -= 26

        add_checkbox(_t("Ljudfeedback vid inspelning", "Sound feedback when recording"),
                    y, self.settings.get("sounds", False), "soundsChanged:")
        y -= 26

        add_checkbox(_t("Visa senaste transkriptioner i menyn", "Show recent transcriptions in menu"),
                    y, self.settings.get("history", True), "historyChanged:")

        y -= 30

        # --- Footer ---
        add_label(
            _t("PTT — lokal transkribering med MLX Whisper",
               "PTT — local transcription with MLX Whisper"),
            LX, y, w=360, size=10, color=ter,
        )

        self._settings_win = win
        win.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    # ---- Action callbacks ------------------------------------------------

    def _on_edit_prompt(self, _sender=None):
        ensure_prompt_file()
        subprocess.Popen(["open", "-t", PROMPT_PATH])

    def _on_edit_polish_prompt(self, _sender=None):
        _ensure_polish_prompt_file()
        subprocess.Popen(["open", "-t", POLISH_PROMPT_PATH])

    def _on_save_polish_prompt(self, _sender=None):
        if not self._polish_prompt_tv:
            return
        text = str(self._polish_prompt_tv.string()).strip()
        with open(POLISH_PROMPT_PATH, "w") as f:
            f.write(text + "\n")
        self._notify(
            _t("Prompt sparad", "Prompt saved"),
            _t("Poleringsprompt uppdaterad", "Polish prompt updated"),
        )

    def _on_clear_log(self, _sender=None):
        try:
            open(LOG_PATH, "w").close()
            log.info("Log cleared")
            self._notify(
                _t("Logg rensad", "Log cleared"),
                "ptt.log",
            )
        except Exception:
            pass

    def _on_save_groq_key(self, _sender=None):
        if not self._groq_key_field:
            return
        key = str(self._groq_key_field.stringValue()).strip()
        if not key:
            return
        if _set_groq_key(key):
            log.info("Groq API key saved")
            self._notify(
                _t("Nyckel sparad", "Key saved"),
                _t("Groq API-nyckel sparad i Keychain", "Groq API key saved to Keychain"),
            )
            # Refresh settings window
            if self._settings_win:
                self._settings_win.close()
                self._settings_win = None
                self._on_open_settings()
        else:
            self._notify(
                _t("Fel", "Error"),
                _t("Kunde inte spara nyckeln", "Could not save key"),
            )

    def _on_remove_groq_key(self, _sender=None):
        _delete_groq_key()
        log.info("Groq API key removed")
        self._notify(
            _t("Nyckel borttagen", "Key removed"),
            _t("Groq API-nyckel borttagen", "Groq API key removed"),
        )
        if self._settings_win:
            self._settings_win.close()
            self._settings_win = None
            self._on_open_settings()

    def _on_recalibrate(self, _sender=None):
        def do():
            self._notify(
                _t("Kalibrerar…", "Calibrating…"),
                _t("Var tyst i 2 sekunder", "Stay quiet for 2 seconds"),
            )
            self._calibrate()
            self._notify(
                _t("Klart!", "Done!"),
                f"{_t('Tröskel', 'Threshold')}: {self.silence_threshold:.4f}",
            )
        threading.Thread(target=do, daemon=True).start()

    def _on_open_log(self, _sender=None):
        subprocess.Popen(["open", "-a", "Console", LOG_PATH])

    def _on_quit(self, _sender=None):
        import rumps
        self.recording = False
        self._worker.stop()
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
        rumps.quit_application()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def list_devices():
    import sounddevice as sd
    devices = sd.query_devices()
    builtin_idx, _ = find_builtin_mic()
    print("\n  Audio input devices:\n")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            tags = []
            if i == sd.default.device[0]:
                tags.append("default")
            if i == builtin_idx:
                tags.append("built-in")
            suffix = f"  ({', '.join(tags)})" if tags else ""
            print(f"    {i}: {d['name']}{suffix}")
    print()


def main():
    parser = argparse.ArgumentParser(
        prog="ptt",
        description="PTT -- Push-to-Talk Transcription for macOS",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--install", action="store_true", help="Auto-start on login")
    parser.add_argument("--uninstall", action="store_true", help="Remove auto-start")
    parser.add_argument("--reset", action="store_true", help="Reset settings")
    parser.add_argument("--lang", default=None, metavar="CODE")
    parser.add_argument("--model", default=None, choices=list(MODELS))
    parser.add_argument("--key", default=None, choices=list(HOTKEYS))
    parser.add_argument("--device", type=int, default=None, metavar="N")
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return
    if args.install:
        settings = load_settings()
        settings["autostart"] = True
        save_settings(settings)
        set_autostart(True)
        print("  PTT starts automatically on login.")
        print("  Remove with: ptt --uninstall")
        return
    if args.uninstall:
        settings = load_settings()
        settings["autostart"] = False
        save_settings(settings)
        set_autostart(False)
        print("  LaunchAgent removed.")
        return
    if args.reset:
        if os.path.isfile(SETTINGS_PATH):
            os.remove(SETTINGS_PATH)
        print("  Settings reset to defaults.")
        return

    _kill_other_instances()

    settings = load_settings()
    if args.lang:
        settings["language"] = args.lang
    if args.model:
        settings["model"] = args.model
    if args.key:
        settings["hotkey"] = args.key
    if args.device is not None:
        settings["device"] = args.device
    save_settings(settings)

    # Apply logging setting
    set_logging_enabled(settings.get("logging", True))

    PTTApp(settings).run()


if __name__ == "__main__":
    main()
