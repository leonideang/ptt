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

_log_handlers = [logging.FileHandler(LOG_PATH)]
if os.isatty(2):  # Only add stderr handler when running in a terminal
    _log_handlers.append(logging.StreamHandler())
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=_log_handlers,
)
log = logging.getLogger("ptt")

# Suppress noisy third-party loggers
for _name in ("httpx", "huggingface_hub", "tqdm"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = {
    "turbo": {
        "repo": "mlx-community/whisper-large-v3-turbo",
        "label": "Turbo -- snabb, alla sprak",
    },
    "kb-sv": {
        "repo": "bratland/kb-whisper-large-mlx",
        "label": "KB Swedish -- bast pa svenska",
    },
}

# ---------------------------------------------------------------------------
# Audio constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.1
MIN_SPEECH_SECONDS = 0.3
PAUSE_SECONDS = 0.8
PRE_ROLL_SECONDS = 0.4
CALIBRATION_SECONDS = 2.0
THRESHOLD_MULTIPLIER = 3.0
MIN_THRESHOLD = 0.002
ANIM_INTERVAL = 3

# ---------------------------------------------------------------------------
# Hotkeys
# ---------------------------------------------------------------------------

HOTKEYS = {
    "alt_r":  {"code": 61, "flag": 1 << 19, "label": "Hoger Option"},
    "alt":    {"code": 58, "flag": 1 << 19, "label": "Vanster Option"},
    "ctrl_r": {"code": 62, "flag": 1 << 18, "label": "Hoger Control"},
    "ctrl":   {"code": 59, "flag": 1 << 18, "label": "Vanster Control"},
}

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
    "model": "turbo",
    "hotkey": "alt_r",
    "device": None,
    "autostart": False,
    "intro_shown": False,
}


def detect_system_language() -> str:
    try:
        out = subprocess.run(
            ["defaults", "read", "-g", "AppleLanguages"],
            capture_output=True, text=True,
        ).stdout.lower()
        if "sv" in out:
            return "sv"
    except Exception:
        pass
    return "en"


def system_has_swedish() -> bool:
    try:
        out = subprocess.run(
            ["defaults", "read", "-g", "AppleLanguages"],
            capture_output=True, text=True,
        ).stdout.lower()
        return "sv" in out
    except Exception:
        return False


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
    return settings


def save_settings(settings: dict):
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


# ---------------------------------------------------------------------------
# Prompt / word hints
# ---------------------------------------------------------------------------


def read_prompt() -> str | None:
    if not os.path.isfile(PROMPT_PATH):
        return None
    lines = []
    for line in open(PROMPT_PATH):
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    text = ", ".join(lines).strip()
    return text if text else None


def ensure_prompt_file():
    if not os.path.isfile(PROMPT_PATH):
        with open(PROMPT_PATH, "w") as f:
            f.write(
                "# PTT -- Ordlista\n"
                "#\n"
                "# Skriv ord och namn som ofta hors fel.\n"
                "# En post per rad, eller kommaseparerat.\n"
                "# Rader som borjar med # ignoreras.\n"
                "#\n"
                "# Exempel:\n"
                "# Claude Code, Kajabi\n"
            )


# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------

_ARTIFACTS = frozenset({
    "thank you", "thanks for watching", "subscribe", "you",
    "tack for att ni tittade", "tack for att ni tittar",
    "undertextning", "musik", "textning", "textning.nu",
    "untertitelung", "sous-titrage", "amara.org",
    "text", ".", "..", "...",
})


def is_hallucination(text: str) -> bool:
    t = text.strip().lower()
    if not t or len(t) <= 1:
        return True
    if t[0] in "([":
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


def paste_text(text: str):
    from Quartz import (
        CGEventCreateKeyboardEvent, CGEventPost,
        CGEventSetFlags, kCGEventFlagMaskCommand, kCGHIDEventTap,
    )

    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    time.sleep(0.03)
    for pressed in (True, False):
        ev = CGEventCreateKeyboardEvent(None, 9, pressed)
        CGEventSetFlags(ev, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, ev)
        if pressed:
            time.sleep(0.03)


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

    _SettingsDelegate = _SD
    return _SettingsDelegate


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
        self.silence_threshold = MIN_THRESHOLD

        self.audio_queue: queue.Queue = queue.Queue()
        self.block_size = int(SAMPLE_RATE * BLOCK_DURATION)
        self.pre_roll: collections.deque = collections.deque(
            maxlen=int(PRE_ROLL_SECONDS / BLOCK_DURATION)
        )

        self._rec_thread: threading.Thread | None = None
        self._stream = None
        self._app = None

        # Menu item reference
        self._status_item = None

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

    def _show_icon(self, nsimage):
        try:
            b = self._app._nsapp.nsstatusitem.button()
            b.setImage_(nsimage)
            b.setTitle_("")
        except Exception as e:
            log.debug("_show_icon failed: %s", e)

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

        self._status_item = rumps.MenuItem("Startar...")

        self._app.menu = [
            self._status_item,
            None,
            rumps.MenuItem("Installningar...", callback=self._on_open_settings),
            rumps.MenuItem("Visa logg...", callback=self._on_open_log),
            None,
            rumps.MenuItem("Avsluta", callback=self._on_quit),
        ]

        # Accessibility check
        if not check_accessibility():
            log.warning("Accessibility permission missing")
            rumps.alert(
                title="PTT behover Accessibility",
                message=(
                    "PTT behover behorighet for att lasa tangenter "
                    "och klistra in text.\n\n"
                    "Lagg till appen i:\n"
                    "Installningar > Integritet och sakerhet > Hjalpmedel"
                ),
                ok="Oppna installningar",
            )
            subprocess.Popen([
                "open",
                "x-apple.systempreferences:"
                "com.apple.preference.security?Privacy_Accessibility",
            ])

        threading.Thread(target=self._init, daemon=True).start()
        self._app.run()

    def _set_status(self, text: str):
        try:
            if self._status_item:
                self._status_item.title = text
        except Exception:
            pass

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
                if si and si.button():
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
            self._set_status("Laddar modell...")
            log.info("Loading model: %s", self.model_repo)

            import mlx_whisper
            mlx_whisper.transcribe(
                np.zeros(SAMPLE_RATE, dtype=np.float32),
                path_or_hf_repo=self.model_repo,
                language=self.language,
            )
            log.info("Model loaded")

            if self.device_idx is not None:
                self.device_name = sd.query_devices(self.device_idx)["name"]
            else:
                self.device_idx, self.device_name = find_builtin_mic()
            log.info("Mic: %s (device %s)", self.device_name, self.device_idx)

            self._calibrate()

            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=self.block_size,
                device=self.device_idx,
                callback=self._audio_cb,
            )
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

            self.ready = True
            self._show_icon(self._icon_idle)
            self._set_status(f"Redo | {self.device_name}")
            label = HOTKEYS[self.hotkey]["label"]
            log.info("PTT ready")

            # First-run intro
            if not self.settings.get("intro_shown"):
                self._notify(
                    "Redo!",
                    f"Hall {label} och prata.\n"
                    "Slapp for att transkribera och klistra in.\n"
                    "Klicka ikonen for installningar.",
                )
                self.settings["intro_shown"] = True
                save_settings(self.settings)
            else:
                self._notify("Redo!", f"Hall {label} och prata")

        except Exception as e:
            log.exception("Init failed")
            self._set_status(f"Fel: {e}")
            self._notify("Kunde inte starta", str(e))

    def _calibrate(self):
        import sounddevice as sd

        log.info("Calibrating...")
        rec = sd.rec(
            int(SAMPLE_RATE * CALIBRATION_SECONDS),
            samplerate=SAMPLE_RATE, channels=1, device=self.device_idx,
        )
        sd.wait()
        ambient = float(np.sqrt(np.mean(rec ** 2)))
        self.silence_threshold = max(ambient * THRESHOLD_MULTIPLIER, MIN_THRESHOLD)
        log.info("Ambient=%.5f -> threshold=%.5f", ambient, self.silence_threshold)

    # ---- Hotkey ----------------------------------------------------------

    def _on_modifier(self, event):
        if not self.ready or event.keyCode() != self.hotkey_code:
            return
        pressed = bool(event.modifierFlags() & self.hotkey_flag)
        if pressed and not self.recording:
            self._start_rec()
        elif not pressed and self.recording:
            self._stop_rec()

    # ---- Recording -------------------------------------------------------

    def _audio_cb(self, indata, frames, time_info, status):
        block = indata.copy()
        if self.recording:
            self.audio_queue.put(block)
        else:
            self.pre_roll.append(block)

    def _start_rec(self):
        if self._rec_thread and self._rec_thread.is_alive():
            return
        self.recording = True
        self._show_icon(self._icon_rec[0])
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
        log.info("REC stop")

    def _rec_loop(self):
        speech_buf: list = []
        silent_blocks = 0
        has_speech = False
        block_n = 0
        frame = 0

        while self.recording:
            try:
                block = self.audio_queue.get(timeout=0.05)
            except queue.Empty:
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
                    self._transcribe(list(speech_buf))
                    speech_buf.clear()
                    silent_blocks = 0
                    has_speech = False

        if speech_buf and has_speech:
            self._transcribe(speech_buf)

        self._show_icon(self._icon_idle)

    # ---- Transcription ---------------------------------------------------

    def _transcribe(self, blocks):
        import mlx_whisper

        audio = np.concatenate(blocks).flatten().astype(np.float32)
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_SPEECH_SECONDS:
            return

        self._show_icon(self._icon_busy)
        t0 = time.monotonic()

        kwargs = {
            "path_or_hf_repo": self.model_repo,
            "language": self.language,
        }
        prompt = read_prompt()
        if prompt:
            kwargs["initial_prompt"] = prompt

        try:
            result = mlx_whisper.transcribe(audio, **kwargs)
        except Exception:
            log.exception("Transcription error")
            if self.recording:
                self._show_icon(self._icon_rec[0])
            return

        elapsed = time.monotonic() - t0
        text = result.get("text", "").strip()

        if text and not is_hallucination(text):
            try:
                paste_text(text + " ")
                speed = duration / elapsed if elapsed > 0 else 0
                log.info("Pasted %d chars  (%.1fs -> %.1fs, %.0fx)", len(text), duration, elapsed, speed)
            except Exception:
                log.exception("Paste failed -- text in clipboard")
        else:
            log.info("Filtered hallucination")

        if self.recording:
            self._show_icon(self._icon_rec[0])

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
        self._notify("Byter modell...", MODELS[key]["label"])

        def preload():
            self._show_icon(self._icon_busy)
            self._set_status("Laddar modell...")
            try:
                import mlx_whisper
                mlx_whisper.transcribe(
                    np.zeros(SAMPLE_RATE, dtype=np.float32),
                    path_or_hf_repo=self.model_repo,
                    language=self.language,
                )
                self._set_status(f"Redo | {self.device_name}")
                self._notify("Modell laddad", MODELS[key]["label"])
            except Exception as e:
                log.exception("Model switch failed")
                self._notify("Fel vid modellbyte", str(e))
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
        self._notify("Tangent andrad", hk["label"])

    def _set_autostart(self, enabled: bool):
        self.settings["autostart"] = enabled
        save_settings(self.settings)
        set_autostart(enabled)
        msg = "Startar automatiskt vid inloggning" if enabled else "Autostart avstangd"
        self._notify("Autostart", msg)

    # ---- Settings window -------------------------------------------------

    def _on_open_settings(self, _sender=None):
        from AppKit import (
            NSWindow, NSTextField, NSPopUpButton, NSButton,
            NSFont, NSApp, NSBackingStoreBuffered, NSTextView,
            NSScrollView, NSBorderlessWindowMask,
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

        W, H = 400, 440
        style = 1 | 2  # NSWindowStyleMaskTitled | NSWindowStyleMaskClosable

        win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, W, H), style, NSBackingStoreBuffered, False
        )
        win.setTitle_("PTT -- Installningar")
        win.center()

        content = win.contentView()
        y = H - 50
        LX = 20       # label x
        PX = 160      # popup x
        PW = 210      # popup width

        def add_label(text, x, yy, w=340, bold=False, size=12):
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

        # --- Language ---
        add_label("Sprak", LX, y)
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

        y -= 38

        # --- Model ---
        add_label("Modell", LX, y)
        self._win_model_keys = []
        model_labels = []
        for key, info in MODELS.items():
            self._win_model_keys.append(key)
            model_labels.append(info["label"])
        try:
            model_idx = self._win_model_keys.index(self.model_key)
        except ValueError:
            model_idx = 0
        add_popup(model_labels, model_idx, "modelChanged:", y)

        y -= 38

        # --- Hotkey ---
        add_label("Tangent", LX, y)
        self._win_hotkey_keys = []
        hotkey_labels = []
        for key, info in HOTKEYS.items():
            self._win_hotkey_keys.append(key)
            hotkey_labels.append(info["label"])
        try:
            hotkey_idx = self._win_hotkey_keys.index(self.hotkey)
        except ValueError:
            hotkey_idx = 0
        add_popup(hotkey_labels, hotkey_idx, "hotkeyChanged:", y)

        y -= 50

        # --- Microphone ---
        add_label("Mikrofon", LX, y, bold=True)
        y -= 24
        mic_text = self.device_name if self.device_name else "Inte initierad"
        add_label(mic_text, LX, y, w=220)

        cal_btn = NSButton.alloc().initWithFrame_(NSMakeRect(280, y - 4, 95, 28))
        cal_btn.setTitle_("Kalibrera")
        cal_btn.setBezelStyle_(1)  # rounded
        cal_btn.setTarget_(delegate)
        cal_btn.setAction_("calibrateClicked:")
        content.addSubview_(cal_btn)

        y -= 45

        # --- Word hints ---
        add_label("Ordlista", LX, y, bold=True)
        y -= 24
        add_label("Ord som Whisper ofta missar", LX, y, w=220)

        edit_btn = NSButton.alloc().initWithFrame_(NSMakeRect(280, y - 4, 95, 28))
        edit_btn.setTitle_("Redigera...")
        edit_btn.setBezelStyle_(1)
        edit_btn.setTarget_(delegate)
        edit_btn.setAction_("editPromptClicked:")
        content.addSubview_(edit_btn)

        y -= 50

        # --- Autostart ---
        autostart_btn = NSButton.alloc().initWithFrame_(NSMakeRect(LX, y, 300, 22))
        autostart_btn.setButtonType_(3)  # switch/checkbox
        autostart_btn.setTitle_("Starta PTT vid inloggning")
        autostart_btn.setState_(1 if self.settings.get("autostart") else 0)
        autostart_btn.setTarget_(delegate)
        autostart_btn.setAction_("autostartChanged:")
        content.addSubview_(autostart_btn)

        y -= 40

        # --- Footer ---
        add_label(
            "PTT -- lokal transkribering med MLX Whisper",
            LX, y, w=360, size=10,
        )

        self._settings_win = win
        win.makeKeyAndOrderFront_(None)
        NSApp.activateIgnoringOtherApps_(True)

    # ---- Action callbacks ------------------------------------------------

    def _on_edit_prompt(self, _sender=None):
        subprocess.Popen(["open", "-t", PROMPT_PATH])

    def _on_recalibrate(self, _sender=None):
        def do():
            self._notify("Kalibrerar...", "Var tyst i 2 sekunder")
            self._calibrate()
            self._notify("Klart!", f"Troskel: {self.silence_threshold:.4f}")
        threading.Thread(target=do, daemon=True).start()

    def _on_open_log(self, _sender=None):
        subprocess.Popen(["open", "-a", "Console", LOG_PATH])

    def _on_quit(self, _sender=None):
        import rumps
        self.recording = False
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
        # Clean up PID file
        try:
            if os.path.isfile(PID_PATH):
                os.remove(PID_PATH)
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

    PTTApp(settings).run()


if __name__ == "__main__":
    main()
