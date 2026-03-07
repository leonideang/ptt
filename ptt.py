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
"""PTT — Push-to-Talk Transcription for macOS (Apple Silicon).

Hold a key, speak, release — text appears where your cursor is.
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
LAUNCHAGENT_LABEL = "com.ptt.transcription"
PLIST_PATH = os.path.expanduser(
    f"~/Library/LaunchAgents/{LAUNCHAGENT_LABEL}.plist"
)

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("ptt")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = {
    "turbo": {
        "repo": "mlx-community/whisper-large-v3-turbo",
        "label": "Turbo — snabb, bra på alla språk",
    },
    "kb-sv": {
        "repo": "bratland/kb-whisper-large-mlx",
        "label": "KB Swedish — bäst på svenska",
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
    "alt_r":  {"code": 61, "flag": 1 << 19, "label": "Höger Option (⌥)"},
    "alt":    {"code": 58, "flag": 1 << 19, "label": "Vänster Option (⌥)"},
    "ctrl_r": {"code": 62, "flag": 1 << 18, "label": "Höger Control (⌃)"},
    "ctrl":   {"code": 59, "flag": 1 << 18, "label": "Vänster Control (⌃)"},
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
                "# PTT — Ordlista\n"
                "#\n"
                "# Skriv ord och namn som ofta hörs fel.\n"
                "# En post per rad, eller kommaseparerat.\n"
                "# Rader som börjar med # ignoreras.\n"
                "#\n"
                "# Exempel:\n"
                "# Claude Code, Kajabi\n"
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

        # Menu item references
        self._status_item = None
        self._lang_items: dict = {}
        self._model_items: dict = {}
        self._hotkey_items: dict = {}
        self._autostart_item = None

        # Icons
        self._icon_idle = None
        self._icon_rec: list = []
        self._icon_busy = None

    # ---- Persistence -----------------------------------------------------

    def _save(self):
        self.settings.update({
            "language": self.language,
            "model": self.model_key,
            "hotkey": self.hotkey,
            "device": self.device_idx,
        })
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
        except Exception:
            pass

    # ---- Menu ------------------------------------------------------------

    def run(self):
        import rumps
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory
        )

        self._create_icons()
        ensure_prompt_file()

        # Use text title initially — icon set after app loop starts
        self._app = rumps.App("PTT", title="◉", quit_button=None)

        self._status_item = rumps.MenuItem("Startar…")

        # --- Language submenu ---
        lang_menu = rumps.MenuItem("Språk")
        available_langs = [("en", "English")]
        if system_has_swedish():
            available_langs.append(("sv", "Svenska"))
        for code, label in available_langs:
            item = rumps.MenuItem(label, callback=self._on_language)
            item._lang_code = code
            item.state = 1 if code == self.language else 0
            self._lang_items[code] = item
            lang_menu.add(item)

        # --- Model submenu ---
        model_menu = rumps.MenuItem("Modell")
        for key, info in MODELS.items():
            item = rumps.MenuItem(info["label"], callback=self._on_model)
            item._model_key = key
            item.state = 1 if key == self.model_key else 0
            self._model_items[key] = item
            model_menu.add(item)

        # --- Hotkey submenu ---
        hotkey_menu = rumps.MenuItem("Tangent")
        for key, info in HOTKEYS.items():
            item = rumps.MenuItem(info["label"], callback=self._on_hotkey)
            item._hotkey_key = key
            item.state = 1 if key == self.hotkey else 0
            self._hotkey_items[key] = item
            hotkey_menu.add(item)

        # --- Autostart ---
        self._autostart_item = rumps.MenuItem(
            "Starta vid inloggning",
            callback=self._on_autostart,
        )
        self._autostart_item.state = 1 if self.settings.get("autostart") else 0

        # --- Settings submenu ---
        settings_menu = rumps.MenuItem("Inställningar")
        settings_menu.add(lang_menu)
        settings_menu.add(model_menu)
        settings_menu.add(hotkey_menu)
        settings_menu.add(None)  # separator
        settings_menu.add(self._autostart_item)
        settings_menu.add(rumps.MenuItem("Redigera ordlista…", callback=self._on_edit_prompt))
        settings_menu.add(rumps.MenuItem("Kalibrera mikrofon", callback=self._on_recalibrate))

        # --- Top-level menu ---
        self._app.menu = [
            self._status_item,
            None,
            settings_menu,
            None,
            rumps.MenuItem("Visa logg…", callback=self._on_open_log),
            rumps.MenuItem("Avsluta", callback=self._on_quit),
        ]

        # Accessibility check
        if not check_accessibility():
            log.warning("Accessibility permission missing")
            rumps.alert(
                title="PTT behöver Accessibility",
                message=(
                    "PTT behöver behörighet för att läsa tangenter "
                    "och klistra in text.\n\n"
                    "Lägg till appen i:\n"
                    "Systeminställningar → Integritet och säkerhet → Hjälpmedel"
                ),
                ok="Öppna inställningar",
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

    def _init(self):
        import sounddevice as sd
        from AppKit import NSEvent

        # Wait for the app loop to create the NSStatusItem
        time.sleep(0.5)

        try:
            self._show_icon(self._icon_busy)
            self._set_status("Laddar modell…")
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
            self._set_status(f"Mikrofon: {self.device_name}")
            label = HOTKEYS[self.hotkey]["label"]
            self._notify("Redo!", f"Håll {label} och prata")
            log.info("PTT ready")

        except Exception as e:
            log.exception("Init failed")
            self._set_status(f"Fel: {e}")
            self._notify("Kunde inte starta", str(e))

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
        log.info("Ambient=%.5f → threshold=%.5f", ambient, self.silence_threshold)

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
                log.info("Pasted %d chars  (%.1fs → %.1fs, %.0f×)", len(text), duration, elapsed, speed)
            except Exception:
                log.exception("Paste failed — text in clipboard")
        else:
            log.info("Filtered hallucination")

        if self.recording:
            self._show_icon(self._icon_rec[0])

    # ---- Settings callbacks ----------------------------------------------

    def _on_language(self, sender):
        code = sender._lang_code
        if code == self.language:
            return
        self.language = code
        for c, item in self._lang_items.items():
            item.state = 1 if c == code else 0
        self._save()
        label = "Svenska" if code == "sv" else "English"
        log.info("Language → %s", code)
        self._notify("Språk", label)

    def _on_model(self, sender):
        key = sender._model_key
        if key == self.model_key:
            return
        self.model_key = key
        self.model_repo = MODELS[key]["repo"]
        for k, item in self._model_items.items():
            item.state = 1 if k == key else 0
        self._save()
        log.info("Model → %s", key)
        self._notify("Byter modell…", MODELS[key]["label"])

        def preload():
            self._show_icon(self._icon_busy)
            self._set_status("Laddar modell…")
            try:
                import mlx_whisper
                mlx_whisper.transcribe(
                    np.zeros(SAMPLE_RATE, dtype=np.float32),
                    path_or_hf_repo=self.model_repo,
                    language=self.language,
                )
                self._set_status(f"Mikrofon: {self.device_name}")
                self._notify("Modell laddad", MODELS[key]["label"])
            except Exception as e:
                log.exception("Model switch failed")
                self._notify("Fel vid modellbyte", str(e))
            self._show_icon(self._icon_idle)

        threading.Thread(target=preload, daemon=True).start()

    def _on_hotkey(self, sender):
        key = sender._hotkey_key
        if key == self.hotkey:
            return
        self.hotkey = key
        hk = HOTKEYS[key]
        self.hotkey_code = hk["code"]
        self.hotkey_flag = hk["flag"]
        for k, item in self._hotkey_items.items():
            item.state = 1 if k == key else 0
        self._save()
        log.info("Hotkey → %s", key)
        self._notify("Tangent ändrad", hk["label"])

    def _on_autostart(self, sender):
        enabled = not bool(sender.state)
        sender.state = 1 if enabled else 0
        self.settings["autostart"] = enabled
        save_settings(self.settings)
        set_autostart(enabled)
        msg = "PTT startar automatiskt vid inloggning" if enabled else "Autostart avstängd"
        self._notify("Autostart", msg)

    def _on_edit_prompt(self, _sender=None):
        subprocess.Popen(["open", "-t", PROMPT_PATH])

    def _on_recalibrate(self, _sender=None):
        def do():
            self._notify("Kalibrerar…", "Var tyst i 2 sekunder")
            self._calibrate()
            self._notify("Klart!", f"Tröskel: {self.silence_threshold:.4f}")
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
        description="PTT — Push-to-Talk Transcription for macOS",
    )
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--install", action="store_true", help="Auto-start on login")
    parser.add_argument("--uninstall", action="store_true", help="Remove auto-start")
    parser.add_argument("--reset", action="store_true", help="Reset settings")
    # Override flags (optional, settings file is primary)
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

    settings = load_settings()
    # CLI overrides
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
