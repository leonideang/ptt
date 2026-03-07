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

Menu bar app. Hold a modifier key to record speech, release to transcribe
and paste the text where your cursor is. Runs entirely on-device using
MLX Whisper — no API keys, no cloud, no data leaves your Mac.

Usage:
  uv run ptt.py                     Start (Swedish, Right Option)
  uv run ptt.py --lang en           English mode
  uv run ptt.py --model kb-sv       Swedish-optimized model
  uv run ptt.py --list-devices      Show audio input devices
  uv run ptt.py --install           Auto-start on login
  uv run ptt.py --uninstall         Remove auto-start

Requires: macOS 13+, Apple Silicon, Accessibility permission.
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
import sys
import threading
import time

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = os.path.expanduser("~/Library/Logs")
LOG_PATH = os.path.join(LOG_DIR, "ptt.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("ptt")

# ---------------------------------------------------------------------------
# Config directory
# ---------------------------------------------------------------------------

CONFIG_DIR = os.path.expanduser("~/.config/ptt")
PROMPT_PATH = os.path.join(CONFIG_DIR, "prompt.txt")

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

MODELS = {
    "turbo": {
        "repo": "mlx-community/whisper-large-v3-turbo",
        "label": "Turbo — snabb, bra på de flesta språk",
        "size": "~1.5 GB",
    },
    "kb-sv": {
        "repo": "bratland/kb-whisper-large-mlx",
        "label": "KB Swedish — bäst på svenska, långsammare",
        "size": "~3 GB",
    },
}

DEFAULT_MODEL = "turbo"
DEFAULT_LANG = "sv"

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
BLOCK_DURATION = 0.1
MIN_SPEECH_SECONDS = 0.3
PAUSE_SECONDS = 0.8
PRE_ROLL_SECONDS = 0.4
CALIBRATION_SECONDS = 2.0
THRESHOLD_MULTIPLIER = 3.0
MIN_THRESHOLD = 0.002

# ---------------------------------------------------------------------------
# Hotkeys (macOS virtual keycodes)
# ---------------------------------------------------------------------------

KEYCODES = {"alt_r": 61, "alt": 58, "ctrl_r": 62, "ctrl": 59}
MODIFIER_FLAGS = {61: 1 << 19, 58: 1 << 19, 62: 1 << 18, 59: 1 << 18}
HOTKEY_LABELS = {
    "alt_r": "Right Option (⌥)",
    "alt": "Left Option (⌥)",
    "ctrl_r": "Right Control (⌃)",
    "ctrl": "Left Control (⌃)",
}

LAUNCHAGENT_LABEL = "com.ptt.transcription"

# ---------------------------------------------------------------------------
# Waveform icons
# ---------------------------------------------------------------------------

# Bar heights (0–1) for each visual state
WAVE_IDLE = [0.15, 0.30, 0.50, 0.30, 0.15]
WAVE_REC_FRAMES = [
    [0.35, 0.75, 0.45, 0.90, 0.50],
    [0.50, 0.40, 0.85, 0.35, 0.70],
    [0.70, 0.90, 0.35, 0.65, 0.40],
    [0.45, 0.55, 0.70, 0.45, 0.85],
]
WAVE_PROCESSING = [0.25, 0.50, 0.25, 0.50, 0.25]
WAVE_ERROR = [0.10, 0.10, 0.80, 0.10, 0.10]

# How often to advance the animation frame (in audio blocks)
ANIM_INTERVAL = 3  # every 0.3s at BLOCK_DURATION=0.1

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


def open_accessibility_prefs():
    subprocess.Popen([
        "open",
        "x-apple.systempreferences:"
        "com.apple.preference.security?Privacy_Accessibility",
    ])


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
        CGEventCreateKeyboardEvent,
        CGEventPost,
        CGEventSetFlags,
        kCGEventFlagMaskCommand,
        kCGHIDEventTap,
    )

    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    time.sleep(0.03)
    for pressed in (True, False):
        ev = CGEventCreateKeyboardEvent(None, 9, pressed)  # 9 = v
        CGEventSetFlags(ev, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, ev)
        if pressed:
            time.sleep(0.03)


# ---------------------------------------------------------------------------
# Prompt / word list
# ---------------------------------------------------------------------------


def read_prompt() -> str | None:
    if os.path.isfile(PROMPT_PATH):
        text = open(PROMPT_PATH).read().strip()
        return text if text else None
    return None


def ensure_prompt_file():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    if not os.path.isfile(PROMPT_PATH):
        with open(PROMPT_PATH, "w") as f:
            f.write(
                "# PTT — Ordlista / Word hints\n"
                "#\n"
                "# Skriv ord och fraser som Whisper ofta missar.\n"
                "# Write words and phrases that Whisper often gets wrong.\n"
                "# Allt på en rad, kommaseparerat. Rader som börjar med # ignoreras.\n"
                "#\n"
                "# Exempel / Example:\n"
                "# Claude Code, Kajabi, Clawdbot, PTT\n"
            )


def open_prompt_file():
    ensure_prompt_file()
    subprocess.Popen(["open", "-t", PROMPT_PATH])


# ---------------------------------------------------------------------------
# Icon generation
# ---------------------------------------------------------------------------


def make_waveform_icon(bars: list[float], size: int = 18):
    """Generate an NSImage waveform icon (template, adapts to dark/light)."""
    from AppKit import NSBezierPath, NSColor, NSImage
    from Foundation import NSMakeRect, NSMakeSize

    img = NSImage.alloc().initWithSize_(NSMakeSize(size, size))
    img.lockFocus()

    n = len(bars)
    bar_w = 2.0
    gap = 1.5
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
# App
# ---------------------------------------------------------------------------


class PTTApp:
    def __init__(
        self,
        model_key: str,
        language: str,
        hotkey: str,
        device: int | None,
    ):
        self.model_key = model_key
        self.model_repo = MODELS[model_key]["repo"]
        self.language = language
        self.hotkey = hotkey
        self.hotkey_code = KEYCODES.get(hotkey, 61)
        self.hotkey_flag = MODIFIER_FLAGS.get(self.hotkey_code, 1 << 19)
        self.device_idx = device
        self.device_name = ""

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
        self._status_item = None
        self._model_items: dict = {}

        # Icons (created after AppKit is available)
        self._icon_idle = None
        self._icon_rec_frames: list = []
        self._icon_processing = None
        self._icon_error = None

    # ---- Icons -----------------------------------------------------------

    def _create_icons(self):
        self._icon_idle = make_waveform_icon(WAVE_IDLE)
        self._icon_rec_frames = [make_waveform_icon(f) for f in WAVE_REC_FRAMES]
        self._icon_processing = make_waveform_icon(WAVE_PROCESSING)
        self._icon_error = make_waveform_icon(WAVE_ERROR)

    def _show_icon(self, nsimage):
        try:
            button = self._app._nsapp.nsstatusitem.button()
            button.setImage_(nsimage)
            button.setTitle_("")
        except Exception:
            pass

    def _show_text(self, text: str):
        try:
            button = self._app._nsapp.nsstatusitem.button()
            button.setImage_(None)
            button.setTitle_(text)
        except Exception:
            pass

    # ---- Menu bar --------------------------------------------------------

    def run(self):
        import rumps
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory
        )

        self._create_icons()
        ensure_prompt_file()

        self._app = rumps.App("PTT", quit_button=None)
        # Set initial icon
        self._show_icon(self._icon_idle)

        self._status_item = rumps.MenuItem("Startar…")
        lang_label = "Svenska" if self.language == "sv" else "English"
        lang_item = rumps.MenuItem(
            f"Språk: {lang_label}",
            callback=self._toggle_language,
        )

        model_menu = rumps.MenuItem("Modell")
        for key, info in MODELS.items():
            item = rumps.MenuItem(info["label"], callback=self._switch_model)
            item._model_key = key
            item.state = 1 if key == self.model_key else 0
            self._model_items[key] = item
            model_menu.add(item)

        hotkey_label = HOTKEY_LABELS.get(self.hotkey, self.hotkey)

        self._app.menu = [
            self._status_item,
            None,
            lang_item,
            model_menu,
            None,
            rumps.MenuItem("Redigera ordlista…", callback=self._edit_prompt),
            rumps.MenuItem("Kalibrera mikrofon", callback=self._recalibrate_cb),
            None,
            rumps.MenuItem(f"Tangent: {hotkey_label}"),
            rumps.MenuItem("Visa logg…", callback=self._open_log),
            None,
            rumps.MenuItem("Avsluta PTT", callback=self._quit),
        ]

        if not check_accessibility():
            log.warning("Accessibility permission missing")
            rumps.alert(
                title="PTT behöver Accessibility",
                message=(
                    "PTT behöver Accessibility-behörighet för att läsa "
                    "tangenter och klistra in text.\n\n"
                    "Lägg till terminalen (eller PTT) i:\n"
                    "Systeminställningar → Integritet och säkerhet → Hjälpmedel"
                ),
                ok="Öppna inställningar",
            )
            open_accessibility_prefs()

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

    # ---- Initialization --------------------------------------------------

    def _init(self):
        import sounddevice as sd
        from AppKit import NSEvent

        try:
            self._show_icon(self._icon_processing)
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
            log.info("Audio stream started")

            mask = 1 << 12  # NSEventMaskFlagsChanged

            def on_global(event):
                self._on_modifier(event)

            def on_local(event):
                self._on_modifier(event)
                return event

            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, on_global)
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, on_local)
            log.info("Hotkey active (keycode %d)", self.hotkey_code)

            prompt = read_prompt()
            if prompt:
                log.info("Prompt loaded: %s", prompt[:80])

            self.ready = True
            self._show_icon(self._icon_idle)
            self._set_status(f"Mikrofon: {self.device_name}")
            hotkey_label = HOTKEY_LABELS.get(self.hotkey, self.hotkey)
            self._notify("Redo!", f"Håll {hotkey_label} och prata")
            log.info("PTT ready")

        except Exception as e:
            log.exception("Init failed")
            self._show_icon(self._icon_error)
            self._set_status(f"Fel: {e}")
            self._notify("Kunde inte starta", str(e))

    def _calibrate(self):
        import sounddevice as sd

        log.info("Calibrating…")
        rec = sd.rec(
            int(SAMPLE_RATE * CALIBRATION_SECONDS),
            samplerate=SAMPLE_RATE,
            channels=1,
            device=self.device_idx,
        )
        sd.wait()
        ambient_rms = float(np.sqrt(np.mean(rec ** 2)))
        self.silence_threshold = max(ambient_rms * THRESHOLD_MULTIPLIER, MIN_THRESHOLD)
        log.info("Ambient=%.5f → threshold=%.5f", ambient_rms, self.silence_threshold)

    # ---- Hotkey ----------------------------------------------------------

    def _on_modifier(self, event):
        if not self.ready:
            return
        if event.keyCode() != self.hotkey_code:
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
            return  # previous recording still finishing

        self.recording = True
        self._show_icon(self._icon_rec_frames[0])
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
        # Don't join — let the thread finish transcription in background.
        # The thread sets the icon back to idle when done.

    def _rec_loop(self):
        speech_buf: list = []
        silent_blocks = 0
        has_speech = False
        block_count = 0
        anim_frame = 0

        while self.recording:
            try:
                block = self.audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

            # Animate icon
            block_count += 1
            if block_count % ANIM_INTERVAL == 0:
                anim_frame = (anim_frame + 1) % len(self._icon_rec_frames)
                self._show_icon(self._icon_rec_frames[anim_frame])

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
            log.info("Skipped (%.2fs < min)", duration)
            return

        self._show_icon(self._icon_processing)
        t0 = time.monotonic()

        prompt = read_prompt()
        try:
            kwargs = {
                "path_or_hf_repo": self.model_repo,
                "language": self.language,
            }
            if prompt:
                kwargs["initial_prompt"] = prompt

            result = mlx_whisper.transcribe(audio, **kwargs)
        except Exception:
            log.exception("Transcription error")
            if self.recording:
                self._show_icon(self._icon_rec_frames[0])
            return

        elapsed = time.monotonic() - t0
        text = result.get("text", "").strip()

        if text and not is_hallucination(text):
            try:
                paste_text(text + " ")
                speed = duration / elapsed if elapsed > 0 else 0
                log.info("'%s'  (%.1fs → %.1fs, %.0f×)", text, duration, elapsed, speed)
            except Exception:
                log.exception("Paste failed — text copied to clipboard")
        else:
            log.info("Filtered: '%s'", text)

        if self.recording:
            self._show_icon(self._icon_rec_frames[0])

    # ---- Menu callbacks --------------------------------------------------

    def _switch_model(self, sender):
        key = sender._model_key
        if key == self.model_key:
            return

        self.model_key = key
        self.model_repo = MODELS[key]["repo"]
        for k, item in self._model_items.items():
            item.state = 1 if k == key else 0

        log.info("Model → %s (%s)", key, self.model_repo)
        self._notify("Byter modell…", MODELS[key]["label"])

        def preload():
            self._show_icon(self._icon_processing)
            self._set_status("Laddar modell…")
            try:
                import mlx_whisper

                mlx_whisper.transcribe(
                    np.zeros(SAMPLE_RATE, dtype=np.float32),
                    path_or_hf_repo=self.model_repo,
                    language=self.language,
                )
                log.info("Model %s loaded", key)
                self._set_status(f"Mikrofon: {self.device_name}")
                self._notify("Modell laddad", MODELS[key]["label"])
            except Exception as e:
                log.exception("Model switch failed")
                self._notify("Fel vid modellbyte", str(e))
            self._show_icon(self._icon_idle)

        threading.Thread(target=preload, daemon=True).start()

    def _toggle_language(self, sender):
        self.language = "en" if self.language == "sv" else "sv"
        label = "Svenska" if self.language == "sv" else "English"
        sender.title = f"Språk: {label}"
        self._notify("Språk", label)
        log.info("Language → %s", self.language)

    def _edit_prompt(self, _sender=None):
        open_prompt_file()

    def _recalibrate_cb(self, _sender=None):
        def do():
            self._notify("Kalibrerar…", "Var tyst i 2 sekunder")
            self._calibrate()
            self._notify("Klart!", f"Tröskel: {self.silence_threshold:.4f}")

        threading.Thread(target=do, daemon=True).start()

    def _open_log(self, _sender=None):
        subprocess.Popen(["open", "-a", "Console", LOG_PATH])

    def _quit(self, _sender=None):
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


def _find_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    for p in ("~/.local/bin/uv", "~/.cargo/bin/uv"):
        expanded = os.path.expanduser(p)
        if os.path.isfile(expanded):
            return expanded
    return "uv"


def install_launchagent():
    plist_dir = os.path.expanduser("~/Library/LaunchAgents")
    plist_path = os.path.join(plist_dir, f"{LAUNCHAGENT_LABEL}.plist")
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

    os.makedirs(plist_dir, exist_ok=True)
    if os.path.exists(plist_path):
        subprocess.run(["launchctl", "unload", plist_path], capture_output=True)

    with open(plist_path, "w") as f:
        f.write(plist)

    subprocess.run(["launchctl", "load", plist_path], check=True)
    print(f"  Installed: {plist_path}")
    print("  PTT starts automatically on login.")
    print("  Remove with: ptt --uninstall")


def uninstall_launchagent():
    plist_path = os.path.expanduser(
        f"~/Library/LaunchAgents/{LAUNCHAGENT_LABEL}.plist"
    )
    if os.path.exists(plist_path):
        subprocess.run(["launchctl", "unload", plist_path], capture_output=True)
        os.remove(plist_path)
        print("  LaunchAgent removed.")
    else:
        print("  No LaunchAgent found.")


def main():
    parser = argparse.ArgumentParser(
        prog="ptt",
        description="PTT — Push-to-Talk Transcription for macOS",
    )
    parser.add_argument(
        "--lang", default=DEFAULT_LANG, metavar="CODE",
        help="Language code, e.g. sv, en, de (default: sv)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, choices=list(MODELS),
        help="Whisper model preset (default: turbo)",
    )
    parser.add_argument(
        "--key", default="alt_r", choices=list(KEYCODES),
        help="Hotkey modifier (default: alt_r = Right Option)",
    )
    parser.add_argument(
        "--device", type=int, default=None, metavar="N",
        help="Audio input device index (default: auto-detect built-in mic)",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List audio input devices and exit",
    )
    parser.add_argument(
        "--install", action="store_true",
        help="Install LaunchAgent for auto-start on login",
    )
    parser.add_argument(
        "--uninstall", action="store_true",
        help="Remove LaunchAgent",
    )
    args = parser.parse_args()

    if args.list_devices:
        list_devices()
        return
    if args.install:
        install_launchagent()
        return
    if args.uninstall:
        uninstall_launchagent()
        return

    PTTApp(
        model_key=args.model,
        language=args.lang,
        hotkey=args.key,
        device=args.device,
    ).run()


if __name__ == "__main__":
    main()
