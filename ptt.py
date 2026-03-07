#!/usr/bin/env python3
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

Features:
  - Auto-calibrates mic sensitivity (works with whispering)
  - Pre-roll buffer captures word onsets before key press
  - Chunked transcription pastes text during natural pauses
  - Built-in mic auto-detection (skips AirPlay/HomePod)
  - Switchable models: fast general or optimized Swedish
  - Auto-start via LaunchAgent

Usage:
  uv run ptt.py                     # Start (Swedish, Right Option)
  uv run ptt.py --lang en           # English mode
  uv run ptt.py --model kb-sv       # Swedish-optimized model
  uv run ptt.py --key alt           # Use Left Option instead
  uv run ptt.py --list-devices      # Show audio input devices
  uv run ptt.py --install           # Auto-start on login
  uv run ptt.py --uninstall         # Remove auto-start

Requires: macOS 13+, Apple Silicon, Accessibility permission.
"""

import argparse
import collections
import ctypes
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
BLOCK_DURATION = 0.1  # seconds per audio block
MIN_SPEECH_SECONDS = 0.3  # skip chunks shorter than this
PAUSE_SECONDS = 0.8  # silence duration before chunk boundary
PRE_ROLL_SECONDS = 0.4  # audio kept before key press
CALIBRATION_SECONDS = 2.0  # ambient noise measurement
THRESHOLD_MULTIPLIER = 3.0  # silence threshold = ambient × this
MIN_THRESHOLD = 0.002  # absolute floor for silence detection

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

# LaunchAgent identifier
LAUNCHAGENT_LABEL = "com.ptt.transcription"

# ---------------------------------------------------------------------------
# Hallucination filter
# ---------------------------------------------------------------------------

_HALLUCINATION_ARTIFACTS = frozenset({
    "thank you", "thanks for watching", "subscribe", "you",
    "tack för att ni tittade", "tack för att ni tittar",
    "undertextning", "musik", "textning", "textning.nu",
    "untertitelung", "sous-titrage", "amara.org",
    "text", ".", "..", "...",
})


def is_hallucination(text: str) -> bool:
    """Return True if the transcription looks like a Whisper hallucination."""
    t = text.strip().lower()
    if not t or len(t) <= 1:
        return True
    if t[0] in "([♪♫":
        return True
    if t in _HALLUCINATION_ARTIFACTS:
        return True
    # Repeated phrases: "Thank you. Thank you. Thank you."
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
    """Check Accessibility permission (needed for global hotkey + paste)."""
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
    """Find the built-in microphone, skipping AirPlay / HomePod / Bluetooth."""
    import sounddevice as sd

    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] <= 0:
            continue
        name = d["name"].lower()
        if any(w in name for w in ("macbook", "built-in", "mikrofon", "internal")):
            return i, d["name"]
    # Fallback to system default
    idx = sd.default.device[0]
    if idx is not None:
        return idx, sd.query_devices(idx)["name"]
    return None, "Unknown"


def paste_text(text: str):
    """Copy text to clipboard and simulate Cmd+V via Quartz CGEvent."""
    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventPost,
        CGEventSetFlags,
        kCGEventFlagMaskCommand,
        kCGHIDEventTap,
    )

    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    time.sleep(0.03)

    v_keycode = 9
    for pressed in (True, False):
        ev = CGEventCreateKeyboardEvent(None, v_keycode, pressed)
        CGEventSetFlags(ev, kCGEventFlagMaskCommand)
        CGEventPost(kCGHIDEventTap, ev)
        if pressed:
            time.sleep(0.03)


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

    # ---- Menu bar --------------------------------------------------------

    def run(self):
        import rumps
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory
        )

        self._app = rumps.App("PTT", title="◉", quit_button=None)

        self._status_item = rumps.MenuItem("Startar…")
        lang_item = rumps.MenuItem(
            f"Språk: {'Svenska' if self.language == 'sv' else 'English'}",
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

    def _set_title(self, title: str):
        try:
            if self._app:
                self._app.title = title
        except Exception:
            pass

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
            self._set_title("⏳")
            self._set_status("Laddar modell…")
            log.info("Loading model: %s", self.model_repo)

            import mlx_whisper
            mlx_whisper.transcribe(
                np.zeros(SAMPLE_RATE, dtype=np.float32),
                path_or_hf_repo=self.model_repo,
                language=self.language,
            )
            log.info("Model loaded")

            # Mic
            if self.device_idx is not None:
                self.device_name = sd.query_devices(self.device_idx)["name"]
            else:
                self.device_idx, self.device_name = find_builtin_mic()
            log.info("Mic: %s (device %s)", self.device_name, self.device_idx)

            # Calibrate
            self._calibrate()

            # Audio stream
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=self.block_size,
                device=self.device_idx,
                callback=self._audio_cb,
            )
            self._stream.start()
            log.info("Audio stream started")

            # Hotkey monitors
            mask = 1 << 12  # NSEventMaskFlagsChanged

            def on_global(event):
                self._on_modifier(event)

            def on_local(event):
                self._on_modifier(event)
                return event

            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, on_global)
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, on_local)
            log.info("Hotkey active (keycode %d)", self.hotkey_code)

            # Ready
            self.ready = True
            self._set_title("◉")
            self._set_status(f"Mikrofon: {self.device_name}")
            hotkey_label = HOTKEY_LABELS.get(self.hotkey, self.hotkey)
            self._notify("Redo!", f"Håll {hotkey_label} och prata")
            log.info("PTT ready")

        except Exception as e:
            log.exception("Init failed")
            self._set_title("⚠")
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
        log.info(
            "Ambient RMS=%.5f → threshold=%.5f",
            ambient_rms,
            self.silence_threshold,
        )

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
        self.recording = True
        self._set_title("●")
        log.info("REC start")

        # Drain stale data
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break

        # Pre-roll: capture word onsets before key was pressed
        for block in self.pre_roll:
            self.audio_queue.put(block)
        self.pre_roll.clear()

        self._rec_thread = threading.Thread(target=self._rec_loop, daemon=True)
        self._rec_thread.start()

    def _stop_rec(self):
        self.recording = False
        log.info("REC stop")
        if self._rec_thread:
            self._rec_thread.join(timeout=15)
        self._set_title("◉")

    def _rec_loop(self):
        speech_buf: list = []
        silent_blocks = 0
        has_speech = False

        while self.recording:
            try:
                block = self.audio_queue.get(timeout=0.05)
            except queue.Empty:
                continue

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

    # ---- Transcription ---------------------------------------------------

    def _transcribe(self, blocks):
        import mlx_whisper

        audio = np.concatenate(blocks).flatten().astype(np.float32)
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_SPEECH_SECONDS:
            log.info("Skipped chunk (%.2fs < min)", duration)
            return

        self._set_title("⏳")
        t0 = time.monotonic()

        try:
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model_repo,
                language=self.language,
            )
        except Exception:
            log.exception("Transcription error")
            self._set_title("●" if self.recording else "◉")
            return

        elapsed = time.monotonic() - t0
        text = result.get("text", "").strip()

        if text and not is_hallucination(text):
            try:
                paste_text(text + " ")
                speed = duration / elapsed if elapsed > 0 else 0
                log.info(
                    "'%s'  (%.1fs audio → %.1fs, %.0f×)",
                    text, duration, elapsed, speed,
                )
            except Exception:
                log.exception("Paste failed — text is in clipboard")
        else:
            log.info("Filtered: '%s'", text)

        self._set_title("●" if self.recording else "◉")

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
            self._set_title("⏳")
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
            self._set_title("◉")

        threading.Thread(target=preload, daemon=True).start()

    def _toggle_language(self, sender):
        self.language = "en" if self.language == "sv" else "sv"
        label = "Svenska" if self.language == "sv" else "English"
        sender.title = f"Språk: {label}"
        self._notify("Språk", label)
        log.info("Language → %s", self.language)

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
    """Locate the uv binary."""
    uv = shutil.which("uv")
    if uv:
        return uv
    candidate = os.path.expanduser("~/.local/bin/uv")
    if os.path.isfile(candidate):
        return candidate
    candidate = os.path.expanduser("~/.cargo/bin/uv")
    if os.path.isfile(candidate):
        return candidate
    return "uv"


def install_launchagent():
    """Install a LaunchAgent so PTT starts automatically on login."""
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
    print("  PTT will start automatically on login.")
    print("  Remove with: ptt --uninstall")


def uninstall_launchagent():
    plist_path = os.path.expanduser(
        f"~/Library/LaunchAgents/{LAUNCHAGENT_LABEL}.plist"
    )
    if os.path.exists(plist_path):
        subprocess.run(["launchctl", "unload", plist_path], capture_output=True)
        os.remove(plist_path)
        print("  LaunchAgent removed. PTT will no longer auto-start.")
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
