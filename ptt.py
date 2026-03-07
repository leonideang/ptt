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
"""PTT — Push-to-Talk Transcription for macOS.

Menu bar app. Hold Right Option to record, release to transcribe and paste.
Auto-calibrates mic for whispering. Runs in background without terminal.

Usage:
  uv run ptt.py                  # Start menu bar app
  uv run ptt.py --lang en        # English mode
  uv run ptt.py --list-devices   # Show audio devices
"""

import argparse
import collections
import ctypes
import logging
import os
import queue
import subprocess
import sys
import threading
import time

import numpy as np

# --- Logging ---
LOG_PATH = "/tmp/ptt.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
)
log = logging.getLogger("ptt")

# --- Config ---
MODELS = {
    "turbo": {
        "repo": "mlx-community/whisper-large-v3-turbo",
        "label": "Turbo (snabb, bra)",
    },
    "kb-sv": {
        "repo": "bratland/kb-whisper-large-mlx",
        "label": "KB Swedish (bast svenska, langsammare)",
    },
}
DEFAULT_MODEL_KEY = "turbo"
DEFAULT_LANG = "sv"
SAMPLE_RATE = 16000
BLOCK_DURATION = 0.1
MIN_SPEECH_SECONDS = 0.3
PAUSE_SECONDS = 0.8
PRE_ROLL_SECONDS = 0.4
CALIBRATION_SECONDS = 2.0
THRESHOLD_MULTIPLIER = 3.0
MIN_THRESHOLD = 0.002

# macOS keycodes
KEYCODES = {"alt_r": 61, "alt": 58, "ctrl_r": 62, "ctrl": 59}
KEYCODE_FLAG = {61: 1 << 19, 58: 1 << 19, 62: 1 << 18, 59: 1 << 18}
KEYCODE_LABELS = {
    "alt_r": "Right Option",
    "alt": "Left Option",
    "ctrl_r": "Right Control",
    "ctrl": "Left Control",
}


# --- Helpers ---


def check_accessibility():
    """Check if we have Accessibility permissions (needed for hotkey + paste)."""
    try:
        lib = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/ApplicationServices.framework/ApplicationServices"
        )
        lib.AXIsProcessTrusted.restype = ctypes.c_bool
        return lib.AXIsProcessTrusted()
    except Exception:
        return True  # Assume yes if check fails


def open_accessibility_prefs():
    subprocess.Popen(
        [
            "open",
            "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility",
        ]
    )


def find_builtin_mic():
    """Find the built-in MacBook mic, avoiding AirPlay/HomePod/external."""
    import sounddevice as sd

    devices = sd.query_devices()
    for i, d in enumerate(devices):
        if d["max_input_channels"] <= 0:
            continue
        name = d["name"].lower()
        if "macbook" in name or "built-in" in name or "mikrofon" in name:
            return i, d["name"]
    idx = sd.default.device[0]
    if idx is not None:
        return idx, sd.query_devices(idx)["name"]
    return None, "Unknown"


def is_hallucination(text: str) -> bool:
    t = text.strip().lower()
    if not t or len(t) <= 1:
        return True
    if t.startswith(("(", "[", "♪", "♫")):
        return True
    words = t.split()
    if len(words) >= 4:
        half = len(words) // 2
        if words[:half] == words[half : half * 2]:
            return True
    artifacts = {
        "thank you", "thanks for watching", "subscribe", "you",
        "tack för att ni tittade", "tack för att ni tittar",
        "undertextning", "musik", "textning", "textning.nu",
        "text", ".", "..", "...", "untertitelung", "sous-titrage",
    }
    return t in artifacts


def paste_text(text: str):
    """Copy to clipboard + simulate Cmd+V via CGEvent."""
    from Quartz import (
        CGEventCreateKeyboardEvent,
        CGEventPost,
        CGEventSetFlags,
        kCGEventFlagMaskCommand,
        kCGHIDEventTap,
    )

    subprocess.run(["pbcopy"], input=text.encode("utf-8"), check=True)
    time.sleep(0.03)

    v_key = 9  # macOS keycode for 'v'
    down = CGEventCreateKeyboardEvent(None, v_key, True)
    CGEventSetFlags(down, kCGEventFlagMaskCommand)
    up = CGEventCreateKeyboardEvent(None, v_key, False)
    CGEventSetFlags(up, kCGEventFlagMaskCommand)
    CGEventPost(kCGHIDEventTap, down)
    time.sleep(0.03)
    CGEventPost(kCGHIDEventTap, up)


# --- App ---


class PTTApp:
    def __init__(self, model_key: str, language: str, hotkey: str, device: int | None):
        self.model_key = model_key
        self.model = MODELS[model_key]["repo"]
        self.language = language
        self.hotkey = hotkey
        self.hotkey_code = KEYCODES.get(hotkey, 61)
        self.hotkey_flag = KEYCODE_FLAG.get(self.hotkey_code, 1 << 19)
        self.device_idx = device
        self.device_name = ""
        self.recording = False
        self.ready = False
        self.silence_threshold = 0.005
        self.audio_queue: queue.Queue = queue.Queue()
        self.block_size = int(SAMPLE_RATE * BLOCK_DURATION)
        self.pre_roll: collections.deque = collections.deque(
            maxlen=int(PRE_ROLL_SECONDS / BLOCK_DURATION)
        )
        self._rec_thread: threading.Thread | None = None
        self._stream = None
        self._app = None
        self._lang_item = None
        self._status_item = None

    # --- rumps menu bar ---

    def run(self):
        import rumps
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

        # Hide from Dock — menu bar only
        NSApplication.sharedApplication().setActivationPolicy_(
            NSApplicationActivationPolicyAccessory
        )

        self._app = rumps.App("PTT", title="◉", quit_button=None)

        self._status_item = rumps.MenuItem("Laddar...")
        self._lang_item = rumps.MenuItem(
            f"Sprak: {'Svenska' if self.language == 'sv' else 'English'}",
            callback=self._toggle_language,
        )

        # Model submenu
        self._model_menu = rumps.MenuItem("Modell")
        self._model_items = {}
        for key, info in MODELS.items():
            item = rumps.MenuItem(info["label"], callback=self._switch_model)
            item._model_key = key
            if key == self.model_key:
                item.state = 1  # checkmark
            self._model_items[key] = item
            self._model_menu.add(item)

        self._app.menu = [
            self._status_item,
            None,
            self._lang_item,
            self._model_menu,
            rumps.MenuItem("Kalibrera mikrofon", callback=self._recalibrate_cb),
            None,
            rumps.MenuItem("Visa logg", callback=self._open_log),
            rumps.MenuItem("Avsluta PTT", callback=self._quit),
        ]

        # Check accessibility before starting
        if not check_accessibility():
            log.warning("No Accessibility permissions!")
            import rumps as r

            r.alert(
                title="PTT behover Accessibility",
                message=(
                    "PTT behover Accessibility-behorighet for att lasa tangenter "
                    "och klistra in text.\n\n"
                    "Lagg till din terminal (eller PTT) i:\n"
                    "System Settings > Privacy & Security > Accessibility"
                ),
                ok="Oppna installningar",
            )
            open_accessibility_prefs()

        threading.Thread(target=self._init_background, daemon=True).start()
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

    def _notify(self, title: str, msg: str):
        try:
            import rumps
            rumps.notification("PTT", title, msg, sound=False)
        except Exception:
            pass

    # --- Initialization ---

    def _init_background(self):
        import sounddevice as sd
        from AppKit import NSEvent

        try:
            # 1. Load model
            self._set_title("⏳")
            self._set_status("Laddar modell...")
            log.info("Loading model: %s", self.model)

            import mlx_whisper

            mlx_whisper.transcribe(
                np.zeros(SAMPLE_RATE, dtype=np.float32),
                path_or_hf_repo=self.model,
                language=self.language,
            )
            log.info("Model loaded")

            # 2. Find mic
            if self.device_idx is not None:
                self.device_name = sd.query_devices(self.device_idx)["name"]
            else:
                self.device_idx, self.device_name = find_builtin_mic()
            log.info("Mic: %s (device %s)", self.device_name, self.device_idx)
            self._set_status(f"Mic: {self.device_name}")

            # 3. Calibrate
            self._calibrate()

            # 4. Audio stream
            self._stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=1,
                blocksize=self.block_size,
                device=self.device_idx,
                callback=self._audio_cb,
            )
            self._stream.start()
            log.info("Audio stream started")

            # 5. Hotkey monitors (NSEvent)
            mask = 1 << 12  # NSEventMaskFlagsChanged

            def on_global(event):
                self._on_modifier(event)

            def on_local(event):
                self._on_modifier(event)
                return event  # pass event through

            NSEvent.addGlobalMonitorForEventsMatchingMask_handler_(mask, on_global)
            NSEvent.addLocalMonitorForEventsMatchingMask_handler_(mask, on_local)
            log.info("Hotkey monitors active (keycode %d)", self.hotkey_code)

            # Ready
            self.ready = True
            self._set_title("◉")
            hotkey_label = KEYCODE_LABELS.get(self.hotkey, "Right Option")
            self._notify("Redo!", f"Hall {hotkey_label} och prata")
            log.info("PTT ready!")

        except Exception as e:
            log.exception("Init failed: %s", e)
            self._set_title("⚠")
            self._set_status(f"Fel: {e}")
            self._notify("Fel vid start", str(e))

    def _calibrate(self):
        import sounddevice as sd

        log.info("Calibrating mic...")
        rec = sd.rec(
            int(SAMPLE_RATE * CALIBRATION_SECONDS),
            samplerate=SAMPLE_RATE,
            channels=1,
            device=self.device_idx,
        )
        sd.wait()
        ambient_rms = np.sqrt(np.mean(rec**2))
        self.silence_threshold = max(ambient_rms * THRESHOLD_MULTIPLIER, MIN_THRESHOLD)
        log.info(
            "Ambient RMS=%.5f, threshold=%.5f", ambient_rms, self.silence_threshold
        )

    # --- Hotkey handling ---

    def _on_modifier(self, event):
        if not self.ready:
            return
        if event.keyCode() != self.hotkey_code:
            return

        flags = event.modifierFlags()
        pressed = bool(flags & self.hotkey_flag)

        if pressed and not self.recording:
            self._start_rec()
        elif not pressed and self.recording:
            self._stop_rec()

    # --- Recording ---

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

        # Pre-roll: capture word onsets before key press
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

            rms = np.sqrt(np.mean(block**2))

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

        # Final chunk on release
        if speech_buf and has_speech:
            self._transcribe(speech_buf)

    def _transcribe(self, blocks):
        import mlx_whisper

        audio = np.concatenate(blocks).flatten().astype(np.float32)
        duration = len(audio) / SAMPLE_RATE
        if duration < MIN_SPEECH_SECONDS:
            log.info("Skipped: too short (%.2fs)", duration)
            return

        self._set_title("⏳")
        t0 = time.monotonic()

        try:
            result = mlx_whisper.transcribe(
                audio,
                path_or_hf_repo=self.model,
                language=self.language,
            )
        except Exception as e:
            log.error("Transcription error: %s", e)
            self._set_title("●" if self.recording else "◉")
            return

        elapsed = time.monotonic() - t0
        text = result.get("text", "").strip()

        if text and not is_hallucination(text):
            try:
                paste_text(text + " ")
                speed = duration / elapsed if elapsed > 0 else 0
                log.info("'%s' (%.1fs audio, %.1fs proc, %.0fx)", text, duration, elapsed, speed)
            except Exception as e:
                log.error("Paste failed: %s (text in clipboard)", e)
        else:
            log.info("Filtered hallucination: '%s'", text)

        self._set_title("●" if self.recording else "◉")

    # --- Menu callbacks ---

    def _switch_model(self, sender):
        key = sender._model_key
        if key == self.model_key:
            return
        self.model_key = key
        self.model = MODELS[key]["repo"]
        # Update checkmarks
        for k, item in self._model_items.items():
            item.state = 1 if k == key else 0
        log.info("Switching model to: %s (%s)", key, self.model)
        self._notify("Modell", MODELS[key]["label"])

        # Pre-load model in background
        def preload():
            self._set_title("⏳")
            self._set_status("Laddar modell...")
            try:
                import mlx_whisper
                mlx_whisper.transcribe(
                    np.zeros(SAMPLE_RATE, dtype=np.float32),
                    path_or_hf_repo=self.model,
                    language=self.language,
                )
                log.info("Model %s loaded", key)
                self._set_status(f"Mic: {self.device_name}")
                self._notify("Modell laddad", MODELS[key]["label"])
            except Exception as e:
                log.error("Model load failed: %s", e)
                self._notify("Fel", str(e))
            self._set_title("◉")

        threading.Thread(target=preload, daemon=True).start()

    def _toggle_language(self, sender):
        self.language = "en" if self.language == "sv" else "sv"
        label = "Svenska" if self.language == "sv" else "English"
        sender.title = f"Sprak: {label}"
        self._notify("Sprak", label)
        log.info("Language: %s", self.language)

    def _recalibrate_cb(self, sender=None):
        def do():
            self._notify("Kalibrerar...", "Var tyst i 2 sekunder")
            self._calibrate()
            self._notify("Klart!", f"Troskel: {self.silence_threshold:.4f}")

        threading.Thread(target=do, daemon=True).start()

    def _open_log(self, sender=None):
        subprocess.Popen(["open", "-a", "Console", LOG_PATH])

    def _quit(self, sender=None):
        import rumps

        self.recording = False
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
        rumps.quit_application()


# --- CLI ---


def list_devices():
    import sounddevice as sd

    devices = sd.query_devices()
    builtin_idx, _ = find_builtin_mic()
    print("\n  Audio input devices:\n")
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            markers = []
            if i == sd.default.device[0]:
                markers.append("default")
            if i == builtin_idx:
                markers.append("built-in")
            suffix = f"  ({', '.join(markers)})" if markers else ""
            print(f"    {i}: {d['name']}{suffix}")
    print()


def install_launchagent():
    """Install LaunchAgent for auto-start on login."""
    plist_name = "com.leon.ptt.plist"
    plist_dir = os.path.expanduser("~/Library/LaunchAgents")
    plist_path = os.path.join(plist_dir, plist_name)
    uv_path = os.path.expanduser("~/.local/bin/uv")
    script_path = os.path.abspath(__file__)

    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.leon.ptt</string>
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
    <string>/tmp/ptt-stdout.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/ptt-stderr.log</string>
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

    # Unload existing if present
    if os.path.exists(plist_path):
        subprocess.run(["launchctl", "unload", plist_path], capture_output=True)

    with open(plist_path, "w") as f:
        f.write(content)

    subprocess.run(["launchctl", "load", plist_path], check=True)
    print(f"  Installed: {plist_path}")
    print("  PTT will auto-start on login.")
    print("  To remove: ptt --uninstall")


def uninstall_launchagent():
    plist_path = os.path.expanduser("~/Library/LaunchAgents/com.leon.ptt.plist")
    if os.path.exists(plist_path):
        subprocess.run(["launchctl", "unload", plist_path], capture_output=True)
        os.remove(plist_path)
        print("  LaunchAgent removed. PTT will no longer auto-start.")
    else:
        print("  No LaunchAgent found.")


def main():
    parser = argparse.ArgumentParser(description="PTT — Push-to-Talk Transcription")
    model_choices = list(MODELS.keys())
    parser.add_argument("--lang", default=DEFAULT_LANG, help="Language (sv/en)")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL_KEY, choices=model_choices,
        help=f"Model preset ({', '.join(model_choices)})",
    )
    parser.add_argument("--key", default="alt_r", choices=KEYCODES.keys(), help="Hotkey")
    parser.add_argument("--device", type=int, default=None, help="Audio device index")
    parser.add_argument("--list-devices", action="store_true", help="List audio devices")
    parser.add_argument("--install", action="store_true", help="Install auto-start")
    parser.add_argument("--uninstall", action="store_true", help="Remove auto-start")
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

    app = PTTApp(
        model_key=args.model,
        language=args.lang,
        hotkey=args.key,
        device=args.device,
    )
    app.run()


if __name__ == "__main__":
    main()
