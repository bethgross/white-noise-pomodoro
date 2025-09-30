import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd
import tkinter as tk

try:
    from tkmacosx import Button as MacOSButton
except ImportError:  # pragma: no cover - fallback when dependency missing
    MacOSButton = tk.Button


POMODORO_DURATION_SECONDS = 25 * 60
SHORT_BREAK_DURATION_SECONDS = 5 * 60


@dataclass
class NoiseProfile:
    """Encapsulates the tunable properties of the generated noise."""

    sample_rate: int = 44_100
    amplitude: float = 0.12  # Keep below 0.3 to avoid clipping/distortion


WHITE_NOISE_PROFILE = NoiseProfile()
DING_FREQUENCY_HZ = 880
DING_DURATION_SECONDS = 0.6


class WhiteNoisePlayer:
    """Continuously streams procedurally generated white noise."""

    def __init__(self, profile: NoiseProfile) -> None:
        self.profile = profile
        self._stream: Optional[sd.OutputStream] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._stream is None:
                self._stream = sd.OutputStream(
                    callback=self._callback,
                    channels=1,
                    dtype="float32",
                    samplerate=self.profile.sample_rate,
                )
                self._stream.start()

    def stop(self) -> None:
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                finally:
                    self._stream = None

    def _callback(self, outdata, frames, time_info, status) -> None:  # type: ignore[override]
        if status:
            print(f"Audio stream warning: {status}")
        noise = np.random.uniform(-1.0, 1.0, frames).astype(np.float32)
        outdata[:, 0] = noise * self.profile.amplitude


class PomodoroApp:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("White Noise Pomodoro")
        self.root.configure(bg="black")
        self.root.geometry("320x320")
        self.root.resizable(width=False, height=False)

        self.timer_var = tk.StringVar(value="00:00")
        self.white_noise_var = tk.BooleanVar(value=True)

        self._build_ui()

        self.noise_player = WhiteNoisePlayer(WHITE_NOISE_PROFILE)

        self.remaining_seconds = 0
        self.timer_running = False
        self.timer_id: Optional[str] = None
        self.noise_desired = False

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        self.timer_label = tk.Label(
            self.root,
            textvariable=self.timer_var,
            fg="white",
            bg="black",
            font=("Helvetica Neue", 48),
        )
        self.timer_label.pack(pady=(24, 12))

        button_frame = tk.Frame(self.root, bg="black")
        button_frame.pack(pady=12)

        pomodoro_kwargs = dict(
            fg="#0A0A0A",
            bg="#89CFF0",
            activebackground="#6FBDE9",
            activeforeground="#0A0A0A",
            relief="flat",
            padx=20,
            pady=12,
            font=("Helvetica Neue", 14),
            borderwidth=0,
            highlightthickness=0,
        )
        if MacOSButton is not tk.Button:
            pomodoro_kwargs.update(borderless=1, focuscolor="")

        self.pomodoro_button = MacOSButton(
            button_frame,
            text="Start Pomodoro",
            command=self.start_pomodoro,
            **pomodoro_kwargs,
        )
        self.pomodoro_button.pack(fill="x", pady=6)

        break_kwargs = dict(
            fg="#0A0A0A",
            bg="#89CFF0",
            activebackground="#6FBDE9",
            activeforeground="#0A0A0A",
            relief="flat",
            padx=20,
            pady=12,
            font=("Helvetica Neue", 14),
            borderwidth=0,
            highlightthickness=0,
        )
        if MacOSButton is not tk.Button:
            break_kwargs.update(borderless=1, focuscolor="")

        self.break_button = MacOSButton(
            button_frame,
            text="Start Break",
            command=self.start_break,
            **break_kwargs,
        )
        self.break_button.pack(fill="x", pady=6)

        toggle_frame = tk.Frame(self.root, bg="black")
        toggle_frame.pack(pady=(8, 16))

        self.white_noise_toggle = tk.Checkbutton(
            toggle_frame,
            text="White Noise",
            variable=self.white_noise_var,
            command=self.on_noise_toggle,
            fg="white",
            bg="black",
            activebackground="black",
            activeforeground="white",
            selectcolor="black",
            font=("Helvetica Neue", 12),
        )
        self.white_noise_toggle.pack()

    def start(self) -> None:
        self.root.mainloop()

    def start_pomodoro(self) -> None:
        self._start_timer(POMODORO_DURATION_SECONDS, wants_noise=True)

    def start_break(self) -> None:
        self._start_timer(SHORT_BREAK_DURATION_SECONDS, wants_noise=False)

    def _start_timer(self, duration_seconds: int, *, wants_noise: bool) -> None:
        self._stop_timer_internal()
        self.remaining_seconds = duration_seconds
        self.timer_running = True
        self.noise_desired = wants_noise
        self._update_timer_label()
        self._sync_noise_state()
        self.timer_id = self.root.after(1000, self._tick)

    def _tick(self) -> None:
        if not self.timer_running:
            return

        self.remaining_seconds -= 1
        self._update_timer_label()

        if self.remaining_seconds <= 0:
            self.timer_running = False
            self.timer_id = None
            self.noise_desired = False
            self.noise_player.stop()
            self._play_ding_async()
            return

        self.timer_id = self.root.after(1000, self._tick)

    def _update_timer_label(self) -> None:
        minutes, seconds = divmod(max(self.remaining_seconds, 0), 60)
        self.timer_var.set(f"{minutes:02}:{seconds:02}")

    def on_noise_toggle(self) -> None:
        self._sync_noise_state()

    def _sync_noise_state(self) -> None:
        if self.timer_running and self.noise_desired and self.white_noise_var.get():
            try:
                self.noise_player.start()
            except Exception as exc:  # pragma: no cover - defensive guardrail
                print(f"Unable to start white noise: {exc}")
        else:
            self.noise_player.stop()

    def _play_ding_async(self) -> None:
        threading.Thread(target=self._play_ding, daemon=True).start()

    def _play_ding(self) -> None:
        samples = int(WHITE_NOISE_PROFILE.sample_rate * DING_DURATION_SECONDS)
        t = np.linspace(0, DING_DURATION_SECONDS, samples, endpoint=False)
        waveform = 0.3 * np.sin(2 * np.pi * DING_FREQUENCY_HZ * t)
        try:
            sd.play(waveform.astype(np.float32), WHITE_NOISE_PROFILE.sample_rate, blocking=True)
        except Exception as exc:  # pragma: no cover - defensive guardrail
            print(f"Unable to play ding: {exc}")

    def on_close(self) -> None:
        self._stop_timer_internal()
        self.noise_player.stop()
        self.root.destroy()

    def _stop_timer_internal(self) -> None:
        self.timer_running = False
        self.noise_desired = False
        if self.timer_id is not None:
            self.root.after_cancel(self.timer_id)
            self.timer_id = None
        self.noise_player.stop()
        self.remaining_seconds = 0
        self._update_timer_label()


def main() -> None:
    app = PomodoroApp()
    app.start()


if __name__ == "__main__":
    main()
