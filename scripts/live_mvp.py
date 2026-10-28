"""Desktop panel for the current local-bot MVP."""

import argparse
import json
import multiprocessing as mp
import threading
import time
import tkinter as tk
import traceback
from collections import deque
from dataclasses import asdict
from pathlib import Path
from queue import Empty
from tkinter import ttk

from mvp_core import Controller, Detection, Settings
from mvp_motion import MotionHistory, MotionPlan
from mvp_runtime import inference_process
from mvp_windows import KERNEL, USER, C, Mouse, W, client_rect, foreground, input_environment

ROOT = Path(__file__).resolve().parents[1]
TITLE = "CS2 ML — MVP v3.7"


class Engine:
    def __init__(
        self,
        mode,
        settings,
        imgsz,
        hz,
        pipeline="full",
        model_backend="auto",
        smooth_tracking=True,
        capture_backend="dxgi",
        blocking_gpu_wait=False,
        death_source="manual",
    ):
        self.mode = mode
        self.settings = settings
        self.pipeline = pipeline
        self.stop = threading.Event()
        self.lock = threading.RLock()
        self.armed = False
        self.fire = True
        self.ready = False
        self.hwnd = None
        self.last_motion_at = 0
        self.controller = Controller(settings)
        self.mouse = Mouse()
        self.motion = MotionPlan(duration=0.018 if smooth_tracking else 0)
        self.history = MotionHistory()
        self.last_snap_at = 0
        self.state = {
            "status": "Загрузка модели в отдельном процессе…",
            "frames": 0,
            "shots": 0,
            "moves": 0,
            "fps": 0,
            "latency_ms": 0,
            "predict_ms": 0,
            "capture": "WGC",
            "input_error": "",
            "version": "3.7",
            "blocking_gpu_wait": blocking_gpu_wait,
            "smooth_tracking": smooth_tracking,
            "input_loop_p95_ms": 0,
            "input_loop_hz": 0,
            "glide_steps": 0,
            "acquisition": "whole frame" if settings.radius <= 0 else f"radius {settings.radius}",
            "death_source": death_source,
            "visual_feed_status": "Плашка: ожидание захвата",
            "confirmed_kill_releases": 0,
        }
        self.state["capture"] = capture_backend.upper()
        self.output = ROOT / "runs/live-mvp" / time.strftime("%Y%m%d-%H%M%S")
        self.output.mkdir(parents=True, exist_ok=False)
        config = dict(
            mode=mode,
            imgsz=imgsz,
            hz=hz,
            pipeline=pipeline,
            model_backend=model_backend,
            capture_backend=capture_backend,
            blocking_gpu_wait=blocking_gpu_wait,
            death_source=death_source,
            output=str(self.output),
        )
        (self.output / "settings.json").write_text(
            json.dumps(
                dict(
                    config,
                    controller=asdict(settings),
                    version="3.7",
                    capture_warmup_seconds=10,
                    input_wait="CPython high-resolution time.sleep",
                    tracking_duration_ms=self.motion.duration * 1000,
                    tracking_tick_ms=4,
                ),
                indent=2,
            )
        )
        (self.output / "diagnostics.json").write_text(json.dumps(input_environment(), indent=2))
        ctx = mp.get_context("spawn")
        self.worker_stop = ctx.Event()
        self.worker_active = ctx.Event()
        self.packets = ctx.Queue(maxsize=1)
        self.messages = ctx.Queue(maxsize=2)
        self.worker = ctx.Process(
            target=inference_process,
            args=(config, self.worker_stop, self.packets, self.messages, self.worker_active),
            daemon=True,
        )
        self.worker.start()
        from mvp_metrics import memory_monitor

        self.metrics_thread = threading.Thread(
            target=memory_monitor,
            args=(self.stop, self.state, self.output, self.worker.pid),
            daemon=True,
        )
        self.metrics_thread.start()
        self.thread = threading.Thread(target=self.inputs, daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            return dict(
                self.state, armed=self.armed, fire=self.fire, ready=self.ready, mode=self.mode
            )

    def disarm(self, reason=None):
        with self.lock:
            self.armed = False
            self.motion.cancel()
            self.history.clear()
            self.controller.reset()
            if hasattr(self, "worker_active"):
                self.worker_active.clear()
            try:
                self.mouse.release()
            except OSError as error:
                self.state["input_error"] = str(error)
            if reason:
                self.state["status"] = reason

    def send_pending(self, log, frame):
        with self.lock:
            if self.stop.is_set() or not self.armed or not foreground(self.hwnd):
                self.motion.cancel()
                return
            event = self.motion.poll(time.perf_counter())
            if event is None:
                return
            if (
                client_rect(self.hwnd) != event["rect"]
                or time.perf_counter() - event["source"] > self.settings.max_age
            ):
                self.motion.cancel()
                return
            try:
                start = time.perf_counter()
                self.mouse.act(event["dx"], event["dy"], event["shoot"])
                sent = time.perf_counter()
                if event["dx"] or event["dy"]:
                    self.last_motion_at = sent
                    self.state["moves"] += 1
                    if event["glide"]:
                        self.state["glide_steps"] = self.state.get("glide_steps", 0) + 1
                    self.controller.acknowledge(event["dx"], event["dy"], *event["size"])
                    self.history.record(
                        sent, event["dx"], event["dy"], event["size"], self.settings
                    )
                    if not event["glide"]:
                        self.last_snap_at = sent
                if event["shoot"]:
                    self.state["shots"] += 1
                    self.controller.last_shot = sent
                log.write(
                    json.dumps(
                        dict(
                            time=time.time(),
                            frame=frame,
                            capture_to_input_ms=(sent - event["source"]) * 1000,
                            control_ms=(sent - start) * 1000,
                            input_timing=getattr(self.mouse, "last_call", {}),
                            **event,
                        )
                    )
                    + "\n"
                )
            except OSError as error:
                self.state["input_error"] = str(error)
                self.disarm("Windows отклонила ввод. Наблюдение продолжается.")
                (self.output / "input-error.txt").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )

    def inputs(self):
        previous = {0x77: False, 0x78: False, 0x75: False, 0x76: False}
        seen = None
        last_accepted = 0
        last_flush = 0
        loop_times = deque(maxlen=256)
        last_loop = None
        last_metrics = 0
        visual_sequence = 0
        log = (self.output / "actions.jsonl").open("w", encoding="utf-8")
        decisions = (self.output / "control.jsonl").open("w", encoding="utf-8")
        input_timings = (self.output / "input-timing.jsonl").open("w", encoding="utf-8")
        try:
            while not self.stop.is_set():
                now = time.perf_counter()
                if self.armed and last_loop is not None:
                    loop_times.append((now - last_loop) * 1000)
                last_loop = now if self.armed else None
                if loop_times and now - last_metrics > 0.5:
                    samples = sorted(loop_times)
                    self.state["input_loop_p95_ms"] = samples[
                        min(len(samples) - 1, int(len(samples) * 0.95))
                    ]
                    self.state["input_loop_hz"] = 1000 / (sum(samples) / len(samples))
                    last_metrics = now
                try:
                    while True:
                        message = self.messages.get_nowait()
                        self.state.update(
                            {k: v for k, v in message.items() if k != "suspend_input"}
                        )
                        self.ready = message.get("ready", self.ready)
                        if "hwnd" in message:
                            self.hwnd = message["hwnd"]
                        if message.get("suspend_input"):
                            self.disarm(message["status"])
                        if "error" in message:
                            self.disarm()
                            self.stop.set()
                except Empty:
                    pass
                if not self.worker.is_alive() and not self.worker_stop.is_set():
                    self.ready = False
                    self.disarm("Процесс вычислений остановлен; смотри error.txt")
                    self.stop.set()
                    break

                for key in previous:
                    value = USER.GetAsyncKeyState(key)
                    down = bool(value & 0x8000)
                    edge = bool(value & 1) or (down and not previous[key])
                    previous[key] = down
                    if not edge:
                        continue
                    if key == 0x75:
                        self.disarm("Остановлено F6")
                        self.stop.set()
                        break
                    if key == 0x78:
                        self.fire = not self.fire
                        self.motion.cancel()
                        self.mouse.release()
                    if key == 0x76:
                        with self.lock:
                            self.motion.cancel()
                            self.history.clear()
                            self.controller.reset()
                            self.mouse.release()
                            self.last_snap_at = time.perf_counter()
                            self.state["status"] = "Цель отпущена вручную. Ожидание нового кадра."
                    if key == 0x77:
                        if self.armed:
                            self.disarm("Выключено. F8 — включить")
                        elif self.state["input_error"]:
                            self.state["status"] = (
                                "Ошибка ввода; перезапусти панель после устранения причины"
                            )
                        elif self.pipeline != "full":
                            self.state["status"] = "Диагностика: управление отключено"
                        elif self.ready and foreground(self.hwnd):
                            self.armed = True
                            self.controller.reset()
                            self.state["status"] = "ACTIVE"
                            if hasattr(self, "worker_active"):
                                self.worker_active.set()
                        else:
                            self.state["status"] = (
                                "Дождись Готово, активируй целевое окно и нажми F8"
                            )
                if self.armed and not foreground(self.hwnd):
                    self.disarm("Потерян фокус. Для возобновления F8")
                packet = None
                try:
                    while True:
                        packet = self.packets.get_nowait()
                except Empty:
                    pass
                if packet is not None and packet["id"] != seen:
                    seen = packet["id"]
                    self.hwnd = packet["hwnd"]
                    now = time.perf_counter()
                    age = now - packet["captured_at"]
                    self.state.update(
                        {
                            k: v
                            for k, v in packet.items()
                            if k not in ("detections", "hwnd", "id", "size", "rect", "captured_at")
                        }
                    )
                    self.state["latency_ms"] = age * 1000
                    sequence = packet.get("visual_kill_sequence", visual_sequence)
                    if sequence != visual_sequence:
                        visual_sequence = sequence
                        event_at = packet.get("visual_kill_at")
                        if (
                            self.armed
                            and foreground(self.hwnd)
                            and event_at is not None
                            and 0 <= now - event_at <= 0.5
                            and 0 <= event_at - self.controller.last_shot <= 2
                        ):
                            with self.lock:
                                if self.controller.release_after_confirmed_kill(now):
                                    self.motion.cancel()
                                    self.mouse.release()

                                    self.last_snap_at = max(self.last_snap_at, event_at - 1e-6)
                                    self.state["confirmed_kill_releases"] += 1
                                    decisions.write(
                                        json.dumps(
                                            dict(
                                                time=time.time(),
                                                frame=seen,
                                                reason="visual_kill_confirmed",
                                                heads=[],
                                                selected=None,
                                                dx=0,
                                                dy=0,
                                                shoot=False,
                                            )
                                        )
                                        + "\n"
                                    )
                    if not self.armed:
                        self.state["status"] = (
                            "Нейросеть на паузе, захват сохраняется. F8 — включить"
                        )
                    valid = (
                        self.armed
                        and foreground(self.hwnd)
                        and packet["rect"] == client_rect(self.hwnd)
                        and packet["captured_at"] > self.last_snap_at
                        and 0 <= age <= self.settings.max_age
                    )
                    if valid:
                        detections = [
                            Detection(
                                d["cls"],
                                d["score"],
                                tuple(d["box"]),
                                tuple(d["head"]),
                                d["head_score"],
                            )
                            for d in packet["detections"]
                        ]
                        with self.lock:
                            start = time.perf_counter()
                            if not self.stop.is_set() and self.armed:
                                detections = self.history.rebase(
                                    detections, packet["captured_at"], packet["size"]
                                )
                                action = self.controller.step(
                                    detections,
                                    *packet["size"],
                                    start,
                                    packet["captured_at"],
                                    enabled=True,
                                    foreground=True,
                                    fire=self.fire,
                                )
                                decisions.write(
                                    json.dumps(
                                        dict(
                                            time=time.time(),
                                            frame=seen,
                                            age_ms=age * 1000,
                                            heads=[
                                                dict(
                                                    cls=d.cls,
                                                    box=d.box,
                                                    head=d.head,
                                                    confidence=d.head_score,
                                                )
                                                for d in detections
                                            ],
                                            selected=asdict(action["target"])
                                            if action["target"]
                                            else None,
                                            **{k: v for k, v in action.items() if k != "target"},
                                        )
                                    )
                                    + "\n"
                                )
                                self.motion.begin(
                                    action,
                                    packet["size"],
                                    packet["rect"],
                                    packet["captured_at"],
                                    start,
                                    self.settings.max_age,
                                )
                                self.state["status"] = (
                                    "Цель потеряна: удержание сохранено. F7 — отпустить."
                                    if action["reason"] == "target_temporarily_lost"
                                    else action["reason"]
                                )
                                last_accepted = start
                    elif self.armed:
                        self.motion.cancel()
                        self.state["status"] = "Кадр устарел или камера ещё не обновилась"
                self.send_pending(log, seen)
                self.mouse.tick()
                while self.mouse.timings:
                    input_timings.write(json.dumps(self.mouse.timings.popleft()) + "\n")
                if self.armed and now - last_accepted > 0.2:
                    self.mouse.release()
                if now - last_flush > 0.25:
                    log.flush()
                    decisions.flush()
                    input_timings.flush()
                    last_flush = now

                if self.armed or self.motion.pending:
                    time.sleep(0.001)
                else:
                    self.stop.wait(0.01)
        except Exception:
            self.state["input_error"] = traceback.format_exc()
            self.stop.set()
            (self.output / "input-error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finally:
            self.disarm()
            self.ready = False
            self.worker_stop.set()
            log.close()
            decisions.close()
            while self.mouse.timings:
                input_timings.write(json.dumps(self.mouse.timings.popleft()) + "\n")
            input_timings.close()

    def close(self):
        self.stop.set()
        self.worker_stop.set()
        self.disarm()
        self.thread.join(timeout=2)
        self.worker.join(timeout=3)
        self.metrics_thread.join(timeout=2)
        if self.worker.is_alive():
            self.worker.terminate()
            self.worker.join(timeout=2)
        self.ready = False
        (self.output / "summary.json").write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for q in (self.packets, self.messages):
            q.close()
            q.cancel_join_thread()


class App:
    def __init__(self, args):
        self.args = args
        self.engine = None
        self.root = tk.Tk()
        self.root.title(TITLE)
        self.root.geometry("820x820")
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame, text="MVP v3.7 — захват цели по всему кадру", font=("Arial", 14, "bold")
        ).pack(anchor="w")
        ttk.Label(frame, text="Для контролируемого локального матча с ботами.").pack(
            anchor="w", pady=7
        )
        self.mode = tk.StringVar(value=args.mode)
        self.sens = tk.StringVar(value=str(args.sensitivity))
        self.size = tk.StringVar(value=str(args.imgsz))
        self.hz = tk.StringVar(value=str(args.hz))
        self.enemy = tk.StringVar(value="T")
        self.pipeline = tk.StringVar(value=args.pipeline)
        self.backend = tk.StringVar(value=args.model_backend)
        self.interval = tk.StringVar(value="0.40")
        self.smooth = tk.BooleanVar(value=True)
        self.blocking_wait = tk.BooleanVar(value=args.blocking_gpu_wait)
        self.capture_backend = tk.StringVar(value=args.capture_backend)
        self.death_source = tk.StringVar(value=args.death_source)
        form = ttk.Frame(frame)
        form.pack(fill="x", pady=10)
        fields = [
            ("Режим", self.mode, ["local-bots"]),
            ("Чувствительность CS2", self.sens, None),
            ("Размер входа (960 легче, 1280 детальнее)", self.size, ["960", "1280"]),
            ("Частота модели", self.hz, ["20", "30", "45", "60"]),
            ("Противник (для игрока CT → T)", self.enemy, ["T", "CT"]),
            ("Проверка нагрузки", self.pipeline, ["full", "capture", "inference"]),
            ("Движок модели", self.backend, ["auto", "native", "tensorrt", "pytorch"]),
            ("Захват (DXGI легче; WGC альтернативный)", self.capture_backend, ["dxgi", "wgc"]),
            ("Подтверждение убийства", self.death_source, ["visual", "manual"]),
        ]
        for row, (label, var, values) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w", pady=5)
            widget = (
                ttk.Combobox(form, textvariable=var, values=values, state="readonly", width=25)
                if values
                else ttk.Entry(form, textvariable=var, width=28)
            )
            widget.grid(row=row, column=1, sticky="w", padx=16)
        ttk.Label(form, text="Интервал выстрелов, секунды").grid(
            row=len(fields), column=0, sticky="w", pady=5
        )
        ttk.Entry(form, textvariable=self.interval, width=28).grid(
            row=len(fields), column=1, sticky="w", padx=16
        )
        ttk.Checkbutton(
            frame, text="Плавное сопровождение после быстрого захвата", variable=self.smooth
        ).pack(anchor="w", pady=5)
        ttk.Checkbutton(
            frame,
            text="Снижать расход CPU при ожидании GPU (проверено с лимитом 120 FPS)",
            variable=self.blocking_wait,
        ).pack(anchor="w")
        ttk.Label(
            frame,
            text="Живая цель удерживается. visual — наш ник слева в новой плашке убийства (игрок CT).\nF7 — вручную отпустить цель.",
        ).pack(anchor="w")
        ttk.Label(
            frame, text="Дождись «Готово» перед игрой. F8 — пауза нейросети; F6 — выключить всё."
        ).pack(anchor="w", pady=6)
        buttons = ttk.Frame(frame)
        buttons.pack(fill="x", pady=12)
        ttk.Button(buttons, text="Запустить мониторинг", command=self.start).pack(side="left")
        ttk.Button(buttons, text="Стоп", command=self.stop).pack(side="left", padx=10)
        ttk.Label(
            frame,
            text="F8 — включить/выключить коррекцию и стрельбу\nF9 — только коррекция / коррекция + выстрел; F7 — отпустить цель\nF6 — остановить; Alt+Tab отключает управление",
        ).pack(anchor="w")
        self.status = tk.StringVar(value="Готов к запуску")
        ttk.Label(frame, textvariable=self.status, wraplength=650).pack(anchor="w", pady=12)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        if args.autostart:
            self.root.after(300, self.start)
        if args.duration:
            self.root.after(int(args.duration * 1000), self.close)
        self.tick()
        self.root.mainloop()

    def start(self):
        self.stop()
        try:
            sens = float(self.sens.get().replace(",", "."))
            if not 0.01 <= sens <= 20:
                raise ValueError("Sensitivity: 0.01–20")
            interval = float(self.interval.get().replace(",", "."))
            if not 0.12 <= interval <= 3:
                raise ValueError("Интервал выстрелов: 0.12–3 секунды")
            self.engine = Engine(
                self.mode.get(),
                Settings(
                    enemy=0 if self.enemy.get() == "T" else 1,
                    sensitivity=sens,
                    shot_interval=interval,
                ),
                int(self.size.get()),
                int(self.hz.get()),
                self.pipeline.get(),
                self.backend.get(),
                self.smooth.get(),
                self.capture_backend.get(),
                self.blocking_wait.get(),
                self.death_source.get(),
            )
        except Exception as error:
            self.status.set(str(error))

    def stop(self):
        if self.engine:
            self.engine.close()
            self.engine = None
        self.status.set("Остановлено")

    def close(self):
        self.stop()
        self.root.destroy()

    def tick(self):
        if self.engine:
            if (self.engine.output / "close.request").exists():
                self.close()
                return
            if (self.engine.output / "stop.request").exists():
                self.stop()
            if self.engine:
                s = self.engine.snapshot()
                prefix = "ACTIVE" if s["armed"] else "OBSERVE"
                smooth = "ВКЛ" if s["smooth_tracking"] else "ВЫКЛ"
                self.status.set(
                    f"{prefix} | {s['input_error'] or s['status']}\n{s['fps']:.1f} Hz | кадр→действие {s['latency_ms']:.1f} ms | модель {s['predict_ms']:.1f} ms | кликов {s['shots']} | {s.get('model_backend', 'загрузка')}\nПлавность {smooth} | цикл ввода {s['input_loop_hz']:.0f} Hz | p95 {s['input_loop_p95_ms']:.1f} мс | захват создан {s.get('capture_creations', 0)} раз\nRAM {s.get('ram_percent', 0):.0f}% | свободно {s.get('available_mb', 0):.0f} МБ | модель RAM {s.get('worker_rss_mb', 0):.0f} МБ | подкачка {s.get('pages_input_per_sec', 0):.0f} стр/с"
                )
                if s.get("death_source") == "visual":
                    self.status.set(
                        self.status.get()
                        + f"\nПлашки: своих {s.get('visual_feed_rows', 0)} | разбор {s.get('visual_feed_ms', 0):.1f} мс | смен цели {s.get('confirmed_kill_releases', 0)}"
                    )
                (self.engine.output / "status.json").write_text(
                    json.dumps(s, ensure_ascii=False), encoding="utf-8"
                )
        self.root.after(250, self.tick)


if __name__ == "__main__":
    mp.freeze_support()
    KERNEL.CreateMutexW.argtypes = [C.c_void_p, W.BOOL, W.LPCWSTR]
    KERNEL.CreateMutexW.restype = W.HANDLE
    C.set_last_error(0)
    instance_mutex = KERNEL.CreateMutexW(None, False, "Local\\CS2MLMVPPanelV2")
    if not instance_mutex:
        raise C.WinError(C.get_last_error())
    if C.get_last_error() == 183:
        USER.FindWindowW.argtypes = [W.LPCWSTR, W.LPCWSTR]
        USER.FindWindowW.restype = W.HWND
        USER.ShowWindow.argtypes = [W.HWND, C.c_int]
        USER.SetForegroundWindow.argtypes = [W.HWND]
        hwnd = USER.FindWindowW(None, TITLE)
        if hwnd:
            USER.ShowWindow(hwnd, 9)
            USER.SetForegroundWindow(hwnd)
        raise SystemExit(0)
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["local-bots"], default="local-bots")
    p.add_argument("--sensitivity", type=float, default=0.97)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--hz", type=int, default=30)
    p.add_argument("--autostart", action="store_true")
    p.add_argument("--pipeline", choices=["full", "capture", "inference"], default="full")
    p.add_argument(
        "--model-backend", choices=["auto", "native", "tensorrt", "pytorch"], default="auto"
    )
    p.add_argument("--capture-backend", choices=["dxgi", "wgc"], default="dxgi")
    p.add_argument("--duration", type=float, default=0)
    p.add_argument("--blocking-gpu-wait", action="store_true")
    p.add_argument("--death-source", choices=["visual", "manual"], default="manual")
    App(p.parse_args())
