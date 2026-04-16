import json
import os
import queue
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import torch
from PIL import Image, ImageTk

from config.config import MobileTrainConfig, TrainConfig
from src.datasets.factory import apply_dataset_profile, normalize_dataset_name

try:
    from scripts.video_stream_infer import (
        SENTINEL,
        blend_overlay,
        build_model,
        build_palette,
        clear_pngs,
        colorize_pred,
        extract_frames,
        load_frame_tensor,
        probe_video_info,
        producer,
        rebuild_video,
        save_image,
    )
except ImportError:
    from video_stream_infer import (
        SENTINEL,
        blend_overlay,
        build_model,
        build_palette,
        clear_pngs,
        colorize_pred,
        extract_frames,
        load_frame_tensor,
        probe_video_info,
        producer,
        rebuild_video,
        save_image,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "video_gui"
DATASET_OPTIONS = ("cityscapes", "camvid", "kitti_semantic")
MODEL_TYPE_OPTIONS = ("teacher", "mobile")
RESAMPLING_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
VIDEO_FILE_TYPES = [
    ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
    ("All files", "*.*"),
]
CHECKPOINT_FILE_TYPES = [
    ("PyTorch checkpoints", "*.pth *.pt"),
    ("All files", "*.*"),
]


@dataclass(frozen=True)
class ModelEntry:
    ckpt_path: Path
    dataset_name: str
    model_type: str

    @property
    def label(self) -> str:
        return f"{self.ckpt_path.name} | {self.dataset_name} | {self.model_type}"


def build_gui_output_layout(output_root: Path, video_path: Path, model_entry: ModelEntry) -> dict[str, Path]:
    run_name = f"{video_path.stem}_{model_entry.ckpt_path.stem}_{model_entry.dataset_name}_{model_entry.model_type}"
    run_dir = output_root / run_name
    return {
        "run_dir": run_dir,
        "frames_dir": run_dir / "frames",
        "overlay_dir": run_dir / "overlay_frames",
        "pred_color_dir": run_dir / "pred_color",
        "pred_id_dir": run_dir / "pred_id",
        "final_video": run_dir / f"{video_path.stem}_overlay.mp4",
        "stats_json": run_dir / "stats.json",
    }


def queue_latest(target_queue: queue.Queue, item) -> None:
    while True:
        try:
            target_queue.put_nowait(item)
            return
        except queue.Full:
            try:
                target_queue.get_nowait()
            except queue.Empty:
                return


def log_message(log_queue: queue.Queue, message: str) -> None:
    log_queue.put(message)


@torch.inference_mode()
def gui_consumer(
    model: torch.nn.Module,
    task_queue: queue.Queue,
    device: torch.device,
    id2color: list[tuple[int, int, int]],
    overlay_dir: Path,
    pred_color_dir: Path | None,
    pred_id_dir: Path | None,
    overlay_alpha: float,
    use_amp: bool,
    total_frames: int,
    stats: dict[str, float | int],
    preview_queue: queue.Queue,
    status_queue: queue.Queue,
    log_queue: queue.Queue,
    stop_event: threading.Event,
    error_queue: queue.Queue,
) -> None:
    processed = 0
    infer_time_sum = 0.0

    try:
        while True:
            try:
                item = task_queue.get(timeout=0.1)
            except queue.Empty:
                if stop_event.is_set():
                    break
                continue

            if item is SENTINEL:
                task_queue.task_done()
                break

            if stop_event.is_set():
                task_queue.task_done()
                continue

            frame_rgb, img_t = load_frame_tensor(item.frame_path)
            img_t = img_t.to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(img_t)
            pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype("uint8")
            if device.type == "cuda":
                torch.cuda.synchronize()
            infer_time_sum += time.perf_counter() - t0

            pred_color = colorize_pred(pred, id2color)
            overlay = blend_overlay(frame_rgb, pred_color, alpha=overlay_alpha)

            frame_name = item.frame_path.name
            save_image(overlay_dir / frame_name, overlay, mode="RGB")
            if pred_color_dir is not None:
                save_image(pred_color_dir / frame_name, pred_color, mode="RGB")
            if pred_id_dir is not None:
                save_image(pred_id_dir / frame_name, pred, mode="L")

            processed += 1
            stats["processed_frames"] = processed
            task_queue.task_done()

            queue_latest(preview_queue, {"index": item.index, "image": overlay})
            if processed == 1 or processed % 10 == 0 or processed == total_frames:
                avg_ms = (infer_time_sum / max(processed, 1)) * 1000.0
                infer_fps = processed / max(infer_time_sum, 1e-12)
                queue_latest(
                    status_queue,
                    {
                        "type": "progress",
                        "processed": processed,
                        "total_frames": total_frames,
                        "infer_avg_ms": avg_ms,
                        "infer_fps": infer_fps,
                    },
                )
                log_message(
                    log_queue,
                    f"[INFO] processed={processed}/{total_frames} infer_avg_ms={avg_ms:.2f} infer_fps={infer_fps:.2f}",
                )
    except Exception as exc:
        stop_event.set()
        error_queue.put(exc)
    finally:
        stats["infer_time_sec"] = infer_time_sum


class StreamInferenceWorker(threading.Thread):
    def __init__(
        self,
        *,
        video_path: Path,
        model_entry: ModelEntry,
        output_root: Path,
        overlay_alpha: float,
        queue_size: int,
        producer_delay_ms: float,
        fps_override: float,
        save_pred_color: bool,
        save_pred_id: bool,
        log_queue: queue.Queue,
        preview_queue: queue.Queue,
        status_queue: queue.Queue,
    ) -> None:
        super().__init__(daemon=True)
        self.video_path = video_path
        self.model_entry = model_entry
        self.output_root = output_root
        self.overlay_alpha = overlay_alpha
        self.queue_size = queue_size
        self.producer_delay_ms = producer_delay_ms
        self.fps_override = fps_override
        self.save_pred_color = save_pred_color
        self.save_pred_id = save_pred_id
        self.log_queue = log_queue
        self.preview_queue = preview_queue
        self.status_queue = status_queue
        self.stop_event = threading.Event()

    def stop(self) -> None:
        self.stop_event.set()

    def _log(self, message: str) -> None:
        log_message(self.log_queue, message)

    def _emit_status(self, payload: dict) -> None:
        queue_latest(self.status_queue, payload)

    def run(self) -> None:
        try:
            self._run_impl()
        except Exception as exc:
            self._log(f"[ERROR] {exc}")
            self._emit_status({"type": "error", "message": str(exc)})

    def _run_impl(self) -> None:
        if not self.video_path.exists():
            raise FileNotFoundError(f"Input video not found: {self.video_path}")
        if not self.model_entry.ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.model_entry.ckpt_path}")

        cfg = TrainConfig() if self.model_entry.model_type == "teacher" else MobileTrainConfig()
        cfg.dataset_name = normalize_dataset_name(self.model_entry.dataset_name)
        apply_dataset_profile(cfg)
        id2color = build_palette(cfg)
        cfg.num_classes = len(id2color)

        layout = build_gui_output_layout(self.output_root, self.video_path, self.model_entry)
        for key in ("run_dir", "frames_dir", "overlay_dir"):
            layout[key].mkdir(parents=True, exist_ok=True)
        clear_pngs(layout["overlay_dir"])
        if self.save_pred_color:
            layout["pred_color_dir"].mkdir(parents=True, exist_ok=True)
            clear_pngs(layout["pred_color_dir"])
        if self.save_pred_id:
            layout["pred_id_dir"].mkdir(parents=True, exist_ok=True)
            clear_pngs(layout["pred_id_dir"])

        video_info = probe_video_info(self.video_path)
        fps = float(self.fps_override) if self.fps_override > 0 else float(video_info["fps"])
        if fps <= 0:
            raise ValueError("Unable to determine FPS. Set a manual FPS override.")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = bool(cfg.use_amp and device.type == "cuda")

        self._log(f"[INFO] video={self.video_path}")
        self._log(f"[INFO] ckpt={self.model_entry.ckpt_path}")
        self._log(f"[INFO] dataset_name={cfg.dataset_name}")
        self._log(f"[INFO] model_type={self.model_entry.model_type}")
        self._log(f"[INFO] device={device} amp={use_amp}")
        self._log(f"[INFO] fps={fps:.6f} size={video_info['width']}x{video_info['height']}")
        self._log(f"[INFO] output_dir={layout['run_dir']}")
        self._emit_status({"type": "stage", "message": "Loading model..."})

        model = build_model(cfg=cfg, ckpt_path=self.model_entry.ckpt_path, device=device, model_type=self.model_entry.model_type)

        self._log("[INFO] extracting frames...")
        self._emit_status({"type": "stage", "message": "Extracting frames..."})
        t_extract_start = time.perf_counter()
        frame_paths = extract_frames(self.video_path, layout["frames_dir"])
        extract_time = time.perf_counter() - t_extract_start
        total_frames = len(frame_paths)
        self._log(f"[INFO] extracted_frames={total_frames} extract_time={extract_time:.3f}s")
        self._emit_status(
            {
                "type": "progress",
                "processed": 0,
                "total_frames": total_frames,
                "infer_avg_ms": 0.0,
                "infer_fps": 0.0,
            }
        )

        task_queue: queue.Queue = queue.Queue(maxsize=max(1, int(self.queue_size)))
        error_queue: queue.Queue = queue.Queue()
        stats: dict[str, float | int] = {"processed_frames": 0, "infer_time_sec": 0.0}

        producer_thread = threading.Thread(
            target=producer,
            args=(frame_paths, task_queue, float(self.producer_delay_ms), self.stop_event, error_queue),
            daemon=True,
        )
        consumer_thread = threading.Thread(
            target=gui_consumer,
            args=(
                model,
                task_queue,
                device,
                id2color,
                layout["overlay_dir"],
                layout["pred_color_dir"] if self.save_pred_color else None,
                layout["pred_id_dir"] if self.save_pred_id else None,
                float(self.overlay_alpha),
                use_amp,
                total_frames,
                stats,
                self.preview_queue,
                self.status_queue,
                self.log_queue,
                self.stop_event,
                error_queue,
            ),
            daemon=True,
        )

        self._log("[INFO] starting producer-consumer inference...")
        self._emit_status({"type": "stage", "message": "Running inference..."})
        t_stream_start = time.perf_counter()
        producer_thread.start()
        consumer_thread.start()

        while producer_thread.is_alive() or consumer_thread.is_alive():
            producer_thread.join(timeout=0.1)
            consumer_thread.join(timeout=0.1)
            if not error_queue.empty():
                raise RuntimeError(f"Video stream inference failed: {error_queue.get()}")

        stream_time = time.perf_counter() - t_stream_start
        if not error_queue.empty():
            raise RuntimeError(f"Video stream inference failed: {error_queue.get()}")

        processed = int(stats["processed_frames"])
        infer_time_sec = float(stats["infer_time_sec"])

        if self.stop_event.is_set():
            self._log(f"[INFO] stopped processed_frames={processed}")
            self._emit_status(
                {
                    "type": "stopped",
                    "processed": processed,
                    "total_frames": total_frames,
                }
            )
            return

        self._log("[INFO] rebuilding output video...")
        self._emit_status({"type": "stage", "message": "Rebuilding video..."})
        rebuild_video(self.video_path, layout["overlay_dir"], layout["final_video"], fps=fps)

        infer_fps = processed / max(infer_time_sec, 1e-12)
        end_to_end_fps = processed / max(stream_time, 1e-12)
        stats_payload = {
            "video": str(self.video_path),
            "checkpoint": str(self.model_entry.ckpt_path),
            "dataset_name": cfg.dataset_name,
            "model_type": self.model_entry.model_type,
            "num_frames": processed,
            "video_fps": fps,
            "extract_time_sec": extract_time,
            "stream_time_sec": stream_time,
            "infer_time_sec": infer_time_sec,
            "infer_avg_ms": (infer_time_sec / max(processed, 1)) * 1000.0,
            "infer_fps": infer_fps,
            "end_to_end_stream_fps": end_to_end_fps,
            "overlay_alpha": float(self.overlay_alpha),
            "queue_size": int(self.queue_size),
            "producer_delay_ms": float(self.producer_delay_ms),
            "final_video": str(layout["final_video"]),
        }
        layout["stats_json"].write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False), encoding="utf-8")

        self._log("[INFO] finished")
        self._log(f"[METRIC] processed_frames={processed}")
        self._log(f"[METRIC] infer_avg_ms={stats_payload['infer_avg_ms']:.3f}")
        self._log(f"[METRIC] infer_fps={infer_fps:.2f}")
        self._log(f"[METRIC] end_to_end_stream_fps={end_to_end_fps:.2f}")
        self._log(f"[INFO] final_video={layout['final_video']}")
        self._log(f"[INFO] stats_json={layout['stats_json']}")
        self._emit_status(
            {
                "type": "done",
                "processed": processed,
                "total_frames": total_frames,
                "infer_avg_ms": stats_payload["infer_avg_ms"],
                "infer_fps": infer_fps,
                "final_video": str(layout["final_video"]),
                "run_dir": str(layout["run_dir"]),
            }
        )


class VideoInferGUI(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Video Stream Inference")
        self.geometry("1680x900")
        self.minsize(1360, 760)

        self.videos: list[Path] = []
        self.models: list[ModelEntry] = []
        self.worker: StreamInferenceWorker | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.last_run_dir: Path | None = None
        self.preview_tabs: ttk.Notebook | None = None
        self.live_preview_tab: ttk.Frame | None = None
        self.sketch_canvas: tk.Canvas | None = None
        self.sketch_last_point: tuple[int, int] | None = None

        self.log_queue: queue.Queue = queue.Queue()
        self.preview_queue: queue.Queue = queue.Queue(maxsize=1)
        self.status_queue: queue.Queue = queue.Queue(maxsize=8)

        self.dataset_var = tk.StringVar(value="cityscapes")
        self.model_type_var = tk.StringVar(value="teacher")
        self.selected_video_var = tk.StringVar(value="")
        self.selected_model_var = tk.StringVar(value="")
        self.output_root_var = tk.StringVar(value=str(DEFAULT_OUTPUT_ROOT))
        self.overlay_alpha_var = tk.StringVar(value="0.55")
        self.queue_size_var = tk.StringVar(value="32")
        self.producer_delay_var = tk.StringVar(value="0")
        self.fps_override_var = tk.StringVar(value="0")
        self.save_pred_color_var = tk.BooleanVar(value=False)
        self.save_pred_id_var = tk.BooleanVar(value=False)
        self.status_text_var = tk.StringVar(value="Idle")

        self._build_ui()
        self._refresh_video_selector()
        self._refresh_model_selector()
        self._set_controls_for_idle()
        self.after(100, self._poll_queues)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self, padding=10)
        toolbar.grid(row=0, column=0, sticky="ew")
        toolbar.columnconfigure(3, weight=2)
        toolbar.columnconfigure(8, weight=2)

        self.start_button = ttk.Button(toolbar, text="Start", command=self._start_run)
        self.start_button.grid(row=0, column=0, padx=(0, 8))
        self.stop_button = ttk.Button(toolbar, text="Stop", command=self._stop_run)
        self.stop_button.grid(row=0, column=1, padx=(0, 12))

        ttk.Label(toolbar, text="Video").grid(row=0, column=2, sticky="w")
        self.video_combobox = ttk.Combobox(
            toolbar,
            textvariable=self.selected_video_var,
            state="readonly",
            values=(),
            width=48,
        )
        self.video_combobox.grid(row=0, column=3, sticky="ew", padx=(4, 8))
        self.add_video_button = ttk.Button(toolbar, text="Add Video", command=self._add_videos)
        self.add_video_button.grid(row=0, column=4, padx=(0, 4))
        self.remove_video_button = ttk.Button(toolbar, text="Remove", command=self._remove_selected_video)
        self.remove_video_button.grid(row=0, column=5, padx=4)
        self.clear_videos_button = ttk.Button(toolbar, text="Clear", command=self._clear_videos)
        self.clear_videos_button.grid(row=0, column=6, padx=(4, 12))

        ttk.Label(toolbar, text="Model").grid(row=0, column=7, sticky="w")
        self.model_combobox = ttk.Combobox(
            toolbar,
            textvariable=self.selected_model_var,
            state="readonly",
            values=(),
            width=44,
        )
        self.model_combobox.grid(row=0, column=8, sticky="ew", padx=(4, 8))
        self.add_model_button = ttk.Button(toolbar, text="Add Model", command=self._add_models)
        self.add_model_button.grid(row=0, column=9, padx=(0, 4))
        self.remove_model_button = ttk.Button(toolbar, text="Remove", command=self._remove_selected_model)
        self.remove_model_button.grid(row=0, column=10, padx=4)
        self.clear_models_button = ttk.Button(toolbar, text="Clear", command=self._clear_models)
        self.clear_models_button.grid(row=0, column=11, padx=(4, 0))

        ttk.Label(toolbar, text="Dataset").grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.dataset_combobox = ttk.Combobox(
            toolbar,
            textvariable=self.dataset_var,
            values=DATASET_OPTIONS,
            state="readonly",
            width=16,
        )
        self.dataset_combobox.grid(row=1, column=1, sticky="ew", padx=(4, 12), pady=(8, 0))

        ttk.Label(toolbar, text="Model Type").grid(row=1, column=2, sticky="w", pady=(8, 0))
        self.model_type_combobox = ttk.Combobox(
            toolbar,
            textvariable=self.model_type_var,
            values=MODEL_TYPE_OPTIONS,
            state="readonly",
            width=12,
        )
        self.model_type_combobox.grid(row=1, column=3, sticky="w", padx=(4, 12), pady=(8, 0))

        ttk.Label(toolbar, text="Overlay alpha").grid(row=1, column=4, sticky="w", pady=(8, 0))
        ttk.Entry(toolbar, textvariable=self.overlay_alpha_var, width=6).grid(row=1, column=5, padx=(4, 12), pady=(8, 0))

        ttk.Label(toolbar, text="Queue").grid(row=1, column=6, sticky="w", pady=(8, 0))
        ttk.Entry(toolbar, textvariable=self.queue_size_var, width=6).grid(row=1, column=7, padx=(4, 12), pady=(8, 0))

        ttk.Label(toolbar, text="Delay ms").grid(row=1, column=8, sticky="w", pady=(8, 0))
        ttk.Entry(toolbar, textvariable=self.producer_delay_var, width=8).grid(row=1, column=9, padx=(4, 12), pady=(8, 0))

        ttk.Label(toolbar, text="FPS override").grid(row=1, column=10, sticky="w", pady=(8, 0))
        ttk.Entry(toolbar, textvariable=self.fps_override_var, width=8).grid(row=1, column=11, padx=(4, 12), pady=(8, 0))

        ttk.Checkbutton(toolbar, text="Save color mask", variable=self.save_pred_color_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(toolbar, text="Save id mask", variable=self.save_pred_id_var).grid(row=2, column=2, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(toolbar, text="Output root").grid(row=2, column=4, sticky="w", pady=(8, 0))
        ttk.Entry(toolbar, textvariable=self.output_root_var).grid(row=2, column=5, columnspan=6, sticky="ew", pady=(8, 0))
        ttk.Button(toolbar, text="Browse", command=self._browse_output_root).grid(row=2, column=11, sticky="e", pady=(8, 0))

        content = ttk.Frame(self, padding=(10, 0, 10, 10))
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=5)
        content.columnconfigure(1, weight=2)
        content.rowconfigure(0, weight=1)

        self._build_preview_panel(content)
        self._build_log_panel(content)

    def _build_video_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Videos", padding=10)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.video_listbox = tk.Listbox(frame, exportselection=False)
        self.video_listbox.grid(row=0, column=0, sticky="nsew")
        video_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.video_listbox.yview)
        video_scroll.grid(row=0, column=1, sticky="ns")
        self.video_listbox.configure(yscrollcommand=video_scroll.set)

        button_row = ttk.Frame(frame)
        button_row.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for column in range(3):
            button_row.columnconfigure(column, weight=1)
        ttk.Button(button_row, text="Add video", command=self._add_videos).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(button_row, text="Remove", command=self._remove_selected_video).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(button_row, text="Clear", command=self._clear_videos).grid(row=0, column=2, sticky="ew", padx=(4, 0))

    def _build_model_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Models", padding=10)
        frame.grid(row=0, column=1, sticky="nsew", padx=8)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        options = ttk.Frame(frame)
        options.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        options.columnconfigure(1, weight=1)
        options.columnconfigure(3, weight=1)

        ttk.Label(options, text="Dataset").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            options,
            textvariable=self.dataset_var,
            values=DATASET_OPTIONS,
            state="readonly",
            width=16,
        ).grid(row=0, column=1, sticky="ew", padx=(6, 12))

        ttk.Label(options, text="Model").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            options,
            textvariable=self.model_type_var,
            values=MODEL_TYPE_OPTIONS,
            state="readonly",
            width=10,
        ).grid(row=0, column=3, sticky="ew", padx=(6, 0))

        self.model_listbox = tk.Listbox(frame, exportselection=False)
        self.model_listbox.grid(row=1, column=0, sticky="nsew")
        model_scroll = ttk.Scrollbar(frame, orient="vertical", command=self.model_listbox.yview)
        model_scroll.grid(row=1, column=1, sticky="ns")
        self.model_listbox.configure(yscrollcommand=model_scroll.set)

        button_row = ttk.Frame(frame)
        button_row.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        for column in range(3):
            button_row.columnconfigure(column, weight=1)
        ttk.Button(button_row, text="Add model", command=self._add_models).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(button_row, text="Remove", command=self._remove_selected_model).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(button_row, text="Clear", command=self._clear_models).grid(row=0, column=2, sticky="ew", padx=(4, 0))

    def _build_preview_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Live Preview", padding=10)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.preview_tabs = ttk.Notebook(frame)
        self.preview_tabs.grid(row=0, column=0, sticky="nsew")

        preview_tab = ttk.Frame(self.preview_tabs)
        preview_tab.columnconfigure(0, weight=1)
        preview_tab.rowconfigure(0, weight=1)
        self.preview_tabs.add(preview_tab, text="Live Preview")
        self.live_preview_tab = preview_tab

        sketch_tab = ttk.Frame(self.preview_tabs)
        sketch_tab.columnconfigure(0, weight=1)
        sketch_tab.rowconfigure(0, weight=1)
        self.preview_tabs.add(sketch_tab, text="Sketch Pad")

        self.preview_label = tk.Label(
            preview_tab,
            text="No output yet",
            anchor="center",
            background="#111111",
            foreground="#eeeeee",
        )
        self.preview_label.grid(row=0, column=0, sticky="nsew")

        self.sketch_canvas = tk.Canvas(
            sketch_tab,
            background="#fbfcfe",
            highlightthickness=0,
            cursor="pencil",
        )
        self.sketch_canvas.grid(row=0, column=0, sticky="nsew")
        self.sketch_canvas.bind("<ButtonPress-1>", self._on_sketch_press)
        self.sketch_canvas.bind("<B1-Motion>", self._on_sketch_drag)
        self.sketch_canvas.bind("<ButtonRelease-1>", self._on_sketch_release)

        preview_controls = ttk.Frame(frame)
        preview_controls.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        preview_controls.columnconfigure(0, weight=1)

        ttk.Label(preview_controls, textvariable=self.status_text_var).grid(row=0, column=0, sticky="w")
        ttk.Button(preview_controls, text="Clear sketch", command=self._clear_sketch).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(preview_controls, text="Draw demo", command=self._draw_demo_sketch).grid(row=0, column=2, padx=(8, 0))
        ttk.Button(preview_controls, text="Open run dir", command=self._open_last_run_dir).grid(row=0, column=3, padx=(8, 0))

        self._draw_demo_sketch()
        self.preview_tabs.select(sketch_tab)

    def _build_log_panel(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Logs", padding=10)
        frame.grid(row=0, column=1, sticky="nsew", padx=(8, 0))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        self.log_text = ScrolledText(frame, wrap="word", state="disabled", font=("Consolas", 10))
        self.log_text.grid(row=0, column=0, sticky="nsew")

    def _browse_output_root(self) -> None:
        selected = filedialog.askdirectory(title="Select output root", initialdir=self.output_root_var.get())
        if selected:
            self.output_root_var.set(selected)

    def _refresh_video_selector(self) -> None:
        values = [str(path) for path in self.videos]
        self.video_combobox.configure(values=values)
        current = self.selected_video_var.get()
        if not values:
            self.selected_video_var.set("")
            return
        if current not in values:
            self.selected_video_var.set(values[0])

    def _refresh_model_selector(self) -> None:
        values = [entry.label for entry in self.models]
        self.model_combobox.configure(values=values)
        current = self.selected_model_var.get()
        if not values:
            self.selected_model_var.set("")
            return
        if current not in values:
            self.selected_model_var.set(values[0])

    def _add_videos(self) -> None:
        paths = filedialog.askopenfilenames(title="Select videos", filetypes=VIDEO_FILE_TYPES)
        first_added = None
        for raw_path in paths:
            path = Path(raw_path)
            if path in self.videos:
                continue
            self.videos.append(path)
            if first_added is None:
                first_added = path
        self._refresh_video_selector()
        if first_added is not None:
            self.selected_video_var.set(str(first_added))

    def _remove_selected_video(self) -> None:
        selected = self._selected_video()
        if selected is None:
            return
        self.videos = [path for path in self.videos if path != selected]
        self._refresh_video_selector()

    def _clear_videos(self) -> None:
        self.videos.clear()
        self._refresh_video_selector()

    def _add_models(self) -> None:
        paths = filedialog.askopenfilenames(title="Select checkpoints", filetypes=CHECKPOINT_FILE_TYPES)
        dataset_name = normalize_dataset_name(self.dataset_var.get())
        model_type = self.model_type_var.get().strip().lower()
        if model_type not in MODEL_TYPE_OPTIONS:
            messagebox.showerror("Invalid model type", "Model type must be teacher or mobile.")
            return

        first_added = None
        for raw_path in paths:
            entry = ModelEntry(ckpt_path=Path(raw_path), dataset_name=dataset_name, model_type=model_type)
            if entry in self.models:
                continue
            self.models.append(entry)
            if first_added is None:
                first_added = entry
        self._refresh_model_selector()
        if first_added is not None:
            self.selected_model_var.set(first_added.label)

    def _remove_selected_model(self) -> None:
        selected = self._selected_model()
        if selected is None:
            return
        self.models = [entry for entry in self.models if entry != selected]
        self._refresh_model_selector()

    def _clear_models(self) -> None:
        self.models.clear()
        self._refresh_model_selector()

    def _selected_video(self) -> Path | None:
        selected = self.selected_video_var.get().strip()
        if not selected:
            return None
        for path in self.videos:
            if str(path) == selected:
                return path
        return None

    def _selected_model(self) -> ModelEntry | None:
        selected = self.selected_model_var.get().strip()
        if not selected:
            return None
        for entry in self.models:
            if entry.label == selected:
                return entry
        return None

    def _start_run(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            messagebox.showwarning("Busy", "A run is already in progress.")
            return

        video_path = self._selected_video()
        model_entry = self._selected_model()
        if video_path is None:
            messagebox.showerror("Missing video", "Select one video before starting.")
            return
        if model_entry is None:
            messagebox.showerror("Missing model", "Select one model before starting.")
            return

        try:
            overlay_alpha = float(self.overlay_alpha_var.get())
            queue_size = max(1, int(self.queue_size_var.get()))
            producer_delay_ms = float(self.producer_delay_var.get())
            fps_override = float(self.fps_override_var.get())
            output_root = Path(self.output_root_var.get()).expanduser()
        except ValueError as exc:
            messagebox.showerror("Invalid numeric input", str(exc))
            return

        self.last_run_dir = None
        self._clear_preview()
        self._append_log("[INFO] ------------------------------------------------------------")
        self._append_log(f"[INFO] starting video={video_path}")
        self._append_log(f"[INFO] model={model_entry.label}")

        self.worker = StreamInferenceWorker(
            video_path=video_path,
            model_entry=model_entry,
            output_root=output_root,
            overlay_alpha=overlay_alpha,
            queue_size=queue_size,
            producer_delay_ms=producer_delay_ms,
            fps_override=fps_override,
            save_pred_color=bool(self.save_pred_color_var.get()),
            save_pred_id=bool(self.save_pred_id_var.get()),
            log_queue=self.log_queue,
            preview_queue=self.preview_queue,
            status_queue=self.status_queue,
        )
        self.worker.start()
        self._set_controls_for_running()

    def _stop_run(self) -> None:
        if self.worker is None or not self.worker.is_alive():
            return
        self.worker.stop()
        self._append_log("[INFO] stop requested")
        self.status_text_var.set("Stopping...")

    def _open_last_run_dir(self) -> None:
        if self.last_run_dir is None or not self.last_run_dir.exists():
            messagebox.showinfo("No output", "No completed run directory is available yet.")
            return
        os.startfile(str(self.last_run_dir))

    def _clear_preview(self) -> None:
        self.preview_photo = None
        self.preview_label.configure(image="", text="No output yet")

    def _clear_sketch(self) -> None:
        if self.sketch_canvas is None:
            return
        self.sketch_canvas.delete("all")
        self.sketch_last_point = None
        width = max(self.sketch_canvas.winfo_width(), 640)
        height = max(self.sketch_canvas.winfo_height(), 480)
        self.sketch_canvas.create_text(
            width / 2,
            height / 2,
            text="Drag with the left mouse button to draw here",
            fill="#6b7280",
            font=("Segoe UI", 16),
            tags=("hint",),
        )

    def _draw_demo_sketch(self) -> None:
        if self.sketch_canvas is None:
            return

        canvas = self.sketch_canvas
        canvas.delete("all")
        self.sketch_last_point = None

        width = max(canvas.winfo_width(), 900)
        height = max(canvas.winfo_height(), 600)

        canvas.create_rectangle(0, 0, width, height, fill="#f8fbff", outline="")
        canvas.create_rectangle(0, height * 0.68, width, height, fill="#d9f99d", outline="")
        canvas.create_oval(
            width * 0.74,
            height * 0.08,
            width * 0.88,
            height * 0.23,
            fill="#fbbf24",
            outline="#f59e0b",
            width=3,
        )
        canvas.create_polygon(
            0,
            height * 0.70,
            width * 0.18,
            height * 0.35,
            width * 0.36,
            height * 0.70,
            fill="#93c5fd",
            outline="#60a5fa",
            width=3,
        )
        canvas.create_polygon(
            width * 0.22,
            height * 0.70,
            width * 0.46,
            height * 0.28,
            width * 0.70,
            height * 0.70,
            fill="#60a5fa",
            outline="#3b82f6",
            width=3,
        )
        canvas.create_polygon(
            width * 0.56,
            height * 0.70,
            width * 0.82,
            height * 0.40,
            width,
            height * 0.70,
            fill="#3b82f6",
            outline="#2563eb",
            width=3,
        )
        canvas.create_polygon(
            width * 0.42,
            height,
            width * 0.48,
            height * 0.68,
            width * 0.52,
            height * 0.68,
            width * 0.58,
            height,
            fill="#4b5563",
            outline="#374151",
            width=2,
        )
        canvas.create_line(
            width * 0.5,
            height * 0.68,
            width * 0.5,
            height * 0.90,
            fill="#f9fafb",
            width=6,
            dash=(10, 14),
        )
        canvas.create_rectangle(
            width * 0.16,
            height * 0.56,
            width * 0.30,
            height * 0.69,
            fill="#fef3c7",
            outline="#d97706",
            width=3,
        )
        canvas.create_polygon(
            width * 0.14,
            height * 0.56,
            width * 0.23,
            height * 0.46,
            width * 0.32,
            height * 0.56,
            fill="#ef4444",
            outline="#b91c1c",
            width=3,
        )
        canvas.create_rectangle(
            width * 0.215,
            height * 0.61,
            width * 0.245,
            height * 0.69,
            fill="#93c5fd",
            outline="#1d4ed8",
            width=2,
        )
        canvas.create_text(
            width * 0.08,
            height * 0.10,
            anchor="w",
            text="Sketch Pad",
            fill="#0f172a",
            font=("Segoe UI Semibold", 24),
        )
        canvas.create_text(
            width * 0.08,
            height * 0.16,
            anchor="w",
            text="You can keep this demo or draw over it.",
            fill="#475569",
            font=("Segoe UI", 14),
        )

    def _on_sketch_press(self, event: tk.Event) -> None:
        if self.sketch_canvas is None:
            return
        self.sketch_canvas.delete("hint")
        self.sketch_last_point = (event.x, event.y)

    def _on_sketch_drag(self, event: tk.Event) -> None:
        if self.sketch_canvas is None:
            return
        if self.sketch_last_point is None:
            self.sketch_last_point = (event.x, event.y)
            return

        x0, y0 = self.sketch_last_point
        self.sketch_canvas.create_line(
            x0,
            y0,
            event.x,
            event.y,
            fill="#111827",
            width=4,
            capstyle=tk.ROUND,
            smooth=True,
        )
        self.sketch_last_point = (event.x, event.y)

    def _on_sketch_release(self, event: tk.Event) -> None:
        self.sketch_last_point = None

    def _set_controls_for_running(self) -> None:
        self.start_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self.video_combobox.configure(state="disabled")
        self.model_combobox.configure(state="disabled")
        self.dataset_combobox.configure(state="disabled")
        self.model_type_combobox.configure(state="disabled")
        for button in (
            self.add_video_button,
            self.remove_video_button,
            self.clear_videos_button,
            self.add_model_button,
            self.remove_model_button,
            self.clear_models_button,
        ):
            button.state(["disabled"])
        self.status_text_var.set("Running...")

    def _set_controls_for_idle(self) -> None:
        self.start_button.state(["!disabled"])
        self.stop_button.state(["disabled"])
        self.video_combobox.configure(state="readonly")
        self.model_combobox.configure(state="readonly")
        self.dataset_combobox.configure(state="readonly")
        self.model_type_combobox.configure(state="readonly")
        for button in (
            self.add_video_button,
            self.remove_video_button,
            self.clear_videos_button,
            self.add_model_button,
            self.remove_model_button,
            self.clear_models_button,
        ):
            button.state(["!disabled"])
        if self.worker is None or not self.worker.is_alive():
            self.status_text_var.set("Idle")

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _poll_queues(self) -> None:
        self._drain_logs()
        self._drain_previews()
        self._drain_status()
        self.after(100, self._poll_queues)

    def _drain_logs(self) -> None:
        while True:
            try:
                message = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log(message)

    def _drain_previews(self) -> None:
        latest = None
        while True:
            try:
                latest = self.preview_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            return

        image = Image.fromarray(latest["image"])
        image.thumbnail((1240, 780), RESAMPLING_LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(image)
        self.preview_label.configure(image=self.preview_photo, text="")
        if self.preview_tabs is not None and self.live_preview_tab is not None:
            self.preview_tabs.select(self.live_preview_tab)

    def _drain_status(self) -> None:
        latest = None
        while True:
            try:
                latest = self.status_queue.get_nowait()
            except queue.Empty:
                break

        if latest is None:
            return

        status_type = latest.get("type")
        if status_type == "stage":
            self.status_text_var.set(latest.get("message", "Running..."))
            return

        if status_type == "progress":
            processed = int(latest.get("processed", 0))
            total_frames = int(latest.get("total_frames", 0))
            infer_avg_ms = float(latest.get("infer_avg_ms", 0.0))
            infer_fps = float(latest.get("infer_fps", 0.0))
            self.status_text_var.set(
                f"Processed {processed}/{total_frames} | {infer_avg_ms:.1f} ms | {infer_fps:.2f} FPS"
            )
            return

        if status_type == "done":
            self.last_run_dir = Path(latest["run_dir"])
            self.status_text_var.set(
                f"Done | {int(latest['processed'])}/{int(latest['total_frames'])} | "
                f"{float(latest['infer_avg_ms']):.1f} ms | {float(latest['infer_fps']):.2f} FPS"
            )
            self._set_controls_for_idle()
            self.worker = None
            return

        if status_type == "stopped":
            self.status_text_var.set(
                f"Stopped | {int(latest.get('processed', 0))}/{int(latest.get('total_frames', 0))}"
            )
            self._set_controls_for_idle()
            self.worker = None
            return

        if status_type == "error":
            self.status_text_var.set("Error")
            self._set_controls_for_idle()
            self.worker = None
            messagebox.showerror("Inference error", latest.get("message", "Unknown error"))

    def _on_close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            self.worker.stop()
        self.destroy()


def main() -> None:
    app = VideoInferGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
