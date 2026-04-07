import argparse
import json
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config.config import MobileTrainConfig, TrainConfig
from src.datasets.camvid_labels import CAMVID_IGNORE_LABEL_NAMES, CAMVID_LABELS
from src.datasets.cityscapes_labels import CITYSCAPES_19_ID2COLOR
from src.datasets.factory import apply_dataset_profile, normalize_dataset_name
from src.datasets.transforms import normalize_img, pil_to_tensor
from src.models.deeplabv3_plus import DeepLabV3Plus
from src.models.deeplabv3_plus_moblie import DeepLabV3PlusMobile
from src.utils.Id2Mask import load_class_dict_csv


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SENTINEL = object()


@dataclass(frozen=True)
class FrameTask:
    index: int
    frame_path: Path


def detect_teacher_segmentation_head(state: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("hybrid_neck.") for k in state.keys()):
        return "hybrid"
    if any(k.startswith("ocr_pre.") or k.startswith("ocr_head.") for k in state.keys()):
        return "ocr"
    return "aspp"


def detect_teacher_hybrid_variant(state: dict[str, torch.Tensor]) -> str:
    if any(k.startswith("hybrid_neck.mid_kernel_branch.") or k == "hybrid_neck.mid_scale_logit" for k in state.keys()):
        return "large_v3"
    return "large"


def detect_decoder_upsample_mode(state: dict[str, torch.Tensor]) -> str:
    if any(
        k.startswith("decoder.aspp_upsample.pre.") or k.startswith("decoder.aspp_upsample.post.")
        for k in state.keys()
    ):
        return "learnable"
    return "bilinear"


def build_model(cfg: TrainConfig, ckpt_path: Path, device: torch.device, model_type: str) -> torch.nn.Module:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    if not isinstance(state, dict):
        raise TypeError(f"Invalid checkpoint format: expected dict-like state_dict, got {type(state)}")

    decoder_upsample_mode = detect_decoder_upsample_mode(state)

    if model_type == "teacher":
        teacher_head = detect_teacher_segmentation_head(state)
        teacher_hybrid_variant = detect_teacher_hybrid_variant(state) if teacher_head == "hybrid" else "large"
        model = DeepLabV3Plus(
            num_classes=cfg.num_classes,
            backbone_pretrained=False,
            backbone_name=cfg.backbone_name,
            output_stride=cfg.output_stride,
            segmentation_head=teacher_head,
            aspp_dropout=cfg.aspp_dropout,
            hybrid_variant=teacher_hybrid_variant,
            hybrid_use_strip=cfg.hybrid_use_strip,
            hybrid_strip_kernel=cfg.hybrid_strip_kernel,
            hybrid_mid_kernel=cfg.hybrid_mid_kernel,
            hybrid_large_kernel=cfg.hybrid_large_kernel,
            hybrid_gate_reduction=cfg.hybrid_gate_reduction,
            hybrid_residual_channels=cfg.hybrid_residual_channels,
            hybrid_residual_init=cfg.hybrid_residual_init,
            hybrid_dropout=cfg.hybrid_dropout,
            ocr_mid_channels=cfg.ocr_mid_channels,
            ocr_key_channels=cfg.ocr_key_channels,
            ocr_dropout=cfg.ocr_dropout,
            decoder_upsample_mode=decoder_upsample_mode,
            decoder_dropout=cfg.decoder_dropout,
        )
    elif model_type == "mobile":
        mobile_output_stride = cfg.output_stride if cfg.output_stride in (16, 32) else 16
        model = DeepLabV3PlusMobile(
            num_classes=cfg.num_classes,
            output_stride=mobile_output_stride,
            backbone_pretrained=False,
            aspp_dropout=cfg.aspp_dropout,
            decoder_upsample_mode=decoder_upsample_mode,
            decoder_dropout=cfg.decoder_dropout,
        )
    else:
        raise ValueError(f"Unsupported model_type={model_type}. Use teacher/mobile.")

    model.load_state_dict(state, strict=True)
    model = model.to(device)
    model.eval()
    return model


def _normalize_label_name(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def build_palette(cfg: TrainConfig) -> list[tuple[int, int, int]]:
    dataset_name = normalize_dataset_name(cfg.dataset_name)
    if dataset_name in {"cityscapes", "kitti_semantic"}:
        return list(CITYSCAPES_19_ID2COLOR)
    if dataset_name == "camvid":
        loaded = None
        for csv_name in ("class_dict.csv", "class_palette.csv"):
            loaded = load_class_dict_csv(Path(cfg.camvid_root) / csv_name)
            if loaded is not None:
                break

        rows = CAMVID_LABELS if loaded is None else list(zip(loaded[2], loaded[1]))
        ignore_name_set = {_normalize_label_name(name) for name in CAMVID_IGNORE_LABEL_NAMES}
        colors = [tuple(int(v) for v in color) for name, color in rows if _normalize_label_name(name) not in ignore_name_set]
        return colors

    raise ValueError(f"Unsupported dataset_name={cfg.dataset_name}. Use cityscapes/camvid/kitti_semantic.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Producer-consumer video inference with overlay reconstruction.")
    parser.add_argument("--video", type=Path, required=True, help="Input video path.")
    parser.add_argument("--ckpt", type=Path, required=True, help="Checkpoint path.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "video_stream", help="Output root dir.")
    parser.add_argument("--dataset", type=str, default="cityscapes", help="Dataset palette to use.")
    parser.add_argument("--model-type", type=str, default="teacher", choices=("teacher", "mobile"), help="Checkpoint model type.")
    parser.add_argument("--overlay-alpha", type=float, default=0.55, help="Overlay alpha in [0,1].")
    parser.add_argument("--queue-size", type=int, default=32, help="Producer-consumer queue size.")
    parser.add_argument("--producer-delay-ms", type=float, default=0.0, help="Optional delay per frame to simulate stream input.")
    parser.add_argument("--fps", type=float, default=0.0, help="Override output FPS. <=0 means probe from source video.")
    parser.add_argument("--save-pred-color", action="store_true", help="Also save color segmentation masks.")
    parser.add_argument("--save-pred-id", action="store_true", help="Also save train-id prediction masks.")
    return parser.parse_args()


def run_command(command: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, capture_output=True)


def find_binary(name: str) -> str | None:
    return shutil.which(name)


def require_binary(name: str) -> str:
    binary = find_binary(name)
    if binary is None:
        raise FileNotFoundError(f"Required binary not found in PATH: {name}")
    return binary


def require_opencv():
    try:
        import cv2
    except ImportError as exc:
        raise FileNotFoundError(
            "OpenCV is required as a fallback when ffprobe/ffmpeg is unavailable. "
            "Install ffmpeg or install opencv-python in the active environment."
        ) from exc
    return cv2


def probe_video_info(video_path: Path) -> dict[str, float | int]:
    ffprobe = find_binary("ffprobe")
    if ffprobe is not None:
        result = run_command([
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(video_path),
        ])
        payload = json.loads(result.stdout)
        streams = payload.get("streams", [])
        if not streams:
            raise RuntimeError(f"No video stream found in {video_path}")

        stream = streams[0]
        rate_text = stream.get("avg_frame_rate") or stream.get("r_frame_rate") or "0/1"
        fps = float(Fraction(rate_text)) if rate_text not in {"0/0", "0/1"} else 0.0
        return {
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": fps,
        }

    cv2 = require_opencv()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Failed to probe video geometry: {video_path}")
        return {
            "width": width,
            "height": height,
            "fps": fps,
        }
    finally:
        cap.release()


def clear_pngs(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.glob("*.png"):
        path.unlink()


def extract_frames(video_path: Path, frame_dir: Path) -> list[Path]:
    clear_pngs(frame_dir)
    ffmpeg = find_binary("ffmpeg")
    if ffmpeg is not None:
        run_command([
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vsync",
            "0",
            str(frame_dir / "frame_%06d.png"),
        ])
    else:
        cv2 = require_opencv()
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        try:
            index = 1
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_path = frame_dir / f"frame_{index:06d}.png"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Failed to write frame: {frame_path}")
                index += 1
        finally:
            cap.release()

    frame_paths = sorted(frame_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"No frames extracted from {video_path}")
    return frame_paths


def rebuild_video(source_video: Path, overlay_dir: Path, output_video: Path, fps: float) -> None:
    ffmpeg = find_binary("ffmpeg")
    if ffmpeg is not None:
        output_video.parent.mkdir(parents=True, exist_ok=True)
        run_command([
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(source_video),
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(overlay_dir / "frame_%06d.png"),
            "-map",
            "1:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-shortest",
            str(output_video),
        ])
        return

    cv2 = require_opencv()
    frame_paths = sorted(overlay_dir.glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError(f"No overlay frames found in {overlay_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame: {frame_paths[0]}")
    height, width = first_frame.shape[:2]

    output_video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(float(fps), 1.0),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_video}")

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            writer.write(frame)
    finally:
        writer.release()


def blend_overlay(frame_rgb: np.ndarray, pred_color: np.ndarray, alpha: float) -> np.ndarray:
    alpha = min(max(float(alpha), 0.0), 1.0)
    overlay = frame_rgb.astype(np.float32) * (1.0 - alpha) + pred_color.astype(np.float32) * alpha
    return np.clip(overlay, 0.0, 255.0).astype(np.uint8)


def colorize_pred(pred: np.ndarray, id2color: list[tuple[int, int, int]]) -> np.ndarray:
    color = np.zeros((pred.shape[0], pred.shape[1], 3), dtype=np.uint8)
    for class_id, rgb in enumerate(id2color):
        color[pred == class_id] = rgb
    return color


def load_frame_tensor(frame_path: Path) -> tuple[np.ndarray, torch.Tensor]:
    img = Image.open(frame_path).convert("RGB")
    frame_rgb = np.asarray(img, dtype=np.uint8)
    img_t = normalize_img(pil_to_tensor(img)).unsqueeze(0)
    return frame_rgb, img_t


def save_image(path: Path, array: np.ndarray, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode=mode).save(path)


def safe_put(task_queue: queue.Queue, item, stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            task_queue.put(item, timeout=0.1)
            return
        except queue.Full:
            continue
    raise RuntimeError("Queue put interrupted by stop event.")


def enqueue_sentinel(task_queue: queue.Queue, error_queue: queue.Queue) -> None:
    while True:
        try:
            task_queue.put(SENTINEL, timeout=0.1)
            return
        except queue.Full:
            if not error_queue.empty():
                return
            continue


def producer(
    frame_paths: list[Path],
    task_queue: queue.Queue,
    delay_ms: float,
    stop_event: threading.Event,
    error_queue: queue.Queue,
) -> None:
    try:
        for index, frame_path in enumerate(frame_paths, start=1):
            if stop_event.is_set():
                break
            safe_put(task_queue, FrameTask(index=index, frame_path=frame_path), stop_event)
            if delay_ms > 0:
                time.sleep(delay_ms / 1000.0)
    except RuntimeError as exc:
        if not stop_event.is_set():
            stop_event.set()
            error_queue.put(exc)
    except Exception as exc:
        stop_event.set()
        error_queue.put(exc)
    finally:
        enqueue_sentinel(task_queue, error_queue)


@torch.inference_mode()
def consumer(
    model: torch.nn.Module,
    task_queue: queue.Queue,
    device: torch.device,
    id2color: list[tuple[int, int, int]],
    overlay_dir: Path,
    pred_color_dir: Path | None,
    pred_id_dir: Path | None,
    overlay_alpha: float,
    use_amp: bool,
    stats: dict[str, float | int],
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
            pred = torch.argmax(logits, dim=1)[0].detach().cpu().numpy().astype(np.uint8)
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
    except Exception as exc:
        stop_event.set()
        error_queue.put(exc)
    finally:
        stats["infer_time_sec"] = infer_time_sum


def build_output_layout(
    output_root: Path,
    video_path: Path,
    ckpt_path: Path,
    dataset_name: str,
    model_type: str,
) -> dict[str, Path]:
    run_name = f"{video_path.stem}_{ckpt_path.stem}_{dataset_name}_{model_type}"
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


def main() -> None:
    args = parse_args()
    if not args.video.exists():
        raise FileNotFoundError(f"Input video not found: {args.video}")
    if not args.ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    cfg = TrainConfig() if args.model_type == "teacher" else MobileTrainConfig()
    cfg.dataset_name = normalize_dataset_name(args.dataset)
    apply_dataset_profile(cfg)
    id2color = build_palette(cfg)
    cfg.num_classes = len(id2color)

    layout = build_output_layout(
        args.output_dir,
        args.video,
        args.ckpt,
        cfg.dataset_name,
        args.model_type,
    )
    for key in ("run_dir", "frames_dir", "overlay_dir"):
        layout[key].mkdir(parents=True, exist_ok=True)
    clear_pngs(layout["overlay_dir"])
    if args.save_pred_color:
        layout["pred_color_dir"].mkdir(parents=True, exist_ok=True)
        clear_pngs(layout["pred_color_dir"])
    if args.save_pred_id:
        layout["pred_id_dir"].mkdir(parents=True, exist_ok=True)
        clear_pngs(layout["pred_id_dir"])

    video_info = probe_video_info(args.video)
    fps = float(args.fps) if args.fps > 0 else float(video_info["fps"])
    if fps <= 0:
        raise ValueError("Unable to determine FPS. Pass --fps manually.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(cfg.use_amp and device.type == "cuda")
    model = build_model(cfg=cfg, ckpt_path=args.ckpt, device=device, model_type=args.model_type)

    print(f"[INFO] video={args.video}")
    print(f"[INFO] ckpt={args.ckpt}")
    print(f"[INFO] dataset_name={cfg.dataset_name}")
    print(f"[INFO] model_type={args.model_type}")
    print(f"[INFO] device={device}, amp={use_amp}")
    print(f"[INFO] fps={fps:.6f}, size={video_info['width']}x{video_info['height']}")
    print(f"[INFO] output_dir={layout['run_dir']}")

    t_extract_start = time.perf_counter()
    frame_paths = extract_frames(args.video, layout["frames_dir"])
    extract_time = time.perf_counter() - t_extract_start
    print(f"[INFO] extracted_frames={len(frame_paths)} extract_time={extract_time:.3f}s")

    task_queue: queue.Queue = queue.Queue(maxsize=max(1, int(args.queue_size)))
    stats: dict[str, float | int] = {"processed_frames": 0, "infer_time_sec": 0.0}

    stop_event = threading.Event()
    error_queue: queue.Queue = queue.Queue()

    producer_thread = threading.Thread(
        target=producer,
        args=(frame_paths, task_queue, float(args.producer_delay_ms), stop_event, error_queue),
        daemon=True,
    )
    consumer_thread = threading.Thread(
        target=consumer,
        args=(
            model,
            task_queue,
            device,
            id2color,
            layout["overlay_dir"],
            layout["pred_color_dir"] if args.save_pred_color else None,
            layout["pred_id_dir"] if args.save_pred_id else None,
            float(args.overlay_alpha),
            use_amp,
            stats,
            stop_event,
            error_queue,
        ),
        daemon=True,
    )

    t_infer_start = time.perf_counter()
    producer_thread.start()
    consumer_thread.start()
    producer_thread.join()
    consumer_thread.join()
    stream_time = time.perf_counter() - t_infer_start

    if not error_queue.empty():
        raise RuntimeError(f"Video stream inference failed: {error_queue.get()}")

    rebuild_video(args.video, layout["overlay_dir"], layout["final_video"], fps=fps)

    processed = int(stats["processed_frames"])
    infer_time_sec = float(stats["infer_time_sec"])
    infer_fps = processed / max(infer_time_sec, 1e-12)
    end_to_end_fps = processed / max(stream_time, 1e-12)
    stats_payload = {
        "video": str(args.video),
        "checkpoint": str(args.ckpt),
        "dataset_name": cfg.dataset_name,
        "model_type": args.model_type,
        "num_frames": processed,
        "video_fps": fps,
        "extract_time_sec": extract_time,
        "stream_time_sec": stream_time,
        "infer_time_sec": infer_time_sec,
        "infer_avg_ms": (infer_time_sec / max(processed, 1)) * 1000.0,
        "infer_fps": infer_fps,
        "end_to_end_stream_fps": end_to_end_fps,
        "overlay_alpha": float(args.overlay_alpha),
        "queue_size": int(args.queue_size),
        "producer_delay_ms": float(args.producer_delay_ms),
        "final_video": str(layout["final_video"]),
    }
    layout["stats_json"].write_text(json.dumps(stats_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[INFO] finished")
    print(f"[METRIC] processed_frames={processed}")
    print(f"[METRIC] infer_avg_ms={stats_payload['infer_avg_ms']:.3f}")
    print(f"[METRIC] infer_fps={infer_fps:.2f}")
    print(f"[METRIC] end_to_end_stream_fps={end_to_end_fps:.2f}")
    print(f"[INFO] final_video={layout['final_video']}")
    print(f"[INFO] stats_json={layout['stats_json']}")


if __name__ == "__main__":
    main()
