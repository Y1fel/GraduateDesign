import argparse
import shutil
import statistics
import subprocess
import tempfile
from pathlib import Path
import re


FRAME_NAME_RE = re.compile(r"^(?P<city>.+)_(?P<seq>\d{6})_(?P<frame>\d{6})_leftImg8bit$")
TIMESTAMP_NAME_RE = re.compile(r"^(?P<city>.+)_(?P<seq>\d{6})_(?P<frame>\d{6})_timestamp$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a video from image frames, optionally using per-frame timestamp files.",
        epilog=(
            "Examples:\n"
            "  python scripts/build_video_from_frames.py "
            "--frames-dir data/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_00 "
            "--output outputs/stuttgart_00.mp4 --fps 17\n"
            "  python scripts/build_video_from_frames.py "
            "--frames-dir data/leftImg8bit_demoVideo/leftImg8bit/demoVideo/stuttgart_00 "
            "--timestamp-root data/timestamp_sequence/timestamp_sequence/train/stuttgart "
            "--timestamp-mode cityscapes-demo --dry-run\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--frames-dir", type=Path, required=True, help="Directory containing ordered frame images.")
    parser.add_argument("--output", type=Path, default=None, help="Output video path. Defaults to outputs/<frames-dir>.mp4.")
    parser.add_argument(
        "--timestamp-root",
        type=Path,
        default=None,
        help="Timestamp directory or root directory containing timestamp txt files.",
    )
    parser.add_argument(
        "--timestamp-mode",
        choices=("none", "exact", "cityscapes-demo"),
        default="none",
        help=(
            "Timestamp matching mode.\n"
            "none: ignore timestamps and use constant FPS.\n"
            "exact: match '<frame>_leftImg8bit.png' to '<frame>_timestamp.txt'.\n"
            "cityscapes-demo: heuristic mapping for Cityscapes leftImg8bit_demoVideo + timestamp_sequence."
        ),
    )
    parser.add_argument(
        "--timestamp-city",
        type=str,
        default=None,
        help="Override the city name used when timestamp-mode=cityscapes-demo.",
    )
    parser.add_argument(
        "--timestamp-offset",
        type=str,
        default="0",
        help=(
            "Frame-index offset used by cityscapes-demo mode. Use an integer or 'auto'. "
            "The current Cityscapes timestamp_sequence package is not a direct demoVideo match, "
            "so this may need adjustment."
        ),
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=30,
        help="Frames per Cityscapes timestamp sequence segment. Used by cityscapes-demo mode.",
    )
    parser.add_argument("--fps", type=float, default=17.0, help="Fallback FPS when timestamps are unavailable.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg"],
        help="Frame file extensions to include.",
    )
    parser.add_argument(
        "--ffmpeg",
        type=Path,
        default=None,
        help="Path to ffmpeg. If omitted, the script uses ffmpeg from PATH.",
    )
    parser.add_argument("--codec", type=str, default="libx264", help="Video codec passed to ffmpeg.")
    parser.add_argument("--pix-fmt", type=str, default="yuv420p", help="Pixel format passed to ffmpeg.")
    parser.add_argument("--crf", type=int, default=18, help="CRF passed to ffmpeg for libx264-like encoders.")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="Optional path to keep the generated ffconcat manifest.",
    )
    parser.add_argument(
        "--strict-timestamps",
        action="store_true",
        help="Fail when any frame cannot be assigned a timestamp-derived duration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect frames/timestamps and print the plan without calling ffmpeg.",
    )
    return parser.parse_args()


def find_ffmpeg(custom_path: Path | None) -> str | None:
    if custom_path is not None:
        if not custom_path.exists():
            raise FileNotFoundError(f"ffmpeg not found: {custom_path}")
        return str(custom_path)

    binary = shutil.which("ffmpeg")
    return binary


def collect_frame_paths(frames_dir: Path, extensions: list[str]) -> list[Path]:
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    if not frames_dir.is_dir():
        raise NotADirectoryError(f"Expected a directory: {frames_dir}")

    ext_set = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    frame_paths = sorted(
        path for path in frames_dir.iterdir() if path.is_file() and path.suffix.lower() in ext_set
    )
    if not frame_paths:
        raise RuntimeError(f"No frame images found in {frames_dir}")
    return frame_paths


def read_timestamp_file(path: Path) -> int:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Timestamp file is empty: {path}")
    return int(text)


def build_exact_timestamp_map(timestamp_root: Path) -> dict[str, int]:
    timestamp_paths = sorted(timestamp_root.rglob("*_timestamp.txt"))
    if not timestamp_paths:
        raise RuntimeError(f"No timestamp txt files found under {timestamp_root}")

    mapping: dict[str, int] = {}
    for path in timestamp_paths:
        mapping[path.name] = read_timestamp_file(path)
    return mapping


def parse_frame_name(frame_path: Path) -> tuple[str, int, int]:
    match = FRAME_NAME_RE.match(frame_path.stem)
    if match is None:
        raise ValueError(
            f"Unsupported frame name format: {frame_path.name}. "
            "Expected '*_<seq>_<frame>_leftImg8bit.png'."
        )
    city = match.group("city")
    seq = int(match.group("seq"))
    frame = int(match.group("frame"))
    return city, seq, frame


def parse_timestamp_name(timestamp_path: Path) -> tuple[str, int, int]:
    match = TIMESTAMP_NAME_RE.match(timestamp_path.stem)
    if match is None:
        raise ValueError(f"Unsupported timestamp name format: {timestamp_path.name}")
    city = match.group("city")
    seq = int(match.group("seq"))
    frame = int(match.group("frame"))
    return city, seq, frame


def default_timestamp_city(frame_city: str) -> str:
    match = re.match(r"^(?P<base>.+)_\d{2}$", frame_city)
    if match is not None:
        return match.group("base")
    return frame_city


def build_cityscapes_demo_index(
    timestamp_root: Path,
    target_city: str,
    sequence_length: int,
) -> dict[int, int]:
    timestamp_paths = sorted(timestamp_root.rglob("*_timestamp.txt"))
    if not timestamp_paths:
        raise RuntimeError(f"No timestamp txt files found under {timestamp_root}")

    mapping: dict[int, int] = {}
    matched = 0
    for path in timestamp_paths:
        city, seq, frame = parse_timestamp_name(path)
        if city != target_city:
            continue
        synthetic_index = seq * sequence_length + frame
        mapping[synthetic_index] = read_timestamp_file(path)
        matched += 1

    if matched == 0:
        raise RuntimeError(
            f"No timestamp files for city '{target_city}' were found under {timestamp_root}"
        )
    return mapping


def resolve_cityscapes_demo_offset(
    raw_offset: str,
    frame_indices: list[int],
    timestamp_index: dict[int, int],
    sequence_length: int,
) -> int:
    if raw_offset != "auto":
        return int(raw_offset)

    if not frame_indices or not timestamp_index:
        return 0

    min_frame = min(frame_indices)
    max_frame = max(frame_indices)
    min_ts = min(timestamp_index)
    max_ts = max(timestamp_index)
    start = min_ts - max_frame
    stop = max_ts - min_frame

    best_offset = 0
    best_score = -1
    for offset in range(start, stop + 1):
        if sequence_length > 0 and offset % sequence_length != 0:
            continue
        score = sum(1 for frame_index in frame_indices if frame_index + offset in timestamp_index)
        if score > best_score:
            best_score = score
            best_offset = offset
            continue
        if score == best_score and abs(offset) < abs(best_offset):
            best_offset = offset

    return best_offset


def suggest_cityscapes_demo_offset(
    frame_indices: list[int],
    timestamp_index: dict[int, int],
    sequence_length: int,
) -> tuple[int, int]:
    best_offset = resolve_cityscapes_demo_offset(
        "auto",
        frame_indices=frame_indices,
        timestamp_index=timestamp_index,
        sequence_length=sequence_length,
    )
    matched = sum(1 for frame_index in frame_indices if frame_index + best_offset in timestamp_index)
    return best_offset, matched


def build_frame_records(
    frame_paths: list[Path],
    timestamp_mode: str,
    timestamp_root: Path | None,
    timestamp_city: str | None,
    timestamp_offset: str,
    sequence_length: int,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    stats: dict[str, object] = {
        "timestamp_mode": timestamp_mode,
        "matched_timestamps": 0,
        "missing_timestamps": 0,
        "cityscapes_demo_offset": None,
        "suggested_offset": None,
        "suggested_offset_match_count": None,
        "notes": [],
    }

    if timestamp_mode == "none":
        for frame_path in frame_paths:
            records.append({"frame_path": frame_path, "timestamp_ns": None})
        return records, stats

    if timestamp_root is None:
        raise ValueError(f"--timestamp-root is required when --timestamp-mode={timestamp_mode}")

    if timestamp_mode == "exact":
        exact_map = build_exact_timestamp_map(timestamp_root)
        for frame_path in frame_paths:
            expected_name = f"{frame_path.stem.removesuffix('_leftImg8bit')}_timestamp.txt"
            timestamp_ns = exact_map.get(expected_name)
            if timestamp_ns is None:
                stats["missing_timestamps"] = int(stats["missing_timestamps"]) + 1
            else:
                stats["matched_timestamps"] = int(stats["matched_timestamps"]) + 1
            records.append({"frame_path": frame_path, "timestamp_ns": timestamp_ns})
        return records, stats

    first_city, _, _ = parse_frame_name(frame_paths[0])
    target_city = timestamp_city or default_timestamp_city(first_city)
    timestamp_index = build_cityscapes_demo_index(
        timestamp_root=timestamp_root,
        target_city=target_city,
        sequence_length=sequence_length,
    )

    frame_indices: list[int] = []
    for frame_path in frame_paths:
        _, seq, frame = parse_frame_name(frame_path)
        frame_indices.append(seq * sequence_length + frame)

    suggested_offset, suggested_match_count = suggest_cityscapes_demo_offset(
        frame_indices=frame_indices,
        timestamp_index=timestamp_index,
        sequence_length=sequence_length,
    )
    offset = resolve_cityscapes_demo_offset(
        timestamp_offset,
        frame_indices=frame_indices,
        timestamp_index=timestamp_index,
        sequence_length=sequence_length,
    )
    stats["cityscapes_demo_offset"] = offset
    stats["suggested_offset"] = suggested_offset
    stats["suggested_offset_match_count"] = suggested_match_count

    if timestamp_offset == "0":
        stats["notes"].append(
            "Cityscapes demoVideo and timestamp_sequence are different packages; "
            "offset=0 is only a direct index heuristic."
        )

    for frame_path, frame_index in zip(frame_paths, frame_indices):
        timestamp_ns = timestamp_index.get(frame_index + offset)
        if timestamp_ns is None:
            stats["missing_timestamps"] = int(stats["missing_timestamps"]) + 1
        else:
            stats["matched_timestamps"] = int(stats["matched_timestamps"]) + 1
        records.append({"frame_path": frame_path, "timestamp_ns": timestamp_ns, "frame_index": frame_index})

    return records, stats


def median_positive_delta_ns(timestamps_ns: list[int]) -> int | None:
    deltas = [b - a for a, b in zip(timestamps_ns, timestamps_ns[1:]) if b > a]
    if not deltas:
        return None
    return int(statistics.median(deltas))


def build_durations_seconds(
    records: list[dict[str, object]],
    fallback_fps: float,
    strict_timestamps: bool,
) -> tuple[list[float], dict[str, object]]:
    if fallback_fps <= 0:
        raise ValueError("--fps must be > 0")

    fallback_duration = 1.0 / fallback_fps
    timestamps_ns = [int(ts) for ts in (record["timestamp_ns"] for record in records) if ts is not None]
    expected_delta_ns = median_positive_delta_ns(timestamps_ns)
    expected_duration = (
        max(expected_delta_ns / 1_000_000_000.0, 0.001) if expected_delta_ns is not None else fallback_duration
    )

    durations: list[float] = []
    filled_from_fallback = 0
    for current, nxt in zip(records, records[1:]):
        current_ts = current["timestamp_ns"]
        next_ts = nxt["timestamp_ns"]
        if current_ts is not None and next_ts is not None and int(next_ts) > int(current_ts):
            duration = max((int(next_ts) - int(current_ts)) / 1_000_000_000.0, 0.001)
        else:
            duration = expected_duration
            filled_from_fallback += 1
        durations.append(duration)

    if not records:
        raise RuntimeError("No records available to build durations.")
    durations.append(durations[-1] if durations else expected_duration)

    if strict_timestamps and filled_from_fallback > 0:
        raise RuntimeError(
            f"{filled_from_fallback} frame intervals could not be derived from timestamps. "
            "Disable --strict-timestamps to fill them with the median interval."
        )

    stats = {
        "filled_intervals": filled_from_fallback,
        "expected_duration_sec": expected_duration,
        "fallback_duration_sec": fallback_duration,
        "average_duration_sec": statistics.fmean(durations),
        "estimated_fps": 1.0 / statistics.fmean(durations),
    }
    return durations, stats


def ffconcat_quote(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def write_ffconcat_manifest(
    records: list[dict[str, object]],
    durations: list[float],
    manifest_path: Path,
) -> None:
    if len(records) != len(durations):
        raise ValueError("records and durations must have the same length")

    lines = ["ffconcat version 1.0"]
    for record, duration in zip(records, durations):
        frame_path = Path(record["frame_path"])
        lines.append(f"file '{ffconcat_quote(frame_path)}'")
        lines.append(f"duration {duration:.9f}")

    last_path = Path(records[-1]["frame_path"])
    lines.append(f"file '{ffconcat_quote(last_path)}'")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def encode_video(
    ffmpeg_bin: str,
    manifest_path: Path,
    output_path: Path,
    codec: str,
    pix_fmt: str,
    crf: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_bin,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(manifest_path),
        "-vsync",
        "vfr",
        "-c:v",
        codec,
        "-crf",
        str(crf),
        "-pix_fmt",
        pix_fmt,
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(command, check=True)


def encode_video_with_opencv(
    records: list[dict[str, object]],
    durations: list[float],
    output_path: Path,
    fallback_fps: float,
) -> None:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "Neither ffmpeg nor OpenCV is available. Install ffmpeg, or install opencv-python in the active environment."
        ) from exc

    first_frame = cv2.imread(str(records[0]["frame_path"]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read frame: {records[0]['frame_path']}")
    height, width = first_frame.shape[:2]

    average_duration = statistics.fmean(durations)
    output_fps = max(float(fallback_fps), 1.0 / average_duration if average_duration > 0 else float(fallback_fps))

    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    writer_specs = [
        (".mp4", "mp4v"),
        (".avi", "MJPG"),
    ]
    last_error = None
    for suffix, fourcc_text in writer_specs:
        target_path = output_path if output_path.suffix.lower() == suffix else output_path.with_suffix(suffix)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_text)
        candidate = cv2.VideoWriter(str(target_path), fourcc, output_fps, (width, height))
        if candidate.isOpened():
            writer = candidate
            output_path = target_path
            break
        candidate.release()
        last_error = f"VideoWriter open failed for {target_path.name} with codec {fourcc_text}"

    if writer is None:
        raise RuntimeError(last_error or "OpenCV VideoWriter could not be opened.")

    try:
        for record, duration in zip(records, durations):
            frame = cv2.imread(str(record["frame_path"]))
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {record['frame_path']}")
            repeat = max(1, int(round(duration * output_fps)))
            for _ in range(repeat):
                writer.write(frame)
    finally:
        writer.release()

    print(
        "note=ffmpeg not found; used OpenCV fallback with approximate constant-FPS encoding "
        f"(output_fps={output_fps:.6f})"
    )
    print(f"actual_output={output_path}")


def default_output_path(frames_dir: Path) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    return project_root / "outputs" / f"{frames_dir.name}.mp4"


def print_summary(
    frame_paths: list[Path],
    output_path: Path,
    duration_stats: dict[str, object],
    record_stats: dict[str, object],
) -> None:
    print(f"frames_dir={frame_paths[0].parent}")
    print(f"frames={len(frame_paths)}")
    print(f"output={output_path}")
    print(f"timestamp_mode={record_stats['timestamp_mode']}")
    print(f"matched_timestamps={record_stats['matched_timestamps']}")
    print(f"missing_timestamps={record_stats['missing_timestamps']}")

    if record_stats["cityscapes_demo_offset"] is not None:
        print(f"cityscapes_demo_offset={record_stats['cityscapes_demo_offset']}")
        print(
            "suggested_offset="
            f"{record_stats['suggested_offset']} "
            f"(matches={record_stats['suggested_offset_match_count']}/{len(frame_paths)})"
        )

    for note in record_stats["notes"]:
        print(f"note={note}")

    print(f"filled_intervals={duration_stats['filled_intervals']}")
    print(f"average_frame_duration_sec={duration_stats['average_duration_sec']:.9f}")
    print(f"estimated_fps={duration_stats['estimated_fps']:.6f}")


def main() -> None:
    args = parse_args()
    frame_paths = collect_frame_paths(args.frames_dir, args.extensions)
    output_path = args.output or default_output_path(args.frames_dir)

    records, record_stats = build_frame_records(
        frame_paths=frame_paths,
        timestamp_mode=args.timestamp_mode,
        timestamp_root=args.timestamp_root,
        timestamp_city=args.timestamp_city,
        timestamp_offset=args.timestamp_offset,
        sequence_length=args.sequence_length,
    )
    durations, duration_stats = build_durations_seconds(
        records=records,
        fallback_fps=args.fps,
        strict_timestamps=args.strict_timestamps,
    )
    print_summary(
        frame_paths=frame_paths,
        output_path=output_path,
        duration_stats=duration_stats,
        record_stats=record_stats,
    )

    if args.dry_run:
        return

    ffmpeg_bin = find_ffmpeg(args.ffmpeg)

    if ffmpeg_bin is not None and args.manifest_out is not None:
        manifest_path = args.manifest_out
        write_ffconcat_manifest(records, durations, manifest_path)
        encode_video(
            ffmpeg_bin=ffmpeg_bin,
            manifest_path=manifest_path,
            output_path=output_path,
            codec=args.codec,
            pix_fmt=args.pix_fmt,
            crf=args.crf,
        )
        return

    if ffmpeg_bin is not None:
        with tempfile.TemporaryDirectory(prefix="frame_video_") as temp_dir:
            manifest_path = Path(temp_dir) / "frames.ffconcat"
            write_ffconcat_manifest(records, durations, manifest_path)
            encode_video(
                ffmpeg_bin=ffmpeg_bin,
                manifest_path=manifest_path,
                output_path=output_path,
                codec=args.codec,
                pix_fmt=args.pix_fmt,
                crf=args.crf,
            )
        return

    if args.manifest_out is not None:
        write_ffconcat_manifest(records, durations, args.manifest_out)
        print("note=ffmpeg not found; manifest was written but no ffmpeg encode was performed.")

    encode_video_with_opencv(
        records=records,
        durations=durations,
        output_path=output_path,
        fallback_fps=args.fps,
    )


if __name__ == "__main__":
    main()
