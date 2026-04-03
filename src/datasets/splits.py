from pathlib import Path


def _normalize_rel_path(entry: str) -> str:
    text = str(entry).strip().replace("\\", "/").lstrip("./")
    return text


def read_split_entries(root: Path, split: str) -> set[str] | None:
    candidates = [
        root / f"{split}.txt",
        root / "splits" / f"{split}.txt",
        root / "ImageSets" / "Segmentation" / f"{split}.txt",
    ]
    for path in candidates:
        if not path.exists():
            continue

        entries = set()
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            entries.add(_normalize_rel_path(text))
        return entries
    return None


def match_split_entry(img_path: Path, base_dir: Path, entries: set[str] | None) -> bool:
    if entries is None:
        return False

    rel = _normalize_rel_path(str(img_path.relative_to(base_dir)))
    stem_rel = _normalize_rel_path(str((img_path.relative_to(base_dir)).with_suffix("")))
    stem = img_path.stem
    return rel in entries or stem_rel in entries or stem in entries
