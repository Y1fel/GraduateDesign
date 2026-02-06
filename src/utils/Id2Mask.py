from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import csv

import numpy as np
from PIL import Image


RGBTuple = Tuple[int, int, int]  # (r,g,b)

# search for .csv and transform
def load_class_dict_csv(
    csv_path: Union[str, Path],
) -> Optional[Tuple[Dict[RGBTuple, int], List[RGBTuple], List[str]]]:

    csv_path = Path(csv_path)
    if not csv_path.exists():
        return None

    id2color= []
    id2name = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        def find_key(target: str) -> Optional[str]:
            for k in (reader.fieldnames or []):
                if k.strip().lower() == target:
                    return k
            return None

        nk = find_key("name")
        rk = find_key("r")
        gk = find_key("g")
        bk = find_key("b")
        if rk is None or gk is None or bk is None:
            return None

        for row in reader:
            try:
                name = str(row[nk]).strip() if nk else ""
                r = int(str(row[rk]).strip())
                g = int(str(row[gk]).strip())
                b = int(str(row[bk]).strip())
                id2name.append(name)
                id2color.append((r, g, b))
            except Exception:
                continue

    if not id2color:
        return None

    color2id = {}
    for cid, rgb in enumerate(id2color):
        if rgb in color2id:
            raise ValueError(f"Duplicate RGB color in csv: {rgb} (old={color2id[rgb]}, new={cid})")
        color2id[rgb] = cid

    return color2id, id2color, id2name


def color_mask_to_id(
    mask_rgb: Union[Image.Image, np.ndarray],
    color2id: Dict[RGBTuple, int],
    ignore_index: int = 255,
) -> np.ndarray:

    if isinstance(mask_rgb, Image.Image):
        arr = np.array(mask_rgb.convert("RGB"), dtype=np.uint8)  # (H,W,3)
    else:
        arr = np.asarray(mask_rgb, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError(f"mask_rgb must be (H,W,3), got {arr.shape}")

    h, w, _ = arr.shape
    out = np.full((h, w), ignore_index, dtype=np.uint8)

    for (r, g, b), cid in color2id.items():
        m = (arr[:, :, 0] == r) & (arr[:, :, 1] == g) & (arr[:, :, 2] == b)
        out[m] = cid

    return out


def id_mask_to_color(
    mask_id: Union[np.ndarray, Image.Image],
    id2color: List[RGBTuple],
    ignore_index: int = 255,
    ignore_color: RGBTuple = (0, 0, 0),
) -> np.ndarray:

    if isinstance(mask_id, Image.Image):
        mid = np.array(mask_id, dtype=np.int64)
    else:
        mid = np.asarray(mask_id, dtype=np.int64)
        if mid.ndim != 2:
            raise ValueError(f"mask_id must be (H,W), got {mid.shape}")

    h, w = mid.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)

    for cid, (r, g, b) in enumerate(id2color):
        out[mid == cid] = (r, g, b)

    out[mid == ignore_index] = ignore_color
    return out



if __name__ == "__main__":
    res = load_class_dict_csv("D:/MachineLearning/GraduateDesign/data/archive/CamVid/class_dict.csv")
    if res is None:
        raise RuntimeError("Failed to load class_dict.csv")
    color2id, id2color, id2name = res
    print(f"Loaded classes: {len(id2color)}")
    print(list(zip(id2name[:32], id2color[:32])))