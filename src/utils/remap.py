from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np

@dataclass
class RemapPack:
    # old_id -> new_id (0..K-1) or ignore
    old2new: Dict[int, int]
    # LUT: shape (256,), lut[old_id] = new_id or ignore_index
    lut: np.ndarray
    # new_id -> RGB color (for visualization)
    id2color_new: List[Tuple[int, int, int]]
    # new_id -> class name
    id2name_new: List[str]

def build_remap_pack_from_names(
    id2name_old: Dict[int, str],
    id2color_old: Dict[int, Tuple[int, int, int]] | List[Tuple[int, int, int]],
    selected_names: List[str],
    ignore_index: int = 255,
) -> RemapPack:
    """
    你只填 selected_names (长度=5)，顺序就是 new_id=0..4 的顺序。
    其余所有 old_id -> ignore_index
    """
    # 反查 name -> old_id
    name2old = {v: k for k, v in id2name_old.items()}

    # 容错：允许大小写/空格不一致时你自行对齐；这里严格匹配，找不到就报错
    missing = [n for n in selected_names if n not in name2old]
    if missing:
        raise KeyError(f"selected_names not in class_dict.csv: {missing}")

    old_ids = [name2old[n] for n in selected_names]

    old2new = {old_id: new_id for new_id, old_id in enumerate(old_ids)}

    lut = np.full((256,), fill_value=ignore_index, dtype=np.uint8)
    for old_id, new_id in old2new.items():
        lut[int(old_id)] = np.uint8(new_id)

    # 生成新的 5类调色板（new_id -> old_id -> color/name）
    def get_color(oid: int) -> Tuple[int, int, int]:
        if isinstance(id2color_old, dict):
            return tuple(id2color_old[oid])
        return tuple(id2color_old[oid])

    id2color_new = [get_color(oid) for oid in old_ids]
    id2name_new = [id2name_old[oid] for oid in old_ids]

    return RemapPack(old2new=old2new, lut=lut, id2color_new=id2color_new, id2name_new=id2name_new)


CAMVID_11_GROUP_NAME_SPECS: List[List[str]] = [
    ["Sky"],
    ["Building", "Wall", "Archway", "Bridge", "Tunnel"],
    ["Column_Pole", "TrafficCone"],
    ["Road", "LaneMkgsDriv", "LaneMkgsNonDriv"],
    ["Sidewalk", "RoadShoulder", "ParkingBlock"],
    ["Tree", "VegetationMisc"],
    ["SignSymbol", "TrafficLight", "Misc_Text"],
    ["Fence"],
    ["Car", "SUVPickupTruck", "Truck_Bus", "Train", "OtherMoving", "MotorcycleScooter"],
    ["Pedestrian", "Child", "Animal", "CartLuggagePram"],
    ["Bicyclist"],
]


def _build_name2old(id2name: List[str] | Dict[int, str]) -> Dict[str, int]:
    if isinstance(id2name, dict):
        items = id2name.items()
    else:
        items = enumerate(id2name)
    return {name: int(old_id) for old_id, name in items}


def build_camvid_11_groups_from_names(id2name: List[str] | Dict[int, str]) -> List[List[int]]:
    name2old = _build_name2old(id2name)

    groups_11: List[List[int]] = []
    for cls_group in CAMVID_11_GROUP_NAME_SPECS:
        missing = [name for name in cls_group if name not in name2old]
        if missing:
            raise ValueError(f"class_dict.csv 缺少类别: {missing}")
        groups_11.append([name2old[name] for name in cls_group])

    return groups_11


def assert_camvid_key_old_ids(id2name: List[str] | Dict[int, str]) -> None:
    if isinstance(id2name, dict):
        get_name = lambda idx: str(id2name.get(idx, "<MISSING>"))
    else:
        get_name = lambda idx: str(id2name[idx]) if 0 <= idx < len(id2name) else "<MISSING>"

    expected = {
        21: "Sky",
        17: "Road",
        5: "Car",
    }
    print("[CLASS-DICT] key old_id check:")
    mismatch = []
    for old_id, expected_name in expected.items():
        actual_name = get_name(old_id)
        print(f"  - old_id {old_id:>2}: expected={expected_name}, actual={actual_name}")
        if actual_name != expected_name:
            mismatch.append((old_id, expected_name, actual_name))

    if mismatch:
        details = "; ".join(
            f"old_id {old_id} expected {expected_name} but got {actual_name}"
            for old_id, expected_name, actual_name in mismatch
        )
        raise RuntimeError(f"class_dict.csv 顺序与 GROUPS_11 假设不一致: {details}")
