from dataclasses import dataclass


RGB = tuple[int, int, int]


@dataclass(frozen=True)
class SegmentationDatasetMeta:
    dataset_name: str
    num_classes: int
    class_names: tuple[str, ...]
    id2color: tuple[RGB, ...]
