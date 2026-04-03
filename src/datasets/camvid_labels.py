CAMVID_LABELS = [
    ("Sky", (128, 128, 128)),
    ("Building", (128, 0, 0)),
    ("Pole", (192, 192, 128)),
    ("Road", (128, 64, 128)),
    ("Pavement", (60, 40, 222)),
    ("Tree", (128, 128, 0)),
    ("SignSymbol", (192, 128, 128)),
    ("Fence", (64, 64, 128)),
    ("Car", (64, 0, 128)),
    ("Pedestrian", (64, 64, 0)),
    ("Bicyclist", (0, 128, 192)),
    ("Unlabelled", (0, 0, 0)),
]

CAMVID_IGNORE_LABEL_NAMES = {
    "void",
    "unlabelled",
    "unlabeled",
    "background",
    "unknown",
}

CAMVID_11_CLASS_NAMES = tuple(name for name, _ in CAMVID_LABELS if name.lower() not in CAMVID_IGNORE_LABEL_NAMES)
CAMVID_11_ID2COLOR = tuple(color for name, color in CAMVID_LABELS if name.lower() not in CAMVID_IGNORE_LABEL_NAMES)
