CITYSCAPES_34_CLASS_NAMES = [
    "unlabeled",
    "ego vehicle",
    "rectification border",
    "out of roi",
    "static",
    "dynamic",
    "ground",
    "road",
    "sidewalk",
    "parking",
    "rail track",
    "building",
    "wall",
    "fence",
    "guard rail",
    "bridge",
    "tunnel",
    "pole",
    "polegroup",
    "traffic light",
    "traffic sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "caravan",
    "trailer",
    "train",
    "motorcycle",
    "bicycle",
]

# official trainId-compatible 19-class setup (255 means ignored)
CITYSCAPES_34_TO_19 = [
    255,  # 00 unlabeled
    255,  # 01 ego vehicle
    255,  # 02 rectification border
    255,  # 03 out of roi
    255,  # 04 static
    255,  # 05 dynamic
    255,  # 06 ground
    0,    # 07 road
    1,    # 08 sidewalk
    0,    # 09 parking -> road
    255,  # 10 rail track
    2,    # 11 building
    3,    # 12 wall
    4,    # 13 fence
    4,    # 14 guard rail -> fence
    2,    # 15 bridge -> building
    2,    # 16 tunnel -> building
    5,    # 17 pole
    5,    # 18 polegroup -> pole
    6,    # 19 traffic light
    7,    # 20 traffic sign
    8,    # 21 vegetation
    9,    # 22 terrain
    10,   # 23 sky
    11,   # 24 person
    12,   # 25 rider
    13,   # 26 car
    14,   # 27 truck
    15,   # 28 bus
    14,   # 29 caravan -> truck
    14,   # 30 trailer -> truck
    16,   # 31 train
    17,   # 32 motorcycle
    18,   # 33 bicycle
]

CITYSCAPES_19_CLASS_NAMES = [
    "road",           # 0
    "sidewalk",       # 1
    "building",       # 2
    "wall",           # 3
    "fence",          # 4
    "pole",           # 5
    "traffic light",  # 6
    "traffic sign",   # 7
    "vegetation",     # 8
    "terrain",        # 9
    "sky",            # 10
    "person",         # 11
    "rider",          # 12
    "car",            # 13
    "truck",          # 14
    "bus",            # 15
    "train",          # 16
    "motorcycle",     # 17
    "bicycle",        # 18
]

# official Cityscapes palette for labelId 0..33
CITYSCAPES_34_ID2COLOR = [
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (0, 0, 0),
    (111, 74, 0),
    (81, 0, 81),
    (128, 64, 128),
    (244, 35, 232),
    (250, 170, 160),
    (230, 150, 140),
    (70, 70, 70),
    (102, 102, 156),
    (190, 153, 153),
    (180, 165, 180),
    (150, 100, 100),
    (150, 120, 90),
    (153, 153, 153),
    (153, 153, 153),
    (250, 170, 30),
    (220, 220, 0),
    (107, 142, 35),
    (152, 251, 152),
    (70, 130, 180),
    (220, 20, 60),
    (255, 0, 0),
    (0, 0, 142),
    (0, 0, 70),
    (0, 60, 100),
    (0, 0, 90),
    (0, 0, 110),
    (0, 80, 100),
    (0, 0, 230),
    (119, 11, 32),
]

CITYSCAPES_19_ID2COLOR = [
    CITYSCAPES_34_ID2COLOR[7],
    CITYSCAPES_34_ID2COLOR[8],
    CITYSCAPES_34_ID2COLOR[11],
    CITYSCAPES_34_ID2COLOR[12],
    CITYSCAPES_34_ID2COLOR[13],
    CITYSCAPES_34_ID2COLOR[17],
    CITYSCAPES_34_ID2COLOR[19],
    CITYSCAPES_34_ID2COLOR[20],
    CITYSCAPES_34_ID2COLOR[21],
    CITYSCAPES_34_ID2COLOR[22],
    CITYSCAPES_34_ID2COLOR[23],
    CITYSCAPES_34_ID2COLOR[24],
    CITYSCAPES_34_ID2COLOR[25],
    CITYSCAPES_34_ID2COLOR[26],
    CITYSCAPES_34_ID2COLOR[27],
    CITYSCAPES_34_ID2COLOR[28],
    CITYSCAPES_34_ID2COLOR[31],
    CITYSCAPES_34_ID2COLOR[32],
    CITYSCAPES_34_ID2COLOR[33],
]
