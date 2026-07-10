import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_ARCHIVES_DIR = DATA_DIR / "raw_archives"
EXTRACTED_DIR = DATA_DIR / "extracted"
SKELETON_DIR = DATA_DIR / "skeletons_rtmw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}
ARCHIVE_EXTENSIONS = {".zip", ".rar", ".7z", ".tar", ".gz", ".tgz", ".bz2", ".xz"}
NTU_RGB_ARCHIVE_NAMES = ["nturgbd_rgb_s{:03d}.zip".format(i) for i in range(1, 33)]

NTU_NAME_RE = re.compile(
    r"S(?P<setup>\d{3})C(?P<camera>\d{3})P(?P<subject>\d{3})"
    r"R(?P<replication>\d{3})A(?P<action>\d{3})",
    re.IGNORECASE,
)

RTMW_CONFIG = "rtmw-l_8xb320-270e_cocktail14-384x288"
RTMW_WEIGHTS_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmw/"
    "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
)
RTMW_WEIGHTS_FILE = "rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth"
RTMW_WEIGHTS_PATH = MODELS_DIR / RTMW_WEIGHTS_FILE

RTMDET_CONFIG = "rtmdet_tiny_8xb32-300e_coco"
RTMDET_WEIGHTS_URL = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet_tiny_8xb32-300e_coco/"
    "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
)
RTMDET_WEIGHTS_FILE = "rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
RTMDET_WEIGHTS_PATH = MODELS_DIR / RTMDET_WEIGHTS_FILE

YOLO_WEIGHTS_FILE = "yolov8n.pt"
YOLO_WEIGHTS_PATH = MODELS_DIR / YOLO_WEIGHTS_FILE

NTU60_XSUB_TRAIN = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16,
    17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
}

NTU120_XSUB_TRAIN = {
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16,
    17, 18, 19, 25, 27, 28, 31, 34, 35, 38,
    45, 46, 47, 49, 50, 52, 53, 54, 55, 56,
    57, 58, 59, 70, 74, 78, 80, 81, 82, 83,
    84, 85, 86, 89, 91, 92, 93, 94, 95, 97,
    98, 100, 103,
}
