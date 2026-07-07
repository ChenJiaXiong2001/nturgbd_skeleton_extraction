import importlib.util
import platform
import sys
from pathlib import Path

from ntu_rtmw.compat import patch_runtime
from ntu_rtmw.constants import RTMDET_WEIGHTS_PATH, RTMW_WEIGHTS_PATH, YOLO_WEIGHTS_PATH


REQUIRED_MODULES = [
    "numpy",
    "cv2",
    "torch",
    "mmengine",
    "mmcv",
    "mmdet",
    "mmpose",
    "ultralytics",
]


def main():
    print("python:", sys.version.replace("\n", " "))
    print("executable:", sys.executable)
    print("architecture:", platform.architecture()[0])
    if sys.version_info < (3, 10):
        raise SystemExit("Expected Python 3.10 or newer.")

    weights = Path(RTMW_WEIGHTS_PATH)
    print("rtmw weights:", weights if weights.exists() else "missing: {}".format(weights))
    det_weights = Path(RTMDET_WEIGHTS_PATH)
    print("rtmdet weights:", det_weights if det_weights.exists() else "missing: {}".format(det_weights))
    yolo_weights = Path(YOLO_WEIGHTS_PATH)
    print("yolo weights:", yolo_weights if yolo_weights.exists() else "missing: {}".format(yolo_weights))

    missing = [name for name in REQUIRED_MODULES if importlib.util.find_spec(name) is None]
    if missing:
        print("Missing modules: {}".format(", ".join(missing)))
        if "mmcv" in missing:
            print(
                "Install mmcv with MIM, not pip -r requirements.txt. "
                "If MIM downloads mmcv-*.tar.gz on Python 3.13, no matching wheel was found."
            )
        raise SystemExit(1)

    patch_runtime()
    try:
        from mmpose.apis import MMPoseInferencer  # noqa: F401
    except Exception as exc:
        raise SystemExit("MMPoseInferencer import failed: {}".format(exc)) from exc

    import torch

    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda device:", torch.cuda.get_device_name(0))
    print("RTMW environment looks ready.")


if __name__ == "__main__":
    main()
