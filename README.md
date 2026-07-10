# NTU RGB+D Skeleton Extraction with RTMW

This project extracts 2D whole-body skeletons from NTU RGB+D RGB videos with
OpenMMLab MMPose RTMW.

The default model uses the official RTMW-L 384x288 checkpoint with the
`rtmw-l_8xb320-270e_cocktail14-384x288` MMPose config, which predicts
COCO-WholeBody 133 keypoints. For NTU RGB+D, the script keeps at most two
people per frame and writes one compressed `.npz` file per video.

## Environment

Use Python 3.10/3.11 for the accurate full `mmcv + mmdet + RTMDet + RTMW`
pipeline. Python 3.13 can run a compatibility path, but full `mmcv` is much
harder to install on Windows.

Install into the system Python 3.10:

```powershell
py -3.10 -m pip install -U pip setuptools wheel
py -3.10 -m pip install -r requirements.txt
py -3.10 -m mim install "mmcv>=2.0.0,<2.2.0"
py -3.10 -m mim install "mmdet>=3.2.0,<3.3.0"
py -3.10 -m mim install "mmpose>=1.3.0"
py -3.10 check_env.py
```

If you need GPU support, install the correct PyTorch build from the official
PyTorch selector first, then install the OpenMMLab packages. MMPose requires
Python 3.7+, CUDA 9.2+, and PyTorch 1.8+; PyTorch currently supports Python
3.10-3.14 on Windows. MMCV must match your PyTorch/CUDA build.

### Python 3.13 and MMCV

Do not install `mmcv` through `pip install -r requirements.txt`. If pip prints a
line like `Using cached mmcv-2.1.0.tar.gz`, it did not find a prebuilt wheel for
your Python/PyTorch/CUDA combination and is trying to compile MMCV from source.
That is why Python 3.13 can fail with:

```text
ModuleNotFoundError: No module named 'pkg_resources'
```

First try the MIM command above. If it still downloads `mmcv-*.tar.gz`, there is
no matching OpenMMLab wheel for your current combination. Then either:

- build MMCV from source on Python 3.13 after installing Visual Studio Build
  Tools and downgrading setuptools, or
- use Python 3.10 for the OpenMMLab environment while keeping this project code
  compatible with Python 3.13.

Source-build workaround to get past the `pkg_resources` error:

```powershell
python -m pip install "setuptools<81" wheel ninja
python -m pip install --no-build-isolation "mmcv>=2.0.0,<2.2.0"
```

This only fixes the packaging error. A full MMCV source build may still fail if
MSVC, CUDA, or the PyTorch/CUDA versions do not match.

## Data Pipeline

Put official NTU RGB+D archive files in:

```text
data/raw_archives/
  nturgbd_rgb_s001.zip
  nturgbd_rgb_s002.zip
  ...
  nturgbd_rgb_s032.zip
```

Then run the full pipeline with Python 3.10:

```powershell
py -3.10 main.py
```

By default, `--device auto` prefers `cuda:0` when PyTorch can see a GPU, then
falls back to CPU.

The RTMW and RTMDet checkpoints are downloaded once into:

```text
models/rtmw-dw-x-l_simcc-cocktail14_270e-384x288-20231122.pth
models/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth
```

Download checkpoints only:

```powershell
py -3.10 -m ntu_rtmw.download
```

Require all 32 archives before starting:

```powershell
py -3.10 main.py --require-all-archives
```

It will:

- download RTMW and RTMDet weights into `models/`
- extract one archive at a time into `data/extracted/`
- extract RTMW skeletons from that archive into `data/skeletons_rtmw/`
- delete that archive's expanded folder, leaving the original `.zip`
- write NTU protocol manifests into `data/processed/`

This low-storage mode is the default. At most one archive is expanded while the
pipeline is running, so `data/extracted/` should not keep growing. The generated
`.npz` skeleton files are kept for downstream action recognition or analysis.

Keep expanded videos after processing:

```powershell
py -3.10 main.py --keep-extracted
```

Use the old mode that extracts every archive first:

```powershell
py -3.10 main.py --extract-all-first
```

When CPU is pinned but GPU use is low, keep one worker and try batching frames:

```powershell
py -3.10 main.py --pose-batch-size 4 --cpu-threads 8
```

Raise `--pose-batch-size` to `8` if VRAM is comfortable. Lower
`--cpu-threads` to `4` if the machine feels overloaded. Use multiple extraction
workers only when CPU still has headroom:

```powershell
py -3.10 main.py --workers 2
```

Each worker loads its own RTMDet and RTMW models, so CPU and GPU memory use both
increase with the worker count. On Linux with CUDA, parallel extraction uses the
`spawn` multiprocessing start method so each worker can initialize CUDA safely.
If a multi-worker run is unstable, first fall back to `--workers 1` and increase
`--pose-batch-size` instead.

Show skeleton extraction live for one video:

```powershell
py -3.10 main.py --limit 1 --show-skeleton
```

Save skeleton visualization previews:

```powershell
py -3.10 main.py --limit 1 --visualize-dir data\visualizations
```

Re-extract existing skeleton files with the accurate detector path:

```powershell
py -3.10 main.py --limit 1 --overwrite --visualize-dir data\visualizations_310
```

Open a small skeleton preview window:

```powershell
py -3.10 preview.py
py -3.10 preview.py --kpt-thr 0.35 --temporal-min-frames 4 --temporal-min-keypoints 10
py -3.10 preview.py --person-match-distance 220 --person-hold-frames 3
py -3.10 preview.py --no-direct
```

By default, preview draws saved `.npz` skeletons directly on the original RGB
video while playing, without generating a fused preview `.avi`. If a skeleton
`.npz` has not been generated yet, this command builds it first. Preview
generation defaults to `--device auto`, which prefers CUDA when available and
falls back to CPU. On CPU, the preview command limits compute threads to `4` by
default so opening a preview does not take over the whole machine. Use
`--cpu-threads 2` for a lighter preview build, or `--device cuda:0` to force a
GPU.

The preview window auto-plays the next video when one clip ends. It also has
`Prev` and `Next` buttons. You can press `n` for next, `p` for previous, and
`q` or `Esc` to quit. Add `--loop-current` if you want one clip to replay
instead.
For close two-person interactions, direct preview keeps person slots stable
across frames by default. Tune `--person-match-distance` and
`--person-hold-frames` if identities still swap or skeletons briefly disappear.
Use `--regenerate` after changing temporal cleanup settings, because existing
preview `.avi` files already contain whatever skeletons were drawn earlier.
Use `--no-direct` only when you specifically want to generate and play a cached
fused preview `.avi`.

Open realtime RTMW skeletons from a local camera:

```powershell
py -3.10 camera.py
```

The camera path uses YOLOv8n for person boxes by default, then sends those
boxes to RTMW for whole-body keypoints. The YOLO checkpoint is kept at:

```text
models/yolov8n.pt
```

Useful camera options:

```powershell
py -3.10 camera.py --camera 1
py -3.10 camera.py --max-width 720 --max-height 405
py -3.10 camera.py --infer-every 3 --min-interval 0.2
py -3.10 camera.py --no-filter-output-to-bbox
py -3.10 camera.py --output-bbox-margin 0.1
py -3.10 camera.py --pose-input frame
py -3.10 camera.py --crop-margin 0.1
py -3.10 camera.py --det-backend mmdet
py -3.10 camera.py --temporal-min-frames 3 --temporal-min-keypoints 8
```

The camera window is responsive while RTMW runs in a background thread. On CPU,
pose updates are slower than the camera preview; use `--infer-every` or
`--min-interval` to reduce load. The camera output filters RTMW keypoints
outside the detected person box by default: those coordinates become `NaN` and
their scores become `0`. Add `--no-filter-output-to-bbox` only when you want to
inspect raw RTMW drift. With YOLO, the default `--pose-input crop` explicitly
crops each detected person box, runs RTMW on the crop, then offsets keypoints
back into the full camera frame.
Increase `--temporal-min-frames` or `--temporal-min-keypoints` if you want the
camera display to hide more one-frame skeleton flashes.

## Single-Step Commands

Process the standard NTU RGB video names ending in `_rgb.avi`:

```powershell
py -3.10 -m ntu_rtmw.extract --input D:\datasets\nturgbd_rgb --output D:\datasets\ntu_rtmw_skeletons --device cuda:0
```

Run several videos in parallel:

```powershell
py -3.10 -m ntu_rtmw.extract --input D:\datasets\nturgbd_rgb --output D:\datasets\ntu_rtmw_skeletons --device cuda:0 --workers 2
```

If CPU is already saturated, prefer a single worker with batched inference:

```powershell
py -3.10 -m ntu_rtmw.extract --input D:\datasets\nturgbd_rgb --output D:\datasets\ntu_rtmw_skeletons --device cuda:0 --pose-batch-size 4 --cpu-threads 8
```

Smoke test on one video:

```powershell
py -3.10 -m ntu_rtmw.extract --input D:\datasets\nturgbd_rgb --output outputs\rtmw --device cuda:0 --limit 1
```

Smoke test with live skeleton window:

```powershell
py -3.10 -m ntu_rtmw.extract --input D:\datasets\nturgbd_rgb --output outputs\rtmw --limit 1 --show-skeleton
```

Process every common video file recursively:

```powershell
py -3.10 -m ntu_rtmw.extract --input D:\datasets\nturgbd_rgb --output outputs\rtmw --all-videos
```

## Output

Each `.npz` contains:

- `keypoints`: shape `(frames, max_persons, 133, 2)`, pixel-space `x, y`
- `scores`: shape `(frames, max_persons, 133)`
- `bboxes`: shape `(frames, max_persons, 4)`, `x1, y1, x2, y2`
- `bbox_scores`: shape `(frames, max_persons)`
- `frame_indices`: frame indices emitted by the video reader
- `metadata`: JSON string with NTU filename fields when available

Missing people/keypoints are padded with `NaN` coordinates and zero scores.
By default, keypoints outside the detected person box or outside the video frame
are also written as `NaN` with score `0`, which avoids stray skeleton nodes in
low-confidence poses such as squats. Use `--no-filter-output-to-bbox` to inspect
raw RTMW drift, or `--output-bbox-margin 0.1` to allow a small margin around the
detected box.
The extractor also applies temporal cleanup: a keypoint must appear for at
least two consecutive frames, one-frame position jumps over 150 pixels are
removed, and person detections with fewer than five valid body keypoints are
hidden. Tune this with `--temporal-min-frames`, `--temporal-max-jump`, and
`--temporal-min-keypoints` if you need stricter or looser previews.

## Notes

The default detector is `rtmdet_tiny_8xb32-300e_coco`, restricted to COCO
person category by `det_cat_ids=[0]`. Use `--det-model whole_image` only for
the lower-accuracy compatibility path.

Useful options:

```powershell
py -3.10 main.py --help
```
