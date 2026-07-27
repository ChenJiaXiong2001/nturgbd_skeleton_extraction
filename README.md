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

Run CUDA extraction and use otherwise-idle CPU cores for additional videos from
the same input set:

```powershell
py -3.10 main.py --device cuda:0 --workers 4 --pose-batch-size 8 --cpu-workers 8 --cpu-worker-threads 4 --cpu-pose-batch-size 1 --scan-workers 32
```

`--workers` remains the number of CUDA workers. `--cpu-workers` adds CPU-only
workers. All workers consume a shared dynamic queue, so faster CUDA workers
automatically process more videos and no input is assigned twice. Each worker
loads its own RTMDet and RTMW models; start with 4 to 8 CPU workers and watch
system RAM and storage throughput before increasing the count.

With `--skip-existing` (the default in `main.py`), output existence checks run
in parallel before inference. `--scan-workers` controls this file-system scan;
32 is a suitable starting point for SSD storage. Existing files are summarized
instead of logged one per line.

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
py -3.10 preview.py --list
py -3.10 preview.py --index 5
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

The preview playlist includes videos inside `data/raw_archives/*.zip`; it does
not expand a whole archive. It copies only the selected AVI into a temporary
cache, preloads the next AVI in the background, and removes old cached AVIs as
you move through the playlist. Use `--no-preload` to keep only the current AVI.

The preview window auto-plays the next video when one clip ends. Its controls
provide previous/next, a searchable `Select` file picker, pause/play, replay,
speed adjustment, and a clickable seek bar. Keyboard shortcuts are `n`/`p`,
`g` (select), Space, `r`, `-`/`+`, and `q` or `Esc`. Add `--loop-current` if
you want one clip to replay instead.

The right sidebar groups large playlists by archive and blocks of 100 files,
for example `s001 1-100` and `s001 101-200`. Use the section buttons to move
between blocks, the mouse wheel to scroll inside a block, and click a filename
to load it. Only indexed `*_rgb.avi` members can be opened; rapid duplicate
clicks are ignored and ZIP extraction is serialized to protect the cache.
Use `--list` to print numbered videos, then `--index N` to start previewing
from the Nth video.
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

Compare YOLO and RTMDet person boxes while keeping the downstream pose path
fixed to explicit person crops and RTMW-L:

```powershell
py -3.10 compare_backends.py `
  --video data\extracted\s001\nturgb+d_rgb\S001C001P001R001A055_rgb.avi `
  --output-dir data\comparisons\A055_yolo_vs_rtmdet_rtmw_crop_all_frames `
  --frames 95 --stride 1 `
  --variants rtmdet_crop yolo_crop --reference rtmdet_crop --video-preview
```

Both variants use the same crop and RTMW-L pose inference code. The only
changed variable is whether person boxes come from `models/yolo26x.pt` or the
RTMDet detector used by the standard OpenMMLab pipeline.
The comparison command supports striding for quick diagnostics, but skeleton
files used for training are produced by the main extraction pipeline without
frame sampling. Missing detections remain as zero/NaN entries at their original
frame positions rather than shortening the sequence.

The camera path uses YOLO26-X for person boxes by default, then sends those
boxes to RTMW for whole-body keypoints. The YOLO checkpoint is kept at:

```text
models/yolo26x.pt
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

## Skeleton Quality Check and Retry

All default paths are derived from the repository location, so the normal
check needs no path arguments on either Windows or Linux:

```bash
python check_skeletons.py
```

The checker prints the number of discovered files and live progress, throughput,
elapsed time, and estimated remaining time. For HDDs or network-mounted data,
start with `--workers 2`; using too many workers can reduce throughput because
compressed `.npz` files compete for disk reads.

This automatically uses:

```text
data/skeletons_rtmw
data/extracted
data/quality_reports
data/skeletons_rtmw_retry
```

An explicit input path can still be provided when skeletons are stored outside
the project:

```powershell
py -3.10 check_skeletons.py data\skeletons_rtmw
```

The default rules expect one person for ordinary actions and two people for
NTU interaction actions A050-A060 and A106-A120. The checker validates the array structure, continuous
frame indices, expected-person recall, longest missing-person run, valid body
keypoints, unexpected extra people in single-person actions, normalized
frame-to-frame body/slot jumps, and duplicate NTU sample
IDs. When duplicates exist, the passing RTMDet result with the highest quality
score is preferred and the extra copies fail the check. A failed scan exits
with status code 1 so it can stop a training pipeline. Add `--no-fail-exit` when
only a report is needed.

For single-person actions, a second active skeleton is allowed for at most 10%
of frames and no more than 10 consecutive frames. Override these safeguards
with `--max-unexpected-person-rate` and `--max-unexpected-person-run`. New
extraction and retry runs also retain at most one active track for single-person
actions while preserving two array slots for downstream shape compatibility.

Useful threshold overrides:

```powershell
py -3.10 check_skeletons.py data\skeletons_rtmw `
  --two-person-min-recall 0.9 `
  --max-missing-run 8 `
  --max-large-jump-rate 0.02
```

Custom two-person labels can be supplied as ranges or individual actions, for
example `--two-person-actions 50-60,106-120`.

Failed files can be re-extracted from their source videos into a separate
directory. With the standard project layout, only the action flags are needed:

```bash
python check_skeletons.py \
  --reextract-failed \
  --retry-profile relaxed \
  --device cuda:0
```

Expanded videos are not required. If a recorded video path is missing, the
checker automatically finds the corresponding
`data/raw_archives/nturgbd_rgb_sXXX.zip`, copies only that failed sample's AVI
to a temporary location, retries skeleton extraction, and immediately deletes
the temporary AVI. It never expands a complete NTU archive. On Ubuntu, put the
single temporary AVI in RAM instead of persistent storage with:

```bash
python check_skeletons.py \
  --reextract-failed \
  --archives-dir data/raw_archives \
  --retry-temp-dir /dev/shm \
  --retry-profile relaxed \
  --device cuda:0
```

Archive fallback is enabled by default. Use `--no-retry-from-archives` to
require already expanded videos under `data/extracted`.

For close two-person actions where RTMDet-tiny still misses the second person,
retry the same failed samples with YOLO26-X person boxes and RTMW-L pose crops.
Use a separate output directory so the RTMDet pilot remains available:

```bash
python check_skeletons.py \
  --reextract-failed \
  --retry-det-backend yolo26 \
  --retry-yolo-model models/yolo26x.pt \
  --retry-yolo-conf 0.15 \
  --retry-yolo-iou 0.70 \
  --retry-yolo-imgsz 960 \
  --retry-crop-margin 0.10 \
  --retry-output data/skeletons_rtmw_retry_yolo \
  --retry-temp-dir /dev/shm \
  --retry-limit 100 \
  --retry-profile relaxed \
  --device cuda:0
```

YOLO26-X changes only person detection. RTMW-L remains the pose model, and the
same quality thresholds decide whether a retry passes.

To complete checking, YOLO26 repair, replacement, and failed-file cleanup in
one command, use:

```bash
python check_skeletons.py data/skeletons_rtmw \
  --repair-failed-yolo26 \
  --archives-dir data/raw_archives \
  --retry-temp-dir /dev/shm \
  --retry-output data/skeletons_rtmw_retry_yolo \
  --retry-gpu-workers 1 \
  --retry-cpu-workers 2 \
  --retry-cpu-worker-threads 4 \
  --device cuda:0 \
  --workers 2 \
  --no-fail-exit
```

This shortcut enables `--reextract-failed`, selects the YOLO26 detector,
replaces an original when its retry passes, and deletes the selected original
when it still fails. It defaults to one CUDA retry worker plus one CPU retry
worker on a shared dynamic queue. The example raises the CPU count to two;
start there because every process loads its own YOLO26-X and RTMW models.
`--workers` controls only parallel NPZ quality checks, while
`--retry-gpu-workers` and `--retry-cpu-workers` control inference. Archive
videos are materialized one at a time inside each worker and removed
immediately; `/dev/shm` keeps those temporary AVIs in RAM. Replaced originals are backed up under
`data/skeletons_rtmw_retry_yolo/original_backup`. Add `--retry-limit 100` for a
pilot run; only those 100 selected failures can be replaced or deleted, and all
unselected files remain untouched.

To delete failed files without attempting re-extraction, use
`--delete-failed`. This is irreversible for originals and should normally be
used only after preserving the JSON/CSV quality report or the source videos.

The equivalent command with explicit Windows paths is:

```powershell
py -3.10 check_skeletons.py data\skeletons_rtmw `
  --reextract-failed `
  --video-root data\extracted `
  --retry-output data\skeletons_rtmw_retry `
  --retry-profile relaxed `
  --device cuda:0
```

The `relaxed` retry profile keeps a 10% bbox margin and disables destructive
temporal removal, which helps distinguish detector loss from post-processing
loss. It does not overwrite the original files. Add `--replace-if-better` only
when passing retry results should replace failed originals; each replaced
original is first copied under `original_backup` in the retry directory. Here
“retry” means running skeleton extraction again, not fine-tuning the RTMW model.

## Notes

The default detector is `rtmdet_tiny_8xb32-300e_coco`, restricted to COCO
person category by `det_cat_ids=[0]`. Use `--det-model whole_image` only for
the lower-accuracy compatibility path.

Useful options:

```powershell
py -3.10 main.py --help
```
