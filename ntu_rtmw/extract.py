import argparse
import importlib.util
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

from .compat import patch_runtime
from .constants import (
    NTU_NAME_RE,
    RTMDET_CONFIG,
    RTMDET_WEIGHTS_PATH,
    RTMW_CONFIG,
    RTMW_WEIGHTS_PATH,
    VIDEO_EXTENSIONS,
)
from .device import resolve_device
from .download import ensure_rtmdet_weights, ensure_rtmw_weights

np = None
CPU_THREADS_CONFIGURED = False
OPENMMLAB_MODULES = ["mmengine", "mmcv", "mmdet", "mmpose"]
BODY_EDGES = [
    (5, 7), (7, 9), (6, 8), (8, 10), (5, 6),
    (5, 11), (6, 12), (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16), (0, 1), (0, 2), (1, 3),
    (2, 4), (0, 5), (0, 6),
]


def ensure_supported_python():
    if sys.version_info < (3, 10):
        raise SystemExit("Use Python 3.10 or newer.")


def require_numpy():
    global np
    if np is not None:
        return
    try:
        import numpy as numpy_module
    except ImportError as exc:
        raise SystemExit("Missing numpy. Install project requirements first.") from exc
    np = numpy_module


def require_mmpose():
    patch_runtime()
    try:
        from mmpose.apis import MMPoseInferencer
    except ImportError as exc:
        raise SystemExit(openmmlab_install_message()) from exc
    return MMPoseInferencer


def missing_openmmlab_modules():
    return [name for name in OPENMMLAB_MODULES if importlib.util.find_spec(name) is None]


def openmmlab_install_message():
    missing = missing_openmmlab_modules()
    if not missing:
        return "OpenMMLab import failed. Run: py -3.13 check_env.py"
    return (
        "Missing OpenMMLab modules: {}\n"
        "Install them before RTMW extraction:\n"
        "  python -m pip install -r requirements.txt\n"
        "  python -m mim install \"mmcv>=2.0.0,<2.2.0\"\n"
        "  python -m mim install \"mmdet>=3.2.0\"\n"
        "  python -m mim install \"mmpose>=1.3.0\"\n"
        "If mmcv downloads mmcv-*.tar.gz, there is no matching wheel "
        "for this Python/PyTorch/CUDA combination."
    ).format(", ".join(missing))


def ensure_openmmlab_ready():
    missing = missing_openmmlab_modules()
    if missing:
        print(openmmlab_install_message(), flush=True)
        raise SystemExit(1)
    patch_runtime()
    try:
        from mmpose.apis import MMPoseInferencer  # noqa: F401
    except Exception as exc:
        print("OpenMMLab import failed: {}".format(exc), flush=True)
        raise SystemExit(1) from exc


def find_videos(root, pattern="*_rgb.avi", all_videos=False):
    root = Path(root)
    if root.is_file():
        return [root]
    if all_videos:
        return sorted(p for p in root.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)
    return sorted(p for p in root.rglob(pattern) if p.is_file())


def ntu_metadata(path):
    match = NTU_NAME_RE.search(Path(path).stem)
    return {k: int(v) for k, v in match.groupdict().items()} if match else {}


def output_path(video, input_root, output_root):
    video = Path(video)
    input_root = Path(input_root)
    output_root = Path(output_root)
    if input_root.is_dir():
        return output_root / video.relative_to(input_root).with_suffix(".npz")
    return output_root / "{}.npz".format(video.stem)


def path_exists(path):
    return Path(path).exists()


def plan_video_jobs(videos, args):
    """Build pending jobs, checking existing outputs concurrently when requested."""
    total = len(videos)
    candidates = []
    for index, video in enumerate(videos, 1):
        out = output_path(video, args.input, args.output)
        candidates.append((index, total, str(video), str(out)))

    if not args.skip_existing:
        return [], candidates

    scan_workers = min(max(1, int(getattr(args, "scan_workers", 32) or 32)), total)
    print("checking existing outputs with {} threads".format(scan_workers), flush=True)
    with ThreadPoolExecutor(max_workers=scan_workers) as executor:
        existing = list(executor.map(path_exists, (job[3] for job in candidates)))

    outputs = [Path(job[3]) for job, exists in zip(candidates, existing) if exists]
    jobs = [job for job, exists in zip(candidates, existing) if not exists]
    print("existing outputs: {}; missing: {}".format(len(outputs), len(jobs)), flush=True)
    return outputs, jobs


def unwrap_predictions(result):
    predictions = result.get("predictions", [])
    if isinstance(predictions, list) and len(predictions) == 1 and isinstance(predictions[0], list):
        return predictions[0]
    return predictions if isinstance(predictions, list) else []


def arr(value, shape):
    if value is None:
        return np.full(shape, np.nan, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    return value if value.size else np.full(shape, np.nan, dtype=np.float32)


def clean_instance(raw):
    keypoints = arr(raw.get("keypoints"), (0, 2))
    if keypoints.ndim == 3:
        keypoints = keypoints[0]
    scores = arr(raw.get("keypoint_scores"), (keypoints.shape[0],))
    if scores.ndim == 2:
        scores = scores[0]
    bbox = arr(raw.get("bbox", raw.get("bboxes")), (4,)).reshape(-1)
    bbox = bbox[:4] if bbox.size >= 4 else np.full((4,), np.nan, dtype=np.float32)
    bbox_score = raw.get("bbox_score", raw.get("bbox_scores", 1.0))
    bbox_score = float(np.asarray(bbox_score).reshape(-1)[0])
    mean_score = float(np.nanmean(scores)) if scores.size else 0.0
    if np.isnan(bbox).any():
        center = np.array([np.nan, np.nan], dtype=np.float32)
        area = 0.0
    else:
        center = np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)
        area = max(0.0, float(bbox[2] - bbox[0])) * max(0.0, float(bbox[3] - bbox[1]))
    return {
        "keypoints": keypoints.astype(np.float32),
        "scores": scores.astype(np.float32),
        "bbox": bbox.astype(np.float32),
        "bbox_score": bbox_score,
        "mean_score": mean_score,
        "center": center,
        "area": area,
    }


def select_instances(raw_instances, bbox_thr, kpt_thr):
    items = [clean_instance(x) for x in raw_instances if isinstance(x, dict)]
    items = [x for x in items if x["bbox_score"] >= bbox_thr and x["mean_score"] >= kpt_thr]
    return sorted(items, key=lambda x: (x["bbox_score"], x["mean_score"], x["area"]), reverse=True)


def assign_slots(items, previous, max_persons, max_distance):
    slots = [None] * max_persons
    unmatched = list(range(len(items)))
    if previous is not None and max_distance > 0:
        pairs = []
        for det_idx in unmatched:
            center = items[det_idx]["center"]
            if np.isnan(center).any():
                continue
            for slot_idx, prev in enumerate(previous):
                if not np.isnan(prev).any():
                    pairs.append((float(np.linalg.norm(center - prev)), det_idx, slot_idx))
        for distance, det_idx, slot_idx in sorted(pairs):
            if distance <= max_distance and slots[slot_idx] is None and det_idx in unmatched:
                slots[slot_idx] = items[det_idx]
                unmatched.remove(det_idx)
    for slot_idx in range(max_persons):
        if slots[slot_idx] is None and unmatched:
            slots[slot_idx] = items[unmatched.pop(0)]
    centers = np.full((max_persons, 2), np.nan, dtype=np.float32)
    for i, item in enumerate(slots):
        if item is not None:
            centers[i] = item["center"]
    return slots, centers


def copy_pose_arrays(arrays):
    return {
        "keypoints": arrays["keypoints"].copy(),
        "scores": arrays["scores"].copy(),
        "bboxes": arrays["bboxes"].copy(),
        "bbox_scores": arrays["bbox_scores"].copy(),
        "frame_indices": arrays["frame_indices"].copy(),
    }


def invalidate_keypoints(arrays, invalid):
    arrays["keypoints"][invalid] = np.nan
    arrays["scores"][invalid] = 0


def filter_arrays_to_bboxes(arrays, margin_ratio):
    require_numpy()
    filtered = copy_pose_arrays(arrays)
    keypoints = filtered["keypoints"]
    bboxes = filtered["bboxes"]
    invalid = np.zeros(filtered["scores"].shape, dtype=bool)
    for frame_idx in range(keypoints.shape[0]):
        for person_idx in range(keypoints.shape[1]):
            bbox = bboxes[frame_idx, person_idx]
            if np.isnan(bbox).any():
                invalid[frame_idx, person_idx] = True
                continue
            x1, y1, x2, y2 = bbox.astype(float)
            width = max(1.0, x2 - x1)
            height = max(1.0, y2 - y1)
            margin_x = width * margin_ratio
            margin_y = height * margin_ratio
            x1 -= margin_x
            x2 += margin_x
            y1 -= margin_y
            y2 += margin_y
            points = keypoints[frame_idx, person_idx]
            invalid[frame_idx, person_idx] = (
                (points[:, 0] < x1)
                | (points[:, 0] > x2)
                | (points[:, 1] < y1)
                | (points[:, 1] > y2)
                | np.isnan(points[:, 0])
                | np.isnan(points[:, 1])
            )
    invalidate_keypoints(filtered, invalid)
    return filtered


def filter_arrays_to_frame(arrays, width, height):
    require_numpy()
    filtered = copy_pose_arrays(arrays)
    points = filtered["keypoints"]
    invalid = (
        (points[..., 0] < 0)
        | (points[..., 0] >= width)
        | (points[..., 1] < 0)
        | (points[..., 1] >= height)
        | np.isnan(points[..., 0])
        | np.isnan(points[..., 1])
    )
    invalidate_keypoints(filtered, invalid)
    return filtered


def remove_short_valid_runs(valid, min_frames):
    if min_frames <= 1 or valid.size == 0:
        return valid
    cleaned = valid.copy()
    frames, persons, keypoints = valid.shape
    for person_idx in range(persons):
        for keypoint_idx in range(keypoints):
            start = None
            for frame_idx in range(frames + 1):
                active = frame_idx < frames and valid[frame_idx, person_idx, keypoint_idx]
                if active and start is None:
                    start = frame_idx
                elif not active and start is not None:
                    if frame_idx - start < min_frames:
                        cleaned[start:frame_idx, person_idx, keypoint_idx] = False
                    start = None
    return cleaned


def remove_one_frame_jumps(valid, points, max_jump):
    require_numpy()
    if max_jump <= 0 or valid.shape[0] < 3:
        return valid
    cleaned = valid.copy()
    for frame_idx in range(1, valid.shape[0] - 1):
        prev_valid = valid[frame_idx - 1]
        curr_valid = valid[frame_idx]
        next_valid = valid[frame_idx + 1]
        common = prev_valid & curr_valid & next_valid
        if not common.any():
            continue
        prev_dist = np.linalg.norm(points[frame_idx] - points[frame_idx - 1], axis=-1)
        next_dist = np.linalg.norm(points[frame_idx] - points[frame_idx + 1], axis=-1)
        bridge_dist = np.linalg.norm(points[frame_idx - 1] - points[frame_idx + 1], axis=-1)
        spike = common & (prev_dist > max_jump) & (next_dist > max_jump) & (bridge_dist <= max_jump)
        cleaned[frame_idx, spike] = False
    return cleaned


def hide_sparse_people(arrays, valid, min_keypoints):
    if min_keypoints <= 0:
        return valid
    cleaned = valid.copy()
    body_count = cleaned[:, :, :17].sum(axis=2)
    sparse = body_count < min_keypoints
    if sparse.any():
        cleaned[sparse] = False
        arrays["bboxes"][sparse] = np.nan
        arrays["bbox_scores"][sparse] = 0
    return cleaned


def temporal_cleanup_arrays(arrays, min_frames, max_jump, min_keypoints, kpt_thr):
    require_numpy()
    cleaned = copy_pose_arrays(arrays)
    points = cleaned["keypoints"]
    valid = (cleaned["scores"] >= kpt_thr) & ~np.isnan(points[..., 0]) & ~np.isnan(points[..., 1])
    valid = remove_one_frame_jumps(valid, points, max_jump)
    valid = remove_short_valid_runs(valid, min_frames)
    valid = hide_sparse_people(cleaned, valid, min_keypoints)
    invalidate_keypoints(cleaned, ~valid)
    return cleaned


def video_size(video):
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height


def postprocess_arrays(arrays, video, args):
    require_numpy()
    processed = arrays
    if getattr(args, "filter_output_to_bbox", True):
        processed = filter_arrays_to_bboxes(processed, getattr(args, "output_bbox_margin", 0.0))
    if getattr(args, "filter_output_to_frame", True):
        size = video_size(video)
        if size is not None:
            processed = filter_arrays_to_frame(processed, size[0], size[1])
    processed = temporal_cleanup_arrays(
        processed,
        getattr(args, "temporal_min_frames", 1),
        getattr(args, "temporal_max_jump", 0.0),
        getattr(args, "temporal_min_keypoints", 0),
        getattr(args, "kpt_thr", 0.1),
    )
    return processed


class TemporalDisplayFilter:
    def __init__(self, min_frames=2, max_jump=150.0, min_keypoints=5, kpt_thr=0.1):
        require_numpy()
        self.min_frames = max(1, int(min_frames or 1))
        self.max_jump = float(max_jump or 0.0)
        self.min_keypoints = max(0, int(min_keypoints or 0))
        self.kpt_thr = float(kpt_thr)
        self.streaks = None
        self.previous_points = None
        self.previous_valid = None

    def _ensure_shape(self, keypoints):
        shape = keypoints.shape[:-1]
        if self.streaks is None or self.streaks.shape != shape:
            self.streaks = np.zeros(shape, dtype=np.int32)
            self.previous_points = np.full(keypoints.shape, np.nan, dtype=np.float32)
            self.previous_valid = np.zeros(shape, dtype=bool)

    def apply_arrays(self, keypoints, scores, bboxes, bbox_scores):
        self._ensure_shape(keypoints)
        raw_points = keypoints.copy()
        raw_valid = (scores >= self.kpt_thr) & ~np.isnan(raw_points[..., 0]) & ~np.isnan(raw_points[..., 1])
        stable_valid = raw_valid.copy()
        if self.max_jump > 0 and self.previous_valid is not None:
            compared = raw_valid & self.previous_valid
            jumps = np.zeros(raw_valid.shape, dtype=bool)
            if compared.any():
                dist = np.linalg.norm(raw_points - self.previous_points, axis=-1)
                jumps = compared & (dist > self.max_jump)
            stable_valid &= ~jumps

        self.streaks = np.where(stable_valid, self.streaks + 1, 0)
        visible = stable_valid & (self.streaks >= self.min_frames)

        update = stable_valid
        self.previous_points[update] = raw_points[update]
        self.previous_valid = update

        filtered_keypoints = keypoints.copy()
        filtered_scores = scores.copy()
        filtered_bboxes = bboxes.copy()
        filtered_bbox_scores = bbox_scores.copy()
        if self.min_keypoints > 0:
            body_count = visible[:, :17].sum(axis=1)
            sparse = body_count < self.min_keypoints
            visible[sparse] = False
            filtered_bboxes[sparse] = np.nan
            filtered_bbox_scores[sparse] = 0
        filtered_keypoints[~visible] = np.nan
        filtered_scores[~visible] = 0
        return filtered_keypoints, filtered_scores, filtered_bboxes, filtered_bbox_scores

    def apply_latest(self, latest):
        if latest is None:
            return None
        filtered = dict(latest)
        k, s, b, bs = self.apply_arrays(
            latest["keypoints"],
            latest["scores"],
            latest["bboxes"],
            latest["bbox_scores"],
        )
        filtered["keypoints"] = k
        filtered["scores"] = s
        filtered["bboxes"] = b
        filtered["bbox_scores"] = bs
        return filtered


def infer_video(inferencer, video, args):
    require_numpy()
    keypoints, scores, bboxes, bbox_scores = [], [], [], []
    previous = None
    num_kpts = 133
    visualizer = SkeletonVisualizer(video, args)
    display_filter = TemporalDisplayFilter(
        getattr(args, "temporal_min_frames", 1),
        getattr(args, "temporal_max_jump", 0.0),
        getattr(args, "temporal_min_keypoints", 0),
        args.kpt_thr,
    )
    kwargs = {"show": False, "return_vis": False, "draw_bbox": False, "kpt_thr": args.kpt_thr}
    pose_batch_size = max(1, int(getattr(args, "pose_batch_size", 1) or 1))
    if pose_batch_size > 1:
        kwargs["batch_size"] = pose_batch_size
    try:
        for frame_idx, result in enumerate(inferencer(str(video), **kwargs)):
            items = select_instances(unwrap_predictions(result), args.bbox_thr, args.kpt_thr)
            for item in items:
                if item["keypoints"].shape[0]:
                    num_kpts = int(item["keypoints"].shape[0])
                    break
            slots, previous = assign_slots(items, previous, args.max_persons, args.tracking_distance)
            k = np.full((args.max_persons, num_kpts, 2), np.nan, dtype=np.float32)
            s = np.zeros((args.max_persons, num_kpts), dtype=np.float32)
            b = np.full((args.max_persons, 4), np.nan, dtype=np.float32)
            bs = np.zeros((args.max_persons,), dtype=np.float32)
            for person_idx, item in enumerate(slots):
                if item is None:
                    continue
                n = min(num_kpts, item["keypoints"].shape[0])
                k[person_idx, :n] = item["keypoints"][:n, :2]
                s[person_idx, :n] = item["scores"][:n]
                b[person_idx] = item["bbox"]
                bs[person_idx] = item["bbox_score"]
            keypoints.append(k)
            scores.append(s)
            bboxes.append(b)
            bbox_scores.append(bs)
            visualizer.draw_live(k, s, b, bs, args.kpt_thr, display_filter)
            if (frame_idx + 1) % 100 == 0:
                print("  {} frames".format(frame_idx + 1), flush=True)
    finally:
        visualizer.close()
    if not keypoints:
        return {
            "keypoints": np.empty((0, args.max_persons, 133, 2), dtype=np.float32),
            "scores": np.empty((0, args.max_persons, 133), dtype=np.float32),
            "bboxes": np.empty((0, args.max_persons, 4), dtype=np.float32),
            "bbox_scores": np.empty((0, args.max_persons), dtype=np.float32),
            "frame_indices": np.empty((0,), dtype=np.int32),
        }
    arrays = {
        "keypoints": np.stack(keypoints),
        "scores": np.stack(scores),
        "bboxes": np.stack(bboxes),
        "bbox_scores": np.stack(bbox_scores),
        "frame_indices": np.arange(len(keypoints), dtype=np.int32),
    }
    arrays = postprocess_arrays(arrays, video, args)
    visualizer.draw_sequence(arrays, args.kpt_thr)
    return arrays


class SkeletonVisualizer:
    def __init__(self, video, args):
        self.video = video
        self.show = getattr(args, "show_skeleton", False)
        self.out_dir = Path(args.visualize_dir) if getattr(args, "visualize_dir", None) else None
        self.cap = None
        self.writer = None
        self.preview_path = None
        self.preview_written = False
        self.window = "RTMW skeleton"
        if not self.show and self.out_dir is None:
            return
        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            print("  warning: cannot open video for visualization: {}".format(video), flush=True)
            self.cap = None
            return
        if self.out_dir is not None:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.out_path = self.out_dir / "{}_skeleton.avi".format(Path(video).stem)
            self.preview_path = self.out_dir / "{}_skeleton.jpg".format(Path(video).stem)
            print("  visualization {}".format(self.out_path), flush=True)

    def draw_live(self, keypoints, scores, bboxes, bbox_scores, kpt_thr, display_filter):
        if self.cap is None or not self.show:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        keypoints, scores, bboxes, bbox_scores = display_filter.apply_arrays(keypoints, scores, bboxes, bbox_scores)
        draw_skeleton(frame, keypoints, scores, bboxes, bbox_scores, kpt_thr, self.cv2)
        self.cv2.imshow(self.window, frame)
        if self.cv2.waitKey(1) & 0xFF == ord("q"):
            self.show = False
            self.cv2.destroyWindow(self.window)

    def draw_sequence(self, arrays, kpt_thr):
        if self.out_dir is None or self.cap is None:
            return
        self.cap.release()
        self.cap = self.cv2.VideoCapture(str(self.video))
        if not self.cap.isOpened():
            return
        fourcc = self.cv2.VideoWriter_fourcc(*"MJPG")
        self.writer = self.cv2.VideoWriter(str(self.out_path), fourcc, self.fps, (self.width, self.height))
        for frame_idx in range(arrays["keypoints"].shape[0]):
            ok, frame = self.cap.read()
            if not ok:
                break
            draw_skeleton(
                frame,
                arrays["keypoints"][frame_idx],
                arrays["scores"][frame_idx],
                arrays["bboxes"][frame_idx],
                arrays["bbox_scores"][frame_idx],
                kpt_thr,
                self.cv2,
            )
            self.writer.write(frame)
            if self.preview_path is not None and not self.preview_written:
                self.cv2.imwrite(str(self.preview_path), frame)
                self.preview_written = True

    def close(self):
        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()
        if getattr(self, "show", False):
            self.cv2.destroyWindow(self.window)


def draw_skeleton(frame, keypoints, scores, bboxes, bbox_scores, kpt_thr, cv2):
    colors = [(0, 255, 0), (0, 180, 255)]
    for person_idx in range(keypoints.shape[0]):
        color = colors[person_idx % len(colors)]
        if bbox_scores[person_idx] > 0 and not np.isnan(bboxes[person_idx]).any():
            x1, y1, x2, y2 = bboxes[person_idx].astype(int).tolist()
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        pts = keypoints[person_idx]
        scr = scores[person_idx]
        for a, b in BODY_EDGES:
            if a >= len(pts) or b >= len(pts):
                continue
            if scr[a] < kpt_thr or scr[b] < kpt_thr:
                continue
            if np.isnan(pts[a]).any() or np.isnan(pts[b]).any():
                continue
            pa = tuple(pts[a].astype(int).tolist())
            pb = tuple(pts[b].astype(int).tolist())
            cv2.line(frame, pa, pb, color, 2, lineType=cv2.LINE_AA)
        for point, score in zip(pts, scr):
            if score >= kpt_thr and not np.isnan(point).any():
                cv2.circle(frame, tuple(point.astype(int).tolist()), 2, color, -1, lineType=cv2.LINE_AA)


def save_npz(path, video, arrays, args):
    require_numpy()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = ntu_metadata(video)
    meta.update({
        "video_name": Path(video).name,
        "video_path": str(video),
        "pose_model": args.pose2d,
        "pose_weights": args.pose2d_weights,
        "det_model": args.det_model,
        "keypoint_convention": "coco_wholebody_133",
        "max_persons": args.max_persons,
    })
    np.savez_compressed(path, metadata=json.dumps(meta, ensure_ascii=False), **arrays)


def build_inferencer(args):
    kwargs = {
        "pose2d": args.pose2d,
        "pose2d_weights": args.pose2d_weights,
        "det_model": args.det_model,
        "det_cat_ids": [0],
        "device": args.device,
    }
    if args.det_weights:
        kwargs["det_weights"] = args.det_weights
    return require_mmpose()(**kwargs)


def configure_cpu_threads(args):
    global CPU_THREADS_CONFIGURED
    threads = int(getattr(args, "cpu_threads", 0) or 0)
    if threads <= 0 or CPU_THREADS_CONFIGURED:
        return
    CPU_THREADS_CONFIGURED = True
    value = str(threads)
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = value
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(threads)
    except Exception:
        pass
    print("CPU threads per process: {}".format(threads), flush=True)


def effective_workers(args, pending_count):
    workers = max(1, int(getattr(args, "workers", 1) or 1))
    workers = min(workers, max(1, pending_count))
    if getattr(args, "show_skeleton", False) and workers > 1:
        print("show-skeleton is interactive; using workers=1", flush=True)
        return 1
    return workers


def split_jobs(jobs, workers):
    chunks = [[] for _ in range(workers)]
    for idx, job in enumerate(jobs):
        chunks[idx % workers].append(job)
    return [chunk for chunk in chunks if chunk]


def process_pool_context(args):
    device = str(getattr(args, "device", "") or "").lower()
    if device.startswith("cuda"):
        return multiprocessing.get_context("spawn")
    return None


def run_worker(worker_id, jobs, args):
    inferencer = build_inferencer(args)
    outputs = []
    for index, total, video, out in jobs:
        video = Path(video)
        out = Path(out)
        print("[{}/{}] worker {} {}".format(index, total, worker_id, video), flush=True)
        save_npz(out, video, infer_video(inferencer, video, args), args)
        print("  saved {}".format(out), flush=True)
        outputs.append(str(out))
    return outputs


def run_queued_worker(worker_id, worker_kind, job_queue, args):
    """Consume videos from a shared queue so faster devices take more work."""
    configure_cpu_threads(args)
    inferencer = build_inferencer(args)
    outputs = []
    while True:
        job = job_queue.get()
        if job is None:
            break
        index, total, video, out = job
        video = Path(video)
        out = Path(out)
        print("[{}/{}] {} worker {} {}".format(index, total, worker_kind, worker_id, video), flush=True)
        save_npz(out, video, infer_video(inferencer, video, args), args)
        print("  saved {}".format(out), flush=True)
        outputs.append(str(out))
    return outputs


def run_hybrid_workers(jobs, args, gpu_workers, cpu_workers):
    """Run CUDA and CPU inferencers together against one dynamic job queue."""
    gpu_workers = min(max(1, gpu_workers), len(jobs))
    cpu_workers = min(max(0, cpu_workers), max(0, len(jobs) - gpu_workers))
    if cpu_workers == 0:
        return None

    gpu_args = argparse.Namespace(**vars(args))
    cpu_args = argparse.Namespace(**vars(args))
    cpu_args.device = "cpu"
    cpu_args.cpu_threads = max(1, int(getattr(args, "cpu_worker_threads", 4) or 4))
    cpu_args.pose_batch_size = max(1, int(getattr(args, "cpu_pose_batch_size", 1) or 1))

    mp_context = multiprocessing.get_context("spawn")
    total_workers = gpu_workers + cpu_workers
    print(
        "hybrid workers: {} GPU + {} CPU (dynamic queue, spawn)".format(gpu_workers, cpu_workers),
        flush=True,
    )
    with mp_context.Manager() as manager:
        job_queue = manager.Queue()
        for job in jobs:
            job_queue.put(job)
        for _ in range(total_workers):
            job_queue.put(None)

        outputs = []
        with ProcessPoolExecutor(max_workers=total_workers, mp_context=mp_context) as executor:
            futures = []
            for worker_id in range(1, gpu_workers + 1):
                futures.append(executor.submit(run_queued_worker, worker_id, "GPU", job_queue, gpu_args))
            for worker_id in range(1, cpu_workers + 1):
                futures.append(executor.submit(run_queued_worker, worker_id, "CPU", job_queue, cpu_args))
            for future in as_completed(futures):
                outputs.extend(Path(path) for path in future.result())
    return outputs


def run(args):
    ensure_supported_python()
    configure_cpu_threads(args)
    args.device = resolve_device(args.device)
    print("device {}".format(args.device), flush=True)
    if str(args.pose2d_weights) == str(RTMW_WEIGHTS_PATH):
        args.pose2d_weights = str(RTMW_WEIGHTS_PATH if Path(RTMW_WEIGHTS_PATH).exists() else ensure_rtmw_weights())
    if args.det_model != "whole_image" and str(args.det_weights) == str(RTMDET_WEIGHTS_PATH):
        args.det_weights = str(RTMDET_WEIGHTS_PATH if Path(RTMDET_WEIGHTS_PATH).exists() else ensure_rtmdet_weights())
    videos = find_videos(args.input, args.pattern, args.all_videos)
    if args.limit is not None:
        videos = videos[: args.limit]
    if not videos:
        print("No videos found in {}".format(args.input), flush=True)
        return []
    print("RTMW extracting {} video(s)".format(len(videos)), flush=True)
    outputs, jobs = plan_video_jobs(videos, args)
    if not jobs:
        return outputs

    workers = effective_workers(args, len(jobs))
    cpu_workers = max(0, int(getattr(args, "cpu_workers", 0) or 0))
    if cpu_workers and str(args.device).lower().startswith("cuda") and not args.show_skeleton:
        hybrid_outputs = run_hybrid_workers(jobs, args, workers, cpu_workers)
        if hybrid_outputs is not None:
            outputs.extend(hybrid_outputs)
            return outputs
    elif cpu_workers:
        reason = "interactive visualization is enabled" if args.show_skeleton else "the primary device is not CUDA"
        print("CPU hybrid workers disabled because {}; using the primary workers only".format(reason), flush=True)

    if workers == 1:
        inferencer = build_inferencer(args)
        for i, total, video, out in jobs:
            video = Path(video)
            out = Path(out)
            print("[{}/{}] {}".format(i, total, video), flush=True)
            save_npz(out, video, infer_video(inferencer, video, args), args)
            print("  saved {}".format(out), flush=True)
            outputs.append(out)
        return outputs

    mp_context = process_pool_context(args)
    start_method = mp_context.get_start_method() if mp_context else multiprocessing.get_start_method()
    print("parallel RTMW workers: {} ({})".format(workers, start_method), flush=True)
    chunks = split_jobs(jobs, workers)
    executor_kwargs = {"max_workers": workers}
    if mp_context is not None:
        executor_kwargs["mp_context"] = mp_context
    with ProcessPoolExecutor(**executor_kwargs) as executor:
        futures = [executor.submit(run_worker, idx + 1, chunk, args) for idx, chunk in enumerate(chunks)]
        for future in as_completed(futures):
            outputs.extend(Path(path) for path in future.result())
    return outputs


def parser():
    p = argparse.ArgumentParser(description="Extract NTU RGB+D skeletons with RTMW.")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pattern", default="*_rgb.avi")
    p.add_argument("--all-videos", action="store_true")
    p.add_argument("--pose2d", default=RTMW_CONFIG)
    p.add_argument("--pose2d-weights", default=str(RTMW_WEIGHTS_PATH))
    p.add_argument("--det-model", default=RTMDET_CONFIG)
    p.add_argument("--det-weights", default=str(RTMDET_WEIGHTS_PATH))
    p.add_argument("--device", default="auto", help="Device for pose extraction. Default auto prefers cuda:0, then cpu.")
    p.add_argument("--max-persons", type=int, default=2)
    p.add_argument("--bbox-thr", type=float, default=0.3)
    p.add_argument("--kpt-thr", type=float, default=0.1)
    p.add_argument("--tracking-distance", type=float, default=150.0)
    p.add_argument("--filter-output-to-bbox", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-bbox-margin", type=float, default=0.0)
    p.add_argument("--filter-output-to-frame", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--temporal-min-frames", type=int, default=2)
    p.add_argument("--temporal-max-jump", type=float, default=150.0)
    p.add_argument("--temporal-min-keypoints", type=int, default=5)
    p.add_argument("--show-skeleton", action="store_true")
    p.add_argument("--visualize-dir")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--cpu-threads", type=int, default=0, help="Limit CPU compute threads per process. Try 4 or 8 if CPU is pinned.")
    p.add_argument("--pose-batch-size", type=int, default=1, help="MMPose inferencer batch size. Try 4 or 8 when CPU is saturated and GPU is underused.")
    p.add_argument("--workers", type=int, default=1, help="Parallel video extraction workers. Try 2 first on one GPU.")
    p.add_argument("--cpu-workers", type=int, default=0, help="Additional CPU-only workers sharing a dynamic video queue with CUDA workers.")
    p.add_argument("--cpu-worker-threads", type=int, default=4, help="Compute threads used by each additional CPU-only worker.")
    p.add_argument("--cpu-pose-batch-size", type=int, default=1, help="MMPose batch size for additional CPU-only workers.")
    p.add_argument("--scan-workers", type=int, default=32, help="Threads used to check existing output files before extraction.")
    return p


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
