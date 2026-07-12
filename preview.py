import argparse
import shutil
import tempfile
import threading
import time
import zipfile
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

from ntu_rtmw.constants import (
    EXTRACTED_DIR,
    RAW_ARCHIVES_DIR,
    RTMDET_CONFIG,
    RTMDET_WEIGHTS_PATH,
    RTMW_CONFIG,
    RTMW_WEIGHTS_PATH,
    SKELETON_DIR,
)
from ntu_rtmw.extract import TemporalDisplayFilter, draw_skeleton, run as extract_run


DEFAULT_VIS_DIR = Path("data") / "visualizations_preview"
DEFAULT_MAX_WIDTH = 960
DEFAULT_MAX_HEIGHT = 540
SIDEBAR_WIDTH = 330
SIDEBAR_ROW_HEIGHT = 27


@dataclass(frozen=True)
class ArchiveVideo:
    archive: Path
    member: str

    @property
    def name(self):
        return Path(self.member).name

    @property
    def stem(self):
        return Path(self.member).stem

    def __str__(self):
        return "{}::{}".format(self.archive, self.member)


def video_name(video):
    return video.name if isinstance(video, ArchiveVideo) else Path(video).name


def video_stem(video):
    return video.stem if isinstance(video, ArchiveVideo) else Path(video).stem


def video_label(video):
    if isinstance(video, ArchiveVideo):
        return "{} / {}".format(video.archive.name, video.name)
    return str(Path(video))


def video_group(video):
    if isinstance(video, ArchiveVideo):
        name = video.archive.stem.lower()
        return name.removeprefix("nturgbd_rgb_")
    path = Path(video)
    try:
        relative = path.resolve().relative_to(EXTRACTED_DIR.resolve())
        return relative.parts[0] if len(relative.parts) > 1 else path.parent.name
    except ValueError:
        return path.parent.name or "files"


def build_video_sections(videos, section_size=100):
    groups = {}
    for index, video in enumerate(videos):
        groups.setdefault(video_group(video), []).append(index)
    sections = []
    for group, indices in groups.items():
        for start in range(0, len(indices), section_size):
            chunk = indices[start:start + section_size]
            sections.append(("{}  {}-{}".format(group, start + 1, start + len(chunk)), chunk))
    return sections


class ArchiveVideoCache:
    """Materialize only selected ZIP members and preload one upcoming video."""

    def __init__(self):
        self._temp = tempfile.TemporaryDirectory(prefix="ntu_preview_")
        self.root = Path(self._temp.name)
        self._paths = {}
        self._jobs = {}
        self._errors = {}
        self._lock = threading.Lock()
        self._extract_lock = threading.Lock()
        self._closed = False

    def _target(self, source):
        archive_tag = source.archive.stem.replace(".", "_")
        return self.root / archive_tag / source.name

    def _extract(self, source):
        target = self._target(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        try:
            with self._extract_lock:
                with zipfile.ZipFile(source.archive) as zf:
                    info = zf.getinfo(source.member)
                    if info.is_dir() or not source.name.lower().endswith("_rgb.avi") or info.file_size <= 0:
                        raise ValueError("Refusing unsafe or invalid video member: {}".format(source.member))
                    with zf.open(info) as src, partial.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=4 * 1024 * 1024)
                partial.replace(target)
            with self._lock:
                if not self._closed:
                    self._paths[source] = target
        except Exception as exc:
            with self._lock:
                self._errors[source] = exc
            try:
                partial.unlink()
            except FileNotFoundError:
                pass

    def preload(self, source):
        if not isinstance(source, ArchiveVideo):
            return
        with self._lock:
            existing = self._jobs.get(source)
            if source in self._paths or (existing is not None and existing.is_alive()) or self._closed:
                return
            job = threading.Thread(target=self._extract, args=(source,), daemon=True)
            self._jobs[source] = job
            job.start()

    def materialize(self, source, progress=None):
        if not isinstance(source, ArchiveVideo):
            return Path(source)
        self.preload(source)
        with self._lock:
            job = self._jobs[source]
        while job.is_alive():
            job.join(0.05)
            if progress:
                progress(source)
        with self._lock:
            error = self._errors.pop(source, None)
            path = self._paths.get(source)
        if error:
            raise SystemExit("Failed to read {}: {}".format(source, error))
        if path is None:
            raise SystemExit("Archive member was not materialized: {}".format(source))
        return path

    def retain(self, sources):
        keep = {source for source in sources if isinstance(source, ArchiveVideo)}
        with self._lock:
            stale = [(source, path) for source, path in self._paths.items() if source not in keep]
            for source, _ in stale:
                self._paths.pop(source, None)
        for _, path in stale:
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def close(self):
        with self._lock:
            self._closed = True
            jobs = list(self._jobs.values())
        for job in jobs:
            job.join()
        self._temp.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def find_first_video(root):
    return list_videos(root)[0]


def list_videos(root, archives_dir=RAW_ARCHIVES_DIR):
    root = Path(root)
    extracted = sorted(root.rglob("*_rgb.avi")) if root.exists() else []
    by_name = {path.name.lower(): path for path in extracted}
    archives_dir = Path(archives_dir)
    if archives_dir.exists():
        for archive in sorted(archives_dir.glob("*.zip")):
            try:
                with zipfile.ZipFile(archive) as zf:
                    for member in zf.namelist():
                        name = Path(member).name
                        if name.lower().endswith("_rgb.avi"):
                            by_name.setdefault(name.lower(), ArchiveVideo(archive, member))
            except zipfile.BadZipFile as exc:
                print("skip bad ZIP {}: {}".format(archive, exc), flush=True)
    videos = sorted(by_name.values(), key=lambda item: video_name(item).lower())
    if not videos:
        raise SystemExit("No NTU RGB video found under {} or {}".format(root, archives_dir))
    return videos


def visualization_paths(video, vis_dir):
    stem = video_stem(video)
    vis_dir = Path(vis_dir)
    return vis_dir / "{}_skeleton.avi".format(stem), vis_dir / "{}_skeleton.jpg".format(stem)


def skeleton_candidates(video, out_dir):
    video = Path(video).resolve()
    out_dir = Path(out_dir)
    candidates = [out_dir / "{}.npz".format(video.stem)]
    try:
        candidates.append(out_dir / video.relative_to(EXTRACTED_DIR.resolve()).with_suffix(".npz"))
    except ValueError:
        pass
    seen = set()
    unique = []
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique


def find_skeleton_path(video, out_dir):
    for candidate in skeleton_candidates(video, out_dir):
        if candidate.exists() and candidate.stat().st_size > 1024:
            return candidate
    matches = sorted(Path(out_dir).rglob("{}.npz".format(Path(video).stem)))
    return matches[0] if matches else None


def preview_is_ready(path):
    path = Path(path)
    return path.exists() and path.stat().st_size > 1024


def ensure_skeleton(
    video,
    out_dir,
    regenerate=False,
    device="auto",
    cpu_threads=4,
    pose_batch_size=1,
    temporal_min_frames=2,
    temporal_max_jump=150.0,
    temporal_min_keypoints=5,
):
    skeleton_path = find_skeleton_path(video, out_dir)
    if skeleton_path is not None and not regenerate:
        print("skeleton ready: {}".format(skeleton_path), flush=True)
        return skeleton_path

    extract_run(Namespace(
        input=str(video),
        output=str(out_dir),
        pattern="*_rgb.avi",
        all_videos=False,
        pose2d_weights=str(RTMW_WEIGHTS_PATH),
        pose2d=RTMW_CONFIG,
        det_model=RTMDET_CONFIG,
        det_weights=str(RTMDET_WEIGHTS_PATH),
        device=device,
        max_persons=2,
        bbox_thr=0.3,
        kpt_thr=0.1,
        tracking_distance=150.0,
        filter_output_to_bbox=True,
        output_bbox_margin=0.0,
        filter_output_to_frame=True,
        temporal_min_frames=temporal_min_frames,
        temporal_max_jump=temporal_max_jump,
        temporal_min_keypoints=temporal_min_keypoints,
        show_skeleton=False,
        visualize_dir=None,
        skip_existing=False,
        limit=None,
        cpu_threads=cpu_threads,
        pose_batch_size=pose_batch_size,
        workers=1,
    ))
    skeleton_path = find_skeleton_path(video, out_dir)
    if skeleton_path is None:
        raise SystemExit("Skeleton file was not generated for: {}".format(video))
    return skeleton_path


def ensure_visualization(
    video,
    out_dir,
    vis_dir,
    regenerate=False,
    device="auto",
    cpu_threads=4,
    pose_batch_size=1,
    temporal_min_frames=2,
    temporal_max_jump=150.0,
    temporal_min_keypoints=5,
):
    vis_video, vis_image = visualization_paths(video, vis_dir)
    if preview_is_ready(vis_video) and not regenerate:
        print("preview ready: {}".format(vis_video), flush=True)
        return vis_video, vis_image

    extract_run(Namespace(
        input=str(video),
        output=str(out_dir),
        pattern="*_rgb.avi",
        all_videos=False,
        pose2d_weights=str(RTMW_WEIGHTS_PATH),
        pose2d=RTMW_CONFIG,
        det_model=RTMDET_CONFIG,
        det_weights=str(RTMDET_WEIGHTS_PATH),
        device=device,
        max_persons=2,
        bbox_thr=0.3,
        kpt_thr=0.1,
        tracking_distance=150.0,
        filter_output_to_bbox=True,
        output_bbox_margin=0.0,
        filter_output_to_frame=True,
        temporal_min_frames=temporal_min_frames,
        temporal_max_jump=temporal_max_jump,
        temporal_min_keypoints=temporal_min_keypoints,
        show_skeleton=False,
        visualize_dir=str(vis_dir),
        skip_existing=False,
        limit=None,
        cpu_threads=cpu_threads,
        pose_batch_size=pose_batch_size,
        workers=1,
    ))
    return vis_video, vis_image


def load_skeleton_arrays(path):
    import numpy as np

    data = np.load(str(path), allow_pickle=False)
    keypoints = data["keypoints"]
    scores = data["scores"]
    bboxes = data["bboxes"] if "bboxes" in data else np.full(keypoints.shape[:2] + (4,), np.nan, dtype=np.float32)
    bbox_scores = data["bbox_scores"] if "bbox_scores" in data else np.zeros(keypoints.shape[:2], dtype=np.float32)
    frame_indices = data["frame_indices"] if "frame_indices" in data else np.arange(keypoints.shape[0], dtype=np.int32)
    return {
        "keypoints": keypoints,
        "scores": scores,
        "bboxes": bboxes,
        "bbox_scores": bbox_scores,
        "frame_indices": frame_indices,
    }


def filter_direct_points(frame, keypoints, scores, bboxes, bbox_scores, args):
    import numpy as np

    filtered_keypoints = keypoints.copy()
    filtered_scores = scores.copy()
    filtered_bboxes = bboxes.copy()
    filtered_bbox_scores = bbox_scores.copy()
    height, width = frame.shape[:2]
    for person_idx, bbox in enumerate(filtered_bboxes):
        if filtered_bbox_scores[person_idx] < args.direct_bbox_thr or np.isnan(bbox).any():
            filtered_keypoints[person_idx] = np.nan
            filtered_scores[person_idx] = 0
            filtered_bboxes[person_idx] = np.nan
            filtered_bbox_scores[person_idx] = 0
            continue
        x1, y1, x2, y2 = bbox.astype(float)
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        margin_x = box_w * args.direct_bbox_margin
        margin_y = box_h * args.direct_bbox_margin
        x1 -= margin_x
        x2 += margin_x
        y1 -= margin_y
        y2 += margin_y
        points = filtered_keypoints[person_idx]
        valid = (
            (filtered_scores[person_idx] >= args.kpt_thr)
            & (points[:, 0] >= 0)
            & (points[:, 0] < width)
            & (points[:, 1] >= 0)
            & (points[:, 1] < height)
            & (points[:, 0] >= x1)
            & (points[:, 0] <= x2)
            & (points[:, 1] >= y1)
            & (points[:, 1] <= y2)
            & ~np.isnan(points[:, 0])
            & ~np.isnan(points[:, 1])
        )
        filtered_keypoints[person_idx, ~valid] = np.nan
        filtered_scores[person_idx, ~valid] = 0
        if valid[:17].sum() < args.temporal_min_keypoints:
            filtered_keypoints[person_idx] = np.nan
            filtered_scores[person_idx] = 0
            filtered_bboxes[person_idx] = np.nan
            filtered_bbox_scores[person_idx] = 0
    return filtered_keypoints, filtered_scores, filtered_bboxes, filtered_bbox_scores


class PersonStabilizer:
    def __init__(self, max_distance=180.0, hold_frames=2, kpt_thr=0.25):
        import numpy as np

        self.max_distance = float(max_distance or 0.0)
        self.hold_frames = max(0, int(hold_frames or 0))
        self.kpt_thr = float(kpt_thr)
        self.previous_centers = None
        self.previous_keypoints = None
        self.previous_scores = None
        self.previous_bboxes = None
        self.previous_bbox_scores = None
        self.missing_counts = None
        self.np = np

    def _center(self, points, scores, bbox):
        np = self.np
        valid = (scores[:17] >= self.kpt_thr) & ~np.isnan(points[:17, 0]) & ~np.isnan(points[:17, 1])
        if valid.sum() >= 3:
            return np.nanmedian(points[:17][valid], axis=0).astype(np.float32)
        if not np.isnan(bbox).any():
            return np.array([(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5], dtype=np.float32)
        return np.array([np.nan, np.nan], dtype=np.float32)

    def _ensure_shape(self, keypoints):
        np = self.np
        person_count = keypoints.shape[0]
        if self.previous_centers is None or self.previous_centers.shape[0] != person_count:
            self.previous_centers = np.full((person_count, 2), np.nan, dtype=np.float32)
            self.previous_keypoints = np.full_like(keypoints, np.nan)
            self.previous_scores = np.zeros(keypoints.shape[:2], dtype=np.float32)
            self.previous_bboxes = np.full((person_count, 4), np.nan, dtype=np.float32)
            self.previous_bbox_scores = np.zeros((person_count,), dtype=np.float32)
            self.missing_counts = np.zeros((person_count,), dtype=np.int32)

    def apply(self, keypoints, scores, bboxes, bbox_scores):
        np = self.np
        self._ensure_shape(keypoints)
        person_count = keypoints.shape[0]
        current_centers = np.stack([
            self._center(keypoints[i], scores[i], bboxes[i]) for i in range(person_count)
        ])
        current_valid = ~np.isnan(current_centers).any(axis=1)

        ordered_keypoints = np.full_like(keypoints, np.nan)
        ordered_scores = np.zeros_like(scores)
        ordered_bboxes = np.full_like(bboxes, np.nan)
        ordered_bbox_scores = np.zeros_like(bbox_scores)
        ordered_centers = np.full_like(current_centers, np.nan)
        used_detections = set()

        candidates = []
        for det_idx, center in enumerate(current_centers):
            if not current_valid[det_idx]:
                continue
            for slot_idx, prev_center in enumerate(self.previous_centers):
                if np.isnan(prev_center).any():
                    continue
                distance = float(np.linalg.norm(center - prev_center))
                candidates.append((distance, slot_idx, det_idx))
        for distance, slot_idx, det_idx in sorted(candidates):
            if self.max_distance > 0 and distance > self.max_distance:
                continue
            if det_idx in used_detections or not np.isnan(ordered_centers[slot_idx]).any():
                continue
            ordered_keypoints[slot_idx] = keypoints[det_idx]
            ordered_scores[slot_idx] = scores[det_idx]
            ordered_bboxes[slot_idx] = bboxes[det_idx]
            ordered_bbox_scores[slot_idx] = bbox_scores[det_idx]
            ordered_centers[slot_idx] = current_centers[det_idx]
            used_detections.add(det_idx)

        for det_idx in range(person_count):
            if det_idx in used_detections or not current_valid[det_idx]:
                continue
            empty_slots = [slot_idx for slot_idx in range(person_count) if np.isnan(ordered_centers[slot_idx]).any()]
            if not empty_slots:
                break
            slot_idx = empty_slots[0]
            ordered_keypoints[slot_idx] = keypoints[det_idx]
            ordered_scores[slot_idx] = scores[det_idx]
            ordered_bboxes[slot_idx] = bboxes[det_idx]
            ordered_bbox_scores[slot_idx] = bbox_scores[det_idx]
            ordered_centers[slot_idx] = current_centers[det_idx]
            used_detections.add(det_idx)

        for slot_idx in range(person_count):
            if not np.isnan(ordered_centers[slot_idx]).any():
                self.missing_counts[slot_idx] = 0
                continue
            self.missing_counts[slot_idx] += 1
            if self.missing_counts[slot_idx] <= self.hold_frames and not np.isnan(self.previous_centers[slot_idx]).any():
                ordered_keypoints[slot_idx] = self.previous_keypoints[slot_idx]
                ordered_scores[slot_idx] = self.previous_scores[slot_idx]
                ordered_bboxes[slot_idx] = self.previous_bboxes[slot_idx]
                ordered_bbox_scores[slot_idx] = self.previous_bbox_scores[slot_idx]
                ordered_centers[slot_idx] = self.previous_centers[slot_idx]

        self.previous_centers = ordered_centers.copy()
        self.previous_keypoints = ordered_keypoints.copy()
        self.previous_scores = ordered_scores.copy()
        self.previous_bboxes = ordered_bboxes.copy()
        self.previous_bbox_scores = ordered_bbox_scores.copy()
        return ordered_keypoints, ordered_scores, ordered_bboxes, ordered_bbox_scores


def resize_for_preview(frame, max_width, max_height):
    import cv2

    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def _format_time(seconds):
    seconds = max(0, int(seconds or 0))
    return "{:02d}:{:02d}".format(seconds // 60, seconds % 60)


def draw_controls(
    frame,
    label,
    index,
    total,
    auto_play=True,
    paused=False,
    speed=1.0,
    position=0,
    duration=0,
):
    import cv2

    height, width = frame.shape[:2]
    controls = {}
    buttons = [
        ("previous", "Prev", 82),
        ("select", "Select", 88),
        ("pause", "Play" if paused else "Pause", 88),
        ("replay", "Replay", 92),
        ("slower", "-Speed", 92),
        ("faster", "+Speed", 92),
        ("next", "Next", 82),
    ]
    gap, button_h = 8, 38
    total_width = sum(item[2] for item in buttons) + gap * (len(buttons) - 1)
    x = max(10, (width - total_width) // 2)
    y = 10
    for action, text, button_w in buttons:
        rect = (x, y, min(width - 4, x + button_w), y + button_h)
        controls[action] = rect
        color = (30, 120, 30) if action == "next" else (65, 65, 65)
        cv2.rectangle(frame, rect[:2], rect[2:], color, -1)
        cv2.rectangle(frame, rect[:2], rect[2:], (210, 210, 210), 1)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0]
        text_x = rect[0] + max(4, (rect[2] - rect[0] - text_size[0]) // 2)
        cv2.putText(frame, text, (text_x, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        x += button_w + gap

    panel_top = max(0, height - 67)
    cv2.rectangle(frame, (0, panel_top), (width, height), (0, 0, 0), -1)
    bar_x1, bar_x2 = 18, max(19, width - 18)
    bar_y1, bar_y2 = panel_top + 10, panel_top + 22
    controls["progress"] = (bar_x1, bar_y1, bar_x2, bar_y2)
    cv2.rectangle(frame, (bar_x1, bar_y1), (bar_x2, bar_y2), (80, 80, 80), -1)
    ratio = 0.0 if duration <= 0 else min(1.0, max(0.0, position / duration))
    fill_x = bar_x1 + int((bar_x2 - bar_x1) * ratio)
    cv2.rectangle(frame, (bar_x1, bar_y1), (fill_x, bar_y2), (70, 210, 70), -1)
    cv2.circle(frame, (fill_x, (bar_y1 + bar_y2) // 2), 7, (230, 230, 230), -1)

    mode = "Loop" if not auto_play else "Auto"
    status = "Paused" if paused else "Playing"
    info = "{}/{}  {}  {}  {:.2f}x  {} / {}".format(
        index + 1,
        total,
        status,
        mode,
        speed,
        _format_time(position),
        _format_time(duration),
    )
    cv2.putText(frame, info, (18, panel_top + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    label_limit = max(12, (width - 36) // 9)
    short_label = label if len(label) <= label_limit else "..." + label[-(label_limit - 3):]
    cv2.putText(frame, short_label, (18, panel_top + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 190, 190), 1)
    return controls


def draw_sidebar(frame, videos, current_index, state, cv2):
    import numpy as np

    height = frame.shape[0]
    panel = np.full((height, SIDEBAR_WIDTH, 3), 24, dtype=np.uint8)
    cv2.rectangle(panel, (0, 0), (SIDEBAR_WIDTH - 1, height - 1), (85, 85, 85), 1)
    if not state.sidebar_sections:
        return np.concatenate((frame, panel), axis=1)

    section_index = min(max(0, state.sidebar_section), len(state.sidebar_sections) - 1)
    title, indices = state.sidebar_sections[section_index]
    cv2.putText(panel, "Files", (14, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1)
    cv2.putText(panel, "section {}/{}".format(section_index + 1, len(state.sidebar_sections)), (175, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)

    prev_rect = (12, 36, 82, 65)
    next_rect = (SIDEBAR_WIDTH - 82, 36, SIDEBAR_WIDTH - 12, 65)
    state.controls["section_previous"] = tuple(value + (frame.shape[1] if pos % 2 == 0 else 0) for pos, value in enumerate(prev_rect))
    state.controls["section_next"] = tuple(value + (frame.shape[1] if pos % 2 == 0 else 0) for pos, value in enumerate(next_rect))
    cv2.rectangle(panel, prev_rect[:2], prev_rect[2:], (65, 65, 65), -1)
    cv2.rectangle(panel, next_rect[:2], next_rect[2:], (65, 65, 65), -1)
    cv2.putText(panel, "< 100", (22, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1)
    cv2.putText(panel, "100 >", (next_rect[0] + 15, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1)
    cv2.putText(panel, title, (92, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (90, 220, 90), 1)

    list_top = 76
    footer_height = 34
    visible_count = max(1, (height - list_top - footer_height) // SIDEBAR_ROW_HEIGHT)
    max_offset = max(0, len(indices) - visible_count)
    state.sidebar_offset = min(max(0, state.sidebar_offset), max_offset)
    visible = indices[state.sidebar_offset:state.sidebar_offset + visible_count]
    for row, video_index in enumerate(visible):
        y1 = list_top + row * SIDEBAR_ROW_HEIGHT
        y2 = y1 + SIDEBAR_ROW_HEIGHT - 2
        selected = video_index == current_index
        color = (45, 105, 45) if selected else ((42, 42, 42) if row % 2 == 0 else (32, 32, 32))
        cv2.rectangle(panel, (7, y1), (SIDEBAR_WIDTH - 15, y2), color, -1)
        text = "{:4d}  {}".format(video_index + 1, video_name(videos[video_index]))
        if len(text) > 42:
            text = text[:39] + "..."
        cv2.putText(panel, text, (12, y1 + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.39, (245, 245, 245), 1)
        state.controls[("video", video_index)] = (
            frame.shape[1] + 7,
            y1,
            frame.shape[1] + SIDEBAR_WIDTH - 15,
            y2,
        )

    if len(indices) > visible_count:
        track_top, track_bottom = list_top, height - footer_height
        thumb_h = max(24, int((track_bottom - track_top) * visible_count / len(indices)))
        travel = max(1, track_bottom - track_top - thumb_h)
        thumb_y = track_top + int(travel * state.sidebar_offset / max_offset)
        cv2.rectangle(panel, (SIDEBAR_WIDTH - 10, track_top), (SIDEBAR_WIDTH - 5, track_bottom), (55, 55, 55), -1)
        cv2.rectangle(panel, (SIDEBAR_WIDTH - 10, thumb_y), (SIDEBAR_WIDTH - 5, thumb_y + thumb_h), (150, 150, 150), -1)

    cv2.putText(panel, "Wheel: scroll   Click: open", (12, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (170, 170, 170), 1)
    return np.concatenate((frame, panel), axis=1)


def point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def choose_video_index(videos, current_index):
    """Show a small native picker and return the chosen playlist index."""
    try:
        import tkinter as tk
        from tkinter import ttk
    except ImportError:
        return current_index
    result = [current_index]
    try:
        root = tk.Tk()
    except tk.TclError:
        return current_index
    root.title("Select NTU preview video")
    root.geometry("760x520")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(1, weight=1)
    query = tk.StringVar()
    top = ttk.Frame(root)
    top.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=(10, 4))
    ttk.Label(top, text="Type to filter, then double-click a video:").pack(side="left")
    ttk.Entry(top, textvariable=query, width=42).pack(side="right", fill="x", expand=True, padx=(12, 0))
    box = tk.Listbox(root, activestyle="dotbox")
    box.grid(row=1, column=0, sticky="nsew", padx=10, pady=4)
    scroll = ttk.Scrollbar(root, orient="vertical", command=box.yview)
    scroll.grid(row=1, column=1, sticky="ns", pady=4)
    box.configure(yscrollcommand=scroll.set)
    shown = []

    def refresh(*_):
        needle = query.get().strip().lower()
        shown[:] = [(idx, video_label(item)) for idx, item in enumerate(videos) if not needle or needle in video_name(item).lower() or needle in video_label(item).lower()]
        box.delete(0, tk.END)
        for idx, label in shown:
            box.insert(tk.END, "{:4d}  {}".format(idx + 1, label))

    def accept(*_):
        selected = box.curselection()
        if selected:
            result[0] = shown[selected[0]][0]
            root.destroy()

    query.trace_add("write", refresh)
    box.bind("<Double-Button-1>", accept)
    ttk.Button(root, text="Open", command=accept).grid(row=2, column=0, sticky="e", padx=10, pady=10)
    refresh()
    root.mainloop()
    return result[0]


class PreviewState:
    def __init__(self):
        self.next_requested = False
        self.previous_requested = False
        self.pause_requested = False
        self.replay_requested = False
        self.speed_delta = 0
        self.seek_ratio = None
        self.select_requested = False
        self.jump_index = None
        self.sidebar_sections = []
        self.sidebar_videos = []
        self.sidebar_section = 0
        self.sidebar_offset = 0
        self.last_file_click = 0.0
        self.controls = {}

    def reset_requests(self):
        self.next_requested = False
        self.previous_requested = False
        self.pause_requested = False
        self.replay_requested = False
        self.speed_delta = 0
        self.seek_ratio = None
        self.select_requested = False

    def configure_sidebar(self, videos):
        self.sidebar_videos = videos
        self.sidebar_sections = build_video_sections(videos)

    def focus_sidebar(self, video_index):
        for section_index, (_, indices) in enumerate(self.sidebar_sections):
            if video_index in indices:
                self.sidebar_section = section_index
                position = indices.index(video_index)
                self.sidebar_offset = max(0, position - 5)
                return

    def change_section(self, delta):
        if not self.sidebar_sections:
            return
        self.sidebar_section = min(max(0, self.sidebar_section + delta), len(self.sidebar_sections) - 1)
        self.sidebar_offset = 0


class PreviewBuildJob:
    def __init__(self, video, args):
        self.video = video
        self.args = args
        self.done = False
        self.result = None
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            if self.args.direct:
                self.result = ensure_skeleton(
                    video=self.video,
                    out_dir=self.args.out_dir,
                    regenerate=self.args.regenerate,
                    device=self.args.device,
                    cpu_threads=self.args.cpu_threads,
                    pose_batch_size=self.args.pose_batch_size,
                    temporal_min_frames=self.args.temporal_min_frames,
                    temporal_max_jump=self.args.temporal_max_jump,
                    temporal_min_keypoints=self.args.temporal_min_keypoints,
                )
            else:
                self.result = ensure_visualization(
                    video=self.video,
                    out_dir=self.args.out_dir,
                    vis_dir=self.args.vis_dir,
                    regenerate=self.args.regenerate,
                    device=self.args.device,
                    cpu_threads=self.args.cpu_threads,
                    pose_batch_size=self.args.pose_batch_size,
                    temporal_min_frames=self.args.temporal_min_frames,
                    temporal_max_jump=self.args.temporal_max_jump,
                    temporal_min_keypoints=self.args.temporal_min_keypoints,
                )
        except Exception as exc:
            self.error = exc
        finally:
            self.done = True


def draw_loading(frame, message, detail):
    import cv2
    import numpy as np

    if frame is None:
        frame = np.zeros((DEFAULT_MAX_HEIGHT, DEFAULT_MAX_WIDTH, 3), dtype=np.uint8)
    else:
        frame = frame.copy()
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 0), 3)
    cv2.rectangle(frame, (40, h // 2 - 62), (w - 40, h // 2 + 64), (0, 0, 0), -1)
    cv2.rectangle(frame, (40, h // 2 - 62), (w - 40, h // 2 + 64), (70, 220, 70), 2)
    cv2.putText(frame, message, (62, h // 2 - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2)
    cv2.putText(frame, detail, (62, h // 2 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (210, 210, 210), 1)
    cv2.putText(frame, "Press q or Esc to quit", (62, h // 2 + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    return frame


def wait_for_preview(video, args, cv2, window, state, last_frame):
    if args.direct:
        skeleton_path = find_skeleton_path(video, args.out_dir)
        if skeleton_path is not None and not args.regenerate:
            print("skeleton ready: {}".format(skeleton_path), flush=True)
            return skeleton_path

    vis_video, vis_image = visualization_paths(video, args.vis_dir)
    if not args.direct and preview_is_ready(vis_video) and not args.regenerate:
        print("preview ready: {}".format(vis_video), flush=True)
        return vis_video, vis_image

    action = "building skeleton in background" if args.direct else "building preview in background"
    print("{}: {}".format(action, video), flush=True)
    state.controls = {}
    job = PreviewBuildJob(video, args).start()
    while not job.done:
        frame = draw_loading(
            last_frame,
            "Generating skeleton preview...",
            Path(video).name,
        )
        cv2.imshow(window, frame)
        key = cv2.waitKey(100) & 0xFF
        if key in (ord("q"), 27):
            return None
    if job.error:
        raise SystemExit("Failed to build preview for {}: {}".format(video, job.error))
    return job.result


def draw_direct_frame(frame, arrays, array_idx, args, display_filter, person_stabilizer, cv2):
    if array_idx < 0 or array_idx >= arrays["keypoints"].shape[0]:
        return frame
    keypoints = arrays["keypoints"][array_idx]
    scores = arrays["scores"][array_idx]
    bboxes = arrays["bboxes"][array_idx]
    bbox_scores = arrays["bbox_scores"][array_idx]
    keypoints, scores, bboxes, bbox_scores = filter_direct_points(frame, keypoints, scores, bboxes, bbox_scores, args)
    if args.stabilize_people:
        keypoints, scores, bboxes, bbox_scores = person_stabilizer.apply(keypoints, scores, bboxes, bbox_scores)
    if args.direct_temporal_display:
        keypoints, scores, bboxes, bbox_scores = display_filter.apply_arrays(keypoints, scores, bboxes, bbox_scores)
    draw_skeleton(frame, keypoints, scores, bboxes, bbox_scores, args.kpt_thr, cv2)
    return frame


def play_direct_video(video, skeleton_path, args, cv2, state, index=0, total=1, label=None):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit("Cannot open video: {}".format(video))
    arrays = load_skeleton_arrays(skeleton_path)
    frame_to_array = {int(frame_idx): idx for idx, frame_idx in enumerate(arrays["frame_indices"])}
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    speed = args.speed
    paused = False
    display_filter = TemporalDisplayFilter(
        args.temporal_min_frames,
        args.temporal_max_jump,
        args.temporal_min_keypoints,
        args.kpt_thr,
    )
    person_stabilizer = PersonStabilizer(
        args.person_match_distance,
        args.person_hold_frames,
        args.kpt_thr,
    )
    frame_idx = 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or arrays["keypoints"].shape[0])
    last_frame = None
    while True:
        if not paused or last_frame is None:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                return "ended"
            array_idx = frame_to_array.get(frame_idx, frame_idx)
            frame = draw_direct_frame(frame, arrays, array_idx, args, display_filter, person_stabilizer, cv2)
            last_frame = frame.copy()
        else:
            frame = last_frame.copy()
        frame = resize_for_preview(frame, args.max_width, args.max_height)
        state.controls = draw_controls(
            frame,
            label or video_name(video),
            index,
            total,
            auto_play=not args.loop_current,
            paused=paused,
            speed=speed,
            position=frame_idx / fps,
            duration=frame_count / fps,
        )
        frame = draw_sidebar(frame, state.sidebar_videos, index, state, cv2)
        cv2.imshow("NTU RTMW skeleton preview", frame)
        delay = 50 if paused else max(1, int(1000 / max(1, fps * speed)))
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), 27):
            cap.release()
            return "quit"
        if key in (ord("n"), ord("d")):
            cap.release()
            return "next"
        if key in (ord("p"), ord("a")):
            cap.release()
            return "previous"
        if key == ord("g"):
            cap.release()
            return "select"
        if key == ord(" "):
            paused = not paused
        if key == ord("r"):
            state.seek_ratio = 0.0
        if key in (ord("-"), ord("_")):
            speed = max(0.25, speed / 1.25)
        if key in (ord("+"), ord("=")):
            speed = min(4.0, speed * 1.25)
        if state.previous_requested:
            state.previous_requested = False
            cap.release()
            return "previous"
        if state.next_requested:
            state.next_requested = False
            cap.release()
            return "next"
        if state.select_requested:
            state.select_requested = False
            cap.release()
            return "select"
        if state.jump_index is not None:
            cap.release()
            return "jump"
        if state.pause_requested:
            state.pause_requested = False
            paused = not paused
        if state.replay_requested:
            state.replay_requested = False
            state.seek_ratio = 0.0
        if state.speed_delta:
            speed = min(4.0, max(0.25, speed * (1.25 ** state.speed_delta)))
            state.speed_delta = 0
        if state.seek_ratio is not None:
            frame_idx = min(max(0, int(frame_count * state.seek_ratio)), max(0, frame_count - 1))
            state.seek_ratio = None
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            last_frame = None
            display_filter = TemporalDisplayFilter(
                args.temporal_min_frames,
                args.temporal_max_jump,
                args.temporal_min_keypoints,
                args.kpt_thr,
            )
            person_stabilizer = PersonStabilizer(
                args.person_match_distance,
                args.person_hold_frames,
                args.kpt_thr,
            )
        elif not paused:
            frame_idx += 1


def play_playlist(videos, start_index, args, cache):
    import cv2

    state = PreviewState()
    state.configure_sidebar(videos)
    window = "NTU RTMW skeleton preview"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.max_width + SIDEBAR_WIDTH, args.max_height)
    last_frame = None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEWHEEL:
            delta = cv2.getMouseWheelDelta(flags) if hasattr(cv2, "getMouseWheelDelta") else flags
            state.sidebar_offset = max(0, state.sidebar_offset + (-5 if delta > 0 else 5))
            return
        if event != cv2.EVENT_LBUTTONUP:
            return
        for action, rect in state.controls.items():
            if not point_in_rect(x, y, rect):
                continue
            if action == "previous":
                state.previous_requested = True
            elif action == "next":
                state.next_requested = True
            elif action == "select":
                state.select_requested = True
            elif action == "section_previous":
                state.change_section(-1)
            elif action == "section_next":
                state.change_section(1)
            elif isinstance(action, tuple) and action[0] == "video":
                now = time.monotonic()
                if now - state.last_file_click >= 0.3:
                    state.jump_index = action[1]
                    state.last_file_click = now
            elif action == "pause":
                state.pause_requested = True
            elif action == "replay":
                state.replay_requested = True
            elif action == "slower":
                state.speed_delta = -1
            elif action == "faster":
                state.speed_delta = 1
            elif action == "progress":
                state.seek_ratio = (x - rect[0]) / max(1, rect[2] - rect[0])
            break

    cv2.setMouseCallback(window, on_mouse)
    index = start_index
    while True:
        state.reset_requests()
        state.focus_sidebar(index)
        source = videos[index]
        next_source = videos[(index + 1) % len(videos)]
        cache.preload(source)

        def show_extracting(item):
            frame = draw_loading(last_frame, "Loading selected video from ZIP...", video_label(item))
            cv2.imshow(window, frame)
            cv2.waitKey(1)

        video = cache.materialize(source, show_extracting)
        if args.preload:
            cache.preload(next_source)
        cache.retain((source, next_source) if args.preload else (source,))
        preview = wait_for_preview(video, args, cv2, window, state, last_frame)
        if preview is None:
            cv2.destroyAllWindows()
            return
        if args.direct:
            result = play_direct_video(
                video,
                preview,
                args,
                cv2,
                state,
                index,
                len(videos),
                label=video_name(source),
            )
            if result == "quit":
                cv2.destroyAllWindows()
                return
            if result == "select":
                index = choose_video_index(videos, index)
                continue
            if result == "jump":
                if state.jump_index is not None and 0 <= state.jump_index < len(videos):
                    index = state.jump_index
                state.jump_index = None
                continue
            if result == "previous":
                index = (index - 1) % len(videos)
            else:
                if args.loop_current and result == "ended":
                    continue
                index = (index + 1) % len(videos)
            continue

        vis_video, vis_image = preview
        print("playing: {}".format(vis_video.resolve()), flush=True)

        cap = cv2.VideoCapture(str(vis_video))
        if not cap.isOpened():
            print("bad preview, regenerating: {}".format(vis_video), flush=True)
            args.regenerate = True
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        speed = args.speed
        paused = False
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        frame_idx = 0
        playback_frame = None
        select_after = False
        while True:
            if not paused or playback_frame is None:
                ok, frame = cap.read()
                if not ok:
                    if args.loop_current:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        continue
                    index = (index + 1) % len(videos)
                    state.next_requested = False
                    break
                playback_frame = frame.copy()
                last_frame = frame.copy()
            else:
                frame = playback_frame.copy()
            frame = resize_for_preview(frame, args.max_width, args.max_height)
            state.controls = draw_controls(
                frame,
                video_name(source),
                index,
                len(videos),
                auto_play=not args.loop_current,
                paused=paused,
                speed=speed,
                position=frame_idx / fps,
                duration=frame_count / fps,
            )
            frame = draw_sidebar(frame, videos, index, state, cv2)
            cv2.imshow(window, frame)
            delay = 50 if paused else max(1, int(1000 / max(1, fps * speed)))
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27):
                cap.release()
                cv2.destroyAllWindows()
                return
            if key in (ord("n"), ord("d")):
                state.next_requested = True
            if key in (ord("p"), ord("a")):
                index = (index - 1) % len(videos)
                state.previous_requested = False
                state.next_requested = False
                break
            if key == ord("g"):
                select_after = True
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("r"):
                state.seek_ratio = 0.0
            if key in (ord("-"), ord("_")):
                speed = max(0.25, speed / 1.25)
            if key in (ord("+"), ord("=")):
                speed = min(4.0, speed * 1.25)
            if state.previous_requested:
                index = (index - 1) % len(videos)
                state.previous_requested = False
                state.next_requested = False
                break
            if state.next_requested:
                index = (index + 1) % len(videos)
                state.previous_requested = False
                state.next_requested = False
                break
            if state.select_requested:
                state.select_requested = False
                select_after = True
                break
            if state.jump_index is not None:
                if 0 <= state.jump_index < len(videos):
                    index = state.jump_index
                state.jump_index = None
                break
            if state.pause_requested:
                state.pause_requested = False
                paused = not paused
            if state.replay_requested:
                state.replay_requested = False
                state.seek_ratio = 0.0
            if state.speed_delta:
                speed = min(4.0, max(0.25, speed * (1.25 ** state.speed_delta)))
                state.speed_delta = 0
            if state.seek_ratio is not None:
                frame_idx = min(max(0, int(frame_count * state.seek_ratio)), max(0, frame_count - 1))
                state.seek_ratio = None
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                playback_frame = None
            elif not paused:
                frame_idx += 1
        cap.release()
        if select_after:
            index = choose_video_index(videos, index)


def play_video(path, speed=1.0, max_width=DEFAULT_MAX_WIDTH, max_height=DEFAULT_MAX_HEIGHT):
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit("Cannot open preview video: {}".format(path))

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    delay = max(1, int(1000 / max(1, fps * speed)))
    window = "NTU RTMW skeleton preview - press q to quit"
    print("playing: {}".format(path), flush=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frame = resize_for_preview(frame, max_width, max_height)
        cv2.imshow(window, frame)
        key = cv2.waitKey(delay) & 0xFF
        if key in (ord("q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()


def play_direct_single(video, skeleton_path, args):
    import cv2

    state = PreviewState()
    window = "NTU RTMW skeleton preview"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.max_width, args.max_height)
    while True:
        result = play_direct_video(video, skeleton_path, args, cv2, state)
        if result != "ended" or not args.loop_current:
            break
    if result != "quit":
        cv2.destroyAllWindows()


def parser():
    p = argparse.ArgumentParser(description="Generate and play one RTMW skeleton preview.")
    p.add_argument("--video", help="Specific NTU *_rgb.avi file. Defaults to the first extracted video.")
    p.add_argument("--index", type=int, default=1, help="1-based video index in the sorted preview playlist.")
    p.add_argument("--list", action="store_true", help="List indexed preview videos and exit.")
    p.add_argument("--vis-dir", default=str(DEFAULT_VIS_DIR))
    p.add_argument("--out-dir", default=str(SKELETON_DIR))
    p.add_argument("--archives-dir", default=str(RAW_ARCHIVES_DIR), help="ZIP directory used when videos are not already extracted.")
    p.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True, help="Preload the next archive video into the temporary cache.")
    p.add_argument("--device", default="auto", help="Device for preview generation. Default auto prefers cuda:0, then cpu.")
    p.add_argument("--cpu-threads", type=int, default=4, help="Limit CPU threads while generating a missing preview.")
    p.add_argument("--pose-batch-size", type=int, default=1, help="MMPose inferencer batch size while generating a missing preview.")
    p.add_argument("--direct", action=argparse.BooleanOptionalAction, default=True, help="Draw skeletons directly from .npz on the original video instead of generating a fused preview .avi.")
    p.add_argument("--direct-temporal-display", action=argparse.BooleanOptionalAction, default=True, help="Apply temporal display filtering while drawing direct previews.")
    p.add_argument("--kpt-thr", type=float, default=0.25, help="Keypoint score threshold used while drawing direct previews.")
    p.add_argument("--direct-bbox-thr", type=float, default=0.3, help="Hide direct-preview people below this bbox score.")
    p.add_argument("--direct-bbox-margin", type=float, default=0.0, help="Allow this bbox-relative margin when filtering direct-preview keypoints.")
    p.add_argument("--stabilize-people", action=argparse.BooleanOptionalAction, default=True, help="Keep direct-preview person slots stable across frames.")
    p.add_argument("--person-match-distance", type=float, default=180.0, help="Maximum pixel distance for matching people to the previous frame.")
    p.add_argument("--person-hold-frames", type=int, default=2, help="Hold the last visible skeleton for this many missing frames.")
    p.add_argument("--temporal-min-frames", type=int, default=3, help="Hide preview keypoints unless they survive this many consecutive frames.")
    p.add_argument("--temporal-max-jump", type=float, default=100.0, help="Hide one-frame preview keypoint jumps larger than this many pixels.")
    p.add_argument("--temporal-min-keypoints", type=int, default=8, help="Hide a preview person if fewer than this many body keypoints are visible.")
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--no-window", action="store_true", help="Only generate/check files, do not open OpenCV window.")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH)
    p.add_argument("--max-height", type=int, default=DEFAULT_MAX_HEIGHT)
    p.add_argument("--loop-current", action="store_true", help="Replay the current video instead of auto-playing the next one.")
    return p


def main():
    args = parser().parse_args()
    videos = list_videos(EXTRACTED_DIR, args.archives_dir)
    if args.list:
        for idx, path in enumerate(videos, 1):
            print("{:4d}  {}".format(idx, video_label(path)), flush=True)
        return

    if args.video:
        requested = args.video.lower()
        matches = [
            idx for idx, item in enumerate(videos)
            if requested in {video_name(item).lower(), str(item).lower()}
        ]
        if matches:
            start_index = matches[0]
        else:
            video = Path(args.video).resolve()
            videos = [video]
            start_index = 0
    else:
        if args.index < 1 or args.index > len(videos):
            raise SystemExit("--index must be between 1 and {}.".format(len(videos)))
        start_index = args.index - 1
    source = videos[start_index]
    with ArchiveVideoCache() as cache:
        if args.no_window:
            video = cache.materialize(source)
            if args.direct:
                skeleton_path = ensure_skeleton(
                    video=video,
                    out_dir=args.out_dir,
                    regenerate=args.regenerate,
                    device=args.device,
                    cpu_threads=args.cpu_threads,
                    pose_batch_size=args.pose_batch_size,
                    temporal_min_frames=args.temporal_min_frames,
                    temporal_max_jump=args.temporal_max_jump,
                    temporal_min_keypoints=args.temporal_min_keypoints,
                )
                print("skeleton file: {}".format(skeleton_path.resolve()), flush=True)
            else:
                vis_video, vis_image = ensure_visualization(
                    video=video,
                    out_dir=args.out_dir,
                    vis_dir=args.vis_dir,
                    regenerate=args.regenerate,
                    device=args.device,
                    cpu_threads=args.cpu_threads,
                    pose_batch_size=args.pose_batch_size,
                    temporal_min_frames=args.temporal_min_frames,
                    temporal_max_jump=args.temporal_max_jump,
                    temporal_min_keypoints=args.temporal_min_keypoints,
                )
                print("preview video: {}".format(vis_video.resolve()), flush=True)
                print("preview image: {}".format(vis_image.resolve()), flush=True)
        else:
            play_playlist(videos, start_index, args, cache)


if __name__ == "__main__":
    main()
