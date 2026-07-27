import argparse
import csv
import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from .constants import EXTRACTED_DIR, SKELETON_DIR, VIDEO_EXTENSIONS
from .manifest import meta_from_path


REQUIRED_ARRAYS = ("keypoints", "scores", "bboxes", "bbox_scores", "frame_indices")
BODY_KEYPOINTS = 17


@dataclass(frozen=True)
class QualityThresholds:
    two_person_actions: tuple = tuple(range(50, 61))
    default_expected_persons: int = 1
    keypoint_score_threshold: float = 0.1
    presence_body_keypoints: int = 5
    complete_body_keypoints: int = 15
    single_person_min_recall: float = 0.95
    two_person_min_recall: float = 0.85
    max_missing_run: int = 10
    min_body_complete_rate: float = 0.80
    large_jump_ratio: float = 0.35
    max_large_jump_rate: float = 0.02
    slot_jump_ratio: float = 0.50
    max_slot_jump_rate: float = 0.05
    max_score_coordinate_mismatch_rate: float = 0.0


def parse_integer_spec(value):
    """Parse values such as ``50-60,106,107`` into a sorted tuple."""
    values = set()
    if value is None:
        return tuple()
    for token in str(value).split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(token))
    return tuple(sorted(values))


def metadata_from_npz(data, path):
    metadata = {}
    if "metadata" in data.files:
        try:
            raw = data["metadata"]
            if getattr(raw, "shape", None) == ():
                raw = raw.item()
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            if isinstance(raw, str):
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    metadata.update(parsed)
        except (TypeError, ValueError, json.JSONDecodeError):
            pass
    filename_meta = meta_from_path(path)
    if filename_meta:
        for key, value in filename_meta.items():
            metadata.setdefault(key, value)
    return metadata


def longest_true_run(mask):
    longest = 0
    current = 0
    for value in mask:
        if bool(value):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def compact_ranges(mask):
    indices = [index for index, value in enumerate(mask) if bool(value)]
    if not indices:
        return []
    ranges = []
    start = previous = indices[0]
    for value in indices[1:]:
        if value != previous + 1:
            ranges.append([start, previous])
            start = value
        previous = value
    ranges.append([start, previous])
    return ranges


def expected_person_count(metadata, thresholds):
    action = metadata.get("action")
    try:
        action = int(action)
    except (TypeError, ValueError):
        action = None
    if action in set(thresholds.two_person_actions):
        return 2
    return max(1, int(thresholds.default_expected_persons))


def _safe_ratio(numerator, denominator, default=0.0):
    return float(numerator) / float(denominator) if denominator else float(default)


def _round_metrics(metrics):
    rounded = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            rounded[key] = round(value, 6)
        else:
            rounded[key] = value
    return rounded


def evaluate_npz(path, thresholds=None):
    import numpy as np

    thresholds = thresholds or QualityThresholds()
    path = Path(path)
    result = {
        "path": str(path.resolve()),
        "status": "error",
        "quality_score": 0.0,
        "expected_persons": None,
        "metadata": {},
        "metrics": {},
        "reasons": [],
        "warnings": [],
    }

    try:
        data_context = np.load(path, allow_pickle=False)
    except Exception as exc:
        result["reasons"].append("cannot_load_npz: {}".format(exc))
        return result

    try:
        with data_context as data:
            result["metadata"] = metadata_from_npz(data, path)
            expected = expected_person_count(result["metadata"], thresholds)
            result["expected_persons"] = expected

            missing_arrays = [name for name in REQUIRED_ARRAYS if name not in data.files]
            if missing_arrays:
                result["reasons"].append("missing_arrays: {}".format(",".join(missing_arrays)))
                return result

            keypoints = np.asarray(data["keypoints"])
            scores = np.asarray(data["scores"])
            bboxes = np.asarray(data["bboxes"])
            bbox_scores = np.asarray(data["bbox_scores"])
            frame_indices = np.asarray(data["frame_indices"])

            structural = []
            if keypoints.ndim != 4 or keypoints.shape[-1] != 2:
                structural.append("keypoints_shape={}".format(tuple(keypoints.shape)))
            if keypoints.ndim == 4 and scores.shape != keypoints.shape[:-1]:
                structural.append("scores_shape={}".format(tuple(scores.shape)))
            if keypoints.ndim == 4:
                frames, persons, points, _ = keypoints.shape
                if bboxes.shape != (frames, persons, 4):
                    structural.append("bboxes_shape={}".format(tuple(bboxes.shape)))
                if bbox_scores.shape != (frames, persons):
                    structural.append("bbox_scores_shape={}".format(tuple(bbox_scores.shape)))
                if frame_indices.shape != (frames,):
                    structural.append("frame_indices_shape={}".format(tuple(frame_indices.shape)))
                if points < BODY_KEYPOINTS:
                    structural.append("keypoint_count={}".format(points))
                if persons < expected:
                    structural.append("person_slots={} expected={}".format(persons, expected))
                if frames == 0:
                    structural.append("empty_sequence")
            if structural:
                result["reasons"].extend("invalid_structure: {}".format(item) for item in structural)
                return result

            frames, persons, points, _ = keypoints.shape
            finite_points = np.isfinite(keypoints).all(axis=-1)
            finite_scores = np.isfinite(scores)
            finite_bboxes = np.isfinite(bboxes).all(axis=-1)
            bbox_widths = bboxes[..., 2] - bboxes[..., 0]
            bbox_heights = bboxes[..., 3] - bboxes[..., 1]
            valid_bbox_geometry = finite_bboxes & (bbox_widths > 0) & (bbox_heights > 0)
            bbox_active = (bbox_scores > 0) & valid_bbox_geometry

            score_positive = finite_scores & (scores > 0)
            mismatch = score_positive & ~finite_points
            mismatch_rate = _safe_ratio(int(mismatch.sum()), int(score_positive.sum()))
            invalid_active_bboxes = (bbox_scores > 0) & ~valid_bbox_geometry

            body_valid = (
                finite_points[:, :, :BODY_KEYPOINTS]
                & finite_scores[:, :, :BODY_KEYPOINTS]
                & (scores[:, :, :BODY_KEYPOINTS] >= thresholds.keypoint_score_threshold)
            )
            body_counts = body_valid.sum(axis=-1)
            active = bbox_active & (body_counts >= thresholds.presence_body_keypoints)
            persons_per_frame = active.sum(axis=1)
            missing_expected = persons_per_frame < expected
            min_recall = (
                thresholds.two_person_min_recall
                if expected >= 2
                else thresholds.single_person_min_recall
            )
            expected_person_recall = float(np.mean(persons_per_frame >= expected))
            longest_missing = longest_true_run(missing_expected)
            no_person_rate = float(np.mean(persons_per_frame == 0))

            active_instances = int(active.sum())
            complete_instances = int((active & (body_counts >= thresholds.complete_body_keypoints)).sum())
            body_complete_rate = _safe_ratio(complete_instances, active_instances)
            mean_valid_body = (
                float(body_counts[active].mean()) if active_instances else 0.0
            )

            bbox_diagonal = np.sqrt(np.maximum(bbox_widths, 0) ** 2 + np.maximum(bbox_heights, 0) ** 2)
            positive_diagonal = bbox_diagonal[np.isfinite(bbox_diagonal) & (bbox_diagonal > 0)]
            fallback_diagonal = float(np.median(positive_diagonal)) if positive_diagonal.size else 1.0
            transition_scale = (bbox_diagonal[1:] + bbox_diagonal[:-1]) * 0.5
            transition_scale = np.where(
                np.isfinite(transition_scale) & (transition_scale > 0),
                transition_scale,
                fallback_diagonal,
            )

            body_displacements = np.linalg.norm(
                keypoints[1:, :, :BODY_KEYPOINTS] - keypoints[:-1, :, :BODY_KEYPOINTS],
                axis=-1,
            )
            common_body = body_valid[1:] & body_valid[:-1] & active[1:, :, None] & active[:-1, :, None]
            body_jump_ratios = body_displacements / transition_scale[:, :, None]
            comparable_body = int(common_body.sum())
            large_body_jumps = common_body & (body_jump_ratios > thresholds.large_jump_ratio)
            large_jump_rate = _safe_ratio(int(large_body_jumps.sum()), comparable_body)
            max_body_jump_ratio = (
                float(np.max(body_jump_ratios[common_body])) if comparable_body else 0.0
            )

            centers = (bboxes[..., :2] + bboxes[..., 2:]) * 0.5
            center_displacements = np.linalg.norm(centers[1:] - centers[:-1], axis=-1)
            common_slots = active[1:] & active[:-1]
            slot_jump_ratios = center_displacements / transition_scale
            comparable_slots = int(common_slots.sum())
            large_slot_jumps = common_slots & (slot_jump_ratios > thresholds.slot_jump_ratio)
            slot_jump_rate = _safe_ratio(int(large_slot_jumps.sum()), comparable_slots)
            max_slot_jump_ratio = (
                float(np.max(slot_jump_ratios[common_slots])) if comparable_slots else 0.0
            )

            frame_indices_contiguous = bool(
                frames > 0
                and int(frame_indices[0]) == 0
                and np.array_equal(frame_indices, np.arange(frames, dtype=frame_indices.dtype))
            )

            metrics = {
                "frames": int(frames),
                "person_slots": int(persons),
                "keypoints_per_person": int(points),
                "expected_person_recall": expected_person_recall,
                "frames_meeting_expected_persons": int((persons_per_frame >= expected).sum()),
                "frames_below_expected_persons": int(missing_expected.sum()),
                "missing_frame_ranges": compact_ranges(missing_expected),
                "longest_missing_run": longest_missing,
                "no_person_rate": no_person_rate,
                "active_person_instances": active_instances,
                "body_complete_rate": body_complete_rate,
                "mean_valid_body_keypoints": mean_valid_body,
                "large_body_jump_rate": large_jump_rate,
                "max_body_jump_ratio": max_body_jump_ratio,
                "large_slot_jump_rate": slot_jump_rate,
                "max_slot_jump_ratio": max_slot_jump_ratio,
                "score_coordinate_mismatch_rate": mismatch_rate,
                "invalid_active_bboxes": int(invalid_active_bboxes.sum()),
                "frame_indices_contiguous": frame_indices_contiguous,
            }
            result["metrics"] = _round_metrics(metrics)

            if not frame_indices_contiguous:
                result["reasons"].append("frame_indices_not_contiguous_from_zero")
            if not bool(finite_scores.all()):
                result["reasons"].append("non_finite_scores")
            if not bool(np.isfinite(bbox_scores).all()):
                result["reasons"].append("non_finite_bbox_scores")
            if int(invalid_active_bboxes.sum()) > 0:
                result["reasons"].append(
                    "invalid_active_bboxes={}".format(int(invalid_active_bboxes.sum()))
                )
            if mismatch_rate > thresholds.max_score_coordinate_mismatch_rate:
                result["reasons"].append(
                    "score_coordinate_mismatch_rate={:.4f}>{:.4f}".format(
                        mismatch_rate,
                        thresholds.max_score_coordinate_mismatch_rate,
                    )
                )
            if expected_person_recall < min_recall:
                result["reasons"].append(
                    "expected_person_recall={:.3f}<{:.3f}".format(
                        expected_person_recall,
                        min_recall,
                    )
                )
            if longest_missing > thresholds.max_missing_run:
                result["reasons"].append(
                    "longest_missing_run={} > {}".format(
                        longest_missing,
                        thresholds.max_missing_run,
                    )
                )
            if body_complete_rate < thresholds.min_body_complete_rate:
                result["reasons"].append(
                    "body_complete_rate={:.3f}<{:.3f}".format(
                        body_complete_rate,
                        thresholds.min_body_complete_rate,
                    )
                )
            if large_jump_rate > thresholds.max_large_jump_rate:
                result["reasons"].append(
                    "large_body_jump_rate={:.4f}>{:.4f}".format(
                        large_jump_rate,
                        thresholds.max_large_jump_rate,
                    )
                )
            if slot_jump_rate > thresholds.max_slot_jump_rate:
                result["reasons"].append(
                    "large_slot_jump_rate={:.4f}>{:.4f}".format(
                        slot_jump_rate,
                        thresholds.max_slot_jump_rate,
                    )
                )

            video_path = result["metadata"].get("video_path")
            if video_path and not Path(str(video_path)).exists():
                result["warnings"].append("metadata_video_path_not_found")

            recall_component = min(1.0, expected_person_recall / max(min_recall, 1e-9))
            body_component = min(
                1.0,
                body_complete_rate / max(thresholds.min_body_complete_rate, 1e-9),
            )
            jump_component = max(
                0.0,
                1.0 - large_jump_rate / max(thresholds.max_large_jump_rate, 1e-9),
            )
            slot_component = max(
                0.0,
                1.0 - slot_jump_rate / max(thresholds.max_slot_jump_rate, 1e-9),
            )
            missing_component = max(
                0.0,
                1.0 - longest_missing / max(float(thresholds.max_missing_run), 1.0),
            )
            quality_score = 100.0 * (
                0.45 * recall_component
                + 0.25 * body_component
                + 0.12 * jump_component
                + 0.08 * slot_component
                + 0.10 * missing_component
            )
            result["quality_score"] = round(quality_score, 3)
            result["status"] = "pass" if not result["reasons"] else "fail"
            return result
    except Exception as exc:
        result["reasons"].append("evaluation_error: {}".format(exc))
        return result


def find_npz_files(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path] if input_path.suffix.lower() == ".npz" else []
    return sorted(path for path in input_path.rglob("*.npz") if path.is_file())


def scan_skeletons(input_path, thresholds=None, workers=4):
    thresholds = thresholds or QualityThresholds()
    paths = find_npz_files(input_path)
    if not paths:
        return []
    workers = min(max(1, int(workers or 1)), len(paths))
    if workers == 1:
        return [evaluate_npz(path, thresholds) for path in paths]

    indexed_results = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(evaluate_npz, path, thresholds): index
            for index, path in enumerate(paths)
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                indexed_results[index] = future.result()
            except Exception as exc:
                indexed_results[index] = {
                    "path": str(paths[index].resolve()),
                    "status": "error",
                    "quality_score": 0.0,
                    "expected_persons": None,
                    "metadata": {},
                    "metrics": {},
                    "reasons": ["worker_error: {}".format(exc)],
                    "warnings": [],
                }
    return [indexed_results[index] for index in range(len(paths))]


def report_summary(results):
    return {
        "total": len(results),
        "passed": sum(result["status"] == "pass" for result in results),
        "failed": sum(result["status"] == "fail" for result in results),
        "errors": sum(result["status"] == "error" for result in results),
        "average_quality_score": round(
            sum(float(result.get("quality_score", 0.0)) for result in results) / len(results),
            3,
        ) if results else 0.0,
    }


def write_reports(results, thresholds, report_dir, prefix="skeleton_quality"):
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "{}.json".format(prefix)
    csv_path = report_dir / "{}.csv".format(prefix)
    failed_path = report_dir / "{}_failed.txt".format(prefix)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "thresholds": asdict(thresholds),
        "summary": report_summary(results),
        "results": results,
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    fields = [
        "status",
        "quality_score",
        "path",
        "action",
        "expected_persons",
        "frames",
        "expected_person_recall",
        "longest_missing_run",
        "body_complete_rate",
        "large_body_jump_rate",
        "large_slot_jump_rate",
        "reasons",
        "warnings",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for result in results:
            metrics = result.get("metrics", {})
            metadata = result.get("metadata", {})
            writer.writerow({
                "status": result.get("status"),
                "quality_score": result.get("quality_score"),
                "path": result.get("path"),
                "action": metadata.get("action"),
                "expected_persons": result.get("expected_persons"),
                "frames": metrics.get("frames"),
                "expected_person_recall": metrics.get("expected_person_recall"),
                "longest_missing_run": metrics.get("longest_missing_run"),
                "body_complete_rate": metrics.get("body_complete_rate"),
                "large_body_jump_rate": metrics.get("large_body_jump_rate"),
                "large_slot_jump_rate": metrics.get("large_slot_jump_rate"),
                "reasons": "; ".join(result.get("reasons", [])),
                "warnings": "; ".join(result.get("warnings", [])),
            })

    failed = [result["path"] for result in results if result["status"] != "pass"]
    failed_path.write_text("\n".join(failed) + ("\n" if failed else ""), encoding="utf-8")
    return json_path, csv_path, failed_path


def build_video_index(video_root):
    index = {}
    video_root = Path(video_root)
    for path in video_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            index.setdefault(path.stem.lower(), []).append(path)
            index.setdefault(path.name.lower(), []).append(path)
    return index


def locate_source_video(result, skeleton_root, video_root, video_index):
    metadata = result.get("metadata", {})
    stored_path = metadata.get("video_path")
    if stored_path:
        candidate = Path(str(stored_path))
        if candidate.exists():
            return candidate

    skeleton_path = Path(result["path"])
    skeleton_root = Path(skeleton_root)
    video_root = Path(video_root)
    if skeleton_root.is_dir():
        try:
            relative = skeleton_path.relative_to(skeleton_root)
        except ValueError:
            relative = Path(skeleton_path.name)
    else:
        relative = Path(skeleton_path.name)

    for suffix in sorted(VIDEO_EXTENSIONS):
        candidate = (video_root / relative).with_suffix(suffix)
        if candidate.exists():
            return candidate

    video_name = metadata.get("video_name")
    lookup_keys = []
    if video_name:
        lookup_keys.append(str(video_name).lower())
        lookup_keys.append(Path(str(video_name)).stem.lower())
    lookup_keys.append(skeleton_path.stem.lower())
    for key in lookup_keys:
        matches = list(dict.fromkeys(video_index.get(key, [])))
        if len(matches) == 1:
            return matches[0]
    return None


def _retry_relative_path(result, skeleton_root):
    path = Path(result["path"])
    skeleton_root = Path(skeleton_root)
    if skeleton_root.is_dir():
        try:
            return path.relative_to(skeleton_root)
        except ValueError:
            pass
    return Path(path.name)


def reextract_failed(
    results,
    thresholds,
    skeleton_root,
    video_root,
    retry_output,
    device="auto",
    profile="relaxed",
    pose_batch_size=1,
    cpu_threads=0,
    retry_limit=None,
    replace_if_better=False,
):
    failed = [result for result in results if result["status"] != "pass"]
    if retry_limit is not None:
        failed = failed[:max(0, int(retry_limit))]
    if not failed:
        return []

    video_root = Path(video_root)
    retry_output = Path(retry_output)
    retry_output.mkdir(parents=True, exist_ok=True)
    video_index = build_video_index(video_root)

    retry_jobs = []
    retry_records = []
    for result in failed:
        video = locate_source_video(result, skeleton_root, video_root, video_index)
        relative = _retry_relative_path(result, skeleton_root)
        output = retry_output / relative
        if video is None:
            retry_records.append({
                "source_npz": result["path"],
                "status": "video_not_found",
                "retry_npz": str(output.resolve()),
            })
            continue
        retry_jobs.append((result, video, output, relative))

    if not retry_jobs:
        return retry_records

    from . import extract
    from .constants import RTMDET_WEIGHTS_PATH, RTMW_WEIGHTS_PATH
    from .device import resolve_device
    from .download import ensure_rtmdet_weights, ensure_rtmw_weights

    extract.ensure_supported_python()
    extract.ensure_openmmlab_ready()
    parser = extract.parser()
    args = parser.parse_args([
        "--input", str(retry_jobs[0][1]),
        "--output", str(retry_output),
        "--device", str(device),
        "--workers", "1",
        "--cpu-workers", "0",
        "--pose-batch-size", str(max(1, int(pose_batch_size or 1))),
        "--cpu-threads", str(max(0, int(cpu_threads or 0))),
    ])
    args.device = resolve_device(args.device)
    args.show_skeleton = False
    args.visualize_dir = None
    args.skip_existing = False
    if profile == "relaxed":
        args.output_bbox_margin = 0.10
        args.temporal_min_frames = 1
        args.temporal_max_jump = 0.0
        args.temporal_min_keypoints = 0
    elif profile != "standard":
        raise ValueError("Unknown retry profile: {}".format(profile))

    if str(args.pose2d_weights) == str(RTMW_WEIGHTS_PATH):
        args.pose2d_weights = str(
            RTMW_WEIGHTS_PATH if Path(RTMW_WEIGHTS_PATH).exists() else ensure_rtmw_weights()
        )
    if args.det_model != "whole_image" and str(args.det_weights) == str(RTMDET_WEIGHTS_PATH):
        args.det_weights = str(
            RTMDET_WEIGHTS_PATH if Path(RTMDET_WEIGHTS_PATH).exists() else ensure_rtmdet_weights()
        )

    extract.configure_cpu_threads(args)
    print("retry device {} profile {}".format(args.device, profile), flush=True)
    inferencer = extract.build_inferencer(args)

    for index, (original, video, output, relative) in enumerate(retry_jobs, 1):
        record = {
            "source_npz": original["path"],
            "source_video": str(video.resolve()),
            "retry_npz": str(output.resolve()),
            "profile": profile,
            "status": "error",
        }
        print("retry [{}/{}] {}".format(index, len(retry_jobs), video), flush=True)
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            arrays = extract.infer_video(inferencer, video, args)
            extract.save_npz(output, video, arrays, args)
            retried = evaluate_npz(output, thresholds)
            record["result"] = retried
            record["status"] = retried["status"]

            better = (
                retried["status"] == "pass"
                and float(retried.get("quality_score", 0.0)) > float(original.get("quality_score", 0.0))
            )
            record["better"] = better
            if replace_if_better and better:
                original_path = Path(original["path"])
                backup = retry_output / "original_backup" / relative
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(original_path, backup)
                shutil.copy2(output, original_path)
                record["replaced"] = True
                record["backup"] = str(backup.resolve())
            else:
                record["replaced"] = False
        except Exception as exc:
            record["error"] = str(exc)
        retry_records.append(record)
    return retry_records


def parser():
    p = argparse.ArgumentParser(
        description="Check RTMW skeleton .npz quality and optionally re-extract failed samples."
    )
    p.add_argument("input", nargs="?", default=str(SKELETON_DIR), help="Skeleton .npz file or directory.")
    p.add_argument("--report-dir", default="data/quality_reports")
    p.add_argument("--report-prefix", default="skeleton_quality")
    p.add_argument("--workers", type=int, default=4, help="Parallel workers used only for .npz checks.")
    p.add_argument("--two-person-actions", default="50-60")
    p.add_argument("--default-expected-persons", type=int, default=1)
    p.add_argument("--kpt-thr", type=float, default=0.1)
    p.add_argument("--presence-body-keypoints", type=int, default=5)
    p.add_argument("--complete-body-keypoints", type=int, default=15)
    p.add_argument("--single-person-min-recall", type=float, default=0.95)
    p.add_argument("--two-person-min-recall", type=float, default=0.85)
    p.add_argument("--max-missing-run", type=int, default=10)
    p.add_argument("--min-body-complete-rate", type=float, default=0.80)
    p.add_argument("--large-jump-ratio", type=float, default=0.35)
    p.add_argument("--max-large-jump-rate", type=float, default=0.02)
    p.add_argument("--slot-jump-ratio", type=float, default=0.50)
    p.add_argument("--max-slot-jump-rate", type=float, default=0.05)
    p.add_argument("--reextract-failed", action="store_true")
    p.add_argument("--video-root", default=str(EXTRACTED_DIR))
    p.add_argument("--retry-output")
    p.add_argument("--retry-profile", choices=["standard", "relaxed"], default="relaxed")
    p.add_argument("--retry-limit", type=int)
    p.add_argument("--device", default="auto")
    p.add_argument("--pose-batch-size", type=int, default=1)
    p.add_argument("--cpu-threads", type=int, default=0)
    p.add_argument(
        "--replace-if-better",
        action="store_true",
        help="Back up and replace an original only when the retried file passes and scores higher.",
    )
    p.add_argument("--no-fail-exit", action="store_true")
    return p


def thresholds_from_args(args):
    return QualityThresholds(
        two_person_actions=parse_integer_spec(args.two_person_actions),
        default_expected_persons=args.default_expected_persons,
        keypoint_score_threshold=args.kpt_thr,
        presence_body_keypoints=args.presence_body_keypoints,
        complete_body_keypoints=args.complete_body_keypoints,
        single_person_min_recall=args.single_person_min_recall,
        two_person_min_recall=args.two_person_min_recall,
        max_missing_run=args.max_missing_run,
        min_body_complete_rate=args.min_body_complete_rate,
        large_jump_ratio=args.large_jump_ratio,
        max_large_jump_rate=args.max_large_jump_rate,
        slot_jump_ratio=args.slot_jump_ratio,
        max_slot_jump_rate=args.max_slot_jump_rate,
    )


def main(argv=None):
    args = parser().parse_args(argv)
    thresholds = thresholds_from_args(args)
    print("checking {}".format(Path(args.input)), flush=True)
    results = scan_skeletons(args.input, thresholds, args.workers)
    if not results:
        raise SystemExit("No .npz skeleton files found under {}".format(args.input))

    retry_records = []
    if args.reextract_failed:
        input_path = Path(args.input)
        retry_output = Path(args.retry_output) if args.retry_output else (
            input_path.parent / "{}_retry".format(input_path.name)
            if input_path.is_dir()
            else Path("data/reextracted_skeletons")
        )
        retry_records = reextract_failed(
            results,
            thresholds,
            args.input,
            args.video_root,
            retry_output,
            device=args.device,
            profile=args.retry_profile,
            pose_batch_size=args.pose_batch_size,
            cpu_threads=args.cpu_threads,
            retry_limit=args.retry_limit,
            replace_if_better=args.replace_if_better,
        )
        if retry_records:
            for result in results:
                matching = [record for record in retry_records if record["source_npz"] == result["path"]]
                if matching:
                    record = matching[0]
                    if record.get("replaced") and record.get("result"):
                        original_path = result["path"]
                        replacement = dict(record["result"])
                        replacement["path"] = original_path
                        result.clear()
                        result.update(replacement)
                    result["retry"] = record

    json_path, csv_path, failed_path = write_reports(
        results,
        thresholds,
        args.report_dir,
        args.report_prefix,
    )
    summary = report_summary(results)
    print("summary {}".format(summary), flush=True)
    print("json {}".format(json_path.resolve()), flush=True)
    print("csv {}".format(csv_path.resolve()), flush=True)
    print("failed list {}".format(failed_path.resolve()), flush=True)

    if not args.no_fail_exit and (summary["failed"] or summary["errors"]):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
