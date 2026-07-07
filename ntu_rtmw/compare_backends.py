import argparse
import json
import time
from argparse import Namespace
from pathlib import Path

from .camera import (
    PoseWorker,
    YoloPersonDetector,
    filter_pose_output_to_bboxes,
    find_rtmw_config_path,
    parser as camera_parser,
)
from .constants import (
    EXTRACTED_DIR,
    RTMDET_CONFIG,
    RTMDET_WEIGHTS_PATH,
    RTMW_WEIGHTS_PATH,
    YOLO_WEIGHTS_PATH,
)
from .extract import draw_skeleton, require_numpy, select_instances


VARIANTS = [
    ("rtmdet_frame", "RTMDet frame"),
    ("rtmdet_crop", "RTMDet crop"),
    ("yolo_frame", "YOLO frame"),
    ("yolo_crop", "YOLO crop"),
]


def first_video(root):
    videos = sorted(Path(root).rglob("*_rgb.avi"))
    if not videos:
        raise SystemExit("No *_rgb.avi found under {}".format(root))
    return videos[0]


def read_frames(video, limit, stride):
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit("Cannot open video: {}".format(video))
    frames = []
    idx = 0
    while len(frames) < limit:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append((idx, frame))
        idx += 1
    cap.release()
    return frames


class RTMDetPersonDetector:
    def __init__(self, device):
        from mmdet.apis import DetInferencer

        self.detector = DetInferencer(
            model=RTMDET_CONFIG,
            weights=str(RTMDET_WEIGHTS_PATH),
            device=device,
            show_progress=False,
        )

    def __call__(self, frame, bbox_thr):
        import numpy as np

        result = self.detector(
            frame,
            return_datasamples=True,
            pred_score_thr=bbox_thr,
            no_save_pred=True,
            no_save_vis=True,
            return_vis=False,
            show=False,
        )
        samples = result.get("predictions", [])
        if not samples:
            return np.empty((0, 5), dtype=np.float32)
        inst = samples[0].pred_instances.cpu().numpy()
        if len(inst.bboxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        labels = inst.labels
        scores = inst.scores
        keep = (labels == 0) & (scores >= bbox_thr)
        if not keep.any():
            return np.empty((0, 5), dtype=np.float32)
        return np.concatenate(
            [inst.bboxes[keep].astype(np.float32), scores[keep].astype(np.float32).reshape(-1, 1)],
            axis=1,
        )


def build_pose_model(device):
    from mmpose.apis import init_model

    return init_model(find_rtmw_config_path(), str(RTMW_WEIGHTS_PATH), device=device)


def build_worker_args(args, detector_name, pose_input):
    worker_args = camera_parser().parse_args([])
    worker_args.device = args.device
    worker_args.det_backend = detector_name
    worker_args.max_persons = args.max_persons
    worker_args.bbox_thr = args.bbox_thr
    worker_args.kpt_thr = args.kpt_thr
    worker_args.tracking_distance = args.tracking_distance
    worker_args.pose_input = pose_input
    worker_args.crop_margin = args.crop_margin
    worker_args.filter_output_to_bbox = args.filter_output_to_bbox
    worker_args.output_bbox_margin = args.output_bbox_margin
    worker_args.yolo_conf = args.yolo_conf
    worker_args.yolo_iou = args.yolo_iou
    worker_args.yolo_imgsz = args.yolo_imgsz
    worker_args.yolo_model = str(YOLO_WEIGHTS_PATH)
    return worker_args


def infer_pose_on_crops(pose_model, frame, bboxes, crop_margin):
    import numpy as np
    from mmpose.apis import inference_topdown

    height, width = frame.shape[:2]
    pose_results = []
    offsets = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.astype(float)
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        mx = bw * crop_margin
        my = bh * crop_margin
        cx1 = int(max(0, np.floor(x1 - mx)))
        cy1 = int(max(0, np.floor(y1 - my)))
        cx2 = int(min(width, np.ceil(x2 + mx)))
        cy2 = int(min(height, np.ceil(y2 + my)))
        if cx2 <= cx1 or cy2 <= cy1:
            continue
        crop = frame[cy1:cy2, cx1:cx2]
        crop_h, crop_w = crop.shape[:2]
        results = inference_topdown(
            pose_model,
            crop,
            np.array([[0, 0, crop_w, crop_h]], dtype=np.float32),
            bbox_format="xyxy",
        )
        if not results:
            continue
        pose_results.append(results[0])
        offsets.append((float(cx1), float(cy1)))
    return pose_results, offsets


def infer_pose_on_frame(pose_model, frame, bboxes):
    from mmpose.apis import inference_topdown

    results = inference_topdown(pose_model, frame, bboxes, bbox_format="xyxy")
    return results, [None] * len(results)


def samples_to_instances(detections, pose_results, offsets):
    import numpy as np

    raw = []
    for det, sample, offset in zip(detections, pose_results, offsets):
        pred = sample.pred_instances.cpu().numpy()
        keypoints = pred.keypoints[0] if len(pred.keypoints) else np.empty((0, 2), dtype=np.float32)
        if offset is not None and keypoints.size:
            keypoints = keypoints.copy()
            keypoints[:, 0] += offset[0]
            keypoints[:, 1] += offset[1]
        scores = pred.keypoint_scores[0] if hasattr(pred, "keypoint_scores") and len(pred.keypoint_scores) else (
            np.ones((keypoints.shape[0],), dtype=np.float32)
        )
        raw.append({
            "keypoints": keypoints,
            "keypoint_scores": scores,
            "bbox": det[:4],
            "bbox_score": float(det[4]),
        })
    return raw


def infer_variant(frames, variant_name, args):
    import numpy as np

    require_numpy()
    detector_kind, pose_input = variant_name.split("_", 1)
    worker_args = build_worker_args(args, detector_kind, pose_input)
    worker = PoseWorker(worker_args)
    pose_model = build_pose_model(args.device)
    detector = (
        RTMDetPersonDetector(args.device)
        if detector_kind == "rtmdet"
        else YoloPersonDetector(str(YOLO_WEIGHTS_PATH), args.yolo_conf, args.yolo_iou, args.yolo_imgsz, args.device)
    )

    outputs = []
    for idx, frame in frames:
        started = time.perf_counter()
        detections = detector(frame, args.bbox_thr) if detector_kind == "rtmdet" else detector(frame)
        detections = detections[detections[:, 4] >= args.bbox_thr]
        if len(detections) > 0:
            order = np.argsort(-detections[:, 4])
            detections = detections[order[: args.max_persons]]
            if pose_input == "crop":
                pose_results, offsets = infer_pose_on_crops(pose_model, frame, detections[:, :4], args.crop_margin)
            else:
                pose_results, offsets = infer_pose_on_frame(pose_model, frame, detections[:, :4])
        else:
            pose_results, offsets = [], []

        raw = samples_to_instances(detections, pose_results, offsets)
        latest = worker._instances_to_arrays(
            select_instances(raw, args.bbox_thr, args.kpt_thr),
            time.perf_counter() - started,
        )
        latest["backend"] = detector_kind
        latest["pose_input"] = pose_input
        if args.filter_output_to_bbox:
            latest = filter_pose_output_to_bboxes(latest, args.output_bbox_margin)
        outputs.append((idx, latest))
    return outputs


def finite_valid_points(output, kpt_thr):
    import numpy as np

    points = output["keypoints"][0]
    scores = output["scores"][0]
    valid = (scores >= kpt_thr) & np.isfinite(points).all(axis=1)
    return points[valid], valid


def bbox_iou(a, b):
    import numpy as np

    if np.isnan(a).any() or np.isnan(b).any():
        return None
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    denom = area_a + area_b - inter
    return inter / denom if denom else None


def mean(values):
    values = [x for x in values if x is not None]
    return sum(values) / len(values) if values else None


def summarize(outputs, kpt_thr):
    valid_counts = []
    bbox_scores = []
    persons = []
    times = []
    filtered_counts = []
    for _, out in outputs:
        points, _ = finite_valid_points(out, kpt_thr)
        valid_counts.append(len(points))
        bbox_scores.append(float(out["bbox_scores"][0]))
        persons.append(int(out["count"]))
        times.append(float(out["elapsed"]))
        filtered_counts.append(int(out.get("filtered_keypoints", 0)))
    return {
        "frames": len(outputs),
        "detected_frames": sum(1 for x in bbox_scores if x > 0),
        "avg_ms_per_frame": mean(times) * 1000 if times else None,
        "avg_persons": mean(persons),
        "avg_valid_keypoints_person0": mean(valid_counts),
        "avg_bbox_score_person0": mean(bbox_scores),
        "avg_filtered_keypoints": mean(filtered_counts),
    }


def compare_outputs(reference, candidate, kpt_thr):
    import numpy as np

    ious = []
    dists = []
    common_counts = []
    for (_, a), (_, b) in zip(reference, candidate):
        ious.append(bbox_iou(a["bboxes"][0], b["bboxes"][0]))
        pa = a["keypoints"][0]
        pb = b["keypoints"][0]
        sa = a["scores"][0]
        sb = b["scores"][0]
        valid = (sa >= kpt_thr) & (sb >= kpt_thr) & np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
        common_counts.append(int(valid.sum()))
        if valid.any():
            dists.append(float(np.linalg.norm(pa[valid] - pb[valid], axis=1).mean()))
    return {
        "avg_bbox_iou_person0": mean(ious),
        "avg_common_keypoints_person0": mean(common_counts),
        "avg_keypoint_l2_px_person0": mean(dists),
    }


def draw_label(frame, label, cv2):
    cv2.rectangle(frame, (8, 8), (500, 42), (0, 0, 0), -1)
    cv2.putText(frame, label, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def make_visuals(frames, outputs_by_variant, out_dir, kpt_thr, make_video):
    import cv2
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    first_image = None
    for i, (frame_idx, frame) in enumerate(frames):
        tiles = []
        for variant_name, label in VARIANTS:
            tile = frame.copy()
            out = outputs_by_variant[variant_name][i][1]
            draw_skeleton(tile, out["keypoints"], out["scores"], out["bboxes"], out["bbox_scores"], kpt_thr, cv2)
            draw_label(tile, "{}  frame {}".format(label, frame_idx), cv2)
            tiles.append(tile)
        top = np.concatenate(tiles[:2], axis=1)
        bottom = np.concatenate(tiles[2:], axis=1)
        merged = np.concatenate([top, bottom], axis=0)
        scale = min(1800 / merged.shape[1], 1200 / merged.shape[0], 1.0)
        if scale < 1:
            merged = cv2.resize(merged, (int(merged.shape[1] * scale), int(merged.shape[0] * scale)))
        if i == 0:
            first_image = out_dir / "comparison_first_frame.jpg"
            cv2.imwrite(str(first_image), merged)
            if make_video:
                writer = cv2.VideoWriter(
                    str(out_dir / "comparison.avi"),
                    cv2.VideoWriter_fourcc(*"MJPG"),
                    8,
                    (merged.shape[1], merged.shape[0]),
                )
        if writer is not None:
            writer.write(merged)
    if writer is not None:
        writer.release()
    return first_image


def parser():
    p = argparse.ArgumentParser(description="Compare detector and crop variants for RTMW pose extraction.")
    p.add_argument("--video")
    p.add_argument("--output-dir", default="data/comparisons/backend_compare_4way")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--device", default="cpu")
    p.add_argument("--max-persons", type=int, default=2)
    p.add_argument("--bbox-thr", type=float, default=0.3)
    p.add_argument("--kpt-thr", type=float, default=0.1)
    p.add_argument("--tracking-distance", type=float, default=120.0)
    p.add_argument("--yolo-conf", type=float, default=0.35)
    p.add_argument("--yolo-iou", type=float, default=0.5)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--crop-margin", type=float, default=0.0)
    p.add_argument("--filter-output-to-bbox", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-bbox-margin", type=float, default=0.0)
    p.add_argument("--reference", choices=[name for name, _ in VARIANTS], default="rtmdet_frame")
    p.add_argument("--video-preview", action="store_true")
    return p


def main():
    args = parser().parse_args()
    video = Path(args.video) if args.video else first_video(EXTRACTED_DIR)
    frames = read_frames(video, args.frames, args.stride)
    if not frames:
        raise SystemExit("No frames read from {}".format(video))

    print("video: {}".format(video), flush=True)
    print("frames: {} stride: {}".format(len(frames), args.stride), flush=True)
    outputs_by_variant = {}
    for variant_name, label in VARIANTS:
        print("running {}...".format(label), flush=True)
        outputs_by_variant[variant_name] = infer_variant(frames, variant_name, args)

    report = {
        "video": str(video),
        "frames": len(frames),
        "stride": args.stride,
        "reference": args.reference,
        "variants": {
            name: summarize(outputs, args.kpt_thr)
            for name, outputs in outputs_by_variant.items()
        },
        "against_reference": {
            name: compare_outputs(outputs_by_variant[args.reference], outputs, args.kpt_thr)
            for name, outputs in outputs_by_variant.items()
            if name != args.reference
        },
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "comparison_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    image_path = make_visuals(frames, outputs_by_variant, out_dir, args.kpt_thr, args.video_preview)

    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("report: {}".format(report_path.resolve()), flush=True)
    if image_path:
        print("image: {}".format(image_path.resolve()), flush=True)


if __name__ == "__main__":
    main()
