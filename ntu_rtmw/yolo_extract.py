"""YOLO26-X person detection followed by RTMW top-down pose extraction."""

from pathlib import Path

from .camera import YoloPersonDetector, find_rtmw_config_path
from .extract import (
    assign_slots,
    expected_person_limit,
    postprocess_arrays,
    require_numpy,
    select_instances,
)


def ensure_ready():
    missing = []
    for module_name in ("cv2", "mmpose", "ultralytics"):
        try:
            __import__(module_name)
        except ImportError:
            missing.append(module_name)
    if missing:
        raise SystemExit(
            "Missing YOLO retry dependencies: {}\n"
            "Install project requirements in the Python 3.10 OpenMMLab environment."
            .format(", ".join(missing))
        )


def build_inferencer(args):
    from mmpose.apis import init_model

    model_path = Path(args.yolo_model)
    if not model_path.exists():
        raise SystemExit(
            "Cannot find YOLO26-X weights: {}\n"
            "Copy yolo26x.pt into the project models directory or pass "
            "--retry-yolo-model."
            .format(model_path)
        )
    detector = YoloPersonDetector(
        str(model_path),
        args.yolo_conf,
        args.yolo_iou,
        args.yolo_imgsz,
        args.device,
    )
    pose_model = init_model(
        find_rtmw_config_path(),
        args.pose2d_weights,
        device=args.device,
    )
    return detector, pose_model


def _pose_item(pose_model, frame, detection, args, inference_topdown):
    import numpy as np

    height, width = frame.shape[:2]
    x1, y1, x2, y2 = detection[:4].astype(float)
    bbox_width = max(1.0, x2 - x1)
    bbox_height = max(1.0, y2 - y1)
    margin_x = bbox_width * float(args.crop_margin)
    margin_y = bbox_height * float(args.crop_margin)
    crop_x1 = int(max(0, np.floor(x1 - margin_x)))
    crop_y1 = int(max(0, np.floor(y1 - margin_y)))
    crop_x2 = int(min(width, np.ceil(x2 + margin_x)))
    crop_y2 = int(min(height, np.ceil(y2 + margin_y)))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    crop_height, crop_width = crop.shape[:2]
    samples = inference_topdown(
        pose_model,
        crop,
        np.array([[0, 0, crop_width, crop_height]], dtype=np.float32),
        bbox_format="xyxy",
    )
    if not samples:
        return None

    pred = samples[0].pred_instances.cpu().numpy()
    if not len(pred.keypoints):
        return None
    keypoints = pred.keypoints[0].astype(np.float32)
    keypoints[:, 0] += float(crop_x1)
    keypoints[:, 1] += float(crop_y1)
    if hasattr(pred, "keypoint_scores") and len(pred.keypoint_scores):
        scores = pred.keypoint_scores[0].astype(np.float32)
    else:
        scores = np.ones((keypoints.shape[0],), dtype=np.float32)
    return {
        "keypoints": keypoints,
        "keypoint_scores": scores,
        "bbox": detection[:4].astype(np.float32),
        "bbox_score": float(detection[4]),
    }


def infer_video(inferencer, video, args):
    import cv2
    import numpy as np
    from mmpose.apis import inference_topdown

    require_numpy()
    detector, pose_model = inferencer
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError("Cannot open video: {}".format(video))

    keypoints = []
    scores = []
    bboxes = []
    bbox_scores = []
    previous = None
    frame_idx = 0
    person_limit = expected_person_limit(video, args)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            detections = detector(frame)
            if len(detections):
                detections = detections[detections[:, 4] >= float(args.bbox_thr)]
                if len(detections):
                    order = np.argsort(-detections[:, 4])
                    detections = detections[order[:person_limit]]

            raw_instances = []
            for detection in detections:
                item = _pose_item(
                    pose_model,
                    frame,
                    detection,
                    args,
                    inference_topdown,
                )
                if item is not None:
                    raw_instances.append(item)

            items = select_instances(raw_instances, args.bbox_thr, args.kpt_thr)
            slots, previous = assign_slots(
                items,
                previous,
                args.max_persons,
                args.tracking_distance,
            )
            frame_keypoints = np.full(
                (args.max_persons, 133, 2), np.nan, dtype=np.float32
            )
            frame_scores = np.zeros((args.max_persons, 133), dtype=np.float32)
            frame_bboxes = np.full(
                (args.max_persons, 4), np.nan, dtype=np.float32
            )
            frame_bbox_scores = np.zeros((args.max_persons,), dtype=np.float32)
            for person_idx, item in enumerate(slots):
                if item is None:
                    continue
                count = min(133, item["keypoints"].shape[0])
                frame_keypoints[person_idx, :count] = item["keypoints"][:count, :2]
                frame_scores[person_idx, :count] = item["scores"][:count]
                frame_bboxes[person_idx] = item["bbox"]
                frame_bbox_scores[person_idx] = item["bbox_score"]

            keypoints.append(frame_keypoints)
            scores.append(frame_scores)
            bboxes.append(frame_bboxes)
            bbox_scores.append(frame_bbox_scores)
            frame_idx += 1
            if frame_idx % 100 == 0:
                print("  {} frames".format(frame_idx), flush=True)
    finally:
        cap.release()

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
    return postprocess_arrays(arrays, video, args)
