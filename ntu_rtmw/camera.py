import argparse
import threading
import time
from argparse import Namespace
from pathlib import Path

from .constants import (
    RTMDET_CONFIG,
    RTMDET_WEIGHTS_PATH,
    RTMW_CONFIG,
    RTMW_WEIGHTS_PATH,
    YOLO_WEIGHTS_PATH,
)
from .download import ensure_rtmdet_weights, ensure_rtmw_weights
from .extract import (
    assign_slots,
    build_inferencer,
    draw_skeleton,
    require_numpy,
    select_instances,
    unwrap_predictions,
)


def find_rtmw_config_path():
    import mmpose

    root = Path(mmpose.__file__).resolve().parent
    path = (
        root / ".mim" / "configs" / "wholebody_2d_keypoint" / "rtmpose"
        / "cocktail14" / "rtmw-l_8xb320-270e_cocktail14-384x288.py"
    )
    if not path.exists():
        raise FileNotFoundError("Cannot find RTMW config: {}".format(path))
    return str(path)


class YoloPersonDetector:
    def __init__(self, model_name, conf, iou, imgsz, device):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise SystemExit(
                "Missing ultralytics. Install it with:\n"
                "  py -3.10 -m pip install ultralytics"
            ) from exc
        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.device = None if device == "cpu" else device

    def __call__(self, frame):
        import numpy as np

        results = self.model.predict(
            frame,
            classes=[0],
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            device=self.device,
            verbose=False,
        )
        if not results:
            return np.empty((0, 5), dtype=np.float32)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return np.empty((0, 5), dtype=np.float32)
        xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
        conf = boxes.conf.detach().cpu().numpy().astype(np.float32).reshape(-1, 1)
        return np.concatenate([xyxy, conf], axis=1)


class PoseWorker:
    def __init__(self, args):
        self.args = args
        self.condition = threading.Condition()
        self.pending_frame = None
        self.latest = None
        self.previous = None
        self.stop_requested = False
        self.busy = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()
        return self

    def stop(self):
        with self.condition:
            self.stop_requested = True
            self.condition.notify_all()
        self.thread.join(timeout=2)

    def submit(self, frame):
        with self.condition:
            self.pending_frame = frame.copy()
            self.condition.notify_all()

    def snapshot(self):
        with self.condition:
            return self.latest, self.busy, self.error

    def _run(self):
        require_numpy()
        try:
            if self.args.det_backend == "yolo":
                inferencer = self._build_yolo_pipeline()
            else:
                inferencer = build_inferencer(Namespace(
                    pose2d=self.args.pose2d,
                    pose2d_weights=self.args.pose2d_weights,
                    det_model=self.args.det_model,
                    det_weights=self.args.det_weights,
                    device=self.args.device,
                ))
        except Exception as exc:
            with self.condition:
                self.error = exc
            return
        while True:
            with self.condition:
                while self.pending_frame is None and not self.stop_requested:
                    self.condition.wait()
                if self.stop_requested:
                    return
                frame = self.pending_frame
                self.pending_frame = None
                self.busy = True

            started = time.perf_counter()
            try:
                if self.args.det_backend == "yolo":
                    latest = self._infer_yolo(frame, inferencer, time.perf_counter())
                else:
                    result = next(iter(inferencer(
                        frame,
                        show=False,
                        return_vis=False,
                        draw_bbox=False,
                        kpt_thr=self.args.kpt_thr,
                    )))
                    latest = self._result_to_arrays(result, time.perf_counter() - started)
                if self.args.filter_output_to_bbox:
                    latest = filter_pose_output_to_bboxes(latest, self.args.output_bbox_margin)
                with self.condition:
                    self.latest = latest
                    self.error = None
            except Exception as exc:
                with self.condition:
                    self.error = exc
            finally:
                with self.condition:
                    self.busy = False

    def _build_yolo_pipeline(self):
        from mmpose.apis import init_model

        detector = YoloPersonDetector(
            self.args.yolo_model,
            self.args.yolo_conf,
            self.args.yolo_iou,
            self.args.yolo_imgsz,
            self.args.device,
        )
        pose_model = init_model(
            find_rtmw_config_path(),
            self.args.pose2d_weights,
            device=self.args.device,
        )
        return detector, pose_model

    def _infer_yolo(self, frame, pipeline, started):
        import numpy as np
        from mmpose.apis import inference_topdown

        require_numpy()
        detector, pose_model = pipeline
        detections = detector(frame)
        detections = detections[detections[:, 4] >= self.args.bbox_thr]
        if len(detections) > 0:
            order = np.argsort(-detections[:, 4])
            detections = detections[order[: self.args.max_persons]]
            if self.args.pose_input == "crop":
                pose_results, keypoint_offsets = self._infer_pose_on_crops(
                    pose_model,
                    frame,
                    detections[:, :4],
                    inference_topdown,
                )
            else:
                pose_results = inference_topdown(pose_model, frame, detections[:, :4], bbox_format="xyxy")
                keypoint_offsets = [None] * len(pose_results)
        else:
            pose_results = []
            keypoint_offsets = []

        raw_instances = []
        for det, sample, offset in zip(detections, pose_results, keypoint_offsets):
            pred = sample.pred_instances.cpu().numpy()
            keypoints = pred.keypoints[0] if len(pred.keypoints) else np.empty((0, 2), dtype=np.float32)
            if offset is not None and keypoints.size:
                keypoints = keypoints.copy()
                keypoints[:, 0] += offset[0]
                keypoints[:, 1] += offset[1]
            if hasattr(pred, "keypoint_scores") and len(pred.keypoint_scores):
                scores = pred.keypoint_scores[0]
            else:
                scores = np.ones((keypoints.shape[0],), dtype=np.float32)
            raw_instances.append({
                "keypoints": keypoints,
                "keypoint_scores": scores,
                "bbox": det[:4],
                "bbox_score": float(det[4]),
            })

        return self._instances_to_arrays(
            select_instances(raw_instances, self.args.bbox_thr, self.args.kpt_thr),
            time.perf_counter() - started,
        )

    def _infer_pose_on_crops(self, pose_model, frame, bboxes, inference_topdown):
        import numpy as np

        height, width = frame.shape[:2]
        pose_results = []
        offsets = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox.astype(float)
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)
            margin_x = bw * self.args.crop_margin
            margin_y = bh * self.args.crop_margin
            cx1 = int(max(0, np.floor(x1 - margin_x)))
            cy1 = int(max(0, np.floor(y1 - margin_y)))
            cx2 = int(min(width, np.ceil(x2 + margin_x)))
            cy2 = int(min(height, np.ceil(y2 + margin_y)))
            if cx2 <= cx1 or cy2 <= cy1:
                continue
            crop = frame[cy1:cy2, cx1:cx2]
            crop_h, crop_w = crop.shape[:2]
            crop_results = inference_topdown(
                pose_model,
                crop,
                np.array([[0, 0, crop_w, crop_h]], dtype=np.float32),
                bbox_format="xyxy",
            )
            if not crop_results:
                continue
            pose_results.append(crop_results[0])
            offsets.append((float(cx1), float(cy1)))
        return pose_results, offsets

    def _result_to_arrays(self, result, elapsed):
        items = select_instances(unwrap_predictions(result), self.args.bbox_thr, self.args.kpt_thr)
        return self._instances_to_arrays(items, elapsed)

    def _instances_to_arrays(self, items, elapsed):
        import numpy as np

        num_kpts = 133
        for item in items:
            if item["keypoints"].shape[0]:
                num_kpts = int(item["keypoints"].shape[0])
                break

        slots, self.previous = assign_slots(
            items,
            self.previous,
            self.args.max_persons,
            self.args.tracking_distance,
        )
        keypoints = np.full((self.args.max_persons, num_kpts, 2), np.nan, dtype=np.float32)
        scores = np.zeros((self.args.max_persons, num_kpts), dtype=np.float32)
        bboxes = np.full((self.args.max_persons, 4), np.nan, dtype=np.float32)
        bbox_scores = np.zeros((self.args.max_persons,), dtype=np.float32)
        for person_idx, item in enumerate(slots):
            if item is None:
                continue
            n = min(num_kpts, item["keypoints"].shape[0])
            keypoints[person_idx, :n] = item["keypoints"][:n, :2]
            scores[person_idx, :n] = item["scores"][:n]
            bboxes[person_idx] = item["bbox"]
            bbox_scores[person_idx] = item["bbox_score"]
        return {
            "keypoints": keypoints,
            "scores": scores,
            "bboxes": bboxes,
            "bbox_scores": bbox_scores,
            "elapsed": elapsed,
            "count": len(items),
            "backend": self.args.det_backend,
            "pose_input": getattr(self.args, "pose_input", "frame"),
        }


def resize_frame(frame, max_width, max_height):
    import cv2

    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame, 1.0
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return resized, scale


def scale_pose(latest, scale):
    if latest is None or scale == 1.0:
        return latest
    copied = dict(latest)
    copied["keypoints"] = latest["keypoints"].copy() * scale
    copied["bboxes"] = latest["bboxes"].copy() * scale
    return copied


def clip_pose_to_bboxes(latest, margin_ratio):
    if latest is None:
        return None

    import numpy as np

    clipped = dict(latest)
    clipped["scores"] = latest["scores"].copy()
    for person_idx, bbox in enumerate(latest["bboxes"]):
        if np.isnan(bbox).any():
            clipped["scores"][person_idx] = 0
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

        points = latest["keypoints"][person_idx]
        valid = (
            (points[:, 0] >= x1)
            & (points[:, 0] <= x2)
            & (points[:, 1] >= y1)
            & (points[:, 1] <= y2)
            & ~np.isnan(points[:, 0])
            & ~np.isnan(points[:, 1])
        )
        clipped["scores"][person_idx, ~valid] = 0
    return clipped


def filter_pose_output_to_bboxes(latest, margin_ratio):
    if latest is None:
        return None

    import numpy as np

    filtered = dict(latest)
    filtered["keypoints"] = latest["keypoints"].copy()
    filtered["scores"] = latest["scores"].copy()
    removed = 0
    for person_idx, bbox in enumerate(latest["bboxes"]):
        if np.isnan(bbox).any():
            removed += int((filtered["scores"][person_idx] > 0).sum())
            filtered["keypoints"][person_idx] = np.nan
            filtered["scores"][person_idx] = 0
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

        points = latest["keypoints"][person_idx]
        valid = (
            (points[:, 0] >= x1)
            & (points[:, 0] <= x2)
            & (points[:, 1] >= y1)
            & (points[:, 1] <= y2)
            & ~np.isnan(points[:, 0])
            & ~np.isnan(points[:, 1])
        )
        removed += int(((~valid) & (filtered["scores"][person_idx] > 0)).sum())
        filtered["keypoints"][person_idx, ~valid] = np.nan
        filtered["scores"][person_idx, ~valid] = 0

    filtered["filtered_to_bbox"] = True
    filtered["filtered_keypoints"] = removed
    return filtered


def draw_status(frame, latest, busy, error, cv2):
    if error is not None:
        text = "RTMW error: {}".format(str(error)[:80])
        color = (40, 40, 220)
    elif latest is None:
        text = "Initializing RTMW..."
        color = (0, 160, 220)
    else:
        ms = latest["elapsed"] * 1000
        status = "inferencing" if busy else "ready"
        backend = latest.get("backend", "mmdet")
        pose_input = latest.get("pose_input", "frame")
        filtered = " | bbox-filter {}".format(latest.get("filtered_keypoints", 0)) if latest.get("filtered_to_bbox") else ""
        text = "RTMW camera | {} | {}:{} | persons {} | {:.0f} ms{} | q/Esc quit".format(
            status,
            backend,
            pose_input,
            latest["count"],
            ms,
            filtered,
        )
        color = (0, 0, 0)
    cv2.rectangle(frame, (8, 8), (min(frame.shape[1] - 8, 820), 42), color, -1)
    cv2.putText(frame, text, (18, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)


def run(args):
    import cv2

    args.infer_every = max(1, args.infer_every)
    args.pose2d_weights = str(RTMW_WEIGHTS_PATH if RTMW_WEIGHTS_PATH.exists() else ensure_rtmw_weights())
    if args.det_backend == "mmdet":
        args.det_weights = str(RTMDET_WEIGHTS_PATH if RTMDET_WEIGHTS_PATH.exists() else ensure_rtmdet_weights())

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera index: {}".format(args.camera))
    if args.width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    window = "RTMW realtime camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.max_width, args.max_height)

    worker = PoseWorker(args).start()
    frame_idx = 0
    last_submit = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("camera frame read failed", flush=True)
                break
            if args.flip:
                frame = cv2.flip(frame, 1)

            now = time.perf_counter()
            latest, busy, error = worker.snapshot()
            if not busy and (frame_idx % args.infer_every == 0) and now - last_submit >= args.min_interval:
                worker.submit(frame)
                last_submit = now

            display, scale = resize_frame(frame.copy(), args.max_width, args.max_height)
            scaled = scale_pose(latest, scale)
            if args.clip_to_bbox:
                scaled = clip_pose_to_bboxes(scaled, args.bbox_margin)
            if scaled is not None:
                draw_skeleton(
                    display,
                    scaled["keypoints"],
                    scaled["scores"],
                    scaled["bboxes"],
                    scaled["bbox_scores"],
                    args.kpt_thr,
                    cv2,
                )
            draw_status(display, latest, busy, error, cv2)
            cv2.imshow(window, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            frame_idx += 1
    finally:
        worker.stop()
        cap.release()
        cv2.destroyAllWindows()


def parser():
    p = argparse.ArgumentParser(description="Show realtime RTMW skeletons from a local camera.")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--det-backend", choices=["yolo", "mmdet"], default="yolo")
    p.add_argument("--pose2d", default=RTMW_CONFIG)
    p.add_argument("--pose2d-weights", default=str(RTMW_WEIGHTS_PATH))
    p.add_argument("--det-model", default=RTMDET_CONFIG)
    p.add_argument("--det-weights", default=str(RTMDET_WEIGHTS_PATH))
    p.add_argument("--yolo-model", default=str(YOLO_WEIGHTS_PATH))
    p.add_argument("--yolo-conf", type=float, default=0.35)
    p.add_argument("--yolo-iou", type=float, default=0.5)
    p.add_argument("--yolo-imgsz", type=int, default=640)
    p.add_argument("--pose-input", choices=["crop", "frame"], default="crop")
    p.add_argument("--crop-margin", type=float, default=0.0)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--max-width", type=int, default=960)
    p.add_argument("--max-height", type=int, default=540)
    p.add_argument("--max-persons", type=int, default=2)
    p.add_argument("--bbox-thr", type=float, default=0.3)
    p.add_argument("--kpt-thr", type=float, default=0.1)
    p.add_argument("--tracking-distance", type=float, default=120.0)
    p.add_argument("--infer-every", type=int, default=1)
    p.add_argument("--min-interval", type=float, default=0.0)
    p.add_argument("--flip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--filter-output-to-bbox", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--output-bbox-margin", type=float, default=0.0)
    p.add_argument("--clip-to-bbox", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--bbox-margin", type=float, default=0.0)
    return p


def main():
    run(parser().parse_args())


if __name__ == "__main__":
    main()
