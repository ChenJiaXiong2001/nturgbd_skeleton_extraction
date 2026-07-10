import argparse
import threading
from argparse import Namespace
from pathlib import Path

from ntu_rtmw.constants import (
    EXTRACTED_DIR,
    RTMDET_CONFIG,
    RTMDET_WEIGHTS_PATH,
    RTMW_CONFIG,
    RTMW_WEIGHTS_PATH,
    SKELETON_DIR,
)
from ntu_rtmw.extract import run as extract_run


DEFAULT_VIS_DIR = Path("data") / "visualizations_preview"
DEFAULT_MAX_WIDTH = 960
DEFAULT_MAX_HEIGHT = 540


def find_first_video(root):
    return list_videos(root)[0]


def list_videos(root):
    root = Path(root)
    videos = sorted(root.rglob("*_rgb.avi"))
    if not videos:
        raise SystemExit("No NTU RGB video found under: {}".format(root))
    return videos


def visualization_paths(video, vis_dir):
    stem = Path(video).stem
    vis_dir = Path(vis_dir)
    return vis_dir / "{}_skeleton.avi".format(stem), vis_dir / "{}_skeleton.jpg".format(stem)


def preview_is_ready(path):
    path = Path(path)
    return path.exists() and path.stat().st_size > 1024


def ensure_visualization(
    video,
    out_dir,
    vis_dir,
    regenerate=False,
    device="auto",
    cpu_threads=4,
    pose_batch_size=1,
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
        temporal_min_frames=2,
        temporal_max_jump=150.0,
        temporal_min_keypoints=5,
        show_skeleton=False,
        visualize_dir=str(vis_dir),
        skip_existing=False,
        limit=None,
        cpu_threads=cpu_threads,
        pose_batch_size=pose_batch_size,
        workers=1,
    ))
    return vis_video, vis_image


def resize_for_preview(frame, max_width, max_height):
    import cv2

    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame
    size = (int(width * scale), int(height * scale))
    return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)


def draw_controls(frame, label, index, total, auto_play=True):
    import cv2

    button_w, button_h = 112, 42
    margin = 14
    x1 = frame.shape[1] - button_w - margin
    y1 = margin
    x2 = frame.shape[1] - margin
    y2 = y1 + button_h
    rect = (x1, y1, x2, y2)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 120, 30), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 255, 80), 2)
    cv2.putText(frame, "Next", (x1 + 23, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)

    mode = "Auto Play" if auto_play else "Loop"
    info = "{}/{}  {}  [{}]".format(index + 1, total, label, mode)
    cv2.rectangle(frame, (10, frame.shape[0] - 38), (min(frame.shape[1] - 10, 760), frame.shape[0] - 8), (0, 0, 0), -1)
    cv2.putText(frame, info, (18, frame.shape[0] - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return rect


def point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


class PreviewState:
    def __init__(self):
        self.next_requested = False
        self.button_rect = None


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
            self.result = ensure_visualization(
                video=self.video,
                out_dir=self.args.out_dir,
                vis_dir=self.args.vis_dir,
                regenerate=self.args.regenerate,
                device=self.args.device,
                cpu_threads=self.args.cpu_threads,
                pose_batch_size=self.args.pose_batch_size,
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
    vis_video, vis_image = visualization_paths(video, args.vis_dir)
    if preview_is_ready(vis_video) and not args.regenerate:
        print("preview ready: {}".format(vis_video), flush=True)
        return vis_video, vis_image

    print("building preview in background: {}".format(video), flush=True)
    state.button_rect = None
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


def play_playlist(videos, start_index, args):
    import cv2

    state = PreviewState()
    window = "NTU RTMW skeleton preview"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, args.max_width, args.max_height)
    last_frame = None

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP and state.button_rect and point_in_rect(x, y, state.button_rect):
            state.next_requested = True

    cv2.setMouseCallback(window, on_mouse)
    index = start_index
    while True:
        video = videos[index]
        preview = wait_for_preview(video, args, cv2, window, state, last_frame)
        if preview is None:
            cv2.destroyAllWindows()
            return
        vis_video, vis_image = preview
        print("playing: {}".format(vis_video.resolve()), flush=True)

        cap = cv2.VideoCapture(str(vis_video))
        if not cap.isOpened():
            print("bad preview, regenerating: {}".format(vis_video), flush=True)
            args.regenerate = True
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(1, int(1000 / max(1, fps * args.speed)))
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.loop_current:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                index = (index + 1) % len(videos)
                state.next_requested = False
                break
            frame = resize_for_preview(frame, args.max_width, args.max_height)
            last_frame = frame.copy()
            state.button_rect = draw_controls(
                frame,
                Path(video).name,
                index,
                len(videos),
                auto_play=not args.loop_current,
            )
            cv2.imshow(window, frame)
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord("q"), 27):
                cap.release()
                cv2.destroyAllWindows()
                return
            if key in (ord("n"), ord("d")):
                state.next_requested = True
            if key in (ord("p"), ord("a")):
                index = (index - 1) % len(videos)
                state.next_requested = False
                break
            if state.next_requested:
                index = (index + 1) % len(videos)
                state.next_requested = False
                break
        cap.release()


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


def parser():
    p = argparse.ArgumentParser(description="Generate and play one RTMW skeleton preview.")
    p.add_argument("--video", help="Specific NTU *_rgb.avi file. Defaults to the first extracted video.")
    p.add_argument("--vis-dir", default=str(DEFAULT_VIS_DIR))
    p.add_argument("--out-dir", default=str(SKELETON_DIR))
    p.add_argument("--device", default="auto", help="Device for preview generation. Default auto prefers cuda:0, then cpu.")
    p.add_argument("--cpu-threads", type=int, default=4, help="Limit CPU threads while generating a missing preview.")
    p.add_argument("--pose-batch-size", type=int, default=1, help="MMPose inferencer batch size while generating a missing preview.")
    p.add_argument("--regenerate", action="store_true")
    p.add_argument("--no-window", action="store_true", help="Only generate/check files, do not open OpenCV window.")
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--max-width", type=int, default=DEFAULT_MAX_WIDTH)
    p.add_argument("--max-height", type=int, default=DEFAULT_MAX_HEIGHT)
    p.add_argument("--loop-current", action="store_true", help="Replay the current video instead of auto-playing the next one.")
    return p


def main():
    args = parser().parse_args()
    videos = list_videos(EXTRACTED_DIR)
    if args.video:
        video = Path(args.video).resolve()
        resolved = [p.resolve() for p in videos]
        if video in resolved:
            start_index = resolved.index(video)
        else:
            videos = [video]
            start_index = 0
    else:
        start_index = 0
    video = videos[start_index]
    if args.no_window:
        vis_video, vis_image = ensure_visualization(
            video=video,
            out_dir=args.out_dir,
            vis_dir=args.vis_dir,
            regenerate=args.regenerate,
            device=args.device,
            cpu_threads=args.cpu_threads,
            pose_batch_size=args.pose_batch_size,
        )
        print("preview video: {}".format(vis_video.resolve()), flush=True)
        print("preview image: {}".format(vis_image.resolve()), flush=True)
    else:
        play_playlist(videos, start_index, args)


if __name__ == "__main__":
    main()
