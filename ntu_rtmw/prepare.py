import argparse
from argparse import Namespace
from pathlib import Path

from .archives import (
    check_ntu_rgb_archives,
    ensure_dirs,
    extract_all,
    extract_one,
    extraction_dir_name,
    list_archives,
    remove_extracted,
)
from .constants import (
    EXTRACTED_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_ARCHIVES_DIR,
    RTMDET_CONFIG,
    RTMDET_WEIGHTS_PATH,
    RTMW_WEIGHTS_URL,
    SKELETON_DIR,
)
from .download import ensure_rtmdet_weights, ensure_rtmw_weights
from .device import resolve_device
from .extract import RTMW_CONFIG, configure_cpu_threads, ensure_openmmlab_ready, run as extract_run
from .manifest import build_manifest


def parser():
    p = argparse.ArgumentParser(description="Prepare NTU RGB+D archives into RTMW skeleton files.")
    p.add_argument("--archives-dir", default=str(RAW_ARCHIVES_DIR))
    p.add_argument("--extract-dir", default=str(EXTRACTED_DIR))
    p.add_argument("--skeleton-dir", default=str(SKELETON_DIR))
    p.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    p.add_argument("--device", default="auto", help="Device for pose extraction. Default auto prefers cuda:0, then cpu.")
    p.add_argument("--det-model", default=RTMDET_CONFIG)
    p.add_argument("--det-weights", default=str(RTMDET_WEIGHTS_PATH))
    p.add_argument("--show-skeleton", action="store_true")
    p.add_argument("--visualize-dir")
    p.add_argument("--protocol", choices=["xsub", "xview", "xset"], default="xsub")
    p.add_argument("--ntu-version", choices=["auto", "60", "120"], default="auto")
    p.add_argument("--limit", type=int)
    p.add_argument("--skip-existing", action="store_true", default=True)
    p.add_argument("--overwrite", action="store_true", help="Re-extract skeleton files even when .npz outputs exist.")
    p.add_argument("--require-all-archives", action="store_true")
    p.add_argument("--no-download-weights", action="store_true")
    p.add_argument("--no-extract-archives", action="store_true")
    p.add_argument("--keep-extracted", action="store_true", help="Keep expanded archive folders after pose extraction.")
    p.add_argument("--extract-all-first", action="store_true", help="Use the old mode: extract every archive before pose extraction.")
    p.add_argument("--no-pose", action="store_true")
    p.add_argument("--cpu-threads", type=int, default=0, help="Limit CPU compute threads per process. Try 4 or 8 if CPU is pinned.")
    p.add_argument("--pose-batch-size", type=int, default=1, help="MMPose inferencer batch size. Try 4 or 8 when CPU is saturated and GPU is underused.")
    p.add_argument("--workers", type=int, default=1, help="Parallel video extraction workers. Try 2 first on one GPU.")
    p.add_argument("--cpu-workers", type=int, default=0, help="Additional CPU-only workers sharing a dynamic video queue with CUDA workers.")
    p.add_argument("--cpu-worker-threads", type=int, default=4, help="Compute threads used by each additional CPU-only worker.")
    p.add_argument("--cpu-pose-batch-size", type=int, default=1, help="MMPose batch size for additional CPU-only workers.")
    p.add_argument("--scan-workers", type=int, default=32, help="Threads used to check existing output files before extraction.")
    return p


def make_extract_args(args, input_dir, output_dir, weights_path, det_weights_path, limit=None):
    return Namespace(
        input=str(input_dir),
        output=str(output_dir),
        pattern="*_rgb.avi",
        all_videos=True,
        pose2d=RTMW_CONFIG,
        pose2d_weights=weights_path,
        det_model=args.det_model,
        det_weights=str(det_weights_path) if det_weights_path else None,
        device=args.device,
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
        show_skeleton=args.show_skeleton,
        visualize_dir=args.visualize_dir,
        skip_existing=args.skip_existing and not args.overwrite,
        limit=limit,
        cpu_threads=args.cpu_threads,
        pose_batch_size=args.pose_batch_size,
        workers=args.workers,
        cpu_workers=args.cpu_workers,
        cpu_worker_threads=args.cpu_worker_threads,
        cpu_pose_batch_size=args.cpu_pose_batch_size,
        scan_workers=args.scan_workers,
    )


def extract_pose_for_all_extracted(args, weights_path, det_weights_path):
    if not args.no_extract_archives:
        extract_all(args.archives_dir, args.extract_dir, skip_existing=True, check_expected=False)
    if args.no_pose:
        return
    extract_run(make_extract_args(args, args.extract_dir, args.skeleton_dir, weights_path, det_weights_path, args.limit))


def extract_pose_archive_by_archive(args, weights_path, det_weights_path):
    archives = list_archives(args.archives_dir)
    if not archives:
        print("No archives found in {}".format(Path(args.archives_dir).resolve()), flush=True)
        return

    remaining = args.limit
    for index, archive in enumerate(archives, 1):
        archive_dir_name = extraction_dir_name(archive)
        target = Path(args.extract_dir) / archive_dir_name
        output_dir = Path(args.skeleton_dir) / archive_dir_name
        marker = target / ".extracted"
        print("archive [{}/{}] {}".format(index, len(archives), archive.name), flush=True)
        if not args.no_extract_archives:
            if marker.exists() and args.skip_existing:
                print("reuse extracted {}".format(target), flush=True)
            else:
                target.mkdir(parents=True, exist_ok=True)
                print("extract {} -> {}".format(archive.name, target), flush=True)
                extract_one(archive, target)
                marker.write_text(str(archive), encoding="utf-8")

        if not args.no_pose:
            current_limit = remaining if remaining is not None else None
            outputs = extract_run(make_extract_args(args, target, output_dir, weights_path, det_weights_path, current_limit))
            if remaining is not None:
                remaining -= len(outputs)
                if remaining <= 0:
                    if not args.keep_extracted and not args.no_extract_archives:
                        remove_extracted(target, args.extract_dir)
                    break

        if not args.keep_extracted and not args.no_extract_archives:
            remove_extracted(target, args.extract_dir)


def main():
    args = parser().parse_args()
    configure_cpu_threads(args)
    args.device = resolve_device(args.device)
    print("device {}".format(args.device), flush=True)
    ensure_dirs(args.archives_dir, args.extract_dir, args.skeleton_dir, args.processed_dir, MODELS_DIR)
    check_ntu_rgb_archives(args.archives_dir, require_all=args.require_all_archives)
    weights_path = RTMW_WEIGHTS_URL if args.no_download_weights else str(ensure_rtmw_weights())
    det_weights_path = args.det_weights
    if args.det_model != "whole_image" and args.det_weights == str(RTMDET_WEIGHTS_PATH):
        det_weights_path = RTMDET_WEIGHTS_PATH if args.no_download_weights else ensure_rtmdet_weights()

    if not args.no_pose:
        ensure_openmmlab_ready()

    if args.extract_all_first or args.no_extract_archives:
        extract_pose_for_all_extracted(args, weights_path, det_weights_path)
    else:
        extract_pose_archive_by_archive(args, weights_path, det_weights_path)

    version = args.ntu_version if args.ntu_version == "auto" else int(args.ntu_version)
    manifest = build_manifest(
        args.skeleton_dir,
        "{}/manifest_{}.csv".format(args.processed_dir, args.protocol),
        protocol=args.protocol,
        ntu_version=version,
    )
