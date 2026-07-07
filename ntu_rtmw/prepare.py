import argparse
from argparse import Namespace

from .archives import check_ntu_rgb_archives, ensure_dirs, extract_all
from .constants import (
    EXTRACTED_DIR,
    MODELS_DIR,
    PROCESSED_DIR,
    RAW_ARCHIVES_DIR,
    RTMDET_CONFIG,
    RTMDET_WEIGHTS_PATH,
    RTMW_WEIGHTS_URL,
    RUNS_DIR,
    SKELETON_DIR,
)
from .download import ensure_rtmdet_weights, ensure_rtmw_weights
from .extract import RTMW_CONFIG, ensure_openmmlab_ready, run as extract_run
from .manifest import build_manifest
from .train import run as train_run


def parser():
    p = argparse.ArgumentParser(description="Prepare NTU RGB+D from archives to training.")
    p.add_argument("--archives-dir", default=str(RAW_ARCHIVES_DIR))
    p.add_argument("--extract-dir", default=str(EXTRACTED_DIR))
    p.add_argument("--skeleton-dir", default=str(SKELETON_DIR))
    p.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    p.add_argument("--device", default="cpu")
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
    p.add_argument("--no-pose", action="store_true")
    p.add_argument("--no-train", action="store_true")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    return p


def main():
    args = parser().parse_args()
    ensure_dirs(args.archives_dir, args.extract_dir, args.skeleton_dir, args.processed_dir, RUNS_DIR, MODELS_DIR)
    check_ntu_rgb_archives(args.archives_dir, require_all=args.require_all_archives)
    weights_path = RTMW_WEIGHTS_URL if args.no_download_weights else str(ensure_rtmw_weights())
    det_weights_path = args.det_weights
    if args.det_model != "whole_image" and args.det_weights == str(RTMDET_WEIGHTS_PATH):
        det_weights_path = RTMDET_WEIGHTS_PATH if args.no_download_weights else ensure_rtmdet_weights()

    if not args.no_pose:
        ensure_openmmlab_ready()

    if not args.no_extract_archives:
        extract_all(args.archives_dir, args.extract_dir, skip_existing=True, check_expected=False)

    if not args.no_pose:
        extract_run(Namespace(
            input=args.extract_dir,
            output=args.skeleton_dir,
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
            show_skeleton=args.show_skeleton,
            visualize_dir=args.visualize_dir,
            skip_existing=args.skip_existing and not args.overwrite,
            limit=args.limit,
        ))

    version = args.ntu_version if args.ntu_version == "auto" else int(args.ntu_version)
    manifest = build_manifest(
        args.skeleton_dir,
        "{}/manifest_{}.csv".format(args.processed_dir, args.protocol),
        protocol=args.protocol,
        ntu_version=version,
    )

    if not args.no_train:
        train_run(Namespace(
            manifest=str(manifest),
            out_dir=str(RUNS_DIR / "gru_{}".format(args.protocol)),
            device=args.device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            frames=64,
            hidden=256,
            lr=3e-4,
        ))
