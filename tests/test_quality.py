import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ntu_rtmw.quality import (
    QualityThresholds,
    build_ntu_zip_index,
    configure_repair_workflow,
    delete_failed_skeletons,
    deletion_summary,
    evaluate_npz,
    locate_archived_video,
    parse_integer_spec,
    parser as quality_parser,
    reextract_failed,
    scan_skeletons,
)


def write_skeleton(path, action, frames=20, persons=2, missing_second=None):
    keypoints = np.zeros((frames, persons, 133, 2), dtype=np.float32)
    scores = np.ones((frames, persons, 133), dtype=np.float32)
    bboxes = np.zeros((frames, persons, 4), dtype=np.float32)
    bbox_scores = np.ones((frames, persons), dtype=np.float32)
    for frame in range(frames):
        for person in range(persons):
            x = 100.0 + person * 300.0 + frame
            y = 100.0
            keypoints[frame, person, :, 0] = x
            keypoints[frame, person, :, 1] = y + np.arange(133, dtype=np.float32)
            bboxes[frame, person] = [x - 25.0, y - 20.0, x + 75.0, y + 500.0]

    if missing_second:
        start, end = missing_second
        keypoints[start:end, 1] = np.nan
        scores[start:end, 1] = 0
        bboxes[start:end, 1] = np.nan
        bbox_scores[start:end, 1] = 0

    metadata = json.dumps({
        "action": action,
        "video_name": "S001C001P001R001A{:03d}_rgb.avi".format(action),
    })
    np.savez_compressed(
        path,
        metadata=metadata,
        keypoints=keypoints,
        scores=scores,
        bboxes=bboxes,
        bbox_scores=bbox_scores,
        frame_indices=np.arange(frames, dtype=np.int32),
    )


class QualityTests(unittest.TestCase):
    def test_parse_integer_spec(self):
        self.assertEqual(parse_integer_spec("50-52,60,59"), (50, 51, 52, 59, 60))

    def test_single_person_sequence_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S001C001P001R001A001_rgb.npz"
            write_skeleton(path, action=1, persons=2, missing_second=(0, 20))
            result = evaluate_npz(path)
            self.assertEqual(result["status"], "pass", result["reasons"])
            self.assertEqual(result["expected_persons"], 1)

    def test_single_person_sequence_with_stable_extra_person_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S001C001P001R001A001_rgb.npz"
            write_skeleton(path, action=1, persons=2)
            result = evaluate_npz(path)
            self.assertEqual(result["expected_persons"], 1)
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["metrics"]["unexpected_person_rate"], 1.0)
            self.assertTrue(
                any("unexpected_person_rate" in reason for reason in result["reasons"])
            )

    def test_complete_two_person_sequence_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S001C001P001R001A055_rgb.npz"
            write_skeleton(path, action=55)
            result = evaluate_npz(path)
            self.assertEqual(result["status"], "pass", result["reasons"])
            self.assertEqual(result["metrics"]["expected_person_recall"], 1.0)

    def test_ntu120_interaction_expects_two_people(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S001C001P001R001A119_rgb.npz"
            write_skeleton(path, action=119, persons=2, missing_second=(5, 17))
            result = evaluate_npz(path)
            self.assertEqual(result["expected_persons"], 2)
            self.assertEqual(result["status"], "fail")

    def test_long_second_person_dropout_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "S001C001P001R001A055_rgb.npz"
            write_skeleton(path, action=55, missing_second=(5, 17))
            result = evaluate_npz(path, QualityThresholds(max_missing_run=5))
            self.assertEqual(result["status"], "fail")
            self.assertEqual(result["metrics"]["longest_missing_run"], 12)
            self.assertTrue(any("expected_person_recall" in reason for reason in result["reasons"]))
            self.assertTrue(any("longest_missing_run" in reason for reason in result["reasons"]))

    def test_duplicate_sample_marks_only_extra_copy_failed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = root / "nested" / "S001C001P001R001A001_rgb.npz"
            second = root / "S001C001P001R001A001_rgb.npz"
            first.parent.mkdir(parents=True)
            write_skeleton(first, action=1, persons=2, missing_second=(0, 20))
            write_skeleton(second, action=1, persons=2, missing_second=(0, 20))
            results = scan_skeletons(root, workers=1)
            self.assertEqual(sum(result["status"] == "pass" for result in results), 1)
            self.assertEqual(sum(result["status"] == "fail" for result in results), 1)
            failed = next(result for result in results if result["status"] == "fail")
            self.assertTrue(any("duplicate_sample_id" in reason for reason in failed["reasons"]))

    def test_retry_skips_duplicate_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = {
                "path": str(root / "duplicate.npz"),
                "status": "fail",
                "quality_score": 85.0,
                "metadata": {},
                "reasons": ["duplicate_sample_id=test preferred=preferred.npz"],
            }
            records = reextract_failed(
                [result],
                QualityThresholds(),
                root,
                root,
                root / "retry",
            )
            self.assertEqual(records[0]["status"], "skipped_duplicate_copy")

    def test_delete_failed_skeletons_only_deletes_eligible_failures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = root / "S001C001P001R001A055_rgb.npz"
            untouched = root / "S001C001P001R001A056_rgb.npz"
            passing = root / "S001C001P001R001A001_rgb.npz"
            write_skeleton(selected, action=55, missing_second=(5, 17))
            write_skeleton(untouched, action=56, missing_second=(5, 17))
            write_skeleton(passing, action=1, missing_second=(0, 20))
            results = [evaluate_npz(path) for path in (selected, untouched, passing)]

            records = delete_failed_skeletons(
                results,
                root,
                eligible_paths=[selected],
            )

            self.assertFalse(selected.exists())
            self.assertTrue(untouched.exists())
            self.assertTrue(passing.exists())
            self.assertEqual(deletion_summary(records)["deleted"], 1)
            self.assertEqual(results[0]["deletion"]["status"], "deleted")
            self.assertNotIn("deletion", results[1])

    def test_repair_yolo26_flag_enables_complete_cleanup_workflow(self):
        args = configure_repair_workflow(
            quality_parser().parse_args(["--repair-failed-yolo26"])
        )
        self.assertTrue(args.reextract_failed)
        self.assertEqual(args.retry_det_backend, "yolo26")
        self.assertTrue(args.replace_if_better)
        self.assertTrue(args.delete_failed)
        self.assertEqual(args.retry_gpu_workers, 1)
        self.assertEqual(args.retry_cpu_workers, 1)

        cpu_disabled = configure_repair_workflow(
            quality_parser().parse_args([
                "--repair-failed-yolo26",
                "--retry-cpu-workers",
                "0",
            ])
        )
        self.assertEqual(cpu_disabled.retry_cpu_workers, 0)

    def test_locates_video_inside_official_ntu_zip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archives = root / "raw_archives"
            archives.mkdir()
            archive = archives / "nturgbd_rgb_s001.zip"
            member = "nturgb+d_rgb/S001C001P001R001A055_rgb.avi"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(member, b"video-bytes")

            result = {
                "path": str(root / "S001C001P001R001A055_rgb.npz"),
                "metadata": {
                    "setup": 1,
                    "video_name": "S001C001P001R001A055_rgb.avi",
                },
            }
            source = locate_archived_video(result, build_ntu_zip_index(archives), {})
            self.assertEqual(source, (archive, member))

    def test_retry_uses_one_video_from_zip_without_expanded_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skeleton_root = root / "skeletons"
            skeleton_root.mkdir()
            original = skeleton_root / "S001C001P001R001A055_rgb.npz"
            write_skeleton(original, action=55, missing_second=(5, 17))
            result = evaluate_npz(original)
            self.assertEqual(result["status"], "fail")

            passing = root / "passing.npz"
            write_skeleton(passing, action=55)
            with np.load(passing, allow_pickle=False) as data:
                arrays = {
                    key: data[key]
                    for key in ("keypoints", "scores", "bboxes", "bbox_scores", "frame_indices")
                }

            archives = root / "raw_archives"
            archives.mkdir()
            archive = archives / "nturgbd_rgb_s001.zip"
            member = "nturgb+d_rgb/S001C001P001R001A055_rgb.avi"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(member, b"video-bytes")

            retry_output = root / "retry"
            retry_temp = root / "retry_temp"
            with (
                patch("ntu_rtmw.extract.ensure_supported_python"),
                patch("ntu_rtmw.extract.ensure_openmmlab_ready"),
                patch("ntu_rtmw.extract.configure_cpu_threads"),
                patch("ntu_rtmw.extract.build_inferencer", return_value=object()),
                patch("ntu_rtmw.extract.infer_video", return_value=arrays),
                patch("ntu_rtmw.download.ensure_rtmw_weights", return_value=root / "pose.pth"),
                patch("ntu_rtmw.download.ensure_rtmdet_weights", return_value=root / "det.pth"),
            ):
                records = reextract_failed(
                    [result],
                    QualityThresholds(),
                    skeleton_root,
                    root / "extracted",
                    retry_output,
                    device="cpu",
                    archives_dir=archives,
                    retry_temp_dir=retry_temp,
                )

            self.assertEqual(records[0]["status"], "pass", records[0])
            self.assertEqual(records[0]["source_archive_member"], member)
            retried = retry_output / original.name
            self.assertTrue(retried.exists())
            with np.load(retried, allow_pickle=False) as data:
                metadata = json.loads(data["metadata"].item())
            self.assertEqual(metadata["video_name"], original.with_suffix(".avi").name)
            self.assertEqual(metadata["video_archive"], str(archive.resolve()))
            self.assertEqual(metadata["video_archive_member"], member)
            self.assertEqual(list(retry_temp.iterdir()), [])

    def test_retry_can_route_zip_video_through_yolo26_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            skeleton_root = root / "skeletons"
            skeleton_root.mkdir()
            original = skeleton_root / "S001C001P001R001A055_rgb.npz"
            write_skeleton(original, action=55, missing_second=(5, 17))
            result = evaluate_npz(original)
            result["quality_score"] = 100.0

            passing = root / "passing.npz"
            write_skeleton(passing, action=55)
            with np.load(passing, allow_pickle=False) as data:
                arrays = {
                    key: data[key]
                    for key in ("keypoints", "scores", "bboxes", "bbox_scores", "frame_indices")
                }

            archives = root / "raw_archives"
            archives.mkdir()
            archive = archives / "nturgbd_rgb_s001.zip"
            member = "nturgb+d_rgb/S001C001P001R001A055_rgb.avi"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr(member, b"video-bytes")
            yolo_model = root / "yolo26x.pt"
            yolo_model.write_bytes(b"weights")

            retry_output = root / "retry_yolo"
            with (
                patch("ntu_rtmw.extract.ensure_supported_python"),
                patch("ntu_rtmw.extract.configure_cpu_threads"),
                patch("ntu_rtmw.yolo_extract.ensure_ready"),
                patch("ntu_rtmw.yolo_extract.build_inferencer", return_value=object()) as build_yolo,
                patch("ntu_rtmw.yolo_extract.infer_video", return_value=arrays) as infer_yolo,
            ):
                records = reextract_failed(
                    [result],
                    QualityThresholds(),
                    skeleton_root,
                    root / "extracted",
                    retry_output,
                    device="cpu",
                    archives_dir=archives,
                    retry_det_backend="yolo26",
                    retry_yolo_model=yolo_model,
                    retry_yolo_conf=0.15,
                    retry_yolo_iou=0.70,
                    retry_yolo_imgsz=960,
                    retry_crop_margin=0.10,
                    replace_if_better=True,
                )

            self.assertEqual(records[0]["status"], "pass", records[0])
            self.assertEqual(records[0]["detector"], "yolo26")
            self.assertTrue(records[0]["replaced"])
            build_yolo.assert_called_once()
            infer_yolo.assert_called_once()
            retried = retry_output / original.name
            with np.load(retried, allow_pickle=False) as data:
                metadata = json.loads(data["metadata"].item())
            self.assertEqual(metadata["det_model"], "yolo26x")
            self.assertEqual(metadata["det_weights"], str(yolo_model))
            self.assertEqual(evaluate_npz(original)["status"], "pass")


if __name__ == "__main__":
    unittest.main()
