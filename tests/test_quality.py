import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ntu_rtmw.quality import (
    QualityThresholds,
    evaluate_npz,
    parse_integer_spec,
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


if __name__ == "__main__":
    unittest.main()
