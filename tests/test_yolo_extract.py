import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from ntu_rtmw.yolo_extract import infer_video


class FakeCapture:
    def __init__(self, _path):
        self.frames = [np.zeros((120, 160, 3), dtype=np.uint8)]

    def isOpened(self):
        return True

    def read(self):
        if self.frames:
            return True, self.frames.pop(0)
        return False, None

    def release(self):
        return None


class FakePredictionInstances:
    def __init__(self):
        points = np.zeros((1, 133, 2), dtype=np.float32)
        points[0, :, 0] = 10.0
        points[0, :, 1] = 20.0
        self.keypoints = points
        self.keypoint_scores = np.ones((1, 133), dtype=np.float32)

    def cpu(self):
        return self

    def numpy(self):
        return self


class FakePoseSample:
    def __init__(self):
        self.pred_instances = FakePredictionInstances()


class FakeDetector:
    def __call__(self, _frame):
        return np.array(
            [
                [10.0, 10.0, 60.0, 110.0, 0.90],
                [90.0, 10.0, 150.0, 110.0, 0.80],
            ],
            dtype=np.float32,
        )


class YoloExtractTests(unittest.TestCase):
    def test_infer_video_keeps_two_yolo_people_and_offsets_crop_keypoints(self):
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.VideoCapture = FakeCapture
        fake_mmpose = types.ModuleType("mmpose")
        fake_apis = types.ModuleType("mmpose.apis")
        fake_apis.inference_topdown = lambda *_args, **_kwargs: [FakePoseSample()]
        fake_mmpose.apis = fake_apis
        args = Namespace(
            bbox_thr=0.15,
            kpt_thr=0.1,
            max_persons=2,
            tracking_distance=150.0,
            crop_margin=0.0,
            filter_output_to_bbox=False,
            filter_output_to_frame=False,
            temporal_min_frames=1,
            temporal_max_jump=0.0,
            temporal_min_keypoints=0,
        )

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules,
            {
                "cv2": fake_cv2,
                "mmpose": fake_mmpose,
                "mmpose.apis": fake_apis,
            },
        ):
            arrays = infer_video(
                (FakeDetector(), object()),
                Path(directory) / "S001C001P001R001A055_rgb.avi",
                args,
            )

        self.assertEqual(arrays["keypoints"].shape, (1, 2, 133, 2))
        self.assertTrue(np.all(arrays["bbox_scores"][0] > 0))
        self.assertAlmostEqual(float(arrays["keypoints"][0, 0, 0, 0]), 20.0)
        self.assertAlmostEqual(float(arrays["keypoints"][0, 1, 0, 0]), 100.0)

    def test_infer_video_keeps_only_one_person_for_single_person_action(self):
        fake_cv2 = types.ModuleType("cv2")
        fake_cv2.VideoCapture = FakeCapture
        fake_mmpose = types.ModuleType("mmpose")
        fake_apis = types.ModuleType("mmpose.apis")
        fake_apis.inference_topdown = lambda *_args, **_kwargs: [FakePoseSample()]
        fake_mmpose.apis = fake_apis
        args = Namespace(
            bbox_thr=0.15,
            kpt_thr=0.1,
            max_persons=2,
            tracking_distance=150.0,
            crop_margin=0.0,
            filter_output_to_bbox=False,
            filter_output_to_frame=False,
            temporal_min_frames=1,
            temporal_max_jump=0.0,
            temporal_min_keypoints=0,
        )

        with tempfile.TemporaryDirectory() as directory, patch.dict(
            sys.modules,
            {
                "cv2": fake_cv2,
                "mmpose": fake_mmpose,
                "mmpose.apis": fake_apis,
            },
        ):
            arrays = infer_video(
                (FakeDetector(), object()),
                Path(directory) / "S001C001P001R001A001_rgb.avi",
                args,
            )

        self.assertGreater(float(arrays["bbox_scores"][0, 0]), 0.0)
        self.assertEqual(float(arrays["bbox_scores"][0, 1]), 0.0)


if __name__ == "__main__":
    unittest.main()
