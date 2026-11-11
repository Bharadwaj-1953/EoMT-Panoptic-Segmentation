"""Synthetic COCO PQ tests; no datasets, model loading, or pretrained inference."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

from scripts import evaluate_panoptic as evaluation
from tests.test_inference import temporary_directory


CATEGORIES = {1: {"id": 1, "name": "thing", "isthing": 1},
              2: {"id": 2, "name": "stuff", "isthing": 0},
              3: {"id": 3, "name": "absent", "isthing": 0}}


def segment(segment_id, category, area, crowd=0):
    return {"id": segment_id, "category_id": category, "area": area, "iscrowd": crowd}


class ReferencePQTests(unittest.TestCase):
    def score(self, gt, pred, gt_segments, pred_segments):
        with temporary_directory() as folder:
            root = Path(folder)
            gt_dir, pred_dir = root / "gt", root / "pred"
            gt_dir.mkdir(); pred_dir.mkdir()
            Image.fromarray(evaluation.id2rgb(np.array([gt], dtype=np.int32))).save(gt_dir / "x.png")
            Image.fromarray(evaluation.id2rgb(np.array([pred], dtype=np.int32))).save(pred_dir / "x.png")
            gt_ann = {"image_id": 1, "file_name": "x.png", "segments_info": gt_segments}
            pred_ann = {"image_id": 1, "file_name": "x.png", "segments_info": pred_segments}
            return evaluation.pq_compute_single_core(0, [(gt_ann, pred_ann)], str(gt_dir), str(pred_dir), CATEGORIES)

    def test_perfect_and_absent_class_macro_average(self):
        stat = self.score([10, 10, 20, 20], [1, 1, 2, 2],
                          [segment(10, 1, 2), segment(20, 2, 2)],
                          [segment(1, 1, 2), segment(2, 2, 2)])
        groups, per_class = evaluation.summarize(stat, CATEGORIES)
        self.assertEqual(groups["All"], {"pq": 1., "sq": 1., "rq": 1., "n": 2})
        self.assertEqual(groups["Things"]["pq"], 1.)
        self.assertEqual(groups["Stuff"]["pq"], 1.)
        self.assertFalse(per_class[3]["included"])

    def test_missing_segment_is_false_negative_even_under_predicted_void(self):
        stat = self.score([10, 10], [0, 0], [segment(10, 1, 2)], [])
        self.assertEqual(stat[1].fn, 1)
        self.assertEqual(stat[1].tp, 0)
        groups, _ = evaluation.summarize(stat, CATEGORIES)
        self.assertEqual(groups["All"]["pq"], 0.)
        self.assertIsNone(groups["Stuff"]["pq"])

    def test_strict_half_iou_does_not_match(self):
        stat = self.score([10, 10, 0, 0], [1, 0, 0, 0],
                          [segment(10, 1, 2)], [segment(1, 1, 1)])
        self.assertEqual((stat[1].tp, stat[1].fp, stat[1].fn), (0, 1, 1))

    def test_void_subtracted_from_union(self):
        stat = self.score([10, 10, 0], [1, 1, 1],
                          [segment(10, 1, 2)], [segment(1, 1, 3)])
        self.assertEqual(stat[1].tp, 1)
        self.assertEqual(stat[1].iou, 1.)

    def test_void_false_positive_suppression_is_strictly_more_than_half(self):
        ignored = self.score([0, 0, 10], [1, 1, 1],
                             [segment(10, 2, 1)], [segment(1, 1, 3)])
        self.assertEqual(ignored[1].fp, 0)
        half = self.score([0, 10], [1, 1], [segment(10, 2, 1)], [segment(1, 1, 2)])
        self.assertEqual(half[1].fp, 1)

    def test_nonperfect_sq_and_rq(self):
        stat = self.score([10, 10, 10, 20], [1, 1, 0, 0],
                          [segment(10, 1, 3), segment(20, 1, 1)], [segment(1, 1, 2)])
        groups, _ = evaluation.summarize(stat, CATEGORIES)
        self.assertAlmostEqual(groups["All"]["sq"], 2 / 3)
        self.assertAlmostEqual(groups["All"]["rq"], 2 / 3)
        self.assertAlmostEqual(groups["All"]["pq"], 4 / 9)

    def test_crowd_ignored_for_matching_and_same_category_false_positive(self):
        stat = self.score([10, 10, 10], [1, 1, 1],
                          [segment(10, 1, 3, crowd=1)], [segment(1, 1, 3)])
        self.assertEqual((stat[1].tp, stat[1].fp, stat[1].fn), (0, 0, 0))
        wrong = self.score([10, 10], [1, 1],
                           [segment(10, 1, 2, crowd=1)], [segment(1, 2, 2)])
        self.assertEqual(wrong[2].fp, 1)

    def test_wrong_class_and_accumulation(self):
        stat = self.score([10, 10], [1, 1], [segment(10, 1, 2)], [segment(1, 2, 2)])
        self.assertEqual(stat[1].fn, 1)
        self.assertEqual(stat[2].fp, 1)
        stat += self.score([10, 10], [1, 1], [segment(10, 1, 2)], [segment(1, 1, 2)])
        groups, _ = evaluation.summarize(stat, CATEGORIES)
        self.assertAlmostEqual(groups["All"]["pq"], 1 / 3)


class EvaluationWorkflowTests(unittest.TestCase):
    def make_dataset(self, root):
        image_dir, gt_dir = root / "images", root / "gt"
        image_dir.mkdir(); gt_dir.mkdir()
        data = {"images": [], "annotations": [], "categories": list(CATEGORIES.values())}
        for i in [9, 2]:
            name = f"{i}.png"
            Image.new("RGB", (2, 2)).save(image_dir / name)
            Image.fromarray(evaluation.id2rgb(np.full((2, 2), 10, dtype=np.int32))).save(gt_dir / name)
            data["images"].append({"id": i, "file_name": name, "width": 2, "height": 2})
            data["annotations"].append({"image_id": i, "file_name": name, "segments_info": [segment(10, 1, 4)]})
        annotation_path = root / "annotations.json"
        evaluation.write_json(annotation_path, data)
        return annotation_path, image_dir, gt_dir, data

    def test_deterministic_subset_full_guard_missing_and_duplicate(self):
        with temporary_directory() as folder:
            root = Path(folder)
            ann, imgs, gt, data = self.make_dataset(root)
            self.assertEqual(evaluation.index_dataset(ann, imgs, gt, limit=1)[-1], [2])
            with self.assertRaisesRegex(ValueError, "5,000"):
                evaluation.index_dataset(ann, imgs, gt, full=True)
            with self.assertRaisesRegex(ValueError, "positive"):
                evaluation.index_dataset(ann, imgs, gt, limit=0)
            (imgs / "2.png").unlink()
            with self.assertRaises(FileNotFoundError):
                evaluation.index_dataset(ann, imgs, gt)
            data["images"].append(copy.deepcopy(data["images"][0]))
            evaluation.write_json(ann, data)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                evaluation.index_dataset(ann, imgs, gt)

    def test_prediction_export_void_roundtrip_and_validation(self):
        with temporary_directory() as folder:
            path = Path(folder) / "prediction.png"
            pred = np.array([[[0, 0], [133, -1]]], dtype=np.int32)
            segments = evaluation.export_prediction(pred, {0: 1}, CATEGORIES, path)
            self.assertEqual(segments, [{"id": 1, "category_id": 1, "area": 1}])
            with Image.open(path) as image:
                self.assertEqual(evaluation.rgb2id(np.array(image)).tolist(), [[1, 0]])
            with self.assertRaisesRegex(ValueError, "void"):
                evaluation.export_prediction(np.array([[[133, 0]]]), {}, CATEGORIES, path)
            with self.assertRaisesRegex(ValueError, "mixed"):
                evaluation.export_prediction(np.array([[[0, 0], [1, 0]]]), {0: 1, 1: 2}, CATEGORIES, path)

    def test_ground_truth_area_validation_and_path_escape(self):
        with temporary_directory() as folder:
            root = Path(folder)
            ann, imgs, gt, data = self.make_dataset(root)
            annotation = data["annotations"][0]
            evaluation.validate_ground_truth(gt / "9.png", annotation, (2, 2))
            annotation["segments_info"][0]["area"] = 3
            with self.assertRaisesRegex(ValueError, "areas"):
                evaluation.validate_ground_truth(gt / "9.png", annotation, (2, 2))
            with self.assertRaisesRegex(ValueError, "escapes"):
                evaluation.child_path(root, "../outside.png")

    def test_end_to_end_synthetic_exports_and_overwrite_rejection(self):
        with temporary_directory() as folder:
            root = Path(folder)
            ann, imgs, gt, data = self.make_dataset(root)
            checkpoint = root / "synthetic.bin"
            checkpoint.write_bytes(b"test fixture, never loaded")
            # Keep all fixtures in the test's temporary subtree, under evaluation/.
            output = evaluation.inference.project_path("results/evaluation") / root.name
            args = SimpleNamespace(images=str(imgs), panoptic=str(gt), annotations=str(ann),
                                   output_dir=str(output), checkpoint=str(checkpoint), limit=1, full=False)
            try:
                result = evaluation.evaluate(args, loader=lambda _: SimpleNamespace(mask_thresh=.8, overlap_thresh=.8),
                    predictor=lambda model, image: np.zeros((2, 2, 2), dtype=np.int32), mapping={0: 1})
                self.assertEqual(result["image_ids"], [2])
                self.assertEqual(result["dataset_size"], 2)
                self.assertEqual(result["metrics"]["All"]["pq"], 1.)
                self.assertGreaterEqual(result["total_evaluation_seconds"], result["inference_seconds"])
                for name in ["metrics.json", "metrics.csv", "timings.csv", "per_class.csv",
                             "predictions.json", "ground_truth_subset.json", "panoptic_predictions/000000000002.png"]:
                    self.assertTrue((output / name).is_file(), name)
                parsed = json.loads((output / "metrics.json").read_text())
                self.assertIsNone(parsed["metrics"]["Stuff"]["pq"])
                with self.assertRaisesRegex(ValueError, "already exists"):
                    evaluation.evaluate(args)
            finally:
                if output.exists():
                    import shutil
                    shutil.rmtree(output)


if __name__ == "__main__":
    unittest.main()
