import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from helpers import create_episode


class TargetEvaluationTests(unittest.TestCase):
    def test_finds_first_close_row_for_both_gripper_directions(self):
        from evaluation.target import find_close_row

        increasing = np.asarray([
            [500, 500, 500, 500, 500, 150],
            [500, 500, 500, 500, 500, 400],
            [500, 500, 500, 500, 500, 520],
        ], dtype=np.float32)
        decreasing = increasing.copy()
        decreasing[:, 5] = [800, 500, 200]

        self.assertEqual(find_close_row(increasing, 150, 520), 1)
        self.assertEqual(find_close_row(decreasing, 800, 200), 1)

    def test_reports_no_close_row_without_fabricating_metrics(self):
        from evaluation.target import evaluate_actions

        actions = np.full((3, 6), 200.0, dtype=np.float32)
        report = evaluate_actions(
            actions,
            target=(500.0,) * 6,
            open_pulse=150.0,
            closed_pulse=520.0,
        )

        self.assertEqual(report, {"status": "no_close_row", "close_row_index": None})

    def test_rejects_malformed_or_nonfinite_actions(self):
        from evaluation.target import find_close_row

        for actions in (
            np.zeros((3, 5), dtype=np.float32),
            np.asarray([[500, 500, 500, 500, 500, np.nan]], dtype=np.float32),
        ):
            with self.assertRaisesRegex(ValueError, "actions"):
                find_close_row(actions, 150, 520)

    def test_joint_endpoint_metrics_preserve_signed_and_absolute_error(self):
        from evaluation.target import joint_endpoint_metrics

        metrics = joint_endpoint_metrics(
            (510, 480, 530, 470, 500, 510),
            (500, 500, 500, 500, 500, 520),
        )

        self.assertEqual(metrics["signed_error"], [10.0, -20.0, 30.0, -30.0, 0.0, -10.0])
        self.assertEqual(metrics["absolute_error"], [10.0, 20.0, 30.0, 30.0, 0.0, 10.0])
        self.assertEqual(metrics["arm_mae"], 18.0)
        self.assertEqual(metrics["arm_max"], 30.0)
        self.assertEqual(metrics["gripper_error"], -10.0)

    def test_cli_uses_anchor_only_and_writes_model_only_result(self):
        from evaluation import main as evaluation_main

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            episode = create_episode(root / "session", source="expert")
            model = root / "model"
            model.mkdir()
            runs = root / "runs"

            class Runner:
                model_path = str(model)
                codec_sha256 = "b" * 64

                def __init__(self, model_path, actions, device):
                    self.model_path = str(model_path)

                def predict(self, image, state, task):
                    self.observation = (image.copy(), tuple(state), task)
                    actions = np.asarray([
                        [500, 500, 500, 500, 500, 150],
                        [520, 530, 540, 550, 510, 400],
                        [520, 530, 540, 550, 510, 520],
                    ], dtype=np.float32)
                    return actions, 0.123

            with patch.object(evaluation_main, "RUNS", runs), patch.object(
                evaluation_main, "SmolVLARunner", Runner
            ):
                code = evaluation_main.main([
                    "--model", str(model),
                    "--episode", str(episode),
                    "--device", "cpu",
                    "--actions", "3",
                ])

            self.assertEqual(code, 0)
            results = list((runs / "evaluation").glob("*/result.json"))
            self.assertEqual(len(results), 1)
            result = json.loads(results[0].read_text(encoding="utf-8"))
            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["close_row_index"], 1)
            self.assertEqual(result["inference_seconds"], 0.123)
            self.assertNotIn("actions", result)


if __name__ == "__main__":
    unittest.main()
