from pathlib import Path
import tempfile
import unittest

from helpers import create_episode
from dataset.raw import RawDataError


class ConverterPeriodTests(unittest.TestCase):
    def test_lerobot_dataset_accepts_150_ms_control_period(self):
        from dataset.converter import convert_sessions

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            session = root / "session"
            create_episode(session)
            output = root / "dataset"
            report = convert_sessions(
                [session], output, "local/test", control_period_ms=150,
                use_videos=False,
            )
            self.assertAlmostEqual(report.fps, 1000.0 / 150.0)
            self.assertEqual(report.control_period_ms, 150)
            self.assertTrue((output / "meta" / "info.json").is_file())

    def test_accepts_image_gap_within_1200_ms_capture_limit(self):
        from dataset.converter import convert_sessions

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            session = root / "session"
            create_episode(
                session,
                timestamps_ns=tuple(range(0, 1_200_000_001, 150_000_000)),
                frame_timestamps_ns=(0, 900_000_000, 1_200_000_000),
            )
            output = root / "dataset"
            try:
                report = convert_sessions(
                    [session], output, "local/test", control_period_ms=150,
                    use_videos=False,
                )
            except RawDataError as error:
                self.fail(f"capture-valid image gap was rejected: {error}")
            self.assertEqual(report.total_frames, 9)

    def test_balances_training_sources_without_repeating_validation_episodes(self):
        from dataset.converter import convert_sessions

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            session = root / "session"
            paths = [
                create_episode(session, index=0),
                create_episode(session, index=1),
                create_episode(session, index=2, source="expert"),
                create_episode(session, index=3, source="expert"),
            ]
            output = root / "dataset"
            try:
                report = convert_sessions(
                    [session],
                    output,
                    "local/test",
                    control_period_ms=150,
                    use_videos=False,
                    validation_episode_ids={paths[1].name, paths[3].name},
                    source_repeats={"manual": 1, "expert": 2},
                )
            except TypeError as error:
                self.fail(f"source-balanced conversion is missing: {error}")
            self.assertEqual(report.unique_episodes, 4)
            self.assertEqual(report.accepted_episodes, 5)
            self.assertEqual(len(report.train_episode_indices), 3)
            self.assertEqual(len(report.validation_episode_indices), 2)
            self.assertEqual(report.source_episode_counts, {"manual": 2, "expert": 2})
            self.assertEqual(
                report.training_source_episode_counts,
                {"manual": 1, "expert": 2},
            )

    def test_writes_one_target_row_per_unique_expert_without_depth_features(self):
        import json
        from dataset.converter import convert_sessions

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            session = root / "session"
            create_episode(session, index=0, source="manual")
            create_episode(session, index=1, source="expert")
            create_episode(
                session,
                index=2,
                source="expert",
                target_source="derived_expert_close",
            )
            output = root / "dataset"

            report = convert_sessions(
                [session],
                output,
                "local/test",
                control_period_ms=150,
                use_videos=False,
                source_repeats={"expert": 2},
            )

            rows = [
                json.loads(line)
                for line in (output / "quality" / "expert-targets.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(
                {row["target_source"] for row in rows},
                {"rgbd_localization", "derived_expert_close"},
            )
            self.assertEqual(report.raw_schema_version, 7)
            self.assertEqual(report.expert_episodes, 2)
            self.assertEqual(report.rgbd_localization_episodes, 1)
            self.assertEqual(report.derived_expert_close_episodes, 1)
            self.assertEqual(report.expert_target_coverage, 1.0)
            self.assertFalse(any("depth" in key or "expert" in key for key in report.feature_keys))


if __name__ == "__main__":
    unittest.main()
