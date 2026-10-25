from pathlib import Path
import hashlib
import json
import tempfile
import unittest

from dataset.raw import ConvertedFrame, RawEpisode, RawSampleEntry
from helpers import create_episode


def _values(gripper: float) -> tuple[float, ...]:
    return (500.0, 500.0, 500.0, 500.0, 500.0, gripper)


def _episode(tmp_path: Path, *, final_gripper: float = 520.0) -> tuple[RawEpisode, list[ConvertedFrame]]:
    image = tmp_path / "last.jpg"
    image.write_bytes(b"image")
    sample = RawSampleEntry(
        0,
        0,
        0,
        0,
        image,
        _values(400.0),
        _values(432.0),
    )
    episode = RawEpisode(
        path=tmp_path / "episode_success",
        episode_index=0,
        task="抓取红色方块",
        image_shape=(4, 5, 3),
        samples=(sample,),
        state_events=(
            (0, _values(400.0)),
            (150_000_000, _values(470.0)),
            (300_000_000, _values(503.0)),
        ),
        action_events=(
            (0, _values(432.0)),
            (150_000_000, _values(490.0)),
            (300_000_000, _values(final_gripper)),
        ),
        source="expert",
        provenance="native_rgb_action_v7",
    )
    frames = [ConvertedFrame(0, image, _values(400.0), _values(432.0), 0)]
    return episode, frames


class TerminalHoldTests(unittest.TestCase):
    def test_extends_late_events_then_appends_ten_terminal_rows(self):
        from dataset.converter import extend_confirmed_terminal_hold

        with tempfile.TemporaryDirectory() as name:
            episode, frames = _episode(Path(name))
            augmented, report = extend_confirmed_terminal_hold(
                episode, frames, period_ns=150_000_000, hold_rows=10
            )

        self.assertEqual([row.action[5] for row in augmented[:3]], [432.0, 490.0, 520.0])
        self.assertEqual(len(augmented), 13)
        self.assertTrue(all(row.action == _values(520.0) for row in augmented[-10:]))
        self.assertTrue(all(row.state == _values(503.0) for row in augmented[-10:]))
        self.assertTrue(all(row.image_path == frames[-1].image_path for row in augmented[-12:]))
        self.assertEqual(report.original_rows, 1)
        self.assertEqual(report.event_extension_rows, 2)
        self.assertEqual(report.hold_rows, 10)
        self.assertEqual(report.output_rows, 13)

    def test_rejects_episode_without_confirmed_closed_terminal_action(self):
        from dataset.converter import extend_confirmed_terminal_hold

        with tempfile.TemporaryDirectory() as name:
            episode, frames = _episode(Path(name), final_gripper=517.0)
            with self.assertRaisesRegex(Exception, "520"):
                extend_confirmed_terminal_hold(
                    episode, frames, period_ns=150_000_000, hold_rows=10
                )

    def test_rejects_nonpositive_hold_rows(self):
        from dataset.converter import extend_confirmed_terminal_hold

        with tempfile.TemporaryDirectory() as name:
            episode, frames = _episode(Path(name))
            with self.assertRaisesRegex(Exception, "hold_rows"):
                extend_confirmed_terminal_hold(
                    episode, frames, period_ns=150_000_000, hold_rows=0
                )

    def test_conversion_report_records_terminal_augmentation(self):
        from dataset.converter import convert_sessions

        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            episode_path = create_episode(root / "session", source="expert")
            actions_path = episode_path / "actions.jsonl"
            actions = [json.loads(line) for line in actions_path.read_text().splitlines()]
            actions[-1]["values"][5] = 520.0
            actions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in actions), encoding="utf-8"
            )
            frames_path = episode_path / "frames.jsonl"
            frames = [json.loads(line) for line in frames_path.read_text().splitlines()]
            action_by_time = {row["timestamp_ns"]: row["values"] for row in actions}
            for row in frames:
                row["action"] = action_by_time[row["action_timestamp_ns"]]
            frames_path.write_text(
                "".join(json.dumps(row) + "\n" for row in frames), encoding="utf-8"
            )
            manifest_path = episode_path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["checksums"]["actions.jsonl"] = hashlib.sha256(
                actions_path.read_bytes()
            ).hexdigest()
            manifest["checksums"]["frames.jsonl"] = hashlib.sha256(
                frames_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            report = convert_sessions(
                [root / "session"],
                root / "dataset",
                "local/test",
                control_period_ms=150,
                use_videos=False,
                terminal_hold_rows=10,
            )

        self.assertEqual(report.total_frames, 13)
        self.assertEqual(report.terminal_hold_rows, 10)
        self.assertEqual(report.terminal_augmented_rows, 10)
        self.assertEqual(report.terminal_reports["0"]["hold_rows"], 10)


if __name__ == "__main__":
    unittest.main()
