import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from helpers import create_episode

try:
    from dataset.curate import trim_episode
except ImportError:
    trim_episode = None


class GraspTrimTests(unittest.TestCase):
    def test_keeps_final_close_and_settle_window_then_recomputes_manifest(self):
        if trim_episode is None:
            self.fail("dataset.curate.trim_episode is missing")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            timestamps = tuple(range(0, 750_000_001, 150_000_000))
            source = create_episode(root / "source", timestamps_ns=timestamps)
            actions_path = source / "actions.jsonl"
            actions = [json.loads(line) for line in actions_path.read_text().splitlines()]
            gripper = (500, 300, 300, 520, 520, 520)
            for row, value in zip(actions, gripper, strict=True):
                row["values"][5] = value
            actions_path.write_text(
                "".join(json.dumps(row) + "\n" for row in actions),
                encoding="utf-8",
            )
            action_by_time = {row["timestamp_ns"]: row["values"] for row in actions}
            frames_path = source / "frames.jsonl"
            frames_before_trim = [
                json.loads(line) for line in frames_path.read_text().splitlines()
            ]
            for row in frames_before_trim:
                row["action"] = action_by_time[row["action_timestamp_ns"]]
            frames_path.write_text(
                "".join(json.dumps(row) + "\n" for row in frames_before_trim),
                encoding="utf-8",
            )
            manifest_path = source / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["checksums"]["actions.jsonl"] = hashlib.sha256(
                actions_path.read_bytes()
            ).hexdigest()
            manifest["checksums"]["frames.jsonl"] = hashlib.sha256(
                frames_path.read_bytes()
            ).hexdigest()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            destination = root / "trimmed" / source.name
            result = trim_episode(
                source,
                destination,
                settle_ns=200_000_000,
                gripper_delta=5.0,
            )

            trimmed_manifest = json.loads(
                (destination / "manifest.json").read_text(encoding="utf-8")
            )
            frames = (destination / "frames.jsonl").read_text().splitlines()
            trimmed_actions = (destination / "actions.jsonl").read_text().splitlines()
            self.assertEqual(result["cutoff_ns"], 660_000_000)
            self.assertEqual(trimmed_manifest["frame_count"], 5)
            self.assertEqual(len(frames), 5)
            self.assertEqual(len(trimmed_actions), 5)
            self.assertEqual(trimmed_manifest["curation"]["settle_ms"], 200)
            self.assertEqual(trimmed_manifest["schema_version"], 7)
            self.assertEqual(trimmed_manifest["provenance"], "native_rgb_action_v7")
            for relative, expected in trimmed_manifest["checksums"].items():
                actual = hashlib.sha256((destination / relative).read_bytes()).hexdigest()
                self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
