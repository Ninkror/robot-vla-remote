import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from dataset.raw import (
    MAX_STATE_SKEW_NS,
    RawDataError,
    load_episode,
    load_session,
    resample_episode,
)
from helpers import create_episode


class RawV7Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = create_episode(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def test_state_skew_limit_is_100_ms(self):
        self.assertEqual(MAX_STATE_SKEW_NS, 100_000_000)

    def test_loads_rgb_and_future_command_at_150_ms_period(self):
        episode = load_episode(self.path)
        self.assertEqual(episode.samples[0].action, (510.0,) * 6)
        self.assertEqual(episode.source, "manual")
        frames = resample_episode(episode, 150_000_000, 1_000_000)
        self.assertEqual(len(frames), 3)
        self.assertEqual(frames[1].source_timestamp_ns, 150_000_000)

    def test_resampling_keeps_state_and_action_streams_independent_of_rgb(self):
        stream_times = tuple(range(0, 1_200_000_001, 150_000_000))
        path = create_episode(
            self.root,
            index=1,
            timestamps_ns=stream_times,
            frame_timestamps_ns=(0, 900_000_000, 1_200_000_000),
        )
        episode = load_episode(path)

        frames = resample_episode(episode, 150_000_000, 600_000_000)

        self.assertEqual(len(frames), 9)
        self.assertEqual(frames[3].source_timestamp_ns, 0)
        self.assertEqual(frames[3].state, (503.0,) * 6)
        self.assertEqual(frames[3].action, (513.0,) * 6)

    def test_depth_fields_are_rejected(self):
        manifest_path = self.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["depth_unit"] = "mm"
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RawDataError, "depth fields"):
            load_episode(self.path)

    def test_previous_schema_is_rejected(self):
        manifest_path = self.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["schema_version"] = 5
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RawDataError, "schema_version 7"):
            load_episode(self.path)

    def test_checksum_mismatch_is_rejected(self):
        (self.path / "rgb/000000.jpg").write_bytes(b"corrupt")
        with self.assertRaisesRegex(RawDataError, "checksum mismatch"):
            load_episode(self.path)

    def test_referenced_image_must_be_in_checksums(self):
        manifest_path = self.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        del manifest["checksums"]["rgb/000000.jpg"]
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RawDataError, "RGB file is absent from checksums"):
            load_episode(self.path)

    def test_session_uses_only_v7_episode_directory(self):
        create_episode(self.root, 1)
        episodes, rejected = load_session(self.root)
        self.assertEqual(len(episodes), 2)
        self.assertEqual(rejected, {})
        self.assertEqual([episode.episode_index for episode in episodes], [0, 1])

    def test_rejects_missing_raw_timing_provenance(self):
        state_path = self.path / "states.jsonl"
        states = [json.loads(line) for line in state_path.read_text().splitlines()]
        del states[0]["request_timestamp_ns"]
        state_path.write_text("".join(json.dumps(row) + "\n" for row in states))
        manifest_path = self.path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["checksums"]["states.jsonl"] = hashlib.sha256(
            state_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

        with self.assertRaisesRegex(RawDataError, "request timestamp"):
            load_episode(self.path)

    def test_loads_localized_expert_target(self):
        path = create_episode(self.root, 2, source="expert")

        episode = load_episode(path)

        self.assertIsNotNone(episode.expert_target)
        self.assertEqual(episode.expert_target.target_source, "rgbd_localization")
        self.assertEqual(episode.expert_target.anchor_state, (500.0,) * 6)
        self.assertEqual(episode.expert_target.grasp_target_base_xyz_m, (0.22, -0.03, 0.04))
        self.assertEqual(len(episode.expert_target.ik_targets), 3)

    def test_loads_explicit_derived_expert_without_invented_object_coordinates(self):
        path = create_episode(
            self.root, 3, source="expert", target_source="derived_expert_close"
        )

        target = load_episode(path).expert_target

        self.assertEqual(target.target_source, "derived_expert_close")
        self.assertIsNone(target.detected_object_base_xyz_m)
        self.assertEqual(target.calibration_status, "unavailable_legacy")
        self.assertEqual(len(target.ik_targets), 1)

    def test_localized_expert_requires_object_coordinates(self):
        path = create_episode(self.root, 4, source="expert")
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["expert"]["anchor"]["detected_object_base_xyz_m"] = None
        manifest_path.write_text(json.dumps(manifest))

        with self.assertRaisesRegex(RawDataError, "detected_object"):
            load_episode(path)

    def test_expert_requires_confirmed_success_and_checksummed_anchor(self):
        path = create_episode(self.root, 5, source="expert")
        manifest_path = path / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["expert"]["operator_confirmed_success"] = False
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RawDataError, "confirmed"):
            load_episode(path)

        manifest["expert"]["operator_confirmed_success"] = True
        del manifest["checksums"]["expert/anchor_rgb.jpg"]
        manifest_path.write_text(json.dumps(manifest))
        with self.assertRaisesRegex(RawDataError, "anchor"):
            load_episode(path)


if __name__ == "__main__":
    unittest.main()
