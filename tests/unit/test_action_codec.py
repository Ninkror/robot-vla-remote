import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from action import (
    ActionCodecConfig,
    ActionCodecError,
    DynamicBoundedRelativeCodec,
)


class ActionCodecTests(unittest.TestCase):
    def setUp(self):
        self.config = ActionCodecConfig.default_v1()
        self.codec = DynamicBoundedRelativeCodec(self.config)
        self.state = np.array([500, 725, 47, 148, 500, 498], dtype=np.float32)

    def test_projects_soft_limit_and_round_trips_endpoint(self):
        absolute = np.array([[500, 725, 0, 148, 500, 498]], dtype=np.float32)
        latent, report = self.codec.encode_episode(self.state, absolute)
        decoded = self.codec.decode_chunk(self.state, latent)
        self.assertTrue(np.isfinite(latent).all())
        self.assertGreaterEqual(float(decoded[0, 2]), 20.0)
        self.assertAlmostEqual(float(decoded[0, 2]), 20.0, delta=0.02)
        self.assertEqual(report.projected_rows, 1)
        self.assertEqual(report.projected_by_servo[2], 1)
        self.assertFalse(latent.flags.writeable)
        self.assertFalse(decoded.flags.writeable)

    def test_extreme_latent_stays_within_recursive_dynamic_bounds(self):
        latent = np.tile(np.array([-1e6, 1e6, -1e6, 1e6, -1e6, 1e6]), (30, 1))
        decoded = self.codec.decode_chunk(self.state, latent)
        previous = self.state.astype(np.float64)
        for row in decoded:
            lower, upper = self.codec.dynamic_bounds(previous)
            self.assertTrue(np.all(row >= lower - 1e-4))
            self.assertTrue(np.all(row <= upper + 1e-4))
            previous = row

    def test_hard_invalid_initial_state_is_rejected_with_servo(self):
        invalid = self.state.copy()
        invalid[1] = 80
        with self.assertRaisesRegex(ActionCodecError, r"servo 2"):
            self.codec.decode_chunk(invalid, np.zeros((1, 6), dtype=np.float32))

    def test_soft_outside_state_can_only_hold_or_move_inward(self):
        below_soft = self.state.copy()
        below_soft[2] = 10
        decoded = self.codec.decode_chunk(
            below_soft,
            np.array([[-1e6, 0, -1e6, 0, 0, 0], [0, 0, 1e6, 0, 0, 0]]),
        )
        self.assertGreaterEqual(float(decoded[0, 2]), 10.0)
        self.assertLessEqual(float(decoded[0, 2]), 20.0)
        self.assertGreaterEqual(float(decoded[1, 2]), float(decoded[0, 2]))
        self.assertLessEqual(float(decoded[1, 2]), 20.0)

    def test_encode_rejects_target_outside_dynamic_interval(self):
        absolute = self.state[None, :].copy()
        absolute[0, 0] += self.config.max_step[0] + 1
        with self.assertRaisesRegex(ActionCodecError, r"row 0.*servo 1"):
            self.codec.encode_episode(self.state, absolute)

    def test_config_json_round_trip_is_byte_stable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "action_codec.json"
            self.config.save(path)
            first = path.read_bytes()
            loaded = ActionCodecConfig.load(path)
            loaded.save(path)
            self.assertEqual(first, path.read_bytes())
            self.assertEqual(self.config, loaded)

    def test_schema_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "action_codec.json"
            payload = self.config.to_dict()
            payload["schema"] = "wrong"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ActionCodecError, "schema"):
                ActionCodecConfig.load(path)


if __name__ == "__main__":
    unittest.main()
