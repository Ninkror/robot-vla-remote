import json
from pathlib import Path
import tempfile
import unittest


class ModelContractTests(unittest.TestCase):
    def test_requires_explicit_rgb_state_contract(self):
        from service.contract import load_model_contract

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with self.assertRaisesRegex(ValueError, "model_contract"):
                load_model_contract(path)
            (path / "model_contract.json").write_text(json.dumps({
                "schema_version": 1,
                "input_profile": "rgb_state_v1",
                "camera_mapping": "single_camera_camera1",
                "action_representation": "dynamic_bounded_relative_v1",
            }))
            self.assertEqual(load_model_contract(path)["input_profile"], "rgb_state_v1")
            (path / "model_contract.json").write_text(json.dumps({
                "schema_version": 1, "input_profile": "rgbd", "camera_mapping": "single_camera_camera1",
                "action_representation": "dynamic_bounded_relative_v1",
            }))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_model_contract(path)
