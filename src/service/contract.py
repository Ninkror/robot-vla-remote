"""Explicit input/output contract for the currently supported inference profile."""

import json
from pathlib import Path


def load_model_contract(model_path: Path | str) -> dict:
    path = Path(model_path) / "model_contract.json"
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"missing or invalid model_contract.json: {error}") from error
    expected = {
        "schema_version": 1,
        "input_profile": "rgb_state_v1",
        "camera_mapping": "single_camera_camera1",
        "action_representation": "dynamic_bounded_relative_v1",
    }
    if not isinstance(contract, dict) or any(contract.get(key) != value for key, value in expected.items()):
        raise ValueError("model_contract.json is incompatible with this inference service")
    return contract
