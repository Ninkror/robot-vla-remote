from __future__ import annotations

import hashlib
from pathlib import Path
import time

import numpy as np
from openpi.policies import policy_config

from action import ActionCodecConfig
from pi05.config import make_train_config


class Pi05Runner:
    def __init__(self, model_path: str, actions_per_chunk: int, device: str, codec_path: str):
        self.model_path = str(Path(model_path).resolve())
        if not Path(self.model_path).is_dir():
            raise ValueError(f"model path is not a directory: {self.model_path}")
        self.codec_sha256 = hashlib.sha256(Path(codec_path).read_bytes()).hexdigest()
        ActionCodecConfig.load(codec_path)
        self.actions_per_chunk = int(actions_per_chunk)
        if self.actions_per_chunk != 72:
            raise ValueError("PI 0.5 ROSOrin checkpoint requires a 72-row action horizon")
        self.device = "gpu"
        config = make_train_config(
            episodes=(0,),
            steps=1,
            batch_size=1,
            experiment="inference",
        )
        self.policy = policy_config.create_trained_policy(config, self.model_path)

    def predict(self, image_rgb: np.ndarray, state_values, task: str):
        image = np.asarray(image_rgb, dtype=np.uint8)
        state = np.asarray(state_values, dtype=np.float32)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape [height, width, 3]")
        if state.shape != (6,) or not np.isfinite(state).all():
            raise ValueError("state must contain six finite values")
        started = time.perf_counter()
        output = self.policy.infer(
            {
                "observation/image": image,
                "observation/state": state,
                "prompt": task.strip(),
            }
        )
        elapsed = time.perf_counter() - started
        actions = np.asarray(output["actions"], dtype=np.float32)
        if actions.shape != (self.actions_per_chunk, 6) or not np.isfinite(actions).all():
            raise ValueError(f"PI policy returned an invalid action shape: {actions.shape}")
        return actions, elapsed
