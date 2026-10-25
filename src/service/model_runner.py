import time
import hashlib
from pathlib import Path

import numpy as np
import torch
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from action import ActionCodecConfig, DynamicBoundedRelativeCodec
from .contract import load_model_contract

from .validation import validate_state


class SmolVLARunner:
    def __init__(self, model_path: str, actions_per_chunk: int = 72, device: str = "cuda"):
        if actions_per_chunk < 1:
            raise ValueError("actions_per_chunk must be positive")
        resolved_model_path = Path(model_path).resolve()
        if not resolved_model_path.is_dir():
            raise ValueError(f"model path is not a directory: {resolved_model_path}")
        self.model_path = str(resolved_model_path)
        load_model_contract(resolved_model_path)
        codec_path = resolved_model_path / "action_codec.json"
        try:
            self.codec_sha256 = hashlib.sha256(codec_path.read_bytes()).hexdigest()
        except OSError as error:
            raise ValueError(f"unable to read action codec: {codec_path}") from error
        codec_config = ActionCodecConfig.load(codec_path)
        if codec_config.servo_ids != (1, 2, 3, 4, 5, 10):
            raise ValueError("action codec servo order must be [1, 2, 3, 4, 5, 10]")
        self.action_codec = DynamicBoundedRelativeCodec(codec_config)
        self.actions_per_chunk = actions_per_chunk
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")

        self.policy = SmolVLAPolicy.from_pretrained(self.model_path)
        self.policy.to(self.device).eval()
        if actions_per_chunk > self.policy.config.chunk_size:
            raise ValueError(
                f"actions_per_chunk {actions_per_chunk} exceeds model chunk size "
                f"{self.policy.config.chunk_size}"
            )
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            self.model_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def predict(
        self,
        image_rgb: np.ndarray,
        state_values,
        task: str,
    ) -> tuple[np.ndarray, float]:
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        image = np.asarray(image_rgb)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must have shape [height, width, 3]")
        if image.dtype != np.uint8:
            raise ValueError("image must have uint8 dtype")
        state = validate_state(state_values)

        image_tensor = (
            torch.from_numpy(np.array(image, copy=True))
            .permute(2, 0, 1)
            .contiguous()
            .to(torch.float32)
            .div_(255.0)
        )
        raw_observation = {
            "observation.state": torch.from_numpy(np.array(state, copy=True)),
            "observation.images.camera1": image_tensor,
            "task": task.strip(),
            "robot_type": "rosorin",
        }
        processed = self.preprocessor(raw_observation)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        with torch.inference_mode():
            chunk = self.policy.predict_action_chunk(processed)
            if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[2] != 6:
                raise ValueError("policy output must be batched six-dimensional actions")
            if chunk.shape[1] < self.actions_per_chunk:
                raise ValueError("policy output contains fewer actions than requested")
            chunk = chunk[:, : self.actions_per_chunk, :]
            postprocessed = self.postprocessor(chunk)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - started

        latent_actions = postprocessed.detach().to(torch.float32).cpu().numpy()
        if latent_actions.shape != (1, self.actions_per_chunk, 6):
            raise ValueError("postprocessed action chunk has an unexpected shape")
        latent_actions = np.array(latent_actions[0], dtype=np.float32, copy=True)
        if not np.isfinite(latent_actions).all():
            raise ValueError("postprocessed action values must be finite")
        actions = self.action_codec.decode_chunk(state, latent_actions)
        return actions, elapsed
