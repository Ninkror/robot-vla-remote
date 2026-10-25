from __future__ import annotations

import dataclasses

import einops
import numpy as np
from openpi import transforms

from action import DynamicBoundedRelativeCodec


def _image(value) -> np.ndarray:
    image = np.asarray(value)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim != 3:
        raise ValueError("arm camera image must have three dimensions")
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    if image.shape[2] != 3:
        raise ValueError("arm camera image must have three channels")
    return np.asarray(image, dtype=np.uint8)


@dataclasses.dataclass(frozen=True)
class RosorinInputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        wrist = _image(data["observation/image"])
        blank = np.zeros_like(wrist)
        result = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": blank,
                "left_wrist_0_rgb": wrist,
                "right_wrist_0_rgb": blank,
            },
            "image_mask": {
                "base_0_rgb": np.False_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
            },
        }
        if "actions" in data:
            result["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            result["prompt"] = data["prompt"]
        return result


@dataclasses.dataclass(frozen=True)
class RosorinOutputs(transforms.DataTransformFn):
    codec: DynamicBoundedRelativeCodec

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"], dtype=np.float32).reshape(-1)[:6]
        latent = np.asarray(data["actions"], dtype=np.float32)
        if latent.ndim != 2 or latent.shape[1] < 6:
            raise ValueError("PI action output must have shape [N, >=6]")
        return {"actions": self.codec.decode_chunk(state, latent[:, :6])}
