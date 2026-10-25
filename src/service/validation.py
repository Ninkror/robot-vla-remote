from collections.abc import Sequence

import numpy as np


def validate_state(values: Sequence[float]) -> np.ndarray:
    state = np.array(values, dtype=np.float32, copy=True)
    if state.shape != (6,):
        raise ValueError("state must contain exactly 6 values")
    if not np.isfinite(state).all():
        raise ValueError("state values must be finite")
    state.setflags(write=False)
    return state


def decode_rgb(height: int, width: int, encoding: str, row_stride: int, data: bytes) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if encoding not in {"rgb8", "bgr8"}:
        raise ValueError(f"unsupported image encoding: {encoding}")
    row_bytes = width * 3
    if row_stride < row_bytes:
        raise ValueError("row stride is smaller than the RGB row width")
    if len(data) != height * row_stride:
        raise ValueError("image buffer length does not match height and row stride")
    rows = np.frombuffer(data, dtype=np.uint8).reshape(height, row_stride)
    image = rows[:, :row_bytes].reshape(height, width, 3)
    if encoding == "bgr8":
        image = image[:, :, ::-1]
    image = np.ascontiguousarray(image)
    image.setflags(write=False)
    return image
