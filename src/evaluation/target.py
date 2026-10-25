from __future__ import annotations

import math

import numpy as np


def _actions(values) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 6 or result.shape[0] < 1:
        raise ValueError("actions must have shape [N, 6]")
    if not np.isfinite(result).all():
        raise ValueError("actions must contain finite values")
    return result


def find_close_row(actions: np.ndarray, open_pulse: float, closed_pulse: float) -> int | None:
    values = _actions(actions)
    open_value = float(open_pulse)
    closed_value = float(closed_pulse)
    if not math.isfinite(open_value) or not math.isfinite(closed_value) or open_value == closed_value:
        raise ValueError("open and closed gripper pulses must be distinct finite values")
    threshold = (open_value + closed_value) / 2.0
    matches = values[:, 5] >= threshold if closed_value > open_value else values[:, 5] <= threshold
    indices = np.flatnonzero(matches)
    return int(indices[0]) if len(indices) else None


def joint_endpoint_metrics(predicted, target) -> dict:
    predicted_values = np.asarray(predicted, dtype=np.float64)
    target_values = np.asarray(target, dtype=np.float64)
    if predicted_values.shape != (6,) or target_values.shape != (6,):
        raise ValueError("predicted and target endpoints must each contain six values")
    if not np.isfinite(predicted_values).all() or not np.isfinite(target_values).all():
        raise ValueError("predicted and target endpoints must be finite")
    signed = predicted_values - target_values
    absolute = np.abs(signed)
    return {
        "signed_error": signed.tolist(),
        "absolute_error": absolute.tolist(),
        "arm_mae": float(absolute[:5].mean()),
        "arm_max": float(absolute[:5].max()),
        "gripper_error": float(signed[5]),
    }


def evaluate_actions(actions, target, open_pulse: float, closed_pulse: float) -> dict:
    values = _actions(actions)
    close_row = find_close_row(values, open_pulse, closed_pulse)
    if close_row is None:
        return {"status": "no_close_row", "close_row_index": None}
    return {
        "status": "ok",
        "close_row_index": close_row,
        **joint_endpoint_metrics(values[close_row], target),
    }
