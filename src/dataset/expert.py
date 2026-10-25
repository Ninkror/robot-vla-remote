from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
import re

import cv2
import numpy as np


@dataclass(frozen=True)
class ExpertTarget:
    target_source: str
    anchor_image_path: Path
    anchor_rgb_timestamp_ns: int
    anchor_state_timestamp_ns: int
    anchor_state: tuple[float, ...]
    pixel_xy: tuple[int, int] | None
    camera_xyz_m: tuple[float, float, float] | None
    detected_object_base_xyz_m: tuple[float, float, float] | None
    grasp_target_base_xyz_m: tuple[float, float, float]
    pregrasp_target_base_xyz_m: tuple[float, float, float] | None
    ik_targets: tuple[tuple[float, ...], ...]
    gripper_open_pulse: float | None
    gripper_closed_pulse: float
    calibration_status: str
    calibration_fingerprint_sha256: str | None
    close_command_timestamp_ns: int
    close_completion_timestamp_ns: int
    close_frame_index: int
    close_action: tuple[float, ...]
    close_state: tuple[float, ...]
    close_state_timestamp_ns: int
    close_fk_xyz_m: tuple[float, float, float]


def _object(value, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _timestamp(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _number(value, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _vector(value, size: int, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{name} must contain {size} values")
    result = tuple(_number(item, name) for item in value)
    return result


def _pulse_vector(value, name: str) -> tuple[float, ...]:
    result = _vector(value, 6, name)
    if any(item < 0.0 or item > 1000.0 for item in result):
        raise ValueError(f"{name} must stay within 0..1000 pulse")
    return result


def _optional_vector(value, size: int, name: str):
    return None if value is None else _vector(value, size, name)


def _safe_file(base: Path, relative, checksums: dict, name: str) -> Path:
    if not isinstance(relative, str) or relative not in checksums:
        raise ValueError(f"{name} must be present in checksums")
    target = (base / relative).resolve()
    if not target.is_relative_to(base.resolve()) or not target.is_file() or target.is_symlink():
        raise ValueError(f"unsafe or missing {name}")
    return target


def parse_expert_target(
    path: Path,
    manifest: dict,
    checksums: dict,
) -> ExpertTarget:
    expert = _object(manifest.get("expert"), "expert")
    if expert.get("operator_confirmed_success") is not True:
        raise ValueError("expert episode must be operator confirmed")
    target_source = expert.get("target_source")
    if target_source not in {"rgbd_localization", "derived_expert_close"}:
        raise ValueError("unknown expert target_source")

    anchor = _object(expert.get("anchor"), "expert.anchor")
    anchor_image = _safe_file(
        path, anchor.get("rgb_file"), checksums, "expert anchor RGB"
    )
    image = cv2.imread(str(anchor_image), cv2.IMREAD_COLOR)
    if image is None or tuple(image.shape) != tuple(manifest.get("image_shape", ())):
        raise ValueError("expert anchor RGB is unreadable or has wrong shape")
    anchor_state = _pulse_vector(anchor.get("state"), "expert anchor state")
    anchor_rgb_timestamp = _timestamp(
        anchor.get("rgb_timestamp_ns"), "expert anchor RGB timestamp"
    )
    anchor_state_timestamp = _timestamp(
        anchor.get("state_timestamp_ns"), "expert anchor state timestamp"
    )

    plan = _object(expert.get("plan"), "expert.plan")
    grasp_target = _vector(
        plan.get("grasp_target_base_xyz_m"), 3, "grasp_target_base_xyz_m"
    )
    raw_targets = plan.get("ik_targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise ValueError("ik_targets must be a non-empty list")
    ik_targets = tuple(_pulse_vector(row, "ik target") for row in raw_targets)
    closed_pulse = _number(plan.get("gripper_closed_pulse"), "gripper_closed_pulse")

    calibration = _object(expert.get("calibration"), "expert.calibration")
    status = calibration.get("status")
    pixel_xy = None
    camera_xyz = None
    object_base_xyz = None
    pregrasp = None
    open_pulse = None
    fingerprint = None
    if target_source == "rgbd_localization":
        pixel = anchor.get("pixel_xy")
        if (
            not isinstance(pixel, (list, tuple))
            or len(pixel) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in pixel)
        ):
            raise ValueError("pixel_xy must contain two integers")
        pixel_xy = (int(pixel[0]), int(pixel[1]))
        camera_xyz = _vector(anchor.get("camera_xyz_m"), 3, "camera_xyz_m")
        object_base_xyz = _vector(
            anchor.get("detected_object_base_xyz_m"),
            3,
            "detected_object_base_xyz_m",
        )
        pregrasp = _vector(
            plan.get("pregrasp_target_base_xyz_m"), 3, "pregrasp_target_base_xyz_m"
        )
        if len(ik_targets) != 3:
            raise ValueError("localized expert must contain three ik_targets")
        _number(plan.get("approach_height_m"), "approach_height_m")
        _vector(plan.get("grasp_offset_m"), 3, "grasp_offset_m")
        _number(plan.get("pitch_deg"), "pitch_deg")
        open_pulse = _number(plan.get("gripper_open_pulse"), "gripper_open_pulse")
        if status != "verified":
            raise ValueError("localized expert calibration must be verified")
        matrix = np.asarray(calibration.get("hand_to_camera"), dtype=np.float64)
        if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
            raise ValueError("hand_to_camera must be a finite homogeneous 4x4 matrix")
        bounds = np.asarray(calibration.get("workspace_bounds_m"), dtype=np.float64)
        if bounds.shape != (3, 2) or not np.isfinite(bounds).all() or np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("workspace_bounds_m must contain three increasing bounds")
        fingerprint = calibration.get("fingerprint_sha256")
        if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
            raise ValueError("calibration fingerprint_sha256 is invalid")
    else:
        for key in ("pixel_xy", "camera_xyz_m", "detected_object_base_xyz_m"):
            if anchor.get(key) is not None:
                raise ValueError(f"derived expert must not invent {key}")
        for key in (
            "pregrasp_target_base_xyz_m",
            "approach_height_m",
            "grasp_offset_m",
            "pitch_deg",
            "gripper_open_pulse",
        ):
            if plan.get(key) is not None:
                raise ValueError(f"derived expert must mark {key} unavailable")
        if status != "unavailable_legacy":
            raise ValueError("derived expert calibration status must be unavailable_legacy")
        for key in ("hand_to_camera", "workspace_bounds_m", "fingerprint_sha256"):
            if calibration.get(key) is not None:
                raise ValueError(f"derived expert must mark calibration {key} unavailable")

    close = _object(expert.get("close_event"), "expert.close_event")
    close_start = _timestamp(
        close.get("command_timestamp_ns"), "close command timestamp"
    )
    close_completion = _timestamp(
        close.get("completion_timestamp_ns"), "close completion timestamp"
    )
    if close_completion < close_start:
        raise ValueError("close completion precedes close command")
    frame_index = _timestamp(close.get("frame_index"), "close frame index")
    if frame_index >= int(manifest.get("frame_count", 0)):
        raise ValueError("close frame index is outside the episode")
    close_state_timestamp = _timestamp(
        close.get("state_timestamp_ns"), "close state timestamp"
    )
    if close_state_timestamp < close_start or close_state_timestamp > close_completion:
        raise ValueError("close state timestamp is outside the close interval")

    return ExpertTarget(
        target_source=target_source,
        anchor_image_path=anchor_image,
        anchor_rgb_timestamp_ns=anchor_rgb_timestamp,
        anchor_state_timestamp_ns=anchor_state_timestamp,
        anchor_state=anchor_state,
        pixel_xy=pixel_xy,
        camera_xyz_m=camera_xyz,
        detected_object_base_xyz_m=object_base_xyz,
        grasp_target_base_xyz_m=grasp_target,
        pregrasp_target_base_xyz_m=pregrasp,
        ik_targets=ik_targets,
        gripper_open_pulse=open_pulse,
        gripper_closed_pulse=closed_pulse,
        calibration_status=str(status),
        calibration_fingerprint_sha256=fingerprint,
        close_command_timestamp_ns=close_start,
        close_completion_timestamp_ns=close_completion,
        close_frame_index=frame_index,
        close_action=_pulse_vector(close.get("action"), "close action"),
        close_state=_pulse_vector(close.get("state"), "close state"),
        close_state_timestamp_ns=close_state_timestamp,
        close_fk_xyz_m=_vector(close.get("fk_xyz_m"), 3, "close fk_xyz_m"),
    )
