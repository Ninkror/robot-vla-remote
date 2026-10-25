import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def create_episode(
    root: Path,
    index: int = 0,
    timestamps_ns: tuple[int, ...] | None = None,
    frame_timestamps_ns: tuple[int, ...] | None = None,
    source: str = "manual",
    target_source: str = "rgbd_localization",
) -> Path:
    episode = root / "episodes" / f"episode_20260929T00000{index}Z_abcd"
    (episode / "rgb").mkdir(parents=True)
    timestamps_ns = timestamps_ns or (0, 150_000_000, 300_000_000)
    frame_timestamps_ns = frame_timestamps_ns or timestamps_ns
    states = [
        {
            "request_timestamp_ns": timestamp,
            "response_timestamp_ns": timestamp,
            "timestamp_ns": timestamp,
            "timestamp_source": "service_request_response_midpoint",
            "values": [500 + index] * 6,
        }
        for index, timestamp in enumerate(timestamps_ns)
    ]
    actions = [
        {
            "duration_s": 0.15,
            "timestamp_ns": timestamp if index == 0 else timestamp + 10_000_000,
            "timestamp_source": "ros_subscription_arrival",
            "values": [510 + index] * 6,
        }
        for index, timestamp in enumerate(timestamps_ns)
    ]
    states_by_timestamp = {
        row["timestamp_ns"]: row["values"] for row in states
    }
    rows = []
    for frame, timestamp in enumerate(frame_timestamps_ns):
        state = states_by_timestamp[timestamp]
        paired_action = max(
            (row for row in actions if row["timestamp_ns"] <= timestamp),
            key=lambda row: row["timestamp_ns"],
        )
        rgb_name = f"rgb/{frame:06d}.jpg"
        cv2.imwrite(str(episode / rgb_name), np.zeros((4, 5, 3), dtype=np.uint8))
        rows.append({"timestamp_ns": timestamp, "rgb_timestamp_ns": timestamp, "rgb": rgb_name,
                     "image_arrival_timestamp_ns": timestamp,
                     "state_timestamp_ns": timestamp,
                     "state_request_timestamp_ns": timestamp,
                     "state_response_timestamp_ns": timestamp,
                     "state": state,
                     "action_timestamp_ns": paired_action["timestamp_ns"],
                     "action_duration_s": 0.15,
                     "action": paired_action["values"]})
    for name, entries in (("frames.jsonl", rows), ("states.jsonl", states), ("actions.jsonl", actions)):
        (episode / name).write_text("".join(json.dumps(item) + "\n" for item in entries), encoding="utf-8")
    expert = None
    if source == "expert":
        expert_dir = episode / "expert"
        expert_dir.mkdir()
        anchor_file = expert_dir / "anchor_rgb.jpg"
        anchor_file.write_bytes((episode / "rgb/000000.jpg").read_bytes())
        state = [float(value) for value in states[0]["values"]]
        close_action = [520.0, 530.0, 540.0, 550.0, 510.0, 520.0]
        localization = target_source == "rgbd_localization"
        expert = {
            "target_source": target_source,
            "anchor": {
                "rgb_timestamp_ns": frame_timestamps_ns[0],
                "rgb_arrival_timestamp_ns": frame_timestamps_ns[0] + 1,
                "state_timestamp_ns": states[0]["timestamp_ns"],
                "state": state,
                "rgb_file": "expert/anchor_rgb.jpg",
                "pixel_xy": [2, 1] if localization else None,
                "camera_xyz_m": [0.01, 0.02, 0.30] if localization else None,
                "detected_object_base_xyz_m": [0.21, -0.03, 0.04] if localization else None,
            },
            "plan": {
                "grasp_target_base_xyz_m": [0.22, -0.03, 0.04],
                "pregrasp_target_base_xyz_m": [0.22, -0.03, 0.08] if localization else None,
                "ik_targets": [
                    [410.0, 420.0, 430.0, 440.0, 450.0, 150.0],
                    [510.0, 520.0, 530.0, 540.0, 550.0, 150.0],
                    close_action,
                ] if localization else [close_action],
                "approach_height_m": 0.04 if localization else None,
                "grasp_offset_m": [0.01, 0.0, 0.0] if localization else None,
                "pitch_deg": 80.0 if localization else None,
                "gripper_open_pulse": 150 if localization else None,
                "gripper_closed_pulse": 520,
            },
            "calibration": {
                "status": "verified" if localization else "unavailable_legacy",
                "hand_to_camera": np.eye(4).tolist() if localization else None,
                "workspace_bounds_m": [[-0.2, 0.4], [-0.2, 0.2], [0.0, 1.0]] if localization else None,
                "fingerprint_sha256": "a" * 64 if localization else None,
            },
            "close_event": {
                "command_timestamp_ns": frame_timestamps_ns[min(1, len(frame_timestamps_ns) - 1)],
                "completion_timestamp_ns": timestamps_ns[-1],
                "frame_index": min(1, len(frame_timestamps_ns) - 1),
                "action": close_action,
                "state": [float(value) for value in states[-1]["values"]],
                "state_timestamp_ns": timestamps_ns[-1],
                "fk_xyz_m": [0.22, -0.03, 0.04],
            },
            "operator_confirmed_success": True,
        }
    checksums = {path.relative_to(episode).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in episode.rglob("*") if path.is_file()}
    manifest = {"schema_version": 7, "episode_id": episode.name, "task": "抓取红色方块",
                "outcome": "success", "operator_confirmed_success": True,
                "source": source, "provenance": "native_rgb_action_v7",
                "recording_profile": "rgb_state_action_v1", "control_period_ms": 150,
                "servo_ids": [1, 2, 3, 4, 5, 10], "frame_count": len(frame_timestamps_ns),
                "image_shape": [4, 5, 3], "checksums": checksums,
                "stream_counts": {"rgb": len(frame_timestamps_ns), "state": len(states),
                                  "action": len(actions), "aligned": len(frame_timestamps_ns)},
                "timebase": {"clock": "monotonic",
                             "rgb_timestamp": "ros_header_projected_to_monotonic",
                             "state_timestamp": "service_request_response_midpoint",
                             "action_timestamp": "ros_subscription_arrival",
                             "alignment": "rgb_anchor_nearest_state_latest_prior_action"},
                "timing": {name: {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
                           for name in ("image_gap", "image_transport_latency", "state_skew",
                                        "state_round_trip", "action_age")}}
    if expert is not None:
        manifest["expert"] = expert
    (episode / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return episode
