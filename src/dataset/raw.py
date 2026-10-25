"""Strict version 7 RGB/state/action episode reader."""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, replace
import hashlib
import json
import math
from numbers import Real
from pathlib import Path

import cv2
import numpy as np

from .expert import ExpertTarget, parse_expert_target


SERVO_IDS = (1, 2, 3, 4, 5, 10)
STATE_LIMITS = {servo_id: (0.0, 1000.0) for servo_id in SERVO_IDS}
ACTION_LIMITS = {1: (0.0, 1000.0), 2: (100.0, 900.0), 3: (0.0, 1000.0),
                 4: (41.0, 959.0), 5: (0.0, 1000.0), 10: (100.0, 900.0)}
MAX_STATE_SKEW_NS = 100_000_000
MAX_ACTION_LOOKAHEAD_NS = 225_000_000


class RawDataError(ValueError):
    pass


@dataclass(frozen=True)
class RawSampleEntry:
    timestamp_ns: int
    image_timestamp_ns: int
    state_timestamp_ns: int
    action_timestamp_ns: int
    image_path: Path
    state: tuple[float, ...]
    action: tuple[float, ...]


@dataclass(frozen=True)
class RawEpisode:
    path: Path
    episode_index: int
    task: str
    image_shape: tuple[int, int, int]
    samples: tuple[RawSampleEntry, ...]
    state_events: tuple[tuple[int, tuple[float, ...]], ...]
    action_events: tuple[tuple[int, tuple[float, ...]], ...]
    source: str
    provenance: str
    expert_target: ExpertTarget | None = None


@dataclass(frozen=True)
class ConvertedFrame:
    timestamp_ns: int
    image_path: Path
    state: tuple[float, ...]
    action: tuple[float, ...]
    source_timestamp_ns: int


def _object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RawDataError(f"invalid JSON in {path}: {error}") from error
    if not isinstance(value, dict):
        raise RawDataError(f"JSON object required in {path}")
    return value


def _timestamp(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RawDataError(f"{name} must be a non-negative integer")
    return value


def _vector(value, limits, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != 6:
        raise RawDataError(f"{name} must contain exactly six values")
    values = []
    for servo_id, raw in zip(SERVO_IDS, value, strict=True):
        if isinstance(raw, bool) or not isinstance(raw, Real) or not math.isfinite(float(raw)):
            raise RawDataError(f"{name} servo {servo_id} must be finite")
        number = float(raw)
        low, high = limits[servo_id]
        if not low <= number <= high:
            raise RawDataError(f"{name} servo {servo_id} is outside {low:g}..{high:g}")
        values.append(number)
    return tuple(values)


def _file(base: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RawDataError("episode file path must be non-empty text")
    target = (base / relative).resolve()
    if not target.is_relative_to(base.resolve()) or not target.is_file() or target.is_symlink():
        raise RawDataError(f"unsafe or missing episode file: {relative}")
    return target


def _events(path: Path, limits, label: str) -> list[tuple[int, tuple[float, ...]]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RawDataError(f"cannot read {path}: {error}") from error
    result = []
    for line in lines:
        try:
            item = json.loads(line)
        except json.JSONDecodeError as error:
            raise RawDataError(f"invalid {label} JSON: {error}") from error
        timestamp = _timestamp(item.get("timestamp_ns"), f"{label} timestamp")
        if label == "state":
            request_ns = _timestamp(
                item.get("request_timestamp_ns"), "state request timestamp"
            )
            response_ns = _timestamp(
                item.get("response_timestamp_ns"), "state response timestamp"
            )
            if response_ns < request_ns:
                raise RawDataError("state response precedes request")
            if timestamp != request_ns + (response_ns - request_ns) // 2:
                raise RawDataError("state timestamp is not the request/response midpoint")
            if item.get("timestamp_source") != "service_request_response_midpoint":
                raise RawDataError("state timestamp source is invalid")
        elif label == "action":
            if item.get("timestamp_source") not in {
                "ros_subscription_arrival",
                "synthetic_initial_state",
            }:
                raise RawDataError("action timestamp source is invalid")
            duration_s = item.get("duration_s")
            if (
                isinstance(duration_s, bool)
                or not isinstance(duration_s, Real)
                or not math.isfinite(float(duration_s))
                or float(duration_s) < 0
            ):
                raise RawDataError("action duration must be finite and non-negative")
        result.append((timestamp, _vector(item.get("values"), limits, label)))
    if not result or any(current[0] <= previous[0] for previous, current in zip(result, result[1:])):
        raise RawDataError(f"{label} timestamps must strictly advance")
    return result


def load_episode(path: Path | str) -> RawEpisode:
    path = Path(path)
    manifest = _object(path / "manifest.json")
    if manifest.get("schema_version") != 7 or manifest.get("episode_id") != path.name:
        raise RawDataError("only schema_version 7 RGB/state/action episodes are accepted")
    if manifest.get("outcome") != "success":
        raise RawDataError("only successful episodes enter training")
    if manifest.get("operator_confirmed_success") is not True:
        raise RawDataError("operator-confirmed success is required")
    if manifest.get("recording_profile") != "rgb_state_action_v1":
        raise RawDataError("recording profile must be rgb_state_action_v1")
    if manifest.get("provenance") != "native_rgb_action_v7":
        raise RawDataError("episode provenance must be native_rgb_action_v7")
    if manifest.get("control_period_ms") != 150:
        raise RawDataError("control period must be 150 ms")
    expected_timebase = {
        "clock": "monotonic",
        "rgb_timestamp": "ros_header_projected_to_monotonic",
        "state_timestamp": "service_request_response_midpoint",
        "action_timestamp": "ros_subscription_arrival",
        "alignment": "rgb_anchor_nearest_state_latest_prior_action",
    }
    if manifest.get("timebase") != expected_timebase:
        raise RawDataError("timebase contract is invalid")
    if manifest.get("source") not in {"manual", "expert"}:
        raise RawDataError("invalid recording source")
    if tuple(manifest.get("servo_ids", ())) != SERVO_IDS:
        raise RawDataError("servo order must be [1,2,3,4,5,10]")
    task = manifest.get("task")
    if not isinstance(task, str) or not task.strip():
        raise RawDataError("task must be non-empty")
    image_shape = tuple(manifest.get("image_shape", ()))
    if len(image_shape) != 3 or image_shape[2] != 3:
        raise RawDataError("RGB image shape must be HWC3")
    if any(str(key).startswith("depth") for key in manifest):
        raise RawDataError("depth fields are not part of schema_version 7")
    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict) or not checksums:
        raise RawDataError("checksums are required")
    for required in ("frames.jsonl", "states.jsonl", "actions.jsonl"):
        if required not in checksums:
            raise RawDataError(f"missing stream: {required}")
    for relative, expected in checksums.items():
        source = _file(path, relative)
        if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise RawDataError(f"checksum mismatch: {relative}")
    expert_target = None
    if manifest["source"] == "expert":
        try:
            expert_target = parse_expert_target(path, manifest, checksums)
        except (TypeError, ValueError) as error:
            raise RawDataError(str(error)) from error
    elif "expert" in manifest:
        raise RawDataError("manual episode cannot contain expert metadata")
    states = _events(path / "states.jsonl", STATE_LIMITS, "state")
    actions = _events(path / "actions.jsonl", ACTION_LIMITS, "action")
    stream_counts = manifest.get("stream_counts")
    if not isinstance(stream_counts, dict) or stream_counts != {
        "rgb": manifest.get("frame_count"),
        "state": len(states),
        "action": len(actions),
        "aligned": manifest.get("frame_count"),
    }:
        raise RawDataError("stream counts do not match recorded streams")
    timing = manifest.get("timing")
    if not isinstance(timing, dict) or set(timing) != {
        "image_gap",
        "image_transport_latency",
        "state_skew",
        "state_round_trip",
        "action_age",
    }:
        raise RawDataError("timing summary is missing or invalid")
    state_times = [event[0] for event in states]
    action_times = [event[0] for event in actions]
    try:
        lines = (path / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise RawDataError(f"cannot read frames: {error}") from error
    if not lines or len(lines) != manifest.get("frame_count"):
        raise RawDataError("frame_count does not match frames.jsonl")
    samples = []
    previous = None
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise RawDataError(f"invalid frame JSON: {error}") from error
        timestamp = _timestamp(row.get("rgb_timestamp_ns"), "rgb timestamp")
        if previous is not None and timestamp <= previous:
            raise RawDataError("RGB timestamps must strictly advance")
        state_time = _timestamp(row.get("state_timestamp_ns"), "state timestamp")
        if abs(timestamp - state_time) > MAX_STATE_SKEW_NS or state_time not in state_times:
            raise RawDataError("RGB/state alignment is invalid")
        state = states[bisect_left(state_times, state_time)][1]
        if tuple(row.get("state", ())) != state:
            raise RawDataError("paired state differs from raw state event")
        if row.get("rgb") not in checksums:
            raise RawDataError("RGB file is absent from checksums")
        image_path = _file(path, row.get("rgb"))
        rgb = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if rgb is None or tuple(rgb.shape) != image_shape:
            raise RawDataError("RGB image is unreadable or has wrong shape")
        if any(str(key).startswith("depth") for key in row):
            raise RawDataError("depth fields are not part of schema_version 7")
        action_duration_s = row.get("action_duration_s")
        if (
            isinstance(action_duration_s, bool)
            or not isinstance(action_duration_s, Real)
            or not math.isfinite(float(action_duration_s))
            or float(action_duration_s) < 0
        ):
            raise RawDataError("paired action duration is missing")
        arrival_ns = _timestamp(
            row.get("image_arrival_timestamp_ns"), "image arrival timestamp"
        )
        if arrival_ns < timestamp:
            raise RawDataError("image arrival precedes the projected source timestamp")
        request_ns = _timestamp(
            row.get("state_request_timestamp_ns"), "paired state request timestamp"
        )
        response_ns = _timestamp(
            row.get("state_response_timestamp_ns"), "paired state response timestamp"
        )
        if response_ns < request_ns:
            raise RawDataError("paired state response precedes request")
        if state_time != request_ns + (response_ns - request_ns) // 2:
            raise RawDataError("paired state timing is invalid")
        paired_action_time = _timestamp(
            row.get("action_timestamp_ns"), "paired action timestamp"
        )
        if paired_action_time > timestamp or paired_action_time not in action_times:
            raise RawDataError("paired action is not a causal raw action event")
        paired_action = actions[bisect_left(action_times, paired_action_time)][1]
        if _vector(row.get("action"), ACTION_LIMITS, "paired action") != paired_action:
            raise RawDataError("paired action differs from raw action event")
        future = bisect_left(action_times, timestamp)
        if future < len(actions) and action_times[future] - timestamp <= MAX_ACTION_LOOKAHEAD_NS:
            action_time, action = actions[future]
        else:
            action_time = paired_action_time
            if action_time > timestamp or action_time not in action_times:
                raise RawDataError("no valid action target or causal hold")
            action = actions[bisect_left(action_times, action_time)][1]
        samples.append(RawSampleEntry(timestamp, timestamp, state_time, action_time,
                                      image_path, state, action))
        previous = timestamp
    return RawEpisode(
        path=path,
        episode_index=0,
        task=task.strip(),
        image_shape=image_shape,
        samples=tuple(samples),
        state_events=tuple(states),
        action_events=tuple(actions),
        source=manifest["source"],
        provenance=str(manifest["provenance"]),
        expert_target=expert_target,
    )


def load_session(source_root: Path | str) -> tuple[list[RawEpisode], dict[str, str]]:
    episodes_root = Path(source_root) / "episodes"
    if not episodes_root.is_dir():
        raise RawDataError("source has no episodes directory")
    episodes: list[RawEpisode] = []
    rejected: dict[str, str] = {}
    image_shape = None
    for path in sorted(episodes_root.glob("episode_*")):
        try:
            episode = load_episode(path)
            if image_shape is None:
                image_shape = episode.image_shape
            elif episode.image_shape != image_shape:
                raise RawDataError("mixed RGB shapes are not accepted")
        except RawDataError as error:
            rejected[path.name] = str(error)
            continue
        episodes.append(replace(episode, episode_index=len(episodes)))
    if not episodes:
        raise RawDataError("source contains no valid v7 RGB/state/action episodes")
    return episodes, rejected


def resample_episode(episode: RawEpisode, period_ns: int, tolerance_ns: int) -> list[ConvertedFrame]:
    if period_ns <= 0 or tolerance_ns < 0:
        raise RawDataError("period must be positive and tolerance non-negative")
    start = episode.samples[0].timestamp_ns
    end = episode.samples[-1].timestamp_ns
    state_times = [event[0] for event in episode.state_events]
    action_times = [event[0] for event in episode.action_events]
    frames = []
    for timestamp in range(start, end + 1, period_ns):
        observation = min(episode.samples, key=lambda sample: abs(sample.timestamp_ns - timestamp))
        if abs(observation.timestamp_ns - timestamp) > tolerance_ns:
            raise RawDataError(f"no RGB observation within tolerance at {timestamp}")
        state_position = bisect_left(state_times, timestamp)
        state_candidates = []
        if state_position:
            state_candidates.append(episode.state_events[state_position - 1])
        if state_position < len(episode.state_events):
            state_candidates.append(episode.state_events[state_position])
        state_time, state = min(
            state_candidates, key=lambda event: abs(event[0] - timestamp)
        )
        if abs(state_time - timestamp) > MAX_STATE_SKEW_NS:
            raise RawDataError(f"no state observation within tolerance at {timestamp}")

        action_position = bisect_left(action_times, timestamp)
        if (
            action_position < len(episode.action_events)
            and action_times[action_position] - timestamp <= MAX_ACTION_LOOKAHEAD_NS
        ):
            _action_time, action = episode.action_events[action_position]
        elif action_position:
            _action_time, action = episode.action_events[action_position - 1]
        else:
            raise RawDataError(f"no action target near timestamp {timestamp}")
        frames.append(
            ConvertedFrame(
                timestamp,
                observation.image_path,
                state,
                action,
                observation.timestamp_ns,
            )
        )
    return frames
