from bisect import bisect_left
from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from action import (
    ActionCodecConfig,
    ActionCodecError,
    DynamicBoundedRelativeCodec,
)

from .quality import write_episode_quality, write_expert_targets
from .raw import (
    ConvertedFrame,
    RawDataError,
    RawEpisode,
    SERVO_IDS,
    load_session,
    resample_episode,
)


ROBOT_TYPE = "hiwonder_rosorin_pro_arm"
CAMERA_KEY = "observation.images.arm_camera"
# Recorder accepts an RGB gap of up to 1.2 s.  Nearest-neighbour
# resampling therefore needs half that interval to cover every grid point.
RESAMPLE_TOLERANCE_NS = 600_000_000


@dataclass(frozen=True)
class TerminalHoldReport:
    original_rows: int
    event_extension_rows: int
    hold_rows: int
    output_rows: int


def _nearest_event(events, timestamp_ns: int):
    times = [event[0] for event in events]
    position = bisect_left(times, timestamp_ns)
    candidates = []
    if position:
        candidates.append(events[position - 1])
    if position < len(events):
        candidates.append(events[position])
    return min(candidates, key=lambda event: abs(event[0] - timestamp_ns))


def _action_at_or_after(events, timestamp_ns: int):
    times = [event[0] for event in events]
    position = bisect_left(times, timestamp_ns)
    return events[min(position, len(events) - 1)]


def extend_confirmed_terminal_hold(
    episode: RawEpisode,
    frames: list[ConvertedFrame],
    period_ns: int,
    hold_rows: int,
) -> tuple[list[ConvertedFrame], TerminalHoldReport]:
    if period_ns <= 0:
        raise RawDataError("period_ns must be positive")
    if isinstance(hold_rows, bool) or not isinstance(hold_rows, int) or hold_rows <= 0:
        raise RawDataError("hold_rows must be a positive integer")
    if episode.source != "expert":
        raise RawDataError("terminal hold augmentation requires confirmed expert data")
    if not frames:
        raise RawDataError("terminal hold augmentation requires converted frames")
    final_state = episode.state_events[-1][1]
    final_action = episode.action_events[-1][1]
    if len(final_state) != 6 or len(final_action) != 6:
        raise RawDataError("terminal state and action must contain six values")
    if not np.isfinite(final_state).all() or not np.isfinite(final_action).all():
        raise RawDataError("terminal state and action must be finite")
    if not np.isclose(final_action[5], 520.0, atol=0.5, rtol=0.0):
        raise RawDataError(
            f"confirmed terminal gripper action must be 520 pulse, got {final_action[5]:g}"
        )

    result = list(frames)
    original_rows = len(result)
    last_image = episode.samples[-1]
    terminal_event_ns = max(
        episode.action_events[-1][0], episode.state_events[-1][0]
    )
    timestamp_ns = result[-1].timestamp_ns + period_ns
    while timestamp_ns <= terminal_event_ns:
        _state_ns, state = _nearest_event(episode.state_events, timestamp_ns)
        _action_ns, action = _action_at_or_after(episode.action_events, timestamp_ns)
        result.append(
            ConvertedFrame(
                timestamp_ns,
                last_image.image_path,
                state,
                action,
                last_image.timestamp_ns,
            )
        )
        timestamp_ns += period_ns
    event_extension_rows = len(result) - original_rows
    for _ in range(hold_rows):
        result.append(
            ConvertedFrame(
                timestamp_ns,
                last_image.image_path,
                final_state,
                final_action,
                last_image.timestamp_ns,
            )
        )
        timestamp_ns += period_ns

    try:
        DynamicBoundedRelativeCodec(ActionCodecConfig.default_v1()).encode_episode(
            np.asarray(result[0].state, dtype=np.float32),
            np.asarray([frame.action for frame in result], dtype=np.float32),
        )
    except ActionCodecError as error:
        raise RawDataError(f"terminal hold violates the action codec: {error}") from error
    return result, TerminalHoldReport(
        original_rows=original_rows,
        event_extension_rows=event_extension_rows,
        hold_rows=hold_rows,
        output_rows=len(result),
    )


@dataclass(frozen=True)
class ConversionReport:
    repo_id: str
    fps: float
    control_period_ms: int
    accepted_episodes: int
    rejected_episodes: int
    total_frames: int
    source_episodes: list[str]
    train_episode_indices: list[int]
    validation_episode_indices: list[int]
    rejection_reasons: dict[str, str]
    dataset_reloaded: bool
    feature_keys: list[str]
    unique_episodes: int
    source_episode_counts: dict[str, int]
    training_source_episode_counts: dict[str, int]
    action_representation: str = "absolute_pulse"
    projected_action_rows: int = 0
    projected_by_servo: tuple[int, ...] = (0, 0, 0, 0, 0, 0)
    latent_min: float | None = None
    latent_max: float | None = None
    codec_sha256: str | None = None
    raw_schema_version: int = 7
    input_profile: str = "rgb_state_v1"
    expert_episodes: int = 0
    rgbd_localization_episodes: int = 0
    derived_expert_close_episodes: int = 0
    expert_target_coverage: float = 1.0
    terminal_hold_rows: int = 0
    terminal_augmented_rows: int = 0
    terminal_reports: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _split(indices: list[int]) -> tuple[list[int], list[int]]:
    if len(indices) == 1:
        return indices, []
    validation_count = max(1, round(len(indices) * 0.10))
    return indices[:-validation_count], indices[-validation_count:]


def convert_session(
    source_root: Path,
    output_root: Path,
    repo_id: str,
    control_period_ms: int,
    use_videos: bool,
    action_codec_config: Path | None = None,
) -> ConversionReport:
    return convert_sessions(
        source_roots=[source_root],
        output_root=output_root,
        repo_id=repo_id,
        control_period_ms=control_period_ms,
        use_videos=use_videos,
        action_codec_config=action_codec_config,
    )


def convert_sessions(
    source_roots: list[Path],
    output_root: Path,
    repo_id: str,
    control_period_ms: int,
    use_videos: bool,
    task_override: str | None = None,
    action_codec_config: Path | None = None,
    validation_episode_ids: set[str] | None = None,
    source_repeats: dict[str, int] | None = None,
    terminal_hold_rows: int = 0,
) -> ConversionReport:
    source_roots = [Path(root) for root in source_roots]
    output_root = Path(output_root)
    if not source_roots:
        raise RawDataError("at least one source session is required")
    if isinstance(control_period_ms, bool) or not isinstance(control_period_ms, int) or control_period_ms <= 0:
        raise RawDataError("control_period_ms must be a positive integer")
    fps = 1000.0 / control_period_ms
    period_ns = control_period_ms * 1_000_000
    if "/" not in repo_id or repo_id.startswith("/") or repo_id.endswith("/"):
        raise RawDataError("repo_id must have owner/name form")
    if output_root.exists():
        raise RawDataError("output_root already exists")
    if task_override is not None:
        task_override = task_override.strip()
        if not task_override:
            raise RawDataError("task_override must be non-empty when provided")

    codec = None
    codec_config_path = None
    codec_sha256 = None
    if action_codec_config is not None:
        codec_config_path = Path(action_codec_config)
        try:
            codec = DynamicBoundedRelativeCodec(ActionCodecConfig.load(codec_config_path))
        except ActionCodecError as error:
            raise RawDataError(f"invalid action codec config: {error}") from error
        codec_sha256 = hashlib.sha256(codec_config_path.read_bytes()).hexdigest()

    unique_episodes = []
    rejected = {}
    source_labels = {}
    image_shape = None
    for source_number, source_root in enumerate(source_roots):
        session_episodes, session_rejected = load_session(source_root)
        for episode_name, reason in session_rejected.items():
            rejected[f"{source_number}:{source_root.name}/{episode_name}"] = reason
        for episode in session_episodes:
            if image_shape is None:
                image_shape = episode.image_shape
            elif episode.image_shape != image_shape:
                raise RawDataError("mixed image shapes across sessions are not accepted")
            if episode.path.name in source_labels:
                raise RawDataError(f"duplicate episode id: {episode.path.name}")
            source_labels[episode.path.name] = f"{source_root.name}/{episode.path.name}"
            unique_episodes.append(
                replace(
                    episode,
                    episode_index=len(unique_episodes),
                    task=task_override if task_override is not None else episode.task,
                )
            )

    unique_ids = {episode.path.name for episode in unique_episodes}
    if validation_episode_ids is None:
        train_positions, validation_positions = _split(list(range(len(unique_episodes))))
        validation_ids = {unique_episodes[index].path.name for index in validation_positions}
    else:
        validation_ids = {str(value) for value in validation_episode_ids}
        unknown = validation_ids - unique_ids
        if unknown:
            raise RawDataError(f"unknown validation episodes: {', '.join(sorted(unknown))}")
        if not validation_ids:
            raise RawDataError("at least one validation episode is required")
        train_positions = [
            index for index, episode in enumerate(unique_episodes)
            if episode.path.name not in validation_ids
        ]
        validation_positions = [
            index for index, episode in enumerate(unique_episodes)
            if episode.path.name in validation_ids
        ]
    if not train_positions:
        raise RawDataError("at least one training episode is required")
    repeats = {str(key): value for key, value in (source_repeats or {}).items()}
    for source, repeat_count in repeats.items():
        if isinstance(repeat_count, bool) or not isinstance(repeat_count, int) or repeat_count < 1:
            raise RawDataError(f"source repeat for {source!r} must be a positive integer")

    episodes = []
    source_episodes = []
    train_indices = []
    validation_indices = []
    source_episode_counts = {}
    training_source_episode_counts = {}
    train_position_set = set(train_positions)
    for position, episode in enumerate(unique_episodes):
        source_episode_counts[episode.source] = source_episode_counts.get(episode.source, 0) + 1
        converted_episode = replace(episode, episode_index=len(episodes))
        episodes.append(converted_episode)
        source_episodes.append(source_labels[episode.path.name])
        if position in train_position_set:
            train_indices.append(converted_episode.episode_index)
            training_source_episode_counts[episode.source] = (
                training_source_episode_counts.get(episode.source, 0) + 1
            )
        else:
            validation_indices.append(converted_episode.episode_index)
    for position in train_positions:
        episode = unique_episodes[position]
        for repeat_number in range(1, repeats.get(episode.source, 1)):
            converted_episode = replace(episode, episode_index=len(episodes))
            episodes.append(converted_episode)
            source_episodes.append(
                f"{source_labels[episode.path.name]}#repeat{repeat_number}"
            )
            train_indices.append(converted_episode.episode_index)
            training_source_episode_counts[episode.source] = (
                training_source_episode_counts.get(episode.source, 0) + 1
            )

    if (
        isinstance(terminal_hold_rows, bool)
        or not isinstance(terminal_hold_rows, int)
        or terminal_hold_rows < 0
    ):
        raise RawDataError("terminal_hold_rows must be a non-negative integer")
    converted = {}
    terminal_reports = {}
    for episode in episodes:
        frames = resample_episode(
            episode, period_ns=period_ns, tolerance_ns=RESAMPLE_TOLERANCE_NS
        )
        if terminal_hold_rows:
            frames, terminal_report = extend_confirmed_terminal_hold(
                episode, frames, period_ns, terminal_hold_rows
            )
            terminal_reports[str(episode.episode_index)] = asdict(terminal_report)
        converted[episode.episode_index] = frames
    encoded_actions = {}
    projected_action_rows = 0
    projected_by_servo = np.zeros(len(SERVO_IDS), dtype=np.int64)
    latent_min = None
    latent_max = None
    if codec is not None:
        all_latent = []
        for episode in episodes:
            frames = converted[episode.episode_index]
            try:
                latent, encoding_report = codec.encode_episode(
                    np.asarray(frames[0].state, dtype=np.float32),
                    np.asarray([frame.action for frame in frames], dtype=np.float32),
                )
            except ActionCodecError as error:
                raise RawDataError(
                    f"cannot encode {episode.path.name}: {error}"
                ) from error
            encoded_actions[episode.episode_index] = latent
            projected_action_rows += encoding_report.projected_rows
            projected_by_servo += np.asarray(
                encoding_report.projected_by_servo, dtype=np.int64
            )
            all_latent.append(latent)
        flattened = np.concatenate(all_latent, axis=0)
        latent_min = float(flattened.min())
        latent_max = float(flattened.max())
    image_shape = episodes[0].image_shape
    names = [f"servo_{servo_id}" for servo_id in SERVO_IDS]
    features = {
        "observation.state": {"dtype": "float32", "shape": (6,), "names": names},
        "action": {"dtype": "float32", "shape": (6,), "names": names},
        CAMERA_KEY: {
            "dtype": "video" if use_videos else "image",
            "shape": image_shape,
            "names": ["height", "width", "channels"],
        },
    }
    dataset = None
    try:
        dataset = LeRobotDataset.create(
            repo_id,
            fps,
            root=output_root,
            robot_type=ROBOT_TYPE,
            features=features,
            use_videos=use_videos,
        )
        for episode in episodes:
            for frame_index, frame in enumerate(converted[episode.episode_index]):
                image_bgr = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
                if image_bgr is None:
                    raise RawDataError(f"image became unreadable during conversion: {frame.image_path}")
                dataset.add_frame(
                    {
                        CAMERA_KEY: cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
                        "observation.state": np.asarray(frame.state, dtype=np.float32),
                        "action": (
                            encoded_actions[episode.episode_index][frame_index]
                            if codec is not None
                            else np.asarray(frame.action, dtype=np.float32)
                        ),
                        "task": episode.task,
                    }
                )
            dataset.save_episode()
        dataset.finalize()
        dataset = None
        reloaded = LeRobotDataset(repo_id, root=output_root)
        if reloaded.meta.total_episodes != len(episodes):
            raise RawDataError("reloaded episode count does not match conversion")
        total_frames = reloaded.meta.total_frames
        feature_keys = sorted(reloaded.features)
        if codec_config_path is not None:
            shutil.copyfile(codec_config_path, output_root / "action_codec.json")
        quality_root = output_root / "quality"
        for episode in episodes:
            write_episode_quality(quality_root, episode, converted[episode.episode_index], fps)
        write_expert_targets(quality_root, unique_episodes)
        expert_episodes = [episode for episode in unique_episodes if episode.source == "expert"]
        target_episodes = [episode for episode in expert_episodes if episode.expert_target is not None]
        report = ConversionReport(
            repo_id=repo_id,
            fps=fps,
            control_period_ms=control_period_ms,
            accepted_episodes=len(episodes),
            rejected_episodes=len(rejected),
            total_frames=total_frames,
            source_episodes=source_episodes,
            train_episode_indices=train_indices,
            validation_episode_indices=validation_indices,
            rejection_reasons=rejected,
            dataset_reloaded=True,
            feature_keys=feature_keys,
            unique_episodes=len(unique_episodes),
            source_episode_counts=source_episode_counts,
            training_source_episode_counts=training_source_episode_counts,
            action_representation=(
                codec.config.schema if codec is not None else "absolute_pulse"
            ),
            projected_action_rows=projected_action_rows,
            projected_by_servo=tuple(int(value) for value in projected_by_servo),
            latent_min=latent_min,
            latent_max=latent_max,
            codec_sha256=codec_sha256,
            raw_schema_version=7,
            expert_episodes=len(expert_episodes),
            rgbd_localization_episodes=sum(
                episode.expert_target is not None
                and episode.expert_target.target_source == "rgbd_localization"
                for episode in expert_episodes
            ),
            derived_expert_close_episodes=sum(
                episode.expert_target is not None
                and episode.expert_target.target_source == "derived_expert_close"
                for episode in expert_episodes
            ),
            expert_target_coverage=(
                len(target_episodes) / len(expert_episodes) if expert_episodes else 1.0
            ),
            terminal_hold_rows=terminal_hold_rows,
            terminal_augmented_rows=sum(
                report["event_extension_rows"] + report["hold_rows"]
                for report in terminal_reports.values()
            ),
            terminal_reports=terminal_reports,
        )
        (output_root / "quality-report.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return report
    except Exception:
        if dataset is not None:
            try:
                dataset.finalize()
            except Exception:
                pass
        if output_root.exists():
            failed_root = output_root.with_name(output_root.name + ".failed")
            if not failed_root.exists():
                shutil.move(str(output_root), str(failed_root))
        raise
