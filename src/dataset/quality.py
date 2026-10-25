from pathlib import Path
import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .raw import ConvertedFrame, RawEpisode, SERVO_IDS


def write_expert_targets(output_dir: Path, episodes: list[RawEpisode]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "expert-targets.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for episode in episodes:
            target = episode.expert_target
            if target is None:
                continue
            row = {
                "episode_index": episode.episode_index,
                "episode_id": episode.path.name,
                "task": episode.task,
                "target_source": target.target_source,
                "anchor": {
                    "rgb_file": target.anchor_image_path.relative_to(episode.path).as_posix(),
                    "rgb_timestamp_ns": target.anchor_rgb_timestamp_ns,
                    "state_timestamp_ns": target.anchor_state_timestamp_ns,
                    "state": list(target.anchor_state),
                    "pixel_xy": None if target.pixel_xy is None else list(target.pixel_xy),
                    "camera_xyz_m": None if target.camera_xyz_m is None else list(target.camera_xyz_m),
                    "detected_object_base_xyz_m": (
                        None
                        if target.detected_object_base_xyz_m is None
                        else list(target.detected_object_base_xyz_m)
                    ),
                },
                "grasp_target_base_xyz_m": list(target.grasp_target_base_xyz_m),
                "ik_close_target": list(target.ik_targets[-1]),
                "calibration_status": target.calibration_status,
                "calibration_fingerprint_sha256": target.calibration_fingerprint_sha256,
                "close_event": {
                    "command_timestamp_ns": target.close_command_timestamp_ns,
                    "completion_timestamp_ns": target.close_completion_timestamp_ns,
                    "frame_index": target.close_frame_index,
                    "action": list(target.close_action),
                    "state": list(target.close_state),
                    "state_timestamp_ns": target.close_state_timestamp_ns,
                    "fk_xyz_m": list(target.close_fk_xyz_m),
                },
            }
            stream.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                + "\n"
            )


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"quality image is unreadable: {path}")
    return image


def write_episode_quality(
    output_dir: Path,
    episode: RawEpisode,
    frames: list[ConvertedFrame],
    fps: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"episode_{episode.episode_index:06d}"
    selection = np.linspace(0, len(frames) - 1, num=min(9, len(frames)), dtype=int)
    images = [_read_image(frames[int(index)].image_path) for index in selection]
    height, width = images[0].shape[:2]
    contact = np.full((height * 3, width * 3, 3), 32, dtype=np.uint8)
    for slot, image in enumerate(images):
        row, column = divmod(slot, 3)
        contact[row * height : (row + 1) * height, column * width : (column + 1) * width] = image
    if not cv2.imwrite(str(output_dir / f"{stem}_contact.jpg"), contact):
        raise RuntimeError("contact sheet write failed")

    preview_path = output_dir / f"{stem}_preview.mp4"
    writer = cv2.VideoWriter(
        str(preview_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("preview video writer could not open")
    try:
        for frame in frames:
            writer.write(_read_image(frame.image_path))
    finally:
        writer.release()

    times = np.arange(len(frames), dtype=np.float64) / fps
    states = np.asarray([frame.state for frame in frames], dtype=np.float32)
    actions = np.asarray([frame.action for frame in frames], dtype=np.float32)
    figure, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for index, axis in enumerate(axes.flat):
        axis.plot(times, states[:, index], label="state", linewidth=1.5)
        axis.plot(times, actions[:, index], label="action", linewidth=1.2)
        axis.set_title(f"servo {SERVO_IDS[index]}")
        axis.set_ylabel("pulse")
        axis.grid(True, alpha=0.3)
    axes.flat[0].legend()
    axes[-1, 0].set_xlabel("seconds")
    axes[-1, 1].set_xlabel("seconds")
    figure.tight_layout()
    figure.savefig(output_dir / f"{stem}_signals.png", dpi=120)
    plt.close(figure)

