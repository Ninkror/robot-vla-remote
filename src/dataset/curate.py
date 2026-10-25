from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import shutil
import sys
import tarfile

import numpy as np
import pyarrow as pa

from common.config import load_json
from common.paths import resolve_path
from common.paths import DATA
from dataset.main import _extract_archive
from dataset.raw import RawDataError, load_episode


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _timing_stats_ns(values: list[int]) -> dict:
    samples = np.asarray(values, dtype=np.float64)
    if not samples.size:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
    return {
        "count": int(samples.size),
        "p50_ms": float(np.percentile(samples, 50) / 1_000_000.0),
        "p95_ms": float(np.percentile(samples, 95) / 1_000_000.0),
        "max_ms": float(np.max(samples) / 1_000_000.0),
    }


def _timing_summary(
    frames: list[dict], states: list[dict], actions: list[dict]
) -> dict:
    del actions  # Raw action timing is represented by the frame-to-action age below.
    image_times = [int(row["rgb_timestamp_ns"]) for row in frames]
    return {
        "image_gap": _timing_stats_ns(
            [right - left for left, right in zip(image_times, image_times[1:])]
        ),
        "image_transport_latency": _timing_stats_ns([
            max(0, int(row["image_arrival_timestamp_ns"]) - int(row["rgb_timestamp_ns"]))
            for row in frames
        ]),
        "state_skew": _timing_stats_ns([
            abs(int(row["rgb_timestamp_ns"]) - int(row["state_timestamp_ns"]))
            for row in frames
        ]),
        "state_round_trip": _timing_stats_ns([
            int(row["response_timestamp_ns"]) - int(row["request_timestamp_ns"])
            for row in states
        ]),
        "action_age": _timing_stats_ns([
            max(0, int(row["rgb_timestamp_ns"]) - int(row["action_timestamp_ns"]))
            for row in frames
        ]),
    }


def _verified_manifest(episode: Path) -> dict:
    manifest_path = episode / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 7 or manifest.get("episode_id") != episode.name:
        raise RawDataError(f"invalid v7 episode manifest: {episode.name}")
    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict) or not checksums:
        raise RawDataError(f"episode has no checksums: {episode.name}")
    for relative, expected in checksums.items():
        target = (episode / relative).resolve()
        if not target.is_relative_to(episode.resolve()) or not target.is_file() or target.is_symlink():
            raise RawDataError(f"unsafe episode entry: {episode.name}/{relative}")
        if _sha256(target) != expected:
            raise RawDataError(f"checksum mismatch: {episode.name}/{relative}")
    return manifest


def trim_episode(
    source: Path,
    destination: Path,
    *,
    settle_ns: int,
    gripper_delta: float,
) -> dict:
    if settle_ns < 0 or gripper_delta <= 0:
        raise ValueError("settle_ns must be non-negative and gripper_delta positive")
    manifest = _verified_manifest(source)
    load_episode(source)
    if manifest.get("source") != "manual":
        raise RawDataError(f"only manual episodes may be trimmed: {source.name}")
    actions = _jsonl(source / "actions.jsonl")
    states = _jsonl(source / "states.jsonl")
    frames = _jsonl(source / "frames.jsonl")
    changes = [
        index
        for index in range(1, len(actions))
        if abs(float(actions[index]["values"][5]) - float(actions[index - 1]["values"][5]))
        >= gripper_delta
    ]
    if not changes:
        raise RawDataError(f"no final gripper close transition: {source.name}")
    close_index = changes[-1]
    close_value = float(actions[close_index]["values"][5])
    prior_minimum = min(float(row["values"][5]) for row in actions[: close_index + 1])
    if close_value - prior_minimum < 40.0:
        raise RawDataError(f"final gripper transition is not a close: {source.name}")
    close_ns = int(actions[close_index]["timestamp_ns"])
    cutoff_ns = close_ns + settle_ns
    kept_frames = [row for row in frames if int(row["rgb_timestamp_ns"]) <= cutoff_ns]
    if not kept_frames:
        raise RawDataError(f"trim removed every RGB frame: {source.name}")
    referenced_state_times = {int(row["state_timestamp_ns"]) for row in kept_frames}
    referenced_action_times = {int(row["action_timestamp_ns"]) for row in kept_frames}
    kept_states = [
        row
        for row in states
        if int(row["timestamp_ns"]) <= cutoff_ns
        or int(row["timestamp_ns"]) in referenced_state_times
    ]
    kept_actions = [
        row
        for row in actions
        if int(row["timestamp_ns"]) <= cutoff_ns
        or int(row["timestamp_ns"]) in referenced_action_times
    ]
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "rgb").mkdir()
    copied_images = set()
    for row in kept_frames:
        relative = PurePosixPath(str(row["rgb"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RawDataError(f"unsafe RGB path: {row['rgb']}")
        if relative.as_posix() in copied_images:
            continue
        copied_images.add(relative.as_posix())
        target = destination / relative.as_posix()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative.as_posix(), target)
    _write_jsonl(destination / "frames.jsonl", kept_frames)
    _write_jsonl(destination / "states.jsonl", kept_states)
    _write_jsonl(destination / "actions.jsonl", kept_actions)

    image_times = [int(row["rgb_timestamp_ns"]) for row in kept_frames]
    duration_s = (image_times[-1] - image_times[0]) / 1_000_000_000.0
    manifest["frame_count"] = len(kept_frames)
    manifest["duration_s"] = duration_s
    manifest["effective_fps"] = (
        (len(kept_frames) - 1) / duration_s if duration_s > 0 and len(kept_frames) > 1 else 0.0
    )
    manifest["max_image_gap_ns"] = max(
        (right - left for left, right in zip(image_times, image_times[1:])),
        default=0,
    )
    manifest["max_state_skew_ns"] = max(
        abs(int(row["rgb_timestamp_ns"]) - int(row["state_timestamp_ns"]))
        for row in kept_frames
    )
    manifest["provenance"] = "native_rgb_action_v7"
    manifest["stream_counts"] = {
        "rgb": len(kept_frames),
        "state": len(kept_states),
        "action": len(kept_actions),
        "aligned": len(kept_frames),
    }
    manifest["timing"] = _timing_summary(kept_frames, kept_states, kept_actions)
    manifest["curation"] = {
        "method": "final_gripper_close_plus_settle",
        "close_timestamp_ns": close_ns,
        "cutoff_timestamp_ns": cutoff_ns,
        "settle_ms": settle_ns // 1_000_000,
        "gripper_delta": float(gripper_delta),
        "original_frame_count": len(frames),
    }
    checksums = {
        path.relative_to(destination).as_posix(): _sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file()
    }
    manifest["checksums"] = checksums
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "episode_id": source.name,
        "original_frames": len(frames),
        "kept_frames": len(kept_frames),
        "close_ns": close_ns,
        "cutoff_ns": cutoff_ns,
        "kept_duration_s": duration_s,
    }


def _tar_info(name: str, size: int, *, directory: bool = False) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name + ("/" if directory and not name.endswith("/") else ""))
    info.type = tarfile.DIRTYPE if directory else tarfile.REGTYPE
    info.size = 0 if directory else size
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    info.mode = 0o755 if directory else 0o644
    return info


def _episode_files(episode: Path) -> list[Path]:
    manifest = _verified_manifest(episode)
    return [episode / "manifest.json"] + [
        episode / relative for relative in sorted(manifest["checksums"])
    ]


def _build_archive(episodes: list[Path], output: Path, archive_id: str) -> str:
    if output.exists():
        raise RawDataError(f"curated archive already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    partial.unlink(missing_ok=True)
    episode_ids = [episode.name for episode in sorted(episodes, key=lambda item: item.name)]
    archive_manifest = {
        "schema_version": 2,
        "archive_id": archive_id,
        "episodes": episode_ids,
    }
    manifest_bytes = (
        json.dumps(archive_manifest, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    try:
        with pa.CompressedOutputStream(str(partial), "zstd") as writer:
            with tarfile.open(fileobj=writer, mode="w|") as archive:
                archive.addfile(_tar_info("manifest.json", len(manifest_bytes)), io.BytesIO(manifest_bytes))
                for episode in sorted(episodes, key=lambda item: item.name):
                    prefix = f"episodes/{episode.name}"
                    archive.addfile(_tar_info(prefix, 0, directory=True))
                    for source in _episode_files(episode):
                        relative = source.relative_to(episode).as_posix()
                        with source.open("rb") as stream:
                            archive.addfile(
                                _tar_info(f"{prefix}/{relative}", source.stat().st_size),
                                stream,
                            )
        os.replace(partial, output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return _sha256(output)


def curate(config_path: Path) -> int:
    config = load_json(config_path)
    name = str(config["name"])
    manual_archive = resolve_path(config["manual_archive"])
    expert_archive = resolve_path(config["expert_archive"])
    output_archive = resolve_path(config["output_archive"])
    for label, archive, expected in (
        ("manual", manual_archive, str(config["manual_sha256"])),
        ("expert", expert_archive, str(config["expert_sha256"])),
    ):
        if not archive.is_file() or _sha256(archive) != expected:
            raise RawDataError(f"{label} source archive is missing or has wrong sha256")
    workspace = DATA / "temp" / f"{name}-curation"
    if workspace.exists():
        raise RawDataError(f"curation workspace already exists: {workspace}")
    workspace.mkdir(parents=True)
    manual_manifest = _extract_archive(manual_archive, workspace / "manual")
    expert_manifest = _extract_archive(expert_archive, workspace / "expert")
    selected_root = workspace / "selected"
    selected_root.mkdir()
    excluded = set(str(value) for value in config.get("exclude_manual_episodes", []))
    settle_ns = int(config["settle_ms"]) * 1_000_000
    gripper_delta = float(config.get("gripper_delta", 5.0))
    trimmed = []
    episodes = []
    for episode_id in manual_manifest["episodes"]:
        if episode_id in excluded:
            continue
        source = workspace / "manual" / "episodes" / episode_id
        destination = selected_root / episode_id
        trimmed.append(
            trim_episode(
                source,
                destination,
                settle_ns=settle_ns,
                gripper_delta=gripper_delta,
            )
        )
        episodes.append(destination)
    for episode_id in expert_manifest["episodes"]:
        source = workspace / "expert" / "episodes" / episode_id
        destination = selected_root / episode_id
        if destination.exists():
            raise RawDataError(f"duplicate episode id across sources: {episode_id}")
        _verified_manifest(source)
        shutil.copytree(source, destination)
        episodes.append(destination)
    archive_sha256 = _build_archive(episodes, output_archive, str(config["archive_id"]))
    report = {
        "schema_version": 1,
        "name": name,
        "output_archive": str(output_archive),
        "output_sha256": archive_sha256,
        "manual_source_episodes": len(manual_manifest["episodes"]),
        "manual_excluded_episodes": sorted(excluded),
        "manual_trimmed_episodes": len(trimmed),
        "manual_original_frames": sum(row["original_frames"] for row in trimmed),
        "manual_kept_frames": sum(row["kept_frames"] for row in trimmed),
        "expert_episodes": len(expert_manifest["episodes"]),
        "combined_unique_episodes": len(episodes),
        "trimmed": trimmed,
    }
    report_path = output_archive.with_name(output_archive.name + ".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    shutil.rmtree(workspace)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"CURATED_ARCHIVE_READY path={output_archive} sha256={archive_sha256}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Trim manual grasps and combine them with expert episodes")
    result.add_argument("--config", type=Path, required=True)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return curate(args.config)
    except (KeyError, OSError, RawDataError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
