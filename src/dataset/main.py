import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile

import pyarrow as pa

from action import ActionCodecConfig
from common.config import load_json
from common.paths import resolve_path
from common.paths import CONFIGS, DATA
from .converter import convert_sessions
from .raw import RawDataError


DEFAULT_CONFIG = CONFIGS / "dataset.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member(member: tarfile.TarInfo) -> None:
    name = PurePosixPath(member.name)
    if name.is_absolute() or ".." in name.parts:
        raise RawDataError(f"unsafe archive member: {member.name}")
    if member.issym() or member.islnk() or member.isdev():
        raise RawDataError(f"unsupported archive member: {member.name}")


def _extract_archive(archive: Path, destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=False)
    archive_manifest = None
    with pa.CompressedInputStream(str(archive), "zstd") as reader:
        with tarfile.open(fileobj=reader, mode="r|") as bundle:
            for member in bundle:
                _safe_member(member)
                if member.name == "manifest.json":
                    extracted = bundle.extractfile(member)
                    if extracted is None:
                        raise RawDataError("archive manifest is unreadable")
                    archive_manifest = json.load(extracted)
                    continue
                bundle.extract(member, destination)
    if not isinstance(archive_manifest, dict) or archive_manifest.get("schema_version") != 2:
        raise RawDataError("archive-level schema_version 2 manifest.json is required")
    return archive_manifest


def _prepare_session(config: dict, workspace: Path) -> Path:
    archive = resolve_path(config["source_archive"])
    expected = str(config["source_sha256"])
    if not archive.is_file():
        raise RawDataError(f"source archive is missing: {archive}")
    actual = _sha256(archive)
    if actual != expected:
        raise RawDataError(f"source archive sha256 mismatch: {actual}")

    extracted = workspace / "archive"
    manifest = _extract_archive(archive, extracted)
    episode_ids = manifest.get("episodes")
    if not isinstance(episode_ids, list) or not episode_ids:
        raise RawDataError("archive manifest contains no episodes")

    source_root = extracted / "episodes"
    session_root = workspace / "session"
    episodes_root = session_root / "episodes"
    episodes_root.mkdir(parents=True)
    excluded = set(config.get("exclude_episodes", []))
    selected = []
    for episode_id in episode_ids:
        source = source_root / str(episode_id)
        source_manifest_path = source / "manifest.json"
        if not source_manifest_path.is_file():
            raise RawDataError(f"episode manifest is missing: {episode_id}")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        for name, checksum in source_manifest.get("checksums", {}).items():
            target = source / name
            if not target.is_file() or _sha256(target) != checksum:
                raise RawDataError(f"checksum mismatch: {episode_id}/{name}")
        if episode_id in excluded:
            continue
        if source_manifest.get("schema_version") != 7:
            raise RawDataError(f"episode must use schema_version 7: {episode_id}")
        selected.append((source_manifest.get("completed_at", ""), str(episode_id), source))

    selected.sort()
    for _completed, episode_id, source in selected:
        target = episodes_root / episode_id
        shutil.copytree(source, target)
    if not selected:
        raise RawDataError("all source episodes were excluded")
    return session_root


def prepare(config_path: Path) -> int:
    config = load_json(config_path)
    name = str(config["name"])
    workspace = DATA / "temp" / name
    output = DATA / "datasets" / name
    if workspace.exists():
        raise RawDataError(f"workspace already exists; run data.sh clean first: {workspace}")
    if output.exists():
        raise RawDataError(f"dataset already exists: {output}")
    ActionCodecConfig.load(CONFIGS / "action_codec.json")
    if config.get("input_profile") != "rgb_state_v1":
        raise RawDataError("this training pipeline supports only rgb_state_v1")
    session = _prepare_session(config, workspace)
    report = convert_sessions(
        source_roots=[session],
        output_root=output,
        repo_id=f"local/{name}",
        control_period_ms=int(config["control_period_ms"]),
        use_videos=False,
        task_override=None,
        action_codec_config=CONFIGS / "action_codec.json",
        validation_episode_ids=(
            set(str(value) for value in config["validation_episode_ids"])
            if "validation_episode_ids" in config
            else None
        ),
        source_repeats=config.get("source_repeats"),
        terminal_hold_rows=int(config.get("terminal_hold_rows", 0)),
    )
    report_path = output / "quality-report.json"
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    print(f"DATASET_READY path={output} report={report_path}")
    return 0


def check(config_path: Path) -> int:
    config = load_json(config_path)
    root = DATA / "datasets" / str(config["name"])
    report_path = root / "quality-report.json"
    info_path = root / "meta" / "info.json"
    if not report_path.is_file() or not info_path.is_file():
        raise RawDataError(f"prepared dataset is missing: {root}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    info = json.loads(info_path.read_text(encoding="utf-8"))
    summary = {
        "path": str(root),
        "episodes": info["total_episodes"],
        "frames": info["total_frames"],
        "fps": info["fps"],
        "unique_episodes": report.get("unique_episodes", info["total_episodes"]),
        "source_episode_counts": report.get("source_episode_counts", {}),
        "training_source_episode_counts": report.get("training_source_episode_counts", {}),
        "train_episodes": report["train_episode_indices"],
        "validation_episodes": report["validation_episode_indices"],
        "projected_action_rows": report["projected_action_rows"],
        "projected_by_servo": report["projected_by_servo"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def clean(config_path: Path) -> int:
    config = load_json(config_path)
    workspace = (DATA / "temp" / str(config["name"])).resolve()
    allowed = (DATA / "temp").resolve()
    if workspace.parent != allowed:
        raise RawDataError(f"refusing unsafe workspace path: {workspace}")
    if workspace.exists():
        shutil.rmtree(workspace)
        print(f"REMOVED {workspace}")
    else:
        print(f"ALREADY_CLEAN {workspace}")
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Prepare and validate the VLA training dataset")
    result.add_argument("command", choices=("prepare", "check", "clean"))
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        return {"prepare": prepare, "check": check, "clean": clean}[args.command](args.config)
    except (KeyError, OSError, RawDataError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
