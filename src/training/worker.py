import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from common.config import load_json
from common.paths import selected_gpu
from common.paths import CONFIGS, DATA, LEROBOT_ROOT, MODELS, ROOT, RUNS


def policy_horizon_args(config: dict) -> list[str]:
    chunk_size = config.get("chunk_size")
    n_action_steps = config.get("n_action_steps")
    for name, value in (
        ("chunk_size", chunk_size),
        ("n_action_steps", n_action_steps),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if n_action_steps > chunk_size:
        raise ValueError("n_action_steps must not exceed chunk_size")
    return [
        f"--policy.chunk_size={chunk_size}",
        f"--policy.n_action_steps={n_action_steps}",
    ]


def _write_status(path: Path, **values) -> None:
    payload = {"updated_at_unix": time.time(), **values}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) != 1:
        raise SystemExit("usage: python -m training.worker RUN_NAME")
    run_name = argv[0]
    config = load_json(CONFIGS / "training.json")
    dataset_config = load_json(CONFIGS / "dataset.json")
    dataset_name = str(dataset_config["name"])
    dataset_root = DATA / "datasets" / dataset_name
    report = json.loads((dataset_root / "quality-report.json").read_text(encoding="utf-8"))
    train_episodes = report["train_episode_indices"]
    output = MODELS / "trained" / run_name
    run_dir = RUNS / "training" / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    status_path = run_dir / "status.json"
    log_path = run_dir / "train.log"
    _write_status(status_path, state="starting", pid=os.getpid(), model_path=str(output))

    steps = int(config["steps"])
    rename_map = config.get("rename_map", {})
    if not isinstance(rename_map, dict) or not all(
        isinstance(source, str) and isinstance(target, str)
        for source, target in rename_map.items()
    ):
        raise ValueError("training rename_map must be a string-to-string mapping")
    command = [
        str(LEROBOT_ROOT / ".venv" / "bin" / "lerobot-train"),
        f"--policy.path={MODELS / 'base' / 'smolvla'}",
        f"--policy.vlm_model_name={MODELS / 'base' / 'smolvlm2'}",
        f"--dataset.repo_id=local/{dataset_name}",
        f"--dataset.root={dataset_root}",
        f"--dataset.episodes={json.dumps(train_episodes, separators=(',', ':'))}",
        f"--output_dir={output}",
        f"--job_name={run_name}",
        f"--steps={steps}",
        f"--batch_size={int(config['batch_size'])}",
        f"--num_workers={int(config['num_workers'])}",
        f"--save_freq={int(config.get('save_freq', steps))}",
        f"--log_freq={int(config.get('log_freq', 10))}",
        f"--policy.scheduler_warmup_steps={int(config.get('warmup_steps', 50))}",
        f"--policy.scheduler_decay_steps={steps}",
        "--policy.push_to_hub=false",
    ]
    command.extend(policy_horizon_args(config))
    if rename_map:
        command.append(
            f"--rename_map={json.dumps(rename_map, ensure_ascii=False, separators=(',', ':'))}"
        )
    (run_dir / "command.json").write_text(
        json.dumps(command, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copyfile(CONFIGS / "training.json", run_dir / "training.json")
    shutil.copyfile(CONFIGS / "dataset.json", run_dir / "dataset.json")
    shutil.copyfile(CONFIGS / "action_codec.json", run_dir / "action_codec.json")

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", selected_gpu()),
            "HF_HOME": str(MODELS / "base" / "huggingface"),
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_DISABLED": "true",
            "PYTHONPATH": str(ROOT / "src"),
        }
    )
    _write_status(status_path, state="running", pid=os.getpid(), model_path=str(output))
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode != 0:
        _write_status(
            status_path,
            state="failed",
            pid=os.getpid(),
            returncode=completed.returncode,
            model_path=str(output),
        )
        return completed.returncode

    checkpoints = sorted(output.glob("checkpoints/*/pretrained_model"))
    if not checkpoints:
        _write_status(status_path, state="failed", pid=os.getpid(), error="no checkpoint found")
        return 2
    checkpoint = checkpoints[-1]
    shutil.copyfile(CONFIGS / "action_codec.json", checkpoint / "action_codec.json")
    contract = {
        "schema_version": 1,
        "input_profile": str(report.get("input_profile", "rgb_state_v1")),
        "raw_schema_versions": [int(report.get("raw_schema_version", 7))],
        "camera_mapping": "single_camera_camera1",
        "action_representation": report.get("action_representation", "absolute_pulse"),
    }
    (checkpoint / "model_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_status(
        status_path,
        state="completed",
        pid=os.getpid(),
        model_path=str(output),
        checkpoint=str(checkpoint),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
