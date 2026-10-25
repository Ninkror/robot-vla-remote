import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common.config import load_json
from common.paths import LOGS, CONFIGS, DATA, MODELS, ROOT, RUNS


CURRENT = RUNS / "training" / "current.json"


def _current() -> dict:
    if not CURRENT.is_file():
        raise RuntimeError("no training run has been started")
    return json.loads(CURRENT.read_text(encoding="utf-8"))


def start(gpu: str | None = None) -> int:
    config = load_json(CONFIGS / "training.json")
    dataset_config = load_json(CONFIGS / "dataset.json")
    dataset_name = str(dataset_config["name"])
    dataset_root = DATA / "datasets" / dataset_name
    if not (dataset_root / "quality-report.json").is_file():
        raise RuntimeError("prepared dataset is missing; run data.sh prepare first")
    if not (MODELS / "base" / "smolvla" / "model.safetensors").is_file():
        raise RuntimeError("SmolVLA base model is missing")
    run_name = f"{dataset_name}-{int(config['steps'])}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    if (MODELS / "trained" / run_name).exists() or (RUNS / "training" / run_name).exists():
        raise RuntimeError(f"run already exists: {run_name}")
    launcher_log = LOGS / f"train-{run_name}.launcher.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    from common.paths import selected_gpu
    environment["CUDA_VISIBLE_DEVICES"] = selected_gpu(gpu)
    launcher_log.parent.mkdir(parents=True, exist_ok=True)
    with launcher_log.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [str(ROOT / ".venv" / "bin" / "python"), "-m", "training.worker", run_name],
            cwd=ROOT,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    CURRENT.parent.mkdir(parents=True, exist_ok=True)
    CURRENT.write_text(
        json.dumps({"run_name": run_name, "pid": process.pid}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"TRAINING_STARTED run={run_name} pid={process.pid}")
    return 0


def status() -> int:
    current = _current()
    run_dir = RUNS / "training" / current["run_name"]
    status_path = run_dir / "status.json"
    if status_path.is_file():
        print(status_path.read_text(encoding="utf-8"), end="")
    else:
        alive = Path(f"/proc/{current['pid']}").exists()
        print(json.dumps({**current, "state": "launching" if alive else "not-running"}, indent=2))
    return 0


def logs(lines: int) -> int:
    current = _current()
    log_path = RUNS / "training" / current["run_name"] / "train.log"
    if not log_path.is_file():
        raise RuntimeError(f"training log is not available yet: {log_path}")
    content = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(content[-lines:]))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Start and inspect SmolVLA training")
    result.add_argument("command", choices=("start", "status", "logs"))
    result.add_argument("--gpu", default=None)
    result.add_argument("--lines", type=int, default=80)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "start":
            return start(args.gpu)
        if args.command == "status":
            return status()
        return logs(args.lines)
    except (KeyError, OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
