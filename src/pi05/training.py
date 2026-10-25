from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

from common.paths import CONFIGS, DATA, OPENPI_ROOT
from common.config import load_json
from pi05.config import CONFIG_NAME, DATASET_NAME, make_train_config
from pi05.data_loader import install as install_data_loader


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load OpenPI script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _train_episodes() -> tuple[int, ...]:
    report_path = DATA / "datasets" / DATASET_NAME / "quality-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    values = tuple(int(value) for value in report["train_episode_indices"])
    if not values:
        raise RuntimeError("PI training has no training episodes")
    return values


def _config(args):
    return make_train_config(
        episodes=_train_episodes(),
        steps=args.steps,
        batch_size=args.batch_size,
        experiment=args.experiment,
        overwrite=args.overwrite,
    )


def compute_stats(args) -> int:
    config = _config(args)
    install_data_loader()
    from openpi.training import config as openpi_config

    openpi_config._CONFIGS_DICT[CONFIG_NAME] = config
    module = _module("robot_vla_compute_norm_stats", OPENPI_ROOT / "scripts" / "compute_norm_stats.py")
    module.main(CONFIG_NAME)
    return 0


def train(args) -> int:
    config = _config(args)
    install_data_loader()
    module = _module("robot_vla_openpi_train", OPENPI_ROOT / "scripts" / "train.py")
    module.main(config)
    return 0


def evaluate(args) -> int:
    from pi05.evaluation import evaluate_checkpoint

    report = evaluate_checkpoint(args.checkpoint, args.output)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


def parser() -> argparse.ArgumentParser:
    defaults = load_json(CONFIGS / "training.json")["pi05"]
    result = argparse.ArgumentParser(description="Train PI 0.5 for the ROSOrin arm")
    subparsers = result.add_subparsers(dest="command", required=True)
    for command in ("stats", "train"):
        item = subparsers.add_parser(command)
        item.add_argument("--steps", type=int, default=int(defaults["steps"]))
        item.add_argument("--batch-size", type=int, default=int(defaults["batch_size"]))
        item.add_argument("--experiment", default=defaults["experiment"])
        item.add_argument("--overwrite", action="store_true")
    item = subparsers.add_parser("evaluate")
    item.add_argument("--checkpoint", type=Path, required=True)
    item.add_argument("--output", type=Path)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    if args.command == "evaluate":
        return evaluate(args)
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError("steps and batch size must be positive")
    if args.command == "stats":
        return compute_stats(args)
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
