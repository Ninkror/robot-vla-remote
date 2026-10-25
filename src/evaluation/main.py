from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import cv2

from common.paths import RUNS
from dataset.raw import RawDataError, load_episode
from service.model_runner import SmolVLARunner

from .target import evaluate_actions


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Evaluate one model against a v7 expert anchor")
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--episode", type=Path, required=True)
    result.add_argument("--actions", type=int, default=72)
    result.add_argument("--device", default="cuda")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        episode = load_episode(args.episode)
        target = episode.expert_target
        if target is None:
            raise ValueError("evaluation requires a schema v7 expert episode")
        image_bgr = cv2.imread(str(target.anchor_image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError("expert anchor RGB is unreadable")
        runner = SmolVLARunner(str(args.model), args.actions, args.device)
        actions, latency = runner.predict(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB),
            target.anchor_state,
            episode.task,
        )
        open_pulse = (
            target.gripper_open_pulse
            if target.gripper_open_pulse is not None
            else target.anchor_state[5]
        )
        evaluation = evaluate_actions(
            actions,
            target.ik_targets[-1],
            open_pulse,
            target.gripper_closed_pulse,
        )
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = RUNS / "evaluation" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        report = {
            "episode_id": episode.path.name,
            "target_source": target.target_source,
            "task": episode.task,
            "model": runner.model_path,
            "action_rows": int(actions.shape[0]),
            "inference_seconds": float(latency),
            "codec_sha256": runner.codec_sha256,
            **evaluation,
        }
        (run_dir / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except (OSError, RawDataError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
