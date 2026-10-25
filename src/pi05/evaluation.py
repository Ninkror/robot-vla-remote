from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from action import ActionCodecConfig, DynamicBoundedRelativeCodec
from common.paths import CONFIGS, DATA, RUNS
from pi05.config import DATASET_NAME
from pi05.data_loader import LeRobotV3Dataset
from pi05.runner import Pi05Runner


def validation_episodes() -> tuple[int, ...]:
    report_path = DATA / "datasets" / DATASET_NAME / "quality-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    values = tuple(int(value) for value in report["validation_episode_indices"])
    if not values:
        raise RuntimeError("PI evaluation has no validation episodes")
    return values


def evaluate_checkpoint(checkpoint: Path | str, output: Path | str | None = None) -> dict:
    checkpoint = Path(checkpoint).resolve()
    episodes = validation_episodes()
    codec = DynamicBoundedRelativeCodec(
        ActionCodecConfig.load(CONFIGS / "action_codec.json")
    )
    runner = Pi05Runner(
        str(checkpoint),
        actions_per_chunk=72,
        device="cuda",
        codec_path=str(CONFIGS / "action_codec.json"),
    )
    results = []
    for episode in episodes:
        dataset = LeRobotV3Dataset(
            DATA / "datasets" / DATASET_NAME,
            episodes=(episode,),
            action_horizon=72,
        )
        sample = dataset[0]
        state = np.asarray(sample["observation.state"], dtype=np.float32)
        expected = codec.decode_chunk(state, np.asarray(sample["action"], dtype=np.float32))
        predicted, inference_seconds = runner.predict(
            sample["observation.images.arm_camera"],
            state,
            str(sample["prompt"]),
        )
        absolute_error = np.abs(predicted - expected)
        results.append(
            {
                "episode_index": episode,
                "inference_seconds": float(inference_seconds),
                "mean_absolute_error": float(absolute_error.mean()),
                "final_absolute_error": float(absolute_error[-1].mean()),
                "max_absolute_error": float(absolute_error.max()),
                "per_joint_mean_absolute_error": absolute_error.mean(axis=0).tolist(),
                "predicted_final": predicted[-1].tolist(),
                "expected_final": expected[-1].tolist(),
            }
        )
    report = {
        "checkpoint": str(checkpoint),
        "dataset": DATASET_NAME,
        "episodes": list(episodes),
        "metrics": results,
        "aggregate_mean_absolute_error": float(
            np.mean([item["mean_absolute_error"] for item in results])
        ),
        "aggregate_final_absolute_error": float(
            np.mean([item["final_absolute_error"] for item in results])
        ),
    }
    destination = Path(output) if output else RUNS / "pi05" / "evaluation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
