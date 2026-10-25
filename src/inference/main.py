import argparse
import json
from pathlib import Path
import sys
import time

import cv2

from common.paths import RUNS
from service.model_runner import SmolVLARunner


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run one offline SmolVLA inference")
    result.add_argument("--model", type=Path, required=True)
    result.add_argument("--image", type=Path, required=True)
    result.add_argument("--state", required=True, help="JSON array or path to a JSON file")
    result.add_argument("--task", required=True)
    result.add_argument("--actions", type=int, default=72)
    result.add_argument("--device", default="cuda")
    return result


def _state(value: str):
    path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else json.loads(value)
    if isinstance(payload, dict):
        payload = payload.get("state")
    return payload


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        image_bgr = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise ValueError(f"image is unreadable: {args.image}")
        runner = SmolVLARunner(str(args.model), args.actions, args.device)
        actions, latency = runner.predict(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), _state(args.state), args.task
        )
        run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        run_dir = RUNS / "inference" / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        report = {
            "task": args.task,
            "model": runner.model_path,
            "state": _state(args.state),
            "action_rows": int(actions.shape[0]),
            "actions": actions.tolist(),
            "inference_seconds": float(latency),
            "codec_sha256": runner.codec_sha256,
        }
        (run_dir / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
