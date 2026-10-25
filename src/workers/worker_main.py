from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
import signal
import socket
import time

import numpy as np

from workers.protocol import receive, send


def _write_status(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps({"updated_at_unix": time.time(), **values}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_runner(args):
    if args.model_id == "smolvla":
        from service.model_runner import SmolVLARunner

        return SmolVLARunner(args.checkpoint, args.actions, args.device)
    if args.model_id == "pi05":
        from pi05.runner import Pi05Runner

        return Pi05Runner(args.checkpoint, args.actions, args.device, args.codec)
    raise ValueError(f"unsupported worker model: {args.model_id}")


def _predict(runner, request: dict) -> dict:
    if request.get("operation") != "predict":
        raise ValueError("unknown worker operation")
    height = int(request["height"])
    width = int(request["width"])
    raw = base64.b64decode(request["rgb_base64"], validate=True)
    image = np.frombuffer(raw, dtype=np.uint8)
    if image.size != height * width * 3:
        raise ValueError("worker RGB payload has an invalid size")
    image = image.reshape(height, width, 3)
    actions, elapsed = runner.predict(image, request["state"], request["task"])
    return {
        "ok": True,
        "actions": np.asarray(actions, dtype=np.float32).tolist(),
        "inference_seconds": float(elapsed),
        "model_path": str(runner.model_path),
        "codec_sha256": str(runner.codec_sha256),
        "device": str(runner.device),
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--model-id", required=True, choices=("smolvla", "pi05"))
    result.add_argument("--checkpoint", required=True)
    result.add_argument("--socket", type=Path, required=True)
    result.add_argument("--status", type=Path, required=True)
    result.add_argument("--codec", required=True)
    result.add_argument("--actions", type=int, required=True)
    result.add_argument("--device", default="cuda")
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    args.socket.unlink(missing_ok=True)
    _write_status(args.status, state="loading", model_id=args.model_id, pid=os.getpid())
    try:
        runner = _load_runner(args)
    except BaseException as error:
        _write_status(
            args.status,
            state="failed",
            model_id=args.model_id,
            pid=os.getpid(),
            error=f"{type(error).__name__}: {error}",
        )
        raise

    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True
        server.close()

    signal.signal(signal.SIGTERM, stop)
    server.bind(str(args.socket))
    os.chmod(args.socket, 0o600)
    server.listen(1)
    server.settimeout(1.0)
    _write_status(
        args.status,
        state="ready",
        model_id=args.model_id,
        pid=os.getpid(),
        model_path=str(runner.model_path),
        device=str(runner.device),
        codec_sha256=str(runner.codec_sha256),
    )
    try:
        while not stopping:
            try:
                connection, _address = server.accept()
            except socket.timeout:
                continue
            except OSError:
                if stopping:
                    break
                raise
            with connection:
                connection.settimeout(300.0)
                try:
                    send(connection, _predict(runner, receive(connection)))
                except BaseException as error:
                    send(connection, {"ok": False, "error": f"{type(error).__name__}: {error}"})
    finally:
        try:
            server.close()
        finally:
            args.socket.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
