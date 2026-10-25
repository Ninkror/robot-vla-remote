import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from common.config import load_json
from common.paths import LOGS, CONFIGS, ROOT, RUNTIME


PID_PATH = RUNTIME / "pids" / "service.pid"
LOG_PATH = LOGS / "service.log"


def _pid() -> int | None:
    try:
        return int(PID_PATH.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None


def _alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(") ", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def start(allow_motion: bool, gpu: str | None = None) -> int:
    existing = _pid()
    if _alive(existing):
        raise RuntimeError(f"service is already running: pid={existing}")
    config = load_json(CONFIGS / "server.json")
    command = [
        str(ROOT / ".venv" / "bin" / "python"), "-m", "service.main",
        "--host", str(config["host"]), "--port", str(config["port"]),
        "--registry", str(CONFIGS / "models.json"),
        "--codec", str(CONFIGS / "action_codec.json"),
        "--actions", str(config["actions_per_chunk"]),
    ]
    from common.paths import selected_gpu
    command.extend(["--gpu", selected_gpu(gpu)])
    if allow_motion or config.get("allow_motion", False):
        command.append("--allow-motion")
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(ROOT / "src"),
        "HF_HOME": str(ROOT / "models" / "base" / "huggingface"),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
    })
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    PID_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdin=subprocess.DEVNULL, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    PID_PATH.write_text(f"{process.pid}\n", encoding="utf-8")
    time.sleep(2)
    if process.poll() is not None:
        raise RuntimeError(f"service failed to start; inspect {LOG_PATH}")
    print(f"SERVICE_STARTED pid={process.pid} router=multi-model")
    return 0


def stop() -> int:
    pid = _pid()
    if not _alive(pid):
        PID_PATH.unlink(missing_ok=True)
        print("SERVICE_NOT_RUNNING")
        return 0
    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    if "service.main" not in cmdline:
        raise RuntimeError(f"refusing to stop unrelated pid {pid}: {cmdline}")
    os.kill(pid, signal.SIGTERM)
    for _ in range(300):
        if not _alive(pid):
            break
        time.sleep(0.1)
    if _alive(pid):
        raise RuntimeError(f"service did not stop: pid={pid}")
    PID_PATH.unlink(missing_ok=True)
    print(f"SERVICE_STOPPED pid={pid}")
    return 0


def status() -> int:
    pid = _pid()
    state = "running" if _alive(pid) else "stopped"
    print(json.dumps({"state": state, "pid": pid, "log": str(LOG_PATH)}, indent=2))
    return 0 if state == "running" else 1


def logs(lines: int) -> int:
    if not LOG_PATH.is_file():
        raise RuntimeError("service log does not exist")
    print("\n".join(LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]))
    return 0


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Manage the remote inference service")
    result.add_argument("command", choices=("start", "stop", "status", "logs"))
    result.add_argument("--gpu", default=None)
    result.add_argument("--allow-motion", action="store_true")
    result.add_argument("--lines", type=int, default=80)
    return result


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "start":
            return start(args.allow_motion, args.gpu)
        if args.command == "stop":
            return stop()
        if args.command == "status":
            return status()
        return logs(args.lines)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
