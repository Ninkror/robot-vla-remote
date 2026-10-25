from __future__ import annotations

import base64
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time

import numpy as np

from common.paths import LOGS, CONFIGS, ROOT, RUNTIME
from router.registry import ModelRegistry, ModelSpec
from workers.protocol import receive, send


class ModelUnavailable(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


class WorkerManager:
    WORKER_STOP_GRACE_S = 3.0
    WORKER_KILL_TIMEOUT_S = 5.0

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        actions_per_chunk: int,
        codec_path: Path | str,
        lease_timeout_s: float = 600.0,
    ) -> None:
        self.registry = registry
        self.actions_per_chunk = int(actions_per_chunk)
        self.codec_path = Path(codec_path)
        self.lease_timeout_s = float(lease_timeout_s)
        self.worker_root = RUNTIME / "workers"
        self.worker_root.mkdir(parents=True, exist_ok=True)
        self.active_model_id = ""
        self._process: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._session_id = ""
        self._session_model_id = ""
        self._session_deadline = 0.0

    def _socket_path(self, model_id: str) -> Path:
        return self.worker_root / f"{model_id}.sock"

    def _status_path(self, model_id: str) -> Path:
        return self.worker_root / f"{model_id}.json"

    def _log_path(self, model_id: str) -> Path:
        return LOGS / f"worker-{model_id}.log"

    def _configured_info(self, spec: ModelSpec) -> dict:
        status = spec.configured_status
        message = spec.message
        if status in {"trained_dry_run", "validated"} and not spec.checkpoint.exists():
            status = "failed"
            message = f"checkpoint is missing: {spec.checkpoint}"
        return {
            "model_id": spec.model_id,
            "display_name": spec.display_name,
            "framework": spec.framework,
            "status": status,
            "can_infer": spec.can_infer,
            "can_execute": spec.can_execute,
            "message": message,
            "active": spec.model_id == self.active_model_id,
        }

    def _read_worker_status(self, model_id: str) -> dict | None:
        try:
            value = json.loads(self._status_path(model_id).read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def info(self, model_id: str) -> dict:
        with self._lock:
            spec = self.registry.get(model_id)
            result = self._configured_info(spec)
            if model_id != self.active_model_id:
                return result
            runtime = self._read_worker_status(model_id)
            process = self._process
            if runtime is not None:
                runtime_state = str(runtime.get("state", "loading"))
                if runtime_state == "ready":
                    result["status"] = spec.configured_status
                    result["message"] = ""
                elif runtime_state == "failed":
                    result["status"] = "failed"
                    result["message"] = str(runtime.get("error", "worker failed"))
                else:
                    result["status"] = "loading"
                    result["message"] = "模型正在加载"
            elif process is not None and process.poll() is None:
                result["status"] = "loading"
                result["message"] = "模型正在加载"
            elif process is not None:
                result["status"] = "failed"
                result["message"] = f"worker exited with code {process.returncode}"
            result["active"] = True
            return result

    def list_models(self) -> list[dict]:
        return [self.info(model_id) for model_id in self.registry.models]

    def _expire_session(self) -> None:
        if self._session_id and time.monotonic() >= self._session_deadline:
            self._session_id = ""
            self._session_model_id = ""
            self._session_deadline = 0.0

    def _stop_worker(self) -> None:
        process = self._process
        model_id = self.active_model_id
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=self.WORKER_STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    process.wait(timeout=self.WORKER_KILL_TIMEOUT_S)
                except subprocess.TimeoutExpired as error:
                    raise ModelUnavailable(
                        "MODEL_STOP_TIMEOUT",
                        f"model worker {model_id} did not stop after SIGKILL",
                    ) from error
        self._process = None
        if model_id:
            self._socket_path(model_id).unlink(missing_ok=True)
            self._status_path(model_id).unlink(missing_ok=True)
        self.active_model_id = ""

    def close(self) -> None:
        with self._lock:
            self._stop_worker()

    def _start_worker(self, spec: ModelSpec) -> None:
        self._stop_worker()
        socket_path = self._socket_path(spec.model_id)
        status_path = self._status_path(spec.model_id)
        socket_path.unlink(missing_ok=True)
        status_path.unlink(missing_ok=True)
        command = [
            str(spec.python),
            "-m",
            "workers.worker_main",
            "--model-id",
            spec.model_id,
            "--checkpoint",
            str(spec.checkpoint.resolve()),
            "--socket",
            str(socket_path),
            "--status",
            str(status_path),
            "--codec",
            str(self.codec_path.resolve()),
            "--actions",
            str(self.actions_per_chunk),
            "--device",
            spec.device,
        ]
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONPATH": str(ROOT / "src"),
                "HF_HOME": str(ROOT / "models" / "base" / "huggingface"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "OPENPI_DATA_HOME": str(ROOT / "models" / "base" / "openpi"),
                "PYTHONUNBUFFERED": "1",
            }
        )
        if spec.cuda_visible_devices:
            environment["CUDA_VISIBLE_DEVICES"] = spec.cuda_visible_devices
        if spec.framework == "openpi":
            environment.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
        log_path = self._log_path(spec.model_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            self._process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        self.active_model_id = spec.model_id

    def ensure_ready(self, model_id: str, session_id: str) -> tuple[ModelSpec, dict]:
        with self._lock:
            self._expire_session()
            try:
                spec = self.registry.get(model_id)
            except KeyError as error:
                raise ModelUnavailable("MODEL_NOT_FOUND", str(error)) from error
            if not spec.can_infer:
                raise ModelUnavailable(
                    "MODEL_INCOMPATIBLE",
                    spec.message or f"model {model_id} is not ready for robot inference",
                )
            if self._session_id and self._session_id != session_id and self._session_model_id != model_id:
                raise ModelUnavailable("MODEL_SWITCH_LOCKED", "另一个执行会话正在锁定当前模型")
            if self.active_model_id != model_id:
                self._start_worker(spec)
                raise ModelUnavailable("MODEL_LOADING", f"model {model_id} is loading")
            info = self.info(model_id)
            if info["status"] == "loading":
                raise ModelUnavailable("MODEL_LOADING", f"model {model_id} is loading")
            if info["status"] == "failed":
                raise ModelUnavailable("MODEL_LOAD_FAILED", info["message"])
            return spec, info

    def predict(self, spec: ModelSpec, image: np.ndarray, state, task: str, timeout_s: float = 300.0) -> dict:
        request = {
            "operation": "predict",
            "height": int(image.shape[0]),
            "width": int(image.shape[1]),
            "rgb_base64": base64.b64encode(image.tobytes()).decode("ascii"),
            "state": [float(value) for value in state],
            "task": self.registry.canonical_prompt(spec.model_id, task),
        }
        with self._lock:
            path = self._socket_path(spec.model_id)
            if not path.exists():
                raise ModelUnavailable("MODEL_LOAD_FAILED", "model worker socket is unavailable")
            connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            connection.settimeout(timeout_s)
            try:
                connection.connect(str(path))
                send(connection, request)
                response = receive(connection)
            finally:
                connection.close()
        if not response.get("ok"):
            raise RuntimeError(str(response.get("error", "model worker failed")))
        return response

    def authorize_session(self, session_id: str, model_id: str) -> None:
        if not session_id:
            raise ValueError("session_id is required for executable predictions")
        with self._lock:
            self._expire_session()
            if self._session_id and self._session_id != session_id:
                raise ModelUnavailable("MODEL_SWITCH_LOCKED", "另一个执行会话正在锁定当前模型")
            self._session_id = session_id
            self._session_model_id = model_id
            self._session_deadline = time.monotonic() + self.lease_timeout_s

    def release_session(self, session_id: str) -> bool:
        with self._lock:
            self._expire_session()
            if session_id and session_id == self._session_id:
                self._session_id = ""
                self._session_model_id = ""
                self._session_deadline = 0.0
                return True
            return False
