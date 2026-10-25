import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

from common.config import load_json
from common.paths import resolve_path
from common.paths import (
    CONFIGS,
    LEROBOT_PYTHON,
    LEROBOT_ROOT,
    MODELS,
    OPENPI_PYTHON,
    OPENPI_ROOT,
)
from router.registry import ModelRegistry


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _probe(python: Path, code: str, environment: dict | None = None) -> dict:
    if not python.is_file():
        return {"ready": False, "error": f"missing interpreter: {python}"}
    merged = os.environ.copy()
    merged.update(environment or {})
    try:
        output = subprocess.check_output(
            [str(python), "-c", code],
            text=True,
            stderr=subprocess.STDOUT,
            env=merged,
            timeout=60,
        )
        return json.loads(output.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {"ready": False, "error": str(error)}


def main() -> int:
    checks = {"python": sys.version.split()[0]}
    try:
        for package in ("grpc", "numpy"):
            importlib.import_module(package)
            checks[f"robot_vla_import_{package}"] = True

        checks["lerobot"] = _probe(
            LEROBOT_PYTHON,
            "import json,lerobot,torch; "
            "print(json.dumps({'ready': True, 'torch': torch.__version__, "
            "'cuda_available': torch.cuda.is_available(), "
            "'cuda_devices': torch.cuda.device_count(), 'lerobot': lerobot.__file__}))",
        )
        checks["openpi"] = _probe(
            OPENPI_PYTHON,
            "import json,jax,openpi; "
            "print(json.dumps({'ready': True, 'jax': jax.__version__, "
            "'backend': jax.default_backend(), 'devices': len(jax.devices()), "
            "'openpi': openpi.__file__}))",
            {"XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        )
        checks["lerobot_revision"] = _revision(LEROBOT_ROOT)
        checks["openpi_revision"] = _revision(OPENPI_ROOT)
        frameworks = load_json(CONFIGS / "frameworks.json")
        checks["framework_revisions_match"] = (
            checks["lerobot_revision"] == frameworks["lerobot"]["revision"]
            and checks["openpi_revision"] == frameworks["openpi"]["revision"]
        )

        dataset = load_json(CONFIGS / "dataset.json")
        archive = resolve_path(dataset["source_archive"])
        checks["source_archive"] = archive.is_file()
        checks["source_archive_sha256"] = (
            archive.is_file() and _sha256(archive) == dataset["source_sha256"]
        )
        checks["smolvla_base"] = (
            MODELS / "base" / "smolvla" / "model.safetensors"
        ).is_file()
        checks["smolvlm2"] = (
            MODELS / "base" / "smolvlm2" / "config.json"
        ).is_file()
        pi05 = MODELS / "base" / "pi05"
        checks["pi05_base"] = (
            (pi05 / "params" / "_METADATA").is_file()
            and any((pi05 / "params").glob("**/*"))
        )
        registry = ModelRegistry(CONFIGS / "models.json")
        checks["models"] = {
            name: {"checkpoint": str(spec.checkpoint), "can_infer": spec.can_infer,
                   "can_execute": spec.can_execute and spec.can_infer}
            for name, spec in registry.models.items()
        }
        required = (
            "robot_vla_import_grpc",
            "robot_vla_import_numpy",
            "framework_revisions_match",
        )
        checks["installation_ok"] = (
            all(checks.get(key) is True for key in required)
            and checks["lerobot"].get("ready") is True
            and checks["lerobot"].get("cuda_available") is True
            and checks["openpi"].get("ready") is True
            and checks["openpi"].get("backend") == "gpu"
        )
        checks["training_archive_checksum_ok"] = checks["source_archive"] and checks["source_archive_sha256"]
        checks["ok"] = checks["installation_ok"] and any(value["can_infer"] for value in checks["models"].values())
    except Exception as error:
        checks["ok"] = False
        checks["error"] = str(error)
    print(json.dumps(checks, ensure_ascii=False, indent=2))
    return 0 if checks.get("ok") else 2


if __name__ == "__main__":
    raise SystemExit(main())
