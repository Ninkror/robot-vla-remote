import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "config"
DATA = ROOT / "data"
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
RUNS = LOGS
RUNTIME = ROOT / "run"

def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()

def framework_path(name: str) -> Path:
    config = json.loads((CONFIGS / "frameworks.json").read_text(encoding="utf-8"))
    return resolve_path(config[name]["path"])

def selected_gpu(override: str | None = None) -> str:
    config = json.loads((CONFIGS / "server.json").read_text(encoding="utf-8"))
    value = str(override if override is not None else config["gpu"])
    if not value or not all(part.isdigit() for part in value.split(",")):
        raise ValueError("GPU must be a visible device index or comma-separated indices")
    return value

LEROBOT_ROOT = framework_path("lerobot")
OPENPI_ROOT = framework_path("openpi")
LEROBOT_PYTHON = LEROBOT_ROOT / ".venv/bin/python"
OPENPI_PYTHON = OPENPI_ROOT / ".venv/bin/python"
