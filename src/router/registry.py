from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.config import load_json
from common.paths import ROOT, CONFIGS, resolve_path, framework_path, selected_gpu


INFERABLE_STATUSES = frozenset({"trained_dry_run", "validated"})


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    framework: str
    python: Path
    checkpoint: Path
    configured_status: str
    robot_compatible: bool
    device: str
    cuda_visible_devices: str
    message: str
    prompt_aliases: dict[str, str]

    @property
    def can_infer(self) -> bool:
        return self.configured_status in INFERABLE_STATUSES and self.checkpoint.exists()

    @property
    def can_execute(self) -> bool:
        return self.configured_status == "validated" and self.robot_compatible


class ModelRegistry:
    def __init__(self, config_path: Path | str, gpu: str | None = None):
        payload = load_json(Path(config_path))
        self.default_model_id = str(payload["default_model"])
        self.gpu_policy = str(payload.get("gpu_policy", "single_active"))
        if self.gpu_policy != "single_active":
            raise ValueError("only the single_active GPU policy is supported")
        raw_models = payload.get("models")
        if not isinstance(raw_models, dict) or not raw_models:
            raise ValueError("model registry must contain models")
        models = {}
        for model_id, raw in raw_models.items():
            if not isinstance(raw, dict):
                raise ValueError(f"model {model_id!r} must be an object")
            aliases = raw.get("prompt_aliases", {})
            if not isinstance(aliases, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in aliases.items()
            ):
                raise ValueError(f"model {model_id!r} prompt aliases are invalid")
            spec = ModelSpec(
                model_id=str(model_id),
                display_name=str(raw.get("display_name", model_id)),
                framework=str(raw["framework"]),
                python=framework_path(str(raw["framework"])) / ".venv/bin/python",
                checkpoint=resolve_path(raw["checkpoint"]),
                configured_status=str(raw["status"]),
                robot_compatible=bool(raw.get("robot_compatible", False)),
                device=str(raw.get("device", "cuda")),
                cuda_visible_devices=selected_gpu(gpu),
                message=str(raw.get("reason", "")),
                prompt_aliases={str(key).strip(): str(value).strip() for key, value in aliases.items()},
            )
            if spec.configured_status not in {"base_only", "trained_dry_run", "validated", "disabled"}:
                raise ValueError(f"model {model_id!r} has an unknown status")
            models[spec.model_id] = spec
        if self.default_model_id not in models:
            raise ValueError("default model is not registered")
        self.models = models

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self.models[model_id]
        except KeyError as error:
            raise KeyError(f"unknown model: {model_id}") from error

    def canonical_prompt(self, model_id: str, prompt: str) -> str:
        spec = self.get(model_id)
        normalized = prompt.strip()
        return spec.prompt_aliases.get(normalized, normalized)
