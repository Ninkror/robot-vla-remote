from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCHEMA = "dynamic_bounded_relative_v1"


class ActionCodecError(ValueError):
    """Raised when action codec inputs violate the declared safety contract."""


@dataclass(frozen=True)
class EncodingReport:
    projected_rows: int
    projected_by_servo: tuple[int, ...]


@dataclass(frozen=True)
class ActionCodecConfig:
    schema: str
    servo_ids: tuple[int, ...]
    hard_lower: tuple[float, ...]
    hard_upper: tuple[float, ...]
    soft_lower: tuple[float, ...]
    soft_upper: tuple[float, ...]
    max_step: tuple[float, ...]
    epsilon: float
    fps: float

    @classmethod
    def default_v1(cls) -> "ActionCodecConfig":
        return cls(
            schema=SCHEMA,
            servo_ids=(1, 2, 3, 4, 5, 10),
            hard_lower=(0, 100, 0, 41, 0, 100),
            hard_upper=(1000, 900, 1000, 959, 1000, 900),
            soft_lower=(20, 120, 20, 61, 20, 120),
            soft_upper=(980, 880, 980, 939, 980, 880),
            max_step=(30, 70, 90, 80, 20, 60),
            epsilon=1e-4,
            fps=1000.0 / 150.0,
        )

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise ActionCodecError(
                f"unsupported action codec schema {self.schema!r}; expected {SCHEMA!r}"
            )
        width = len(self.servo_ids)
        if width != 6 or len(set(self.servo_ids)) != width:
            raise ActionCodecError("servo_ids must contain six unique entries")
        for name in (
            "hard_lower",
            "hard_upper",
            "soft_lower",
            "soft_upper",
            "max_step",
        ):
            values = getattr(self, name)
            if len(values) != width or not np.isfinite(values).all():
                raise ActionCodecError(f"{name} must contain six finite values")
        hard_lower = np.asarray(self.hard_lower, dtype=np.float64)
        hard_upper = np.asarray(self.hard_upper, dtype=np.float64)
        soft_lower = np.asarray(self.soft_lower, dtype=np.float64)
        soft_upper = np.asarray(self.soft_upper, dtype=np.float64)
        max_step = np.asarray(self.max_step, dtype=np.float64)
        if np.any(hard_lower >= hard_upper):
            raise ActionCodecError("hard lower limits must be below upper limits")
        if np.any(soft_lower < hard_lower) or np.any(soft_upper > hard_upper):
            raise ActionCodecError("soft limits must be inside hard limits")
        if np.any(soft_lower >= soft_upper):
            raise ActionCodecError("soft lower limits must be below upper limits")
        if np.any(max_step <= 0):
            raise ActionCodecError("max_step values must be positive")
        if not (0.0 < self.epsilon < 0.5):
            raise ActionCodecError("epsilon must be between zero and one half")
        if not np.isfinite(self.fps) or self.fps <= 0:
            raise ActionCodecError("fps must be positive and finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsilon": self.epsilon,
            "fps": self.fps,
            "hard_lower": [float(value) for value in self.hard_lower],
            "hard_upper": [float(value) for value in self.hard_upper],
            "max_step": [float(value) for value in self.max_step],
            "schema": self.schema,
            "servo_ids": list(self.servo_ids),
            "soft_lower": [float(value) for value in self.soft_lower],
            "soft_upper": [float(value) for value in self.soft_upper],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ActionCodecConfig":
        required = {
            "schema",
            "servo_ids",
            "hard_lower",
            "hard_upper",
            "soft_lower",
            "soft_upper",
            "max_step",
            "epsilon",
            "fps",
        }
        missing = sorted(required - payload.keys())
        if missing:
            raise ActionCodecError(f"missing action codec fields: {', '.join(missing)}")
        return cls(
            schema=str(payload["schema"]),
            servo_ids=tuple(int(value) for value in payload["servo_ids"]),
            hard_lower=tuple(float(value) for value in payload["hard_lower"]),
            hard_upper=tuple(float(value) for value in payload["hard_upper"]),
            soft_lower=tuple(float(value) for value in payload["soft_lower"]),
            soft_upper=tuple(float(value) for value in payload["soft_upper"]),
            max_step=tuple(float(value) for value in payload["max_step"]),
            epsilon=float(payload["epsilon"]),
            fps=float(payload["fps"]),
        )

    @classmethod
    def load(cls, path: str | Path) -> "ActionCodecConfig":
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ActionCodecError(f"cannot load action codec config: {exc}") from exc
        if not isinstance(payload, dict):
            raise ActionCodecError("action codec config must be a JSON object")
        return cls.from_dict(payload)

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class DynamicBoundedRelativeCodec:
    def __init__(self, config: ActionCodecConfig):
        self.config = config
        self._hard_lower = np.asarray(config.hard_lower, dtype=np.float64)
        self._hard_upper = np.asarray(config.hard_upper, dtype=np.float64)
        self._soft_lower = np.asarray(config.soft_lower, dtype=np.float64)
        self._soft_upper = np.asarray(config.soft_upper, dtype=np.float64)
        self._max_step = np.asarray(config.max_step, dtype=np.float64)

    def _matrix(self, values: np.ndarray, name: str) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(self.config.servo_ids):
            raise ActionCodecError(f"{name} must have shape [N, 6], got {matrix.shape}")
        bad = np.argwhere(~np.isfinite(matrix))
        if bad.size:
            row, column = bad[0]
            raise ActionCodecError(
                f"{name} row {row} servo {self.config.servo_ids[column]} is not finite"
            )
        return matrix

    def _state(self, values: np.ndarray, name: str = "initial state") -> np.ndarray:
        state = np.asarray(values, dtype=np.float64)
        if state.shape != (len(self.config.servo_ids),):
            raise ActionCodecError(f"{name} must have shape [6], got {state.shape}")
        for column, value in enumerate(state):
            servo_id = self.config.servo_ids[column]
            if not np.isfinite(value):
                raise ActionCodecError(f"{name} servo {servo_id} is not finite")
            if value < self._hard_lower[column] or value > self._hard_upper[column]:
                raise ActionCodecError(
                    f"{name} servo {servo_id}={value:g} is outside hard limits "
                    f"[{self._hard_lower[column]:g}, {self._hard_upper[column]:g}]"
                )
        return state

    def dynamic_bounds(self, previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        state = self._state(previous, "previous state")
        lower = np.empty_like(state)
        upper = np.empty_like(state)
        below = state < self._soft_lower
        above = state > self._soft_upper
        inside = ~(below | above)
        lower[below] = state[below]
        upper[below] = np.minimum(
            self._soft_lower[below], state[below] + self._max_step[below]
        )
        lower[above] = np.maximum(
            self._soft_upper[above], state[above] - self._max_step[above]
        )
        upper[above] = state[above]
        lower[inside] = np.maximum(
            self._soft_lower[inside], state[inside] - self._max_step[inside]
        )
        upper[inside] = np.minimum(
            self._soft_upper[inside], state[inside] + self._max_step[inside]
        )
        return lower, upper

    def encode_episode(
        self, initial_state: np.ndarray, absolute_actions: np.ndarray
    ) -> tuple[np.ndarray, EncodingReport]:
        previous = self._state(initial_state).copy()
        actions = self._matrix(absolute_actions, "absolute actions")
        projected_counts = np.zeros(len(self.config.servo_ids), dtype=np.int64)
        projected_rows = 0
        latent_rows: list[np.ndarray] = []
        for row_index, raw_target in enumerate(actions):
            for column, value in enumerate(raw_target):
                if value < self._hard_lower[column] or value > self._hard_upper[column]:
                    raise ActionCodecError(
                        f"absolute actions row {row_index} servo "
                        f"{self.config.servo_ids[column]}={value:g} is outside hard limits"
                    )
            safe_target = np.clip(raw_target, self._soft_lower, self._soft_upper)
            projected = ~np.isclose(safe_target, raw_target, atol=1e-9, rtol=0.0)
            if projected.any():
                projected_rows += 1
                projected_counts += projected.astype(np.int64)
            lower, upper = self.dynamic_bounds(previous)
            outside = np.argwhere((safe_target < lower - 1e-9) | (safe_target > upper + 1e-9))
            if outside.size:
                column = int(outside[0, 0])
                raise ActionCodecError(
                    f"absolute actions row {row_index} servo {self.config.servo_ids[column]}="
                    f"{safe_target[column]:g} is outside dynamic interval "
                    f"[{lower[column]:g}, {upper[column]:g}]"
                )
            width = upper - lower
            ratio = np.divide(
                safe_target - lower,
                width,
                out=np.full_like(safe_target, 0.5),
                where=width > 0,
            )
            ratio = np.clip(ratio, self.config.epsilon, 1.0 - self.config.epsilon)
            latent_rows.append(np.log(ratio / (1.0 - ratio)))
            previous = safe_target
        latent = np.asarray(latent_rows, dtype=np.float32).reshape((-1, len(self.config.servo_ids)))
        latent.setflags(write=False)
        return latent, EncodingReport(
            projected_rows=projected_rows,
            projected_by_servo=tuple(int(value) for value in projected_counts),
        )

    def decode_chunk(self, initial_state: np.ndarray, latent_actions: np.ndarray) -> np.ndarray:
        previous = self._state(initial_state).copy()
        latent = self._matrix(latent_actions, "latent actions")
        decoded_rows: list[np.ndarray] = []
        for row in latent:
            lower, upper = self.dynamic_bounds(previous)
            stable = np.clip(row, -80.0, 80.0)
            ratio = 1.0 / (1.0 + np.exp(-stable))
            ratio = np.clip(ratio, self.config.epsilon, 1.0 - self.config.epsilon)
            target = lower + ratio * (upper - lower)
            decoded_rows.append(target)
            previous = target
        decoded = np.asarray(decoded_rows, dtype=np.float32).reshape(
            (-1, len(self.config.servo_ids))
        )
        decoded.setflags(write=False)
        return decoded
