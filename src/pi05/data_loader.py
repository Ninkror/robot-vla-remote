from __future__ import annotations

import io
from pathlib import Path
from typing import SupportsIndex

import numpy as np
from openpi.training import data_loader as _loader
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq

from pi05.config import RosorinData


_ORIGINAL = _loader.create_torch_dataset
_IMAGE_KEY = "observation.images.arm_camera"
_COLUMNS = (
    "observation.state",
    "action",
    _IMAGE_KEY,
    "episode_index",
    "frame_index",
    "task_index",
)


class LeRobotV3Dataset:
    """Read a local LeRobot v3 parquet dataset without Hugging Face Hub.

    OpenPI currently pins LeRobot 0.1, whose built-in reader expects the v2.1
    metadata layout. The recorder produces v3 datasets. This adapter leaves
    the canonical v3 dataset in place and supplies the random-access samples
    OpenPI needs for normalization and training.
    """

    def __init__(self, root: str | Path, episodes: tuple[int, ...], action_horizon: int):
        self._root = Path(root)
        self._action_horizon = int(action_horizon)
        if self._action_horizon < 1:
            raise ValueError("action horizon must be positive")

        data_files = sorted((self._root / "data").glob("chunk-*/*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"no LeRobot v3 parquet data found under {self._root}")
        self._table = pa.concat_tables(
            [pq.read_table(path, columns=list(_COLUMNS)) for path in data_files],
            promote_options="default",
        ).combine_chunks()

        all_episode_indices = np.asarray(self._table["episode_index"], dtype=np.int64)
        selected = set(int(value) for value in episodes)
        available = set(int(value) for value in np.unique(all_episode_indices))
        missing = sorted(selected - available)
        if missing:
            raise ValueError(f"requested episodes are missing from dataset: {missing}")

        self._sample_rows: list[int] = []
        self._future_rows: dict[int, np.ndarray] = {}
        for episode in sorted(selected):
            rows = np.flatnonzero(all_episode_indices == episode)
            if not len(rows):
                continue
            frame_indices = np.asarray(
                [self._table["frame_index"][int(row)].as_py() for row in rows],
                dtype=np.int64,
            )
            rows = rows[np.argsort(frame_indices, kind="stable")]
            self._sample_rows.extend(int(row) for row in rows)
            future_offsets = np.arange(self._action_horizon, dtype=np.int64)
            for offset, row in enumerate(rows):
                selected_offsets = np.minimum(offset + future_offsets, len(rows) - 1)
                self._future_rows[int(row)] = rows[selected_offsets]

        if not self._sample_rows:
            raise ValueError("no frames remain after episode selection")
        self._tasks = self._read_tasks(self._root / "meta" / "tasks.parquet")

    @staticmethod
    def _read_tasks(path: Path) -> dict[int, str]:
        if not path.is_file():
            raise FileNotFoundError(f"LeRobot v3 tasks metadata is missing: {path}")
        result: dict[int, str] = {}
        for row in pq.read_table(path).to_pylist():
            task_index = int(row["task_index"])
            task = row.get("task") or row.get("__index_level_0__")
            if not task:
                raise ValueError(f"task text is missing for task index {task_index}")
            result[task_index] = str(task)
        return result

    def __len__(self) -> int:
        return len(self._sample_rows)

    def __getitem__(self, index: SupportsIndex) -> dict:
        row = self._sample_rows[index.__index__()]
        image_record = self._table[_IMAGE_KEY][row].as_py()
        payload = image_record.get("bytes")
        if payload is None:
            relative_path = image_record.get("path")
            if not relative_path:
                raise ValueError(f"image payload is missing at dataset row {row}")
            payload = (self._root / relative_path).read_bytes()
        with Image.open(io.BytesIO(payload)) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

        future_rows = self._future_rows[row]
        actions = np.asarray(
            [self._table["action"][int(value)].as_py() for value in future_rows],
            dtype=np.float32,
        )
        task_index = int(self._table["task_index"][row].as_py())
        return {
            "observation.state": np.asarray(
                self._table["observation.state"][row].as_py(), dtype=np.float32
            ),
            _IMAGE_KEY: rgb,
            "action": actions,
            "prompt": self._tasks[task_index],
        }


def create_torch_dataset(data_config, action_horizon: int, model_config):
    if not isinstance(data_config, RosorinData):
        return _ORIGINAL(data_config, action_horizon, model_config)
    return LeRobotV3Dataset(data_config.root, data_config.episodes, action_horizon)


def install() -> None:
    _loader.create_torch_dataset = create_torch_dataset
