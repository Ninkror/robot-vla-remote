from __future__ import annotations

import dataclasses
from pathlib import Path

from openpi import transforms as _transforms
from openpi.models import pi0_config
from openpi.training import config as _config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

from action import ActionCodecConfig, DynamicBoundedRelativeCodec
from common.paths import DATA, MODELS
from pi05.transforms import RosorinInputs, RosorinOutputs


CONFIG_NAME = "pi05_rosorin_red_block"
DATASET_NAME = "red_block_expert20_terminal_v2"


@dataclasses.dataclass(frozen=True)
class RosorinData(_config.DataConfig):
    root: str = ""
    episodes: tuple[int, ...] = ()


@dataclasses.dataclass(frozen=True)
class RosorinDataConfig(_config.DataConfigFactory):
    root: str = str(DATA / "datasets" / DATASET_NAME)
    episodes: tuple[int, ...] = ()
    codec_path: str = str(MODELS.parent / "config" / "action_codec.json")

    def create(self, assets_dirs: Path, model_config) -> _config.DataConfig:
        base = self.create_base_config(assets_dirs, model_config)
        codec = DynamicBoundedRelativeCodec(ActionCodecConfig.load(self.codec_path))
        repack = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.arm_camera",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data = _transforms.Group(
            inputs=[RosorinInputs()],
            outputs=[RosorinOutputs(codec)],
        )
        return RosorinData(
            repo_id=base.repo_id,
            asset_id=base.asset_id,
            norm_stats=base.norm_stats,
            root=self.root,
            episodes=self.episodes,
            repack_transforms=repack,
            data_transforms=data,
            model_transforms=_config.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
            prompt_from_task=True,
            use_quantile_norm=True,
        )


def make_train_config(
    *,
    episodes: tuple[int, ...],
    steps: int,
    batch_size: int,
    experiment: str,
    overwrite: bool = False,
) -> _config.TrainConfig:
    model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=72,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    return _config.TrainConfig(
        name=CONFIG_NAME,
        project_name="robot-vla",
        exp_name=experiment,
        model=model,
        data=RosorinDataConfig(
            repo_id=f"local/{DATASET_NAME}",
            base_config=_config.DataConfig(prompt_from_task=True),
            episodes=episodes,
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(
            str(MODELS / "base" / "pi05" / "params")
        ),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=min(200, max(1, steps // 10), max(0, steps - 1)),
            peak_lr=2.5e-5,
            decay_steps=steps,
            decay_lr=2.5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        assets_base_dir=str(MODELS / "assets"),
        checkpoint_base_dir=str(MODELS / "trained"),
        batch_size=batch_size,
        num_workers=0,
        num_train_steps=steps,
        log_interval=10,
        save_interval=max(50, min(500, steps // 5)),
        keep_period=steps + 1,
        overwrite=overwrite,
        wandb_enabled=False,
        fsdp_devices=1,
        policy_metadata={
            "robot_type": "hiwonder_rosorin_pro_arm",
            "action_codec": "dynamic_bounded_relative_v1",
            "control_period_ms": 150,
            "active_action_dims": 6,
        },
    )
