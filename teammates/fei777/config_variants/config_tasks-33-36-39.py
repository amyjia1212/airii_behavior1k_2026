"""Training configuration for BEHAVIOR-1K challenge.

Reference: https://github.com/Physical-Intelligence/openpi
"""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import os
import pathlib
from typing import Any, Literal, List, Protocol, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

# Import from OpenPI
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.aloha_policy as aloha_policy
import openpi.policies.droid_policy as droid_policy
import openpi.policies.libero_policy as libero_policy
import openpi.shared.download as _download
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.optimizer as _optimizer
import openpi.transforms as _transforms

# Import from B1K custom modules
from b1k.models import pi_behavior_config
from b1k.policies import b1k_policy
from b1k.shared import normalize as _normalize
from b1k.training import weight_loaders
from b1k import transforms as b1k_transforms

ModelType: TypeAlias = _model.ModelType
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.
    """
    # Assets directory. If not provided, the config assets_dirs will be used.
    assets_dir: str | None = None
    # Asset id. If not provided, the repo id will be used.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False
    # If true, will use per-timestamp normalization for actions
    use_per_timestamp_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt (not used for PI_BEHAVIOR).
    prompt_from_task: bool = False

    # Only used for RLDS data loader.
    rlds_data_dir: str | None = None

    # Only used for B1K data loader.
    behavior_dataset_root: str | None = None

    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # Episodes index to use for training 
    episodes_index: List[int] | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for B1K."""

    default_prompt: str | None = None  # Not used (task embeddings instead)
    # Maps dataset task_index -> model task_id. Needed when the dataset's task_index exceeds
    # the model's task/stage embedding table size (TASK_NUM_STAGES only covers 50 task slots,
    # from the original 2025 challenge; the 2026 challenge has 100 tasks). E.g. {83: 0} trains
    # store_honey (task_index=83) using task-slot 0's embeddings/stage-count.
    task_mapping: dict[int, int] | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        return _transforms.Group(
            inputs=[
                _transforms.ResizeImages(224, 224),
                b1k_transforms.ComputeSubtaskStateFromMeta(dataset=None, task_mapping=self.task_mapping),
                b1k_transforms.TaskIndexToTaskId(task_mapping=self.task_mapping),
                _transforms.PadStatesAndActions(model_config.action_dim),
            ],
        )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=False,  # Always use z-score normalization for B1K
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class LeRobotB1KDataConfig(DataConfigFactory):
    """Data configuration for BEHAVIOR-1K dataset."""

    action_sequence_keys: Sequence[str] = ("action",)
    use_delta_joint_actions: bool = False

    # FAST auxiliary tokenization (only for PI_BEHAVIOR with use_fast_auxiliary)
    use_fast_tokenization: bool = False

    # See ModelTransformFactory.task_mapping: needed for any 2026-challenge task_index >= 50.
    task_mapping: dict[int, int] | None = None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Repack transforms for B1K observations
        repack_mapping = {
            # Raw feature names from the behavior-1k/2026-challenge-demos LeRobot v3 dataset
            # (R1Pro camera link names): zed=head, left/right_realsense=wrist cameras.
            "observation/egocentric_camera": "observation.rgb.zed_link_camera_0",
            "observation/wrist_image_left": "observation.rgb.left_realsense_link_camera_0",
            "observation/wrist_image_right": "observation.rgb.right_realsense_link_camera_0",
            "observation/state": "observation.state",
            "actions": "action",
            "task_index": "task_index",  # Always preserve task_index
            "timestamp": "timestamp",    # Preserve timestamp for subtask state computation
            "episode_index": "episode_index",  # Preserve episode_index for episode length lookup
            "index": "index",           # Preserve index
        }
            
        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_mapping)]
        )

        # Prepare data for policy training
        data_transforms = _transforms.Group(
            inputs=[b1k_policy.B1kInputs(model_type=model_config.model_type)],
            outputs=[b1k_policy.B1kOutputs()],
        )

        # Delta action transforms
        if self.use_delta_joint_actions:
            delta_action_mask = _transforms.make_bool_mask(-3, 3, -1, 7, -1, 7, -1)
        else:
            delta_action_mask = _transforms.make_bool_mask(-23)
        
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        # Model transforms (subtask state, task ID, padding)
        model_transforms = ModelTransformFactory(task_mapping=self.task_mapping)(model_config)
        
        # FAST tokenization (if enabled for PI_BEHAVIOR)
        if self.use_fast_tokenization and hasattr(model_config, 'use_fast_auxiliary') and model_config.use_fast_auxiliary:
            asset_id = self.assets.asset_id or self.repo_id
            tokenizer_path = assets_dirs / asset_id / "fast_tokenizer"
            
            # Get base config to access norm_stats
            base_config = self.create_base_config(assets_dirs, model_config)
            
            # Only add transform if tokenizer directory exists
            if tokenizer_path.exists():
                model_transforms = model_transforms.push(
                    inputs=[b1k_transforms.TokenizeFASTActions(
                        tokenizer_path=str(tokenizer_path),
                        encoded_dim_ranges=model_config.get_fast_dim_ranges(),
                        max_fast_tokens=model_config.max_fast_tokens,
                        norm_stats=base_config.norm_stats,
                        use_per_timestamp=base_config.use_per_timestamp_norm,
                    )],
                )
            else:
                logging.warning(
                    f"FAST tokenizer not found at {tokenizer_path}. "
                    "FAST auxiliary training will be disabled (inference mode)."
                )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "B1K"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config (PI_BEHAVIOR only for B1K).
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi_behavior_config.PiBehaviorConfig)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Note: PyTorch support removed - JAX only

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=LeRobotB1KDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int | None = None
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # FSDP configuration for model sharding across devices.
    fsdp_devices: int = 1
    
    # Validation configuration
    val_log_interval: int = 100
    val_batch_size: int | None = None
    val_num_batches: int = 10
    val_repo_id: str | None = None
    val_episodes_index: List[int] | None = None
    
    # Number of flow matching samples per training step
    num_flow_samples: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# B1K Training Configurations
_CONFIGS = [
    TrainConfig(
        name="pi_behavior_b1k_fast",
        exp_name="openpi",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=True,
            correlation_beta=0.5,
            # FAST auxiliary training
            use_fast_auxiliary=True,
            fast_loss_weight=0.05,
            fast_encoded_dims="0:6,7:23",  # Encode 22 dimensions
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.1,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="IliaLarchenko/behavior_224_rgb",
            base_config=DataConfig(
                prompt_from_task=False,  # No text prompts for PI_BEHAVIOR
                behavior_dataset_root="~/data/behavior_224_rgb",
                use_per_timestamp_norm=True,  # Enable per-timestamp normalization
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,  # Enable FAST tokenization in data pipeline
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1000,
            peak_lr=1e-4,
            decay_steps=20_000,
            decay_lr=1e-5,
        ),
        num_flow_samples=15,
        weight_loader=weight_loaders.PiBehaviorWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=200_000,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=80,
        save_interval=500,
        keep_period=2000,
    ),
    TrainConfig(
        name="pi_behavior_b1k_radio",
        exp_name="turning_on_radio",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=True,
            fast_loss_weight=0.05,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.1,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2026-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root="./data/behavior_2026_demos",
                use_per_timestamp_norm=True,
                # task_index=0 (turning_on_radio) has 200 episodes. Originally held out the last 20
                # (this codebase's val_repo_id/val_episodes_index fields are unwired, no val loop
                # exists, so there was no reason to exclude them) -- now using all 200 for a further
                # fine-tune pass to give the model exposure to the extra data.
                episodes_index=list(range(200)),
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=1e-4,
            decay_steps=15_000,
            decay_lr=1e-5,
        ),
        num_flow_samples=15,
        weight_loader=weight_loaders.PiBehaviorWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=20_000,
        batch_size=16,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=16,
        save_interval=1000,
        keep_period=5000,
    ),
    TrainConfig(
        name="pi_behavior_b1k_store_honey",
        exp_name="store_honey",
        project_name="B1K",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=True,
            fast_loss_weight=0.05,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.1,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2026-challenge-demos",
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root="./data/behavior_2026_demos_task83",
                use_per_timestamp_norm=True,
                # task_index=83 (store_honey), 200 episodes, absolute indices 16600-16799
                # (task_index N occupies episodes [N*200, N*200+200) in this dataset).
                episodes_index=list(range(16600, 16800)),
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
            # TASK_NUM_STAGES / task_embeddings only cover 50 slots (the 2025 challenge's task
            # count); task_index 83 is from the 2026 challenge's added 50 tasks and has no slot
            # of its own. Remap to slot 0 (turning_on_radio's slot: TASK_NUM_STAGES[0]=5, the
            # fewest stages of any slot, a reasonable default for another simple task). NOTE:
            # correction_rules.py's MIN_STAGE_FOR_CLOSURE has a radio-specific entry keyed on
            # task_id 0 (don't close gripper before stage 2) -- harmless-by-default for a
            # pick-and-place task too, but worth revisiting if eval behavior looks gripper-shy.
            task_mapping={83: 0},
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=200,
            peak_lr=1e-4,
            decay_steps=4_500,
            decay_lr=1e-5,
        ),
        num_flow_samples=15,
        weight_loader=weight_loaders.PiBehaviorWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
        num_train_steps=5_000,
        batch_size=16,
        assets_base_dir="./outputs/assets",
        checkpoint_base_dir="./outputs/checkpoints",
        num_workers=16,
        save_interval=500,
        keep_period=2500,
    ),
]


# ---- BEHAVIOR Challenge 2026: fine-tunes of the 2025 winning checkpoints on the 2026 demos ----
# Policy A: tasks 2,3,6,19,28,44,47 initialised from IliaLarchenko/behavior_submission checkpoint_1 (rev 89545bc).
# Policy B: tasks 33,36,39 initialised from IliaLarchenko/behavior_submission checkpoint_3 (rev 89545bc).
# Same model/loss settings as pi_behavior_b1k_fast; norm stats + FAST tokenizer are reused unchanged from the
# initial checkpoint (asset id IliaLarchenko/behavior_224_rgb) so the pretrained input/output scaling is kept.
# Task slot = real task id (tasks < 50 keep their 2025 slots), 61-dim 2026 proprio layout (b1k.shared.proprioception).
def _b1k2026_ft_config(name: str, tasks: list[int], init_params: str, num_train_steps: int) -> "TrainConfig":
    return TrainConfig(
        name=name,
        exp_name=name,
        project_name="B1K-2026",
        model=pi_behavior_config.PiBehaviorConfig(
            action_horizon=30,
            action_dim=32,
            use_correlated_noise=True,
            correlation_beta=0.5,
            use_fast_auxiliary=True,
            fast_loss_weight=0.05,
            fast_encoded_dims="0:6,7:23",
            fast_vocab_size=1024,
            max_fast_tokens=200,
            use_kv_transform=True,
            use_knowledge_insulation=False,
            subtask_loss_weight=0.1,
            freeze_vision_backbone=True,
        ),
        data=LeRobotB1KDataConfig(
            repo_id="behavior-1k/2026-challenge-demos",
            assets=AssetsConfig(asset_id="IliaLarchenko/behavior_224_rgb"),
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root=os.environ.get("B1K_2026_DATA_ROOT", "/root/data/b1k2026"),
                use_per_timestamp_norm=True,
                # task_index N occupies absolute episodes [200N, 200N+200) of behavior-1k/2026-challenge-demos
                episodes_index=[200 * t + i for t in tasks for i in range(200)],
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
        ),
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=500,
            peak_lr=5e-5,
            decay_steps=num_train_steps,
            decay_lr=5e-6,
        ),
        num_flow_samples=15,
        weight_loader=weight_loaders.PiBehaviorWeightLoader(init_params),
        num_train_steps=num_train_steps,
        batch_size=32,
        assets_base_dir="/workspace/b1k2026_assets",
        checkpoint_base_dir="/workspace/b1k2026_checkpoints",
        num_workers=24,
        save_interval=1000,
        keep_period=5000,
        log_interval=50,
    )


_CONFIGS += [
    _b1k2026_ft_config(
        "pi_behavior_b1k2026_ft_ckpt1",
        tasks=[2, 3, 6, 19, 28, 44, 47],
        init_params="/workspace/models/behavior_submission/checkpoint_1/params",
        num_train_steps=int(os.environ.get("B1K_FT_STEPS_A", 40_000)),
    ),
    _b1k2026_ft_config(
        "pi_behavior_b1k2026_ft_ckpt3",
        tasks=[33, 36, 39],
        init_params="/workspace/models/behavior_submission/checkpoint_3/params",
        num_train_steps=int(os.environ.get("B1K_FT_STEPS_B", 16_000)),
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]

