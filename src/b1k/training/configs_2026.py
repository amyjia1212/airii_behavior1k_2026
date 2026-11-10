"""Training configs for the 2026 BEHAVIOR Challenge (100 tasks, LeRobot v3 dataset).

These configs were reconstructed from the run configs logged to Weights & Biases
(entity `b1k-2026`, projects `b1k-2026-chunks`, `b1k-2026-single-task`, `b1k-train`). Hyperparameters,
task splits, init checkpoints and step counts are copied verbatim from those logs. Machine-specific paths
were replaced by environment variables so the configs work on any machine:

    B1K_DATA_ROOT    local copy of behavior-1k/2026-challenge-demos       (default ~/data/2026-challenge-demos)
    B1K_MODELS_DIR   directory holding init checkpoints (see scripts/download_checkpoints.py)
                                                                          (default ~/models)
    B1K_OUTPUTS_DIR  where assets (norm stats / FAST tokenizer) and checkpoints are written
                                                                          (default ./outputs)

Training recipe (see REPORT.md for the full write-up):

  stage 0  IliaLarchenko/behavior_50t_checkpoint: the 2025 winners' 50-task generalist PI_BEHAVIOR checkpoint.
  stage 1  t5 base: 5-task model on tasks 90/91/93/96/97, 100k steps, initialised from stage 0.
           (published as amyjia1212/b1k-2026-t90-91-93-96-97; its W&B logs were deleted, see note below.)
  stage 2  "chunk" models: 4-5 tasks each, 50k steps, initialised from the stage-1 model.
  stage 3  single-task fine-tunes (6k-10k steps) for tasks where a chunk model was weak, initialised either from
           stage 0 or from the corresponding chunk model.
"""

from __future__ import annotations

import os

import openpi.training.optimizer as _optimizer

from b1k.models import pi_behavior_config
from b1k.training import weight_loaders

import dataclasses

import optax


DATA_ROOT = os.path.expanduser(os.environ.get("B1K_DATA_ROOT", "~/data/2026-challenge-demos"))
MODELS_DIR = os.path.expanduser(os.environ.get("B1K_MODELS_DIR", "~/models"))
OUTPUTS_DIR = os.environ.get("B1K_OUTPUTS_DIR", "./outputs")

REPO_ID = "behavior-1k/2026-challenge-demos"
CAMERA_KEYS = (
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
)
WANDB_ENTITY = "b1k-2026"
HF_USER = "amyjia1212"


@dataclasses.dataclass(frozen=True)
class AdamWMuDtype(_optimizer.AdamW):
    """openpi AdamW with an optional low-precision first moment (`mu_dtype`).

    The chunk / single-task runs stored Adam's first moment in bfloat16 to fit a batch of 16 at 280x280 on a
    single A100-80GB.
    """

    mu_dtype: str | None = None

    def create(self, lr, weight_decay_mask=None):
        tx = optax.adamw(
            lr,
            b1=self.b1,
            b2=self.b2,
            eps=self.eps,
            weight_decay=self.weight_decay,
            mask=weight_decay_mask,
            mu_dtype=self.mu_dtype,
        )
        return optax.chain(optax.clip_by_global_norm(self.clip_gradient_norm), tx)


def _model_2026() -> pi_behavior_config.PiBehaviorConfig:
    # Same architecture / losses as the 2025 `pi_behavior_b1k_fast` config, with 100 tasks and 280x280 images.
    return pi_behavior_config.PiBehaviorConfig(
        action_horizon=30,
        action_dim=32,
        num_tasks=100,
        image_resolution=(280, 280),
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
        time_threshold_inpaint=0.3,
    )


def _ckpt(*parts: str) -> str:
    return os.path.join(MODELS_DIR, *parts, "params")


# Init checkpoints (local paths; scripts/download_checkpoints.py fetches the HF ones into B1K_MODELS_DIR)
INIT_50T = _ckpt("behavior_50t_checkpoint")
INIT_T5_BASE = _ckpt("b1k-2026-t90-91-93-96-97", "step_099999")
INIT_T5_50K = _ckpt("b1k-2026-t90-91-93-96-97", "step_049999")


def make_2026_configs(TrainConfig, LeRobotB1KDataConfig, DataConfig) -> list:
    """Build all 2026 configs. Classes are passed in to avoid a circular import with config.py."""

    def data(task_indices):
        return LeRobotB1KDataConfig(
            repo_id=REPO_ID,
            camera_keys=CAMERA_KEYS,
            base_config=DataConfig(
                prompt_from_task=False,
                behavior_dataset_root=DATA_ROOT,
                use_per_timestamp_norm=True,
                task_indices=list(task_indices),
            ),
            use_delta_joint_actions=True,
            use_fast_tokenization=True,
        )

    common = dict(
        model=_model_2026(),
        ema_decay=0.99,
        batch_size=16,
        num_flow_samples=15,
        log_interval=100,
        save_interval=2000,
        assets_base_dir=f"{OUTPUTS_DIR}/assets",
        checkpoint_base_dir=f"{OUTPUTS_DIR}/checkpoints",
        wandb_entity=WANDB_ENTITY,
    )

    configs = []

    # ---- stage 1: 5-task base model --------------------------------------------------------------------------
    # Settings from our run notes of 2026-09-26 (the W&B project of this run was deleted): 1x A100-80GB PCIe,
    # batch 16 (32 needed ~109 GB) without gradient accumulation, AdamW with bf16 first moment, 1k warm-up to 5e-5
    # and cosine decay to 5e-6 over 50k steps, EMA 0.99, 4 data workers (the container had a 117 GB RAM limit),
    # weights pushed to HF every 1k steps. The first 50k steps ended 2026-09-29 (step_049999); the run was then
    # continued for another 50k steps with a warm restart (2026-10-03..05, step_099999): see `t5_cont` below.
    configs.append(
        TrainConfig(
            name="pi_behavior_b1k_2026_t5",
            exp_name="t90-91-93-96-97",
            project_name="b1k-2026",
            data=data([90, 91, 93, 96, 97]),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1000, peak_lr=5e-5, decay_steps=50_000, decay_lr=5e-6
            ),
            optimizer=AdamWMuDtype(mu_dtype="bfloat16"),
            weight_loader=weight_loaders.PiBehaviorWeightLoader(INIT_50T),
            num_train_steps=50_000,
            num_workers=4,
            keep_period=None,
            max_to_keep=2,
            fsdp_devices=1,
            hf_repo_id=f"{HF_USER}/b1k-2026-t90-91-93-96-97",
            hf_push_interval=1000,  # the published repo has a checkpoint every 1000 steps
            restart_data_workers_after_checkpoint=True,
            **{**common, "save_interval": 1000},
        )
    )
    # Stage-1 continuation, steps 50k-100k (published as step_055000 ... step_099999). Its LR was not recorded;
    # we MEASURED it from the published checkpoints. With Adam, the weight change between two checkpoints is
    # proportional to the LR; over the first 50k (known schedule) the ratio for task_stage_embeddings is constant
    # (44-50 per 5k steps for LRs from 4.3e-5 down to 5.4e-6). Applied to 50k-100k, it gives an LR of about 1.7e-5
    # at 50k-60k, falling along a cosine to about 5e-6 at 95k-100k: a warm restart with a peak of about 1.8e-5
    # (+-10%) decaying to 5e-6. 2e-5 is our best guess for the value we typed. Here the step counter restarts at 0
    # and the optimizer state starts fresh, so checkpoint step N here corresponds to published step 50k + N.
    configs.append(
        TrainConfig(
            name="pi_behavior_b1k_2026_t5_cont",
            exp_name="t90-91-93-96-97-cont",
            project_name="b1k-2026",
            data=data([90, 91, 93, 96, 97]),
            lr_schedule=_optimizer.CosineDecaySchedule(
                warmup_steps=1000, peak_lr=2e-5, decay_steps=50_000, decay_lr=5e-6
            ),
            optimizer=AdamWMuDtype(mu_dtype="bfloat16"),
            weight_loader=weight_loaders.PiBehaviorWeightLoader(INIT_T5_50K),
            num_train_steps=50_000,
            num_workers=4,
            keep_period=None,
            max_to_keep=2,
            fsdp_devices=1,
            hf_repo_id=f"{HF_USER}/b1k-2026-t90-91-93-96-97-cont",
            hf_push_interval=5000,
            restart_data_workers_after_checkpoint=True,
            **{**common, "save_interval": 1000},
        )
    )

    # ---- stage 2: chunk models (W&B project b1k-2026-chunks, 5x A100-80GB PCIe, one run per GPU) -------------
    chunks = {
        "c1": [50, 70, 72, 85],
        "c2": [52, 53, 73, 66, 86],
        "c3": [65, 71, 74, 75],
        "c4": [79, 81, 84, 88],
        "c5": [94, 95, 98],
    }
    for i, (cid, tasks) in enumerate(chunks.items(), start=1):
        tag = "t" + "-".join(map(str, tasks))
        configs.append(
            TrainConfig(
                name=f"pi_behavior_b1k_2026_{cid}",
                exp_name=tag,
                project_name="b1k-2026-chunks",
                wandb_run_name=f"chunk{i}-{tag}",
                data=data(tasks),
                lr_schedule=_optimizer.CosineDecaySchedule(
                    warmup_steps=1000, peak_lr=5e-5, decay_steps=50_000, decay_lr=5e-6
                ),
                optimizer=AdamWMuDtype(mu_dtype="bfloat16"),
                weight_loader=weight_loaders.PiBehaviorWeightLoader(INIT_T5_BASE),
                num_train_steps=50_000,
                num_workers=8,
                keep_period=None,
                max_to_keep=2,
                fsdp_devices=1,
                hf_repo_id=f"{HF_USER}/b1k-2026-{tag}",
                hf_push_interval=2000,
                restart_data_workers_after_checkpoint=True,
                **common,
            )
        )

    # ---- stage 3a: single-task fine-tunes from the 2025 50-task checkpoint (project b1k-2026-single-task) -----
    for task in (50, 52, 72):
        configs.append(
            TrainConfig(
                name=f"pi_behavior_b1k_2026_s{task}",
                exp_name=f"t{task}",
                project_name="b1k-2026-single-task",
                wandb_run_name=f"single-t{task}-from-50t",
                data=data([task]),
                lr_schedule=_optimizer.CosineDecaySchedule(
                    warmup_steps=1000, peak_lr=3e-5, decay_steps=30_000, decay_lr=3e-6
                ),
                optimizer=AdamWMuDtype(mu_dtype="bfloat16"),
                weight_loader=weight_loaders.PiBehaviorWeightLoader(INIT_50T),
                num_train_steps=10_000,
                num_workers=4,
                keep_period=None,
                max_to_keep=2,
                fsdp_devices=1,
                hf_repo_id=f"{HF_USER}/b1k-2026-t{task}",
                hf_push_interval=2000,
                restart_data_workers_after_checkpoint=True,
                **common,
            )
        )

    # ---- stage 3b: single-task runs on 2-4x A100-SXM4-80GB with FSDP (project b1k-train) ----------------------
    # (name suffix, task, init checkpoint, steps, cosine decay steps)
    single_runs = [
        ("t53", 53, INIT_50T, 6_000, 20_000),
        ("t65", 65, INIT_50T, 6_000, 20_000),
        ("t66", 66, INIT_50T, 6_000, 20_000),
        ("t71", 71, INIT_50T, 6_000, 20_000),
        ("t74", 74, INIT_50T, 6_000, 20_000),
        ("t79_ft10k", 79, _ckpt("b1k-2026-t79-81-84-88", "step_049999"), 10_000, 10_000),
        ("t81_ft10k", 81, _ckpt("b1k-2026-t79-81-84-88", "step_049999"), 10_000, 10_000),
        ("t86_ft10k", 86, _ckpt("b1k-2026-t52-53-73-66-86", "step_049999"), 10_000, 10_000),
        ("t94_ft10k", 94, _ckpt("b1k-2026-t94-95-98", "step_049999"), 10_000, 10_000),
        ("t52_cont10k", 52, _ckpt("b1k-2026-t52", "step_009999"), 10_000, 10_000),
        ("t53_cont10k", 53, _ckpt("b1k-2026-t53", "step_005999"), 10_000, 10_000),
    ]
    for suffix, task, init, steps, decay in single_runs:
        configs.append(
            TrainConfig(
                name=f"pi_behavior_b1k_2026_{suffix}",
                exp_name=f"b1k-2026-{suffix.replace('_', '-')}",
                project_name="b1k-train",
                data=data([task]),
                lr_schedule=_optimizer.CosineDecaySchedule(
                    warmup_steps=1000, peak_lr=1e-4, decay_steps=decay, decay_lr=1e-5
                ),
                optimizer=_optimizer.AdamW(),
                weight_loader=weight_loaders.PiBehaviorWeightLoader(init),
                num_train_steps=steps,
                num_workers=10,
                keep_period=2000,
                fsdp_devices=2,
                **{**common, "wandb_entity": WANDB_ENTITY},
            )
        )

    return configs
