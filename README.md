# BEHAVIOR Challenge 2026: PI_BEHAVIOR on 100 tasks

Code, configs and checkpoints for our team's 2026 [BEHAVIOR Challenge](https://behavior.stanford.edu/challenge/) entry (standard track, R1Pro, RGB + proprioception).

The model is **PI_BEHAVIOR**, the π0.5-based policy from the [1st-place 2025 solution](https://github.com/IliaLarchenko/behavior-1k-solution) by Ilia Larchenko. We adapted it to the 2026 challenge, which has 100 tasks, a LeRobot v3 dataset and BEHAVIOR-1K v3.9.3. This repository is a fork of that solution: its full git history is kept, and everything 2026-specific is in the commits on top.

* **Technical report:** [REPORT.md](REPORT.md)
* **Self-evaluation results:** [results/SELF_EVAL_2026.md](results/SELF_EVAL_2026.md)
* **Checkpoints:** Hugging Face, see [Checkpoints](#checkpoints)
* **Training logs:** W&B entity `b1k-2026`, projects `b1k-2026-chunks`, `b1k-2026-single-task`, `b1k-train`, `b1k-eval`, `b1k-eval-stagefix`

> [!IMPORTANT]
> **How this code was recovered (please read).** We trained on short-lived cloud GPU pods (RunPod), and our modified copy of the code was lost when those pods were deleted after the challenge. This repository was **rebuilt from what survived**:
>
> * **Base code:** the public 2025 solution at commit [`ca556f7`](https://github.com/IliaLarchenko/behavior-1k-solution/commit/ca556f74a455cef7987a2be4537b5ac85cc56dd7). The later runs logged exactly this commit; the earlier ones logged `92a031f`, a local commit that was lost.
> * **Configs:** the exact run configs, commands, package versions and logs of every training run, from W&B. Every hyperparameter in [`configs_2026.py`](src/b1k/training/configs_2026.py) comes from there.
> * **Code changes:** the log messages printed by our modified code, and the tensor shapes stored in our published checkpoints. Together these show what each 2026 change did.
>
> The 2026 changes were therefore **re-implemented**, not copied. What has been verified:
>
> * the new LeRobot v3 reader returns exactly the stored states, actions and video frames of the 2026 dataset;
> * the checkpoint-resizing code produces the exact shapes in our checkpoints (100 tasks, 1087 stage embeddings, 280px);
> * all 2026 stage counts match the checkpoint shapes.
>
> **Full end-to-end training and evaluation with this exact code has not been re-run yet.** The original simulator evaluations of the published checkpoints were done with our original code and are logged in W&B. The details are in [Reconstruction notes](#reconstruction-notes).

---

## What's in this repository

Our team trained models for 95 of the 100 tasks. Different team members used different models; **this repository contains the PI_BEHAVIOR pipeline, which produced the checkpoints for 25 tasks** (50, 52, 53, 65, 66, 70–75, 79, 81, 84–86, 88, 90, 91, 93–98). The checkpoints for the other tasks were produced by teammates, and their code will be added to this repository (see [Team checkpoints](#team-checkpoints)).

### Changes vs. the 2025 solution

| Area | Change | Files |
|---|---|---|
| Tasks | 50 → **100 tasks**. 2025 stage counts kept for tasks 0–49; tasks 50–99 use `floor(mean episode length / 900)` clamped to [5, 15], giving 1087 task-stage embeddings | `models/pi_behavior_config.py`, `transforms.py`, `scripts/compute_task_num_stages.py` |
| Images | **280×280** inputs (configurable `image_resolution`; SigLIP 20×20 patches instead of 16×16) | `models/pi_behavior_config.py`, `models/pi_behavior.py`, `training/config.py` |
| Init from 2025 checkpoints | Task and stage embedding tables are extended (new rows = mean of existing rows), and the SigLIP position embedding is interpolated bicubically from 16×16 to 20×20 | `training/weight_loaders.py` |
| Data | Standalone **LeRobot v3** reader for `behavior-1k/2026-challenge-demos` (shared parquet/mp4 files, HEVC video via PyAV) with task selection. The norm-stats and FAST-tokenizer scripts read v3 episodes too | `training/lerobot_v3.py`, `training/data_loader.py`, `scripts/compute_norm_stats.py`, `scripts/train_fast_tokenizer.py` |
| Training infra | Checkpoints are uploaded to Hugging Face every N steps (pods were ephemeral); W&B entity and run names; bf16 Adam first moment (`mu_dtype`); `max_to_keep`; data workers restarted after each checkpoint | `scripts/train.py`, `training/config.py`, `training/checkpoints.py`, `training/configs_2026.py` |
| Eval | BEHAVIOR-1K v3.9.3 import paths; 280px resize; **gripper recovery** (a gripper closed on nothing for 90 steps is reopened for 15 steps); checkpoint switching for tasks 0–99; full-resolution RGB eval wrapper | `shared/eval_b1k_wrapper.py`, `scripts/serve_b1k.py`, `policies/checkpoint_switcher.py`, `eval/` |
| Configs | All 2026 training configs (19 recovered from W&B and 1 partly assumed, see notes) | `training/configs_2026.py` |

The model architecture and losses are unchanged from 2025: task and stage conditioning, FAST auxiliary loss, KV transform, correlated noise, per-timestep normalisation, and the inference-time inpainting and compression. See [docs/README_2025.md](docs/README_2025.md) and the [2025 write-up](https://github.com/IliaLarchenko/behavior-1k-solution).

---

## Setup

Requirements: Linux, Python 3.11, CUDA 12, [uv](https://docs.astral.sh/uv/). Training used A100-80GB GPUs. Evaluation needs one GPU with 24 GB or more for the policy, plus the BEHAVIOR-1K simulator.

```bash
git clone --recurse-submodules https://github.com/amyjia1212/airii_behavior1k_2026.git
cd airii_behavior1k_2026
bash setup_remote.sh          # installs openpi + this package (uv) and BEHAVIOR-1K v3.9.3-post1 (OmniGibson[eval])
```

`requirements-2026-frozen.txt` is the exact `pip freeze` of the environment our 2026 training runs used (logged by W&B), for reference.

For the simulator evaluator, follow the official [BEHAVIOR-1K v3.9.3 setup](https://behavior.stanford.edu/challenge/baselines.html), which creates the `behavior` conda env.

## Checkpoints

```bash
uv run scripts/download_checkpoints.py          # 8 checkpoints -> ~/models (override with --models-dir / $B1K_MODELS_DIR)
```

| Checkpoint (Hugging Face) | Tasks served | Trained by config |
|---|---|---|
| [`amyjia1212/b1k-2026-t50-70-72-85`](https://huggingface.co/amyjia1212/b1k-2026-t50-70-72-85/tree/main/step_049999) `step_049999` | 50, 70, 72, 85 | `pi_behavior_b1k_2026_c1` |
| [`amyjia1212/b1k-2026-t52`](https://huggingface.co/amyjia1212/b1k-2026-t52/tree/main/step_008000) `step_008000` | 52 | `pi_behavior_b1k_2026_s52` |
| [`amyjia1212/b1k-2026-t52-53-73-66-86`](https://huggingface.co/amyjia1212/b1k-2026-t52-53-73-66-86/tree/main/step_049999) `step_049999` | 53, 66, 73, 86 | `pi_behavior_b1k_2026_c2` |
| [`amyjia1212/b1k-2026-t65-71-74-75`](https://huggingface.co/amyjia1212/b1k-2026-t65-71-74-75/tree/main/step_049999) `step_049999` | 65, 71, 74, 75 | `pi_behavior_b1k_2026_c3` |
| [`amyjia1212/b1k-2026-t79-81-84-88`](https://huggingface.co/amyjia1212/b1k-2026-t79-81-84-88/tree/main/step_049999) `step_049999` | 79, 84, 88 | `pi_behavior_b1k_2026_c4` |
| [`amyjia1212/b1k-2026-t81-ft10k`](https://huggingface.co/amyjia1212/b1k-2026-t81-ft10k/tree/main/step_009999) `step_009999` | 81 | `pi_behavior_b1k_2026_t81_ft10k` |
| [`amyjia1212/b1k-2026-t90-91-93-96-97`](https://huggingface.co/amyjia1212/b1k-2026-t90-91-93-96-97/tree/main/step_099999) `step_099999` | 90, 91, 93, 96, 97 | `pi_behavior_b1k_2026_t5_base` (partly assumed) |
| [`amyjia1212/b1k-2026-t94-95-98`](https://huggingface.co/amyjia1212/b1k-2026-t94-95-98/tree/main/step_049999) `step_049999` | 94, 95, 98 | `pi_behavior_b1k_2026_c5` |

Each checkpoint folder has `params/` (Orbax) and `assets/behavior-1k/2026-challenge-demos/`, which holds the norm stats with the action correlation matrix and the FAST tokenizer. The task → checkpoint routing is in [`task_checkpoint_mapping_2026.json`](task_checkpoint_mapping_2026.json).

## Evaluation

**1. Start the policy server.** It switches checkpoints by the `task_id` the evaluator sends. These are the settings we used for all 2026 evaluations:

```bash
uv run scripts/serve_b1k.py \
  --task-checkpoint-mapping task_checkpoint_mapping_2026.json \
  --image-size 280 \
  --actions-to-execute 13 --actions-to-keep 4 --execute-in-n-steps 10 \
  --num-steps 20 \
  --gripper-stuck-steps 90 --gripper-open-hold-steps 15 \
  --port 8000 \
  policy:checkpoint \
  --policy.config pi_behavior_b1k_2026_c1 \
  --policy.dir ~/models/b1k-2026-t50-70-72-85/step_049999
```

(`--policy.config` only defines the architecture, which is identical for all 2026 checkpoints. `--policy.dir` is the checkpoint loaded at start-up.)

**2. Run the BEHAVIOR-1K evaluator** in the `behavior` env, from the BEHAVIOR-1K v3.9.3 checkout:

```bash
python -m omnigibson.eval.eval \
  --task-name putting_dirty_dishes_in_sink \
  --host 127.0.0.1 --port 8000 \
  --instance-indices 0 1 2 3 4 5 6 7 8 9 --num-rollouts 1 \
  --robot-config /path/to/airii_behavior1k_2026/eval/r1pro.yaml \
  --env-wrapper rgb_full_res_wrapper.RGBFullResWrapper \
  --output-dir eval_logs/putting_dirty_dishes_in_sink --write-video
```

Put `eval/` on `PYTHONPATH` so the evaluator can import `rgb_full_res_wrapper`. Our self-evaluation used `RGBFullResWrapper`: head camera 720×720 and wrists 480×480, RGB only, resized to 280 by the server. `eval/default_wrapper.py` is the official low-resolution (224×224) wrapper that was included in our submission package. `eval/r1pro.yaml` is the robot config we submitted.

## Training

1. **Data.** Download `behavior-1k/2026-challenge-demos` (LeRobot v3) to `$B1K_DATA_ROOT` (default `~/data/2026-challenge-demos`). Only the chunks of the tasks you train on are needed (`chunk-XXX` = task XXX), plus `meta/`:

   ```bash
   huggingface-cli download behavior-1k/2026-challenge-demos --repo-type dataset --local-dir ~/data/2026-challenge-demos \
     --include "meta/**" --include "data/chunk-050/**" --include "videos/*/chunk-050/**"   # repeat per task
   ```

2. **Norm stats and FAST tokenizer.** Do this once per config. Alternatively, copy `assets/` from one of our checkpoints into `outputs/assets/<config>/`.

   ```bash
   uv run scripts/compute_norm_stats.py --config-name pi_behavior_b1k_2026_c1 --correlation
   uv run scripts/train_fast_tokenizer.py --config-name pi_behavior_b1k_2026_c1 --encoded-dims="0:6,7:23" --vocab-size=1024
   ```

3. **Train.** Set `hf_repo_id` in the config, or pass `--hf_repo_id=None` to disable uploads.

   ```bash
   uv run scripts/train.py pi_behavior_b1k_2026_c1 --overwrite                 # 1x A100-80GB, batch 16
   uv run scripts/train.py pi_behavior_b1k_2026_t81_ft10k --overwrite          # 2x A100 (fsdp_devices=2)
   ```

Our training was staged; [REPORT.md](REPORT.md) has the full description:

| Stage | Configs | Init | Steps | LR (cosine, peak → final) |
|---|---|---|---|---|
| 0 | n/a | 2025 PI_BEHAVIOR 50-task checkpoint (`behavior_50t_checkpoint`) | n/a | n/a |
| 1 | `pi_behavior_b1k_2026_t5_base` (tasks 90, 91, 93, 96, 97) | stage 0 | 100k | 5e-5 → 5e-6 *(assumed)* |
| 2 | `pi_behavior_b1k_2026_c1` … `c5` (4–5 tasks each) | stage 1, step 99,999 | 50k | 5e-5 → 5e-6 |
| 3a | `pi_behavior_b1k_2026_s50`, `s52`, `s72` | stage 0 | 10k | 3e-5 → 3e-6 (decay 30k) |
| 3b | `pi_behavior_b1k_2026_t53`, `t65`, `t66`, `t71`, `t74` | stage 0 | 6k | 1e-4 → 1e-5 (decay 20k) |
| 3c | `pi_behavior_b1k_2026_t79_ft10k`, `t81_ft10k`, `t86_ft10k`, `t94_ft10k` | matching stage-2 chunk | 10k | 1e-4 → 1e-5 |
| 3d | `pi_behavior_b1k_2026_t52_cont10k`, `t53_cont10k` | stage-3a / 3b result | 10k | 1e-4 → 1e-5 |

All stages use batch 16, EMA 0.99, 15 flow samples per step, the FAST auxiliary loss (weight 0.05) and the subtask loss (0.1), with the vision backbone frozen. Set `$B1K_MODELS_DIR` and put the stage-0 checkpoint at `$B1K_MODELS_DIR/behavior_50t_checkpoint/params`. `scripts/download_checkpoints.py --training-inits` fetches the remaining init checkpoints.

## Team checkpoints

The full routing of our team's submission is below; ✅ marks the tasks that use checkpoints from this repository. Rows without ✅ link to teammates' Hugging Face repos; their training code is being added to this repository. Tasks 13, 41, 49, 58 and 68 had no trained model.

<details>
<summary>Task → checkpoint table (100 tasks)</summary>

| Task | Checkpoint | |
|---:|---|---|
| 0 | [shi2bis/task0_turning_on_radio_cont5000](https://huggingface.co/shi2bis/task0_turning_on_radio_cont5000) | |
| 1 | [Rena-Tian/b1k-2026-task1-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task1-ft1) | |
| 2, 3, 6, 19, 28, 44, 47 | [fei777/b1k2026-pi05-ft-tasks-2-3-6-19-28-44-47 @73057b3](https://huggingface.co/fei777/b1k2026-pi05-ft-tasks-2-3-6-19-28-44-47/tree/73057b3d130c9ff6753bd39ba115b66fae9a382e) | |
| 4 | [Rena-Tian/b1k-2026-task4-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task4-ft1) | |
| 5 | [estjtmi/behavior-2026-pi05-15tasks / checkpoint_40000](https://huggingface.co/estjtmi/behavior-2026-pi05-15tasks/tree/main/checkpoint_40000) | |
| 7 | [RichardCTian/b1k-2026-task7-ft1](https://huggingface.co/RichardCTian/b1k-2026-task7-ft1) | |
| 8, 9 | [Rena-Tian/b1k-2026-task8-9-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task8-9-ft1) | |
| 10 | [RichardCTian/b1k-2026-task10-ft1](https://huggingface.co/RichardCTian/b1k-2026-task10-ft1) | |
| 11 | [RichardCTian/b1k-2026-task11-ft1](https://huggingface.co/RichardCTian/b1k-2026-task11-ft1) | |
| 12 | [RichardCTian/b1k-2026-task12-ft1](https://huggingface.co/RichardCTian/b1k-2026-task12-ft1) | |
| 14 | [Rena-Tian/b1k-2026-task14-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task14-ft1) | |
| 15 | [RichardCTian/b1k-2026-task15-ft1](https://huggingface.co/RichardCTian/b1k-2026-task15-ft1) | |
| 16 | [Rena-Tian/b1k-2026-task16-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task16-ft1) | |
| 17 | [RichardCTian/b1k-2026-task17-ft1](https://huggingface.co/RichardCTian/b1k-2026-task17-ft1) | |
| 18 | [RichardCTian/b1k-2026-task18-ft1](https://huggingface.co/RichardCTian/b1k-2026-task18-ft1) | |
| 20 | [Rena-Tian/b1k-2026-task20-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task20-ft1) | |
| 21 | [Rena-Tian/b1k-2026-task21-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task21-ft1) | |
| 22 | [Blakesss/task22-pi05-backup @ task22-official-final-backup-20261004-075738-UTC](https://huggingface.co/Blakesss/task22-pi05-backup/tree/task22-official-final-backup-20261004-075738-UTC) | |
| 23, 48 | [fei777/b1k2026-pi05-ft-tasks-4-23-48 @38a169c](https://huggingface.co/fei777/b1k2026-pi05-ft-tasks-4-23-48/tree/38a169cdd213de3d28ab0abfcab6343b9df31f72) | |
| 24 | [RichardCTian/b1k-2026-task24-ft1](https://huggingface.co/RichardCTian/b1k-2026-task24-ft1) | |
| 25 | [RichardCTian/b1k-2026-task25-ft1](https://huggingface.co/RichardCTian/b1k-2026-task25-ft1) | |
| 26 | [Rena-Tian/b1k-2026-task26-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task26-ft1) | |
| 27, 31 | [estjtmi/behavior-2026-five-winner-ckpt3-10k](https://huggingface.co/estjtmi/behavior-2026-five-winner-ckpt3-10k) | |
| 29 | [RichardCTian/b1k-2026-task29-ft1](https://huggingface.co/RichardCTian/b1k-2026-task29-ft1) | |
| 30 | [RichardCTian/b1k-2026-task30-ft1](https://huggingface.co/RichardCTian/b1k-2026-task30-ft1) | |
| 32 | [RichardCTian/b1k-2026-task32-ft1](https://huggingface.co/RichardCTian/b1k-2026-task32-ft1) | |
| 33, 36, 39 | [fei777/b1k2026-pi05-ft-tasks-33-36-39 @1441005](https://huggingface.co/fei777/b1k2026-pi05-ft-tasks-33-36-39/tree/1441005be9eb1825c1c5e95ec14d61b9e7a1e894) | |
| 34 | [Rena-Tian/b1k-2026-task34-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task34-ft1) | |
| 35, 63 | [shi2bis/joint_35_63_78](https://huggingface.co/shi2bis/joint_35_63_78) | |
| 37 | [jermo1120/b1k-task37-clean-a-trumpet-25k](https://huggingface.co/jermo1120/b1k-task37-clean-a-trumpet-25k) | |
| 38 | [jermo1120/pi05-b1k-task38-spraying-for-bugs-25k](https://huggingface.co/jermo1120/pi05-b1k-task38-spraying-for-bugs-25k) | |
| 40 | [RichardCTian/b1k-2026-task40-ft1](https://huggingface.co/RichardCTian/b1k-2026-task40-ft1) | |
| 42 | [Rena-Tian/b1k-2026-task42-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task42-ft1) | |
| 43 | [Rena-Tian/b1k-2026-task43-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task43-ft1) | |
| 45 | [Rena-Tian/b1k-2026-task45-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task45-ft1) | |
| 46 | [Rena-Tian/b1k-2026-task46-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task46-ft1) | |
| 50, 70, 72, 85 | [amyjia1212/b1k-2026-t50-70-72-85 / step_049999](https://huggingface.co/amyjia1212/b1k-2026-t50-70-72-85/tree/main/step_049999) | ✅ |
| 51, 61 | [fei777/b1k2026-pi05-ft-tasks-51-61 @0184638](https://huggingface.co/fei777/b1k2026-pi05-ft-tasks-51-61/tree/0184638e2ccad6f1fc9db5228df5c3b8581d8821) | |
| 52 | [amyjia1212/b1k-2026-t52 / step_008000](https://huggingface.co/amyjia1212/b1k-2026-t52/tree/main/step_008000) | ✅ |
| 53, 66, 73, 86 | [amyjia1212/b1k-2026-t52-53-73-66-86 / step_049999](https://huggingface.co/amyjia1212/b1k-2026-t52-53-73-66-86/tree/main/step_049999) | ✅ |
| 54 | [RichardCTian/b1k-2026-task54-putting-away-toys](https://huggingface.co/RichardCTian/b1k-2026-task54-putting-away-toys) | |
| 55 | [jermo1120/pi05-b1k-task55-re-shelving-library-books](https://huggingface.co/jermo1120/pi05-b1k-task55-re-shelving-library-books) | |
| 56, 78 | [Rena-Tian/b1k-2026-3more-ft1](https://huggingface.co/Rena-Tian/b1k-2026-3more-ft1) | |
| 57 | [Luke-Xing/task57b1k](https://huggingface.co/Luke-Xing/task57b1k) | |
| 59 | [RichardCTian/b1k-2026-task59-organizing-art-supplies](https://huggingface.co/RichardCTian/b1k-2026-task59-organizing-art-supplies) | |
| 60 | [shi2bis/joint_57_60_69_93](https://huggingface.co/shi2bis/joint_57_60_69_93) | |
| 62 | [RichardCTian/b1k-2026-task62-halve-an-egg](https://huggingface.co/RichardCTian/b1k-2026-task62-halve-an-egg) | |
| 64 | [RichardCTian/b1k-2026-task64-ft1](https://huggingface.co/RichardCTian/b1k-2026-task64-ft1) | |
| 65, 71, 74, 75 | [amyjia1212/b1k-2026-t65-71-74-75 / step_049999](https://huggingface.co/amyjia1212/b1k-2026-t65-71-74-75/tree/main/step_049999) | ✅ |
| 67 | [RichardCTian/b1k-2026-task67-ft1](https://huggingface.co/RichardCTian/b1k-2026-task67-ft1) | |
| 69 | [Rena-Tian/b1k-2026-task69-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task69-ft1) | |
| 76, 82, 92 | [fei777/b1k2026-pi05-ft-tasks-76-82-92 @73da80f](https://huggingface.co/fei777/b1k2026-pi05-ft-tasks-76-82-92/tree/73da80fe71101cd7d6a35ebd5ed861965dbfb39c) | |
| 77 | [hlhl188/b1k-2026-cluster-77-89 / runs/cluster_installing_devices_lora_v1/step_20000](https://huggingface.co/hlhl188/b1k-2026-cluster-77-89/tree/main/runs/cluster_installing_devices_lora_v1/step_20000) | |
| 79, 84, 88 | [amyjia1212/b1k-2026-t79-81-84-88 / step_049999](https://huggingface.co/amyjia1212/b1k-2026-t79-81-84-88/tree/main/step_049999) | ✅ |
| 80, 87 | [shi2bis/joint_76_80_87](https://huggingface.co/shi2bis/joint_76_80_87) | |
| 81 | [amyjia1212/b1k-2026-t81-ft10k / step_009999](https://huggingface.co/amyjia1212/b1k-2026-t81-ft10k/tree/main/step_009999) | ✅ |
| 83 | [angegiga/pi05-store-honey / checkpoints/from9500_90500](https://huggingface.co/angegiga/pi05-store-honey/tree/main/checkpoints/from9500_90500) | |
| 89 | [Rena-Tian/b1k-2026-task89-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task89-ft1) | |
| 90, 91, 93, 96, 97 | [amyjia1212/b1k-2026-t90-91-93-96-97 / step_099999](https://huggingface.co/amyjia1212/b1k-2026-t90-91-93-96-97/tree/main/step_099999) | ✅ |
| 94, 95, 98 | [amyjia1212/b1k-2026-t94-95-98 / step_049999](https://huggingface.co/amyjia1212/b1k-2026-t94-95-98/tree/main/step_049999) | ✅ |
| 99 | [Rena-Tian/b1k-2026-task99-ft1](https://huggingface.co/Rena-Tian/b1k-2026-task99-ft1) | |
| 13, 41, 49, 58, 68 | n/a (not trained) | |

</details>

## Reconstruction notes

| Component | Source | Status |
|---|---|---|
| Hyperparameters, task splits, init checkpoints, step counts, GPUs | W&B run configs and commands (19 runs) | **exact** |
| Stage-1 5-task model config (`t5_base`) | Tasks, steps and init from the HF repo and downstream configs; LR schedule **assumed** equal to stage 2 | partly assumed |
| `TASK_NUM_STAGES` for tasks 50–99 | Recomputed from dataset metadata; total of 1087 checked against checkpoint tensor shapes | verified |
| Embedding extension and position-embedding interpolation | Re-implemented from the training log messages; output shapes checked against checkpoints | re-implemented, unit-tested |
| LeRobot v3 reader | Re-implemented; states, actions and frames checked bit-exact against the raw 2026 dataset | re-implemented, tested |
| HF upload, W&B naming, data-worker restarts | Re-implemented from log messages | re-implemented |
| Eval settings (13/4/10, 20 flow steps, 280px, gripper 90/15) | Logged in each W&B eval run's config | **exact settings** |
| Gripper-recovery trigger condition | Only "after 90 stuck steps, hold open 15" was logged; the "closed on nothing" trigger is our reconstruction | approximate |
| "stagefix" | Our eval runs before and after a fix to stage handling for tasks ≥ 50. The 100-entry `TASK_NUM_STAGES` in this repo covers it | re-implemented |
| Our eval-side W&B logging script (`log_wandb.py`) | Lost; not needed to run evaluation | not included |

## Acknowledgements and license

Built on the [2025 BEHAVIOR Challenge 1st-place solution](https://github.com/IliaLarchenko/behavior-1k-solution) by Ilia Larchenko, [openpi](https://github.com/Physical-Intelligence/openpi) (via the [BEHAVIOR fork](https://github.com/wensi-ai/openpi)) and [BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K). Licensed under Apache-2.0, the same as the original solution ([LICENSE](LICENSE)).
