# Team code and checkpoints

Our 2026 submission routed each task to a checkpoint trained by one team member. Almost every member built on the
same base, the 2025 1st-place PI_BEHAVIOR solution (π0.5 with task-ID and stage embeddings), with their own 2026 data
pipeline and recipe. A few checkpoints are plain openpi π0.5. This folder collects the team members' code, or points to
where it is published.

**Columns:**
- **Model**: PIB = PI_BEHAVIOR; π0.5 = openpi π0.5 (language-conditioned). Table sizes come from the checkpoints' own metadata.
- **Init**: `50t` = [`IliaLarchenko/behavior_50t_checkpoint`](https://huggingface.co/IliaLarchenko/behavior_50t_checkpoint); `ckptN` = [`IliaLarchenko/behavior_submission`](https://huggingface.co/IliaLarchenko/behavior_submission) `checkpoint_N`.
- **Reported Q**: self-evaluation from each model card, using the leaderboard protocol (public instances 0–9, 1 rollout).
  The protocol is the same as ours, but the runs, pods and settings differ.

| Member (HF) | Tasks in submission | Model | Init | Code | Reported Q |
|---|---|---|---|---|---|
| amyjia1212 | 50, 52, 53, 65, 66, 70–75, 79, 81, 84–86, 88, 90, 91, 93–98 | PIB, 100 tasks, 1087 stages, 280 px | 50t | this repository (`src/`, `scripts/`) | see `results/SELF_EVAL_2026.md` |
| fei777 | 2, 3, 6, 19, 23, 28, 33, 36, 39, 44, 47, 48, 51, 61, 76, 82, 92 | PIB, 50 slots (new tasks borrow a 2025 slot), 224 px | ckpt1 / 50t | [`fei777/`](fei777/) (full tree + `uv.lock` + lerobot patch, from the HF model repos) | not in model cards |
| Rena-Tian | 1, 4, 8, 9, 14, 16, 20, 21, 26, 34, 42, 43, 45, 46, 56, 78, 89, 99 | PIB (50 slots, or 100 tasks with 1105 stages) | ckpt2 / 50t | [`rena_richard/`](rena_richard/) submodule ([Rena-Tian/b1k-2026-pipeline](https://github.com/Rena-Tian/b1k-2026-pipeline)). **Complete**: every submitted task maps to a config, init, step count and LR in [`CONFIGS_AFTER_2026-10-04.md`](https://github.com/Rena-Tian/b1k-2026-pipeline/blob/main/CONFIGS_AFTER_2026-10-04.md) | 0.03–0.60 (e.g. task 45: 0.60, task 1: 0.567, task 16: 0.50) |
| Rena-Tian | 69 | π0.5 (openpi, 3-task model for tasks 34/63/69) | pi05_base | `rena_richard/patches/openpi-pi05.wensi-ai.vs-0cc8e35.diff` (`pi05_b1k_3task_calib`, LoRA, against `wensi-ai/openpi@0cc8e35`); run details in [`CONFIGS_AFTER_2026-10-04.md`](https://github.com/Rena-Tian/b1k-2026-pipeline/blob/main/CONFIGS_AFTER_2026-10-04.md) | 0.00 |
| RichardCTian | 7, 10, 11, 12, 15, 17, 18, 24, 25, 29, 30, 32, 40, 54, 59, 62, 64, 67 | PIB (50 slots, or 100 tasks with 1105 stages) | ckpt2 / 50t | same [`rena_richard/`](rena_richard/) submodule ([Rena-Tian/b1k-2026-pipeline](https://github.com/Rena-Tian/b1k-2026-pipeline)). **Complete except task 54**: every other submitted task maps to a config, init, step count, submitted step and LR in [`CONFIGS_AFTER_2026-10-04.md`](https://github.com/Rena-Tian/b1k-2026-pipeline/blob/main/CONFIGS_AFTER_2026-10-04.md); the task-54 config is not pushed yet | 0.04–0.40 (e.g. task 40: 0.40, task 12: 0.367, task 30: 0.363) |
| shi2bis / shi3bis (Scarlett) | 0, 35, 60, 63, 80, 87 | PIB, 100 tasks, 627–673 stages, 224 px | 50t | [`scarlett/`](scarlett/) submodule ([shi3bis/1k-repo-scarlett](https://github.com/shi3bis/1k-repo-scarlett)). **Partial**: the pushed code reserves 627 stages, but the joint checkpoints have 649 / 656 / 673 | `eval_metrics.json` in HF repos |
| Blakesss | 22 | PIB | 2025 ckpt | archived in the HF repo ([`archives/current_policy.tar.zst`, patches, `RESTORE_OFFICIAL.md`](https://huggingface.co/Blakesss/task22-pi05-backup/tree/task22-official-final-backup-20261004-075738-UTC)) | 0.58 (2/10 successes) |
| jermo1120 | 38 | PIB (`pi_behavior_b1k_fast`, single task, 25k steps, LR 5e-6 → 5e-7; step 24999 submitted) | ckpt3 | [`jermo1120/task38/`](jermo1120/task38/) patches + 2026 serving adapter (from the [model card](https://huggingface.co/jermo1120/pi05-b1k-task38-spraying-for-bugs-25k)). **Complete** | 0.20 |
| jermo1120 | 55 | π0.5 (openpi, full fine-tune from pi05_base) | pi05_base | [`jermo1120/`](jermo1120/) patches (from the HF model repo) | 0.067 |
| jermo1120 | 37 | PIB (`pi_behavior_b1k_task37_clean_a_trumpet`, single task, 25k steps, LR 5e-6 → 5e-7; `step_24000` submitted) | ckpt3 | [`jermo1120/task37/`](jermo1120/task37/) patches + launch script (from the [model card](https://huggingface.co/jermo1120/b1k-task37-clean-a-trumpet-25k)). **Complete** | 0.00 (the task's only goal gives no partial credit) |
| angegiga | 83 | π0.5 (openpi, `wensi-ai/openpi` behavior branch + patches) | pi05_base | in the HF repo ([`openpi_patches/`, `rebuild.sh`, `SUBMISSION.md`](https://huggingface.co/angegiga/pi05-store-honey)) | 3/10 successes |
| estjtmi | 5, 27, 31 | PIB, 50 slots | ? | **missing** (no model card) | ? |
| Luke-Xing | 57 | PIB in 2025 slot 36 (shapes unchanged), 3 stages: LoRA 12k → LoRA 8k → + action expert 7.5k, batch 32, 1× RTX 5090 | ckpt3 | recipe, LRs and exact commands in the [model card](https://huggingface.co/Luke-Xing/task57b1k); **code missing**: the configs, `serve_b1k_2026.py` (slot mapping, torso cap) and data loader are not published | 0.10 (1/10; a repeat run scored 0/10) |
| hlhl188 | 77 | π0.5 + LoRA (`cluster_installing_devices_lora_v1`; trained jointly on 77 + 89, step 20000 submitted) | ckpt2 | [`helen_task77/`](helen_task77/) submodule ([hehe6-88/b1k-2026-task77](https://github.com/hehe6-88/b1k-2026-task77)). **Complete**: training config, code, model card, per-instance results | 0.2500 |

Tasks 13, 41, 49, 58 and 68 had no trained model.

**Provenance of the vendored code:**
- `fei777/` is the `code.tgz` tree uploaded with fei777's checkpoints for tasks 4/23/48, 51/61 and 76/82/92.
  The checkpoints for 2/3/6/19/28/44/46/47 and 33/36/39 were trained with the same tree except for `src/b1k/training/config.py`,
  and those two config versions are in `fei777/config_variants/`.
- `jermo1120/` holds the patches uploaded with `jermo1120/pi05-b1k-task55-re-shelving-library-books` (task 55), and in
  `task37/` and `task38/` the files uploaded with the task-37 and task-38 checkpoints (copied unchanged).
- `scarlett/` is a git submodule pinned to `shi3bis/1k-repo-scarlett@4b2436f`.
- `helen_task77/` is a git submodule pinned to `hehe6-88/b1k-2026-task77@042588a`: the task-77 training config, openpi π0.5 + LoRA
  training/serving/sweep scripts, dataset converter, model card and the per-instance results behind q = 0.25.
- `rena_richard/` is a git submodule pinned to `Rena-Tian/b1k-2026-pipeline@64de0a3`: a snapshot of the 2025 winner's
  code with their 2026 changes (LeRobot v3 loader, 1105-stage table, skill-annotation stage labels, skill oversampling,
  every per-task config through the 10/16 deadline), exact diffs against upstream, the openpi π0.5 diff for task 69,
  train/eval/sweep scripts, their runbook `CLAUDE.md`, and a task → config map (`CONFIGS_AFTER_2026-10-04.md`).

## Still needed (status as of 2026-11-10)

1. RichardCTian: push the config (and init, steps, LR) for task 54. Everything else is done.
2. estjtmi (tasks 5, 27, 31): code, configs, and ideally a model card.
   Luke-Xing (57): the model card is done; still needed is the code it describes (`src/`, `scripts/` with the three
   `pi_behavior_sweep_*` configs and `serve_b1k_2026.py`), e.g. the `policy_code/` folder from the submission zip.
3. Scarlett: push the code used for the `joint_*` checkpoints, and say which sub-checkpoint of each `joint_*` repo was submitted
   (the repos hold several, e.g. `joint_35_63_78_noise_step22500` and `joint_35_63_78_pickup_step27500`).
