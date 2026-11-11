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
| RichardCTian | 7, 10, 11, 12, 15, 17, 18, 24, 25, 29, 30, 32, 40, 54, 59, 62, 64, 67 | PIB (50 slots, or 100 tasks with 1105 stages) | ckpt2 / 50t | same [`rena_richard/`](rena_richard/) submodule ([Rena-Tian/b1k-2026-pipeline](https://github.com/Rena-Tian/b1k-2026-pipeline)). **Complete**: every submitted task maps to a config, init, step count, submitted step and LR in [`CONFIGS_AFTER_2026-10-04.md`](https://github.com/Rena-Tian/b1k-2026-pipeline/blob/task54-config/CONFIGS_AFTER_2026-10-04.md); task 54 (`pi_behavior_b1k_2026_task54`, 50t init, 10k steps, batch 16, peak LR 5e-5) is on the `task54-config` branch | 0.04–0.40 (e.g. task 40: 0.40, task 12: 0.367, task 30: 0.363) |
| shi2bis / shi3bis (Scarlett) | 0, 35, 60, 63, 80, 87 | PIB, 100 tasks, 627–673 stages, 224 px | 50t | [`scarlett/`](scarlett/) submodule ([shi3bis/1k-repo-scarlett](https://github.com/shi3bis/1k-repo-scarlett)). **Complete**: one branch per run family, each with the matching stage table; submitted sub-checkpoints and run lineage in [Scarlett's runs](#scarletts-runs) below | `eval_metrics.json` in HF repos |
| Blakesss | 22 | PIB | 2025 ckpt | archived in the HF repo ([`archives/current_policy.tar.zst`, patches, `RESTORE_OFFICIAL.md`](https://huggingface.co/Blakesss/task22-pi05-backup/tree/task22-official-final-backup-20261004-075738-UTC)) | 0.58 (2/10 successes) |
| jermo1120 | 38 | PIB (`pi_behavior_b1k_fast`, single task, 25k steps, LR 5e-6 → 5e-7; step 24999 submitted) | ckpt3 | [`jermo1120/task38/`](jermo1120/task38/) patches + 2026 serving adapter (from the [model card](https://huggingface.co/jermo1120/pi05-b1k-task38-spraying-for-bugs-25k)). **Complete** | 0.20 |
| jermo1120 | 55 | π0.5 (openpi, full fine-tune from pi05_base) | pi05_base | [`jermo1120/`](jermo1120/) patches (from the HF model repo) | 0.067 |
| jermo1120 | 37 | PIB (`pi_behavior_b1k_task37_clean_a_trumpet`, single task, 25k steps, LR 5e-6 → 5e-7; `step_24000` submitted) | ckpt3 | [`jermo1120/task37/`](jermo1120/task37/) patches + launch script (from the [model card](https://huggingface.co/jermo1120/b1k-task37-clean-a-trumpet-25k)). **Complete** | 0.00 (the task's only goal gives no partial credit) |
| angegiga | 83 | π0.5 (openpi, `wensi-ai/openpi` behavior branch + patches) | pi05_base | in the HF repo ([`openpi_patches/`, `rebuild.sh`, `SUBMISSION.md`](https://huggingface.co/angegiga/pi05-store-honey)) | 3/10 successes |
| estjtmi (Thomas) | 5, 27, 31 | PIB, 50 slots (27, 31: 5-task model 4/27/31/32/35 from ckpt3, 10k steps) | ckpt3 (27, 31); ? (5) | **Partial**: fork [estjtmi/behavior-1k-solution](https://github.com/estjtmi/behavior-1k-solution) has his serving/eval changes (commit `6750a2b` on top of the winners' `ca556f7`) and the 27/31 eval runs (branch [`eval/ckpt3-10k-tasks-31-27-20261005`](https://github.com/estjtmi/behavior-1k-solution/tree/eval/ckpt3-10k-tasks-31-27-20261005)); the training configs for both models are not pushed | 27: 0.1875, 31: 0.15 (his eval on 301–310) |
| Luke-Xing | 57 | PIB in 2025 slot 36 (shapes unchanged), 3 stages: LoRA 12k → LoRA 8k → + action expert 7.5k, batch 32, 1× RTX 5090 | ckpt3 | recipe, LRs and exact commands in the [model card](https://huggingface.co/Luke-Xing/task57b1k); **code missing**: the configs, `serve_b1k_2026.py` (slot mapping, torso cap) and data loader are not published | 0.10 (1/10; a repeat run scored 0/10) |
| hlhl188 | 77 | π0.5 + LoRA (`cluster_installing_devices_lora_v1`; trained jointly on 77 + 89, step 20000 submitted) | ckpt2 | [`helen_task77/`](helen_task77/) submodule ([hehe6-88/b1k-2026-task77](https://github.com/hehe6-88/b1k-2026-task77)). **Complete**: training config, code, model card, per-instance results | 0.2500 |

Tasks 13, 41, 49, 58 and 68 had no trained model.

**Provenance of the vendored code:**
- `fei777/` is the `code.tgz` tree uploaded with fei777's checkpoints for tasks 4/23/48, 51/61 and 76/82/92.
  The checkpoints for 2/3/6/19/28/44/46/47 and 33/36/39 were trained with the same tree except for `src/b1k/training/config.py`,
  and those two config versions are in `fei777/config_variants/`.
- `jermo1120/` holds the patches uploaded with `jermo1120/pi05-b1k-task55-re-shelving-library-books` (task 55), and in
  `task37/` and `task38/` the files uploaded with the task-37 and task-38 checkpoints (copied unchanged).
- `scarlett/` is a git submodule pinned to `shi3bis/1k-repo-scarlett@4b2436f` (`main`). The code for each submitted model is
  on its own branch of that repo; see [Scarlett's runs](#scarletts-runs).
- `helen_task77/` is a git submodule pinned to `hehe6-88/b1k-2026-task77@042588a`: the task-77 training config, openpi π0.5 + LoRA
  training/serving/sweep scripts, dataset converter, model card and the per-instance results behind q = 0.25.
- `rena_richard/` is a git submodule pinned to `Rena-Tian/b1k-2026-pipeline@f50bce2` (branch `task54-config`, one commit ahead of `main`): a snapshot of the 2025 winner's
  code with their 2026 changes (LeRobot v3 loader, 1105-stage table, skill-annotation stage labels, skill oversampling,
  every per-task config through the 10/16 deadline), exact diffs against upstream, the openpi π0.5 diff for task 69,
  train/eval/sweep scripts, their runbook `CLAUDE.md`, and a task → config map (`CONFIGS_AFTER_2026-10-04.md`).

## Still needed (status as of 2026-11-10)

1. Thomas (estjtmi, tasks 5, 27, 31): his serving/eval code is on GitHub, but not the training setup. Needed for each of
   his two models (`behavior-2026-pi05-15tasks` → task 5, `behavior-2026-five-winner-ckpt3-10k` → 27, 31): the training
   config (or the `config.py` diff), the train command, and init, task list, steps, batch and LR; plus the task-5 score.
2. Luke-Xing (57): the model card is done; still needed is the code it describes (`src/`, `scripts/` with the three
   `pi_behavior_sweep_*` configs and `serve_b1k_2026.py`), e.g. the `policy_code/` folder from the submission zip.

## Scarlett's runs

All start from `IliaLarchenko/behavior_50t_checkpoint` with the task embeddings grown from 50 to 100 rows (new rows set
to the mean of the trained rows) and the stage table grown as below. Common settings: batch 32, RGB only, Ilia's
per-timestep norm stats, FAST auxiliary loss 0.05, stage loss 0.1, correlated flow-matching noise; LR warmup over about
10% of each run to 2e-5, then cosine to 2e-6. Joint runs draw mixed batches from the union of their tasks' episodes.
"noise" = proprioceptive state noise; "pickup" = oversampling of pick-up frames for task 63 (approach ×4, lift ×3,
retry episodes ×3).

**Stage tables** are cumulative; new rows are inserted in task-index order, `ceil(mean episode length / 900)` without
the minimum-5 clamp. Each branch's `src/b1k/models/pi_behavior_config.py` has the matching table.

| Table | New tasks (stages) | Used by |
|---:|---|---|
| 627 | 596 original + 55 (11), 76 (11), 82 (9) | task 0 |
| 649 | + 80 (10), 87 (12) | `joint_76_80_87` |
| 656 | + 63 (3), 78 (4) | `joint_35_63_78` |
| 673 | + 57 (5), 60 (4), 69 (3), 93 (5) | `joint_57_60_69_93` |

**Submitted checkpoints and how they were trained:**

| Tasks | Submitted sub-checkpoint | Code (branch @ head) | Lineage |
|---|---|---|---|
| 0 | `task0_turning_on_radio_cont5000` | [`task0-radio-2000`](https://github.com/shi3bis/1k-repo-scarlett/tree/task0-radio-2000) @ `a6bbd76` | 627 rows, no state noise, 7,000 steps total: 2,000 from the extended base (warmup 200); a continuation that crashed at its step-1000 save (its step-500 save survived); 4,500 more from that step-500 save |
| 35, 63 | `joint_35_63_78_pickup_step27500` | [`pickup-63`](https://github.com/shi3bis/1k-repo-scarlett/tree/pickup-63) @ `448ffc4` (earlier stages: [`attach-35-63-78`](https://github.com/shi3bis/1k-repo-scarlett/tree/attach-35-63-78) @ `7bef7b4`) | `step4500`: 4,500 steps from the 656-row base → `noise_step7500`: +3,000 with state noise → `noise_step22500`: +15,000 (warmup 1,500; `noise_step17500` is a mid-run push) → `pickup_step27500`: +5,000 with pickup oversampling (warmup not recorded) |
| 60 | `joint_57_60_69_93_step11000` | [`covered-57-60-69-93`](https://github.com/shi3bis/1k-repo-scarlett/tree/covered-57-60-69-93) @ `1875a17` | `step6000`: 6,000 steps from the 673-row base with state noise → `step11000`: +5,000 (warmup 500) |
| 80, 87 | `joint_76_80_87_step9000` | [`task76-resume-2000-lowlr`](https://github.com/shi3bis/1k-repo-scarlett/tree/task76-resume-2000-lowlr) @ `ae8a4dd` | `step4500`: 4,500 steps from the 649-row base (warmup 450) → `step9000`: +4,500 with LR re-warmed. No state noise. (`step3000` is probably a mid-run save; not recorded.) |

Not submitted: `shi2bis/task0_turning_on_radio_statenoise` (3,000 steps from `cont5000` with state noise, then 10,000 more,
warmup 1,000, pushed at step 18000 and 20000). Scarlett notes the pickup oversampling did not help.
