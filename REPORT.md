# Technical Report: Adapting PI_BEHAVIOR to the 100-task 2026 BEHAVIOR Challenge

*BEHAVIOR Challenge 2026 (CoRL 2026), standard track. Code: this repository.*

> **Scope.** Our team's submission combined models trained by several team members. This report covers the
> **PI_BEHAVIOR pipeline**, which produced the checkpoints for 25 of the 100 tasks (50, 52, 53, 65, 66, 70–75, 79, 81, 84–86,
> 88, 90, 91, 93–98). Section 8 gives a short overview of the other team members' approaches. All numbers are self-evaluations
> on public test instances, not official hidden-test results.

## 1. Summary

We started from **PI_BEHAVIOR**, the π0.5-based policy that won the 2025 BEHAVIOR Challenge. In 2025 it was trained on 50 tasks.
We extended it to the 2026 setting, which has 100 tasks, a new LeRobot v3 dataset and the BEHAVIOR-1K v3.9.3 evaluator, under a
small compute budget (single A100-80GB GPUs per run; 2-GPU FSDP for the last fine-tunes):

1. **Warm start** from a 2025 50-task PI_BEHAVIOR checkpoint. The task and stage embedding tables grow to 100 tasks (new rows
   start at the mean of the existing rows), and the SigLIP position embedding is interpolated from 224 px to **280 px** inputs.
2. **Curriculum by task groups.** We first trained a 5-task model for 100k steps, then five 3–5-task "chunk" models for 50k steps
   each, then short single-task fine-tunes for tasks where the chunk models were weak.
3. **Inference** kept the 2025 tricks (rolling inpainting, action compression, stage voting), with a faster control loop (replan
   every 10 env steps). We added a **gripper-recovery** rule: a gripper that has been stuck (fully closed, or blocked while
   commanded open) for 90 steps is forced open for 15 steps.

On the public instances 0–9, the 20 self-evaluated tasks reach a mean **Q-score of 0.105** (0 full successes). The best tasks are
`laying_tile_floors` (0.425), `putting_dirty_dishes_in_sink` (0.275; 0.625 and 0.400 in repeat runs), `thawing_frozen_food` (0.267),
`turning_out_all_lights_before_sleep` (0.220) and `tidying_bathroom` (0.200).

## 2. Background: PI_BEHAVIOR

PI_BEHAVIOR (Larchenko, Zarin and Karnatak, 2025) modifies π0.5 (PaliGemma 2B VLM and a 300M action expert, flow matching, 30-step action chunks over
the 23-D R1Pro action space):

* **No language prompt.** The task is given by a learned **task embedding**. The current **stage** of the task (episodes are split
  into 5–15 equal-time stages per task) is given by a task-specific stage embedding plus a sin/cos encoding. The VLM predicts the
  stage, and at inference the stage is tracked by majority voting.
* **Auxiliary losses:** FAST-token prediction (weight 0.05) and stage classification (0.1).
* **KV transform** between VLM and action-expert layers, **correlated flow noise** (from the action covariance, with shrinkage
  β = 0.5), per-timestep action normalisation, and delta joint actions.
* **Inference:** soft inpainting of the next chunk from the previous one, temporal compression (execute *k* predicted actions in
  fewer env steps using cubic interpolation), and task-specific correction rules.

We kept the architecture and losses unchanged, so differences come from the data, the initialisation and the training schedule.

## 3. Adapting to 2026

**Data.** `behavior-1k/2026-challenge-demos` has 20,000 teleoperated demos (200 per task, 30 fps, 3.27 TB) in the LeRobot v3
layout: episodes are packed into shared parquet and HEVC mp4 files. Our environment's `lerobot` (0.3.4) cannot read v3, so we wrote a
small standalone reader (`src/b1k/training/lerobot_v3.py`). It finds a task's episodes through `meta/episodes`, reads states and
actions from the shared parquet, and decodes frames with PyAV at `from_timestamp + t`. Its output is checked to be bit-exact against
an independent sequential decode. The state is the 61-D proprioception vector; as in 2025, we extract base velocity, trunk and arm
joint positions and normalised gripper widths (23-D).

**100 tasks.** Tasks 0–49 are the 2025 tasks, so we kept their stage counts. That lets 2025 checkpoints load unchanged. The new
tasks 50–99 get 10 stages each, except tasks 90 (5), 91 (11) and 93 (5). That adds 491 stage embeddings (1087 in total).
The stage table must match the checkpoint exactly. A different per-task split with the same total of 1087 still loads without an
error, but every task from the first difference on reads another task's rows. Our early evaluations had exactly this bug: they
used counts recomputed from demo lengths. We found and fixed it on 2026-10-14 ("stagefix"), and only the corrected evaluations
are reported here.

**Resolution.** All 2026 models use 280×280 inputs, up from 224. Because SigLIP patches are 14 px, this changes the token grid from
16×16 to 20×20. We bicubically interpolate the pretrained position embedding.

**Weight initialisation from 2025.** The 2025 checkpoint has 50 task embeddings and 596 stage embeddings. We append rows equal to the
mean of the existing ones, so new tasks start from an "average task" rather than random vectors.

## 4. Training

All runs use batch 16, AdamW (β = 0.9/0.95, weight decay 1e-10, gradient clip 1.0), cosine LR with 1k warm-up steps, EMA 0.99,
15 flow samples per step and a frozen vision backbone. Norm stats (with the action correlation matrix) and a FAST tokenizer
(1024 tokens, dims 0:6 and 7:23) are computed per config.

| Stage | Models | Init | Steps | Peak → final LR | Hardware |
|---|---|---|---|---|---|
| 1 | 5-task base: 90, 91, 93, 96, 97 | 2025 50-task ckpt | 50k + 50k* | 5e-5 → 5e-6* | 1× A100-80GB PCIe, bf16 Adam μ |
| 2 | chunks {50,70,72,85}, {52,53,73,66,86}, {65,71,74,75}, {79,81,84,88}, {94,95,98} | stage 1 | 50k | 5e-5 → 5e-6 | one 5× A100-80GB PCIe pod (one run per GPU), bf16 Adam μ |
| 3a | single-task 50, 52, 72 | 2025 ckpt | 10k | 3e-5 → 3e-6 | 1× A100 |
| 3b | single-task 53, 65, 66, 71, 74 | 2025 ckpt | 6k | 1e-4 → 1e-5 | 2× A100-SXM4 (FSDP) |
| 3c | single-task 79, 81, 86, 94 | stage-2 chunk | 10k | 1e-4 → 1e-5 | 2× A100-SXM4 (FSDP) |
| 3d | continued 52, 53 | stage 3a/3b | +10k | 1e-4 → 1e-5 | 2× A100-SXM4 (FSDP) |

\* The stage-1 W&B project was deleted; these settings come from our run notes. The schedule decays over the first 50k steps;
the model was then trained for another 50k steps, and the LR of that continuation is not recorded. Batch 32 did not fit
(about 109 GB of GPU memory), so we used batch 16 with the first Adam moment in bf16. The container's 117 GB RAM limit forced
us down to 4 data workers, because each checkpoint save copies about 44 GB to host memory.

Stage 2 was set up as an experiment: does fine-tuning from a model already trained on several 2026 tasks generalise better
than training each task from the 2025 checkpoint? The plan was batch 32 at 280 px per 80 GB GPU, stepping down (smaller batch,
remat, and lower resolution only as a last resort) if it ran out of memory. The logged runs use batch 16 at 280 px.
Before PI_BEHAVIOR, we also tried the plain openpi π0.5 baseline on tasks 90, 93 and 96 ([`early_experiments/`](early_experiments/)).

Choosing the init was the first issue we ran into. In the corrected (stagefix) evaluations, the single-task models fine-tuned
directly from the 2025 checkpoint (stage 3a/3b) scored 0 on 7 of their 8 tasks: 50, 53, 65, 66, 71, 72 and 74. Only
`sorting_bottles_cans_and_paper` (task 52) reached 0.05–0.075. The stage-2 chunk models, initialised from a model that had already
been trained on 2026 tasks at 280 px, scored higher on 5 of those 7 tasks (for example 0.220 vs 0 on task 66) and tied on the
other two. So in the final routing, most tasks use chunk models. Single-task fine-tunes were kept only where they matched or beat
the chunk model: task 52 at step 8k (0.075 vs 0.056) and task 81 (0.083 vs 0.000).

## 5. Inference

The policy server (`scripts/serve_b1k.py`) routes each task to its checkpoint (`task_checkpoint_mapping_2026.json`). Settings used
for every evaluation:

* 280×280 inputs. The head (720²) and wrist (480²) RGB images are resized with padding. No depth is used.
* 20 flow-matching steps; 30-action chunks, of which **13 are executed in 10 env steps** (1.3× speed-up, cubic interpolation, base
  velocities scaled) and **4 are kept** to inpaint the next chunk. Compression is switched off when the gripper commands vary a lot.
* Stage tracking by majority vote over the last 3 predictions, plus the 2025 correction rules.
* **Gripper recovery (new):** a gripper counts as stuck if its fingers are fully closed (closed on nothing), or if it is commanded
  at least 0.3 (normalised) wider than it is but has not moved. After 90 stuck env steps it is forced open for 15 steps, and the
  next chunk is not inpainted from the old plan, so the policy can grasp again.

## 6. Results

Self-evaluation on public instances 0–9, one rollout each, default 1.5× human-length timeout. Q-score is the fraction of goal
predicates satisfied. The full table is in [results/SELF_EVAL_2026.md](results/SELF_EVAL_2026.md) and the raw runs in
`results/selfeval_runs_2026.json`.

| Group | Tasks (Q-score) |
|---|---|
| chunk 1 | 50 freeze_fruit 0.014 · 70 thawing_frozen_food 0.267 · 72 cook_a_frozen_pie 0.000 · 85 putting_dirty_dishes_in_sink 0.275 |
| chunk 2 | 53 tidying_living_room 0.025 · 66 turning_out_all_lights 0.220 · 73 organizing_school_stuff 0.133 · 86 make_gift_bags 0.017 |
| chunk 3 | 65 unloading_the_car 0.000 · 71 clean_rusty_garden_tools 0.080 · 74 carrying_out_garden_furniture 0.050 · 75 pruning_kit 0.000 |
| chunk 4 | 79 polishing_shoes 0.067 · 84 tidying_bathroom 0.200 · 88 rearrange_your_room 0.067 |
| chunk 5 | 94 dispose_of_batteries 0.100 · 95 cook_brussels_sprouts 0.000 · 98 laying_tile_floors 0.425 |
| single | 52 sorting_bottles_cans_and_paper 0.075 · 81 packing_meal_for_delivery 0.083 |
| 5-task base | 90, 91, 93, 96, 97: no self-evaluation recorded |

Mean over the 20 evaluated tasks: **Q = 0.105**, success rate 0.00.

**Variance.** Repeating `putting_dirty_dishes_in_sink` with the same checkpoint and settings on different pods gave Q = 0.275, 0.400
and 0.625. Simulator nondeterminism is therefore of the same size as most of the differences between checkpoints, and we treat
differences below about 0.1 on 10 rollouts as noise.

## 7. Limitations and lessons

* **Compute.** The 2025 solution trained on 50 tasks with batch 2048 on 8 GPUs. We used batch 16 on single GPUs. With 200 demos
  per task and long-horizon tasks of about 6 minutes, our models were clearly undertrained, and no task reached a full success.
* **Pick the init carefully.** Fine-tuning a 2025 checkpoint directly on one new 2026 task did not work in our budget. An
  intermediate model trained on 2026 data at the new resolution was a much better starting point.
* **Make infrastructure reproducible.** Our modified code lived only on ephemeral pods and was lost when they were deleted.
  Checkpoints survived because we uploaded them to Hugging Face during training, and configs survived because W&B logged them. The
  code in this repository was rebuilt from those records, as described in the README. Next time the code would be pushed to git from
  the start.

## 8. Team overview

The final submission routed each task to the best checkpoint any team member had for it (95 tasks; 13, 41, 49, 58 and 68 had
no model). Almost every member fine-tuned the same 2025 PI_BEHAVIOR model, but with different 2026 adaptations:

* **50-slot reuse** (fei777, Rena-Tian, RichardCTian, estjtmi, jermo1120 for task 38, Blakesss). 2026 tasks 0–49 are the 2025
  tasks, so the 2025 task slots, stage tables and 224 px inputs were kept, starting from a 2025 checkpoint
  (for example `behavior_submission` checkpoint 1 or 2). fei777 also served new tasks (51, 61, 76, 82, 92) through borrowed 2025 slots that
  have no task-specific correction rules. Rena-Tian and RichardCTian used low-LR (5e-6) fine-tunes with skill-annotation stages for
  many tasks. Their reported Q-scores reach 0.60 (task 45), 0.57 (task 1) and 0.58 (task 22, Blakesss).
* **100-task extensions** like ours, starting from `behavior_50t_checkpoint`, but with different stage tables: 1105 stages
  (Rena-Tian and RichardCTian, for new tasks such as 54, 56, 62, 64, 67, 89, 99) and 627–673 stages (Scarlett/shi2bis, joint
  multi-task models).
* **Plain π0.5 (openpi)** for tasks 55 (jermo1120), 69 (Rena-Tian), 77 (hlhl188, LoRA) and 83 (angegiga, with a stuck-recovery
  wrapper; 3/10 successes).

Rena-Tian and RichardCTian's pipeline (code snapshot, diffs against upstream, scripts and runbook) is included as the
[`teammates/rena_richard`](teammates/rena_richard) submodule. The per-member details, code locations and reported scores are in
[`teammates/README.md`](teammates/README.md).

## References

* I. Larchenko, G. Zarin, A. Karnatak. *1st place solution, 2025 BEHAVIOR Challenge.* https://arxiv.org/abs/2512.06951, https://github.com/IliaLarchenko/behavior-1k-solution
* Physical Intelligence. *π0.5: a Vision-Language-Action Model with Open-World Generalization*, 2025. https://github.com/Physical-Intelligence/openpi
* C. Li et al. *BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation.* https://behavior.stanford.edu
