---
license: apache-2.0
tags:
- behavior-1k
- robotics
---

# BEHAVIOR 2026 eval code and answers

Code and settings used to evaluate the `amyjia1212/b1k-2026-*` checkpoints, with answers to three questions about them.
As of 2026-10-14.

## Q1: How do camera images become 280x280?

The simulator renders full-resolution images, and the policy server downscales them to 280x280 with plain bilinear
interpolation.

| Camera | Rendered (official eval resolution) | Fed to the model |
| --- | --- | --- |
| Head (zed) | 720x720 RGB | 280x280 |
| Left wrist (realsense) | 480x480 RGB | 280x280 |
| Right wrist (realsense) | 480x480 RGB | 280x280 |

- The evaluator uses the official camera resolutions, RGB only (no depth), through `RGBFullResWrapper`
  (`b1k_eval/b1k_eval_wrappers.py`). Full-resolution frames are sent to the policy server.
- On the server, `B1KPolicyWrapper.process_obs` (`src/b1k/shared/eval_b1k_wrapper.py`) resizes each camera with
  `openpi_client.image_tools.resize_with_pad(img, 280, 280)`. That is PIL bilinear. All three cameras are square, so
  no padding is added.
- 280x280 is what the checkpoints expect: their SigLIP position embedding has 400 = 20x20 patches.

## Q2: Stage counts per task

Yes: tasks 90 and 91 share 16 rows as 5 for task 90 and 11 for task 91. The full table:

| Tasks | Stages each | Rows of `task_stage_embeddings` |
| --- | --- | --- |
| 0-49 | 2025 counts, unchanged | 0-595 (596 rows) |
| 50-89 | 10 | 596-995 |
| 90 | 5 | 996-1000 |
| 91 | 11 | 1001-1011 |
| 92 | 10 | 1012-1021 |
| 93 | 5 | 1022-1026 |
| 94-99 | 10 | 1027-1086 |

The total is 1087 rows, which matches the checkpoints' `task_stage_embeddings` table (1087 x 1024).

**Our earlier eval results are invalid.** Until 2026-10-14 the eval code used per-task counts reconstructed from demo
lengths: round(mean episode length / 900), clipped to 5-15. That table also sums to 1087, so the checkpoints loaded
without error. But it reads every task from 4 on at the wrong rows: for example, task 52 used rows 601-611 (11
stages) instead of 616-625 (10). The code here has the corrected table (`src/b1k/models/pi_behavior_config.py`).
All affected evals are being rerun, and their results are tagged `-stagefix`.

## Q3: Code and the policy server command

| Component | Source | Commit |
| --- | --- | --- |
| Policy repo | [IliaLarchenko/behavior-1k-solution](https://github.com/IliaLarchenko/behavior-1k-solution) (Apache-2.0) | `ca556f74a455cef7987a2be4537b5ac85cc56dd7` |
| openpi submodule | as pinned by that repo | `01177e0242a1c7e8fad2547caa0e987def614cda` |
| Simulator / evaluator | [StanfordVL/BEHAVIOR-1K](https://github.com/StanfordVL/BEHAVIOR-1K) | tag `v3.9.3-post1` |

Files in this repo:

| File | What it is |
| --- | --- |
| `b1k-eval-code-20261014.tar.gz` | Full policy source as it runs, including the openpi submodule, all local changes and the corrected stage table |
| `behavior-1k-solution.diff` | Our changes to tracked files of the policy repo at the commit above |
| `openpi.diff` | Our changes to the openpi submodule |
| `b1k_eval-harness.tar.gz` | Eval harness: `run_all.py` (orchestration), `eval_multi_ckpt.py` (evaluator client), `b1k_eval_wrappers.py` (cameras) |

The two new files `scripts/serve_b1k_2026.py` (the policy server) and `src/b1k/shared/eval_2026_compat.py` are not
in the diff, because git does not track them yet; they are in the source tarball.

Policy server command, one server per GPU, run from the policy repo root. Example: single-task model 52, checkpoint
`step_009999`, on GPU 2:

```bash
CUDA_VISIBLE_DEVICES=<GPU UUID> XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 XLA_PYTHON_CLIENT_ALLOCATOR=platform \
taskset -c 24-31 .venv/bin/python scripts/serve_b1k_2026.py \
  --checkpoint-dir /dev/shm/ckpts/b1k-2026-t52 --checkpoints step_009999 --port 8102 \
  --staging-dir /dev/shm/ckpt_gpu2 \
  --actions-to-execute 13 --execute-in-n-steps 10 \
  --gripper-recovery-steps 90 --gripper-recovery-hold-steps 15
```

- `--checkpoint-dir` holds the step directories downloaded from `amyjia1212/<model>`. `--checkpoints` takes several
  of them; the server keeps all of them in host memory and switches on `GET /switch?ckpt=<name>`.
- Inference settings: replan every 10 env steps, executing 13 of each 30-action prediction (the 2025 1.3x speedup),
  with inpainting. Gripper recovery forces a gripper open for 15 steps after 90 stuck steps. "No gripper recovery"
  runs pass `--gripper-recovery-steps 0`.
- The evaluator client is `eval_multi_ckpt.py`. It wraps `omnigibson.eval`'s `BatchedEvaluator` with the official
  public-test instances (indices 0-9 = ids 301-310), the `RGBFullResWrapper` cameras and the default 1.5x
  human-length timeout. `run_all.py` (function `run_wave`) builds its exact command line.
