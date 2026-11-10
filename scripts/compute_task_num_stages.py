"""Recompute TASK_NUM_STAGES for the 2026 tasks from the dataset metadata.

Tasks 0-49 keep the 2025 values (so 2025 checkpoints load unchanged). For tasks 50-99 we used
floor(mean episode length / 900) clamped to [5, 15], which gives 491 extra stage embeddings
(596 + 491 = 1087, matching the task_stage_embeddings table in our 2026 checkpoints).

    uv run scripts/compute_task_num_stages.py --root ~/data/2026-challenge-demos
    (only meta/episodes/** is needed:  huggingface-cli download behavior-1k/2026-challenge-demos \
        --repo-type dataset --include "meta/**" --local-dir ~/data/2026-challenge-demos)
"""

import argparse
import os

from b1k.models.pi_behavior_config import TASK_NUM_STAGES
from b1k.training.lerobot_v3 import load_episodes_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.environ.get("B1K_DATA_ROOT", "~/data/2026-challenge-demos"))
    args = ap.parse_args()
    meta = load_episodes_meta(os.path.expanduser(args.root))
    avg = meta.groupby("task_index")["length"].mean()
    new = [max(5, min(15, int(avg[t] / 900))) for t in range(50, 100)]
    print("tasks 50-99:", new)
    print("sum 50-99:", sum(new), "| total with 2025 tasks:", sum(TASK_NUM_STAGES[:50]) + sum(new))
    print("matches pi_behavior_config.TASK_NUM_STAGES:", tuple(new) == tuple(TASK_NUM_STAGES[50:]))


if __name__ == "__main__":
    main()
