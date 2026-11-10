"""Download our 2026 checkpoints (and optionally the init checkpoints needed to re-train) from Hugging Face.

Each repo stores checkpoints as `step_XXXXXX/{params,assets}`; only the requested steps are downloaded.

    uv run scripts/download_checkpoints.py                 # the 8 checkpoints used for evaluation
    uv run scripts/download_checkpoints.py --training-inits  # + the init checkpoints referenced by configs_2026.py
"""

import argparse
import os

from huggingface_hub import snapshot_download

HF_USER = "amyjia1212"

# repo name -> steps (the eval checkpoints listed in task_checkpoint_mapping_2026.json)
EVAL_CHECKPOINTS = {
    "b1k-2026-t50-70-72-85": ["step_049999"],
    "b1k-2026-t52": ["step_008000"],
    "b1k-2026-t52-53-73-66-86": ["step_049999"],
    "b1k-2026-t65-71-74-75": ["step_049999"],
    "b1k-2026-t79-81-84-88": ["step_049999"],
    "b1k-2026-t81-ft10k": ["step_009999"],
    "b1k-2026-t90-91-93-96-97": ["step_099999"],
    "b1k-2026-t94-95-98": ["step_049999"],
}

# extra checkpoints used only as initialisation by the stage-3 configs
TRAINING_INITS = {
    "b1k-2026-t52": ["step_009999"],
    "b1k-2026-t53": ["step_005999"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default=os.environ.get("B1K_MODELS_DIR", "~/models"))
    ap.add_argument("--training-inits", action="store_true", help="also download init checkpoints for re-training")
    args = ap.parse_args()
    models_dir = os.path.expanduser(args.models_dir)

    wanted = {k: list(v) for k, v in EVAL_CHECKPOINTS.items()}
    if args.training_inits:
        for k, v in TRAINING_INITS.items():
            wanted.setdefault(k, []).extend(v)

    for repo, steps in wanted.items():
        local = os.path.join(models_dir, repo)
        print(f"{HF_USER}/{repo}: {steps} -> {local}")
        snapshot_download(
            repo_id=f"{HF_USER}/{repo}",
            local_dir=local,
            allow_patterns=[f"{s}/**" for s in steps],
        )

    if args.training_inits:
        print(
            "\nNOTE: the stage-0 init `behavior_50t_checkpoint` (2025 PI_BEHAVIOR model on the 50 tasks of the 2025 "
            "challenge) must be placed at "
            f"{os.path.join(models_dir, 'behavior_50t_checkpoint', 'params')}. See README, 'Training from scratch'."
        )


if __name__ == "__main__":
    main()
