"""Log BEHAVIOR eval results to W&B (project b1k-eval).

One run per checkpoint (id <model>-<ckpt>, resumed across calls) plus one summary run.
--task <name> : log that task's metrics, raw result JSON files and videos into every checkpoint's run (once per task).
--finalize    : write overall means into each checkpoint run's summary and (re)build the summary run with
                metrics vs. training step.
"""

import argparse
import glob
import json
import os

import wandb

from run_all import CHECKPOINTS, INFERENCE, INSTANCE_OFFSET, MODEL, NUM_INSTANCES, RESULTS_ROOT, RUN_TAG, TASKS, step_of

PROJECT = "b1k-eval"
INSTANCE_TAG = RUN_TAG  # run-name/group suffix: inference settings + instance set
EVAL_CONFIG = {
    "model_repo": f"amyjia1212/{MODEL}",
    "tasks": TASKS,
    "instances": f"public_test indices {INSTANCE_OFFSET}-{INSTANCE_OFFSET + NUM_INSTANCES - 1}, 1 rollout each",
    "evaluator": "BEHAVIOR-1K v3.9.3-post1 omnigibson.eval.eval (default 1.5x human-length timeout)",
    "env_wrapper": "RGBFullResWrapper (head 720x720, wrists 480x480, RGB; resized to 280x280 by server)",
    "policy": f"PiBehavior (behavior-1k-solution) - soft inpainting, 1.3x cubic speedup, 20 flow steps, {INFERENCE}",
}


def load_results(ckpt):
    rows = []
    for task in TASKS.values():
        for path in sorted(glob.glob(os.path.join(RESULTS_ROOT, ckpt, task, "json", "*.json"))):
            r = json.load(open(path))
            video = os.path.join(RESULTS_ROOT, ckpt, task, "videos", os.path.basename(path)[:-5] + ".mp4")
            r["video_path"] = video if os.path.exists(video) else None
            rows.append(r)
    return rows


def q(r):
    return float(r.get("q_score", {}).get("final", 0.0))


def aggregate(rows):
    metrics = {}
    for task in TASKS.values():
        task_rows = [r for r in rows if r["task"] == task]
        if task_rows:
            metrics[f"q_score/{task}"] = sum(map(q, task_rows)) / len(task_rows)
            metrics[f"success_rate/{task}"] = sum(bool(r["success"]) for r in task_rows) / len(task_rows)
            metrics[f"num_rollouts/{task}"] = len(task_rows)
    if rows:
        metrics["q_score/mean"] = sum(map(q, rows)) / len(rows)
        metrics["success_rate/mean"] = sum(bool(r["success"]) for r in rows) / len(rows)
        metrics["num_rollouts/total"] = len(rows)
    return metrics


def scalar_items(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            yield from scalar_items(v, f"{prefix}{k}.")
        elif isinstance(v, (int, float, bool)):
            yield f"{prefix}{k}", v


def init_checkpoint_run(ckpt):
    return wandb.init(
        project=PROJECT,
        name=f"{MODEL}/{ckpt}{INSTANCE_TAG}",
        id=f"{MODEL}-{ckpt}{INSTANCE_TAG}",
        resume="allow",
        group=MODEL + INSTANCE_TAG,
        job_type="eval-checkpoint",
        config={**EVAL_CONFIG, "checkpoint": ckpt, "train_step": step_of(ckpt)},
    )


def log_task(task):
    for ckpt in CHECKPOINTS:
        marker = os.path.join(RESULTS_ROOT, ckpt, task, ".wandb_logged")
        rows = [r for r in load_results(ckpt) if r["task"] == task]
        if not rows or os.path.exists(marker):
            continue
        run = init_checkpoint_run(ckpt)
        metrics = {k: v for k, v in aggregate(rows).items() if k.endswith(f"/{task}")}
        # No rollout table (user, 2026-10-14): the evaluator's JSON files are uploaded unchanged as run files
        # (<task>/json/<name>.json) and each video is logged as its own media entry.
        run.log(metrics)
        for r in rows:
            name = f"{r['task']}_{r['instance_id']}_{r['rollout_id']}"
            run.save(os.path.join(RESULTS_ROOT, ckpt, task, "json", name + ".json"), base_path=os.path.join(RESULTS_ROOT, ckpt), policy="now")
            if r["video_path"]:
                run.log({f"videos/{task}/{r['instance_id']}": wandb.Video(r["video_path"], format="mp4")})
        run.summary.update(metrics)
        run.finish()
        open(marker, "w").close()
        print(f"Logged {task} for {ckpt}: q_score={metrics[f'q_score/{task}']:.3f} ({len(rows)} rollouts)")


def finalize():
    for ckpt in CHECKPOINTS:
        rows = load_results(ckpt)
        if not rows:
            continue
        run = init_checkpoint_run(ckpt)
        run.summary.update(aggregate(rows))
        run.finish()


def log_summary():
    run = wandb.init(
        project=PROJECT,
        name=f"{MODEL}/summary{INSTANCE_TAG}",
        id=f"{MODEL}-summary{INSTANCE_TAG}",
        resume="allow",
        group=MODEL + INSTANCE_TAG,
        job_type="eval-summary",
        config=EVAL_CONFIG,
    )
    run.define_metric("train_step")
    run.define_metric("*", step_metric="train_step")
    table = wandb.Table(columns=["checkpoint", "train_step", *[f"q_score/{t}" for t in TASKS.values()], "q_score/mean", "success_rate/mean", "num_rollouts"])
    for ckpt in CHECKPOINTS:
        rows = load_results(ckpt)
        if not rows:
            continue
        m = aggregate(rows)
        run.log({"train_step": step_of(ckpt), **m})
        table.add_data(ckpt, step_of(ckpt), *[m.get(f"q_score/{t}") for t in TASKS.values()], m["q_score/mean"], m["success_rate/mean"], len(rows))
    run.log({"checkpoint_summary": table})
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    os.environ.setdefault("WANDB_API_KEY", open("/root/b1k_eval/.wandb_key").read().strip())
    if args.task:
        log_task(args.task)
    if args.finalize:
        finalize()
        log_summary()
