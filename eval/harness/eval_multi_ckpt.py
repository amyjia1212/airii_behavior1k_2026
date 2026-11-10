"""Evaluate several checkpoints on one task while loading the (10-env) scene only once.

Same flow as ``python -m omnigibson.eval.eval`` (seeding, instance resolution, BatchedEvaluator, default 1.5x
human-length timeout, one rollout per instance), except that before each pass the policy server is asked to
switch checkpoints (GET /switch?ckpt=<name>). Every checkpoint therefore sees exactly the same freshly
re-loaded instances that ``--num-rollouts`` would re-run.

Results: <results_root>/<ckpt>/<task>/{json,videos}/<task>_<instance_id>_0.{json,mp4}
Checkpoints whose result JSONs already exist for all of this process's instances are skipped.

With --phase-dir, several of these processes (each owning a slice of the instances) share one policy server:
the orchestrator switches the server's checkpoint, then creates <phase-dir>/go_<ckpt>; each process runs that
checkpoint and creates <phase-dir>/done_<ckpt>_<rank>.
"""

import argparse
import faulthandler
import glob
import logging
import os
import signal
import time
import urllib.request

from omegaconf import OmegaConf

from omnigibson.eval.evaluator import BatchedEvaluator, resolve_instance_ids
from omnigibson.eval.utils.eval_utils import DEFAULT_EVAL_SEED, seed_everything
from omnigibson.macros import gm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("eval_multi_ckpt")
faulthandler.register(signal.SIGUSR1, all_threads=True)  # `kill -USR1 <pid>` dumps all stacks (py-spy is blocked here)


def bind_render_gpu(gpu):
    """Render on GPU `gpu` (Vulkan/nvidia-smi index) while CUDA_VISIBLE_DEVICES exposes only that GPU as cuda:0.

    OmniGibson's own OMNIGIBSON_GPU_ID sets the renderer and PhysX to the same index, which is wrong once CUDA sees a
    single device, so the two SimulationApp options are set separately here.
    """
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"  # as omnigibson.simulator does right before its own import
    import isaacsim

    app_cls = isaacsim.SimulationApp

    def launch(config, *a, **kw):
        return app_cls({**config, "active_gpu": gpu, "physics_gpu": 0}, *a, **kw)

    isaacsim.SimulationApp = launch


def switch_checkpoint(host, port, ckpt):
    t0 = time.time()
    with urllib.request.urlopen(f"http://{host}:{port}/switch?ckpt={ckpt}", timeout=3600) as r:
        body = r.read().decode()
        if r.status != 200:
            raise RuntimeError(f"switch to {ckpt} failed: {body}")
    logger.info(f"Policy server switched to {ckpt} in {time.time() - t0:.1f}s ({body.strip()})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--results-root", required=True)
    parser.add_argument("--instance-indices", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--env-wrapper", default="b1k_eval_wrappers.RGBFullResWrapper")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--no-switch", action="store_true", help="Serve whatever the server has loaded (debugging).")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--phase-dir", default=None, help="Barrier directory shared with the orchestrator.")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--action-chunk-size", type=int, default=20,
        help="Client-side replay of the server's 20-step open-loop chunk (identical actions, fewer round trips).",
    )
    args = parser.parse_args()

    def out_dirs(ckpt):
        base = os.path.join(args.results_root, ckpt, args.task_name)
        return os.path.join(base, "json"), os.path.join(base, "videos")

    instance_ids = resolve_instance_ids(args.task_name, args.instance_indices, mode="public_test")

    def ckpt_done(ckpt):
        json_dir = out_dirs(ckpt)[0]
        return all(os.path.exists(os.path.join(json_dir, f"{args.task_name}_{i}_0.json")) for i in instance_ids)

    if all(ckpt_done(c) for c in args.checkpoints):
        logger.info("All checkpoints already evaluated for these instances")
        if args.phase_dir:
            for ckpt in args.checkpoints:
                open(os.path.join(args.phase_dir, f"done_{ckpt}_{args.rank}"), "w").close()
        return

    if os.environ.get("B1K_RENDER_GPU"):
        bind_render_gpu(int(os.environ["B1K_RENDER_GPU"]))
    gm.HEADLESS = True
    gm.RENDER_VIEWER_CAMERA = False  # same as eval.py in headless mode
    seed = seed_everything(DEFAULT_EVAL_SEED)
    logger.info(f"Resolved public_test instance ids for {args.task_name}: {instance_ids}; checkpoints: {args.checkpoints}")

    cfg = OmegaConf.create(
        {
            "env_wrapper": {"_target_": args.env_wrapper},
            "policy_name": "websocket",
            "model": {
                "_target_": "omnigibson.eval.policies.WebsocketPolicy",
                "host": args.host,
                "port": args.port,
                "action_chunk_size": args.action_chunk_size,
            },
            "headless": True,
            "partial_scene_load": True,
            "max_steps": args.max_steps,
            "write_video": not args.no_video,
            "mode": "public_test",
            "seed": seed,
            "num_envs": len(instance_ids),
            "task": {"name": args.task_name},
            "robot": None,
        }
    )

    with BatchedEvaluator(cfg) as evaluator:
        for ckpt in args.checkpoints:
            if args.phase_dir:
                go = os.path.join(args.phase_dir, f"go_{ckpt}")
                while not os.path.exists(go):
                    time.sleep(2)
            if not ckpt_done(ckpt):
                json_dir, video_dir = out_dirs(ckpt)
                os.makedirs(json_dir, exist_ok=True)
                os.makedirs(video_dir, exist_ok=True)
                if not args.phase_dir and not args.no_switch:
                    switch_checkpoint(args.host, args.port, ckpt)
                t0 = time.time()
                results = evaluator.run(
                    [int(i) for i in instance_ids],
                    write_video=not args.no_video,
                    video_path=video_dir,
                    metrics_dir=json_dir,
                    rollout_id=0,
                    video_fps=args.video_fps,
                )
                results = list(results.values())
                n_success = sum(r["success"] for r in results)
                mean_q = sum(r.get("q_score", {}).get("final", 0.0) for r in results) / max(len(results), 1)
                logger.info(
                    f"CKPT_DONE {ckpt} {args.task_name} {instance_ids}: {n_success}/{len(results)} success | "
                    f"mean q_score={mean_q:.3f} | {(time.time() - t0) / 60:.1f} min"
                )
            if args.phase_dir:
                open(os.path.join(args.phase_dir, f"done_{ckpt}_{args.rank}"), "w").close()

if __name__ == "__main__":
    main()
