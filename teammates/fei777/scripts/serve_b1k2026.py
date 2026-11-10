#!/usr/bin/env python3
"""Policy server for the BEHAVIOR Challenge 2026 evaluator (BEHAVIOR-1K v3.9.3).

Loads one checkpoint, wraps it in the rollout wrapper (action chunk execution, rolling inpainting,
stage voting) and serves it over the BEHAVIOR websocket protocol. The v3.9.3 evaluator sends every
observation with a leading num_envs axis of size 1 (run it with --num-envs 1); it is removed before the
wrapper sees the observation and restored on the returned 23-dimensional R1Pro action.

Based on the single-checkpoint backend of the team's policy image (docker/backend/serve_checkpoint.py).
"""

import argparse
import asyncio
import dataclasses
import http
import logging
import os
import traceback

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import numpy as np
from openpi_client import msgpack_numpy
from websockets.asyncio.server import serve

from b1k.policies.policy_config import create_trained_policy
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig
from b1k.training.config import get_config

log = logging.getLogger("backend")
ACTION_DIM = 23


def load_wrapper(args):
    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    config = get_config(args.config)
    assets = dataclasses.replace(config.data.assets, assets_dir=os.path.join(checkpoint_dir, "assets"))
    if args.asset_id:
        assets = dataclasses.replace(assets, asset_id=args.asset_id)
    config = dataclasses.replace(config, data=dataclasses.replace(config.data, assets=assets))
    policy = create_trained_policy(config, checkpoint_dir, sample_kwargs={"num_steps": args.num_steps})
    wrapper_config = B1KWrapperConfig(
        actions_to_execute=args.actions_to_execute,
        actions_to_keep=args.actions_to_keep,
        execute_in_n_steps=args.execute_in_n_steps,
        history_len=args.history_len,
        votes_to_promote=args.votes_to_promote,
        time_threshold_inpaint=args.time_threshold_inpaint,
        num_steps=args.num_steps,
        apply_eval_tricks=not args.no_eval_tricks,
    )
    return policy, B1KPolicyWrapper(policy, task_id=args.task_id, config=wrapper_config)


def unbatch(obs):
    """Drop the leading num_envs axis (size 1) that the v3.9.3 evaluator adds to every array."""
    proprio = next((v for k, v in obs.items() if k.endswith("::proprio")), None)
    if proprio is None or np.ndim(proprio) != 2:
        return obs, False
    out = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1:
            if value.shape[0] != 1:
                raise ValueError(f"observation '{key}' has batch size {value.shape[0]}; run the evaluator with --num-envs 1")
            value = value[0]
        out[key] = value
    return out, True


def health_check(connection, request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None


async def run(args):
    policy, wrapper = load_wrapper(args)
    log.info("checkpoint loaded from %s (config %s)", args.checkpoint_dir, args.config)
    lock = asyncio.Lock()

    serve_tasks = set(args.serve_tasks or [])
    task_mapping = dict(pair.split(":") for pair in args.task_mapping or [])
    task_mapping = {int(k): int(v) for k, v in task_mapping.items()}

    def reset():
        wrapper.reset()
        if args.policy_seed is not None:
            policy._rng = jax.random.key(args.policy_seed)

    def route_task(obs):
        """Check the task id of an observation and translate it to the task slot the checkpoint uses."""
        if "task_id" not in obs:
            return
        task_id = int(np.asarray(obs["task_id"]).reshape(-1)[0])
        if serve_tasks and task_id not in serve_tasks:
            raise ValueError(f"this checkpoint serves task ids {sorted(serve_tasks)}, got {task_id}")
        if task_id in task_mapping:
            obs["task_id"] = np.array([task_mapping[task_id]], dtype=np.int64)

    async def handle(socket):
        async with lock:  # wrapper state belongs to one rollout
            packer = msgpack_numpy.Packer()
            reset()
            await socket.send(packer.pack({"checkpoint": os.path.basename(os.path.abspath(args.checkpoint_dir))}))
            async for payload in socket:
                try:
                    obs = msgpack_numpy.unpackb(payload)
                    if "reset" in obs:
                        reset()
                        continue
                    obs, batched = unbatch(obs)
                    route_task(obs)
                    action = wrapper.act(obs).cpu().numpy().astype(np.float32)
                    if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
                        raise ValueError(f"invalid action from policy: shape {action.shape}")
                    await socket.send(packer.pack({"action": action[None] if batched else action}))
                except Exception:
                    log.error("inference failed:\n%s", traceback.format_exc())
                    await socket.send(traceback.format_exc())
                    await socket.close(code=1011, reason="policy backend error")
                    return

    async with serve(
        handle, args.host, args.port, process_request=health_check, compression=None, max_size=None, ping_interval=None
    ) as server:
        log.info("backend listening on %s:%d", args.host, args.port)
        await server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", required=True, help="directory containing params/ and assets/")
    parser.add_argument("--config", default="pi_behavior_b1k_fast", help="training config name the checkpoint was produced with")
    parser.add_argument("--asset-id", default=None, help="assets/<asset id>/norm_stats.json inside the checkpoint")
    parser.add_argument("--task-id", type=int, default=None, help="fixed task id; by default it is read from each observation")
    parser.add_argument("--serve-tasks", type=int, nargs="+", default=None, help="task ids this checkpoint may be used for")
    parser.add_argument(
        "--task-mapping", nargs="+", default=None, metavar="TASK:SLOT",
        help="task ids that were trained on another task slot of the model, e.g. 59:19",
    )  # fmt: skip
    parser.add_argument("--policy-seed", type=int, default=None, help="reseed the sampler at the start of every rollout")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--actions-to-execute", type=int, default=26)
    parser.add_argument("--actions-to-keep", type=int, default=4)
    parser.add_argument("--execute-in-n-steps", type=int, default=20)
    parser.add_argument("--history-len", type=int, default=3)
    parser.add_argument("--votes-to-promote", type=int, default=2)
    parser.add_argument("--time-threshold-inpaint", type=float, default=0.3)
    parser.add_argument("--num-steps", type=int, default=20, help="flow-matching denoising steps")
    parser.add_argument("--no-eval-tricks", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s", force=True)
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
