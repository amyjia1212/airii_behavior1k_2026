"""Serve a PI_BEHAVIOR checkpoint to the 2026 BEHAVIOR evaluator (omnigibson.eval.eval).

Uses the 2025 solution's B1KPolicyWrapper (rolling inpainting, stage voting, action compression)
behind openpi's 2026 websocket server, which speaks the same protocol as the 2026 evaluator client.

    PYTHONPATH=src /workspace/openpi/.venv/bin/python scripts/serve_b1k_2026.py \
        --policy-dir outputs/checkpoints/pi_behavior_sweep_lora/<exp>/<step>

Task mapping: the evaluator sends obs["task_id"] = 57 (sweeping_garage); the model was fine-tuned for
it in slot 36, so 57 is rewritten to 36. The 2025 "eval tricks" (hand-written per-task gripper/stage
correction rules) are OFF by default: slot 36's rules force both grippers open, which would make
holding the broom impossible.
"""

import dataclasses
import logging
import os
import socket

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.45")

import numpy as np
import torch
import tyro

from openpi.serving import websocket_b1k_server

from b1k.policies import policy_config as _policy_config
from b1k.policies.b1k_policy import PROPRIOCEPTION_INDICES
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig
from b1k.training import config as _config


@dataclasses.dataclass
class Args:
    policy_dir: str
    config: str = "pi_behavior_sweep_lora"
    port: int = 8000
    # Evaluator task id -> model task slot.
    task_map: dict[int, int] = dataclasses.field(default_factory=lambda: {57: 36})
    actions_to_execute: int = 26
    actions_to_keep: int = 4
    execute_in_n_steps: int = 20
    history_len: int = 3
    votes_to_promote: int = 2
    time_threshold_inpaint: float = 0.3
    num_steps: int = 20
    apply_eval_tricks: bool = False
    # Clamp trunk commands to the envelope of the 2026 sweeping demos (see cap_torso). Off = raw model output.
    torso_cap: bool = True
    # If set, append per-step trunk state / commanded / capped trunk action to this CSV (diagnostics).
    torso_log: str | None = None


# JoyLo teleop moves the trunk along upright -> downward -> ground (og_teleop_cfg.py R1_*_TORSO_JOINT_POS).
TORSO_PATH = np.array([[0.45, -0.4, 0.0, 0.0], [1.6, -2.5, -0.94, 0.0], [1.735, -2.57, -2.1, 0.0]])
# Per-joint range of trunk actions over all 894,606 frames of the 2026 sweeping_garage demos.
TORSO_DEMO_MIN = np.array([0.45, -2.563, -1.833, 0.0])
TORSO_DEMO_MAX = np.array([1.721, -0.4, 0.078, 0.0])
# 99.9th percentile distance of demo trunk actions from TORSO_PATH (max 0.55).
TORSO_MAX_PATH_DIST = 0.3


def _closest_on_torso_path(q: np.ndarray) -> np.ndarray:
    best, best_d = None, np.inf
    for a, b in zip(TORSO_PATH[:-1], TORSO_PATH[1:]):
        d = b - a
        p = a + np.clip((q - a) @ d / (d @ d), 0.0, 1.0) * d
        if np.linalg.norm(q - p) < best_d:
            best, best_d = p, np.linalg.norm(q - p)
    return best


def cap_torso(q: np.ndarray) -> np.ndarray:
    """Keeps a trunk command inside the demo envelope: per-joint demo range, then at most
    TORSO_MAX_PATH_DIST from the teleop trunk path. Stops the policy from leaning back/over."""
    q = np.clip(q, TORSO_DEMO_MIN, TORSO_DEMO_MAX)
    p = _closest_on_torso_path(q)
    dist = np.linalg.norm(q - p)
    if dist > TORSO_MAX_PATH_DIST:
        q = p + (q - p) * (TORSO_MAX_PATH_DIST / dist)
    return q


def _py(v):
    return float(v) if isinstance(v, (np.floating, float)) else v


class TaskRemapWrapper:
    """Rewrites obs["task_id"] from the evaluator's task index to the model's task slot."""

    def __init__(self, policy: B1KPolicyWrapper, task_map: dict[int, int], torso_cap: bool = True,
                 torso_log: str | None = None):
        self._policy = policy
        self._task_map = task_map
        self._torso_cap = torso_cap
        self._torso_log = open(torso_log, "a") if torso_log else None
        self._episode = 0
        self._step = 0

    def act(self, obs: dict):
        if "task_id" in obs:
            task_id = int(np.asarray(obs["task_id"]).reshape(-1)[0])
            obs["task_id"] = np.array([self._task_map.get(task_id, task_id)], dtype=np.int64)
        action = self._policy.act(obs)
        raw = action[3:7].numpy().copy()
        if self._torso_cap:
            action[3:7] = torch.from_numpy(cap_torso(raw.astype(np.float64))).to(action.dtype)
        if self._torso_log is not None:
            state = np.asarray(obs["robot_r1::proprio"]).reshape(-1)[PROPRIOCEPTION_INDICES["R1Pro"]["trunk_qpos"]]
            row = [self._episode, self._step, *state, *raw, *action[3:7].numpy()]
            self._torso_log.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in map(_py, row)) + "\n")
            if self._step % 100 == 0:
                self._torso_log.flush()
        self._step += 1
        return action

    def reset(self):
        self._policy.reset()
        self._episode += 1
        self._step = 0


def main(args: Args) -> None:
    config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(config, args.policy_dir, sample_kwargs={"num_steps": args.num_steps})
    wrapper_config = B1KWrapperConfig(
        actions_to_execute=args.actions_to_execute,
        actions_to_keep=args.actions_to_keep,
        execute_in_n_steps=args.execute_in_n_steps,
        history_len=args.history_len,
        votes_to_promote=args.votes_to_promote,
        time_threshold_inpaint=args.time_threshold_inpaint,
        num_steps=args.num_steps,
        apply_eval_tricks=args.apply_eval_tricks,
    )
    default_task = next(iter(args.task_map.values()))
    wrapped = B1KPolicyWrapper(policy, task_id=default_task, config=wrapper_config)
    wrapped = TaskRemapWrapper(wrapped, args.task_map, torso_cap=args.torso_cap, torso_log=args.torso_log)
    logging.info(f"Serving {args.policy_dir} on port {args.port}; task map {args.task_map}; torso_cap={args.torso_cap}; {wrapper_config}")
    logging.info("Host: %s", socket.gethostname())
    websocket_b1k_server.WebsocketPolicyServer(
        policy=wrapped, host="0.0.0.0", port=args.port, metadata=policy.metadata
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
