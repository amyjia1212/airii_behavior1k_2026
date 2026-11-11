"""Serve the 2025-format B1K Pi0.5 policy (behavior-1k-solution) to the 2026 BEHAVIOR evaluator.

The 2026 evaluator (BEHAVIOR-1K v3.9.3) differs from the 2025 one the model was trained/served for:
  1. Observations are batched: every value has a leading (num_envs,) dim. We support num_envs == 1.
  2. Proprio is the compact 61-dim layout from omnigibson.eval.r1pro.yaml instead of the 2025
     256-dim layout. We scatter the fields the model reads into their 2025 slots (others stay 0;
     extract_state_from_proprio only reads base_qvel, trunk_qpos, arm_*_qpos, gripper_*_qpos).
  3. obs["task_id"] uses the 2026 B100 numbering. The model's task embedding uses the 2025
     numbering (clean_a_trumpet == 37), so we overwrite it with --task-id.
  4. base_qvel is in the robot-local frame in 2026 (world frame in the 2025 training data).
     Converting back needs the robot's global yaw, which is privileged info. By default it is passed
     through as-is; --world-frame-base-vel converts it using a yaw the patched evaluator exposes
     (B1K_EXPOSE_BASE_YAW=1), and --zero-base-vel hides it.

Run with the behavior-1k-solution uv env, e.g.:
  uv run --project /workspace/behavior-1k-solution python serve_2026_adapter.py \
      --ckpt /workspace/ckpt_task37/lr5e-6/step_09999 --task-id 37 --port 8000
"""

import argparse
import logging
import os

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import numpy as np
import torch

from omnigibson.learning.utils.eval_utils import PROPRIOCEPTION_INDICES as PROPRIO_2025
from omnigibson.learning.utils.network_utils import WebsocketPolicyServer

from b1k.policies import policy_config as _policy_config
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig
from b1k.training import config as _config

logger = logging.getLogger(__name__)

# Copied from BEHAVIOR-1K v3.9.3 OmniGibson/omnigibson/eval/utils/eval_utils.py (R1Pro).
PROPRIO_2026 = {
    "base_qvel": np.s_[0:3],
    "arm_left_qpos": np.s_[3:10],
    "arm_left_qvel": np.s_[10:17],
    "eef_left_pos": np.s_[17:20],
    "eef_left_quat": np.s_[20:24],
    "gripper_left_qpos": np.s_[24:26],
    "gripper_left_qvel": np.s_[26:28],
    "arm_right_qpos": np.s_[28:35],
    "arm_right_qvel": np.s_[35:42],
    "eef_right_pos": np.s_[42:45],
    "eef_right_quat": np.s_[45:49],
    "gripper_right_qpos": np.s_[49:51],
    "gripper_right_qvel": np.s_[51:53],
    "trunk_qpos": np.s_[53:57],
    "trunk_qvel": np.s_[57:61],
}
PROPRIO_2026_DIM = 61
PROPRIO_2025_DIM = 256


def proprio_2026_to_2025(proprio: np.ndarray) -> np.ndarray:
    assert proprio.shape[-1] == PROPRIO_2026_DIM, f"Unexpected 2026 proprio dim {proprio.shape}"
    out = np.zeros(proprio.shape[:-1] + (PROPRIO_2025_DIM,), dtype=np.float32)
    for key, src in PROPRIO_2026.items():
        out[..., PROPRIO_2025["R1Pro"][key]] = proprio[..., src]
    return out


class Eval2026Adapter:
    def __init__(self, inner: B1KPolicyWrapper, task_id: int, zero_base_vel: bool = False, world_frame_base_vel: bool = False):
        self.inner = inner
        self.task_id = task_id
        self.zero_base_vel = zero_base_vel
        self.world_frame_base_vel = world_frame_base_vel
        self._warned_task_id = False

    def reset(self):
        self.inner.reset()

    def act(self, obs: dict) -> torch.Tensor:
        unbatched = {}
        for key, value in obs.items():
            value = np.asarray(value)
            if value.ndim > 0:
                assert value.shape[0] == 1, f"Only --num-envs 1 is supported, got {key} with shape {value.shape}"
                value = value[0]
            unbatched[key] = value

        unbatched["robot_r1::proprio"] = proprio_2026_to_2025(unbatched["robot_r1::proprio"])
        if self.world_frame_base_vel:
            # Undo the 2026 robot-frame rotation (OmniGibson robot.py _get_base_qvel_for_proprioception).
            # Needs the evaluator run with B1K_EXPOSE_BASE_YAW=1 (local patch, privileged info).
            assert "robot_r1::base_yaw" in unbatched, "Run the evaluator with B1K_EXPOSE_BASE_YAW=1"
            yaw = float(np.asarray(unbatched.pop("robot_r1::base_yaw")).ravel()[0])
            idx = PROPRIO_2025["R1Pro"]["base_qvel"]
            vx, vy, wz = unbatched["robot_r1::proprio"][..., idx]
            c, s = np.cos(yaw), np.sin(yaw)
            unbatched["robot_r1::proprio"][..., idx] = [c * vx - s * vy, s * vx + c * vy, wz]
        if self.zero_base_vel:
            # Diagnostic: hide the robot-frame base velocity instead of feeding it where world-frame was trained.
            unbatched["robot_r1::proprio"][..., PROPRIO_2025["R1Pro"]["base_qvel"]] = 0.0
        if not self._warned_task_id and "task_id" in unbatched:
            logger.info(f"Evaluator task_id (2026 numbering) = {unbatched['task_id']}; using 2025 task_id {self.task_id}")
            self._warned_task_id = True
        unbatched["task_id"] = np.array([self.task_id])

        action = self.inner.act(unbatched)
        return action.unsqueeze(0)  # (num_envs=1, 23)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config", default="pi_behavior_b1k_fast")
    parser.add_argument("--task-id", type=int, required=True, help="2025 B1K task index (clean_a_trumpet = 37)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--no-eval-tricks", action="store_true")
    parser.add_argument(
        "--world-frame-base-vel",
        action="store_true",
        help="Rotate base velocity back to the 2025 world frame (needs evaluator with B1K_EXPOSE_BASE_YAW=1)",
    )
    parser.add_argument("--zero-base-vel", action="store_true", help="Diagnostic: feed zero base velocity to the model")
    args = parser.parse_args()

    # Same defaults as behavior-1k-solution/scripts/serve_b1k.py
    train_config = _config.get_config(args.config)
    policy = _policy_config.create_trained_policy(train_config, args.ckpt, sample_kwargs={"num_steps": args.num_steps})
    metadata = policy.metadata
    wrapper = B1KPolicyWrapper(
        policy,
        task_id=args.task_id,
        config=B1KWrapperConfig(num_steps=args.num_steps, apply_eval_tricks=not args.no_eval_tricks),
    )
    server = WebsocketPolicyServer(
        policy=Eval2026Adapter(wrapper, args.task_id, zero_base_vel=args.zero_base_vel, world_frame_base_vel=args.world_frame_base_vel), host="0.0.0.0", port=args.port, metadata=metadata
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
