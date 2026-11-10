"""R1Pro proprioception layout for the 61-dim `observation.state` field recorded in the
BEHAVIOR-1K challenge demo datasets (both 2025 and 2026 LeRobot exports use this layout).

Copied from the now-removed omnigibson.learning.utils.eval_utils (see
StanfordVL/BEHAVIOR-1K PR #2260, which replaced the `learning` module with `eval` and
dropped this constant). Verified against behavior-1k/2026-challenge-demos: gripper qpos
values fall in [0, 0.05] as expected for R1Pro's gripper joint range.
"""

from collections import OrderedDict

import numpy as np

PROPRIOCEPTION_INDICES = {
    "R1Pro": OrderedDict(
        {
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
    ),
}
