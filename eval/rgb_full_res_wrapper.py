from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import EVAL_CAMERA_ROLES, HEAD_RESOLUTION, WRIST_RESOLUTION


class RGBFullResWrapper(EnvironmentWrapper):
    """
    Eval wrapper used for our 2026 self-evaluation: full-resolution RGB observations (head 720x720, wrists
    480x480, i.e. the data-collection resolutions) without depth. The policy server resizes the images to the
    model's 280x280 input.

    Same as BEHAVIOR-1K's ``omnigibson.eval.wrappers.rgbd_full_res_wrapper.RGBDFullResWrapper`` (v3.9.3) minus the
    ``depth_linear`` modality, which our model does not use and which slows rendering down.

    Args:
        env (og.Environment): The environment to wrap.
    """

    @classmethod
    def camera_spec(cls) -> dict:
        """Returns {"modalities": [...], "resolution": {camera_role: (H, W)}} for this eval profile."""
        resolution = {role: (HEAD_RESOLUTION if role == "head" else WRIST_RESOLUTION) for role in EVAL_CAMERA_ROLES}
        return {"modalities": ["rgb"], "resolution": resolution}

    def __init__(self, env: Environment):
        super().__init__(env=env)
