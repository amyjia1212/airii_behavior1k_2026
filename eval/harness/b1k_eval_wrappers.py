from omnigibson.envs import Environment, EnvironmentWrapper
from omnigibson.eval.utils.eval_utils import EVAL_CAMERA_ROLES, HEAD_RESOLUTION, WRIST_RESOLUTION


class RGBFullResWrapper(EnvironmentWrapper):
    """
    Same cameras/resolutions as the official RGBDFullResWrapper (head 720x720, wrists 480x480),
    but RGB only -- the policy never consumes depth, so skipping it just saves render time.
    """

    @classmethod
    def camera_spec(cls) -> dict:
        resolution = {role: (HEAD_RESOLUTION if role == "head" else WRIST_RESOLUTION) for role in EVAL_CAMERA_ROLES}
        return {"modalities": ["rgb"], "resolution": resolution}

    def __init__(self, env: Environment):
        super().__init__(env=env)
