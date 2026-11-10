"""Minimal Policy subclass for PiBehavior models that handles tuple unpacking.

Reference: https://github.com/PhysicalIntelligence/openpi/blob/behavior/src/openpi/policies/policy.py
"""

import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from openpi.policies.policy import Policy
from b1k.models.observation import Observation  # Use our custom Observation with FAST fields


class PiBehaviorPolicy(Policy):
    """Policy for PiBehavior models - only difference is unpacking the tuple return.
    
    PiBehavior.sample_actions() returns (actions, subtask_logits) instead of just actions.
    This minimal subclass unpacks the tuple before output transforms are applied.
    """
    
    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None, initial_actions: np.ndarray | None = None) -> dict:
        """Infer with PiBehavior-specific tuple unpacking.
        
        Identical to parent Policy.infer() except:
        1. Accepts initial_actions parameter for rolling inpainting
        2. Unpacks (actions, subtask_logits) tuple before output transforms
        """
        # Reuse all parent logic for input processing
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        self._rng, sample_rng = jax.random.split(self._rng)

        # Prepare sample_kwargs
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = jnp.asarray(noise)
            if noise.ndim == 2:
                noise = noise[None, ...]
            sample_kwargs["noise"] = noise
            
        if initial_actions is not None:
            initial_actions = jnp.asarray(self._transform_initial_actions(obs, initial_actions))
            if initial_actions.ndim == 2:
                initial_actions = initial_actions[None, ...]
            sample_kwargs["initial_actions"] = initial_actions

        observation = Observation.from_dict(inputs)
        start_time = time.monotonic()
        
        # ONLY DIFFERENCE: Unpack tuple return from PiBehavior.sample_actions
        actions, subtask_logits = self._sample_actions(sample_rng, observation, **sample_kwargs)
        
        outputs = {
            "state": inputs["state"],
            "actions": actions,  # Now an array, not a tuple!
            "subtask_logits": subtask_logits,
        }
        
        model_time = time.monotonic() - start_time
        
        # Convert to numpy (same as parent)
        outputs = {
            k: np.asarray(v[0, ...]) if isinstance(v, (jnp.ndarray, np.ndarray)) else v
            for k, v in outputs.items()
        }

        # Apply output transforms (now works because actions is an array)
        outputs = self._output_transform(outputs)
        
        # Add convenience field
        outputs["predicted_stage"] = int(np.argmax(outputs["subtask_logits"]))
        
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def _transform_initial_actions(self, obs: dict, initial_actions: np.ndarray) -> np.ndarray:
        """Normalize raw inpainting actions with the same input transforms (delta + normalization) as training."""
        # Create training-format observation batch for proper transform processing
        training_obs = {}

        # Map evaluation keys to training keys if needed
        if "observation/state" in obs:
            training_obs["observation/state"] = obs["observation/state"]
        elif "state" in obs:
            training_obs["observation/state"] = obs["state"]

        if "observation/egocentric_camera" in obs:
            training_obs["observation/egocentric_camera"] = obs["observation/egocentric_camera"]
        elif "image" in obs and "base_0_rgb" in obs["image"]:
            training_obs["observation/egocentric_camera"] = obs["image"]["base_0_rgb"]

        if "observation/wrist_image_left" in obs:
            training_obs["observation/wrist_image_left"] = obs["observation/wrist_image_left"]
        elif "image" in obs and "left_wrist_0_rgb" in obs["image"]:
            training_obs["observation/wrist_image_left"] = obs["image"]["left_wrist_0_rgb"]

        if "observation/wrist_image_right" in obs:
            training_obs["observation/wrist_image_right"] = obs["observation/wrist_image_right"]
        elif "image" in obs and "right_wrist_0_rgb" in obs["image"]:
            training_obs["observation/wrist_image_right"] = obs["image"]["right_wrist_0_rgb"]

        # Copy any other keys that might be needed (tokenized_prompt, subtask_state, etc.)
        for key in obs:
            if key not in training_obs and key not in ["image", "state"]:
                training_obs[key] = obs[key]

        initial_batch = {
            **training_obs,  # Include all observation data in training format
            "actions": initial_actions  # Add initial_actions as the actions field
        }

        # Apply the full input transform pipeline (delta transforms + normalization)
        return self._input_transform(initial_batch)["actions"]

    def infer_batch(
        self, obs_list: list[dict], initial_actions_list: list[np.ndarray] | None = None, pad_to: int = 0
    ) -> list[dict]:
        """Batched version of infer(): one model call for several independent observations.

        Per-sample transforms are identical to infer(); only the model forward pass is batched.
        initial_actions_list must be all-None or all-arrays. The batch is padded to `pad_to` by repeating the last
        observation (outputs discarded) so the jitted sampler only ever sees a few distinct shapes.
        """
        n = len(obs_list)
        if pad_to > n:
            obs_list = list(obs_list) + [obs_list[-1]] * (pad_to - n)
            if initial_actions_list is not None:
                initial_actions_list = list(initial_actions_list) + [initial_actions_list[-1]] * (pad_to - n)
        inputs_list = [self._input_transform(jax.tree.map(lambda x: x, obs)) for obs in obs_list]
        inputs = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *inputs_list)
        self._rng, sample_rng = jax.random.split(self._rng)

        sample_kwargs = dict(self._sample_kwargs)
        if initial_actions_list is not None:
            sample_kwargs["initial_actions"] = jnp.stack(
                [jnp.asarray(self._transform_initial_actions(obs, ia)) for obs, ia in zip(obs_list, initial_actions_list)]
            )

        observation = Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions, subtask_logits = self._sample_actions(sample_rng, observation, **sample_kwargs)
        actions, subtask_logits, states = np.asarray(actions), np.asarray(subtask_logits), np.asarray(inputs["state"])
        model_time = time.monotonic() - start_time

        results = []
        for i in range(n):
            outputs = self._output_transform(
                {"state": states[i], "actions": actions[i], "subtask_logits": subtask_logits[i]}
            )
            outputs["predicted_stage"] = int(np.argmax(outputs["subtask_logits"]))
            outputs["policy_timing"] = {"infer_ms": model_time * 1000}
            results.append(outputs)
        return results

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
