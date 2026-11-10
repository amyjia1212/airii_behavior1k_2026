"""Weight loaders for PI_BEHAVIOR model initialization from Pi05 checkpoints.

Reference: https://github.com/Physical-Intelligence
"""

import dataclasses
import logging
import re

import flax.traverse_util
import numpy as np
import orbax.checkpoint as ocp

import openpi.shared.array_typing as at
import openpi.shared.download as download

# Re-export base loaders from OpenPI
from openpi.training.weight_loaders import (
    WeightLoader,
    NoOpWeightLoader,
    CheckpointWeightLoader,
    _merge_params,
)

logger = logging.getLogger(__name__)


# --- 2026 challenge additions -------------------------------------------------------------------
# Our 2026 runs initialise from PI_BEHAVIOR checkpoints trained on the 2025 challenge (50 tasks,
# 224x224 images). The 2026 model has 100 tasks and uses 280x280 images, so a few tables need to be
# resized before the checkpoint can be merged:
#   * task_embeddings/embedding        (50, 2048)  -> (100, 2048)   new rows = mean of existing rows
#   * task_stage_embeddings/embedding  (596, 1024) -> (1087, 1024)  new rows = mean of existing rows
#     (new tasks' stages are appended after the 2025 ones, matching TASK_STAGE_OFFSETS)
#   * PaliGemma/img/pos_embedding      (1, 256, 1152) -> (1, 400, 1152)  bicubic 16x16 -> 20x20 grid
_EXTENDABLE_TABLES = ("task_embeddings/embedding", "task_stage_embeddings/embedding")


def _expected_shape(x):
    return tuple(getattr(x, "shape", ()))


def _adapt_loaded_shapes(loaded_params: at.Params, expected_params: at.Params) -> at.Params:
    """Resize task tables / position embeddings in a loaded checkpoint to match the target model."""
    import jax

    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")
    flat_expected = flax.traverse_util.flatten_dict(expected_params, sep="/")

    for key, value in list(flat_loaded.items()):
        if key not in flat_expected:
            continue
        want = _expected_shape(flat_expected[key])
        have = tuple(np.shape(value))
        if not want or want == have:
            continue

        if key.endswith(_EXTENDABLE_TABLES) and len(have) == 2 and want[1] == have[1] and want[0] > have[0]:
            value = np.asarray(value)
            mean_row = value.mean(axis=0, keepdims=True).astype(value.dtype)
            extra = np.repeat(mean_row, want[0] - have[0], axis=0)
            flat_loaded[key] = np.concatenate([value, extra], axis=0)
            logger.info(f"Extended {key}: {have} -> {want} (new rows = mean of existing rows)")
        elif key.endswith("pos_embedding") and len(have) == 3 and have[0] == want[0] and have[2] == want[2]:
            old_side, new_side = int(round(have[1] ** 0.5)), int(round(want[1] ** 0.5))
            if old_side * old_side != have[1] or new_side * new_side != want[1]:
                raise ValueError(f"Cannot interpolate non-square position embedding {key}: {have} -> {want}")
            value = np.asarray(value, dtype=np.float32)
            grid = value.reshape(have[0], old_side, old_side, have[2])
            grid = jax.image.resize(grid, (have[0], new_side, new_side, have[2]), method="bicubic")
            flat_loaded[key] = np.asarray(grid).reshape(want).astype(np.asarray(flat_loaded[key]).dtype)
            logger.info(f"Interpolated {key}: {have} -> {want}")
        else:
            raise ValueError(f"Shape mismatch for {key}: checkpoint {have} vs model {want}")

    return flax.traverse_util.unflatten_dict(flat_loaded, sep="/")


@dataclasses.dataclass(frozen=True)
class PiBehaviorWeightLoader(WeightLoader):
    """Loads checkpoints for PI_BEHAVIOR model.
    
    Automatically detects:
    - Pi05 checkpoint: Loads weights, preserves new PI_BEHAVIOR parameters
    - PI_BEHAVIOR checkpoint: Loads all weights directly
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # Load checkpoint
        params_path = download.maybe_download(self.params_path)
        
        # Load directly with PyTreeCheckpointer (handles both old and new checkpoint formats)
        with ocp.PyTreeCheckpointer() as ckptr:
            restored = ckptr.restore(params_path)
        
        # Handle nested 'params' key (from some checkpoint formats)
        if isinstance(restored, dict) and "params" in restored:
            loaded_params = restored["params"]
        else:
            loaded_params = restored
        
        # Remove 'value' suffixes (from nnx.State format)
        flat_params = flax.traverse_util.flatten_dict(loaded_params)
        if all(kp[-1] == "value" for kp in flat_params if len(kp) > 0):
            flat_params = {kp[:-1]: v for kp, v in flat_params.items()}
            loaded_params = flax.traverse_util.unflatten_dict(flat_params)
        
        # Detect checkpoint type
        has_task_embeddings = 'task_embeddings' in loaded_params
        
        if has_task_embeddings:
            # Loading PI_BEHAVIOR checkpoint - load ALL weights from checkpoint
            logging.info("Loading PI_BEHAVIOR checkpoint (all weights will be loaded)")
            loaded_params = _adapt_loaded_shapes(loaded_params, params)
            # Use _merge_params with empty missing_regex to validate shapes
            return _merge_params(loaded_params, params, missing_regex="^$")
        else:
            # Loading Pi05 checkpoint - preserve new PI_BEHAVIOR-specific parameters
            logging.info("Loading Pi05 checkpoint (new PI_BEHAVIOR parameters will use random init)")
            loaded_params = _adapt_loaded_shapes(loaded_params, params)
            
            # These parameters are NEW in PI_BEHAVIOR (not in Pi05), so keep them from params (random init)
            missing_regex = (
                ".*task_embeddings.*|"
                ".*stage_pred_from_vlm.*|"
                ".*task_stage_embeddings.*|"
                ".*gate_sincos.*|"
                ".*gate_task_stage.*|"
                ".*gate_task.*|"
                ".*fusion_layer.*|"
                ".*stage_projection.*|"
                ".*task_subtask_fusion.*|"
                ".*fast_token_embedding.*|"
                ".*fast_token_proj.*|"
                ".*kv_transform.*"
            )
            return _merge_params(loaded_params, params, missing_regex=missing_regex)
