"""Data loading for BEHAVIOR-1K dataset.

Reference: https://github.com/wensi-ai/openpi/tree/behavior
"""

import logging

import numpy as np
import time

# Import all base data loading from OpenPI
from openpi.training.data_loader import (
    Dataset,
    IterableDataset,
    DataLoader,
    TransformedDataset,
    IterableTransformedDataset,
    FakeDataset,
    TorchDataLoader,
    RLDSDataLoader,
    create_torch_dataset,
    create_rlds_dataset,
    transform_iterable_dataset,
    create_data_loader,
    create_torch_data_loader,
    create_rlds_data_loader,
)

import openpi.training.config as _config
import openpi.transforms as _transforms

from b1k.models.observation import Observation
from b1k.transforms_normalize import NormalizeWithPerTimestamp


class DataLoaderImpl(DataLoader):
    """Custom DataLoader using our Observation with fast_tokens."""
    
    def __init__(self, data_config: _config.DataConfig, data_loader: TorchDataLoader | RLDSDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self) -> _config.DataConfig:
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield Observation.from_dict(batch), batch["actions"]


def create_behavior_dataset(data_config: _config.DataConfig, action_horizon: int, seed: int | None = None) -> Dataset:
    """Create a BEHAVIOR-1K dataset for training.

    Uses the standard lerobot LeRobotDataset (v3 codebase) to load behavior-1k/2026-challenge-demos.
    NOTE: unlike the original OmniGibson BehaviorLeRobotDataset (built for the older, now-removed
    `omnigibson.learning` module and the 2025 challenge's LeRobot v2.1 dataset layout),
    `data_config.episodes_index` here is a list of ABSOLUTE episode indices in the dataset
    (e.g. [0..199] for task_index=0 / turning_on_radio), not per-task-local indices.

    Args:
        data_config: Data configuration
        action_horizon: Action horizon for delta timestamps
        seed: Unused (kept for signature compatibility; shuffling is handled by the TorchDataLoader).

    Returns:
        Dataset instance with BEHAVIOR-1K data
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(
        repo_id=data_config.repo_id,
        root=data_config.behavior_dataset_root,
        episodes=data_config.episodes_index,
        delta_timestamps={
            key: [t / 30.0 for t in range(action_horizon)] for key in data_config.action_sequence_keys
        },
        # Default tolerance_s=1e-4 is tighter than float32 precision at the large cumulative
        # timestamps used in this dataset's video chunks (ULP ~1.6e-4 at ~1374s), causing
        # spurious FrameTimestampError crashes. Half a frame period (1/60s) is generous enough
        # to absorb float32 rounding while still catching genuine sync issues.
        tolerance_s=1.0 / 60.0,
    )

    _memory_map_episode_subset(dataset)

    if data_config.prompt_from_task:
        dataset = TransformedDataset(dataset, [_transforms.PromptFromLeRobotTask(dataset.meta.tasks)])

    return dataset


class _AbsoluteToRelativeIndex:
    """Absolute dataset index -> row of the episode-subset table (numpy, cheap to pickle)."""

    def __init__(self, absolute_indices: np.ndarray):
        self._order = np.argsort(absolute_indices, kind="stable")
        self._sorted = absolute_indices[self._order]

    def __getitem__(self, absolute_index: int) -> int:
        pos = int(np.searchsorted(self._sorted, absolute_index))
        if pos >= len(self._sorted) or self._sorted[pos] != absolute_index:
            raise KeyError(absolute_index)
        return int(self._order[pos])


def _memory_map_episode_subset(dataset) -> None:
    """Back an episode-subset LeRobotDataset by an on-disk Arrow copy instead of an in-memory table.

    lerobot loads an episode subset as an in-memory Arrow table, and the training loader starts its workers
    with the "spawn" method, which pickles the dataset (and the whole table) into every worker. A
    memory-mapped dataset pickles as a file reference, and the index mapping as two numpy arrays.
    """
    import hashlib
    import os
    import pathlib

    import datasets
    from lerobot.datasets.utils import hf_transform_to_torch

    if dataset.episodes is None:
        return
    key = hashlib.md5(f"{dataset.root}|{sorted(dataset.episodes)}".encode()).hexdigest()
    cache_dir = pathlib.Path(os.environ.get("B1K_ARROW_CACHE", "/root/data/arrow_cache")) / key
    if not (cache_dir / "dataset_info.json").exists():
        logging.info(f"Writing memory-mapped copy of the episode subset to {cache_dir}")
        dataset.hf_dataset.with_format(None).save_to_disk(str(cache_dir), max_shard_size="2GB")
    hf_dataset = datasets.load_from_disk(str(cache_dir))
    hf_dataset.set_transform(hf_transform_to_torch)
    dataset.hf_dataset = hf_dataset
    dataset._absolute_to_relative_idx = _AbsoluteToRelativeIndex(hf_dataset.data.column("index").to_numpy())


def transform_dataset(dataset: Dataset, data_config: _config.DataConfig, *, skip_norm_stats: bool = False) -> Dataset:
    """Transform dataset with B1K-specific per-timestamp normalization support.
    
    CRITICAL: This overrides wensi-ai's transform_dataset to pass use_per_timestamp to Normalize.
    wensi-ai's version doesn't support per-timestamp normalization which causes huge action losses!
    """
    norm_stats = {}
    if data_config.repo_id != "fake" and not skip_norm_stats:
        if data_config.norm_stats is None:
            raise ValueError(
                "Normalization stats not found. "
                "Make sure to run `scripts/compute_norm_stats.py --config-name=<your-config>`."
            )
        norm_stats = data_config.norm_stats

    # Build transform list
    transforms_list = [
        *data_config.repack_transforms.inputs,
        *data_config.data_transforms.inputs,
        # Use custom Normalize with per-timestamp support (wensi-ai's doesn't have it!)
        NormalizeWithPerTimestamp(
            norm_stats, 
            use_quantiles=data_config.use_quantile_norm,
            use_per_timestamp=data_config.use_per_timestamp_norm  # CRITICAL: Per-timestamp normalization!
        ),
    ]
    
    # Add subtask state computation for PI_BEHAVIOR models (needs dataset reference)
    model_transforms = []
    for transform in data_config.model_transforms.inputs:
        # ComputeSubtaskStateFromMeta needs dataset reference to access episode lengths
        if hasattr(transform, '__class__') and transform.__class__.__name__ == 'ComputeSubtaskStateFromMeta':
            # Replace placeholder with dataset-aware version
            from b1k import transforms as b1k_transforms
            if hasattr(dataset, 'meta') and hasattr(dataset.meta, 'episodes'):
                # Preserve task_mapping from the placeholder instance (e.g. for tasks with
                # task_index >= 50, remapped to a valid TASK_NUM_STAGES slot).
                model_transforms.append(
                    b1k_transforms.ComputeSubtaskStateFromMeta(dataset=dataset, task_mapping=transform.task_mapping)
                )
                logging.info("Added dataset-aware ComputeSubtaskStateFromMeta transform")
            else:
                logging.warning("Skipping subtask state computation - dataset has no meta.episodes")
        else:
            model_transforms.append(transform)
    
    transforms_list.extend(model_transforms)

    return TransformedDataset(dataset, transforms_list)


def extract_episode_lengths_from_dataset(dataset) -> dict[int, float]:
    """Extract episode lengths from B1K dataset metadata.
    
    Args:
        dataset: BehaviorLeRobotDataset instance
        
    Returns:
        Dictionary mapping episode_index to episode_length (in frames)
        
    Raises:
        ValueError: If dataset doesn't have required metadata
    """
    if not hasattr(dataset, 'episode_data_index'):
        raise ValueError("Dataset must have episode_data_index attribute")
    
    episode_data_index = dataset.episode_data_index
    if 'to' not in episode_data_index or 'from' not in episode_data_index:
        raise ValueError("episode_data_index must have 'to' and 'from' keys")
    
    episode_to = episode_data_index['to'] 
    episode_from = episode_data_index['from']
    episodes = dataset.episodes
    
    episode_lengths = {}
    for i, episode_index in enumerate(episodes):
        if i < len(episode_to) and i < len(episode_from):
            episode_length = episode_to[i] - episode_from[i]
            episode_lengths[episode_index] = float(episode_length)
    
    logging.info(f"Extracted {len(episode_lengths)} episode lengths from dataset")
    return episode_lengths


def create_behavior_data_loader(
    config: _config.TrainConfig,
    *,
    sharding=None,
    shuffle: bool = False,
    num_batches: int | None = None,
    skip_norm_stats: bool = False,
) -> DataLoader:
    """Create a data loader for BEHAVIOR-1K training."""
    import jax
    import time
    
    data_config = config.data.create(config.assets_dirs, config.model)
    
    # Use random seed if not provided
    seed = config.seed
    if seed is None:
        seed = int(time.time() * 1000) % (2**32)
        logging.info(f"Using random seed: {seed}")
    
    dataset = create_behavior_dataset(data_config, action_horizon=config.model.action_horizon, seed=seed)
    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=config.batch_size // jax.process_count(),
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=config.num_workers,
        seed=seed,
    )
    
    return DataLoaderImpl(data_config, data_loader)
