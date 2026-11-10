"""Minimal reader for the 2026 BEHAVIOR challenge dataset (LeRobot v3.0 layout).

The 2026 demos (behavior-1k/2026-challenge-demos) use the LeRobot v3 layout: many episodes are concatenated
into shared parquet files (data/chunk-XXX/file-YYY.parquet) and shared mp4 files per camera
(videos/<key>/chunk-XXX/file-YYY.mp4). Episode boundaries live in meta/episodes/**.parquet.

The lerobot version pinned by our environment (0.3.4) predates the v3 format, so the 2026 runs used this small
standalone reader instead. It only depends on pandas/pyarrow and PyAV, and returns samples in the same
format OmniGibson's 2025 BehaviorLeRobotDataset did, so the rest of the 2025 pipeline (repack -> B1kInputs ->
normalisation -> model transforms) is unchanged.

Each sample is a dict with:
    <camera_key>            uint8 (H, W, 3) frame for each requested camera
    observation.state       float32 (61,) proprioception (see PROPRIOCEPTION_INDICES["R1Pro"])
    action                  float32 (action_horizon, 23) chunk starting at this frame (edge-padded)
    timestamp, frame_index, episode_index, index, task_index
"""

from __future__ import annotations

import bisect
import collections
import dataclasses
import glob
import json
import logging
import os
import types
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LOW_DIM_COLUMNS = ("observation.state", "action", "timestamp", "frame_index", "episode_index", "index", "task_index")


def load_episodes_meta(root: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(root, "meta", "episodes", "chunk-*", "file-*.parquet")))
    if not files:
        raise FileNotFoundError(f"No meta/episodes/chunk-*/file-*.parquet under {root}")
    cols = None
    frames = []
    for f in files:
        df = pd.read_parquet(f)
        if cols is None:
            cols = [c for c in df.columns if not c.startswith("stats/")]
        frames.append(df[cols])
    return pd.concat(frames, ignore_index=True).sort_values("episode_index").reset_index(drop=True)


@dataclasses.dataclass(frozen=True)
class EpisodeRef:
    """One episode inside a shared v3 data file (picklable, used by the stats/tokenizer scripts)."""

    data_file: str
    row_start: int  # first row of the episode inside data_file
    length: int
    episode_index: int
    task_index: int

    def read(self, columns: Sequence[str] | None = ("observation.state", "action")) -> pd.DataFrame:
        df = pd.read_parquet(self.data_file, columns=list(columns) if columns else None)
        out = df.iloc[self.row_start : self.row_start + self.length].reset_index(drop=True)
        if len(out) != self.length:
            raise ValueError(f"Episode {self.episode_index}: expected {self.length} rows, got {len(out)}")
        return out

    def __str__(self) -> str:  # used for per-episode RNG seeding in the scripts
        return f"{self.data_file}#episode_{self.episode_index:08d}"


def _data_file(root: str, chunk: int, file: int) -> str:
    return os.path.join(root, "data", f"chunk-{chunk:03d}", f"file-{file:03d}.parquet")


def _video_file(root: str, key: str, chunk: int, file: int) -> str:
    return os.path.join(root, "videos", key, f"chunk-{chunk:03d}", f"file-{file:03d}.mp4")


def _with_row_starts(meta: pd.DataFrame) -> pd.DataFrame:
    """Row offset of each episode inside its data file.

    `dataset_from_index` is a global frame index; episodes in a file are contiguous, so the file's first
    global index is the minimum over *all* episodes stored in that file.
    """
    key = ["data/chunk_index", "data/file_index"]
    file_start = meta.groupby(key)["dataset_from_index"].transform("min")
    meta = meta.copy()
    meta["row_start"] = (meta["dataset_from_index"] - file_start).astype(np.int64)
    return meta


def list_episodes(root: str, task_indices: Sequence[int] | None = None, episodes: Sequence[int] | None = None) -> list[EpisodeRef]:
    meta = _with_row_starts(load_episodes_meta(root))
    if task_indices is not None:
        meta = meta[meta["task_index"].isin(list(task_indices))]
    if episodes is not None:
        meta = meta[meta["episode_index"].isin(list(episodes))]
    return [
        EpisodeRef(
            data_file=_data_file(root, int(r["data/chunk_index"]), int(r["data/file_index"])),
            row_start=int(r["row_start"]),
            length=int(r["length"]),
            episode_index=int(r["episode_index"]),
            task_index=int(r["task_index"]),
        )
        for _, r in meta.iterrows()
    ]


class _VideoReader:
    """Sequential-friendly PyAV reader for one mp4 file (keeps the container open, seeks when needed)."""

    def __init__(self, path: str):
        import av  # PyAV ships its own FFmpeg with HEVC decoding

        self.path = path
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.time_base = float(self.stream.time_base)
        self._iter = None
        self._last_pts_s = None
        self._last_frame = None

    def _restart_at(self, t: float):
        self.container.seek(int(max(t, 0.0) / self.time_base), stream=self.stream, backward=True, any_frame=False)
        self._iter = self.container.decode(self.stream)
        self._last_pts_s = None
        self._last_frame = None

    def frame_at(self, t: float, tolerance_s: float, max_forward_s: float = 2.0) -> np.ndarray:
        # Re-seek unless we can decode forward from the current position cheaply.
        if self._iter is None or self._last_pts_s is None or t < self._last_pts_s - 1e-6 or t - self._last_pts_s > max_forward_s:
            self._restart_at(t)
        if self._last_pts_s is not None and abs(self._last_pts_s - t) <= tolerance_s:
            return self._last_frame
        for frame in self._iter:
            pts_s = frame.pts * self.time_base
            if pts_s + tolerance_s < t:
                continue
            self._last_pts_s = pts_s
            self._last_frame = frame.to_ndarray(format="rgb24")
            if abs(pts_s - t) > tolerance_s:
                raise ValueError(f"{self.path}: wanted t={t:.4f}s, nearest decoded frame is at {pts_s:.4f}s")
            return self._last_frame
        raise ValueError(f"{self.path}: t={t:.4f}s is past the end of the video")


class B1KLeRobotV3Dataset:
    """Map-style dataset over selected tasks of the 2026 demos."""

    def __init__(
        self,
        root: str,
        task_indices: Sequence[int],
        camera_keys: Sequence[str],
        action_horizon: int,
        action_key: str = "action",
        episodes: Sequence[int] | None = None,
        tolerance_s: float = 1e-3,
        max_open_files: int = 4,
    ):
        self.root = os.path.expanduser(root)
        self.camera_keys = tuple(camera_keys)
        self.action_horizon = int(action_horizon)
        self.action_key = action_key
        self.tolerance_s = tolerance_s
        self.max_open_files = max_open_files

        with open(os.path.join(self.root, "meta", "info.json")) as f:
            self.info = json.load(f)
        self.fps = float(self.info["fps"])

        meta = _with_row_starts(load_episodes_meta(self.root))
        meta = meta[meta["task_index"].isin(list(task_indices))]
        if episodes is not None:
            meta = meta[meta["episode_index"].isin(list(episodes))]
        if len(meta) == 0:
            raise ValueError(f"No episodes found for tasks {tuple(task_indices)} under {self.root}")
        self._meta = meta.reset_index(drop=True)
        logger.info(f"LeRobot v3: {len(self._meta)} episodes for tasks {tuple(task_indices)}")

        lengths = self._meta["length"].to_numpy(dtype=np.int64)
        self._cum = np.concatenate([[0], np.cumsum(lengths)])
        self.episodes = self._meta["episode_index"].astype(int).tolist()

        # `meta.episodes` mirrors the lerobot v2 API used by ComputeSubtaskStateFromMeta (episode -> {"length"}).
        self.meta = types.SimpleNamespace(
            episodes={int(e): {"length": int(l)} for e, l in zip(self._meta["episode_index"], lengths)},
            fps=self.fps,
        )

        # Per-worker caches (created lazily so the dataset pickles cleanly into DataLoader workers)
        self._tables: collections.OrderedDict | None = None
        self._videos: collections.OrderedDict | None = None

    def __len__(self) -> int:
        return int(self._cum[-1])

    # -- caches -------------------------------------------------------------------------------------------------
    def _table(self, path: str) -> dict[str, np.ndarray]:
        if self._tables is None:
            self._tables = collections.OrderedDict()
        if path in self._tables:
            self._tables.move_to_end(path)
            return self._tables[path]
        df = pd.read_parquet(path, columns=list(LOW_DIM_COLUMNS))
        table = {
            "observation.state": np.stack(df["observation.state"].to_numpy()).astype(np.float32),
            "action": np.stack(df["action"].to_numpy()).astype(np.float32),
            "timestamp": df["timestamp"].to_numpy(dtype=np.float32),
            "frame_index": df["frame_index"].to_numpy(dtype=np.int64),
            "episode_index": df["episode_index"].to_numpy(dtype=np.int64),
            "index": df["index"].to_numpy(dtype=np.int64),
            "task_index": df["task_index"].to_numpy(dtype=np.int64),
        }
        self._tables[path] = table
        while len(self._tables) > self.max_open_files:
            self._tables.popitem(last=False)
        return table

    def _video(self, path: str) -> _VideoReader:
        if self._videos is None:
            self._videos = collections.OrderedDict()
        if path in self._videos:
            self._videos.move_to_end(path)
            return self._videos[path]
        reader = _VideoReader(path)
        self._videos[path] = reader
        while len(self._videos) > self.max_open_files * max(1, len(self.camera_keys)):
            _, old = self._videos.popitem(last=False)
            old.container.close()
        return reader

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tables"] = None
        state["_videos"] = None
        return state

    # -- sampling -----------------------------------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        idx = int(idx)
        if idx < 0:
            idx += len(self)
        ep = bisect.bisect_right(self._cum, idx) - 1
        frame = idx - int(self._cum[ep])
        row = self._meta.iloc[ep]
        length = int(row["length"])

        table = self._table(_data_file(self.root, int(row["data/chunk_index"]), int(row["data/file_index"])))
        r0 = int(row["row_start"])
        r = r0 + frame
        if int(table["episode_index"][r]) != int(row["episode_index"]):
            raise RuntimeError(f"Row mapping mismatch for episode {int(row['episode_index'])} frame {frame}")

        # Action chunk [t, t+H), padded with the episode's last action (same as lerobot's delta_timestamps).
        last = r0 + length - 1
        rows = np.minimum(np.arange(r, r + self.action_horizon), last)
        sample = {
            "observation.state": table["observation.state"][r],
            self.action_key: table["action"][rows],
            "timestamp": table["timestamp"][r],
            "frame_index": table["frame_index"][r],
            "episode_index": table["episode_index"][r],
            "index": table["index"][r],
            "task_index": table["task_index"][r],
        }

        t_in_episode = float(table["timestamp"][r])
        for key in self.camera_keys:
            path = _video_file(self.root, key, int(row[f"videos/{key}/chunk_index"]), int(row[f"videos/{key}/file_index"]))
            t = float(row[f"videos/{key}/from_timestamp"]) + t_in_episode
            sample[key] = self._video(path).frame_at(t, self.tolerance_s)
        return sample
