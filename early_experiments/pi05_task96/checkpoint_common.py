#!/usr/bin/env python3
"""Shared helpers for the openpi checkpoint watchdog + HF Hub uploader.

Single source of truth for:
  - listing step checkpoint dirs
  - verifying a checkpoint is fully written (orbax completion markers + stability)
  - the upload marker used to gate pruning on confirmed HF uploads
"""
import os
import time

UPLOAD_MARKER = ".hf_uploaded"


def list_step_dirs(checkpoint_dir: str):
    """Return [(step:int, path:str), ...] sorted ascending, numeric dirs only."""
    steps = []
    for name in os.listdir(checkpoint_dir):
        p = os.path.join(checkpoint_dir, name)
        if os.path.isdir(p) and name.isdigit():
            steps.append((int(name), p))
    steps.sort()
    return steps


def file_count(root: str) -> int:
    n = 0
    for _, _, files in os.walk(root):
        n += len(files)
    return n


def is_verified_complete(step_dir: str, settle_s: float = 60.0, passes: int = 2) -> bool:
    """True only if the step dir is fully written and stable (orbax write order)."""
    markers = (
        "_CHECKPOINT_METADATA",
        os.path.join("params", "manifest.ocdbt"),
        os.path.join("train_state", "manifest.ocdbt"),
    )
    for rel in markers:
        p = os.path.join(step_dir, rel)
        if not os.path.isfile(p) or os.path.getsize(p) == 0:
            return False

    meta = os.path.join(step_dir, "_CHECKPOINT_METADATA")
    if time.time() - os.path.getmtime(meta) < settle_s:
        return False

    counts = [file_count(step_dir)]
    for _ in range(max(passes - 1, 0)):
        time.sleep(settle_s / max(passes, 1))
        counts.append(file_count(step_dir))
    return len(set(counts)) == 1


def has_upload_marker(step_dir: str) -> bool:
    return os.path.isfile(os.path.join(step_dir, UPLOAD_MARKER))


def write_upload_marker(step_dir: str, info: str) -> str:
    p = os.path.join(step_dir, UPLOAD_MARKER)
    with open(p, "w") as f:
        print(info.rstrip(), file=f)
    return p
