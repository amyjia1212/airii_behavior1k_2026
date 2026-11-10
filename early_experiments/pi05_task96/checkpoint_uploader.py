#!/usr/bin/env python3
"""Upload verified openpi checkpoints to a private HF Hub repo.

Rules:
  - ONLY checkpoints that pass the watchdog's verification
    (is_verified_complete: orbax completion markers + file stability)
    are uploaded. Incomplete checkpoints are never touched.
  - After the upload is CONFIRMED (remote file count matches local),
    a marker file (.hf_uploaded) is written inside the checkpoint dir.
    The watchdog only prunes checkpoints carrying this marker, so nothing
    is ever deleted locally before it exists remotely.
  - Older remote snapshots are pruned (keep last N) only after a new
    upload has succeeded.

Usage:
  checkpoint_uploader.py <checkpoint_dir> [--repo REPO] [--interval SEC]
      [--settle SEC] [--passes N] [--keep-remote N] [--dry-run] [--once]
"""
import argparse
import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoint_common import (
    UPLOAD_MARKER,
    file_count,
    has_upload_marker,
    is_verified_complete,
    list_step_dirs,
    write_upload_marker,
)

from huggingface_hub import HfApi, RepoFile, RepoFolder

LOG_FILE = "/root/checkpoint_uploader.log"
DEFAULT_REPO_SUFFIX = "pi05-b1k-cook-broccolini-checkpoints"  # <-- CHANGE for the new task


def _list_tree(api, repo: str, path: str):
    """Version-tolerant list_repo_tree (path_in_repo vs path kwarg)."""
    try:
        return api.list_repo_tree(repo_id=repo, path=path, recursive=True)
    except TypeError:
        return api.list_repo_tree(repo_id=repo, path_in_repo=path, recursive=True)


def _is_file(item) -> bool:
    """Version-tolerant file check: hub>=1.16 returns RepoFile/RepoFolder,
    older versions return HfFileInfo with a .type attribute."""
    return isinstance(item, RepoFile) or getattr(item, "type", None) == "file"


def _is_folder(item) -> bool:
    return isinstance(item, RepoFolder) or getattr(item, "type", None) == "folder"


def resolve_repo(api, repo: str | None, log) -> str:
    if not repo:
        who = api.whoami()
        repo = f"{who['name']}/{DEFAULT_REPO_SUFFIX}"
    if not api.repo_exists(repo):
        api.create_repo(repo_id=repo, private=True, exist_ok=True)
        log.info("Created private repo %s", repo)
    return repo


def upload_step(api, repo: str, step: int, step_dir: str, log, dry_run: bool) -> bool:
    local_count = file_count(step_dir)
    log.info("Uploading checkpoint %d (%d files) -> %s/checkpoints/%d", step, local_count, repo, step)
    if dry_run:
        return True
    api.upload_folder(
        repo_id=repo,
        folder_path=step_dir,
        path_in_repo=f"checkpoints/{step}",
        commit_message=f"Checkpoint step {step}",
        ignore_patterns=[UPLOAD_MARKER],
    )
    remote = _list_tree(api, repo, f"checkpoints/{step}")
    remote_files = [f for f in remote if _is_file(f)]
    if len(remote_files) != local_count:
        log.error(
            "Remote verification FAILED for step %d: remote=%d files vs local=%d - NOT marking uploaded",
            step,
            len(remote_files),
            local_count,
        )
        return False
    log.info("Remote verified: %d/%d files for step %d", len(remote_files), local_count, step)
    return True


def prune_remote(api, repo: str, keep_remote: int, log, dry_run: bool) -> None:
    """Keep only the newest N remote checkpoint snapshots (non-fatal on error)."""
    try:
        items = _list_tree(api, repo, "checkpoints")
        dirs = []
        for f in items:
            if _is_folder(f):
                try:
                    dirs.append((int(f.path.split("/")[-1]), f.path))
                except ValueError:
                    continue
        for _step, path in sorted(dirs)[:-keep_remote]:
            if dry_run:
                log.info("[dry-run] would delete remote %s", path)
                continue
            api.delete_folder(repo_id=repo, path_in_repo=path, commit_message=f"Prune {path}")
            log.info("Deleted remote %s", path)
    except Exception as exc:
        log.warning("Remote prune failed (non-fatal): %s", exc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--repo", default=None, help="HF repo id (default: <user>/pi05-b1k-cook-broccolini-checkpoints)")
    ap.add_argument("--interval", type=int, default=600, help="seconds between scans")
    ap.add_argument("--settle", type=float, default=60.0, help="min age (s) of completion marker")
    ap.add_argument("--passes", type=int, default=2, help="stability checks per verification")
    ap.add_argument("--keep-remote", type=int, default=2, help="remote snapshots to retain after upload")
    ap.add_argument("--dry-run", action="store_true", help="log decisions without uploading")
    ap.add_argument("--once", action="store_true", help="run a single scan and exit")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("uploader")

    api = HfApi()
    log.info("Uploader started: dir=%s interval=%ds settle=%.0fs keep_remote=%d", args.checkpoint_dir, args.interval, args.settle, args.keep_remote)
    repo = None

    while True:
        try:
            if args.dry_run:
                repo = args.repo or "dry-run-repo"
            elif repo is None:
                repo = resolve_repo(api, args.repo, log)
                log.info("Using repo: %s", repo)
            steps = list_step_dirs(args.checkpoint_dir)
            pending = [(s, p) for s, p in steps if not has_upload_marker(p)]
            for step, p in reversed(pending):  # newest first
                if not is_verified_complete(p, args.settle, args.passes):
                    log.info("Checkpoint %d not verified complete yet - skipping upload", step)
                    continue
                if upload_step(api, repo, step, p, log, args.dry_run):
                    if args.dry_run:
                        log.info("[dry-run] would write upload marker for step %d", step)
                    else:
                        write_upload_marker(p, f"uploaded to {repo} at {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
                        log.info("Upload marker written for step %d", step)
                        prune_remote(api, repo, args.keep_remote, log, False)
            if not pending and not args.dry_run:
                log.info("No pending uploads (all checkpoints verified + uploaded)")
            if args.once:
                break
            time.sleep(args.interval)
        except Exception as exc:
            log.exception("Uploader pass failed: %s", exc)
            if args.once:
                raise
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
