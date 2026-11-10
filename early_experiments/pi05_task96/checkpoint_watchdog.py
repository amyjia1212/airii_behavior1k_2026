#!/usr/bin/env python3
"""Watchdog that prunes old openpi training checkpoints safely.

Pruning rule (all conditions must hold for any deletion):
  1. The NEWEST checkpoint is verified complete (orbax markers + stability).
  2. Every checkpoint being deleted has a confirmed HF Hub upload marker
     (written by checkpoint_uploader.py AFTER a verified, successful upload).

This guarantees there is always a resumable checkpoint on disk AND that no
checkpoint is pruned before it is safely backed up remotely.

Usage:
  checkpoint_watchdog.py <checkpoint_dir> [--keep N] [--interval SEC]
      [--settle SEC] [--passes N] [--dry-run] [--once]
"""
import argparse
import logging
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from checkpoint_common import has_upload_marker, is_verified_complete, list_step_dirs

LOG_FILE = "/root/checkpoint_watchdog.log"


def run_once(checkpoint_dir: str, keep: int, settle_s: float, passes: int, dry_run: bool, log) -> None:
    steps = list_step_dirs(checkpoint_dir)

    if len(steps) <= keep:
        log.info("No pruning needed: %d checkpoint(s) present (keep=%d)", len(steps), keep)
        return

    newest_step, newest_dir = steps[-1]
    if not is_verified_complete(newest_dir, settle_s, passes):
        log.info(
            "Newest checkpoint %d NOT verified complete yet - keeping all older checkpoints",
            newest_step,
        )
        return

    for step, p in steps[:-keep]:
        if not has_upload_marker(p):
            log.warning(
                "Checkpoint %d has NO confirmed HF upload yet - keeping it (upload-confirm-before-prune)",
                step,
            )
            continue
        log.info("Deleting old checkpoint %d (newest %d verified complete, upload confirmed)", step, newest_step)
        if not dry_run:
            shutil.rmtree(p)
            log.info("Deleted %s", p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint_dir")
    ap.add_argument("--keep", type=int, default=1, help="number of newest checkpoints to retain")
    ap.add_argument("--interval", type=int, default=300, help="seconds between scans")
    ap.add_argument("--settle", type=float, default=60.0, help="min age (s) of completion marker")
    ap.add_argument("--passes", type=int, default=2, help="stability checks per verification")
    ap.add_argument("--dry-run", action="store_true", help="log decisions without deleting")
    ap.add_argument("--once", action="store_true", help="run a single scan and exit")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
    )
    log = logging.getLogger("watchdog")

    log.info("Watchdog started: dir=%s keep=%d interval=%ds settle=%.0fs", args.checkpoint_dir, args.keep, args.interval, args.settle)
    while True:
        try:
            run_once(args.checkpoint_dir, args.keep, args.settle, args.passes, args.dry_run, log)
        except Exception as exc:  # never let a scan failure kill the watcher
            log.exception("Scan failed: %s", exc)
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
