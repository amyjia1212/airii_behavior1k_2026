#!/bin/bash
# Training-pod setup for a new B1K/openpi task. No Isaac Sim / Xvfb here on purpose --
# train_b1k.py never imports omnigibson/isaacsim, confirmed on the source pod. That's a
# separate script for whenever you're ready to eval the resulting checkpoint.
#
# Run this on a brand-new pod as root. Assumes a network volume is mounted at /workspace
# (RunPod's default) -- persistent stuff (venv, repo) goes there so it survives pod restarts.
#
# [Release note] The original script wrote checkpoint_common.py / checkpoint_uploader.py /
# checkpoint_watchdog.py to /root with heredocs. Those files are kept next to this script in
# this repository and are copied instead; the code is unchanged.
set -euo pipefail

WORKDIR=/workspace
REPO_DIR="$WORKDIR/openpi"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== [1/6] installing uv (same native installer used on the source pod) ==="
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

echo "=== [2/6] cloning openpi repo onto the network volume ==="
mkdir -p "$WORKDIR"
if [ ! -d "$REPO_DIR" ]; then
  git clone https://github.com/wensi-ai/openpi.git "$REPO_DIR"
else
  echo "  $REPO_DIR already exists, skipping clone"
fi
cd "$REPO_DIR"

echo "=== [3/6] installing deps via uv sync (uses the repo's pyproject.toml + uv.lock) ==="
uv sync

echo "=== [4/6] Hugging Face auth ==="
echo "  needs to be able to read behavior-1k/2026-challenge-demos and push to your"
echo "  private checkpoint-backup repos. Run interactively:"
echo "    uv run hf auth login"
echo "  or set HF_TOKEN as an env var before running this script."
if [ -n "${HF_TOKEN:-}" ]; then
  echo "$HF_TOKEN" | uv run hf auth login --token stdin 2>&1 || true
else
  echo "  HF_TOKEN not set -- run 'uv run hf auth login' manually before training."
fi

echo "=== [5/6] installing the checkpoint safety-net scripts (uploader + watchdog) ==="
cp "$HERE/checkpoint_common.py" "$HERE/checkpoint_uploader.py" "$HERE/checkpoint_watchdog.py" /root/

echo "  NOTE: edit DEFAULT_REPO_SUFFIX in /root/checkpoint_uploader.py for the new task's"
echo "  backup repo name before starting the uploader daemon (currently still says"
echo "  'cook-broccolini' -- copied verbatim from the source pod)."

echo "=== [6/6] done with generic setup ==="
echo
echo "STILL NEEDED before you can actually launch training (task-specific, not automated here):"
echo "  1. Add a new TrainConfig entry in $REPO_DIR/src/openpi/training/config.py for your"
echo "     new task (copy the pi05_b1k_task90 block as a template -- new 'name=', new"
echo "     'dataset_root=' path)."
echo "  2. Figure out how dataset_root actually gets scoped to just the new task's demos --"
echo "     I could NOT verify this mechanism from the source pod (the local dataset caches"
echo "     were already cleaned up, and there's no data-prep script in the repo). Confirm"
echo "     this before a multi-hour run, not after."
echo "  3. Once both of those are sorted, launch training (same pattern as prior runs):"
echo "     cd $REPO_DIR"
echo "     XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 HF_HUB_OFFLINE=1 uv run scripts/b1k/train_b1k.py \\"
echo "       <new_config_name> --exp_name=<new_task> --overwrite --batch_size=64 \\"
echo "       > /root/train.log 2>&1 &"
echo "  4. Start the safety-net daemons pointed at the new checkpoint dir:"
echo "     uv run python /root/checkpoint_uploader.py \\"
echo "       ./outputs/checkpoints/<new_config_name>/<new_task> --interval 600 &"
echo "     uv run python /root/checkpoint_watchdog.py \\"
echo "       ./outputs/checkpoints/<new_config_name>/<new_task> --keep 1 --interval 300 &"
