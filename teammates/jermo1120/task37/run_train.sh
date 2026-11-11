#!/bin/bash
# Run the 25k task 37 training. On exit, kill orphaned data-loader workers so they can't
# eat into the 125 GB container memory limit (they survive the trainer being OOM-killed).
cd /root/work/behavior-1k-solution
HF_HOME=/root/.cache/huggingface UV_CACHE_DIR=/root/.cache/uv XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
  .venv/bin/python3 scripts/train.py pi_behavior_b1k_task37_clean_a_trumpet --log_interval=50 --num_workers=8 "$@" &
echo $! > /root/work/train.pid
wait $!
echo "train exited with code $? at $(date)"
for p in $(ps -eo pid=,cmd= | grep "[m]ultiprocessing.spawn" | grep behavior-1k-solution | awk '{print $1}'); do kill -9 $p; done
echo "cleaned up data-loader workers"
