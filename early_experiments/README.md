# Early experiments (not used in the final submission)

Before switching to the PI_BEHAVIOR model, we fine-tuned the official openpi **π0.5** baseline
(`pi05_b1k` config from the [`wensi-ai/openpi`](https://github.com/wensi-ai/openpi) `behavior` branch) on single tasks.
The checkpoints are on Hugging Face:

| Repo | Tasks |
|---|---|
| [`amyjia1212/pi05-b1k-cook-broccolini-checkpoints`](https://huggingface.co/amyjia1212/pi05-b1k-cook-broccolini-checkpoints) | 96 |
| [`amyjia1212/pi05-b1k-composting-waste-checkpoints`](https://huggingface.co/amyjia1212/pi05-b1k-composting-waste-checkpoints) | 90 |
| [`amyjia1212/pi05-b1k-clean-a-keyboard-checkpoints`](https://huggingface.co/amyjia1212/pi05-b1k-clean-a-keyboard-checkpoints) | 93 |
| [`amyjia1212/pi05-b1k-joint-90-93-96-checkpoints`](https://huggingface.co/amyjia1212/pi05-b1k-joint-90-93-96-checkpoints) | 90, 93, 96 |

The files in [`pi05_task96/`](pi05_task96/) are the **original** scripts from that phase (recovered from our notes):

* `setup_training_pod.sh`: training-pod setup (uv, openpi clone, HF login). The only changes from the original are that it
  copies the three Python files below instead of writing them with heredocs, and a fixed `$HOME`/`$PATH` in the `PATH` line.
* `checkpoint_uploader.py` / `checkpoint_watchdog.py` / `checkpoint_common.py`: a safety net for pod training. The uploader
  pushes every fully-written checkpoint to a private HF repo and verifies the remote file count. The watchdog deletes old
  local checkpoints only after their upload is confirmed, so the pod disk never fills up and nothing is lost.

## Notes from the task-96 run (cook_broccolini)

* **Setup:** 1× A100-SXM4-80GB on RunPod, CUDA 12.8; openpi `behavior` branch at `436812bd` (torch 2.7.1+cu128, JAX 0.5.3,
  the BEHAVIOR lerobot fork 0.5.2). `pi05_b1k` with `repo_id=behavior-1k/2026-challenge-demos`, 50k steps, batch 64,
  checkpoints every 10k steps, π0.5 base weights.
* **Data:** we downloaded only task 96's chunk (episodes 19200–19399, about 21.7 GB with video) plus `meta/` and the
  annotations, instead of the 3.27 TB dataset. Many-small-file downloads hit HTTP 429; `hf download --max-workers 4` with
  `HF_HUB_DISABLE_XET=1` worked reliably.
* **Issues in the lerobot fork with a single-task subset**, patched locally in the installed package:
  1. data rows carry global episode indices (19200–19399), while the local episode table is indexed 0–199. The lookups were
     offset by the table's base index;
  2. the chunk's episode table was padded to 20,000 rows so that row position equals global episode index;
  3. `LeRobotDataset.__init__` falls back to downloading the full dataset when the local load fails, so it was forced to load locally;
  4. `num_frames` came from `info.json` (the full dataset, about 210M frames), so it was changed to the loaded subset's count (about 820k).

  Everything ran with `HF_HUB_OFFLINE=1`, so an accidental download fails instead of pulling terabytes.
* **Eval plan:** `serve_b1k.py` plus the OmniGibson evaluator (BEHAVIOR-1K v3.9.1, `RGBDFullResWrapper`, 10 instances),
  comparing a few checkpoints.

These subset problems are why the final pipeline uses its own LeRobot v3 reader, `src/b1k/training/lerobot_v3.py`. It reads
the shared parquet and video files directly by global episode index, without patching lerobot.
