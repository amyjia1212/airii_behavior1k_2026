"""Rerun every eval affected by the stage-table bug (2026-10-14), single-task models first, on all five GPUs.

Until 2026-10-14 pi_behavior_config.TASK_NUM_STAGES held counts reconstructed from demo lengths, which index the
checkpoints' task_stage_embeddings at the wrong rows; it now holds the training team's counts. run_all.py tags the new
results "-stagefix", so nothing old is overwritten.

JOBS run in order, each as one `run_all.py --no-wandb` on the next free GPU (B1K_PROCS=2), one scene load at a time:
a job starts only once the previously started one has loaded its scenes ("Loaded N policy instance" in all of its
evaluator logs), has ended, or LOAD_MAX_MIN have passed, and only while /workspace answers. Jobs of the same model may
run on several GPUs at once, so W&B is logged per (model, settings) by this script once all its jobs have ended.
Chunk checkpoints (step_049999) are staged from /workspace/ckpts into /dev/shm/ckpts in a background thread
(copy_ckpts.py checks every file's sha256 against HF); single-task checkpoints are already there.

  cd /root/b1k_eval && setsid nohup /root/behavior-1k-solution/.venv/bin/python rerun_stagefix.py \
      >> /root/setup_logs/rerun_stagefix.log 2>&1 < /dev/null &
Rerunning it skips (model, checkpoint, task) results that are complete; jobs still running are not detected, so stop
any leftover run_all.py first.
"""

import glob
import os
import subprocess
import threading
import time

PY = "/root/behavior-1k-solution/.venv/bin/python"
GPUS = [4, 3, 2, 0, 1]
LOAD_MAX_MIN = 45
SHM = "/dev/shm/ckpts"
CHUNKS = {  # chunk model -> tasks (run_all.MODELS)
    "b1k-2026-t50-70-72-85": ["freeze_fruit", "thawing_frozen_food", "cook_a_frozen_pie", "putting_dirty_dishes_in_sink"],
    "b1k-2026-t52-53-73-66-86": ["sorting_bottles_cans_and_paper", "tidying_living_room", "organizing_school_stuff",
                                 "turning_out_all_lights_before_sleep", "make_gift_bags_for_baby_showers"],
    "b1k-2026-t65-71-74-75": ["unloading_the_car", "clean_your_rusty_garden_tools", "carrying_out_garden_furniture",
                              "put_together_a_basic_pruning_kit"],
    "b1k-2026-t79-81-84-88": ["polishing_shoes", "packing_meal_for_delivery", "tidying_bathroom", "rearrange_your_room"],
    "b1k-2026-t94-95-98": ["dispose_of_batteries", "cook_brussels_sprouts", "laying_tile_floors"],
}


def job(model, ckpts, n, label, tasks=None, gr=90):
    return dict(model=model, ckpts=ckpts, n=n, label=label, tasks=tasks, gr=gr)


SINGLE3 = ["b1k-2026-t52", "b1k-2026-t50", "b1k-2026-t72"]
JOBS = (
    [job(m, ["step_009999", "step_008000"], 5, "10k8k") for m in SINGLE3]
    + [job(f"b1k-2026-t{t}", ["step_005999"], 10, "6k") for t in (53, 66, 71, 65)]
    + [job(m, ["step_004000", "step_006000"], 5, "4k6k") for m in SINGLE3]
    + [job("b1k-2026-t52", ["step_009999"], 5, "10k-gr0", gr=0)]
    + [job(m, ["step_002000"], 10, "2k") for m in SINGLE3]
    + [job(m, ["step_049999"], 10, t, tasks=[t]) for m, ts in CHUNKS.items() for t in ts]
)

lock = threading.Lock()
staged = {}  # chunk model -> threading.Event, set once its checkpoint is in /dev/shm (or failed: see stage_ok)
stage_ok = {}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def volume_ok():
    try:
        return subprocess.run(["ls", "/workspace/eval_results"], capture_output=True, timeout=60).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def tag(j):
    return f"-rhc10-gr{j['gr']}-stagefix"


def env_for(j, gpu=None, logs=None):
    e = dict(os.environ, B1K_MODEL=j["model"], B1K_PROCS="2", B1K_INSTANCE_OFFSET="0", B1K_NUM_INSTANCES=str(j["n"]),
             B1K_GRIPPER_RECOVERY_STEPS=str(j["gr"]), B1K_CKPT_ROOT=SHM)
    if gpu is not None:
        e.update(B1K_GPU=str(gpu), B1K_LOG_ROOT=logs)
    return e


def complete(j):
    tasks = j["tasks"] or list(task_names(j["model"]))
    root = f"/workspace/eval_results/{j['model']}{tag(j)}"
    return all(len(glob.glob(f"{root}/{c}/{t}/json/*.json")) >= j["n"] for c in j["ckpts"] for t in tasks)


def task_names(model):
    out = subprocess.check_output([PY, "-c", "import run_all; print('\\n'.join(run_all.TASKS.values()))"],
                                  env=dict(os.environ, B1K_MODEL=model), cwd="/root/b1k_eval", text=True)
    return [t for t in out.splitlines() if t]


def stager():
    for m in CHUNKS:
        if os.path.isdir(f"{SHM}/{m}/step_049999/params"):
            # chunk 2 was staged (and verified) by restage.sh on 2026-10-13
            stage_ok[m] = True
        else:
            log(f"staging {m}/step_049999 from /workspace/ckpts")
            out = subprocess.run([PY, "/root/setup_logs/copy_ckpts.py", m], capture_output=True, text=True,
                                 env=dict(os.environ, HF_HOME="/dev/shm/hf_home"))
            ok = f"COPY {m}" in out.stdout and " OK " in out.stdout.split(f"COPY {m}")[-1]
            stage_ok[m] = ok
            log(f"staged {m}: {'OK' if ok else 'FAILED'} {out.stdout.strip().splitlines()[-2:] if out.stdout else out.stderr[-300:]}")
        staged[m].set()


def loaded(logs, procs=2):
    n = 0
    for f in glob.glob(f"{logs}/eval_*.log"):
        try:
            n += sum("policy instance" in l and "Loaded" in l for l in open(f, errors="replace"))
        except OSError:
            pass
    return n >= procs


def main():
    for m in CHUNKS:
        staged[m] = threading.Event()
    threading.Thread(target=stager, daemon=True).start()
    running = {}  # gpu -> (Popen, job, logs)
    pending = [j for j in JOBS if not complete(j)]
    log(f"{len(pending)} of {len(JOBS)} jobs to run on GPUs {GPUS}")
    last = None  # (Popen, logs, start time) of the most recently started job
    finished = []
    while pending or running:
        for g, (p, j, logs) in list(running.items()):
            if p.poll() is not None:
                del running[g]
                finished.append(j)
                log(f"GPU {g}: {j['model']} {j['label']} ended with {p.returncode}, complete={complete(j)}")
                wandb_if_done(j, pending, running)
        free = [g for g in GPUS if g not in running]
        nxt = next((j for j in pending if j["model"] not in CHUNKS or staged[j["model"]].is_set()), None)
        if free and nxt:
            if nxt["model"] in CHUNKS and not stage_ok.get(nxt["model"]):
                pending.remove(nxt)
                log(f"skipping {nxt['model']} {nxt['label']}: checkpoint not staged")
                continue
            gate = last is None or last[0].poll() is not None or loaded(last[1]) \
                or time.time() - last[2] > LOAD_MAX_MIN * 60
            if gate and volume_ok():
                g = free[0]
                logs = f"/workspace/eval_logs/{nxt['model']}{tag(nxt)}-{nxt['label']}-gpu{g}"
                os.makedirs(logs, exist_ok=True)
                cmd = [PY, "run_all.py", "--checkpoints", *nxt["ckpts"], "--retries", "1", "--no-wandb"]
                if nxt["tasks"]:
                    cmd += ["--tasks", *nxt["tasks"]]
                p = subprocess.Popen(cmd, cwd="/root/b1k_eval", env=env_for(nxt, g, logs),
                                     stdout=open(f"{logs}/run_all.log", "a"), stderr=subprocess.STDOUT,
                                     start_new_session=True)
                running[g] = (p, nxt, logs)
                pending.remove(nxt)
                last = (p, logs, time.time())
                log(f"GPU {g}: START {nxt['model']} {nxt['label']} {nxt['ckpts']} n={nxt['n']} gr={nxt['gr']} "
                    f"pid {p.pid} logs {logs}")
                continue
        time.sleep(30)
    log("ALL DONE")


def wandb_if_done(j, pending, running):
    """Log W&B for j's (model, settings) once no job of that pair is pending or running."""
    key = (j["model"], j["gr"])
    if any((x["model"], x["gr"]) == key for x in pending) or any((x["model"], x["gr"]) == key for _, x, _ in running.values()):
        return
    if j["model"] in CHUNKS and any(x["model"] == j["model"] for x in pending):
        return
    e = env_for(j)
    for t in task_names(j["model"]):
        subprocess.call([PY, "/root/b1k_eval/log_wandb.py", "--task", t], cwd="/root/b1k_eval", env=e,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rc = subprocess.call([PY, "/root/b1k_eval/log_wandb.py", "--finalize"], cwd="/root/b1k_eval", env=e,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log(f"W&B logged {j['model']}{tag(j)} (finalize exit {rc})")


if __name__ == "__main__":
    main()
