"""Keep all five GPUs busy with chunks 2-5: one worker thread per GPU pulls (model, task) pairs from a shared queue.

Replaces the fixed per-GPU task lists of run_chunks_multigpu.sh. Each task is one `run_all.py --tasks <task>` job on
the worker's GPU with B1K_PROCS evaluator processes (see run_all.py). A GPU prefers its own list in JOBS and, once
that is empty, takes tasks from the GPU with the most left, so no GPU idles while tasks remain.

- Jobs already running when this starts (run_chunks_multigpu.sh, or an earlier scheduler) are left alone until their
  current task ends, then killed and their remaining tasks join the queue.
- A job that fails or whose logs stop moving for STALL_MIN minutes is killed and its task requeued with one process
  fewer (3 -> 2 -> 1); after MAX_ATTEMPTS it is given up.
- Phase 1 = public-test indices 0-9 (the requested run). Phase 2 = indices 10-19 of the same tasks, picked by a
  GPU only when no phase-1 task is waiting, into separate `-inst10-19` results dirs / W&B runs.
  `touch /dev/shm/b1k_sched/no_phase2` skips phase 2.
- `echo 2 > /dev/shm/b1k_sched/procs` changes the processes per task for jobs started from then on (default 3).
- W&B summaries (`log_wandb.py --finalize`) are written per model once a phase has no tasks left.
- Staggered starts: the /workspace volume (geesefs) dropped ~20 min after six simulator processes began loading
  scenes from it at once, so only one job loads at a time. A new job starts only once the previously started job
  has loaded its scene (every evaluator log says "Loaded N policy instance(s)"), has ended, or LOAD_MAX_MIN minutes
  have passed, and never less than START_GAP_MIN minutes after the previous start.
- /workspace errors do not kill the scheduler: reads are retried until the volume answers again, no job is started
  while it does not, and a task that ends while the volume is failing is not counted as a failure.
  `touch /dev/shm/b1k_sched/hold` also stops new starts (running jobs continue); remove the file to resume.
- A model's tasks are only handed out once /dev/shm/ckpts/<model>/step_049999.ready exists (checkpoint staged).
- The log lines are mirrored to /dev/shm/b1k_sched/scheduler.log, which stays readable if /workspace is gone.

  cd /root/b1k_eval && setsid nohup /root/behavior-1k-solution/.venv/bin/python \
      /workspace/eval_work/code/schedule_multigpu.py >> /workspace/eval_logs/scheduler.log 2>&1 < /dev/null &
Rerun the same command after a crash: finished tasks are skipped, running jobs are adopted as described above.
"""

import os
import signal
import subprocess
import threading
import time

PY = "/root/behavior-1k-solution/.venv/bin/python"
CKPT = "step_049999"
CTRL = "/dev/shm/b1k_sched"
STALL_MIN = 60
MAX_ATTEMPTS = 3
START_GAP_MIN = 3  # minimum spacing between two job starts
LOAD_MAX_MIN = 45  # a job still loading its scene after this long no longer blocks the next start
VOLUME = "/workspace/eval_results"
JOBS = {  # gpu -> (model, tasks): the original assignment, now only a preference
    0: ("b1k-2026-t52-53-73-66-86", ["sorting_bottles_cans_and_paper", "tidying_living_room", "organizing_school_stuff"]),
    1: ("b1k-2026-t52-53-73-66-86", ["turning_out_all_lights_before_sleep", "make_gift_bags_for_baby_showers"]),
    2: ("b1k-2026-t65-71-74-75", ["unloading_the_car", "clean_your_rusty_garden_tools", "carrying_out_garden_furniture",
                                  "put_together_a_basic_pruning_kit"]),
    3: ("b1k-2026-t79-81-84-88", ["polishing_shoes", "packing_meal_for_delivery", "tidying_bathroom", "rearrange_your_room"]),
    4: ("b1k-2026-t94-95-98", ["dispose_of_batteries", "cook_brussels_sprouts", "laying_tile_floors"]),
}
OFFSETS = [0, 10]  # phase 1, phase 2

lock = threading.Lock()
claimed = {}  # (offset, model, task) -> gpu
failures = {}  # (offset, model, task) -> failed attempts
finalized = set()  # (offset, model)
start_gate = threading.Lock()  # held while a worker waits for its turn to start a job
last_start = {"t": 0.0, "logs": None, "n": 0, "marks": 0, "proc": None}  # the most recently started job
volume_bad = {"t": 0.0}  # last time /workspace failed to answer


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    try:
        with open(os.path.join(CTRL, "scheduler.log"), "a") as f:
            f.write(line + "\n")
    except OSError:
        pass
    try:
        print(line, flush=True)  # stdout is a file on /workspace
    except OSError:
        pass


def volume_ok():
    """True if /workspace answers. Checked in a child process so a hung FUSE mount cannot block the scheduler."""
    try:
        ok = subprocess.run(["ls", VOLUME], capture_output=True, timeout=60).returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        ok = False
    if not ok:
        volume_bad["t"] = time.time()
    return ok


def wait_volume(what):
    """Block until /workspace answers again."""
    waited = False
    while not volume_ok():
        if not waited:
            log(f"WARNING: /workspace is not readable ({what}); retrying every 60 s")
        waited = True
        time.sleep(60)
    if waited:
        log(f"/workspace is readable again ({what})")


def run_tag(offset):
    return "-rhc10-gr90" + ("" if offset == 0 else f"-inst{offset}-{offset + 9}")


def done(key):
    """True if the task has its 10 result JSONs. glob would turn a dead volume into "no results", so list instead."""
    offset, model, task = key
    path = f"/workspace/eval_results/{model}{run_tag(offset)}/{CKPT}/{task}/json"
    while True:
        try:
            return sum(f.endswith(".json") for f in os.listdir(path)) >= 10
        except FileNotFoundError:
            if volume_ok():
                return False
        except OSError as e:
            volume_bad["t"] = time.time()
            log(f"WARNING: cannot list {path}: {e}")
        wait_volume("counting results")


def phase2_enabled():
    return not os.path.exists(os.path.join(CTRL, "no_phase2"))


def ckpt_ready(model):
    """Checkpoint fully staged in /dev/shm (marker written by download_ckpts_to_shm.py / copy_ckpts_to_shm.py)."""
    return os.path.exists(f"/dev/shm/ckpts/{model}/{CKPT}.ready")


def open_keys(offset, gpu, staged=True):
    """Unclaimed, unfinished tasks of GPU `gpu`'s list (only those whose checkpoint is staged, unless staged=False).
    Call with the lock held."""
    model, tasks = JOBS[gpu]
    if staged and not ckpt_ready(model):
        return []
    return [
        k for k in ((offset, model, t) for t in tasks)
        if k not in claimed and failures.get(k, 0) < MAX_ATTEMPTS and not done(k)
    ]


def pick(gpu):
    """Claim the next task for `gpu`: its own list first, else the last task of the GPU with the most left."""
    with lock:
        for offset in OFFSETS:
            if offset and not phase2_enabled():
                break
            own = open_keys(offset, gpu)
            other = max((open_keys(offset, g) for g in JOBS if g != gpu), key=len)
            key = own[0] if own else other[-1] if other else None
            if key:
                claimed[key] = gpu
                return key
    return None


def children(pid, pattern):
    out = subprocess.run(["pgrep", "-P", str(pid), "-f", pattern], capture_output=True, text=True).stdout
    return {int(p) for p in out.split()}


def alive(pid):
    try:
        os.kill(pid, 0)
        return open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def environ(pid):
    try:
        raw = open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace")
    except OSError:
        return {}
    return dict(kv.split("=", 1) for kv in raw.split("\0") if "=" in kv)


def gpu_mem(gpu):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", str(gpu)],
        capture_output=True, text=True,
    ).stdout
    return int(out.strip() or 0)


def cleanup(gpu, run_all_pid=None):
    """Kill a GPU's job (run_all.py, its policy server and evaluators) and wait until the GPU is empty."""
    if run_all_pid and alive(run_all_pid):
        try:
            os.killpg(run_all_pid, signal.SIGTERM)
        except OSError:
            pass
    ports = f"(8{gpu:03d}|81{gpu:02d})"  # the first launch used 8000+GPU, run_all.py now uses 8100+GPU
    pattern = rf"(serve_b1k_2026|eval_multi_ckpt)\.py .*--port {ports} "
    for sig in ("-TERM", "-KILL"):
        subprocess.run(["pkill", sig, "-f", pattern])
        for _ in range(20):
            if subprocess.run(["pgrep", "-f", pattern], capture_output=True).returncode != 0:
                break
            time.sleep(1)
    if run_all_pid and alive(run_all_pid):
        try:
            os.killpg(run_all_pid, signal.SIGKILL)
        except OSError:
            pass
    for _ in range(60):
        if gpu_mem(gpu) < 1500:
            return
        time.sleep(3)
    log(f"GPU {gpu}: WARNING {gpu_mem(gpu)} MiB still in use after cleanup")


def adopt(gpu, pid, key):
    """Let an already running run_all.py finish its current task, then kill it."""
    log(f"GPU {gpu}: adopted running job pid {pid}, waiting for {key[2]} to end")
    first = children(pid, "eval_multi_ckpt.py")
    gone_since = None
    while alive(pid):
        now = children(pid, "eval_multi_ckpt.py")
        if not (now & first):  # the evaluators seen at start have exited; run_all.py logs to W&B, then starts the next
            gone_since = gone_since or time.time()
            if now or time.time() - gone_since > 20 * 60:
                break
        time.sleep(10)
    log(f"GPU {gpu}: {key[2]} ended ({'done' if done(key) else 'INCOMPLETE'}), stopping the old job")
    cleanup(gpu, pid)


def procs_for(key):
    """Processes per task: /dev/shm/b1k_sched/procs, or procs_<task> for one task (e.g. to resume a partly finished
    task with the instance split it was started with, so that no finished instance is run again)."""
    base = 3
    for name in ("procs", f"procs_{key[2]}"):
        try:
            base = int(open(os.path.join(CTRL, name)).read())
        except (OSError, ValueError):
            pass
    return max(1, base - failures.get(key, 0))


def log_dir(gpu, key):
    return f"/workspace/eval_logs/{key[1]}{run_tag(key[0])}-gpu{gpu}"


def newest_mtime(path):
    """Newest mtime in a log dir; None if /workspace did not answer."""
    try:
        return max((e.stat().st_mtime for e in os.scandir(path)), default=0)
    except OSError:
        volume_bad["t"] = time.time()
        return None


def load_marks(logs):
    """Number of "scene and policies loaded" lines in a job's evaluator logs (they are appended to across attempts)."""
    try:
        out = subprocess.run(
            f"cat {logs}/eval_*.log 2>/dev/null | grep -cE 'Loaded [0-9]+ policy instance|All checkpoints already evaluated'", shell=True,
            capture_output=True, text=True, timeout=120,
        ).stdout
        return int(out.strip() or 0)
    except (subprocess.TimeoutExpired, OSError, ValueError):
        return 0


def wait_turn(gpu, task):
    """Called with start_gate held: wait until the previously started job no longer loads its scene (evaluators
    whose instances are all done exit at once without logging, so only the ones still alive have to be loaded)."""
    told = False
    while True:
        prev, why = last_start, None
        age = (time.time() - prev["t"]) / 60
        if os.path.exists(os.path.join(CTRL, "hold")):
            why = f"{CTRL}/hold exists"
        elif not volume_ok():
            why = "/workspace is not readable"
        elif age < START_GAP_MIN:
            why = f"previous job started {age:.0f} min ago"
        elif prev["proc"] is not None and prev["proc"].poll() is None and age < LOAD_MAX_MIN \
                and load_marks(prev["logs"]) - prev["marks"] < min(prev["n"], len(children(prev["proc"].pid, "eval_multi_ckpt.py")) or prev["n"]):
            why = "previous job is still loading its scene"
        if why is None:
            return
        if not told:
            log(f"GPU {gpu}: holding {task}: {why}")
            told = True
        time.sleep(30)


def run(gpu, key):
    offset, model, task = key
    cleanup(gpu)
    logs = log_dir(gpu, key)
    with start_gate:
        wait_turn(gpu, task)
        n = procs_for(key)
        marks = load_marks(logs)
        proc = launch(gpu, key, n, logs)
        last_start.update(t=time.time(), logs=logs, n=n, marks=marks, proc=proc)
    log(f"GPU {gpu}: START {model} {task} (indices {offset}-{offset + 9}) with {n} processes, pid {proc.pid}, logs {logs}")
    t0 = seen = time.time()
    while proc.poll() is None:
        time.sleep(30)
        newest = newest_mtime(logs)
        if newest is None:
            seen = time.time()  # cannot tell: do not call it a stall while the volume is failing
            continue
        seen = max(seen, newest)
        stale = (time.time() - seen) / 60
        if stale > STALL_MIN:
            log(f"GPU {gpu}: {task} logs silent for {stale:.0f} min, killing the job")
            break
    cleanup(gpu, proc.pid)
    proc.wait()
    return (time.time() - t0) / 60, t0


def launch(gpu, key, n, logs):
    offset, model, task = key
    while True:
        try:
            os.makedirs(logs, exist_ok=True)
            out = open(os.path.join(logs, "run_all.log"), "a")
            break
        except OSError as e:
            volume_bad["t"] = time.time()
            log(f"GPU {gpu}: WARNING cannot open logs in {logs}: {e}")
            wait_volume("opening logs")
    env = dict(
        os.environ, B1K_MODEL=model, B1K_GPU=str(gpu), B1K_PROCS=str(n), B1K_INSTANCE_OFFSET=str(offset),
        B1K_CKPT_ROOT="/dev/shm/ckpts",
    )
    return subprocess.Popen(
        [PY, "run_all.py", "--checkpoints", CKPT, "--tasks", task, "--no-finalize", "--retries", "0"],
        cwd="/root/b1k_eval", env=env, stdout=out, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )


def finalize(offset):
    """W&B summaries for every model of this phase with no open or running tasks."""
    todo = []
    with lock:
        for model in sorted({m for m, _ in JOBS.values()}):
            keys = [(offset, m, t) for m, tasks in JOBS.values() if m == model for t in tasks]
            settled = all(done(k) or failures.get(k, 0) >= MAX_ATTEMPTS for k in keys)
            if not any(done(k) for k in keys):
                continue  # nothing to summarize (e.g. phase 2 never ran)
            if settled and not any(k in claimed for k in keys) and (offset, model) not in finalized:
                finalized.add((offset, model))
                todo.append(model)
    for model in todo:
        env = dict(os.environ, B1K_MODEL=model, B1K_INSTANCE_OFFSET=str(offset), B1K_CKPT_ROOT="/dev/shm/ckpts")
        wait_volume("W&B finalize")
        try:
            rc = subprocess.call(
                [PY, "/root/b1k_eval/log_wandb.py", "--finalize"], cwd="/root/b1k_eval", env=env,
                stdout=open(f"/workspace/eval_logs/finalize_{model}{run_tag(offset)}.log", "a"), stderr=subprocess.STDOUT,
            )
        except OSError as e:
            rc = f"not run ({e})"
            with lock:
                finalized.discard((offset, model))
        log(f"W&B finalize {model}{run_tag(offset)}: exit code {rc}")


def worker(gpu, adopted):
    if adopted:
        pid, key = adopted
        adopt(gpu, pid, key)
        with lock:
            claimed.pop(key, None)
        finalize(key[0])
    while True:
        key = pick(gpu)
        if key is None:
            with lock:  # stay while a running task may still fail and come back, or a checkpoint is still being staged
                busy = bool(claimed) or any(open_keys(0, g, staged=False) for g in JOBS)
            if not busy:
                break
            time.sleep(60)
            continue
        minutes, t0 = run(gpu, key)
        wait_volume("checking results")
        ok = done(key)
        counted = not ok and volume_bad["t"] < t0  # a job that died with the volume is retried as it was
        with lock:
            if counted:
                failures[key] = failures.get(key, 0) + 1
            claimed.pop(key, None)
        log(f"GPU {gpu}: {'DONE' if ok else 'FAILED'} {key[1]} {key[2]} (indices {key[0]}-{key[0] + 9}) after {minutes:.0f} min"
            + ("" if ok else f" (failure {failures.get(key, 0)}/{MAX_ATTEMPTS})" if counted
               else " (/workspace failed during the job: not counted, will retry)"))
        finalize(key[0])
    log(f"GPU {gpu}: nothing left to run")


def main():
    os.makedirs(CTRL, exist_ok=True)
    adopted = {}
    out = subprocess.run(["pgrep", "-f", r"run_all\.py --checkpoints"], capture_output=True, text=True).stdout
    for pid in map(int, out.split()):
        env = environ(pid)
        if "B1K_GPU" not in env:
            continue
        gpu, model, offset = int(env["B1K_GPU"]), env["B1K_MODEL"], int(env.get("B1K_INSTANCE_OFFSET", "0"))
        task = None
        for child in children(pid, "eval_multi_ckpt.py"):
            argv = open(f"/proc/{child}/cmdline").read().split("\0")
            task = argv[argv.index("--task-name") + 1]
        key = (offset, model, task)
        claimed[key] = gpu
        adopted[gpu] = (pid, key)
    log(f"Scheduler started; adopted {adopted}")
    wait_volume("startup")
    try:  # e.g. `echo 1 > /dev/shm/b1k_sched/skip_gpus`: no new jobs on those GPUs (read at scheduler start)
        skip = {int(g) for g in open(os.path.join(CTRL, "skip_gpus")).read().split()}
    except (OSError, ValueError):
        skip = set()
    if skip:
        log(f"Not using GPUs {sorted(skip)} (skip_gpus); their tasks go to the other GPUs")
    threads = [threading.Thread(target=worker, args=(g, adopted.get(g))) for g in JOBS if g not in skip or g in adopted]
    for t in threads:
        t.start()
        time.sleep(1)
    for t in threads:
        t.join()
    for offset in OFFSETS:
        finalize(offset)
    log("Scheduler: all done")


if __name__ == "__main__":
    main()
