"""Evaluate checkpoints of one amyjia1212/b1k-2026-t* model (env B1K_MODEL) on the BEHAVIOR 2026 tasks it was trained on.

Inference-time settings (see INFERENCE below) are passed to the policy server and tagged onto the results dir and
W&B run names: receding-horizon control (replan every RHC_EXECUTE_STEPS env steps) and gripper recovery (force a
gripper open after GRIPPER_RECOVERY_STEPS consecutive stuck steps). run_chunks.sh runs all five chunk models.

One policy server holds all checkpoints (host memory) and switches on request. For each task, a few evaluator
processes (each pinned to its own cores, owning a slice of public-test instances 0-9) load their scenes once and
then run every checkpoint in lockstep (eval_multi_ckpt.py): switch server -> go_<ckpt> -> wait for all
done_<ckpt>_<rank>. After each task, its results are logged to W&B.

Multi-GPU (5x L40S pod): one run_all.py process per chunk model, each bound to one GPU (env B1K_GPU, see GPU_LAYOUT):
its own policy-server port, CPU sets, log dir and results dir, so chunks are independent (run_chunks_multigpu.sh).

Throughput notes (AMD EPYC 7543, RTX 5090): unpinned OmniGibson with default torch threads ran ~2 env-steps/s;
pinned to 16 cores with OMP_NUM_THREADS=8 it runs ~10 env-steps/s per process. GPU memory decides the number of processes
per task (see WAVES); the policy server needs ~8 GB. Resumable: finished
(checkpoint, task) pairs are skipped, and tasks already logged to W&B are not logged again.

Run with the b1k-solution venv python (has wandb).
"""

import argparse
import glob
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

MODELS = {  # HF repo amyjia1212/<model> -> {2026 challenge task index: task name}
    "b1k-2026-t90-91-93-96-97": {
        90: "composting_waste", 91: "store_produce", 93: "clean_a_keyboard", 96: "cook_broccolini",
        97: "setup_a_bar_for_a_cocktail_party",
    },
    "b1k-2026-t50-70-72-85": {  # chunk 1
        50: "freeze_fruit", 70: "thawing_frozen_food", 72: "cook_a_frozen_pie", 85: "putting_dirty_dishes_in_sink",
    },
    "b1k-2026-t52-53-73-66-86": {  # chunk 2
        52: "sorting_bottles_cans_and_paper", 53: "tidying_living_room", 73: "organizing_school_stuff",
        66: "turning_out_all_lights_before_sleep", 86: "make_gift_bags_for_baby_showers",
    },
    "b1k-2026-t65-71-74-75": {  # chunk 3
        65: "unloading_the_car", 71: "clean_your_rusty_garden_tools", 74: "carrying_out_garden_furniture",
        75: "put_together_a_basic_pruning_kit",
    },
    "b1k-2026-t79-81-84-88": {  # chunk 4
        79: "polishing_shoes", 81: "packing_meal_for_delivery", 84: "tidying_bathroom", 88: "rearrange_your_room",
    },
    "b1k-2026-t94-95-98": {  # chunk 5
        94: "dispose_of_batteries", 95: "cook_brussels_sprouts", 98: "laying_tile_floors",
    },
    # single-task models (step_002000 on indices 0-9; step_004000/6000/8000 on indices 0-4, B1K_NUM_INSTANCES=5)
    "b1k-2026-t50": {50: "freeze_fruit"},
    "b1k-2026-t52": {52: "sorting_bottles_cans_and_paper"},
    "b1k-2026-t72": {72: "cook_a_frozen_pie"},
    "b1k-2026-t53": {53: "tidying_living_room"},  # step_005999 (6k) on indices 0-9
    "b1k-2026-t66": {66: "turning_out_all_lights_before_sleep"},  # step_005999 (6k) on indices 0-9
    "b1k-2026-t71": {71: "clean_your_rusty_garden_tools"},  # step_005999 (6k) on indices 0-9
    "b1k-2026-t65": {65: "unloading_the_car"},  # step_005999 (6k) on indices 0-9
}
MODEL = os.environ.get("B1K_MODEL", "b1k-2026-t90-91-93-96-97")
# B1K_CKPT_ROOT: where the checkpoints are read from. This pod's network (and with it the /workspace volume) is capped
# at ~12 MB/s, so run_chunks_multigpu.sh points it at a copy re-downloaded from the same HF repos into /dev/shm.
CKPT_ROOT = os.path.join(os.environ.get("B1K_CKPT_ROOT", "/workspace/ckpts"), MODEL)

# Inference-time settings. Receding-horizon control: execute only the first RHC_ACTIONS of each 30-action prediction
# in RHC_EXECUTE_STEPS env steps (same 1.3x speedup as the 2025 defaults of 26 actions in 20 steps), then replan.
# Gripper recovery: force a gripper open for GRIPPER_RECOVERY_HOLD steps once it has been fully closed on nothing
# (or commanded open but not moving) for GRIPPER_RECOVERY_STEPS consecutive env steps; 0 disables.
RHC_EXECUTE_STEPS = int(os.environ.get("B1K_RHC_EXECUTE_STEPS", "10"))
RHC_ACTIONS = round(1.3 * RHC_EXECUTE_STEPS)
GRIPPER_RECOVERY_STEPS = int(os.environ.get("B1K_GRIPPER_RECOVERY_STEPS", "90"))
GRIPPER_RECOVERY_HOLD = 15
BASELINE_INFERENCE = RHC_EXECUTE_STEPS == 20 and GRIPPER_RECOVERY_STEPS == 0  # the 2025 defaults used before 2026-10-11
INFERENCE_TAG = "" if BASELINE_INFERENCE else f"-rhc{RHC_EXECUTE_STEPS}-gr{GRIPPER_RECOVERY_STEPS}"
INFERENCE = (
    f"execute {RHC_ACTIONS}/keep 4 of 30 in {RHC_EXECUTE_STEPS} env steps (replan every {RHC_EXECUTE_STEPS} steps), "
    + (f"gripper recovery after {GRIPPER_RECOVERY_STEPS} stuck steps (hold open {GRIPPER_RECOVERY_HOLD})"
       if GRIPPER_RECOVERY_STEPS else "no gripper recovery")
)
# Public-test instance indices evaluated are INSTANCE_OFFSET .. INSTANCE_OFFSET+9 (the split has 20: ids 301-320).
# Offset 0 (first ten) keeps the original result/W&B names; any other offset gets its own results dir and W&B runs.
INSTANCE_OFFSET = int(os.environ.get("B1K_INSTANCE_OFFSET", "0"))
INSTANCE_TAG = "" if INSTANCE_OFFSET == 0 else f"-inst{INSTANCE_OFFSET}-{INSTANCE_OFFSET + 9}"
# "-stagefix": runs from 2026-10-14 on use the training team's per-task stage counts (pi_behavior_config.py); earlier
# results (same names without the suffix) used reconstructed counts that indexed the wrong stage embeddings.
STAGE_TAG = "-stagefix"
RUN_TAG = INFERENCE_TAG + INSTANCE_TAG + STAGE_TAG
RESULTS_ROOT = f"/workspace/eval_results/{MODEL}" + RUN_TAG
GPU = int(os.environ.get("B1K_GPU", "0"))  # nvidia-smi index of the GPU this process owns
LOG_ROOT = os.environ.get("B1K_LOG_ROOT", f"/workspace/eval_logs/{MODEL}{RUN_TAG}-gpu{GPU}")  # one log dir per job
SOLUTION_DIR = "/root/behavior-1k-solution"
B1K_DIR = "/root/BEHAVIOR-1K"
BEHAVIOR_PY = "/root/miniconda3/envs/behavior/bin/python"

# Every checkpoint downloaded into CKPT_ROOT (download_chunk_ckpts.py / auto_eval_new_ckpts.py) is evaluated.
CHECKPOINTS = sorted(d for d in os.listdir(CKPT_ROOT) if d.startswith("step_")) if os.path.isdir(CKPT_ROOT) else []
TASKS = MODELS[MODEL]  # 2026 challenge task index -> task name
# B1K_NUM_INSTANCES=5 evaluates only the first five (ids 301-305 at offset 0); results dir and W&B names are unchanged.
NUM_INSTANCES = int(os.environ.get("B1K_NUM_INSTANCES", "10"))
# One chunk per GPU. This pod: 2x EPYC 9374F (node 0 = cores 0-31, node 1 = cores 32-63; 64-127 are their SMT
# siblings and stay unused), GPUs 0-1 on node 0 and 2-4 on node 1, and a cgroup quota of 68 CPUs for the whole pod.
# So each chunk gets one simulator process on 10-12 physical cores of its GPU's node, and the five policy servers
# share the 8 remaining cores of node 0.
GPU_LAYOUT = {  # nvidia-smi index -> simulator CPU sets (one per evaluator process)
    0: ["0-11"], 1: ["12-23"], 2: ["32-42"], 3: ["43-53"], 4: ["54-63"],
}
PORT = 8100 + GPU  # not 8000+GPU: nginx listens on 8001 on RunPod pods and answers /healthz
# Evaluation waves per task; each wave is a list of instance-index slices, one evaluator process per slice, and runs
# all checkpoints in lockstep. The small Rs_int scene fits 3 processes (10 envs) next to the policy server on the
# 32 GB GPU. Office/restaurant scenes need ~2.2-2.5 GB per env, so they run as two waves of 2 processes (3+3, 2+2).
# (Those were the 32 GB RTX 5090 numbers. On a 48 GB L40S all 10 instances run as one wave.)
# B1K_PROCS evaluator processes share that wave: one 10-env process is bound by its main thread (~2 steps/s on
# ~2.5 cores), so the instances and the GPU's cores are split evenly over B1K_PROCS processes (default 1).
PROCS = int(os.environ.get("B1K_PROCS", "1"))


def split(seq, n):
    """Split seq into n contiguous parts whose sizes differ by at most one (larger parts first)."""
    k, r = divmod(len(seq), n)
    bounds = [i * k + min(i, r) for i in range(n + 1)]
    return [seq[a:b] for a, b in zip(bounds, bounds[1:])]


def gpu_cores(spec):
    lo, hi = map(int, spec.split("-"))
    return list(range(lo, hi + 1))


WAVES = [split(list(range(NUM_INSTANCES)), PROCS)]
CPU_SETS = [f"{c[0]}-{c[-1]}" for c in split(gpu_cores(GPU_LAYOUT[GPU][0]), PROCS)]
SERVER_CPUS = "24-31"
MAX_STEPS = None  # None = evaluator default (1.5x mean human demo length)


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def step_of(ckpt):
    return int(ckpt.split("_")[1])


def result_jsons(ckpt, task):
    return sorted(glob.glob(os.path.join(RESULTS_ROOT, ckpt, task, "json", "*.json")))


def task_done(ckpt, task):
    return len(result_jsons(ckpt, task)) >= NUM_INSTANCES


def gpu_uuid():
    """UUID of nvidia-smi GPU index GPU (CUDA_VISIBLE_DEVICES by index depends on CUDA_DEVICE_ORDER; a UUID does not)."""
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader", "-i", str(GPU)])
    return out.decode().strip()


def server_healthy():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/healthz", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def start_server(checkpoints):
    log_path = os.path.join(LOG_ROOT, "server.log")
    env = dict(
        os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION="0.35", XLA_PYTHON_CLIENT_ALLOCATOR="platform",
        CUDA_VISIBLE_DEVICES=gpu_uuid(),
    )
    proc = subprocess.Popen(
        [
            "taskset", "-c", SERVER_CPUS,
            f"{SOLUTION_DIR}/.venv/bin/python", "scripts/serve_b1k_2026.py",
            "--checkpoint-dir", CKPT_ROOT, "--checkpoints", *checkpoints, "--port", str(PORT),
            "--staging-dir", f"/dev/shm/ckpt_gpu{GPU}",  # servers of different chunks stage the same step name
            "--actions-to-execute", str(RHC_ACTIONS), "--execute-in-n-steps", str(RHC_EXECUTE_STEPS),
            "--gripper-recovery-steps", str(GRIPPER_RECOVERY_STEPS),
            "--gripper-recovery-hold-steps", str(GRIPPER_RECOVERY_HOLD),
        ],
        cwd=SOLUTION_DIR,
        env=env,
        stdout=open(log_path, "a"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    log(f"Started policy server pid={proc.pid} on GPU {GPU} port {PORT} (log: {log_path}); preloading {len(checkpoints)} checkpoints")
    start = time.time()
    while True:
        if proc.poll() is not None:  # checked first: something else may be answering on the port
            raise RuntimeError(f"policy server exited with code {proc.returncode}")
        if server_healthy():
            break
        if time.time() - start > 3600:
            raise TimeoutError("policy server did not become healthy")
        time.sleep(10)
    log(f"Policy server healthy after {(time.time() - start) / 60:.1f} min")
    return proc


def stop(proc):
    if proc is not None and proc.poll() is None:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)


def switch_server(ckpt):
    t0 = time.time()
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/switch?ckpt={ckpt}", timeout=3600) as r:
        if r.status != 200:
            raise RuntimeError(f"switch to {ckpt} failed: {r.read().decode()}")
    log(f"  server switched to {ckpt} in {time.time() - t0:.0f}s")


def run_task(task, checkpoints):
    """Run all checkpoints on one task, wave by wave. Returns True if every evaluator process exited 0."""
    ok = True
    for wave, slices in enumerate([[[i + INSTANCE_OFFSET for i in s] for s in w] for w in WAVES]):
        ok &= run_wave(task, checkpoints, slices, wave)
    return ok


def run_wave(task, checkpoints, slices, wave):
    """Run all checkpoints with one lockstep evaluator process per slice. Returns True if all exited 0."""
    phase_dir = os.path.join("/dev/shm/b1k_phase", f"gpu{GPU}_{task}_w{wave}")
    shutil.rmtree(phase_dir, ignore_errors=True)
    os.makedirs(phase_dir)
    procs = []
    for rank, (instances, cpus) in enumerate(zip(slices, CPU_SETS)):
        tmp = f"/dev/shm/og_tmp/gpu{GPU}_{task}{RUN_TAG}_w{wave}_{rank}"
        os.makedirs(tmp, exist_ok=True)
        env = dict(
            os.environ,
            OMNIGIBSON_DATA_PATH="/workspace/og_data",
            OMNIGIBSON_HEADLESS="1",
            PYTHONPATH="/root/b1k_eval",
            TMPDIR=tmp,  # USD decrypts every object to a temp file; the network volume is ~10x slower and / is only 50 GB
            # /dev/shm is noexec, so torch.compile's generated .so files must live on the local disk (they are tiny)
            TORCHINDUCTOR_CACHE_DIR=f"/root/.cache/torchinductor/{task}_w{wave}_{rank}",
            TRITON_CACHE_DIR=f"/root/.cache/triton/{task}_w{wave}_{rank}",
            OMP_NUM_THREADS=str(min(8, len(gpu_cores(cpus)))),
            MKL_NUM_THREADS=str(min(8, len(gpu_cores(cpus)))),
            # torch/warp/PhysX only see this chunk's GPU (as cuda:0); Vulkan still lists every GPU, so the renderer
            # is pointed at it by index (eval_multi_ckpt.py).
            CUDA_VISIBLE_DEVICES=gpu_uuid(),
            B1K_RENDER_GPU=str(GPU),
        )
        cmd = [
            "taskset", "-c", cpus, BEHAVIOR_PY, "/root/b1k_eval/eval_multi_ckpt.py",
            "--task-name", task,
            "--checkpoints", *checkpoints,
            "--results-root", RESULTS_ROOT,
            "--instance-indices", *map(str, instances),
            "--port", str(PORT),
            "--phase-dir", phase_dir,
            "--rank", str(rank),
            "--action-chunk-size", str(RHC_EXECUTE_STEPS),  # must equal the server's open-loop length
            *(["--max-steps", str(MAX_STEPS)] if MAX_STEPS else []),
        ]
        log_path = os.path.join(LOG_ROOT, f"eval_{task}{RUN_TAG}_w{wave}_r{rank}.log")
        procs.append(subprocess.Popen(cmd, cwd=B1K_DIR, env=env, stdout=open(log_path, "a"), stderr=subprocess.STDOUT))
    log(f"Evaluating {task} wave {wave}: {len(procs)} processes, instance slices {slices}, checkpoints {checkpoints}")

    t_task = time.time()
    for ckpt in checkpoints:
        switch_server(ckpt)
        open(os.path.join(phase_dir, f"go_{ckpt}"), "w").close()
        t0 = time.time()
        while True:
            waiting = [
                rank for rank, p in enumerate(procs)
                if p.poll() is None and not os.path.exists(os.path.join(phase_dir, f"done_{ckpt}_{rank}"))
            ]
            if not waiting:
                break
            time.sleep(10)
        n = len(result_jsons(ckpt, task))
        log(f"  {task} / {ckpt} (wave {wave}): {n}/{NUM_INSTANCES} results after {(time.time() - t0) / 60:.1f} min")
    rcs = [p.wait() for p in procs]
    log(f"  {task} wave {wave}: exit codes {rcs} after {(time.time() - t_task) / 60:.1f} min")
    return all(rc == 0 for rc in rcs)


def main():
    global MAX_STEPS, RESULTS_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", nargs="+", default=CHECKPOINTS)
    parser.add_argument("--tasks", nargs="+", default=list(TASKS.values()))
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--no-finalize", action="store_true",
                        help="Skip the W&B summary (a model split over two GPUs is finalized once, by the launcher).")
    parser.add_argument("--max-steps", type=int, default=None, help="Testing only: override the episode timeout.")
    parser.add_argument("--results-root", default=RESULTS_ROOT, help="Testing only.")
    args = parser.parse_args()
    MAX_STEPS, RESULTS_ROOT = args.max_steps, args.results_root
    os.makedirs(LOG_ROOT, exist_ok=True)
    log(f"GPU {GPU} (port {PORT}, simulator CPUs {CPU_SETS}, server CPUs {SERVER_CPUS}), logs {LOG_ROOT}")
    log(f"Model {MODEL}: checkpoints {args.checkpoints}, tasks {args.tasks}, results {RESULTS_ROOT}; inference: {INFERENCE}")

    server = None
    try:
        for task in args.tasks:
            for attempt in range(args.retries + 1):
                pending = [c for c in args.checkpoints if not task_done(c, task)]
                if not pending:
                    break
                if server is None or server.poll() is not None:
                    server = start_server(args.checkpoints)
                if not run_task(task, pending):
                    log(f"  attempt {attempt + 1} for {task} had failures")
            missing = [c for c in args.checkpoints if not task_done(c, task)]
            if missing:
                log(f"WARNING: {task} incomplete for {missing} after retries")
            if not args.no_wandb:
                subprocess.call([sys.executable, "/root/b1k_eval/log_wandb.py", "--task", task])
    finally:
        stop(server)
    if not args.no_wandb and not args.no_finalize:
        subprocess.call([sys.executable, "/root/b1k_eval/log_wandb.py", "--finalize"])
    log("All done")


if __name__ == "__main__":
    main()
