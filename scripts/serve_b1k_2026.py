"""Policy server for the 2026 BEHAVIOR evaluator (python -m omnigibson.eval.eval).

Differences from serve_b1k.py (2025 evaluator):
- The 2026 evaluator always sends observations batched as (num_envs, ...), so each environment in the
  batch gets its own B1KPolicyWrapper (stage voting / inpainting / action-queue state) while sharing
  one loaded model.
- Each websocket connection gets its own set of wrappers, so several evaluator processes can share
  one server.
- No OmniGibson dependency: the websocket protocol (msgpack-numpy, /healthz, {"reset": True}) is
  reimplemented with openpi_client.
- Multi-checkpoint mode (--checkpoints): weights of all listed checkpoints are kept in host memory and
  GET /switch?ckpt=<name> swaps the served checkpoint (and resets all per-env state), so an evaluator can
  run every checkpoint on one loaded scene.
"""

import asyncio
import dataclasses
import http
import logging
import os
import pathlib
import shutil
import time
import traceback
import urllib.parse

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.35")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")

import jax
import jax.numpy as jnp
import numpy as np
import openpi.models.model as _model
import tyro
import websockets
import websockets.asyncio.server as _server
from openpi_client import msgpack_numpy

from b1k.policies import policy_config as _policy_config
from b1k.shared.eval_b1k_wrapper import B1KPolicyWrapper, B1KWrapperConfig
from b1k.training import config as _config

logger = logging.getLogger(__name__)

# Model calls are padded to this batch size (evaluator processes of the remaining large-scene tasks hold at most 3 envs) to avoid recompiling per size.
BATCH_PAD = 3


@dataclasses.dataclass
class Args:
    # Checkpoint step directory (contains params/ and assets/). In multi-checkpoint mode: the directory holding
    # the step directories named in --checkpoints.
    checkpoint_dir: str
    # Multi-checkpoint mode: step directory names under checkpoint_dir to preload (first one is served initially).
    checkpoints: list[str] | None = None
    # Params are staged here before restoring (reading large chunks straight from the network volume fails).
    staging_dir: str = "/dev/shm/ckpt"
    config: str = "pi_behavior_b1k_fast_2026"
    port: int = 8000

    # B1K wrapper execution parameters. Receding-horizon control: of each 30-action prediction only the first 13
    # are executed (in 10 env steps, the same 1.3x speedup as the 2025 defaults) before replanning; the 2025
    # defaults were 26 actions in 20 steps. The 4 actions after the executed ones still seed the next plan.
    actions_to_execute: int = 13
    actions_to_keep: int = 4
    execute_in_n_steps: int = 10
    history_len: int = 3
    votes_to_promote: int = 2
    time_threshold_inpaint: float = 0.3
    num_steps: int = 20
    apply_eval_tricks: bool = True
    # Force a gripper open for `gripper_recovery_hold_steps` env steps after it has been stuck (fully closed on
    # nothing, or commanded open but not moving) for this many consecutive env steps. 0 disables.
    gripper_recovery_steps: int = 90
    gripper_recovery_hold_steps: int = 15


class BatchedB1KPolicy:
    """Per-connection state: one B1KPolicyWrapper per environment index in the evaluator batch."""

    def __init__(self, policy, wrapper_config: B1KWrapperConfig):
        self._policy = policy
        self._wrapper_config = wrapper_config
        self._wrappers: list[B1KPolicyWrapper] = []

    def _ensure(self, batch_size: int) -> None:
        while len(self._wrappers) < batch_size:
            self._wrappers.append(B1KPolicyWrapper(self._policy, config=self._wrapper_config))

    def reset(self) -> None:
        for wrapper in self._wrappers:
            wrapper.reset()

    def set_policy(self, policy) -> None:
        """Serve a different model; per-env wrapper state is discarded."""
        self._policy = policy
        self._wrappers = []

    def act(self, obs: dict, chunk_size: int = 1) -> np.ndarray:
        """Returns (num_envs, chunk_size, action_dim) actions for the next `chunk_size` steps.

        B1KPolicyWrapper already executes `execute_in_n_steps` actions open-loop between replans, so when
        chunk_size == execute_in_n_steps (requests land exactly on replan boundaries) replaying the chunk
        client-side is identical to querying every step. Environments that need a new prediction at the same
        step share one batched model call.
        """
        batch_size = int(np.asarray(obs["task_id"]).shape[0])
        self._ensure(batch_size)
        env_obs = [
            {key: np.asarray(value)[i] for key, value in obs.items() if not key.startswith("__")}
            for i in range(batch_size)
        ]
        states = [self._wrappers[i].begin_step(env_obs[i]) for i in range(batch_size)]
        chunks = [[] for _ in range(batch_size)]
        for _ in range(chunk_size):
            needing = [i for i in range(batch_size) if self._wrappers[i].needs_prediction()]
            if needing:
                self._predict(needing, env_obs, states)
            for i in range(batch_size):
                chunks[i].append(self._wrappers[i].next_action().numpy())
        return np.asarray(chunks, dtype=np.float32)

    def _predict(self, env_indices, env_obs, states) -> None:
        model_inputs = {i: self._wrappers[i].build_model_input(env_obs[i]) for i in env_indices}
        # Inpainting and non-inpainting samples are different jitted programs, so batch them separately.
        with_inpaint = [i for i in env_indices if model_inputs[i].get("initial_actions") is not None]
        without_inpaint = [i for i in env_indices if model_inputs[i].get("initial_actions") is None]
        for group, inpaint in ((with_inpaint, True), (without_inpaint, False)):
            if not group:
                continue
            try:
                outputs = self._policy.infer_batch(
                    [model_inputs[i] for i in group],
                    [model_inputs[i]["initial_actions"] for i in group] if inpaint else None,
                    pad_to=BATCH_PAD,
                )
            except Exception as e:  # GPU is shared with the simulators; fall back to lower-peak single-env calls
                if "RESOURCE_EXHAUSTED" not in str(e):
                    raise
                logger.warning("Batched inference ran out of GPU memory; retrying one environment at a time")
                outputs = [
                    self._policy.infer_batch(
                        [model_inputs[i]], [model_inputs[i]["initial_actions"]] if inpaint else None, pad_to=1
                    )[0]
                    for i in group
                ]
            for i, output in zip(group, outputs):
                self._wrappers[i].apply_prediction(output, states[i])


class PolicyBank:
    """Holds the served policy; in multi-checkpoint mode also the host-memory params of every checkpoint."""

    def __init__(self, args: Args, train_config):
        self._args = args
        self._train_config = train_config
        self._sample_kwargs = {"num_steps": args.num_steps}
        self._host_params = {}
        self.connections: set[BatchedB1KPolicy] = set()
        if args.checkpoints:
            for name in args.checkpoints:
                self._host_params[name] = self._restore_to_host(name)
            self.current = args.checkpoints[0]
            self.policy = self._build(self.current)
        else:
            self.current = pathlib.Path(args.checkpoint_dir).name
            self.policy = _policy_config.create_trained_policy(
                train_config, args.checkpoint_dir, sample_kwargs=self._sample_kwargs
            )

    def _restore_to_host(self, name):
        staged = pathlib.Path(self._args.staging_dir) / name
        shutil.rmtree(staged, ignore_errors=True)
        t0 = time.time()
        shutil.copytree(pathlib.Path(self._args.checkpoint_dir) / name / "params", staged / "params")
        params = _model.restore_params(staged / "params", restore_type=np.ndarray, dtype=jnp.bfloat16)
        shutil.rmtree(staged)
        logger.info("Restored %s to host memory in %.0fs", name, time.time() - t0)
        return params

    def _build(self, name):
        # Put the served checkpoint's weights on the GPU once. Handing host arrays to the model instead would re-upload
        # all ~6.6 GB of weights on every inference call (slow, and a large transient GPU-memory spike).
        device_params = jax.device_put(self._host_params[name])
        return _policy_config.create_trained_policy(
            self._train_config,
            pathlib.Path(self._args.checkpoint_dir) / name,  # norm stats are read from <ckpt>/assets
            sample_kwargs=self._sample_kwargs,
            params=device_params,
        )

    def switch(self, name) -> str:
        if name not in self._host_params:
            raise KeyError(f"unknown checkpoint {name!r}; have {sorted(self._host_params)}")
        if name != self.current:
            t0 = time.time()
            # Drop every reference to the previous model so its device buffers are freed before building the next.
            for connection in self.connections:
                connection.set_policy(None)
            self.policy = None
            self.policy = self._build(name)
            self.current = name
            logger.info("Switched to %s in %.1fs", name, time.time() - t0)
        for connection in self.connections:
            connection.set_policy(self.policy)
        return f"serving {name}\n"


async def _serve(args: Args, bank: PolicyBank) -> None:
    wrapper_config = B1KWrapperConfig(
        actions_to_execute=args.actions_to_execute,
        actions_to_keep=args.actions_to_keep,
        execute_in_n_steps=args.execute_in_n_steps,
        history_len=args.history_len,
        votes_to_promote=args.votes_to_promote,
        time_threshold_inpaint=args.time_threshold_inpaint,
        num_steps=args.num_steps,
        apply_eval_tricks=args.apply_eval_tricks,
        gripper_recovery_steps=args.gripper_recovery_steps,
        gripper_recovery_hold_steps=args.gripper_recovery_hold_steps,
    )
    logger.info("Wrapper config: %s", wrapper_config)

    async def handler(websocket):
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()
        batched_policy = BatchedB1KPolicy(bank.policy, wrapper_config)
        bank.connections.add(batched_policy)
        await websocket.send(packer.pack(bank.policy.metadata))
        while True:
            try:
                request = msgpack_numpy.unpackb(await websocket.recv())
                if "reset" in request:
                    batched_policy.reset()
                    continue
                start = time.monotonic()
                # Optional action-chunk protocol (evaluator --replay-action-chunk-size K): only honored when K
                # matches the wrapper's open-loop execution length, otherwise fall back to one action per request.
                chunk_size = int(request.get("__action_chunk_size__", 0))
                use_chunk = chunk_size > 1 and chunk_size == wrapper_config.execute_in_n_steps
                chunk = batched_policy.act(request, chunk_size if use_chunk else 1)
                response = {"action": chunk[:, 0], "server_timing": {"infer_ms": 1000 * (time.monotonic() - start)}}
                if use_chunk:
                    response["action_chunk"] = chunk
                await websocket.send(packer.pack(response))
            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                bank.connections.discard(batched_policy)
                break
            except Exception:
                logger.error("Error serving %s:\n%s", websocket.remote_address, traceback.format_exc())
                await websocket.send(traceback.format_exc())
                await websocket.close(code=1011, reason="Internal server error")
                raise

    def process_request(connection, request):
        url = urllib.parse.urlparse(request.path)
        if url.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if url.path == "/switch":
            # Blocking on purpose: evaluators only call this between passes, when no inference is in flight.
            try:
                return connection.respond(http.HTTPStatus.OK, bank.switch(urllib.parse.parse_qs(url.query)["ckpt"][0]))
            except Exception:
                logger.error("Checkpoint switch failed:\n%s", traceback.format_exc())
                return connection.respond(http.HTTPStatus.INTERNAL_SERVER_ERROR, traceback.format_exc())
        return None

    async with _server.serve(
        handler, "0.0.0.0", args.port, compression=None, max_size=None, process_request=process_request
    ) as server:
        logger.info("Serving on port %d", args.port)
        await server.serve_forever()


def main(args: Args) -> None:
    bank = PolicyBank(args, _config.get_config(args.config))
    logger.info("Loaded %s; serving %s", args.config, bank.current)
    asyncio.run(_serve(args, bank))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
