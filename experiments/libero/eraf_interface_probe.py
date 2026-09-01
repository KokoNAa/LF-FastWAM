"""Passive LIBERO recorder for the opt-in same-cache interface probe."""

import json
import hashlib
from pathlib import Path

import numpy as np

from fastwam.models.wan22.eraf_interface_probe import InterfaceProbe


def observe_simulator(env, metadata):
    """Record all named joints, including articulated fixtures and robot joints.

    Region names (e.g. bottom_region) need not be physical bodies. Predicate
    evaluation is the authority for completion; grasp/lift is not a door proxy.
    No privileged value produced here is fed into the policy.
    """
    inner = env.env
    joints = {}
    for name in inner.sim.model.joint_names:
        if name is not None:
            joints[str(name)] = np.asarray(inner.sim.data.get_joint_qpos(name)).reshape(-1).tolist()
    predicates = {}
    for goal in ("source_goal_state", "counterfactual_goal_state"):
        predicates[goal] = [
            {"predicate": list(p), "holds": bool(inner._eval_predicate(p))}
            for p in metadata[goal]
        ]
    return {"joints_qpos": joints, "predicates": predicates}


class LiberoInterfaceObserver:
    def __init__(self, model, cfg):
        options = cfg.EVALUATION.interface_probe
        self.probe = InterfaceProbe(
            model,
            options.warm_checkpoint,
            cfg.ckpt,
            options.driver,
            allow_hybrid_driver=bool(options.get("execute_hybrid_driver", False)),
        )
        self.trials = {int(x) for x in options.trials}
        self.stride = int(options.stride_replans)
        if self.stride < 1:
            raise ValueError("Interface probe stride must be positive.")
        self.root = Path(cfg.EVALUATION.output_dir) / "interface_probe"
        self.root.mkdir(parents=True, exist_ok=False)
        self.records = (self.root / "records.jsonl").open("x", encoding="utf-8")
        self.cfg = cfg
        self.pending = None
        self.write({"kind": "provenance", **self.probe.provenance,
                    "task_id": int(cfg.EVALUATION.task_id), "probe_trials": sorted(self.trials),
                    "stride_replans": self.stride, "privileged_observer_only": True,
                    "executed_driver": str(options.driver)})

    def write(self, record):
        self.records.write(json.dumps(record, allow_nan=False) + "\n")
        self.records.flush()

    def begin_episode(self, trial, instruction):
        self.trial, self.instruction = int(trial), instruction

    def observe_step(self, env, metadata, policy_step, *, terminal=False):
        self.write({"kind": "simulator", "trial": self.trial,
                    "policy_step": int(policy_step), "terminal": terminal,
                    **observe_simulator(env, metadata)})

    def before_replan(self, *, env, obs, metadata, policy_step, replan_index):
        if replan_index == 0:
            self.observe_step(env, metadata, policy_step)
        if replan_index % self.stride:
            return False
        state = env.env.sim.get_state()
        state = np.asarray(state.flatten() if hasattr(state, "flatten") else state).copy()
        arrays = {"simulator_state": state}
        arrays.update({f"obs/{key}": np.array(value, copy=True)
                       for key, value in obs.items() if isinstance(value, np.ndarray)})
        self.pending = ({"kind": "probe", "trial": self.trial,
                         "policy_step": int(policy_step), "replan_index": int(replan_index),
                         "policy_instruction": self.instruction,
                         "simulator_state_bytes_sha256": hashlib.sha256(state.tobytes()).hexdigest(),
                         "simulator_state_dtype": str(state.dtype),
                         **observe_simulator(env, metadata)}, arrays)
        return True

    def finish_replan(self, payload, *, action_env, input_image, proprio, prompt):
        if self.pending is None:
            raise RuntimeError("Interface probe has no pending same-state observation.")
        record, arrays = self.pending
        record.update(payload["record"])
        record["prompt"] = prompt
        arrays.update(payload["arrays"])
        arrays["driver_action_environment"] = np.array(action_env, copy=True)
        arrays["model_input_image"] = input_image.detach().float().cpu().numpy()
        if proprio is not None:
            arrays["model_input_proprio"] = proprio.detach().float().cpu().numpy()
        filename = f"trial{self.trial:03d}_replan{record['replan_index']:04d}.npz"
        # Exclusive creation: never silently replace a previous diagnostic.
        with (self.root / filename).open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        record["arrays"] = filename
        record["input_image_dtype"] = str(input_image.dtype)
        self.write(record)
        self.pending = None

    def close(self):
        self.records.close()
