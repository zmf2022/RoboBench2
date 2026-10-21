# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jev two-stage client: measured state -> INTENT -> motor labels -> rel-IK.

The decision layer is RoboJEV's, unchanged in structure (see
:mod:`policies.jev.prompts`, :mod:`policies.jev.state`). This module is the
part that cannot be ported verbatim, because RoboJEV executes a decision as an
ABSOLUTE Cartesian target with per-tick feedback until it converges, while
``DroidRelIKActionCfg`` takes open-loop relative increments. Three consequences,
each of which silently breaks the port if skipped:

**The TCP is not ``ee_pos``.** ``ee_pos`` reports ``base_link``, the Robotiq
mount flange (robolab/robots/droid.py). RoboJEV's TCP is the point between the
jaws. Aiming the flange where the grasp centre belongs puts the fingers ~13 cm
low -- the arm descends forever and the object is never between the jaws. The
flange->grasp-centre vector is measured in
``policies.vlm_agent.token_controller.FLANGE_TO_FINGERTIP``.

**One decision is one normalised step, spread over control steps.** Relative IK
achieves a roughly constant ~28% of whatever is commanded, so the commanded
total is ``step_m / REL_IK_ACHIEVED_FRACTION``, and that total is divided by
``control_steps`` -- issuing the whole delta on every control step multiplies
the travel by ``control_steps``.

**Zeros in the rotation slots do not hold the orientation.** In relative mode a
zero rotation delta means "target = whatever you are now", so IK-induced tilt
becomes the next setpoint and is never corrected. RoboJEV keeps the gripper
pointing down by construction; here it has to be driven back every control step.
"""

from __future__ import annotations

import logging

import numpy as np

from policies.jev.execution import ActionGuard
from policies.jev.geometry import tolerance_for_step
from policies.jev.state import STALL_PROGRESS_M, build_state
from policies.jev.types import EefDecision, PolicyError
from policies.vlm_agent.token_controller import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    REL_IK_ACHIEVED_FRACTION,
    fingertip_position,
    gripper_width_m,
    hold_orientation_rotvec,
)
from policies.vlm_pinpoint.pointing_controller import yaw_aligned_quat
from robolab.core.observations.observation_utils import unpack_proprio_obs
from robolab.core.task.instruction import instruction_text
from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)

# Achieved translation per decision, in metres -- RoboJEV's ``step_m``. Its own
# config caps that at 0.02, and this deliberately exceeds it: RoboJEV's cap
# bounds a CLOSED-LOOP step that is guaranteed to land, where overshoot is the
# only risk. Here the risk is the opposite one -- a commanded step big enough to
# saturate the DLS solve, which buys distance with orientation error. What
# governs saturation is the PER-CONTROL-STEP command, not the per-decision
# distance, so this rises together with CONTROL_STEPS below and leaves
# ``per_step_cmd`` exactly where it was at 0.02/8. Raising this one alone would
# not.
STEP_M = 0.03
# Splitting the same total over more control steps shrinks the per-step command
# and with it the orientation transient, at no cost in distance -- but it also
# divides the DECISION budget. BananaInBowlTask runs episode_length_s=50 at
# dt=1/120 with decimation=8, i.e. 750 control steps, so 12 steps per decision
# buys 62 decisions -- against the ~40 a pick-and-place needs at a 3 cm step,
# plus the handful of stationary decisions the aiming gate spends before the
# first approach.
CONTROL_STEPS = 12
# Per-control-step safety cap on the IK target step: a guard so a mis-set config
# cannot ask the solver for a lunge it will diverge on.
MAX_DELTA_M = 0.05
# Control steps to let the scene settle before the first decision. Object motion
# inside robolab.constants.SETTLE_WARMUP_S (1.0 s, i.e. 15 control steps) is the
# scene settling, and the object's resting height is latched from the first
# decision -- latching it mid-settle biases every lift and placement after it.
SETTLE_STEPS = 15
# Consecutive invalid/failed decisions tolerated before giving up on the episode.
MAX_CONSECUTIVE_FAILURES = 10
# ``finish`` is an intention, never an authoritative success label -- the harness
# decides success. This only stops us paying for identical calls once the model
# has said "done" this many times in a row; the arm keeps holding either way.
FINISH_HOLD_STREAK = 3
# How many past actions the model sees, with the motion each actually achieved.
HISTORY_LENGTH = 8

TASKS = {
    # Object/container names must match the scene's contact_object_list.
    "BananaInBowlTask": {
        "object": "banana",
        "container": "bowl",
        # Grasp centre relative to the object's origin. RoboJEV uses +8 mm on a
        # 4 cm cube (flat faces, jaws grip the upper half); a banana is round and
        # this rig measures that closing at 65% of its height grasps air
        # (policies/vlm_pinpoint/pointing_controller.py), so aim at the equator.
        # This is the one number here that wants a probe run to confirm.
        "grasp_hover_m": 0.0,
        "release_hover_m": 0.10,
        # A RELEASE GATE, not the success test: the harness decides success with
        # a convex-hull containment check in the bowl's own frame
        # (core/task/conditionals.py::object_in_container), which this policy
        # deliberately does not try to reproduce. This is just "close enough over
        # the bowl to let go", sized at the bowl's opening radius.
        "container_radius_m": 0.05,
        # Which local axis runs along the object's length. Used only as a
        # fallback: when the ground-truth exporter reports per-object
        # ``dimensions`` this is derived instead. UNVERIFIED -- the scene asset
        # is not in this checkout, so confirm it before trusting a rollout.
        "object_long_axis_local": (1.0, 0.0, 0.0),
        # Lift above the object's resting height before transporting.
        "transport_clearance_m": 0.08,
    },
}


def _task_config(task) -> tuple[str, dict]:
    if isinstance(task, (list, tuple)):
        if len(task) != 1:
            raise ValueError("JevClient requires exactly one task per run")
        task = task[0]
    if task in TASKS:
        return task, TASKS[task]
    key = str(task).lower().replace("_", "").replace("-", "")
    for name, cfg in TASKS.items():
        stem = name.lower().replace("task", "")
        if stem and stem in key:
            return name, cfg
    raise ValueError(f"unsupported task {task!r}; known: {sorted(TASKS)}")


def _find_object(mapping: dict, name: str):
    lowered = {str(k).lower(): k for k in mapping}
    if name in lowered:
        return mapping[lowered[name]]
    for key in mapping:
        if name in str(key).lower():
            return mapping[key]
    return None


def _long_axis_local(obj: dict, fallback):
    """The object's local long axis, measured if the exporter reports extents.

    ``gt_state`` carries pose but not size today, so this falls back to a
    per-task constant. If ``robolab/eval/gt_state.py`` is extended to export
    ``world.get_dimensions(name)`` this becomes fully derived and the constant
    stops mattering -- which is the version worth having.
    """
    dims = obj.get("dimensions")
    if dims is None:
        return fallback
    dims = np.asarray(dims, dtype=float).ravel()[:3]
    if dims.size != 3 or not np.isfinite(dims).all() or dims.max() <= 0:
        return fallback
    axis = np.zeros(3)
    axis[int(np.argmax(dims))] = 1.0
    return axis


class _EnvState:
    """Everything latched per env, cleared at each episode boundary."""

    def __init__(self) -> None:
        self.generation = 0       # distinguishes observations across per-env resets
        self.quat_home = None      # the downward reference latched at reset
        self.quat_ref = None       # ...spun about +Z to aim the jaws
        self.gripper_target = "open"      # RoboJEV semantics: 'hold' keeps this
        self.per_step_cmd = np.zeros(3)
        self.steps_left = 0
        self.settle_left = SETTLE_STEPS
        self.rest_z = None
        self.prev_tip = None
        self.history: list[dict] = []
        self.step_id = 0
        self.failures = 0
        self.finish_streak = 0
        self.stalled = False
        self.holding_pattern = False
        self.pending: dict | None = None


class JevClient(InferenceClient):
    """Drives rel-IK from RoboJEV-style intent + motor decisions."""

    def __init__(self, backend, task: str, step_m: float = STEP_M,
                 control_steps: int = CONTROL_STEPS, ik_scale: float = 0.5,
                 episode_id: str = "episode") -> None:
        super().__init__()
        self.task_id, self.task_cfg = _task_config(task)
        self.backend = backend
        self.step_m = float(step_m)
        self.control_steps = max(1, int(control_steps))
        self.ik_scale = float(ik_scale)
        self.episode_id = episode_id
        self.guard = ActionGuard()
        self._envs: dict[int, _EnvState] = {}
        self._next_env_generation = 0

    # -- lifecycle ------------------------------------------------------

    def begin_episode(self, episode_idx: int) -> None:
        super().begin_episode(episode_idx)
        self.episode_id = f"episode_{episode_idx:04d}"
        self._envs.clear()
        self.guard.reset()

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._envs.clear()
            self.guard.reset()
        else:
            self._envs.pop(env_id, None)
        super().reset(env_id=env_id)

    def close(self) -> None:
        closer = getattr(self.backend, "close", None)
        if callable(closer):
            closer()
        super().close()

    def _env_state(self, env_id: int) -> _EnvState:
        if env_id not in self._envs:
            self._envs[env_id] = _EnvState()
            self._envs[env_id].generation = self._next_env_generation
            self._next_env_generation += 1
        return self._envs[env_id]

    @property
    def commanded_per_step_m(self) -> float:
        """Metres commanded per control step so one decision achieves ``step_m``."""
        return self.step_m / REL_IK_ACHIEVED_FRACTION / self.control_steps

    # -- observation ----------------------------------------------------

    def _pose(self, obs, env_id: int) -> dict:
        proprio = dict(unpack_proprio_obs(obs, env_id=env_id))
        ee_pos = np.asarray(proprio["ee_pos"], dtype=float).ravel()[:3]
        ee_quat = np.asarray(proprio["ee_quat"], dtype=float).ravel()[:4]
        measured = float(np.asarray(proprio.get("gripper_pos", [0.0]),
                                    dtype=float).ravel()[0])
        return {
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            # The point between the jaws -- RoboJEV's TCP, not the flange.
            "tcp": fingertip_position(ee_pos, ee_quat),
            "gripper_width": gripper_width_m(measured),
        }

    def _measured_state(self, obs, instruction: str, env: _EnvState,
                        pose: dict, env_id: int) -> dict:
        gt = self._get_env_gt_state(obs, env_id)
        if not isinstance(gt, dict) or not gt.get("objects"):
            raise PolicyError(
                "Jev decides on measured object state: run with --enable-gt-state "
                "(no ground-truth objects in this observation)")
        cfg = self.task_cfg
        obj = _find_object(gt["objects"], cfg["object"])
        container = _find_object(gt["objects"], cfg["container"])
        if obj is None or container is None:
            raise PolicyError(
                f"scene is missing {cfg['object']!r} or {cfg['container']!r}; "
                f"available: {sorted(gt['objects'])}")
        obj_pos = np.asarray(obj["pos"], dtype=float).ravel()[:3]
        # ``or`` on a numpy array raises; these come off the exporter as arrays.
        velocity = obj.get("vel")
        velocity = np.zeros(3) if velocity is None else np.asarray(velocity, dtype=float).ravel()[:3]
        quaternion = obj.get("quat")
        quaternion = (1.0, 0.0, 0.0, 0.0) if quaternion is None else quaternion
        contacts = (gt.get("robot") or {}).get("objects_in_contact")
        if env.rest_z is None:
            env.rest_z = float(obj_pos[2])
        return build_state(
            task=instruction_text(instruction) if instruction else "",
            task_id=self.task_id,
            state_id=f"{self.episode_id}:{env_id}:{env.generation}:{env.step_id}",
            step_id=env.step_id,
            tcp=pose["tcp"],
            tcp_quaternion=pose["ee_quat"],
            object_pos=obj_pos,
            object_quaternion=quaternion,
            object_velocity=velocity,
            container_pos=np.asarray(container["pos"], dtype=float).ravel()[:3],
            gripper_width=pose["gripper_width"],
            gripper_target=env.gripper_target,
            objects_in_contact=list(contacts) if contacts is not None else [],
            object_label=cfg["object"],
            container_label=cfg["container"],
            grasp_hover_m=cfg["grasp_hover_m"],
            release_hover_m=cfg["release_hover_m"],
            container_radius_m=cfg["container_radius_m"],
            object_rest_z=env.rest_z,
            transport_clearance_m=cfg["transport_clearance_m"],
            object_long_axis_local=_long_axis_local(obj, cfg["object_long_axis_local"]),
            previous_action_stalled=env.stalled,
            direction_tolerance_m=tolerance_for_step(self.step_m),
            recent_actions=env.history,
        )

    # -- control loop ---------------------------------------------------

    def infer(self, obs, instruction: str, *, env_id: int = 0) -> dict:
        env = self._env_state(env_id)
        pose = self._pose(obs, env_id)

        # Latch the orientation to hold on this env's first step, so a drifting
        # DLS solution cannot redefine "pointing down" one step at a time.
        if env.quat_home is None:
            env.quat_home = pose["ee_quat"].copy()
            env.quat_ref = env.quat_home.copy()
            logger.info("[Jev] env %d holding orientation %s",
                        env_id, np.round(env.quat_ref, 4).tolist())

        if env.settle_left > 0:
            env.settle_left -= 1
        elif env.steps_left <= 0 and not env.holding_pattern:
            self._decide(obs, instruction, env, pose, env_id=env_id)

        return {"action": self._action_for_step(env, pose), "viz": None}

    def _decide(self, obs, instruction: str, env: _EnvState, pose: dict,
                *, env_id: int) -> None:
        """One two-stage decision, or no motion at all if anything is invalid."""
        self._close_out_history(env, pose)
        try:
            state = self._measured_state(obs, instruction, env, pose, env_id)
            intent = self.backend.decide_intent(state)
            if intent not in ("finish",):
                env.finish_streak = 0
            answers = self.backend.decide_motor(state, intent)
            self._aim_jaws(state, env, env_id=env_id)
            decision = EefDecision(state_id=state["state_id"], intent=intent,
                                   **{k: answers[k] for k in ("x", "y", "z", "gripper")})
            delta, grip, reason = self.guard.resolve(state, decision, self.step_m)
        except PolicyError as exc:
            # RoboJEV executes NO action on an invalid decision. Hold for one
            # control step, then ask again rather than replaying a stale plan.
            env.failures += 1
            logger.warning("[Jev] env %d decision rejected (%d/%d): %s",
                           env_id, env.failures, MAX_CONSECUTIVE_FAILURES, exc)
            if env.failures >= MAX_CONSECUTIVE_FAILURES:
                raise
            env.per_step_cmd = np.zeros(3)
            env.steps_left = 1
            env.pending = None
            return
        except Exception:
            env.failures += 1
            logger.exception("[Jev] env %d decision failed (%d/%d)",
                             env_id, env.failures, MAX_CONSECUTIVE_FAILURES)
            if env.failures >= MAX_CONSECUTIVE_FAILURES:
                raise
            env.per_step_cmd = np.zeros(3)
            env.steps_left = 1
            env.pending = None
            return

        env.failures = 0
        env.step_id += 1
        env.gripper_target = grip
        requested = np.zeros(3) if delta is None else delta
        # Commanded, not achieved: rel-IK delivers ~28% of what it is asked for,
        # and the total is spread across the control steps of this decision.
        env.per_step_cmd = requested / REL_IK_ACHIEVED_FRACTION / self.control_steps
        env.steps_left = self.control_steps
        env.pending = {"action": decision.action_dict(), "intent": intent,
                       "result": reason,
                       "delta_requested_m": [round(float(v), 4) for v in requested]}

        # Enough to diagnose a stuck episode from the log alone: where the grasp
        # centre is, how far it still has to go, and what the LAST decision
        # actually achieved as opposed to what it asked for.
        grasp_rel = state["relations"]["grasp_tcp_from_tcp"]
        last = env.history[-1] if env.history else {}
        logger.info(
            "[Jev] env %d intent=%s xyz=%s grip=%s -> %s | tcp_z=%.3f "
            "grasp_err=%s yaw_err=%.0fdeg | last moved %s%s",
            env_id, intent, [decision.x, decision.y, decision.z], grip, reason,
            state["robot"]["tcp_position"][2], grasp_rel["delta_m"],
            np.degrees(state["relations"]["grasp_yaw_error_rad"]),
            last.get("delta_measured_m", "n/a"),
            " STALLED" if env.stalled else "")

        if intent == "finish":
            env.finish_streak += 1
            if env.finish_streak >= FINISH_HOLD_STREAK:
                # Not a success claim: the harness owns the verdict. This only
                # stops paying for the same call while the arm holds still.
                env.holding_pattern = True
                logger.info("[Jev] env %d finish x%d -- holding, no further calls",
                            env_id, env.finish_streak)

    def _aim_jaws(self, state: dict, env: _EnvState, *, env_id: int) -> None:
        """Turn the wrist so the jaws close ACROSS the object, not along it.

        The 2F-85 opens 85 mm and a banana is ~190 mm long: closing along its
        length does not fit, so this is a prerequisite for the grasp rather than
        a refinement. The angle is computed, not asked for -- RoboJEV engineers
        its geometric helpers explicitly and leaves the model the discrete
        choice, and a small model has no reliable rotation channel anyway.

        Only the reference moves; the orientation hold already running every
        control step drives the wrist onto it, and the top-down approach that
        the reference encodes is preserved exactly.
        """
        long_rad = state["relations"]["object_long_axis_rad"]
        if long_rad is None or env.quat_home is None:
            return
        if state["robot"]["held_object"] is not None:
            # Spinning the wrist with the object in the jaws would twist it out.
            return
        aimed = yaw_aligned_quat(env.quat_home, float(long_rad) + np.pi / 2.0)
        if not np.allclose(aimed, env.quat_ref, atol=1e-6):
            logger.info("[Jev] env %d aiming jaws across object long axis "
                        "%.1f deg (error %.1f deg)", env_id,
                        np.degrees(long_rad),
                        np.degrees(state["relations"]["grasp_yaw_error_rad"]))
        env.quat_ref = aimed

    def _close_out_history(self, env: _EnvState, pose: dict) -> None:
        """Record what the previous decision physically achieved."""
        tip = np.asarray(pose["tcp"], dtype=float)
        if env.pending is not None:
            measured = (tip - env.prev_tip) if env.prev_tip is not None else np.zeros(3)
            entry = dict(env.pending)
            entry["delta_measured_m"] = [round(float(v), 4) for v in measured]
            # Asked for a full step and got nothing: the arm is against something.
            # Without this the policy pushes into the obstruction for the rest of
            # the episode, which looks exactly like a frozen arm.
            requested = np.asarray(entry["delta_requested_m"], dtype=float)
            env.stalled = bool(requested.any()
                               and float(np.linalg.norm(measured)) < STALL_PROGRESS_M)
            entry["stalled"] = env.stalled
            env.history = (env.history + [entry])[-HISTORY_LENGTH:]
            env.pending = None
        env.prev_tip = tip.copy()

    def _action_for_step(self, env: _EnvState, pose: dict) -> np.ndarray:
        action = np.zeros(7)
        if env.steps_left > 0:
            action[:3] = np.clip(env.per_step_cmd, -MAX_DELTA_M, MAX_DELTA_M) / self.ik_scale
            env.steps_left -= 1
        # Recomputed every control step against the CURRENT pose: that freshness
        # is the whole point, and zeros here would integrate the drift instead.
        if env.quat_ref is not None:
            action[3:6] = hold_orientation_rotvec(env.quat_ref, pose["ee_quat"]) / self.ik_scale
        action[6] = GRIPPER_CLOSE if env.gripper_target == "close" else GRIPPER_OPEN
        return action

    # -- base-class hooks (unused: infer owns the loop, but keep them honest) --

    def _extract_observation(self, raw_obs, *, env_id: int = 0) -> dict:
        return {"pose": self._pose(raw_obs, env_id)}

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        raise NotImplementedError("JevClient.infer owns the two-stage loop")

    def _query_server(self, request: dict) -> dict:
        raise NotImplementedError("JevClient.infer owns the two-stage loop")

    def _unpack_response(self, response: dict) -> np.ndarray:
        raise NotImplementedError("JevClient.infer owns the two-stage loop")
