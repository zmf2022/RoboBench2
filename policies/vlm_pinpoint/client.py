# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pointing policy client: VLM point -> geometric servo -> rel-IK actions.

One VLM query per PRIMITIVE, not per N control steps. The model gets the wrist
frame, answers POINT/PHASE, and that becomes a frozen 3D target plus a short
waypoint list (stage above -> descend -> close -> lift). The client then servos
those waypoints against live proprioception and only asks again when the
primitive finishes -- the model never sits inside the control loop, which is how
every published pointing stack is wired (RoboPoint, MOKA, PIVOT).

The orientation slots carry a correction back to the pose latched at episode
start. They are NOT optional: zeros there let the wrist drift, and the wrist
camera is this policy's only sensor.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import cv2
import numpy as np

from policies.vlm_agent.client import extract_views
from policies.vlm_agent.token_controller import (
    CAMERA_ROTATIONS,
    EMPTY_GRASP_WIDTH_M,
    GRIPPER_OPEN,
    MAX_DELTA_M,
    camera_label,
    gripper_width_m,
    hold_orientation_rotvec,
    preprocess_for_vlm,
    rotate_image,
)
from policies.vlm_pinpoint.connector import fingertip_position
from policies.vlm_pinpoint.pointing_backend import VLLMPointingBackend
from policies.vlm_pinpoint.pointing_controller import (
    GRIPPER_CONTACT_SLACK_M,
    GRIPPER_MIN_TRAVEL_M,
    GRIPPER_SETTLE_MAX_STEPS,
    GRIPPER_SETTLE_STEPS,
    GRIPPER_SQUEEZE_M,
    GRIPPER_STOPPED_FRACTION,
    SERVO_M_PER_STEP,
    STALL_PROGRESS_M,
    STALL_STEPS,
    MissingDepthError,
    NoValidDepthError,
    PointingController,
    Waypoint,
    project,
    sensor_point_to_frame,
    waypoint_budget,
    yaw_aligned_quat,
)
from robolab.core.observations.observation_utils import (
    _label_panel,
    unpack_proprio_obs,
)
from robolab.eval.base_client import InferenceClient

# Which observation keys and which view label each --point-camera choice means.
# The external camera is FIXED: its extrinsics never move, the whole workspace
# stays in frame for the whole episode, and it is not rotated (CAMERA_ROTATIONS
# only turns the wrist view).
# Which camera --point-camera selects. ``keys`` is the single source of truth:
# the RGB view is matched by the label these keys produce, and _harness_obs reads
# depth/K/extrinsics from the same keys, so the frame the model points on and the
# frustum the point is resolved against cannot drift apart.
_CAMERAS = {
    "external": {
        "keys": ("over_shoulder_left_camera", "over_shoulder_left_cam"),
        "role": "external",
    },
    "wrist": {
        "keys": ("wrist_cam", "wrist_camera"),
        "role": "wrist",
    },
}

logger = logging.getLogger(__name__)

# Consecutive per-env query failures tolerated before giving up, mirroring
# vlm_agent. One malformed reply must not kill a sweep that has already banked
# episodes, but a server that is down should not burn a whole episode issuing
# hold actions either.
MAX_CONSECUTIVE_FAILURES = 10

# Servo-kind phases are re-queried every point_every steps so the VLM stays in
# the loop; terminal primitives (GRASP/LIFT/PLACE) run to exhaustion.
SERVO_KINDS = ("APPROACH", "ALIGN")
# Fresh target within this of the leg being servoed keeps the old plan: points
# carry ~20-50 px of noise (>1 cm at grasp range), and chasing it re-aims the
# arm every window instead of converging.
REPLAN_HYST_M = 0.03


@dataclass
class _EnvState:
    # The primitive currently executing, and how far into it we are. An empty
    # plan (or one walked to the end) is what triggers the next VLM query.
    plan: list[Waypoint] = field(default_factory=list)
    wp_idx: int = 0
    wp_steps: int = 0
    wp_budget: int = 0
    wp_best: float = float("inf")   # closest this leg has come, for stall detection
    wp_stalled: int = 0             # consecutive steps that bought no ground
    grip_ref: float = float("inf")  # jaw width at the start of the settle window
    grip_open: float = float("inf")  # jaw width when this dwell began
    grip_peak: float = 0.0          # fastest window of travel seen this dwell
    grip_still: int = 0             # steps elapsed in the current settle window
    gripper_cmd: float = GRIPPER_OPEN
    hold_steps: int = 0
    finished: bool = False
    last_point: dict | None = None
    failures: int = 0
    depth_viz: np.ndarray | None = None
    # What kind of plan is executing (the phase that produced it), whether it may
    # be interrupted by a fresh query, and how long since the last VLM query.
    # Atomicity comes from plan CONTENT, not the phase string: a staged
    # auto-grasp builds a close-dwell plan under phase APPROACH, and killing it
    # on the next window aborts every descend halfway (tip bobs 0.10-0.14 m
    # forever, never reaching the 0.014 m close height). SERVO_KINDS re-query
    # every point_every steps; terminal primitives run to exhaustion.
    plan_kind: str = ""
    plan_atomic: bool = False
    since_query: int = 0
    # Orientation to hold for the whole episode, latched on the first step.
    quat_ref: np.ndarray | None = None


class PinpointClient(InferenceClient):
    """InferenceClient driving rel-IK from VLM image points."""

    def __init__(
        self,
        backend: VLLMPointingBackend,
        controller: PointingController | None = None,
        ik_scale: float = 0.5,
        point_every: int = 8,
        send_depth_image: bool = True,
        point_camera: str = "external",
        viz_dir: str | None = None,
    ) -> None:
        super().__init__()
        self.backend = backend
        self.controller = controller or PointingController()
        self.ik_scale = float(ik_scale)
        # Only a dwell length now (how long to stand still after a failed query
        # before asking again); queries are driven by primitive completion.
        self.point_every = int(point_every)
        # Whether the depth map also goes to the MODEL. Depth is always used
        # numerically for the unprojection -- this only controls the second
        # image. No published pointing stack feeds depth to the VLM (they ground
        # on RGB and consume depth geometrically), so it is worth A/B-ing.
        self.send_depth_image = bool(send_depth_image)
        if point_camera not in _CAMERAS:
            raise ValueError(f"point_camera must be one of {sorted(_CAMERAS)}")
        self.point_camera = point_camera
        self._cam = _CAMERAS[point_camera]
        self.view_rot_deg = int(CAMERA_ROTATIONS.get(self._cam["role"], 0))
        self.viz_dir = viz_dir
        if viz_dir:
            os.makedirs(viz_dir, exist_ok=True)
        self._decision = 0
        self._envs: dict[int, _EnvState] = {}

    def begin_episode(self, episode_idx: int) -> None:
        super().begin_episode(episode_idx)
        self._envs.clear()

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._envs.clear()
        else:
            self._envs.pop(env_id, None)
        super().reset(env_id=env_id)

    def close(self) -> None:
        closer = getattr(self.backend, "close", None)
        if callable(closer):
            closer()

    def _env_state(self, env_id: int) -> _EnvState:
        if env_id not in self._envs:
            self._envs[env_id] = _EnvState()
        return self._envs[env_id]

    # -- base-class hooks (single-shot semantics; the live path is infer()) --
    def _extract_observation(self, raw_obs, *, env_id: int = 0) -> dict:
        views = dict(extract_views(raw_obs, env_id))
        # Match the label the SAME camera key produces, not a prefix of it.
        # ``camera_label`` names every non-wrist camera "AgentView (...)", so a
        # startswith("AgentView") silently takes whichever external view sorts
        # first while ``_harness_obs`` keeps reading depth/K/extrinsics from
        # ``self._cam["keys"]`` -- RGB from one camera, geometry from another,
        # with nothing to see in the logs. Exact match turns a second external
        # camera into a loud error instead of a wrong answer.
        wanted = {camera_label(key) for key in self._cam["keys"]}
        frame = next((img for lbl, img in views.items() if lbl in wanted), None)
        if frame is None:
            raise RuntimeError(
                f"--point-camera {self.point_camera} needs one of {sorted(wanted)}; "
                f"got {sorted(views)}. Check the camera list in run.py."
            )
        proprio = dict(unpack_proprio_obs(raw_obs, env_id=env_id))
        ee_pos = np.asarray(proprio["ee_pos"], dtype=float).ravel()[:3]
        ee_quat = np.asarray(proprio["ee_quat"], dtype=float).ravel()[:4]
        return {
            "views": views,
            "frame": preprocess_for_vlm(frame),
            "tip": fingertip_position(ee_pos, ee_quat),
            "ee_quat": ee_quat,
            "gripper": proprio.get("gripper_pos"),
            "harness_obs": self._harness_obs(raw_obs, env_id),
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        return {
            "frame": extracted_obs["frame"],
            "instruction": instruction,
            "tip": extracted_obs["tip"],
            "harness_obs": extracted_obs["harness_obs"],
        }

    def _query_server(self, request: dict) -> dict:
        point = self.backend.point(request["frame"], request["instruction"])
        return {"point": point, "tip": request["tip"],
                "harness_obs": request["harness_obs"]}

    def _unpack_response(self, response: dict) -> np.ndarray:
        try:
            delta, gripper, _done = self.controller.step(
                response["harness_obs"], response["point"], response["tip"])
        except NoValidDepthError:
            return np.zeros((self.point_every, 7))
        except MissingDepthError as exc:
            raise RuntimeError(str(exc)) from exc
        step = np.zeros(7)
        step[:3] = np.clip(np.asarray(delta, dtype=float).ravel()[:3] / self.point_every,
                           -0.05, 0.05) / self.ik_scale
        step[6] = float(gripper)
        return np.tile(step, (self.point_every, 1))

    def infer(self, obs, instruction: str, *, env_id: int = 0) -> dict:
        state = self._env_state(env_id)
        extracted = self._extract_observation(obs, env_id=env_id)
        frame, tip, ee_quat = extracted["frame"], extracted["tip"], extracted["ee_quat"]

        # Latch the orientation to hold on this env's first step of the episode.
        if state.quat_ref is None:
            state.quat_ref = np.asarray(ee_quat, dtype=float).copy()
            logger.info("[Pinpoint] env %d holding orientation %s",
                        env_id, np.round(state.quat_ref, 4).tolist())

        if state.wp_idx >= len(state.plan) and not state.finished:
            self._plan_next(state, obs, frame, instruction, tip, env_id=env_id)
            state.since_query = 0
        elif (not state.plan_atomic and state.plan_kind in SERVO_KINDS
                and not state.finished
                and state.since_query >= self.point_every):
            self._plan_next(state, obs, frame, instruction, tip, env_id=env_id)
            state.since_query = 0
        else:
            state.since_query += 1

        # Preview depth tracks every step (VLM queries only refresh the plan).
        # Fenced off: preview must never break control (e.g. in tests with stub
        # observations, or when a camera term is momentarily absent).
        if self.send_depth_image:
            try:
                fresh = self._harness_obs(obs, env_id).get("depth")
            except Exception:  # noqa: BLE001 - preview only, control continues
                fresh = None
            state.depth_viz = (self._depth_panel(fresh, frame.shape, self.view_rot_deg)
                               if fresh is not None else None)

        action = np.zeros(7)
        action[:3] = np.clip(self._advance(state, tip, env_id, extracted.get("gripper")),
                             -MAX_DELTA_M, MAX_DELTA_M) / self.ik_scale
        # Zeros here do NOT hold the orientation: in relative mode a zero
        # rotation delta means "target = whatever you have right now", so the
        # saturated DLS solver's orientation error silently becomes the next
        # setpoint and compounds. Measured upstream at 24.5 deg of drift by the
        # end of an episode -- and this policy's ONLY input is the wrist camera,
        # so a tilted gripper means the model is pointing in a frame that no
        # longer matches the geometry. See vlm_agent.kinematics.
        #
        # The reference is the latched pose SPUN to the current waypoint's jaw
        # angle when it has one. A 2F-85 opens 85 mm and a banana is 190 mm long,
        # so jaws that close along the object's long axis cannot grip it at all;
        # the yaw is measured off the depth map in the controller, because the
        # model has no rotation channel to ask.
        action[3:6] = hold_orientation_rotvec(
            self._orientation_goal(state), ee_quat) / self.ik_scale
        action[6] = state.gripper_cmd
        return {"action": action, "viz": self._viz(frame, state)}

    def _orientation_goal(self, state: _EnvState) -> np.ndarray:
        """Latched pose, turned to the active waypoint's jaw angle if it has one."""
        if state.wp_idx < len(state.plan):
            yaw = state.plan[state.wp_idx].yaw_rad
            if yaw is not None:
                return yaw_aligned_quat(state.quat_ref, yaw)
        return state.quat_ref

    @staticmethod
    def _gripper_settled(state: _EnvState, gripper_pos, wp: Waypoint) -> bool:
        """Has the dwell done its job — have the fingers stopped moving?

        Ending a closing dwell on a step count alone starts the lift while the
        jaws are still travelling, which drags the object out of them. The floor
        is ``hold_steps`` because the 2F-85's joint command is binary and only
        travels while it is HELD; above that the measured width decides, and the
        ceiling keeps a jammed close from pinning the episode.

        ``wp.close_to_m`` -- the object's measured width -- is what makes this
        honest. The jaws are done when they have closed PAST it (the pads are
        compliant), or when they stall while already near it. A stall while still
        much wider than the object is not a grasp: it is an actuator that has not
        got going, and accepting it starts the lift with the object untouched.
        """
        width = None
        if gripper_pos is not None:
            width = gripper_width_m(float(gripper_pos))
            if state.grip_open == float("inf"):
                state.grip_open = width      # where the jaws stood when asked

        if state.wp_steps < max(1, wp.hold_steps):
            return False
        if width is None or state.wp_steps >= GRIPPER_SETTLE_MAX_STEPS:
            return True
        if width <= EMPTY_GRASP_WIDTH_M:
            return True                      # fully shut: no travel left to wait for
        # The one end condition that is a MEASUREMENT rather than a guess about an
        # undocumented actuator: the jaws have closed onto the object, and a touch
        # past it so the compliant pads are actually loaded.
        if wp.close_to_m is not None and width <= wp.close_to_m - GRIPPER_SQUEEZE_M:
            return True
        # Stillness only means "finished" once the jaws have actually MOVED.
        # Before the joint starts they are just as still as when they have
        # stopped, and treating that as done begins the lift mid-close.
        if abs(state.grip_open - width) < GRIPPER_MIN_TRAVEL_M:
            return False
        state.grip_still += 1
        if state.grip_still < GRIPPER_SETTLE_STEPS:
            return False
        previous, state.grip_ref, state.grip_still = state.grip_ref, width, 0
        if not np.isfinite(previous):
            return False                     # first window: nothing to compare to
        moved = abs(width - previous)
        state.grip_peak = max(state.grip_peak, moved)
        # A stall out here is the actuator, not the object: nothing to grip yet.
        if (wp.close_to_m is not None
                and width > wp.close_to_m + GRIPPER_CONTACT_SLACK_M):
            return False
        # Stopped RELATIVE to how fast these jaws actually travel. An absolute
        # threshold has to be finer than the actuator's per-step motion, and this
        # one moves 0.16 mm per step -- slower than any epsilon worth writing.
        return moved < GRIPPER_STOPPED_FRACTION * state.grip_peak

    def _advance(self, state: _EnvState, tip, env_id: int,
                 gripper_pos=None) -> np.ndarray:
        """Metres to command this control step, servoing the CURRENT waypoint.

        Closed on live proprioception rather than replaying a precomputed delta:
        the fingertip is re-measured every step, so IK slip and the orientation
        correction do not accumulate into the target.
        """
        if state.wp_idx >= len(state.plan):
            return np.zeros(3)
        wp = state.plan[state.wp_idx]
        state.gripper_cmd = float(wp.gripper)
        state.wp_steps += 1
        if state.wp_steps == 1:
            state.grip_ref, state.grip_still = float("inf"), 0
            state.grip_open, state.grip_peak = float("inf"), 0.0

        if wp.pos is None:  # dwell in place (the grasp/release hold)
            if state.wp_steps == 1 and state.wp_idx:
                # Where the hand ACTUALLY ended up versus where the descent was
                # aimed. "The grasp is too high" has two quite different causes --
                # a closure height planned too high, or relative IK that ran out
                # of travel on the way down -- and they are indistinguishable
                # without both numbers side by side.
                aimed = state.plan[state.wp_idx - 1].pos
                if aimed is not None:
                    logger.info(
                        "[Pinpoint] env %d closing at z=%.3f m (aimed %.3f, "
                        "%+.1f mm)", env_id, float(tip[2]), float(aimed[2]),
                        1000 * (float(tip[2]) - float(aimed[2])))
            if gripper_pos is not None and state.wp_steps % 25 == 0:
                logger.info("[Pinpoint] env %d jaws %.1f mm after %d steps",
                            env_id, 1000 * gripper_width_m(float(gripper_pos)),
                            state.wp_steps)
            if self._gripper_settled(state, gripper_pos, wp):
                width = (gripper_width_m(float(gripper_pos))
                         if gripper_pos is not None else None)
                if (wp.pos is None and wp.gripper == 1.0 and width is not None
                        and width <= EMPTY_GRASP_WIDTH_M):
                    # Closed on air: fingers fully shut means nothing is between
                    # them. Lifting now just restarts the loop from higher up —
                    # drop the rest of the plan (including the lift) and re-query.
                    logger.info(
                        "[Pinpoint] env %d grasp missed (jaws fully shut at %.1f mm); "
                        "replanning WITHOUT lifting", env_id, 1000 * width)
                    state.plan, state.wp_idx, state.wp_steps = [], 0, 0
                    state.plan_kind = ""
                    state.plan_atomic = False
                    return np.zeros(3)
                logger.info(
                    "[Pinpoint] env %d jaws done after %d steps: %.1f -> %.1f mm "
                    "(target %s)", env_id, state.wp_steps, 1000 * state.grip_open,
                    1000 * gripper_width_m(float(gripper_pos))
                    if gripper_pos is not None else float("nan"),
                    "none" if wp.close_to_m is None
                    else "object %.1f mm, close to %.1f mm" % (
                        1000 * wp.close_to_m, 1000 * (wp.close_to_m - GRIPPER_SQUEEZE_M)))
                self._next_waypoint(state)
            return np.zeros(3)

        delta = np.asarray(wp.pos, dtype=float).ravel()[:3] - np.asarray(tip, dtype=float).ravel()[:3]
        dist = float(np.linalg.norm(delta))
        if state.wp_steps == 1:
            # Budget this leg by how far it actually is. A flat cap starves long
            # legs and not short ones, so the arm descends but never travels.
            state.wp_budget = waypoint_budget(dist)
            state.wp_best, state.wp_stalled = dist, 0
        if dist <= wp.tol_m:
            self._next_waypoint(state)
            return np.zeros(3)

        # A free leg converges geometrically, so a run of steps that buys no
        # ground means the hand is against something. On a grasp descent that is
        # ARRIVAL, not failure -- see Waypoint.on_stall.
        if dist < state.wp_best - STALL_PROGRESS_M:
            state.wp_best, state.wp_stalled = dist, 0
        else:
            state.wp_stalled += 1
        stalled = state.wp_stalled >= STALL_STEPS
        if stalled or state.wp_steps >= state.wp_budget:
            why = "contact" if stalled else "budget"
            if wp.on_stall == "advance":
                logger.info(
                    "[Pinpoint] env %d waypoint %d stopped on %s %.3f m short "
                    "after %d steps; closing here",
                    env_id, state.wp_idx, why, dist, state.wp_steps)
                self._next_waypoint(state)
                return np.zeros(3)
            logger.warning(
                "[Pinpoint] env %d waypoint %d unreached (%s) after %d/%d steps "
                "(%.3f m short); re-planning",
                env_id, state.wp_idx, why, state.wp_steps, state.wp_budget, dist)
            state.plan = []  # force a fresh query from wherever the arm is
            state.wp_idx = state.wp_steps = 0
            return np.zeros(3)
        return delta / dist * min(dist, SERVO_M_PER_STEP)

    @staticmethod
    def _next_waypoint(state: _EnvState) -> None:
        state.wp_idx += 1
        state.wp_steps = 0

    def _mark_gripper(self, frame: np.ndarray, hobs: dict):
        """Draw the fingertip into the frame the model is about to see.

        In a fixed third-person view the model has no way to know which of the
        things on screen is the hand it is driving — the arm is just another
        object. Projecting the fingertip back through the same intrinsics and
        marking it is what makes "move the gripper to the banana" a relationship
        it can reason about, instead of a guess. Same trick as MOKA / Set-of-Mark.
        """
        uv = project(hobs.get("tip_pos"), K=hobs.get("K"),
                     cam_pos=hobs.get("cam_pos"), cam_quat=hobs.get("cam_quat"),
                     ee_pos=hobs.get("ee_pos"), ee_quat=hobs.get("ee_quat"))
        ref = hobs.get("depth")
        if ref is None:
            ref = hobs.get("scene")
        if uv is None or ref is None:
            return frame, None
        sh, sw = np.asarray(ref).shape[:2]
        fx, fy = sensor_point_to_frame(uv[0], uv[1], sw, sh, self.view_rot_deg)
        if not (0.0 <= fx <= 1.0 and 0.0 <= fy <= 1.0):
            return frame, None  # the hand is out of this view's frustum
        h, w = frame.shape[:2]
        cx, cy = int(fx * w), int(fy * h)
        marked = frame.copy()
        cv2.drawMarker(marked, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
        cv2.circle(marked, (cx, cy), 13, (0, 255, 0), 2)
        cv2.putText(marked, "GRIPPER", (cx + 16, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return marked, (fx, fy)

    @staticmethod
    def _state_text(hobs: dict, marked) -> str:
        """The proprioception the model cannot read off a picture.

        Whether the fingers are actually holding something is invisible in an
        image — a closed empty gripper and a closed loaded one look identical —
        and it is exactly what the GRASP/LIFT/DONE decision turns on.
        """
        parts = []
        if marked is not None:
            parts.append(
                f"A green crosshair marks YOUR GRIPPER, at ({marked[0]:.2f}, "
                f"{marked[1]:.2f}) in normalized image coordinates.")
        else:
            parts.append("Your gripper is currently outside this view.")
        grip = float(hobs.get("gripper_pos", 0.0))
        width = gripper_width_m(grip)
        if grip > 0.5:
            if width > EMPTY_GRASP_WIDTH_M:
                parts.append(f"The gripper is CLOSED and holding something "
                             f"({width * 100:.1f} cm across).")
            else:
                parts.append("The gripper is CLOSED but EMPTY — the fingers are "
                             "touching, so the last grasp FAILED and you must "
                             "try to pick the object up again.")
        else:
            parts.append("The gripper is OPEN and holding nothing.")
        tip = np.asarray(hobs.get("tip_pos", np.zeros(3)), dtype=float).ravel()
        parts.append(f"The fingertips are {tip[2] * 100:.1f} cm above the table.")
        return " ".join(parts)

    def _save_viz(self, frame, depth_viz, point, env_id: int) -> None:
        """Persist what the model saw and what it answered.

        The recorded mp4 is the harness's own camera tile and the policy overlay
        only reaches cv2.imshow, so under --headless there is otherwise no way to
        tell a bad point from bad geometry.
        """
        shot = frame.copy()
        h, w = shot.shape[:2]
        cx, cy = int(point["x"] / 1000.0 * w), int(point["y"] / 1000.0 * h)
        cv2.circle(shot, (cx, cy), 9, (255, 0, 0), 2)
        cv2.drawMarker(shot, (cx, cy), (255, 0, 0), cv2.MARKER_TILTED_CROSS, 10, 2)
        cv2.putText(shot, point["phase"], (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        if depth_viz is not None:
            shot = np.concatenate([shot, np.asarray(depth_viz)], axis=1)
        name = f"{self._decision:04d}_env{env_id}_{point['phase']}.png"
        cv2.imwrite(os.path.join(self.viz_dir, name),
                    cv2.cvtColor(shot, cv2.COLOR_RGB2BGR))

    @staticmethod
    def _depth_panel(depth, frame_shape, rot_deg):
        """Display/VLM copy of a (already env-sliced) depth map, rotated exactly
        like its RGB frame. Sampling in the controller still uses RAW sensor
        pixels."""
        from policies.vlm_pinpoint.pointing_backend import encode_depth_for_vlm
        depth_rot = rotate_image(np.asarray(depth), rot_deg)
        return cv2.resize(encode_depth_for_vlm(depth_rot),
                          (frame_shape[1], frame_shape[0]))

    def _plan_next(self, state, obs, frame, instruction, tip, *, env_id: int = 0) -> None:
        hobs = self._harness_obs(obs, env_id)
        self._decision += 1
        frame, marked = self._mark_gripper(frame, hobs)
        depth_viz = None
        if self.send_depth_image and hobs.get("depth") is not None:
            # Rotated exactly like the RGB frame it is paired with: the prompt
            # tells the model the two share a viewpoint. Sampling in the
            # controller still uses RAW sensor pixels.
            depth_viz = self._depth_panel(hobs["depth"], frame.shape,
                                          self.view_rot_deg)
        state.depth_viz = depth_viz
        try:
            point = self.backend.point(frame, instruction, depth_image=depth_viz,
                                       state_text=self._state_text(hobs, marked),
                                       view=self.point_camera)
        except Exception as exc:  # noqa: BLE001 - transport failures count too
            # ValueError is an unparsable reply; a timeout or a dropped socket is
            # just as survivable for one decision. Only a run of them means the
            # server is really gone — and holding for a whole episode instead of
            # failing loudly is worse than stopping.
            state.failures += 1
            logger.warning(
                "[Pinpoint] env %d no usable point (%d/%d consecutive): %s; holding",
                env_id, state.failures, MAX_CONSECUTIVE_FAILURES, exc)
            if state.failures >= MAX_CONSECUTIVE_FAILURES:
                raise
            self._hold(state)
            return
        state.failures = 0
        state.last_point = point
        if self.viz_dir:
            self._save_viz(frame, depth_viz, point, env_id)
        logger.info(
            "[Pinpoint] env %d POINT %d %d PHASE %s tip=%s",
            env_id, point["x"], point["y"], point["phase"],
            np.round(tip, 3).tolist(),
        )
        kind = str(point.get("phase", "APPROACH")).upper()
        try:
            fresh_target = self.controller._target_for(hobs, point)
        except NoValidDepthError as exc:
            logger.warning("[Pinpoint] env %d %s; holding", env_id, exc)
            self._hold(state)
            return
        except MissingDepthError as exc:
            raise RuntimeError(str(exc)) from exc
        if self._target_close(state, kind, fresh_target):
            logger.info("[Pinpoint] env %d target stable (<%.0f mm), keeping plan",
                        env_id, REPLAN_HYST_M * 1000)
            state.plan_kind = kind
            return
        try:
            waypoints, done = self.controller.plan(hobs, point, tip)
        except NoValidDepthError as exc:
            # Pointed at sky: hold still this window, re-query next time.
            # (Absent depth key is a config error and still crashes below.)
            logger.warning("[Pinpoint] env %d %s; holding", env_id, exc)
            self._hold(state)
            return
        except MissingDepthError as exc:
            raise RuntimeError(str(exc)) from exc
        if done:
            state.finished = True
            state.plan, state.wp_idx, state.wp_steps = [], 0, 0
            state.plan_kind = ""
            state.plan_atomic = False
            return
        logger.info(
            "[Pinpoint] env %d plan %s",
            env_id,
            [("hold", wp.gripper, wp.hold_steps) if wp.pos is None
             else (np.round(wp.pos, 3).tolist(), wp.gripper) for wp in waypoints],
        )
        state.plan, state.wp_idx, state.wp_steps = waypoints, 0, 0
        state.plan_kind = kind
        state.plan_atomic = self._is_atomic(kind, waypoints)

    @staticmethod
    def _is_atomic(kind: str, waypoints) -> bool:
        """A plan runs to exhaustion iff it commits a dwell.

        Atomicity comes from CONTENT, not the phase string: a staged auto-grasp
        arrives here under phase APPROACH carrying a close dwell, and
        interrupting it on the next window aborts every descend halfway (tip
        bobs 0.10-0.14 m forever, never reaching the close height).
        """
        return (kind not in SERVO_KINDS
                or any(getattr(wp, "pos", None) is None for wp in waypoints))

    @staticmethod
    def _target_close(state: _EnvState, kind: str, fresh_target) -> bool:
        """True when the fresh target matches the leg being servoed.

        Same servo kind on both sides and under REPLAN_HYST_M: the new point
        is measurement noise on the old target, so replacing the plan would
        re-aim the arm instead of converging.
        """
        if not state.plan or state.plan_kind not in SERVO_KINDS:
            return False
        if kind not in SERVO_KINDS:
            return False
        if not (0 <= state.wp_idx < len(state.plan)):
            return False
        cur = state.plan[state.wp_idx].pos
        if cur is None:
            return False
        try:
            dist = float(np.linalg.norm(
                np.asarray(fresh_target, dtype=float).ravel()[:3]
                - np.asarray(cur, dtype=float).ravel()[:3]))
        except (TypeError, ValueError):
            return False
        return dist < REPLAN_HYST_M

    def _hold(self, state: _EnvState) -> None:
        """Stand still for one query window, then ask again.

        A real one-waypoint plan, not an empty one: leaving the plan empty would
        re-query on the very next control step and hammer the server.
        """
        state.plan = [Waypoint(None, state.gripper_cmd, self.point_every)]
        state.wp_idx = state.wp_steps = 0
        state.plan_kind = ""
        state.plan_atomic = True

    def _harness_obs(self, obs, env_id: int) -> dict:
        image_obs = obs.get("image_obs", {}) if isinstance(obs, dict) else {}
        proprio = dict(unpack_proprio_obs(obs, env_id=env_id))
        ee_pos = np.asarray(proprio.get("ee_pos"), dtype=float).ravel()[:3]
        ee_quat = np.asarray(proprio.get("ee_quat"), dtype=float).ravel()[:4]

        def first(rank: int, *keys):
            """Per-env value for the first key present, given its per-env rank.

            Every image_obs term is batched as ``(num_envs, *per_env_shape)``, so
            the env axis is the one that makes ndim exceed ``rank``. Guessing it
            from the shape instead does not work: ``cam_pos`` (N, 3) and
            ``cam_quat`` (N, 4) look identical to unbatched arrays, and for K
            (N, 3, 3) any N >= 3 is indistinguishable from a single matrix.
            """
            for key in keys:
                if key in image_obs:
                    v = image_obs[key]
                    if hasattr(v, "cpu"):
                        v = v.cpu().numpy()
                    v = np.asarray(v)
                    if v.ndim == rank + 1:
                        return np.asarray(v[env_id])
                    if v.ndim == rank:
                        return v  # already per-env (unit tests, flat obs)
                    raise ValueError(
                        f"{key}: expected per-env rank {rank} (optionally batched), "
                        f"got shape {v.shape}"
                    )
            return None

        # Everything geometric must come from the SAME camera the model pointed
        # on, or the unprojection resolves the pixel against the wrong frustum.
        base = self._cam["keys"]
        return {
            "scene": first(3, *base),
            "depth": first(3, *(f"{k}_depth" for k in base)),
            "K": first(2, *(f"{k}_K" for k in base)),
            "cam_pos": first(1, *(f"{k}_pos" for k in base)),
            "cam_quat": first(1, *(f"{k}_quat" for k in base)),
            "view_rot_deg": self.view_rot_deg,
            "ee_pos": ee_pos,
            "ee_quat": ee_quat,
            "tip_pos": fingertip_position(ee_pos, ee_quat),
            "gripper_pos": np.asarray(
                proprio.get("gripper_pos", [0.0]), dtype=float).ravel()[0],
        }

    @staticmethod
    def _viz(frame: np.ndarray, state: _EnvState) -> np.ndarray:
        viz = np.asarray(frame).copy()
        if state.last_point is not None:
            h, w = viz.shape[:2]
            cx = int(state.last_point["x"] / 1000.0 * w)
            cy = int(state.last_point["y"] / 1000.0 * h)
            cv2.circle(viz, (cx, cy), 8, (255, 0, 0), 2)
            cv2.circle(viz, (cx, cy), 2, (255, 0, 0), -1)
            cv2.putText(viz, state.last_point["phase"], (10, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        # Same labelled-panel treatment as the agent preview, plus the depth
        # map whenever one was sent — the preview mirrors the VLM inputs.
        panels = [_label_panel(viz, "WristView")]
        if state.depth_viz is not None:
            panels.append(_label_panel(np.asarray(state.depth_viz), "WristDepth"))
        return np.concatenate(panels, axis=1)
