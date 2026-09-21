# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discrete action units -> RoboLab relative-IK actions, and the prompt that grounds them.

``DroidRelIKActionCfg``'s action layout is::

    [dx, dy, dz, drx, dry, drz, gripper]

Three properties of that action space drive everything in this file, and getting any of
them wrong produces a policy that moves, logs clean numbers, and never completes a task:

**1. The translation slots are divided by the action config's ``scale`` (0.5).** An action
of 1.0 asks the IK for a 0.5 m target step. So a desired displacement in metres has to be
divided by :attr:`TokenController.ik_scale` on the way in. ``run.py`` reads that number
off ``DroidRelIKActionCfg`` at startup rather than hardcoding it, so the two cannot drift.

**2. Relative IK achieves a roughly CONSTANT ~28% of whatever is commanded.** Measured
upstream on BananaInBowlTask by sweeping (n_steps x per-step command): 4x5 mm -> 5.4 mm,
8x10 mm -> 22.2 mm, 16x5 mm -> 22.5 mm — 27-28% of the commanded total every time,
independent of both knobs. That is the rel-IK semantics: each step re-targets "current
pose + delta", so while the solver lags, every step re-bases on a position that has not
arrived. Travel does not accumulate linearly and cannot be recovered by adding steps.
:data:`STEP_M` is therefore the COMMANDED total (0.072 m), which lands on 2 cm physical:

    MV_FWD 20.30 | MV_BACK 20.28 | MV_LEFT 20.12 | MV_RIGHT 20.11 | MV_UP 19.97 | MV_DOWN 19.95
    mean 20.12 mm, sd 0.13 mm, max off-axis 0.32 mm  (0.072 m over 8 control steps)

**3. Zeros in the rotation slots do not hold the orientation** — see
:func:`kinematics.hold_orientation_rotvec`. The rotation slots are filled per control step
from the CURRENT pose, which is why one decision expands into a plan of control steps
executed by the client rather than into a precomputed action chunk.

RE-MEASURE with ``python policies/vlm_agent/probe_axes.py`` after changing the robot, the
task, or ``dt``/``decimation`` — and before trusting any success number.
"""

from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from PIL import Image

from robolab.core.utils.image_utils import convert_to_uint8
from robolab.core.utils.isaaclab_compat import quat_isaaclab_to_wxyz, quat_wxyz_to_isaaclab

logger = logging.getLogger(__name__)


########################################################
# Action vocabulary — the model-facing contract
########################################################

# Incremental end-effector translations.
MOVE_UNITS = ("MV_FWD", "MV_BACK", "MV_LEFT", "MV_RIGHT", "MV_UP", "MV_DOWN")

# Incremental end-effector yaw about the base +Z axis.
ROTATE_UNITS = ("ROTATE_CW", "ROTATE_CCW")

# Gripper units. Each is held for ``gripper_hold_steps`` control steps: the Robotiq
# 2F-85 is driven by a BINARY joint command, so the fingers need time to travel and
# load up on the object.
GRASP_UNIT = "GRASP"

RELEASE_UNIT = "RELEASE"

GRIPPER_UNITS = (GRASP_UNIT, RELEASE_UNIT)

# Hold the current setpoint for one control step.
STILL_UNIT = "STILL"

# Terminal units. NOTE: RoboLab's ``run_episode`` has no client-driven termination hook,
# so these do not end the episode — the env still runs to ``max_episode_length``. What
# they do is stop the client querying the VLM, which is the difference between ~170 and
# ~1350 model calls on a 90 s task. See ``VLMInferenceClient._plan_next``.
DONE_UNIT = "DONE"

GIVE_UP_UNIT = "GIVE_UP"

TERMINAL_UNITS = (DONE_UNIT, GIVE_UP_UNIT)

TOKEN_VOCAB = (
    *MOVE_UNITS,
    *ROTATE_UNITS,
    *GRIPPER_UNITS,
    STILL_UNIT,
    *TERMINAL_UNITS,
)


########################################################
# Pose helpers (thin adapter over isaaclab.utils.math)
########################################################

# Proportional gain and per-control-step clamp for :func:`hold_orientation_rotvec`.
ORIENT_HOLD_GAIN = 1.0
ORIENT_HOLD_MAX_RAD = 0.15

# base_link (the Robotiq 2F-85 mount flange, the body ``ee_pos``/``ee_quat`` report and
# the IK aims) -> THE POINT THE CLOSED JAWS MEET, in base_link's own frame. The IK aims
# the flange; the VLM has to reason about where its grasp lands, so reported state is
# corrected by this.
#
# Two wrong values preceded this one, and both produced the same symptom — an arm that
# descends forever and never arrives, with the object never between the jaws:
#
#   (0, 0, 0.1628)   a scalar along local +z, off the Robotiq spec sheet without checking
#                    which way base_link's z points. The fingers run along +X, so the
#                    estimate was ~22 cm off in a fixed direction.
#   (0.15, 0.03, 0)  ``robolab/eval/phases.py::TCP_OFFSET``. Right axis, wrong point: that
#                    is the *fingertip midpoint* the repo GRADES contact against, and it
#                    sits 36 mm from where a grasp actually closes — 30 mm off the jaw
#                    CENTRELINE (so the aim point lands beside the jaws, not between them)
#                    and 19 mm past the pad face, at the pads' far tip.
#
# Grading a contact and aiming a grasp are not the same point. This one is the grasp
# centre, and three independent derivations agree on it:
#   * PharmFlow's measured DROID contract, ``arena/biomedical_contract.py::
#     DROID_BASE_TO_CLOSED_GRASP_CENTER`` = (0.13108637, 0.00014421, 0.0);
#   * this rig live, from the pad colliders (``robolab/eval/tows.py::PAD_BOX``, an FK
#     validated to 0.00 mm against recorded finger poses) = (0.1301, 0.0, 0.0);
#   * the collider itself: the two pads are symmetric about base_link's y=0, so the
#     centreline y is exactly 0 by construction.
# ``offline_tests/test_vlm_agent_geometry.py`` pins it against the first and the third.
#
# The axis is still +X, cross-checked by ``tests/test_pointing_backend.py::
# test_fingertip_offset_points_where_the_wrist_camera_looks``: wrist_cam is bolted to
# base_link to watch the fingers and its optical axis is 21 deg off this vector (cos
# 0.93) versus 72 deg off +Z.
FLANGE_TO_FINGERTIP = (0.13108637, 0.0, 0.0)

# Fraction of the commanded translation relative IK actually achieves, measured on
# BananaInBowlTask (see this module's docstring). Used only to tell the model how far one
# decision really travels; the action itself is unaffected.
REL_IK_ACHIEVED_FRACTION = 0.28

# The gripper's approach axis in its own frame, and the world direction "straight down".
# +X, for the same reason FLANGE_TO_FINGERTIP is: the jaws close around a line running
# out of base_link's +X. With the old (0, 0, 1) this reported ~90 deg of "tilt" on a
# gripper that was pointing perfectly straight down, because it measured the wrong axis.
_APPROACH_AXIS_LOCAL = (1.0, 0.0, 0.0)
_STRAIGHT_DOWN = np.array([0.0, 0.0, -1.0])


def _math():
    """``isaaclab.utils.math``, imported on first use (post-AppLauncher)."""
    import isaaclab.utils.math as math_utils

    return math_utils


def _to_isaaclab(quat_wxyz) -> torch.Tensor:
    """RoboLab WXYZ (any array-like) -> a ``(1, 4)`` tensor in the installed convention."""
    quat = torch.as_tensor(np.asarray(quat_wxyz, dtype=np.float64).ravel()[:4], dtype=torch.float64)
    return quat_wxyz_to_isaaclab(quat).reshape(1, 4)


def _to_wxyz(quat_il: torch.Tensor) -> np.ndarray:
    """A tensor in the installed convention -> a RoboLab WXYZ numpy quaternion."""
    return quat_isaaclab_to_wxyz(quat_il.reshape(4)).cpu().numpy().astype(np.float64)


def quat_rotate(quat_wxyz, vec) -> np.ndarray:
    """Rotate ``vec`` by a WXYZ quaternion."""
    vec_t = torch.as_tensor(np.asarray(vec, dtype=np.float64).reshape(1, 3), dtype=torch.float64)
    return _math().quat_apply(_to_isaaclab(quat_wxyz), vec_t).reshape(3).cpu().numpy()


def yaw_reference(quat_ref_wxyz, yaw_rad: float) -> np.ndarray:
    """Advance a held-orientation reference by ``yaw_rad`` about the base +Z axis.

    World-frame pre-multiply, matching the ``ref ⊗ cur⁻¹`` error convention in
    :func:`hold_orientation_rotvec`.
    """
    if float(yaw_rad) == 0.0:
        return np.asarray(quat_ref_wxyz, dtype=np.float64).ravel()[:4]
    math_utils = _math()
    yaw = math_utils.quat_from_angle_axis(
        torch.tensor([float(yaw_rad)], dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
    )
    return _to_wxyz(math_utils.quat_mul(yaw, _to_isaaclab(quat_ref_wxyz)))


def fingertip_position(ee_pos, ee_quat_wxyz) -> np.ndarray:
    """Where the fingers are, given the flange pose the IK reports."""
    offset = quat_rotate(ee_quat_wxyz, FLANGE_TO_FINGERTIP)
    return np.asarray(ee_pos, dtype=np.float64).ravel()[:3] + offset


def gripper_tilt_deg(quat_wxyz) -> float:
    """Angle between the gripper's approach axis and straight down, in degrees.

    The one number that says whether the orientation hold is actually holding. A tilted
    gripper is not an error anywhere in RoboLab: it grasps at an angle, the wrist camera
    stops looking down, and every displacement statistic still passes.
    """
    axis = quat_rotate(quat_wxyz, _APPROACH_AXIS_LOCAL)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    return float(np.degrees(np.arccos(np.clip(axis @ _STRAIGHT_DOWN, -1.0, 1.0))))


def hold_orientation_rotvec(
    quat_ref_wxyz,
    quat_cur_wxyz,
    gain: float = ORIENT_HOLD_GAIN,
    max_rad: float = ORIENT_HOLD_MAX_RAD,
) -> np.ndarray:
    """World-frame axis-angle that pulls ``quat_cur`` back onto ``quat_ref``.

    This feeds the ``(drx, dry, drz)`` slots of ``DroidRelIKActionCfg``, and **it is what
    makes "the orientation is held" true.** Commanding zeros there does NOT hold the
    orientation: in relative mode a zero rotation delta means "target = the orientation
    you have right now", i.e. an INTEGRATING reference, so any orientation error the IK
    introduces silently becomes the next setpoint and is never corrected.

    And the IK does introduce it. The position command is deliberately overdriven (see
    ``token_controller.STEP_M``), which leaves the DLS solver saturated, and a saturated
    least-squares solution buys position progress with orientation error. Measured
    upstream on RubiksCubeTask with zeros in these slots: the gripper left reset perfectly
    vertical and was 24.5 deg off by the time it released — no error raised, no failed
    episode, no displacement statistic out of range.

    Measured effect of this correction (same token sequence, world frame, gain 1.0):
    worst tilt 24.5 -> 16.9 deg, final 24.5 -> 5.5 deg, at a 2.5% cost in travel. The
    correction alone does not fix the transient — while the arm moves, the saturated IK
    keeps taking orientation back, and this only catches up once it stops. Splitting the
    same total displacement over more control steps is what fixes the transient, so
    ``TokenController`` does both.

    The frame was determined by experiment upstream: the body-frame alternative diverges
    loudly (worst tilt 112 deg), which is the useful property of having tried both.

    Mirrors Show-Harness ``core/sim/robolab_task.hold_orientation_rotvec``; keep them in
    sync, since the point is that served rollouts and generated data apply the identical
    correction.
    """
    math_utils = _math()
    # q_err = ref ⊗ cur⁻¹ : the WORLD-frame rotation taking cur onto ref.
    q_err = math_utils.quat_mul(
        _to_isaaclab(quat_ref_wxyz), math_utils.quat_conjugate(_to_isaaclab(quat_cur_wxyz))
    )
    # axis_angle_from_quat already picks the shortest arc and Taylor-expands near zero.
    rotvec = math_utils.axis_angle_from_quat(q_err).reshape(3).cpu().numpy().astype(np.float64)
    rotvec = rotvec * float(gain)
    mag = float(np.linalg.norm(rotvec))
    return rotvec * (float(max_rad) / mag) if mag > float(max_rad) else rotvec

########################################################
# Prompt helpers
########################################################

# Resolution sent to the model, and the aspect the frame is cropped to first. Both cameras
# render 1280x720; letterboxing 16:9 straight into a square leaves 256x144 of picture inside
# Output resolution: half of the 1280x720 sensor, no padding.
VLM_IMAGE_WIDTH = 640
VLM_IMAGE_HEIGHT = 360


def camera_role(key: str) -> str:
    """``"wrist"`` for the eye-in-hand camera, ``"external"`` for everything else."""
    return "wrist" if "wrist" in key.lower() else "external"


def camera_label(key: str) -> str:
    """Human name for the view, used verbatim in the prompt."""
    if camera_role(key) == "wrist":
        return "WristView"
    pretty = key.replace("_camera", "").replace("_cam", "").replace("_", " ").strip()
    return f"AgentView ({pretty})" if pretty else "AgentView"


def rotate_image(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate an HWC image by *degrees* CCW (nearest multiple of 90)."""
    k = (int(degrees) % 360) // 90
    return np.rot90(image, k=k) if k else image


def preprocess_for_vlm(
    image: np.ndarray,
    width: int = VLM_IMAGE_WIDTH,
    height: int = VLM_IMAGE_HEIGHT,
) -> np.ndarray:
    """Downscale to 640x360 — exactly the frame the model receives.

    No crop, no padding: full FOV preserved.
    """
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 4:  # RGBA -> RGB
        array = array[:, :, :3]
    array = convert_to_uint8(array)

    return np.asarray(
        Image.fromarray(array).resize((width, height), Image.LANCZOS)
    )


def image_to_data_url(
    image: np.ndarray,
    fmt: str = "PNG",
) -> str:
    """Encode the preprocessed frame as a data URL for the chat API."""
    buf = io.BytesIO()
    Image.fromarray(preprocess_for_vlm(image)).save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{'image/png' if fmt == 'PNG' else 'image/jpeg'};base64,{b64}"


def format_state(state: dict, gripper_cmd: float, *, travel_cm: float | None = None) -> str:
    """The text block the VLM reads. Reports the COMMANDED gripper, not the measured one.

    Showing the measured value invites the model to re-issue GRASP while the fingers are
    still travelling — the command is binary and takes ~10 control steps to complete.
    """
    lines = [f"Gripper: {'CLOSED' if gripper_cmd > 0.5 else 'OPEN'} (commanded)"]
    if "tip_pos" in state:
        tip = state["tip_pos"]
        lines.append(f"Fingertip position (base frame, m): x={tip[0]:.3f} y={tip[1]:.3f} z={tip[2]:.3f}")
        # z IS the height above the table: the base frame's z=0 is the table surface
        # (policies/vlm_pinpoint/client.py reports the same number as "cm above the
        # table"). Without saying so, "z=0.47" is a number the model cannot act on —
        # it reads it as "still high" and has no way to know that is 23 descents away.
        lines.append(
            f"Fingertip height ABOVE THE TABLE: {tip[2] * 100:.0f} cm "
            f"(table surface is z=0; grasp height is ~2 cm)")
        if travel_cm:
            lines.append(
                f"One MV_ unit moves about {travel_cm:.0f} cm, so reaching the table "
                f"from here takes about {max(0.0, tip[2] * 100 - 2.0) / travel_cm:.0f} "
                "more MV_DOWN decisions.")
    elif "ee_pos" in state:
        ee = state["ee_pos"]
        lines.append(f"Flange position (base frame, m): x={ee[0]:.3f} y={ee[1]:.3f} z={ee[2]:.3f}")
    if "tilt_deg" in state:
        lines.append(f"Gripper tilt from straight-down: {state['tilt_deg']:.1f} deg")
    return "\n".join(lines)


########################################################
# Motion calibration
########################################################

# Per-role camera rotation at extraction time (degrees, CCW).
# Show-Harness: Robotiq 2F-85 wrist = 180, Panda hand wrist = 270.
CAMERA_ROTATIONS: dict[str, int] = {"wrist": 180}

# Commanded translation per decision, in metres. NOT the achieved displacement — see the
# module docstring. 0.072 commanded == ~2.0 cm physical.
STEP_M = 0.072

# Control steps one decision is split over. The total travel is set by STEP_M and is
# insensitive to this (the ~28% factor is a fraction of the total), so raising it shrinks
# the per-step command and with it the orientation transient, at no cost in distance.
CONTROL_STEPS = 8

# Commanded yaw per rotation decision, radians (~8.6 deg).
YAW_STEP_RAD = 0.15

# Per-control-step safety cap on the IK target step, in metres. Only a guard so a
# mis-set config cannot ask the solver for a lunge it will diverge on.
MAX_DELTA_M = 0.05

# The Robotiq 2F-85 is driven by a BINARY joint command, so a GRASP has to be HELD for the
# fingers to travel and load up on the object. One step is not enough — this is what makes
# the difference between a grasp and a twitch.
GRIPPER_HOLD_STEPS = 10

# RoboLab's BinaryJointPositionZeroToOneAction rule is ``action > 0.5 -> CLOSE``.
# Flipping these makes every GRASP release and every RELEASE grab.
GRIPPER_OPEN = 0.0
GRIPPER_CLOSE = 1.0

# Robotiq 2F-85 opening, derived from the finger_joint angle the proprio obs reports
# normalised: 0 rad = 85 mm open, pi/4 = fully closed. A CLOSED gripper reading a near-zero
# width has its fingers touching each other, i.e. it is holding NOTHING.
GRIPPER_OPEN_WIDTH_M = 0.085
# Below this a closed gripper counts as empty and is reopened. 5 mm, not the real rigs'
# 1 mm, per Show-Harness's ``empty_width_m`` for RoboLab ("Real-robot default is 0.001,
# which is too tight here") — and RoboLab's own cfg says why: ``gripper_pos`` carries
# ``GaussianNoiseCfg(std=0.05)`` (droid.py), i.e. 4.25 mm of width noise, so a 1 mm window
# is inside the noise floor. At 5 mm an empty grasp (true width 0) is caught on ~88% of
# decision boundaries, while a loaded gripper holding even a 40 mm object sits ~9 sigma
# clear of the threshold and is never touched.
EMPTY_GRASP_WIDTH_M = 0.005


def gripper_width_m(gripper_measured: float) -> float:
    """Normalised ``finger_joint`` (0 = open .. 1 = closed) -> finger opening in metres."""
    return GRIPPER_OPEN_WIDTH_M * (1.0 - float(np.clip(gripper_measured, 0.0, 1.0)))

# Base-frame unit vector per move unit. RoboLab's Franka sits at the origin with +X away
# from the base, robot-LEFT = +Y (ROS: X fwd, Y left, Z up; OverShoulderLeftCam sits at
# +Y), +Z up. CONFIRM with ``probe_axes.py`` before trusting a rollout: a sign error
# here is invisible in the logs and looks like a bad model.
MOVE_VECTORS: dict[str, tuple[float, float, float]] = {
    "MV_FWD": (1.0, 0.0, 0.0),
    "MV_BACK": (-1.0, 0.0, 0.0),
    "MV_LEFT": (0.0, 1.0, 0.0),
    "MV_RIGHT": (0.0, -1.0, 0.0),
    "MV_UP": (0.0, 0.0, 1.0),
    "MV_DOWN": (0.0, 0.0, -1.0),
}

# Yaw sign per rotation unit, calibrated so the token matches the turn the VLM SEES in the
# wrist view: the eye-in-hand camera looks down, so its apparent sense is mirrored versus
# the base +Z right-hand rule.
YAW_SIGNS = {"ROTATE_CW": 1.0, "ROTATE_CCW": -1.0}


@dataclass(frozen=True)
class Decision:
    """One VLM decision, expanded into a plan of control steps.

    The action itself is deliberately NOT stored: the rotation slots depend on the pose at
    the moment each control step is issued, so the client materialises the action per step
    from this plan plus the live observation.
    """

    unit: str
    steps: int
    move_m: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw_rad: float = 0.0
    gripper: float | None = None  # None -> keep the latched command
    terminal: bool = False
    note: str = ""
    # The model's own WRIST: YES/NO judgment, or None when it emitted no marker. Steers
    # nothing — it is logged so a run can be diagnosed by which view the model believed it
    # was steering by. See :meth:`TokenController.parse_wrist_marker`.
    wrist: bool | None = None

    @property
    def ends_episode_planning(self) -> bool:
        return self.terminal


class TokenController:
    """Grounds the discrete action units into metric motion, and builds the prompt."""

    TOKEN_VOCAB = TOKEN_VOCAB

    def __init__(
        self,
        *,
        ik_scale: float = 0.5,
        step_m: float = STEP_M,
        control_steps: int = CONTROL_STEPS,
        yaw_step_rad: float = YAW_STEP_RAD,
        max_delta_m: float = MAX_DELTA_M,
        gripper_hold_steps: int = GRIPPER_HOLD_STEPS,
    ) -> None:
        if ik_scale <= 0.0:
            raise ValueError(f"ik_scale must be > 0, got {ik_scale!r}")
        self.ik_scale = float(ik_scale)
        self.step_m = float(step_m)
        self.control_steps = max(1, int(control_steps))
        self.yaw_step_rad = float(yaw_step_rad)
        self.max_delta_m = float(max_delta_m)
        self.gripper_hold_steps = max(1, int(gripper_hold_steps))

    # ------------------------------------------------------------------
    # Output side: unit -> plan -> per-control-step action
    # ------------------------------------------------------------------

    @property
    def travel_cm(self) -> float:
        """Physical centimetres one decision actually buys.

        ``step_m`` is the COMMANDED total; relative IK delivers ~28% of it (see this
        module's docstring). The model is told this number so that "45 cm above the
        table" converts into a countable number of decisions instead of a vague
        "still high".
        """
        return self.step_m * REL_IK_ACHIEVED_FRACTION * 100.0

    @property
    def per_step_m(self) -> float:
        """Metres commanded per control step so one decision totals ``step_m``."""
        return self.step_m / self.control_steps

    def decide(self, unit: str, *, note: str = "") -> Decision:
        """Expand one action unit into a plan. Raises ``ValueError`` on an unknown unit."""
        unit = unit.strip().upper()
        if unit in MOVE_VECTORS:
            move = np.asarray(MOVE_VECTORS[unit], dtype=np.float64) * self.per_step_m
            return Decision(unit=unit, steps=self.control_steps, move_m=move, note=note)
        if unit in YAW_SIGNS:
            return Decision(
                unit=unit,
                steps=self.control_steps,
                yaw_rad=YAW_SIGNS[unit] * self.yaw_step_rad,
                note=note,
            )
        if unit == GRASP_UNIT:
            return Decision(unit=unit, steps=self.gripper_hold_steps, gripper=GRIPPER_CLOSE, note=note)
        if unit == RELEASE_UNIT:
            return Decision(unit=unit, steps=self.gripper_hold_steps, gripper=GRIPPER_OPEN, note=note)
        if unit == STILL_UNIT:
            return Decision(unit=unit, steps=1, note=note)
        if unit in TERMINAL_UNITS:
            return Decision(unit=unit, steps=1, terminal=True, note=note)
        raise ValueError(f"Unknown action unit: {unit!r}; expected one of {list(TOKEN_VOCAB)}")

    def action_for_step(
        self,
        decision: Decision,
        *,
        gripper_cmd: float,
        rot_correction: np.ndarray,
    ) -> np.ndarray:
        """Build the 7-D action for ONE control step of ``decision``.

        ``rot_correction`` is the world-frame axis-angle from
        :func:`kinematics.hold_orientation_rotvec`, computed by the caller against the
        CURRENT pose — that freshness is the whole point, so it is a parameter rather than
        something cached on the decision.
        """
        delta = np.clip(decision.move_m, -self.max_delta_m, self.max_delta_m) / self.ik_scale
        rot = np.asarray(rot_correction, dtype=np.float64) / self.ik_scale
        return np.array([*delta, *rot, float(gripper_cmd)], dtype=np.float64)

    def rotate_reference(self, quat_ref: np.ndarray, decision: Decision) -> np.ndarray:
        """Advance the held-orientation reference by a rotation decision's yaw.

        Rotation is expressed by MOVING THE REFERENCE, not by writing a raw delta into the
        rotation slots. The orientation hold then drives the arm onto the new reference and
        keeps correcting roll and pitch on the way, so a yaw command cannot quietly become
        a tilt. World-frame pre-multiply, matching the ``ref ⊗ cur⁻¹`` error convention in
        :func:`kinematics.hold_orientation_rotvec`.
        """
        return yaw_reference(quat_ref, decision.yaw_rad)

    # ------------------------------------------------------------------
    # Input side: observation -> prompt
    # ------------------------------------------------------------------

    def system_prompt(self, view_labels: list[str]) -> str:
        """The direction rules, grounded against the views that are actually attached.

        Without this grounding the model has no way to know that MV_LEFT means base +Y
        rather than "left in whichever image it happened to look at" — the vocabulary is
        defined in the base frame, and the model only ever sees pixels.
        """
        wrist = next((label for label in view_labels if label == "WristView"), None)
        agent = next((label for label in view_labels if label != "WristView"), None)
        primary = agent or wrist or "the image"

        lines = [
            "You control a Franka arm with a parallel gripper. Each reply commands ONE "
            "action unit, which moves the gripper about 2 cm or rotates it about 9 degrees.",
            "",
            f"VIEWS, in the order they are attached: {', '.join(view_labels) or 'none'}.",
        ]
        if agent:
            lines.append(
                f"- {agent}: a fixed third-person view. Your primary guide for approaching "
                "a target that the wrist camera cannot see yet."
            )
        if wrist:
            lines.append(
                f"- {wrist}: mounted on the gripper, with the black fingertips entering "
                "from the top of the frame. Table objects appear NEAR THE TOP too, level "
                "with the fingertips or just past them; the lower half of the frame is "
                "usually bare table. Your guide for left/right once the target is in it. "
                "It cannot see straight down between the fingers, so it cannot tell you "
                "how high you are — read that off the state block."
            )
        lines += [
            "",
            "DIRECTION. Judge the offset between the gripper and the target, and command the "
            "single axis with the LARGEST deviation.",
        ]
        if wrist:
            lines += [
                f"When the target is visible in {wrist}, use it for left/right and grasp:",
                "- target left of the fingers -> MV_LEFT ; right of the fingers -> MV_RIGHT",
                "- target centred LEFT/RIGHT between the fingers -> MV_DOWN (this view "
                "settles left/right only; it cannot see the gap under the fingers, so it "
                "can never tell you that you are low enough)",
                f"Do NOT judge forward/backward from up/down position in {wrist}: this "
                "camera looks back along the arm, so +X/-X shows as size change, not "
                "vertical motion. For MV_FWD/MV_BACK judge from "
                f"{agent if agent else 'the other view'} (target higher in that frame -> "
                "MV_FWD ; lower -> MV_BACK), and from target size here "
                "(small/far -> MV_FWD to approach ; overflowing the frame -> MV_BACK).",
            ]
        if agent:
            lines += [
                f"Otherwise use {agent}, judging the target against the gripper:",
                "- MV_FWD moves the gripper toward the image TOP-LEFT in this view, MV_BACK "
                "toward the bottom-right ; MV_LEFT goes left-DOWNWARD, MV_RIGHT right-UPWARD.",
                "- Left/right alone cannot separate MV_FWD from MV_LEFT here (both go left) "
                "— use the vertical direction: target higher in the frame -> MV_FWD ; "
                "lower -> MV_BACK.",
                f"If the two views disagree on left/right while the target is visible in "
                f"{wrist if wrist else 'the wrist view'}, trust the wrist view — it rides "
                "the gripper, while the fixed view distorts lateral offset with perspective.",
            ]
        lines += [
            "MV_UP when you need to lift the grasped object, when you are too low to reach "
            "over an obstacle, or when retreating after RELEASE.",
            "",
            "DEPTH AND HEIGHT. The agent view can lie about distance: looking close there "
            "may just be occlusion or a high viewpoint. The fingertip height in the state "
            "block is the number to trust.",
            # The wrist camera sits behind and above the fingers, so a target you are
            # lined up with is NECESSARILY at the top of the frame, partly occluded. The
            # rule this replaces ordered MV_BACK on exactly that condition, which turned
            # the success state into a retreat trigger: measured 11 of 20 decisions were
            # MV_BACK, and the arm orbited the banana for a whole episode without
            # descending (z went 0.47 -> 0.43 in 160 control steps).
            "- The wrist camera is bolted to the flange and looks out ALONG the jaw "
            "axis, with the fingertips 15 cm ahead of it, so a target you are "
            "correctly lined up with SITS AT THE TOP OF THE WRIST VIEW, partly behind the "
            "fingertips. That is what being aligned LOOKS like, not a lost target: answer "
            "it with MV_DOWN. Never answer it with MV_BACK.",
            "- MV_BACK is only for a real overshoot: the target has grown large and fills "
            "much of the wrist view, or the agent view shows the gripper has travelled "
            "past it. Backing off merely because the fingers cover the target undoes the "
            "approach you just made and you will arrive at the same picture again.",
            # A hard "do not refine above 10 cm" gate was tried here and is worse: the
            # model saw the banana sitting right of the fingers for eight consecutive
            # decisions, was forbidden to act on it, and the offset grew until the
            # target left the wrist view entirely at 19 cm. Lateral error does not
            # shrink on its own, and every centimetre of descent makes it harder to see.
            "- Descent is the DEFAULT, not a phase. Descend on every decision that has no "
            "clearly visible lateral offset to fix; when you can see the target sitting "
            "off the finger centre, correct that first, then resume descending. Do not "
            "bank a visible offset for later, and do not stop descending to re-examine a "
            "target you can already see.",
            "- A run of MV_DOWN from the start pose is expected — the state block says how "
            "many are left. It is a stuck loop only if the height stops falling.",
            "- From high above the table a far-looking target is usually an x overshoot: "
            "MV_DOWN first rather than repeating MV_FWD.",
            "- After a grasp, MV_UP clear of the table before any x/y move.",
            "",
            "GRIPPER.",
            "- GRASP only when the grasp point is between the centre of the two fingers "
            "AND the fingertips are within about 3 cm of the table. Height is the binding "
            "condition: a picture that looks aligned from 20 cm up is still 20 cm up. A "
            "GRASP is held long enough for the fingers to close, so do not repeat it "
            "while it is already CLOSED.",
            "- RELEASE only when the held object is above its destination and lowered onto it.",
            "",
            "ROTATE_CW / ROTATE_CCW turn the gripper as seen in the wrist view. Use them only "
            "when the grasp needs the fingers aligned across the object's short axis.",
            "",
            f"{DONE_UNIT} only when the goal state is already visible in the images. "
            f"{GIVE_UP_UNIT} if the task has become impossible. Both stop the run, so do not "
            "use them speculatively.",
            "",
            f"Units: {', '.join(TOKEN_VOCAB)}",
            "",
            # Thinking happens in the model's own reasoning channel when the server
            # exposes one (the backend turns it on); models served without a separate
            # channel (e.g. Cosmos3-Omni on vLLM, which returns no reasoning_content)
            # must think in the visible reply instead. One short sentence is allowed
            # before the decision lines; the parser takes the LAST markers, so prose
            # cannot hijack the decision.
            "Do not answer a unit with its OPPOSITE on the next decision (MV_FWD then "
            "MV_BACK, MV_LEFT then MV_RIGHT): that is an oscillation and it cancels your "
            "own progress. Repeating the SAME unit many times is fine and often correct — "
            "a long descent is repetition, not oscillation.",
            "",
            # WRIST: YES/NO is Show-Harness's shared wrist-visibility marker
            # (core/prompting/wrist_marker.py). Making the A/B branch an explicit,
            # parseable statement rather than an implicit one both improves the choice and
            # gives the log a signal for which view the model thought it was steering by.
            "Work out the offset in whichever view the DIRECTION rules above put you in "
            "charge of for this decision. You may think in ONE short sentence first, then "
            "give exactly these two final lines and nothing else:",
            "WRIST: <YES if the target is in the wrist view, else NO>",
            "ACTION: <UNIT>",
        ]
        return "\n".join(lines)

    def build_prompt(
        self,
        obs: dict,
        *,
        gripper_cmd: float,
        recent: list[str] | None = None,
    ) -> list[dict]:
        """OpenAI chat messages: stable text first, fresh state, images last.

        Layout is prefix-cache friendly: the vLLM server reuses KV for the
        shared token prefix, so stable content (task, growing history) comes
        before per-step content (live state, fresh frames). History is
        append-only within an episode for the same reason — a sliding window
        would shift every token and defeat the cache. Mirrors Show-Harness's
        imagelabel-free contract otherwise: agentview, then any extra views
        (wrist last). View identity comes from the fixed order plus the
        system prompt's view descriptions (wrist = the one with fingertips).
        """
        views = obs.get("views") or []
        labels = [label for label, _ in views]

        history = ", ".join(recent) if recent else "none"
        content: list[dict] = [
            {"type": "text",
             "text": (
                 f"Task: {obs.get('instruction', '')}\n\n"
                 f"Your last units with measured outcome (oldest first): {history}\n\n"
                 f"Robot state:\n"
                 f"{format_state(obs.get('state', {}), gripper_cmd, travel_cm=self.travel_cm)}\n\n"
                 "Reply with at most one thinking sentence, then exactly two lines: WRIST: <YES|NO> then ACTION: <UNIT>"
             )},
        ]
        for _, image in views:
            content.append(
                {"type": "image_url", "image_url": {"url": image_to_data_url(image)}}
            )

        return [
            {"role": "system", "content": self.system_prompt(labels)},
            {"role": "user", "content": content},
        ]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def parse_wrist_marker(content: str) -> bool | None:
        """The model's own answer to "is the target in the wrist view?", or ``None``.

        Mirrors Show-Harness's ``core/prompting/wrist_marker.py``: take the LAST marker (the
        model's final word) and treat its absence as unknown rather than as an error — the
        marker steers nothing, so a reply without one is still a usable decision.
        """
        matches = re.findall(r"WRIST\s*[:=]\s*(YES|NO)\b", content or "", re.IGNORECASE)
        return matches[-1].upper() == "YES" if matches else None

    def parse_unit(self, content: str, reasoning: str = "") -> str:
        """Recover the commanded unit from the reply.

        Order matters and is the fix for a real failure: scanning the vocabulary and
        returning the first token that appears anywhere makes the answer depend on how long
        the token strings are, not on what the model decided — "MV_UP would overshoot, so
        MV_FWD" resolves to whichever of the two is checked first. So:

        1. the explicit ``ACTION:`` contract, last occurrence (a model may restate it);
        2. failing that, the LAST vocabulary mention in the visible reply;
        3. reasoning text only as a last resort, since it is where rejected options live.
        """
        for text in (content or "", reasoning or ""):
            # Without --reasoning-parser the whole <think>...</think> trace stays inside
            # `content`, and it is full of options the model went on to reject. The answer
            # follows the trace, so drop it before reading anything out.
            text = re.sub(r"<think>.*?</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"^.*</think>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            # Both rules above need the CLOSING tag. A trace cut off at ``max_tokens`` has
            # an opening <think> and no </think>, so neither fires and the whole trace —
            # rejected options and all — would fall through to the last-mention heuristic
            # below and yield a confident-looking random unit. There is no answer in such a
            # reply; say so rather than inventing one.
            if re.search(r"<think>", text, re.IGNORECASE):
                continue
            if not text.strip():
                continue
            contract = re.findall(r"ACTION\s*:\s*\*{0,2}([A-Z_]+)", text.upper())
            for candidate in reversed(contract):
                if candidate in TOKEN_VOCAB:
                    return candidate

            upper = text.upper()
            hits = [
                (match.start(), unit)
                for unit in TOKEN_VOCAB
                # \b would split on the underscore; assert a non-word char instead.
                for match in re.finditer(rf"(?<![A-Z_]){re.escape(unit)}(?![A-Z_])", upper)
            ]
            if hits:
                return max(hits)[1]

        raise ValueError(
            f"No action unit found. content={(content or '')[:200]!r} "
            f"reasoning={(reasoning or '')[:200]!r}"
        )

    def decision_from_message(self, message: dict) -> Decision:
        """OpenAI-style assistant message -> :class:`Decision`."""
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        unit = self.parse_unit(content, reasoning)
        note = (content or reasoning or "").strip().replace("\n", " ")[:200]
        # ``Decision`` is frozen, so the marker is attached with ``replace`` rather than
        # threaded through every ``decide`` return.
        return replace(self.decide(unit, note=note), wrist=self.parse_wrist_marker(content))
