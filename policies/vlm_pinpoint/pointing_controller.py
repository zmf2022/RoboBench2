# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal geometric controller: image point + depth -> rel-IK motion.

Pipeline per decision:
  1. map the 0-1000 point from the VLM input frame back to raw sensor pixels
     (the VLM frame is a uniform downscale of the whole sensor frame, so this
     is a pure scale plus the undo of the 180° wrist rotation)
  2. unproject (u, v, depth) with the pinhole intrinsics to a 3D target in
     the camera frame, then to the robot-root frame with the live wrist
     extrinsics (EE pose composed with the mount offset)
  3. expand the phase into a WAYPOINT PRIMITIVE the caller servos against live
     proprioception -- the model is asked once per primitive, not once per N
     control steps (see :meth:`PointingController.plan`)

Phase machine: APPROACH/ALIGN stage above the target, GRASP descends-closes-
lifts, LIFT goes +z, PLACE descends and releases, DONE terminates.

Requires depth + intrinsics + camera pose in the harness observation. Without
depth the controller raises :class:`MissingDepthError` telling exactly which
camera flag to enable, instead of servoing blind.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from policies.vlm_pinpoint.connector import quat_rotate

logger = logging.getLogger(__name__)

# Wrist PinholeCameraCfg in robolab/robots/droid.py::_WRIST_CAM (1280x720).
_FOCAL_LENGTH = 2.8
_APERTURE_W = 5.376
_APERTURE_H = 3.024
_SENSOR_W = 1280
_SENSOR_H = 720
# Mount offset of wrist_cam on the Robotiq base_link (opengl WXYZ in cfg).
_MOUNT_POS = np.array([0.011, -0.031, -0.074], dtype=float)
_MOUNT_QUAT_WXYZ = np.array([-0.420, 0.570, 0.576, -0.409], dtype=float)

# Per-decision travel cap (m) and grasp hold, mirroring the token pipeline.
STEP_M = 0.072
GRIPPER_HOLD_STEPS = 10
# How close (m) the fingertip must be before ALIGN becomes GRASP/DOWN-to-grasp.
GRASP_RADIUS_M = 0.02
# Horizontal radius (m) inside which the fingers count as STAGED OVER the target,
# so the controller descends and grasps on its own instead of waiting for the model
# to say GRASP. Sized against the OBJECT, not just the pointing noise: at 0.06 the
# arm committed with up to 6 cm of horizontal error on a 4 cm banana, closed on air
# and lifted in a loop. 0.03 still latches under real convergence (measured
# 0.011-0.026 on good approaches) while sending noise spikes back to servoing.
STAGED_RADIUS_M = 0.03

# --- grasp yaw (see object_long_axis_rad) ---------------------------------
# The Robotiq 2F-85 opens 85 mm. A banana is ~190 mm long. If the jaws close
# along the object's LONG axis the grasp cannot succeed -- not "less reliably",
# it physically does not fit. So the wrist yaw has to be aimed, and aiming it
# from the depth map is deterministic where asking a small VLM for an angle is
# not (the model has no rotation channel at all -- see pointing_backend).
# Half-width of the window sampled around the pointed pixel, in METRES at the
# target's own depth, so it is independent of sensor resolution. Wide enough to
# cover a banana end to end, tight enough to exclude the bowl beside it.
GRASP_AXIS_RADIUS_M = 0.09
# A point counts as OBJECT rather than SUPPORT if it stands this far above the
# local surface. The object rests ON the table, so its depth is nearly the
# table's -- separating them by depth band does not work; by height does.
GRASP_AXIS_MIN_HEIGHT_M = 0.008
# Below this elongation the axis is noise (a ball has no long axis), and the
# held orientation is kept instead.
GRASP_AXIS_MIN_ELONGATION = 1.3
GRASP_AXIS_MIN_POINTS = 30
# The hand is directly over the target when the grasp is planned, so it lands in
# the sampling window and its own point cloud gets segmented as "the object" --
# measured at 15.4 cm of "banana" on a 4 cm banana. Points within this distance
# of the flange->fingertip segment are the robot, not the scene. Proprioception
# only: no scene knowledge, so this survives off-sim.
GRIPPER_EXCLUSION_M = 0.05
# ...and the forearm reaches into the window too, which the hand mask does not
# cover (measured: 12.4 cm of "object" left after masking the hand alone). The
# pointed pixel is ON the object, so the object cannot extend far above it --
# that anchors the ceiling to the one thing in the window we actually trust.
OBJECT_TOP_MARGIN_M = 0.03
# Fingertip height above the SUPPORT at closure. A top-down pinch has to get the
# pads around the widest part of the object; closing higher than that squeezes a
# round object out of the jaws instead of gripping it (measured: closing at 65%
# of a banana's height grasps air). Small, because the fingers must reach past
# the equator without ramming the table.
GRASP_CLEARANCE_M = 0.008
# Robotiq 2F-85 inner pad: the gripping face runs from the FINGERTIP back up
# towards the flange. Everything in this file tracks the fingertip, so the pad
# sits ENTIRELY ABOVE whatever height is commanded -- put the tip at the object's
# equator and the pads straddle empty air above it. Half a pad is therefore the
# offset between "where the pad should be centred" and "where to send the tip".
PAD_LENGTH_M = 0.037
# ...and the tip cannot go all the way down: the 2F-85's fingers swing inward AND
# dip as they close, so a tip too near the support jams the arc on the table with
# the jaws still open. Measured floor -- 8 mm jammed, 12 mm worked, 10 mm is the
# compromise for short objects (banana): watch for table jam on regression.
FINGERTIP_FLOOR_M = 0.010
# Hard floor on the descent, in case the support estimate is bad: never drive
# more than this below the surface the model actually pointed at.
MAX_BITE_M = 0.05
# Progress (m) a leg must make to count as still moving, and how many steps of
# no progress mean it has stalled. A free descent converges geometrically -- the
# servo commands the remaining distance, so ~8 steps cover a 10 cm leg -- which
# makes a stall obvious long before the step budget runs out.
STALL_PROGRESS_M = 0.001
STALL_STEPS = 12
# A closing dwell ends when the FINGERS have stopped moving, not when a step
# counter runs out: lifting while the jaws are still travelling drags the object
# out of them. ``gripper_pos`` is the measured finger joint, so "settled" is
# observable -- consecutive steps whose width changes by less than the epsilon.
# hold_steps stays the floor (the binary joint needs the command held to travel
# at all) and the ceiling stops a jammed or empty close from pinning the episode.
# How the dwell decides the jaws have finished travelling.
#
# NOT an absolute speed threshold. Three guesses at one were all wrong in the
# same direction, because this gripper is far slower than any of them: measured
# on this rig, 85.0 mm -> 72.9 mm over 74 control steps, i.e. 0.16 mm per step
# (the finger_joint actuator runs on the USD's default gains --
# DroidCfg.actuators["gripper"] sets stiffness=None, damping=None). Every
# absolute epsilon called a steadily closing gripper "stopped", the lift began,
# and the banana slid out.
#
# So the test is RELATIVE: the jaws have stopped when a window buys much less
# travel than the fastest window did. That calibrates itself to whatever the
# hardware actually does, at any speed.
GRIPPER_SETTLE_STEPS = 6
GRIPPER_STOPPED_FRACTION = 0.25
# Travel the jaws must make before "not moving" is allowed to mean "finished".
# Without it the two are indistinguishable: for the first steps after the command
# the joint has not started, so the width sits constant at fully open and every
# stillness test says "settled".
GRIPPER_MIN_TRAVEL_M = 0.010
# The dwell must end ON the object, never short of it. A rollout that stopped a
# margin WIDER than the measured width (50.8 mm on a 42.8 mm banana) had not
# touched it at all -- and since the close command is still held through the
# lift, the remaining travel happened on the way up. That is exactly the "closing
# while lifting" slip. So the jaw target goes PAST the measurement: the pads are
# compliant and the 5th-95th percentile width runs a little wide.
GRIPPER_SQUEEZE_M = 0.002
# ...and a stall only counts as contact once the jaws are near that width. Further
# out than this the jaws cannot be on the object, so stillness means the actuator
# has not got going yet, not that the grasp is closed.
GRIPPER_CONTACT_SLACK_M = 0.015
# Ceiling, sized against the measured rate: reaching a banana from fully open
# takes ~300 steps at 0.16 mm/step. Cutting a close short loses the whole episode,
# while waiting too long only costs steps, so this errs long.
GRIPPER_SETTLE_MAX_STEPS = 350

# --- primitive geometry (see PointingController.plan) ----------------------
# How high above the target the gripper stages before descending. A top-down
# pick needs the fingers to arrive from above: servoing straight down the
# line-of-sight drives them into the table, or into the side of the object.
PREGRASP_HEIGHT_M = 0.10
# The depth sample is the VISIBLE SURFACE of the object, but the fingers have
# to straddle it, so the grasp waypoint sits this far below that surface.
GRASP_DEPTH_M = 0.015
LIFT_HEIGHT_M = 0.12
# Arrival tolerance and the per-control-step servo speed. The speed matches the
# token pipeline's effective rate (STEP_M spread over CONTROL_STEPS).
WAYPOINT_TOL_M = 0.015
SERVO_M_PER_STEP = STEP_M / 8.0
# Relative IK achieves a roughly constant ~28% of whatever is commanded -- a
# measured property of this harness, not an estimate (see the vlm_agent.
# token_controller module docstring). Every metre figure here is COMMANDED, so
# any step budget has to be sized against the ACHIEVED rate:
# SERVO_M_PER_STEP is 9 mm commanded, i.e. ~2.5 mm of real travel per step.
IK_ACHIEVED_FRACTION = 0.28
# Absolute guard so an unreachable waypoint cannot pin the episode. This is a
# ceiling, NOT the per-waypoint budget: a flat cap is what made long legs fail
# while short ones succeeded (a 0.25 m lateral move needs ~100 steps, a 0.10 m
# descent only ~40), which reads exactly like an arm that can descend but cannot
# travel. Use :func:`waypoint_budget` instead.
MAX_STEPS_PER_WAYPOINT = 250


def waypoint_budget(dist_m: float) -> int:
    """Control steps to allow a leg of ``dist_m``, at the achieved servo rate."""
    steps = dist_m / (SERVO_M_PER_STEP * IK_ACHIEVED_FRACTION)
    return int(min(MAX_STEPS_PER_WAYPOINT, max(20.0, steps * 1.5)))


@dataclass
class Waypoint:
    """One leg of a primitive: servo the fingertip to ``pos``, or dwell in place.

    ``pos is None`` means "stay where you are for ``hold_steps``" — the Robotiq
    2F-85 takes a binary joint command, so a grasp only travels if the closed
    command is HELD (see GRIPPER_HOLD_STEPS).
    """

    pos: np.ndarray | None
    gripper: float
    hold_steps: int = 0
    tol_m: float = WAYPOINT_TOL_M
    # Jaw opening (m) that means "closed onto the object", or None to fall back
    # to the stop-detection heuristics. Measured off the object cloud.
    close_to_m: float | None = None
    # Desired world-frame angle of the JAW AXIS in the XY plane, radians, or
    # None to keep the orientation latched at episode start. The client turns
    # this into a quaternion (see yaw_aligned_quat) because only it owns the
    # latch; the controller stays free of episode state.
    yaw_rad: float | None = None
    # What a stalled or over-budget leg means. "replan" (the default) asks the
    # model again from wherever the arm ended up. "advance" means CONTACT IS THE
    # POINT: on a grasp descent, fingers that stop making progress have ARRIVED
    # -- they are against the object or the table -- and the right response is to
    # close them, not to abandon the grasp. Treating that as failure is what left
    # the hand pressed on the banana for a whole step budget with the jaws open,
    # then lifting away empty.
    on_stall: str = "replan"


class MissingDepthError(RuntimeError):
    """Raised when the observation has no depth/intrinsics for unprojection."""


class NoValidDepthError(MissingDepthError):
    """The pointed ray hits nothing (skybox/inf): transient, safe to hold."""


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = (float(v) for v in np.asarray(a, dtype=float).ravel()[:4])
    bw, bx, by, bz = (float(v) for v in np.asarray(b, dtype=float).ravel()[:4])
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def wrist_extrinsics(ee_pos, ee_quat_wxyz):
    """Wrist camera pose in robot-root frame from the live EE pose."""
    ee_pos = np.asarray(ee_pos, dtype=float).ravel()[:3]
    ee_quat = np.asarray(ee_quat_wxyz, dtype=float).ravel()[:4]
    cam_pos = ee_pos + quat_rotate(ee_quat, _MOUNT_POS)
    cam_quat = _quat_mul_wxyz(ee_quat, _MOUNT_QUAT_WXYZ)
    return cam_pos, cam_quat


def intrinsics():
    """Pinhole (fx, fy, cx, cy) for the 1280x720 wrist sensor."""
    fx = _FOCAL_LENGTH * _SENSOR_W / _APERTURE_W
    fy = _FOCAL_LENGTH * _SENSOR_H / _APERTURE_H
    return fx, fy, _SENSOR_W / 2.0, _SENSOR_H / 2.0


def frame_point_to_sensor(x1000: int, y1000: int,
                          sensor_w: float = _SENSOR_W,
                          sensor_h: float = _SENSOR_H,
                          rot_deg: int = 180) -> tuple[float, float]:
    """0-1000 point in the VLM frame -> raw sensor pixels.

    ``preprocess_for_vlm`` is a pure uniform rescale of the full frame (no crop,
    no pad), so the VLM frame size cancels out entirely and only the view's
    ``CAMERA_ROTATIONS`` entry has to be undone. The wrist view is rotated 180
    (flip both axes); a fixed external view is not rotated at all, and flipping
    it anyway would point the geometry at the opposite corner of the table.
    """
    u = (x1000 / 1000.0) * sensor_w
    v = (y1000 / 1000.0) * sensor_h
    if int(rot_deg) % 360 == 180:
        return sensor_w - u, sensor_h - v
    if int(rot_deg) % 360 != 0:
        raise ValueError(f"unsupported view rotation {rot_deg} (expected 0 or 180)")
    return u, v


def sensor_point_to_frame(u: float, v: float, sensor_w: float, sensor_h: float,
                          rot_deg: int = 180) -> tuple[float, float]:
    """Inverse of :func:`frame_point_to_sensor`, in 0-1 fractions of the frame."""
    if int(rot_deg) % 360 == 180:
        u, v = sensor_w - u, sensor_h - v
    return u / sensor_w, v / sensor_h


def project(point_w, K=None, cam_pos=None, cam_quat=None,
            ee_pos=None, ee_quat=None) -> tuple[float, float] | None:
    """Robot-root 3D point -> sensor pixel, or ``None`` if behind the camera.

    The forward of :func:`unproject`, and it exists for one reason: in a fixed
    third-person view the model cannot tell which of the things on screen is the
    hand it is controlling. Projecting the fingertip back in and drawing it is
    what turns "point at the banana" into a spatial relationship the model can
    actually reason about (the marking trick from MOKA / Set-of-Mark).
    """
    if K is not None:
        K = np.asarray(K, dtype=float).reshape(3, 3)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    else:
        fx, fy, cx, cy = intrinsics()
    if cam_pos is not None and cam_quat is not None:
        cp = np.asarray(cam_pos, dtype=float).ravel()[:3]
        cq = np.asarray(cam_quat, dtype=float).ravel()[:4]
    elif ee_pos is not None and ee_quat is not None:
        cp, cq = wrist_extrinsics(ee_pos, ee_quat)
    else:
        return None
    cq_inv = np.array([cq[0], -cq[1], -cq[2], -cq[3]])
    p_gl = quat_rotate(cq_inv, np.asarray(point_w, dtype=float).ravel()[:3] - cp)
    x, y, z = p_gl[0], -p_gl[1], -p_gl[2]  # OpenGL -> pinhole
    if z <= 1e-6:
        return None
    return fx * x / z + cx, fy * y / z + cy


def unproject(u: float, v: float, depth: float, K=None, cam_pos=None, cam_quat=None,
              ee_pos=None, ee_quat=None) -> np.ndarray:
    """Pixel + depth -> 3D target in robot-root frame.

    ``depth`` is distance along the optical axis, not ray length: IsaacLab's
    ``"depth"`` annotator is ``distance_to_image_plane`` (camera.py), which is
    exactly what the pinhole model below wants.

    Both the ``wrist_cam_quat`` observation (``camera_quat`` returns
    ``quat_w_opengl``) and the mount ``rot`` in ``_WRIST_CAM.offset`` use the
    **OpenGL** camera convention: +X right, +Y **up**, **-Z** forward. Pixel
    coordinates run the other way on two of those axes, so the pinhole point has
    to be flipped into the camera's own frame before it is rotated. Getting this
    wrong mirrors the target through the camera — it lands behind and above the
    lens, and the arm servos away from whatever the model pointed at.
    """
    if K is not None:
        K = np.asarray(K, dtype=float).reshape(3, 3)
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    else:
        fx, fy, cx, cy = intrinsics()
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    p_cam = np.array([x, -y, -depth])  # pinhole (y down, z forward) -> OpenGL
    if cam_pos is not None and cam_quat is not None:
        cp = np.asarray(cam_pos, dtype=float).ravel()[:3]
        cq = np.asarray(cam_quat, dtype=float).ravel()[:4]
    elif ee_pos is not None and ee_quat is not None:
        cp, cq = wrist_extrinsics(ee_pos, ee_quat)
    else:
        raise MissingDepthError(
            "unprojection needs camera extrinsics: pass K/cam_pos/cam_quat or "
            "ee_pos/ee_quat in the harness observation"
        )
    # cam_pos is env-local while ee_pos is robot-root; for Franka-family robots
    # the root sits at the env origin with identity rotation so the two agree
    # numerically (docs/frames.md). A floor-standing base would need the root
    # pose to bridge them.
    return cp + quat_rotate(cq, p_cam)


@dataclass
class ObjectGeometry:
    """What the depth map says about the pointed object, in the robot frame."""

    support_z: float            # local surface the object rests on
    top_z: float                # highest point of the object
    long_axis_rad: float | None  # XY angle of its long axis, None if too round
    width_m: float = 0.0        # extent ACROSS that axis: what the jaws must close to


def object_geometry(depth_map, su: float, sv: float, target=None,
                    K=None, cam_pos=None, cam_quat=None,
                    ee_pos=None, ee_quat=None,
                    flange_pos=None, tip_pos=None) -> ObjectGeometry | None:
    """Measure the pointed object off the depth map, or ``None`` if unusable.

    Segments by HEIGHT, not by depth band: the object rests on the support, so
    from a third-person camera the two are at almost the same *range*, but the
    object is the part standing above the local surface. The support height is a
    low percentile of the window rather than a constant, so this does not need to
    know where the table is.

    ``flange_pos``/``tip_pos`` mask out the robot's own hand, and ``target``
    caps how far above the pointed surface the object may extend. Both are
    needed: the gripper hovers directly over the target exactly when the grasp is
    planned, and the forearm reaches into the window behind it. Measured on one
    4 cm banana -- 15.4 cm of "object" raw, 12.4 cm with the hand masked, 4 cm
    with both -- and a corrupted cloud corrupts the long axis and the top height
    that the closure depends on.

    Everything here comes from the depth map, the camera calibration and forward
    kinematics. No scene knowledge, so it transfers off-sim.
    """
    d = np.asarray(depth_map, dtype=float)
    while d.ndim > 2:
        d = d[..., 0]
    sh, sw = d.shape[:2]

    if K is not None:
        fx = float(np.asarray(K, dtype=float).reshape(3, 3)[0, 0])
    else:
        fx = intrinsics()[0]
    centre_depth = float(d[int(np.clip(sv, 0, sh - 1)), int(np.clip(su, 0, sw - 1))])
    if not np.isfinite(centre_depth) or centre_depth <= 0:
        return None
    rad = int(round(fx * GRASP_AXIS_RADIUS_M / centre_depth))
    if rad < 2:
        return None

    u0, u1 = int(max(0, su - rad)), int(min(sw, su + rad + 1))
    v0, v1 = int(max(0, sv - rad)), int(min(sh, sv + rad + 1))
    window = d[v0:v1, u0:u1]
    vv, uu = np.nonzero(np.isfinite(window) & (window > 0))
    if uu.size < GRASP_AXIS_MIN_POINTS:
        return None

    pts = np.array([
        unproject(u0 + u, v0 + v, window[v, u], K=K, cam_pos=cam_pos,
                  cam_quat=cam_quat, ee_pos=ee_pos, ee_quat=ee_quat)
        for u, v in zip(uu, vv)
    ])
    if flange_pos is not None and tip_pos is not None:
        pts = pts[_distance_to_segment(pts, flange_pos, tip_pos) > GRIPPER_EXCLUSION_M]
        if len(pts) < GRASP_AXIS_MIN_POINTS:
            return None
    if target is not None:
        ceiling = float(np.asarray(target, dtype=float).ravel()[2]) + OBJECT_TOP_MARGIN_M
        pts = pts[pts[:, 2] <= ceiling]
        if len(pts) < GRASP_AXIS_MIN_POINTS:
            return None

    support = float(np.percentile(pts[:, 2], 20))
    obj = pts[pts[:, 2] > support + GRASP_AXIS_MIN_HEIGHT_M]
    if len(obj) < GRASP_AXIS_MIN_POINTS:
        return None

    axis_rad = None
    flat = obj[:, :2] - obj[:, :2].mean(axis=0)
    eigval, eigvec = np.linalg.eigh(flat.T @ flat)
    if eigval[0] > 0 and np.sqrt(eigval[1] / eigval[0]) >= GRASP_AXIS_MIN_ELONGATION:
        axis_rad = float(np.arctan2(eigvec[1, 1], eigvec[0, 1]))
    # Extent across the minor axis: how wide the object is between the jaws, and
    # therefore how far they have to close before they are ON it. Percentiles,
    # not min/max, so a few stray points cannot inflate it.
    minor = flat @ eigvec[:, 0]
    width = float(np.percentile(minor, 95) - np.percentile(minor, 5))
    return ObjectGeometry(support_z=support, top_z=float(obj[:, 2].max()),
                          long_axis_rad=axis_rad, width_m=width)


def _distance_to_segment(points, a, b) -> np.ndarray:
    """Per-point distance to the segment ``a``-``b``."""
    a = np.asarray(a, dtype=float).ravel()[:3]
    b = np.asarray(b, dtype=float).ravel()[:3]
    ab = b - a
    denom = float(ab @ ab)
    if denom <= 1e-12:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip((points - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def yaw_aligned_quat(quat_ref_wxyz, yaw_rad: float) -> np.ndarray:
    """``quat_ref`` spun about WORLD +Z until its jaw axis sits at ``yaw_rad``.

    Only the wrist turns: the top-down approach direction that ``quat_ref``
    encodes is preserved exactly, which is what keeps this composable with the
    orientation hold instead of fighting it.

    The jaw axis is base_link's +Y (``robolab.constants.GRASP_JAW_BODY``), and a
    gripper turned 180 degrees grips identically, so the correction is wrapped
    into +-90 degrees — the arm never takes the long way round for a grasp that
    is the same grasp.
    """
    q = np.asarray(quat_ref_wxyz, dtype=float).ravel()[:4]
    jaw = quat_rotate(q, (0.0, 1.0, 0.0))
    current = float(np.arctan2(jaw[1], jaw[0]))
    delta = (float(yaw_rad) - current + np.pi / 2) % np.pi - np.pi / 2
    half = delta / 2.0
    return _quat_mul_wxyz(np.array([np.cos(half), 0.0, 0.0, np.sin(half)]), q)


class PointingController:
    """Phase machine turning POINT/PHASE replies into metric rel-IK deltas."""

    def __init__(self, step_m: float = STEP_M, gripper_hold_steps: int = GRIPPER_HOLD_STEPS):
        self.step_m = float(step_m)
        # Last grasp yaw/target for hysteresis (see _grasp_plan). Small cache,
        # not hold counting: the caller still owns dwell timing.
        self._last_grasp_yaw: float | None = None
        self._last_grasp_target = None
        # Sensor pixel of the most recent target, so the yaw estimator can window
        # the depth map around it without unprojecting the point twice.
        self._last_sensor_uv: tuple[float, float] = (0.0, 0.0)
        # Advertised to the caller, which owns the latch: GRASP returns a
        # zero delta and a closed gripper, and the CALLER replays that for
        # gripper_hold_steps CONTROL steps. Counting the hold down here as well
        # would stall for that many DECISIONS instead (8x longer by default,
        # each one a wasted VLM call), so this class stays stateless.
        self.gripper_hold_steps = int(gripper_hold_steps)

    def step(self, harness_obs: dict, point: dict, tip_pos) -> tuple[np.ndarray, float, bool]:
        """Return ``(delta_m[3], gripper_cmd, done)`` for one decision.

        ``point`` is ``{"x", "y", "phase"}`` from the backend (0-1000 frame).
        Stateless — see ``gripper_hold_steps``.
        """
        phase = str(point["phase"]).upper()
        if phase == "DONE":
            return np.zeros(3), 0.0, True
        if phase == "GRASP":
            return np.zeros(3), 1.0, False

        target = self._target_for(harness_obs, point)
        tip = np.asarray(tip_pos, dtype=float).ravel()[:3]
        delta = np.asarray(target, dtype=float).ravel()[:3] - tip
        logger.info(
            "[Pinpoint] tip=%s dist=%.4f m",
            np.round(tip, 4).tolist(), float(np.linalg.norm(delta)),
        )
        if phase == "LIFT":
            delta = np.array([0.0, 0.0, self.step_m])
        elif phase == "PLACE":
            delta[2] = min(delta[2], 0.0)
            if float(np.linalg.norm(delta)) < GRASP_RADIUS_M:
                return np.zeros(3), 0.0, False  # arrived: open the fingers
            return self._cap(delta), 1.0, False
        dist = float(np.linalg.norm(delta))
        if dist < GRASP_RADIUS_M and phase in ("ALIGN", "APPROACH"):
            return np.zeros(3), 1.0, False
        gripper = 0.0 if phase in ("APPROACH", "ALIGN") else 1.0
        return self._cap(delta), gripper, False

    def _target_for(self, harness_obs: dict, point: dict) -> np.ndarray:
        """The 3D point the VLM designated, in the robot-root frame."""
        depth_map = self._depth_map(harness_obs)
        # The rendered depth map is the authority on sensor resolution; K is
        # expressed in those same pixels.
        sh, sw = depth_map.shape[:2]
        su, sv = frame_point_to_sensor(point["x"], point["y"], sw, sh,
                                       int(harness_obs.get("view_rot_deg", 180)))
        depth = self._sample_depth(depth_map, su, sv)
        target = unproject(
            su, sv,
            depth,
            K=harness_obs.get("K"),
            cam_pos=harness_obs.get("cam_pos"),
            cam_quat=harness_obs.get("cam_quat"),
            ee_pos=harness_obs.get("ee_pos"),
            ee_quat=harness_obs.get("ee_quat"),
        )
        logger.info("[Pinpoint] target=%s depth=%.3f m @sensor(%d,%d)",
                    np.round(target, 4).tolist(), depth, int(su), int(sv))
        self._last_sensor_uv = (su, sv)
        return np.asarray(target, dtype=float).ravel()[:3]

    def _measure(self, harness_obs: dict, target: np.ndarray) -> ObjectGeometry | None:
        """Depth-map geometry of the object at the last pointed pixel."""
        su, sv = self._last_sensor_uv
        try:
            geom = object_geometry(
                self._depth_map(harness_obs), su, sv, target,
                K=harness_obs.get("K"),
                cam_pos=harness_obs.get("cam_pos"),
                cam_quat=harness_obs.get("cam_quat"),
                ee_pos=harness_obs.get("ee_pos"),
                ee_quat=harness_obs.get("ee_quat"),
                flange_pos=harness_obs.get("ee_pos"),
                tip_pos=harness_obs.get("tip_pos"),
            )
        except Exception:  # noqa: BLE001 - a bad measurement must not lose the grasp
            logger.warning("[Pinpoint] object measurement failed; using defaults",
                           exc_info=True)
            return None
        if geom is None:
            logger.info("[Pinpoint] object not measurable (sparse/occluded); "
                        "falling back to fixed grasp depth and the latched yaw")
        else:
            logger.info("[Pinpoint] object: support z=%.3f top z=%.3f (h=%.1f cm) "
                        "long axis %s",
                        geom.support_z, geom.top_z,
                        100 * (geom.top_z - geom.support_z),
                        "none" if geom.long_axis_rad is None
                        else "%.1f deg" % np.degrees(geom.long_axis_rad))
        return geom

    def plan(self, harness_obs: dict, point: dict, tip_pos) -> tuple[list[Waypoint], bool]:
        """One VLM decision -> a whole primitive, as ``(waypoints, done)``.

        The client re-asks every query window while a servo-kind
        (APPROACH/ALIGN) plan is alive, so the model stays in the loop; a fresh
        target within REPLAN_HYST_M of the leg being servoed keeps the old plan
        instead of re-aiming. That hysteresis is load-bearing: a point carries
        ~20-50 px of error, over a centimetre at grasp range, and re-aiming on
        every query makes the arm chase measurement noise instead of converging.
        Terminal primitives (GRASP/LIFT/PLACE) run to exhaustion uninterrupted.

        The phases keep their semantic meaning; only their execution changes.
        """
        phase = str(point["phase"]).upper()
        tip = np.asarray(tip_pos, dtype=float).ravel()[:3]
        if phase == "DONE":
            return [], True
        if phase == "LIFT":
            # Nothing to unproject: straight up from wherever the fingers are.
            return [Waypoint(tip + np.array([0.0, 0.0, LIFT_HEIGHT_M]), 1.0)], False

        target = self._target_for(harness_obs, point)
        above = np.array([target[0], target[1], target[2] + PREGRASP_HEIGHT_M])

        if phase == "GRASP":
            return self._grasp_plan(target, above, self._measure(harness_obs, target)), False
        if phase == "PLACE":
            release = np.array([target[0], target[1], target[2] + GRASP_DEPTH_M])
            return [
                Waypoint(above, 1.0),
                Waypoint(release, 1.0, tol_m=0.008),
                Waypoint(None, 0.0, self.gripper_hold_steps),  # open and settle
            ], False
        # APPROACH / ALIGN: stage above the target, fingers open -- and once the
        # fingers ARE over it, descend and grasp without another round trip.
        #
        # Arriving is the trigger. This used to return the staging waypoint
        # unconditionally and wait for the model to emit GRASP, which a local
        # Qwen never does: a whole episode came back 26 APPROACH / 1 ALIGN /
        # 0 GRASP, so the arm parked 10 cm over the banana and re-planned the
        # waypoint it was already standing on until the budget ran out. Silent,
        # and indistinguishable from an arm that cannot reach.
        #
        # When to close the hand is a control decision, not a perception one.
        # Every published pointing stack keeps the state machine in the
        # primitive library and asks the VLM only WHERE (RoboPoint, MOKA,
        # VoxPoser); this module's own docstring says so and then did the
        # opposite for the one transition that matters.
        staged = (float(np.linalg.norm(tip[:2] - target[:2])) <= STAGED_RADIUS_M
                  and tip[2] > target[2])
        if staged:
            logger.info(
                "[Pinpoint] staged over target (%.3f m horizontal, %.3f m up): "
                "descending to grasp without re-querying",
                float(np.linalg.norm(tip[:2] - target[:2])), tip[2] - target[2],
            )
            return self._grasp_plan(target, above, self._measure(harness_obs, target)), False
        return [Waypoint(above, 0.0)], False

    def _grasp_plan(self, target: np.ndarray, above: np.ndarray,
                    geom: ObjectGeometry | None = None) -> list[Waypoint]:
        """Stage overhead, descend, close and hold, lift clear.

        The closure height and the wrist yaw both come from ``geom`` — measured
        off the depth map — because both were wrong as constants. ``None`` (too
        sparse, too occluded, too round) falls back to the fixed offset below the
        sampled surface and the latched orientation.

        The yaw rides on EVERY leg, not just the descent: the wrist turns while
        it is still clear of the table, holds through the close so the fingers do
        not twist the object out of the jaws, and holds through the lift so a
        settled grasp is not spun loose.
        """
        yaw_rad = None
        if geom is not None and geom.long_axis_rad is not None:
            yaw_rad = geom.long_axis_rad + np.pi / 2.0
            # Hysteresis: depth noise jitters the long axis a few degrees per
            # decision (68.7 vs 64.1 measured), and chasing it visibly rotates
            # the wrist every grasp cycle. Same neighborhood + <15 deg change
            # keeps the previous yaw. π-periodic: 180 deg apart grips identically.
            if (self._last_grasp_yaw is not None
                    and self._last_grasp_target is not None
                    and float(np.linalg.norm(
                        np.asarray(target, dtype=float).ravel()[:2]
                        - np.asarray(self._last_grasp_target, dtype=float).ravel()[:2]
                    )) < 0.03
                    and abs((yaw_rad - self._last_grasp_yaw + np.pi / 2) % np.pi
                            - np.pi / 2) < np.radians(15)):
                yaw_rad = self._last_grasp_yaw
        self._last_grasp_target = np.asarray(target, dtype=float).ravel()[:2].copy()
        self._last_grasp_yaw = yaw_rad
        grasp = np.array([target[0], target[1],
                          self._closure_height(target[2], geom)])
        return [
            Waypoint(above, 0.0, yaw_rad=yaw_rad),               # stage overhead, open
            # Contact ends this leg, whether or not the waypoint is reached.
            Waypoint(grasp, 0.0, tol_m=0.008, yaw_rad=yaw_rad, on_stall="advance"),
            Waypoint(None, 1.0, self.gripper_hold_steps, yaw_rad=yaw_rad,
                     close_to_m=(geom.width_m if geom is not None
                                 and geom.width_m > 0.0 else None)),
            Waypoint(above, 1.0, yaw_rad=yaw_rad),               # lift clear
        ]

    @staticmethod
    def _closure_height(surface_z: float, geom: ObjectGeometry | None) -> float:
        """Height the fingertips close at.

        The depth sample gives the object's VISIBLE TOP, and a fixed offset below
        it does not know how tall the object is: on a 4 cm banana the old
        constant closed at 65% of its height, above the widest part, which
        squeezes a round object out of the jaws instead of gripping it.

        What has to line up is the PAD, and the pad is not at the commanded
        height: it runs from the fingertip back up towards the flange. Sending
        the tip to the object's equator — the previous rule — therefore parks the
        whole gripping face above the object and closes on its top edge, which is
        still "a bit high" and still squeezes a round object out. The tip goes
        half a pad BELOW the equator so the face is centred on it.

        Bounded three ways: never nearer the support than FINGERTIP_FLOOR_M (the
        2F-85's fingers dip as they swing in, and jam on the table below that),
        never above the old fixed offset, and never more than MAX_BITE_M below
        what was pointed at, so a bad support estimate cannot drive the hand
        through the table.
        """
        fixed = surface_z - GRASP_DEPTH_M
        if geom is None:
            return fixed
        middle = 0.5 * (geom.support_z + geom.top_z)
        centred = middle - 0.5 * PAD_LENGTH_M
        floor = geom.support_z + max(FINGERTIP_FLOOR_M, GRASP_CLEARANCE_M)
        return float(np.clip(max(centred, floor), surface_z - MAX_BITE_M, fixed))

    @staticmethod
    def _depth_map(harness_obs: dict) -> np.ndarray:
        """The wrist depth map as a 2-D array of metres."""
        depth = harness_obs.get("depth")
        if depth is None:
            raise MissingDepthError(
                "pointing needs depth: wrap the wrist camera with with_depth() "
                "(robolab.variations.camera) so image_obs carries wrist_cam_depth "
                "plus wrist_cam_pos/quat/K"
            )
        d = np.asarray(depth, dtype=float)
        while d.ndim > 2:
            d = d[..., 0]
        return d

    def _sample_depth(self, d: np.ndarray, su: float, sv: float) -> float:
        """Median depth in a small window. (su, sv) are sensor pixels."""
        h, w = d.shape[:2]
        u = min(w - 1, max(0, int(round(su))))
        v = min(h - 1, max(0, int(round(sv))))
        # The model rarely lands dead-centre (see the grasp-point prompt rule):
        # spiral out until a valid patch is found, so near-misses servo instead
        # of crashing. Beyond ~40 sensor px the point itself is wrong — hold.
        for r in (2, 5, 10, 20, 40):
            window = d[max(0, v - r):v + r + 1, max(0, u - r):u + r + 1]
            valid = window[np.isfinite(window) & (window > 0)]
            if valid.size >= 3:
                return float(np.median(valid))
        raise NoValidDepthError(
            f"no valid depth at sensor pixel ({u}, {v}): "
            "the pointed ray hits nothing (skybox/inf)"
        )

    def _cap(self, delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=float).ravel()[:3]
        dist = float(np.linalg.norm(delta))
        if dist > self.step_m and dist > 0:
            delta = delta * (self.step_m / dist)
        return delta
