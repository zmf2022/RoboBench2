# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured state for the Jev two-stage decision. No images.

Port of RoboJEV's ``src/jev_vla_sim/state.py``. The schema is the contract the
INTENT and MOTOR criteria are written against: **every criterion must name a
field that exists here**, or the model is guessing. Keep the two in sync --
adding a criterion that mentions a fact this builder does not measure is the
failure mode this module exists to prevent.

Pure numpy on purpose: no simulator, no robolab imports, so the schema and its
derived booleans stay unit-testable without booting Isaac Sim.
"""

from __future__ import annotations

import numpy as np

from policies.jev.geometry import (
    DIRECTION_TOLERANCE_M,
    axis_angle_xy,
    relation,
    signed_axis_error,
)
from policies.vlm_pinpoint.connector import quat_rotate

# A closed gripper whose fingers have collapsed onto each other holds NOTHING.
# 5 mm rather than a real rig's 1 mm because ``gripper_pos`` carries
# GaussianNoiseCfg(std=0.05) (robolab/robots/droid.py), i.e. ~4 mm of width
# noise -- a tighter window sits inside the noise floor.
GRASP_WIDTH_MIN_M = 0.005
# ...and fingers near their full 85 mm opening are not gripping either, they are
# merely touching. RoboJEV uses the same two-sided band on the Panda hand.
GRASP_WIDTH_MAX_M = 0.070
# The jaws separate along GRASP_JAW_BODY's +y (robolab/constants.py), so THIS is
# the axis that must end up across the object, not along it.
JAW_AXIS_LOCAL = (0.0, 1.0, 0.0)
# How far the wrist may be off perpendicular before the grasp stops fitting.
# The 2F-85 opens 85 mm and grips by ~70; a banana is ~190 mm long and ~35 thick,
# and at a yaw error of theta the jaws must span 190*sin(theta) + 35*cos(theta).
# That reaches 70 mm at 10.6 degrees, so this is the angle at which the grasp
# stops being possible -- not the angle at which it gets worse.
GRASP_YAW_TOLERANCE_RAD = 0.185
# Motion below this, on a decision that asked for a full step, is no motion:
# the arm is against something. Mirrors vlm_pinpoint's STALL_PROGRESS_M, and the
# 2F-85's fingers swing inward AND dip as they close, so a descent that stalls is
# usually the finger arc already on the table with the jaws still open.
STALL_PROGRESS_M = 0.001
# Below this the released object counts as at rest rather than still falling.
SETTLED_SPEED_MPS = 0.02
# RoboJEV's withdraw/finish split: retreat until the TCP is this far above the
# object, then hold.
WITHDRAW_CLEARANCE_M = 0.10


def _vec(value, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=float)
    return np.asarray(value, dtype=float).ravel()[:3]


def _quat(value) -> np.ndarray:
    """WXYZ, defaulting to identity. Never truth-test a numpy array."""
    if value is None:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.asarray(value, dtype=float).ravel()[:4]


def build_state(
    *,
    task: str,
    task_id: str,
    state_id: str,
    step_id: int,
    tcp,
    tcp_quaternion,
    object_pos,
    object_quaternion,
    object_velocity,
    container_pos,
    gripper_width: float,
    gripper_target: str,
    objects_in_contact,
    object_label: str,
    container_label: str,
    grasp_hover_m: float,
    release_hover_m: float,
    container_radius_m: float,
    object_rest_z: float,
    transport_clearance_m: float,
    object_long_axis_local=None,
    previous_action_stalled: bool = False,
    direction_tolerance_m: float = DIRECTION_TOLERANCE_M,
    recent_actions=None,
) -> dict:
    """Assemble the decision state. Positions are robot-base metres."""
    tcp = _vec(tcp)
    obj = _vec(object_pos)
    container = _vec(container_pos)
    velocity = _vec(object_velocity)
    width = float(gripper_width)
    contacts = list(objects_in_contact or [])
    tol = float(direction_tolerance_m)

    # Physical, not commanded: the fingers must be TOUCHING the object and the
    # jaws must be somewhere between "collapsed onto each other" and "wide
    # open". A close command alone does not prove a grasp.
    held = (object_label
            if object_label in contacts and GRASP_WIDTH_MIN_M < width < GRASP_WIDTH_MAX_M
            else None)

    object_to_tcp = obj - tcp
    grasp = obj + np.array([0.0, 0.0, grasp_hover_m])

    # Aim the OBJECT at the container, not the TCP: while held, the object sits
    # wherever it ended up in the jaws, and that offset has to come back out of
    # the placement target. RoboJEV does the same subtraction.
    placement = container + np.array([0.0, 0.0, release_hover_m])
    if held is not None:
        placement = placement - object_to_tcp

    # Which way the object lies, and which way the jaws currently close. A cube
    # can be grasped from any yaw; an elongated object cannot, so the two have to
    # be compared before the fingers close.
    jaw_rad = axis_angle_xy(quat_rotate(_quat(tcp_quaternion), JAW_AXIS_LOCAL))
    if object_long_axis_local is None:
        long_rad = None
        yaw_error = 0.0
        yaw_aligned = True
    else:
        long_rad = axis_angle_xy(quat_rotate(_quat(object_quaternion),
                                             object_long_axis_local))
        # The jaws must close ACROSS the long axis, hence the quarter turn.
        yaw_error = signed_axis_error(long_rad + np.pi / 2, jaw_rad)
        yaw_aligned = bool(abs(yaw_error) <= GRASP_YAW_TOLERANCE_RAD)

    grasp_rel = relation(grasp - tcp, tol)
    # The descent ends either by arriving or by running out of room. pinpoint's
    # grasp waypoint carries on_stall="advance" for the same reason: below the
    # fingertip floor the arc jams on the table and no further command helps.
    at_grasp_pose = bool(
        set(grasp_rel["directions"].values()) == {"zero"}
        or (grasp_rel["xy_aligned"] and grasp_rel["directions"]["z"] == "negative"
            and previous_action_stalled))

    lifted_m = float(obj[2] - object_rest_z)
    inside_xy = bool(np.linalg.norm(obj[:2] - container[:2]) <= container_radius_m)
    settled = bool(held is None and inside_xy
                   and float(np.linalg.norm(velocity)) <= SETTLED_SPEED_MPS)

    return {
        "task": task,
        "task_id": task_id,
        "state_id": state_id,
        "step_id": int(step_id),
        "frame": "robot_base",
        "units": "m",
        "quaternion_order": "wxyz",
        "robot": {
            "tcp_position": [round(float(v), 4) for v in tcp],
            "tcp_quaternion": [round(float(v), 4) for v in _quat(tcp_quaternion)],
            "gripper_width": round(width, 4),
            "gripper_target": gripper_target,
            "finger_object_contacts": contacts,
            "held_object": held,
        },
        "objects": [{
            "id": object_label,
            "label": object_label,
            "position": [round(float(v), 4) for v in obj],
            "quaternion": [round(float(v), 4) for v in _quat(object_quaternion)],
            "velocity": [round(float(v), 4) for v in velocity],
        }],
        "target": {
            "id": container_label,
            "label": container_label,
            "position": [round(float(v), 4) for v in container],
            "radius_m": round(float(container_radius_m), 4),
        },
        "relations": {
            "object_from_tcp": relation(obj - tcp, tol),
            "grasp_tcp_from_tcp": grasp_rel,
            "at_grasp_pose": at_grasp_pose,
            "previous_action_stalled": bool(previous_action_stalled),
            "target_from_object": relation(container - obj, tol),
            "placement_tcp_from_tcp": relation(placement - tcp, tol),
            "object_lifted_above_rest_m": round(lifted_m, 4),
            "object_clear_for_transport": bool(lifted_m >= transport_clearance_m),
            "object_inside_target_xy": inside_xy,
            "object_settled_in_target": settled,
            "tcp_above_object_m": round(float(tcp[2] - obj[2]), 4),
            "object_long_axis_rad": None if long_rad is None else round(long_rad, 4),
            "jaw_axis_rad": round(jaw_rad, 4),
            "grasp_yaw_error_rad": round(float(yaw_error), 4),
            "grasp_yaw_aligned": yaw_aligned,
            "transport_clearance_m": round(float(transport_clearance_m), 4),
            "withdraw_clearance_m": WITHDRAW_CLEARANCE_M,
            "direction_tolerance_m": round(tol, 4),
        },
        "recent_actions": list(recent_actions or []),
    }
