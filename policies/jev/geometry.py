# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic geometry -- the layer every decision criterion is written against.

Port of RoboJEV's ``src/jev_vla_sim/geometry.py``. The model is never asked to
do coordinate arithmetic: it reads per-axis ``negative``/``zero``/``positive``
labels plus alignment booleans, and every INTENT and MOTOR criterion quotes
those labels by name ("``grasp_tcp_from_tcp.directions.x`` is negative").

RoboJEV's RULES say so explicitly -- *"Use supplied semantic directions /
alignment / clearance facts rather than doing coordinate arithmetic"* -- so
this module is the keystone of the two-stage design, not a convenience.
"""

from __future__ import annotations

import numpy as np

# The deadband is a FRACTION OF THE STEP, not a constant. RoboJEV pairs a 6 mm
# tolerance with a 10 mm step, and that ratio -- not the millimetre value -- is
# what makes the loop converge: a step taken from just outside the band must
# land inside it. Copy the 6 mm to a 20 mm step and every axis overshoots the
# band on each decision and oscillates forever, which is what the port did.
DIRECTION_TOLERANCE_RATIO = 0.6
# Kept for the default step (RoboJEV's 10 mm), so callers that do not thread a
# step size through still get upstream's numbers.
DIRECTION_TOLERANCE_M = 0.006


def tolerance_for_step(step_m: float) -> float:
    """Deadband matched to one decision's travel."""
    return DIRECTION_TOLERANCE_RATIO * float(step_m)


def direction(value: float, tolerance: float = DIRECTION_TOLERANCE_M) -> str:
    """Signed scalar -> ``negative`` / ``zero`` / ``positive``."""
    value = float(value)
    return ("positive" if value > tolerance
            else "negative" if value < -tolerance
            else "zero")


def relation(delta, tolerance: float = DIRECTION_TOLERANCE_M) -> dict:
    """A target-minus-current vector as the model sees it.

    ``delta_m`` is kept (rounded) so a human reading the log can check the
    labels, but no criterion refers to it.
    """
    delta = np.asarray(delta, dtype=float).ravel()[:3]
    return {
        "delta_m": [round(float(v), 4) for v in delta],
        "directions": {axis: direction(v, tolerance)
                       for axis, v in zip(("x", "y", "z"), delta)},
        "xy_aligned": bool(np.max(np.abs(delta[:2])) <= tolerance),
    }


def axis_angle_xy(vector) -> float:
    """XY heading of an undirected axis, folded into [-pi/2, pi/2).

    An axis has no head or tail: a banana pointing north-east and one pointing
    south-west are the same object in the same pose, and a gripper turned 180
    degrees grips identically. Folding here keeps every later comparison from
    having to re-discover that.
    """
    vector = np.asarray(vector, dtype=float).ravel()[:3]
    return (float(np.arctan2(vector[1], vector[0])) + np.pi / 2) % np.pi - np.pi / 2


def signed_axis_error(target_rad: float, current_rad: float) -> float:
    """Shortest rotation taking ``current`` onto ``target``, in [-pi/2, pi/2)."""
    return (float(target_rad) - float(current_rad) + np.pi / 2) % np.pi - np.pi / 2
