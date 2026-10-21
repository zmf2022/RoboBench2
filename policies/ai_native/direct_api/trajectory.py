# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode Direct-API B-spline outputs into absolute DROID action chunks.

The model-facing coefficients are B-spline control points, not polynomial
coefficients. Each control point is one absolute 8-D DROID action: seven
Franka joint targets followed by a continuous gripper command in ``[0, 1]``.
The policy client may later binarize the gripper (normally at ``0.5``).

The recommended tool schema is therefore::

    {
        "control_points": [[joint_1, ..., joint_7, gripper], ...]
    }

The client can prepend the current 8-D robot state to future model-emitted
control points, decode ``action_horizon + 1`` samples, then discard the first
sample. This anchors the curve at the current state while returning exactly
``action_horizon`` future actions. Gripper interpolation remains continuous so
thresholding happens in one place after decoding.
"""

from __future__ import annotations

from numbers import Integral
from typing import Sequence

import numpy as np

DROID_ARM_DIM = 7
DROID_ACTION_DIM = 8


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_integer(value: int, name: str) -> int:
    parsed = _nonnegative_integer(value, name)
    if parsed == 0:
        raise ValueError(f"{name} must be at least 1")
    return parsed


def _control_point_array(
    control_points: Sequence[Sequence[float]] | np.ndarray,
    *,
    expected_width: int | None,
    name: str,
) -> np.ndarray:
    try:
        points = np.asarray(control_points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc

    expected = "a non-empty 2-D array" if expected_width is None else f"shape (N, {expected_width}) with N >= 1"
    if points.ndim != 2 or not points.shape[0] or (expected_width is not None and points.shape[1] != expected_width):
        raise ValueError(f"{name} must have {expected}")
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{name} must contain only finite values")
    return points


def _open_uniform_knots(num_control_points: int, degree: int) -> np.ndarray:
    """Return the clamped/open-uniform knot vector on ``[0, 1]``."""
    internal_count = num_control_points - degree - 1
    internal = (
        np.arange(1, internal_count + 1, dtype=np.float64) / (internal_count + 1)
        if internal_count
        else np.empty(0, dtype=np.float64)
    )
    return np.concatenate(
        (
            np.zeros(degree + 1, dtype=np.float64),
            internal,
            np.ones(degree + 1, dtype=np.float64),
        )
    )


def _de_boor(
    parameter: float,
    control_points: np.ndarray,
    knots: np.ndarray,
    degree: int,
) -> np.ndarray:
    """Evaluate one point using de Boor's numerically stable recurrence."""
    if parameter <= 0.0:
        return control_points[0].copy()
    if parameter >= 1.0:
        return control_points[-1].copy()

    num_control_points = control_points.shape[0]
    span = int(np.searchsorted(knots, parameter, side="right") - 1)
    span = min(max(span, degree), num_control_points - 1)
    work = control_points[span - degree : span + 1].copy()

    for recurrence in range(1, degree + 1):
        for index in range(degree, recurrence - 1, -1):
            knot_index = span - degree + index
            left = knots[knot_index]
            right = knots[knot_index + degree - recurrence + 1]
            weight = 0.0 if right == left else (parameter - left) / (right - left)
            work[index] = (1.0 - weight) * work[index - 1] + weight * work[index]
    return work[degree]


def evaluate_clamped_bspline(
    control_points: Sequence[Sequence[float]] | np.ndarray,
    *,
    num_samples: int,
    degree: int = 3,
) -> np.ndarray:
    """Evaluate an open-uniform B-spline at evenly spaced samples.

    The effective degree is ``min(degree, N - 1)``. Consequently one control
    point produces a constant trajectory and two points produce a straight
    line even when cubic interpolation was requested. With two or more output
    samples, the first and last results are exactly the first and last control
    points. A single output sample represents the trajectory's final target.
    """
    samples = _positive_integer(num_samples, "num_samples")
    requested_degree = _nonnegative_integer(degree, "degree")
    points = _control_point_array(
        control_points,
        expected_width=None,
        name="control_points",
    )

    if samples == 1:
        return points[-1:].copy()

    effective_degree = min(requested_degree, points.shape[0] - 1)
    if points.shape[0] == 1:
        return np.repeat(points[:1], samples, axis=0)

    knots = _open_uniform_knots(points.shape[0], effective_degree)
    parameters = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    trajectory = np.stack(
        [_de_boor(parameter, points, knots, effective_degree) for parameter in parameters],
        axis=0,
    )
    # Assign endpoints explicitly to make the endpoint contract independent of
    # floating-point details in the recurrence.
    trajectory[0] = points[0]
    trajectory[-1] = points[-1]
    return trajectory


def decode_bspline_trajectory(
    control_points: Sequence[Sequence[float]] | np.ndarray,
    *,
    action_horizon: int,
    degree: int = 3,
) -> np.ndarray:
    """Return exactly ``action_horizon`` absolute 8-D DROID actions.

    Control points have seven absolute arm joint targets followed by a gripper
    value in ``[0, 1]``. Gripper values are deliberately not rounded. The
    integrating client should apply its usual binary threshold and robot joint
    limit/delta checks after decoding.
    """
    horizon = _positive_integer(action_horizon, "action_horizon")
    points = _control_point_array(
        control_points,
        expected_width=DROID_ACTION_DIM,
        name="control_points",
    )
    gripper_points = points[:, DROID_ARM_DIM]
    if np.any((gripper_points < 0.0) | (gripper_points > 1.0)):
        raise ValueError("control-point gripper values must be in [0, 1]")
    return evaluate_clamped_bspline(
        points,
        num_samples=horizon,
        degree=degree,
    ).astype(np.float32)


def decode_anchored_bspline_trajectory(
    current_action: Sequence[float] | np.ndarray,
    future_control_points: Sequence[Sequence[float]] | np.ndarray,
    *,
    action_horizon: int,
    degree: int = 3,
) -> np.ndarray:
    """Decode future model control points along a current-state anchor.

    This is the integration-friendly variant: it prepends ``current_action``,
    evaluates one extra sample so the anchor is included exactly, then drops
    that anchor. The returned array therefore always contains exactly
    ``action_horizon`` future absolute DROID actions.
    """
    try:
        anchor = np.asarray(current_action, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("current_action must be a numeric vector with shape (8,)") from exc
    if anchor.shape != (DROID_ACTION_DIM,):
        raise ValueError("current_action must have shape (8,)")
    if not np.all(np.isfinite(anchor)):
        raise ValueError("current_action must contain only finite values")

    future = _control_point_array(
        future_control_points,
        expected_width=DROID_ACTION_DIM,
        name="future_control_points",
    )
    horizon = _positive_integer(action_horizon, "action_horizon")
    anchored = np.concatenate((anchor[np.newaxis], future), axis=0)
    return decode_bspline_trajectory(
        anchored,
        action_horizon=horizon + 1,
        degree=degree,
    )[1:]
