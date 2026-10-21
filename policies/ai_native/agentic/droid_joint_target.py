# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded local DLS conversion from a DROID EEF target to one joint action.

The model target is the Robotiq ``base_link`` pose in the DROID robot-root
frame.  This adapter reads only current robot kinematics.  It never advances
physics, reads task objects, plans collisions, or assumes the requested pose
will be reached.  Call it again after every real simulator step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from policies.ai_native.agentic.joint_actions import bounded_joint_action
from policies.ai_native.common.agent_eef import AgentEEFPlan

ARM_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))
EEF_BODY_NAME = "base_link"


class DroidEEFControlError(ValueError):
    """A rejected DROID EEF-to-joint conversion; the simulator was not stepped."""


def _array(value: Any, *, shape: tuple[int, ...], name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise DroidEEFControlError(f"{name} must contain finite numbers with shape {shape}") from exc
    if result.shape != shape or not np.all(np.isfinite(result)):
        raise DroidEEFControlError(f"{name} must contain finite numbers with shape {shape}")
    return result


def _unit_quaternion(value: Any, name: str) -> np.ndarray:
    quaternion = _array(value, shape=(4,), name=name)
    norm = float(np.linalg.norm(quaternion))
    if norm <= np.finfo(np.float64).eps:
        raise DroidEEFControlError(f"{name} must be a nonzero quaternion")
    return quaternion / norm


def _rotation(quaternion_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(quaternion_wxyz[[1, 2, 3, 0]])


def _bounded_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector if norm <= maximum or norm == 0.0 else vector * (maximum / norm)


def compute_dls_joint_target(
    joint_positions: Any,
    joint_limits: Any,
    jacobian_root: Any,
    current_position: Any,
    current_quaternion_wxyz: Any,
    target_position: Any,
    target_quaternion_wxyz: Any,
    *,
    damping: float = 0.05,
    max_joint_delta_rad: float = 0.05,
    max_translation_step_m: float = 0.02,
    max_rotation_step_rad: float = 0.1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return one bounded absolute seven-joint target and diagnostics."""

    joints = _array(joint_positions, shape=(7,), name="joint_positions")
    limits = _array(joint_limits, shape=(7, 2), name="joint_limits")
    jacobian = _array(jacobian_root, shape=(6, 7), name="jacobian_root")
    current_position = _array(current_position, shape=(3,), name="current_position")
    target_position = _array(target_position, shape=(3,), name="target_position")
    current_quaternion = _unit_quaternion(current_quaternion_wxyz, "current_quaternion_wxyz")
    target_quaternion = _unit_quaternion(target_quaternion_wxyz, "target_quaternion_wxyz")
    for name, value in (
        ("damping", damping),
        ("max_joint_delta_rad", max_joint_delta_rad),
        ("max_translation_step_m", max_translation_step_m),
        ("max_rotation_step_rad", max_rotation_step_rad),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise DroidEEFControlError(f"{name} must be finite and positive")
    if np.any(limits[:, 0] >= limits[:, 1]):
        raise DroidEEFControlError("joint_limits must have lower values below upper values")

    translation_error = target_position - current_position
    rotation_error = (_rotation(target_quaternion) * _rotation(current_quaternion).inv()).as_rotvec()
    bounded_error = np.concatenate(
        (
            _bounded_norm(translation_error, float(max_translation_step_m)),
            _bounded_norm(rotation_error, float(max_rotation_step_rad)),
        )
    )
    system = jacobian @ jacobian.T + float(damping) ** 2 * np.eye(6)
    try:
        delta = jacobian.T @ np.linalg.solve(system, bounded_error)
    except np.linalg.LinAlgError as exc:
        raise DroidEEFControlError("DLS system is numerically singular") from exc
    delta = np.clip(delta, -float(max_joint_delta_rad), float(max_joint_delta_rad))
    target = np.clip(joints + delta, limits[:, 0], limits[:, 1])
    if not np.all(np.isfinite(target)):
        raise DroidEEFControlError("computed joint target is not finite")
    diagnostics = {
        "target_translation_error_m": float(np.linalg.norm(translation_error)),
        "target_rotation_error_rad": float(np.linalg.norm(rotation_error)),
        "bounded_task_delta": bounded_error.tolist(),
        "requested_joint_delta": delta.tolist(),
        "executed_joint_delta": (target - joints).tolist(),
        "joint_limit_clipped": (~np.isclose(joints + delta, target)).tolist(),
        "physical_tracking_verified": False,
    }
    return target.astype(np.float32), diagnostics


@dataclass
class DroidJointTargetAdapter:
    """Read live DROID kinematics and produce one native joint-position action."""

    asset_name: str = "robot"
    damping: float = 0.05
    max_joint_delta_rad: float = 0.05
    max_translation_step_m: float = 0.02
    max_rotation_step_rad: float = 0.1
    _env: Any = None

    def bind_environment(self, env: Any) -> None:
        if int(getattr(env, "num_envs", 0)) != 1:
            raise ValueError("DROID bounded EEF control currently requires --num-envs 1")
        self._env = env

    def action_for_plan(self, plan: AgentEEFPlan, *, env_id: int) -> tuple[np.ndarray, Mapping[str, Any]]:
        if self._env is None:
            raise RuntimeError("DROID joint-target adapter has not been bound to the RoboLab environment")
        if env_id != 0:
            raise ValueError("DROID bounded EEF control currently supports only env_id 0")
        if set(plan.commanded_state) != {"arm"}:
            raise ValueError("DROID Agent-EEF requires exactly one effector named arm")

        robot = self._env.scene[self.asset_name]
        joint_names_value = getattr(robot, "joint_names", None)
        body_names_value = getattr(robot, "body_names", None)
        joint_names = list(joint_names_value if joint_names_value is not None else robot.data.joint_names)
        body_names = list(body_names_value if body_names_value is not None else robot.data.body_names)
        if any(joint_names.count(name) != 1 for name in (*ARM_JOINT_NAMES, "finger_joint")):
            raise DroidEEFControlError("DROID joint names do not match the bounded EEF controller")
        if body_names.count(EEF_BODY_NAME) != 1:
            raise DroidEEFControlError("DROID must contain exactly one Robotiq base_link body")
        joint_ids = [joint_names.index(name) for name in ARM_JOINT_NAMES]
        body_id = body_names.index(EEF_BODY_NAME)

        joint_positions_all = _array(
            robot.data.joint_pos[env_id], shape=(len(joint_names),), name="robot joint positions"
        )
        limits_all = _array(
            robot.data.soft_joint_pos_limits[env_id], shape=(len(joint_names), 2), name="DROID soft joint limits"
        )
        if "joint_target" in plan.metadata:
            return bounded_joint_action(
                plan.metadata["joint_target"], joint_positions_all[joint_ids], limits_all[joint_ids],
                self.max_joint_delta_rad,
            )

        root_position = _array(robot.data.root_pos_w[env_id], shape=(3,), name="robot root position")
        root_quaternion = _unit_quaternion(robot.data.root_quat_w[env_id], "robot root quaternion")
        body_position = _array(robot.data.body_pos_w[env_id, body_id], shape=(3,), name="EEF body position")
        body_quaternion = _unit_quaternion(
            robot.data.body_quat_w[env_id, body_id], "EEF body quaternion"
        )
        root_inverse = _rotation(root_quaternion).inv()
        current_position = root_inverse.apply(body_position - root_position)
        current_rotation = root_inverse * _rotation(body_quaternion)
        current_quaternion = current_rotation.as_quat()[[3, 0, 1, 2]]

        fixed_base = bool(robot.is_fixed_base)
        jacobian_body = body_id - 1 if fixed_base else body_id
        if jacobian_body < 0:
            raise DroidEEFControlError("cannot control an omitted fixed root body")
        all_jacobians = np.asarray(robot.root_physx_view.get_jacobians()[env_id].detach().cpu(), dtype=np.float64)
        if all_jacobians.ndim != 3 or jacobian_body >= all_jacobians.shape[0]:
            raise DroidEEFControlError("unexpected PhysX Jacobian shape")
        columns = joint_ids if fixed_base else [index + 6 for index in joint_ids]
        jacobian_world = all_jacobians[jacobian_body][:, columns]
        if jacobian_world.shape != (6, 7):
            raise DroidEEFControlError("selected DROID Jacobian must have shape (6, 7)")
        jacobian_root = np.empty_like(jacobian_world)
        jacobian_root[:3] = root_inverse.apply(jacobian_world[:3].T).T
        jacobian_root[3:] = root_inverse.apply(jacobian_world[3:].T).T

        target_state = plan.commanded_state["arm"]
        arm_target, diagnostics = compute_dls_joint_target(
            joint_positions_all[joint_ids],
            limits_all[joint_ids],
            jacobian_root,
            current_position,
            current_quaternion,
            target_state.position_m,
            target_state.quaternion_wxyz,
            damping=self.damping,
            max_joint_delta_rad=self.max_joint_delta_rad,
            max_translation_step_m=self.max_translation_step_m,
            max_rotation_step_rad=self.max_rotation_step_rad,
        )
        gripper = np.float32(1.0 if target_state.gripper_closed else 0.0)
        return np.concatenate((arm_target, np.asarray([gripper], dtype=np.float32))), diagnostics

    def hold_action(self, *, env_id: int) -> np.ndarray:
        """Hold measured arm joints and measured binary gripper state."""

        if self._env is None:
            raise RuntimeError("DROID joint-target adapter has not been bound to the RoboLab environment")
        if env_id != 0:
            raise ValueError("DROID bounded EEF control currently supports only env_id 0")
        robot = self._env.scene[self.asset_name]
        joint_names_value = getattr(robot, "joint_names", None)
        joint_names = list(joint_names_value if joint_names_value is not None else robot.data.joint_names)
        if any(joint_names.count(name) != 1 for name in (*ARM_JOINT_NAMES, "finger_joint")):
            raise DroidEEFControlError("DROID joint names do not match the bounded EEF controller")
        positions = _array(
            robot.data.joint_pos[env_id], shape=(len(joint_names),), name="robot joint positions"
        )
        arm = positions[[joint_names.index(name) for name in ARM_JOINT_NAMES]].astype(np.float32)
        gripper_closed = positions[joint_names.index("finger_joint")] > math.pi / 8
        return np.concatenate((arm, np.asarray([float(gripper_closed)], dtype=np.float32)))


__all__ = [
    "DroidEEFControlError",
    "DroidJointTargetAdapter",
    "compute_dls_joint_target",
]
