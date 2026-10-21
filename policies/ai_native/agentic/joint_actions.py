# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Absolute joint-target alternative with the same one-to-five-step decision cadence."""

from __future__ import annotations

import numpy as np

from policies.ai_native.common.agent_eef import AgentEEFPlan, agent_eef_state, build_agent_eef_tools
from policies.ai_native.common.prompts import (
    DROID_CAMERA_GEOMETRY_PROMPT,
    DROID_EEF_FRAME_PROMPT,
    DROID_GRIPPER_GEOMETRY_PROMPT,
)


def action_tools(spec, options=None):
    if options is None or options.action_space == "eef":
        return build_agent_eef_tools(spec)
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_joint_target",
                "description": "Hold one complete absolute Panda arm/gripper target for 1 to 5 control steps.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "joint_positions": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 7,
                            "maxItems": 7,
                            "description": "panda_joint1 through panda_joint7, absolute radians.",
                        },
                        "gripper_closed": {"type": "boolean"},
                        "steps": {"type": "integer", "minimum": 1, "maximum": 5},
                        "reason": {"type": "string"},
                    },
                    "required": ["joint_positions", "gripper_closed", "steps", "reason"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def joint_policy_instructions(geometry, *, control_hz=15.0, max_target_steps=5, **control):
    prompt = (
        "Control a simulated Franka Panda arm with a Robotiq 2F-85 gripper. "
        "Use execute_joint_target with seven absolute joint angles in radians in panda_joint1..7 order, "
        f"a boolean gripper_closed, and 1 to {max_target_steps} control steps at {control_hz:g} Hz. "
        "arm_joint_pos contains the current measured angles. The same target is held for the entire segment. "
        "At each step the host clamps the requested joint change to 0.05 rad from measured positions "
        "and to the robot's native joint limits. It does not run IK or wait for arrival. "
        "The gripper command executes every step even if the arm has not arrived. "
        "Simulation pauses during inference. Use measured joint tracking feedback after each segment. "
        "EEF observations and the following geometry are spatial references, not the output action. "
        + DROID_EEF_FRAME_PROMPT
        + " Only RoboLab decides success and termination."
    )
    if geometry:
        prompt += "\n\n" + DROID_GRIPPER_GEOMETRY_PROMPT.replace(
            "The action pose and ee_pos/ee_quat", "The observed ee_pos/ee_quat"
        )
        prompt += "\n\n" + DROID_CAMERA_GEOMETRY_PROMPT
    return prompt


def decode_joint_target(spec, name, arguments, proprio, max_target_steps=5):
    if name != "execute_joint_target" or set(arguments) != {"joint_positions", "gripper_closed", "steps", "reason"}:
        raise ValueError("Expected execute_joint_target with the complete target, duration and reason")
    joints = np.asarray(arguments["joint_positions"], dtype=float)
    steps = arguments["steps"]
    if joints.shape != (7,) or not np.isfinite(joints).all():
        raise ValueError("joint_positions must contain seven finite radians")
    if type(steps) is not int or not 1 <= steps <= max_target_steps:
        raise ValueError("steps must be within the configured control-step limit")
    if type(arguments["gripper_closed"]) is not bool or not isinstance(arguments["reason"], str):
        raise ValueError("gripper_closed must be boolean and reason must be text")
    action = np.r_[joints, float(arguments["gripper_closed"])]
    return AgentEEFPlan(
        actions=np.tile(action, (steps, 1)),
        status="ok",
        tool_name=name,
        note=arguments["reason"],
        feedback=None,
        metadata={"joint_target": action.tolist()},
        commanded_state=agent_eef_state(spec, proprio),
    )


def bounded_joint_action(target, measured, limits, max_delta=0.05):
    target, measured, limits = (np.asarray(x, dtype=float) for x in (target, measured, limits))
    if target.shape != (8,) or measured.shape != (7,) or limits.shape != (7, 2):
        raise ValueError("Invalid joint target, measured state or limits shape")
    if not all(np.isfinite(x).all() for x in (target, measured, limits)) or np.any(limits[:, 0] > limits[:, 1]):
        raise ValueError("Joint control requires finite values and ordered limits")
    joints = np.clip(measured + np.clip(target[:7] - measured, -max_delta, max_delta), limits[:, 0], limits[:, 1])
    return np.r_[joints, target[7]].astype(np.float32), {
        "controller": "absolute_joint_target",
        "requested_joint_target": target[:7].tolist(),
        "measured_joint_positions": measured.tolist(),
        "applied_joint_target": joints.tolist(),
    }
