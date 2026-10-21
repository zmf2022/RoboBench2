# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve Agentic policy and tool instructions for setup and live requests."""

from __future__ import annotations

from policies.ai_native.agentic.coding_workspace import CODING_GUIDANCE
from policies.ai_native.agentic.joint_actions import joint_policy_instructions
from policies.ai_native.agentic.runtime_options import RuntimeOptions
from policies.ai_native.common.agent_eef import EmbodimentSpec, agent_eef_policy_instructions
from policies.ai_native.common.prompts import DROID_CAMERA_GEOMETRY_PROMPT, DROID_GRIPPER_GEOMETRY_PROMPT

PLANNING_INSTRUCTIONS = (
    "Before acting, break the task into a short sequence of subgoals. "
    "Track which subgoal is current and revise the plan using observed outcomes."
)


def policy_instructions(
    config: RuntimeOptions | None, spec: EmbodimentSpec, *, prompt_family: str = "baseline", **control,
) -> str:
    if config is None:
        return agent_eef_policy_instructions(spec, prompt_family=prompt_family, **control)
    prompt = agent_eef_policy_instructions(spec, prompt_family=prompt_family, **control)
    if config.action_space == "joint":
        prompt = joint_policy_instructions(config.geometry and prompt_family == "gripper_camera_geometry_prompt", **control)
    if not config.geometry:
        prompt = prompt.replace(DROID_GRIPPER_GEOMETRY_PROMPT, "").replace(DROID_CAMERA_GEOMETRY_PROMPT, "")
    if config.explicit_planning:
        prompt += "\n\n" + PLANNING_INSTRUCTIONS
    if not config.decomposition_scaffolding:
        prompt += (
            "\n\nChoose the next movement directly from the current observations, task goal and feedback. "
            "Do not write multi-step plans, subgoals, current objectives, future-action sequences or checklists. "
            "Notes may contain only measurements, observed events and outcomes of executed actions. "
            "Explain only the immediate action. Conversation history remains available."
        )
    return prompt


def tool_instructions(config: RuntimeOptions | None) -> str:
    if config is None:
        return (
            "Built-in shell, filesystem, web, MCP, image-view, delegation, and code-execution tools are disabled. "
            "Read-only memory_* tools may be called before the one robot-action tool."
        )
    memory = (
        "Episode memory_* tools can retrieve observations, action history and notes. "
        if config.memory_tools
        else "Only the supplied sampled observations and current feedback are available; no episode archive or notes tools. "
    )
    if config.history_channels:
        memory = (
            "Only the conversation and observations supplied in this request are available. "
            "Past user-state and motion-feedback text is omitted; current measured state and feedback are provided. "
            "Episode archives, retrieval tools and persistent notes are disabled. "
        )
    optional = (
        CODING_GUIDANCE
        if config.optional_tools
        else "Built-in shell/code and image-inspection tools are disabled. "
    )
    return (
        memory + optional + "Web, MCP, delegation and planning tools are disabled. Use the robot-action tool to move."
    )
