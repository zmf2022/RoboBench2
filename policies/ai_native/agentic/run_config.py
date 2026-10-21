# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Simulator-independent CLI and factories for AI-native Codex Agent-EEF."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import robolab.constants
from policies.ai_native.agentic.app_server import ASTRA_MODEL_IDS, resolve_agent_memory_root
from policies.ai_native.agentic.client import AgentEEFControlConfig, CodexAgentEEFClient
from policies.ai_native.agentic.droid_joint_target import DroidJointTargetAdapter
from policies.ai_native.agentic.runtime_options import RuntimeOptions
from policies.ai_native.agentic.session import CodexAgentEEFSession
from policies.ai_native.common.agent_eef import (
    AGENT_EEF_PROMPT_FAMILIES,
    EmbodimentSpec,
    validate_agent_eef_prompt_family,
)

DEFAULT_CAMERAS = {
    "droid": ("over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam"),
}
DROID_CAMERA_PRESETS = (
    ("WRIST", ("wrist_cam",)),
    ("LEFT_RIGHT", ("over_shoulder_left_camera", "over_shoulder_right_camera")),
    ("WRIST_LEFT", ("over_shoulder_left_camera", "wrist_cam")),
    ("WRIST_RIGHT", ("over_shoulder_right_camera", "wrist_cam")),
    ("WRIST_LEFT_RIGHT", ("over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam")),
)


def add_agent_eef_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--action-space", choices=("eef", "joint"), default="eef")
    parser.add_argument("--history-offsets", nargs="+", type=int, default=None)
    parser.add_argument("--history-text", choices=("none", "full", "sampled"), default="none")
    parser.add_argument("--history-images", choices=("none", "full"), default="none")
    for name, default, help_text in (
        ("motion-feedback", True, "Include measured motion-tracking feedback."),
        ("optional-tools", False, "Enable shell/Python and image-inspection tools."),
        ("persistent-history", True, "Keep the conversation and episode memory between decisions."),
        ("geometry", True, "Include geometry descriptions provided by the selected prompt family."),
        ("explicit-planning", False, "Ask the model to state and track a plan."),
        ("decomposition-scaffolding", True, "Allow written subgoals and plans."),
        ("include-joint-state", False, "Include measured DROID arm joint angles in the input."),
    ):
        parser.add_argument("--" + name, action=argparse.BooleanOptionalAction, default=default, help=help_text)
    parser.add_argument('--demonstration-manifest', type=Path, default=None)
    parser.add_argument("--prompt-family", choices=AGENT_EEF_PROMPT_FAMILIES,
                        default="gripper_camera_geometry_prompt")
    parser.add_argument("--agent-style", choices=["agent-eef"], default="agent-eef")
    parser.add_argument(
        "--history-length",
        "--history_length",
        type=int,
        default=1,
        help="Maximum latest synchronized image/proprioception observations sent per inference call.",
    )
    parser.add_argument("--embodiment", choices=sorted(DEFAULT_CAMERAS), default="droid")
    parser.add_argument("--model", choices=sorted(ASTRA_MODEL_IDS), default="gpt-6-astra-medium")
    parser.add_argument("--allow-model-requests", "--allow_model_requests", action="store_true")
    parser.add_argument("--codex-bin", "--codex_bin", default="codex")
    parser.add_argument("--decision-timeout", "--decision_timeout", type=float, default=120.0)
    parser.add_argument("--image-size", "--image_size", type=int, default=640)
    parser.add_argument("--jpeg-quality", "--jpeg_quality", type=int, default=85)
    parser.add_argument("--camera", nargs="+", default=None)
    parser.add_argument("--control-hz", "--control_hz", type=float, default=15.0)
    parser.add_argument("--max-target-steps", "--max_target_steps", type=int, default=5)
    parser.add_argument("--max-invalid-actions", "--max_invalid_actions", type=int, default=3)
    parser.add_argument("--max-llm-calls", "--max_llm_calls", type=int, default=150)


def build_runtime_options(args: argparse.Namespace) -> RuntimeOptions:
    options = RuntimeOptions(
        action_space=args.action_space,
        observation_offsets=tuple(args.history_offsets) if args.history_offsets is not None else None,
        history_text=args.history_text,
        history_images=args.history_images,
        motion_feedback=args.motion_feedback,
        optional_tools=args.optional_tools,
        persistent_history=args.persistent_history,
        geometry=args.geometry,
        explicit_planning=args.explicit_planning,
        decomposition_scaffolding=args.decomposition_scaffolding,
        include_joint_state=args.include_joint_state,
        image_demonstration=args.demonstration_manifest is not None,
    )
    if options.joint_state and args.embodiment != "droid":
        raise ValueError("Joint targets and joint-state input require DROID")
    if options.history_images == "full" and args.history_length != 1:
        raise ValueError("Full image history requires --history-length 1")
    return options


def camera_names(args: argparse.Namespace) -> tuple[str, ...]:
    names = tuple(args.camera or DEFAULT_CAMERAS[args.embodiment])
    if len(names) != 3 or len(set(names)) != 3 or not all(isinstance(name, str) and name for name in names):
        raise ValueError("agent-eef requires exactly three unique camera names")
    return names


def resolve_droid_camera_preset(names: tuple[str, ...]) -> str:
    """Name the smallest DROID camera preset covering the requested views."""
    requested = set(names)
    known = {name for _, cameras in DROID_CAMERA_PRESETS for name in cameras}
    unknown = sorted(requested - known)
    if unknown:
        raise ValueError(f"Unknown DROID Agent-EEF cameras: {', '.join(unknown)}")
    for preset, cameras in sorted(DROID_CAMERA_PRESETS, key=lambda item: len(item[1])):
        if requested <= set(cameras):
            return preset
    raise ValueError("No DROID camera preset covers the requested cameras")


def build_control_config(args: argparse.Namespace) -> AgentEEFControlConfig:
    return AgentEEFControlConfig(
        control_hz=args.control_hz,
        max_target_steps=args.max_target_steps,
        max_invalid_actions=args.max_invalid_actions,
        max_llm_calls=args.max_llm_calls,
    )


def build_session_factory(
    args: argparse.Namespace,
    *,
    spec: EmbodimentSpec,
    session_class=CodexAgentEEFSession,
) -> Callable[[int, int], object]:
    if args.model not in ASTRA_MODEL_IDS:
        raise ValueError("Use an exact supported Astra model ID")
    if args.allow_model_requests is not True:
        raise ValueError("Real model requests require --allow-model-requests")
    if type(args.history_length) is not int or args.history_length < 1:
        raise ValueError("history_length must be a positive integer")
    memory_root, memory_root_source = resolve_agent_memory_root(
        default_output_dir=robolab.constants.get_output_dir(),
    )

    def factory(env_id: int, episode_idx: int):
        return session_class(
            spec=spec,
            model_id=args.model,
            allow_model_requests=True,
            codex_bin=args.codex_bin,
            timeout=args.decision_timeout,
            image_size=args.image_size,
            jpeg_quality=args.jpeg_quality,
            memory_root=memory_root,
            memory_root_source=memory_root_source,
            policy_name=f"agent-eef-{spec.name}",
            env_id=env_id,
            episode_idx=episode_idx,
            prompt_family=args.prompt_family,
            control_hz=args.control_hz,
            max_target_steps=args.max_target_steps,
            options=build_runtime_options(args),
            demonstration_path=getattr(args, 'demonstration_manifest', None),
        )

    return factory


def build_client(
    args: argparse.Namespace,
    *,
    spec: EmbodimentSpec,
    session_class=CodexAgentEEFSession,
) -> CodexAgentEEFClient:
    options = build_runtime_options(args)
    family = args.prompt_family
    validate_agent_eef_prompt_family(spec, family)
    control = build_control_config(args)
    adapter = DroidJointTargetAdapter() if spec.name == "droid" else None
    return CodexAgentEEFClient(
        spec=spec,
        session_factory=build_session_factory(args, spec=spec, session_class=session_class),
        camera_names=camera_names(args),
        control=control,
        history_length=len(options.observation_offsets) if options.observation_offsets else args.history_length,
        execution_adapter=adapter,
        prompt_family=family,
        options=options,
        runtime_settings={key: getattr(args, key) for key in (
            "model", "history_length", "max_llm_calls", "decision_timeout", "max_invalid_actions",
            "image_size", "jpeg_quality", "control_hz", "max_target_steps",
        )},
    )


__all__ = [
    "DEFAULT_CAMERAS",
    "add_agent_eef_args",
    "build_client",
    "build_control_config",
    "build_runtime_options",
    "build_session_factory",
    "camera_names",
    "resolve_droid_camera_preset",
]
