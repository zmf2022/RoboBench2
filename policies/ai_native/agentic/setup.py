# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate Agentic environment settings and record the configured prompt."""

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import robolab.constants
from policies.ai_native.agentic.joint_actions import action_tools
from policies.ai_native.agentic.prompts import policy_instructions
from policies.ai_native.common.camera_geometry import DROID_CAMERA_CALIBRATION, validate_droid_camera_calibration


def configure_environment(client, env_cfg):
    family = client.prompt_family
    options = client.options
    expanded = family == "gripper_camera_geometry_prompt"
    control_dt_s = float(env_cfg.sim.dt * env_cfg.decimation)
    if expanded:
        if set(client.camera_names) != set(DROID_CAMERA_CALIBRATION):
            raise ValueError("geometry prompt requires all three calibrated DROID cameras")
        if not math.isclose(control_dt_s, 1 / client.control.control_hz, abs_tol=1e-9):
            raise ValueError("Agent-EEF prompt control rate does not match simulator timing")
        validate_droid_camera_calibration(env_cfg.to_dict())
        adapter = client.execution_adapter
        ik_limits = (adapter.max_translation_step_m, adapter.max_rotation_step_rad, adapter.max_joint_delta_rad)
        if ik_limits != (0.02, 0.1, 0.05):
            raise ValueError("Agent-EEF geometry timing prompt must match native IK limits")

    control = dict(
        control_hz=client.control.control_hz,
        max_target_steps=client.control.max_target_steps,
        max_translation_m=client.control.max_translation_m,
        max_rotation_rad=client.control.max_rotation_rad,
    )
    prompt = policy_instructions(options, client.spec, prompt_family=family, **control)
    geometry = expanded and (options is None or options.geometry)
    joint_targets = options is not None and options.action_space == "joint"
    artifact = {
        "variant": family,
        "prompt_family": family,
        "output_mode": "agent-joint" if joint_targets else "agent-eef",
        "control": asdict(client.control),
        "control_dt_s": control_dt_s,
        "camera_calibration": DROID_CAMERA_CALIBRATION if geometry else None,
        "grasp_instructions_removed": expanded,
        "geometry_prompt_enabled": geometry,
        "camera_geometry_prompt_enabled": geometry,
        "geometry_state_fields": ["ee_pos", "ee_quat", "gripper_pos"] if expanded else [],
        "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "system_prompt": prompt,
        "prompt_scope": "Base policy instructions repeated each turn; native App Server memory/tool instructions also apply.",
        "output_tools": action_tools(client.spec, options),
        "runtime": client.runtime_settings,
    }
    if options:
        artifact["options"] = options.to_dict()
    output = Path(robolab.constants.get_output_dir())
    output.mkdir(parents=True, exist_ok=True)
    (output / "prompt_variant.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (output / "system_prompt.txt").write_text(prompt)
    print(f"[AI-native] Prompt variant {family}; sha256={artifact['sha256']}")
