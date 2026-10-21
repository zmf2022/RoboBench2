# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Write Direct-API evaluation configuration after environment setup."""

import hashlib
import json
from pathlib import Path

import robolab.constants
from policies.ai_native.common.camera_geometry import DROID_CAMERA_CALIBRATION


def write_evaluation_artifacts(client):
    metadata = client.run_metadata
    if metadata is None:
        return
    camera_geometry = client.prompt_family in {"gripper_camera_geometry_prompt", "camera_geometry_prompt"}
    gripper_geometry = client.prompt_family in {"gripper_camera_geometry_prompt", "gripper_geometry_prompt"}
    resolved = {
        "schema_version": 1,
        "output_mode": client.output_mode, "prompt_family": client.prompt_family,
        "model": client.model, "reasoning_effort": client.reasoning_effort,
        "instruction_type": metadata["instruction_type"],
        "history_offsets": client.history_offsets, "history_length": client.history_length,
        "history_stride": client.history_stride, "min_action_horizon": client.min_action_horizon,
        "max_action_horizon": client.max_action_horizon, "bspline_min_steps": client.bspline_min_steps,
        "bspline_control_points": client.bspline_control_points, "bspline_degree": client.bspline_degree,
        "max_completion_tokens": client.max_completion_tokens,
        "image_max_side": client.image_max_side, "image_detail": client.image_detail,
        "jpeg_quality": client.jpeg_quality, "max_api_attempts": client.max_api_attempts,
        "max_retries": client.max_retries, "max_output_retries": client.max_output_retries,
        "timeout_s": client.timeout_s, "max_joint_delta": client.max_joint_delta,
        "max_eef_translation_delta_m": client.max_eef_translation_delta_m,
        "max_eef_rotation_delta_rad": client.max_eef_rotation_delta_rad,
        "camera_names": client.camera_names, "renderer": metadata["renderer"],
        "rendering_type": metadata["rendering_type"],
        "control_dt_s": client.control_dt_s, "num_envs": metadata["num_envs"],
        "num_runs": metadata["num_runs"], "run_index_offset": metadata["run_index_offset"],
        "base_seed": metadata["base_seed"],
        "prompt_sha256": hashlib.sha256(client._method_prompt().encode()).hexdigest(),
    }
    resolved["sha256"] = hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest()
    client.configuration_sha256 = resolved["sha256"]
    config_path = Path(robolab.constants.get_output_dir(), "resolved_config.json")
    if config_path.exists() and json.loads(config_path.read_text()).get("sha256") != resolved["sha256"]:
        raise ValueError("Output folder contains a different resolved configuration; use a new folder")
    config_path.write_text(json.dumps(resolved, indent=2) + "\n")
    if client.prompt_family is not None:
        prompt = client._method_prompt()
        artifact = {
            "variant": client.prompt_family,
            "prompt_family": client.prompt_family,
            "output_mode": client.output_mode,
            "control_dt_s": client.control_dt_s,
            "min_action_horizon": client.min_action_horizon,
            "max_action_horizon": client.max_action_horizon,
            "translation_limit_m": client.max_eef_translation_delta_m,
            "rotation_limit_rad": client.max_eef_rotation_delta_rad,
            "joint_limit_rad": client.max_joint_delta,
            "action_memory_chunks": 0,
            "geometry_prompt_enabled": camera_geometry or gripper_geometry,
            "gripper_geometry_prompt_enabled": gripper_geometry,
            "camera_geometry_prompt_enabled": camera_geometry,
            "camera_calibration": DROID_CAMERA_CALIBRATION if camera_geometry else None,
            "grasp_instructions_removed": True,
            "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "system_prompt": prompt,
        }
        Path(robolab.constants.get_output_dir(), "prompt_variant.json").write_text(
            json.dumps(artifact, indent=2) + "\n"
        )
        print(f"[AI-native] Prompt variant {artifact['variant']}; sha256={artifact['sha256']}")
