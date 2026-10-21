# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Built-in embodiment adapters for the shared Agent-EEF controller."""

from __future__ import annotations

from policies.ai_native.common.agent_eef import EffectorSpec, EmbodimentSpec

# Match the reference EEF controller: local pose-step and joint limits, with
# no additional Cartesian box around the gripper mount.
DROID_WORKSPACE_MIN = None
DROID_WORKSPACE_MAX = None


def make_droid_spec(
    *,
    workspace_min=DROID_WORKSPACE_MIN,
    workspace_max=DROID_WORKSPACE_MAX,
) -> EmbodimentSpec:
    return EmbodimentSpec(
        name="droid",
        action_dim=8,
        effectors=(
            EffectorSpec(
                name="arm",
                position_key="ee_pos",
                quaternion_key="ee_quat",
                gripper_key="gripper_pos",
                position_action_indices=(0, 1, 2),
                quaternion_action_indices=(3, 4, 5, 6),
                gripper_action_index=7,
                workspace_min=workspace_min,
                workspace_max=workspace_max,
                root_frame="DROID robot root",
            ),
        ),
    )


__all__ = [
    "DROID_WORKSPACE_MAX",
    "DROID_WORKSPACE_MIN",
    "make_droid_spec",
]
