# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-effector specifications used by Direct-API action chunks."""

from __future__ import annotations

from policies.ai_native.common.agent_eef import EffectorSpec, EmbodimentSpec

DROID_SPEC = EmbodimentSpec(
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
            root_frame="DROID robot root",
        ),
    ),
)


def get_eef_spec(name: str) -> EmbodimentSpec:
    if name != DROID_SPEC.name:
        raise ValueError(f"unknown AI-native EEF embodiment {name!r}; expected {DROID_SPEC.name!r}")
    return DROID_SPEC


__all__ = ["DROID_SPEC", "get_eef_spec"]
