# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for multi-articulation robot-cfg label resolution."""

import pytest

from robolab.core.environments.base import create_recorder_config, resolve_ee_body_label
from robolab.core.task.conditionals import _resolve_gripper_closure_entry


def test_ee_body_label_bare_string_targets_primary_robot():
    assert resolve_ee_body_label("base_link") == ("robot", "base_link")


def test_ee_body_label_dict_selects_articulation():
    assert resolve_ee_body_label({"asset": "left_robot", "body": "link_6"}) == ("left_robot", "link_6")


def test_ee_body_label_dict_asset_defaults_to_primary_robot():
    assert resolve_ee_body_label({"body": "link_6"}) == ("robot", "link_6")


def test_ee_body_label_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown ee_recorder_bodies keys"):
        resolve_ee_body_label({"asset": "left_robot", "body": "link_6", "bodyy": "typo"})


def test_ee_body_label_requires_body():
    with pytest.raises(ValueError, match="missing 'body'"):
        resolve_ee_body_label({"asset": "left_robot"})


def test_ee_body_label_rejects_other_types():
    with pytest.raises(TypeError, match="must be str or dict"):
        resolve_ee_body_label(("left_robot", "link_6"))


def test_recorder_config_builds_one_term_per_channel_with_articulation():
    config = create_recorder_config(
        ee_recorder_bodies={
            "left_ee_pose": {"asset": "left_robot", "body": "link_6"},
            "right_ee_pose": "link_6",
        }
    )

    left_term = config.record_left_ee_pose
    right_term = config.record_right_ee_pose
    assert (left_term.robot_cfg_name, left_term.ee_body_name, left_term.record_key) == (
        "left_robot",
        "link_6",
        "left_ee_pose",
    )
    assert (right_term.robot_cfg_name, right_term.ee_body_name, right_term.record_key) == (
        "robot",
        "link_6",
        "right_ee_pose",
    )


def test_recorder_config_empty_label_disables_ee_recording():
    config = create_recorder_config(ee_recorder_bodies={})
    assert not any(key.endswith("ee_pose") for key in vars(config))


def test_gripper_closure_legacy_tuple_targets_default_asset():
    resolved = _resolve_gripper_closure_entry(("finger_joint", 0.0, 0.8), "robot")
    assert resolved == ("robot", "finger_joint", 0.0, 0.8, False)


def test_gripper_closure_dict_selects_articulation_and_is_explicit():
    entry = {"asset": "left_robot", "joint": "left_finger", "open": -0.0475, "closed": 0.0}
    resolved = _resolve_gripper_closure_entry(entry, "robot")
    assert resolved == ("left_robot", "left_finger", -0.0475, 0.0, True)


def test_gripper_closure_dict_asset_defaults_to_default_asset():
    entry = {"joint": "left_finger", "open": -0.0475, "closed": 0.0}
    assert _resolve_gripper_closure_entry(entry, "robot")[0] == "robot"


def test_gripper_closure_dict_rejects_unknown_and_missing_keys():
    with pytest.raises(ValueError, match="Unknown gripper_closure_cfg keys"):
        _resolve_gripper_closure_entry({"joint": "j", "open": 0.0, "closed": 1.0, "aset": "x"}, "robot")
    with pytest.raises(ValueError, match="missing"):
        _resolve_gripper_closure_entry({"asset": "left_robot", "joint": "j", "open": 0.0}, "robot")
