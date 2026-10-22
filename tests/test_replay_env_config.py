# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the recorded-env-config overlay used by replay.

Covers the ``Subtask.conditions`` skip added to fix
https://github.com/NVlabs/RoboLab/issues/11: conditions are serialized in
``env_cfg.json`` as ``str(functools.partial(...))`` reprs, which are not
dotted import paths and can't be resolved back into callables. The overlay
must leave the freshly built config's callables in place instead of
clobbering them with unresolvable strings. No simulation is stepped.
"""

import json
from copy import deepcopy
from functools import partial
from types import SimpleNamespace

import pytest

from robolab.core.replay.env_config import apply_recorded_env_cfg
from robolab.core.utils import isaaclab_compat as compat
from robolab.core.utils.function_loader import func_as_str, get_callable_info


def _predicate(object: str) -> bool:
    return True


def test_overlay_skips_subtask_conditions_and_keeps_current_callables():
    current_condition = partial(_predicate, object="rubiks_cube")
    env_cfg = {
        "subtasks": [
            {
                "conditions": {
                    "rubiks_cube": [[current_condition, 1.0]],
                },
                "score": 1.0,
                "logical": "all",
                "K": None,
                "name": "pick_and_place",
            }
        ],
    }
    recorded = {
        "subtasks": [
            {
                "conditions": {
                    "rubiks_cube": [
                        ["functools.partial(<function _predicate at 0x7f0000000000>, object='rubiks_cube')", 1.0]
                    ],
                },
                "score": 1.0,
                "logical": "all",
                "K": None,
                "name": "pick_and_place",
            }
        ],
    }

    skipped = apply_recorded_env_cfg(env_cfg, recorded)

    condition_func = env_cfg["subtasks"][0]["conditions"]["rubiks_cube"][0][0]
    assert callable(condition_func)
    assert condition_func is current_condition
    assert any("conditions" in s for s in skipped)


def test_overlay_still_updates_sibling_subtask_fields():
    env_cfg = {
        "subtasks": [
            {
                "conditions": {"rubiks_cube": [[partial(_predicate, object="rubiks_cube"), 1.0]]},
                "score": 1.0,
                "logical": "all",
                "K": None,
                "name": "pick_and_place",
            }
        ],
    }
    recorded = {
        "subtasks": [
            {
                "conditions": {
                    "rubiks_cube": [
                        ["functools.partial(<function _predicate at 0x7f0000000000>, object='rubiks_cube')", 1.0]
                    ],
                },
                "score": 0.5,
                "logical": "any",
                "K": None,
                "name": "pick_and_place",
            }
        ],
    }

    apply_recorded_env_cfg(env_cfg, recorded)

    subtask = env_cfg["subtasks"][0]
    assert subtask["score"] == 0.5
    assert subtask["logical"] == "any"


def test_get_callable_info_tolerates_unresolved_string():
    func_name, params = get_callable_info("functools.partial(<function _predicate at 0x0>, object='banana')")
    assert func_name == "functools.partial(<function _predicate at 0x0>, object='banana')"
    assert params == {}
    assert func_as_str("some_unresolved_string") == "some_unresolved_string()"


def test_overlay_replaces_instruction_fields_without_merging_current_variants():
    env_cfg = {
        "instruction": {
            "default": "Do the task",
            "vague": "Do it",
            "specific": "Do the exact task",
        }
    }
    recorded = {"instruction": {"vague": "Do it", "video": "demo.mp4"}}

    apply_recorded_env_cfg(env_cfg, recorded)

    assert env_cfg["instruction"] == {"vague": "Do it", "video": "demo.mp4"}


@pytest.mark.parametrize("uses_xyzw", [False, True])
def test_recorded_rotations_round_trip_at_replay_boundary(monkeypatch, uses_xyzw):
    monkeypatch.setattr(compat, "ISAACLAB_USES_XYZW", uses_xyzw)
    native = compat.quat_wxyz_to_isaaclab((.5, .5, -.5, .5))
    identity = compat.quat_wxyz_to_isaaclab((1., 0., 0., 0.))
    values = {
        "scene": {
            "robot": {"init_state": {"rot": native}},
            "camera": {"offset": {"rot": native}},
            "frames": {"target_frames": [{"offset": {"rot": native}}]},
            "light": {"spawn": {"color": [.2, .3, .4, .5]}},
        },
        "actions": {"arm": {"body_offset": {"rot": identity}}},
        "instruction": {"default": "pick"},
    }
    exported = compat.env_cfg_to_recording_dict(SimpleNamespace(to_dict=lambda: deepcopy(values)))
    recorded = json.loads(json.dumps(exported))
    saved = deepcopy(recorded)
    current = deepcopy(values)
    # Start from a different native pose so the overlay has to restore it.
    current["scene"]["robot"]["init_state"]["rot"] = identity
    for _ in range(2):
        assert apply_recorded_env_cfg(current, recorded) == []
        assert current == values
        assert recorded == saved


def test_legacy_wxyz_config_is_converted_before_overlay(monkeypatch):
    monkeypatch.setattr(compat, "ISAACLAB_USES_XYZW", True)
    current = SimpleNamespace(scene=SimpleNamespace(robot=SimpleNamespace(
        init_state=SimpleNamespace(rot=(.5, .5, .5, .5)))))
    # No version/format marker was present in historical recordings.
    recorded = {"scene": {"robot": {"init_state": {"rot": [1., 0., 0., 0.]}}}}
    assert apply_recorded_env_cfg(current, recorded) == []
    assert current.scene.robot.init_state.rot == (0., 0., 0., 1.)
