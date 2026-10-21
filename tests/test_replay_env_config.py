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

from functools import partial

from robolab.core.replay.env_config import apply_recorded_env_cfg
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
