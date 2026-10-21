# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the counting predicates and conditionals.

Covers the pure math (``count_matching_objects``, ``compare_count``), the
candidate-set resolution used by the count conditionals, and the
``verify_task_valid`` acceptance of ``object=None`` termination params.
No simulation is stepped — the world is a ``WorldState`` around a mock env.
"""

from types import SimpleNamespace

import pytest
import torch

from robolab.core.task.conditionals import _resolve_count_candidates
from robolab.core.task.predicate_logic import compare_count, count_matching_objects
from robolab.core.world.world_state import WorldState


def make_world(num_envs: int = 4) -> WorldState:
    return WorldState(SimpleNamespace(num_envs=num_envs, device="cpu"))


#########################################################
# count_matching_objects
#########################################################

def test_count_scalar_path():
    world = make_world()
    inside = {"a", "c"}
    count = count_matching_objects(
        world, ["a", "b", "c"],
        lambda w, obj, env_id=None: obj in inside,
        env_id=0,
    )
    assert count == 2
    assert isinstance(count, int)


def test_count_scalar_empty_candidates():
    world = make_world()
    assert count_matching_objects(world, [], lambda w, o, env_id=None: True, env_id=0) == 0


def test_count_vectorized_path():
    world = make_world(num_envs=3)
    # Per-object, per-env membership: 'a' in envs {0,1}, 'b' in env {1}, 'c' nowhere
    membership = {
        "a": torch.tensor([True, True, False]),
        "b": torch.tensor([False, True, False]),
        "c": torch.tensor([False, False, False]),
    }
    count = count_matching_objects(
        world, ["a", "b", "c"],
        lambda w, obj, env_id=None: membership[obj],
        env_id=None,
    )
    assert count.dtype == torch.long
    assert count.tolist() == [1, 2, 0]


def test_count_vectorized_empty_candidates():
    world = make_world(num_envs=5)
    count = count_matching_objects(world, [], lambda w, o, env_id=None: None, env_id=None)
    assert count.dtype == torch.long
    assert count.tolist() == [0, 0, 0, 0, 0]


#########################################################
# compare_count
#########################################################

@pytest.mark.parametrize("comparison,expected", [
    ("==", False), (">=", True), ("<=", False), (">", True), ("<", False),
])
def test_compare_count_scalar(comparison, expected):
    assert compare_count(3, 2, comparison) is expected


def test_compare_count_scalar_equal():
    assert compare_count(2, 2, "==") is True
    assert compare_count(2, 2, ">=") is True
    assert compare_count(2, 2, "<=") is True
    assert compare_count(2, 2, ">") is False


def test_compare_count_tensor():
    counts = torch.tensor([0, 2, 3], dtype=torch.long)
    assert compare_count(counts, 2, "==").tolist() == [False, True, False]
    assert compare_count(counts, 2, ">=").tolist() == [False, True, True]
    assert compare_count(counts, 2, "<").tolist() == [True, False, False]


def test_compare_count_invalid_comparison():
    with pytest.raises(ValueError, match="Invalid comparison"):
        compare_count(1, 1, "!=")


#########################################################
# _resolve_count_candidates
#########################################################

def make_env_with_contacts(contact_object_list):
    return SimpleNamespace(cfg=SimpleNamespace(contact_object_list=contact_object_list))


def test_resolve_candidates_default_excludes_ignored():
    env = make_env_with_contacts(["banana", "apple", "grey_bin", "table"])
    candidates = _resolve_count_candidates(env, None, ["table", "grey_bin"])
    assert candidates == ["banana", "apple"]


def test_resolve_candidates_explicit_str_and_list():
    env = make_env_with_contacts(["banana", "table"])
    assert _resolve_count_candidates(env, "banana", ["table"]) == ["banana"]
    # Explicit lists are used as-is, even if they overlap ignore_objects.
    assert _resolve_count_candidates(env, ["banana", "apple"], ["table"]) == ["banana", "apple"]


#########################################################
# verify_task_valid accepts object=None termination params
#########################################################

def test_verify_callable_args_none_object_skipped():
    """The object-in-contact-list loop in verify_task_valid must skip None."""
    from isaaclab.managers import TerminationTermCfg as DoneTerm

    from robolab.core.task.conditionals import object_count_in_container
    from robolab.core.task.task import Task, verify_task_valid

    class Terminations:
        success = DoneTerm(
            func=object_count_in_container,
            params={"container": "grey_bin", "count": 2, "object": None},
        )

    class SceneStub:
        __annotations__ = {"grey_bin": None, "table": None}

    class CountTask(Task):
        contact_object_list = ["grey_bin", "table"]
        scene = SceneStub()
        terminations = Terminations
        instruction = "count things"
        episode_length_s = 60

    valid, error = verify_task_valid(CountTask)
    assert valid, error


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
