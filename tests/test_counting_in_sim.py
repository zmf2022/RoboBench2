# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""In-sim tests for the counting conditionals against an existing benchmark scene.

Boots ``BananasInBinThreeTotalTask`` (5 bananas + grey bin + table), settles the
scene, and verifies:

- the count agrees with per-object ``object_in_container`` booleans,
- positive and negative triggering for every comparison operator,
- ``object=None`` counts the same set as the explicit banana list,
- teleporting a banana into the bin raises the count by one (dynamic trigger),
- surface counting on the table drops by one after the teleport.

Requires CUDA (same as test_run_empty). One env, no policy.
"""

import pytest
import torch

from robolab.core.environments.factory import get_envs
from robolab.core.environments.runtime import create_env, end_episode
from robolab.core.task.conditionals import (
    get_object_count_in_container,
    get_object_count_on_surface,
    object_count_in_container,
    object_count_on_surface,
    object_in_container,
)
from robolab.core.world.world_state import get_world
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

BANANAS = ["banana", "banana_01", "banana_02", "banana_03", "banana_04"]
BIN = "grey_bin"


@pytest.fixture(scope="module")
def sim_env():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required for in-sim counting tests")
    auto_register_droid_envs()
    envs = get_envs(task="BananasInBinThreeTotalTask")
    assert envs, "BananasInBinThreeTotalTask has no registered env"
    env, _ = create_env(envs[0], num_envs=1, use_fabric=True)
    try:
        env.reset()
        _settle(env, steps=20)
        yield env
        end_episode(env)
    finally:
        env.close()


def _settle(env, steps: int):
    actions = torch.zeros((env.num_envs, env.action_manager.total_action_dim), device=env.device)
    for _ in range(steps):
        env.step(actions)


def test_count_matches_boolean_conditionals(sim_env):
    """The count must equal the number of per-banana object_in_container Trues."""
    expected = sum(
        bool(object_in_container(sim_env, object=b, container=BIN, env_id=0)) for b in BANANAS
    )
    count_scalar = get_object_count_in_container(sim_env, BIN, object=BANANAS, env_id=0)
    count_vec = get_object_count_in_container(sim_env, BIN, object=BANANAS, env_id=None)
    assert count_scalar == expected
    assert count_vec.shape == (1,)
    assert int(count_vec[0]) == expected


def test_default_candidates_match_explicit_list(sim_env):
    """object=None (contact list minus table/bin) must count the same as the banana list."""
    implicit = get_object_count_in_container(sim_env, BIN, object=None, env_id=0)
    explicit = get_object_count_in_container(sim_env, BIN, object=BANANAS, env_id=0)
    assert implicit == explicit


def test_conditional_triggers_positive_and_negative(sim_env):
    """Every comparison must trigger correctly around the observed count k."""
    k = get_object_count_in_container(sim_env, BIN, object=None, env_id=0)
    assert 0 <= k <= len(BANANAS)

    # Positive triggers
    assert object_count_in_container(sim_env, BIN, count=k, comparison="==", env_id=0)
    assert object_count_in_container(sim_env, BIN, count=k, comparison=">=", env_id=0)
    assert object_count_in_container(sim_env, BIN, count=k, comparison="<=", env_id=0)
    assert object_count_in_container(sim_env, BIN, count=k + 1, comparison="<", env_id=0)

    # Negative triggers
    assert not object_count_in_container(sim_env, BIN, count=k + 1, comparison="==", env_id=0)
    assert not object_count_in_container(sim_env, BIN, count=k + 1, comparison=">=", env_id=0)
    assert not object_count_in_container(sim_env, BIN, count=k, comparison=">", env_id=0)
    assert not object_count_in_container(sim_env, BIN, count=k, comparison="<", env_id=0)

    # Vectorized (termination-manager) path agrees with the scalar path.
    vec = object_count_in_container(sim_env, BIN, count=k, comparison="==", env_id=None)
    assert vec.dtype == torch.bool and vec.shape == (1,)
    assert bool(vec[0])


def test_teleport_into_bin_raises_count(sim_env):
    """Dynamic trigger: dropping an outside banana into the bin bumps the count by one."""
    k_before = get_object_count_in_container(sim_env, BIN, object=None, env_id=0)
    on_table_before = get_object_count_on_surface(sim_env, "table", object=BANANAS, env_id=0)

    outside = [
        b for b in BANANAS
        if not bool(object_in_container(sim_env, object=b, container=BIN, env_id=0))
    ]
    assert outside, "expected at least one banana outside the bin in the initial scene"
    mover = outside[0]

    # Teleport the banana above the bin center and let it drop in.
    world = get_world(sim_env)
    bin_pos, _ = world.get_pose(BIN, env_id=None)  # (1, 3) world frame, meters
    banana = sim_env.scene[mover]
    pose = torch.zeros((1, 7), device=sim_env.device)
    pose[:, :3] = bin_pos + torch.tensor([0.0, 0.0, 0.15], device=sim_env.device)
    pose[:, 3] = 1.0  # identity quaternion, w-x-y-z
    banana.write_root_pose_to_sim(pose)
    banana.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim_env.device))
    _settle(sim_env, steps=30)

    k_after = get_object_count_in_container(sim_env, BIN, object=None, env_id=0)
    assert k_after == k_before + 1

    # The == trigger flips: old count now False, new count now True.
    assert not object_count_in_container(sim_env, BIN, count=k_before, comparison="==", env_id=0)
    assert object_count_in_container(sim_env, BIN, count=k_before + 1, comparison="==", env_id=0)

    # Surface count on the table lost exactly the moved banana.
    on_table_after = get_object_count_on_surface(sim_env, "table", object=BANANAS, env_id=0)
    assert on_table_after == on_table_before - 1
    assert object_count_on_surface(
        sim_env, "table", count=on_table_after, comparison="==", object=BANANAS, env_id=0
    )
    assert not object_count_on_surface(
        sim_env, "table", count=on_table_before, comparison="==", object=BANANAS, env_id=0
    )
