# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every multi-arm public EE pose stays robot-root XYZ + WXYZ on Lab 3."""

from importlib import import_module
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("name", ["aloha", "bimanual_station", "bimanual_yam", "bimanual_franka"])
@pytest.mark.parametrize("proxy", [False, True])
@pytest.mark.parametrize("identity", [False, True])
def test_both_arm_observations_use_public_wxyz(name, proxy, identity):
    module = import_module(f"robolab.robots.{name}")
    left, right = ("left_arm", "right_arm") if name == "bimanual_franka" else ("left", "right")
    h = 2 ** -.5
    root_pos = torch.tensor([[1., 2., 3.], [-1., 1., 0.]])
    root_quat = torch.tensor([[0., 0., h, h], [0., h, 0., h]])
    body_pos = torch.tensor([[[.9, 2.3, 3.2], [.6, 1.8, 3.1]],
                             [[-.8, 1.1, -.3], [-.9, 1.4, .2]]])
    body_quat = torch.tensor([[[0., 0., h, h], [.5, .5, .5, .5]],
                              [[0., h, 0., h], [.5, .5, -.5, .5]]])
    expected_left_pos = torch.tensor([[.3, .1, .2], [.3, .1, .2]])
    expected_right_pos = torch.tensor([[-.2, .4, .1], [-.2, .4, .1]])
    expected_left_quat = torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1)
    expected_right_quat = torch.tensor([[h, h, 0., 0.]]).repeat(2, 1)
    if identity:
        root_pos.zero_()
        root_quat[:] = torch.tensor([0., 0., 0., 1.])
        body_pos[:, 0] = expected_left_pos
        body_pos[:, 1] = expected_right_pos
        body_quat[:] = torch.tensor([0., 0., 0., 1.])
        expected_right_quat = expected_left_quat.clone()
    wrap = (lambda x: SimpleNamespace(torch=x)) if proxy else (lambda x: x)
    data = SimpleNamespace(
        body_names=[module.EE_BODY_NAME[left], module.EE_BODY_NAME[right]],
        root_pos_w=wrap(root_pos), root_quat_w=wrap(root_quat),
        body_pos_w=wrap(body_pos), body_quat_w=wrap(body_quat),
    )
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data)})
    before = body_quat.clone()
    observation_type = (
        module.BimanualProprioceptionObservationCfg if name == "bimanual_franka"
        else module.BimanualYamProprioceptionObservationCfg if name == "bimanual_yam"
        else module.AlohaProprioceptionObservationCfg
    )
    observations = observation_type()
    torch.testing.assert_close(observations.left_ee_pos.func(env), expected_left_pos, atol=1e-6, rtol=0)
    torch.testing.assert_close(observations.right_ee_pos.func(env), expected_right_pos, atol=1e-6, rtol=0)
    torch.testing.assert_close(observations.left_ee_quat.func(env), expected_left_quat, atol=1e-6, rtol=0)
    torch.testing.assert_close(observations.right_ee_quat.func(env), expected_right_quat, atol=1e-6, rtol=0)
    torch.testing.assert_close(body_quat, before)


@pytest.mark.parametrize("name", ["droid", "franka", "kinova_gen3"])
@pytest.mark.parametrize("proxy", [False, True])
def test_frame_observations_are_robot_root_not_world(name, proxy):
    module = import_module(f"robolab.robots.{name}")
    h = 2 ** -.5
    wrap = (lambda x: SimpleNamespace(torch=x)) if proxy else (lambda x: x)
    root_pos = torch.tensor([[1., 2., 3.], [-1., 1., 0.]])
    root_quat = torch.tensor([[0., 0., h, h], [0., h, 0., h]])
    world_pos = torch.tensor([[[.9, 2.3, 3.2]], [[-.8, 1.1, -.3]]])
    frames = SimpleNamespace(data=SimpleNamespace(
        target_frame_names=["eef_frame"], target_pos_w=wrap(world_pos),
        target_quat_w=wrap(root_quat[:, None]),
    ))
    robot = SimpleNamespace(data=SimpleNamespace(root_pos_w=wrap(root_pos), root_quat_w=wrap(root_quat)))

    class Scene(dict):
        env_origins = torch.tensor([[0., 0., 0.], [10., 0., 0.]])

    env = SimpleNamespace(scene=Scene(robot=robot, frames=frames))
    pos_fn = module.ee_frame_pos if name == "franka" else module.eef_pos
    quat_fn = module.ee_frame_quat if name == "franka" else module.eef_quat
    torch.testing.assert_close(pos_fn(env), torch.tensor([[.3, .1, .2], [.3, .1, .2]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(quat_fn(env), torch.tensor([[1., 0., 0., 0.]]).repeat(2, 1), atol=1e-6, rtol=0)
