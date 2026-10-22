# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agentic DROID kinematics consume native simulator arrays and quaternions."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from policies.ai_native.agentic.droid_joint_target import ARM_JOINT_NAMES, DroidJointTargetAdapter
from policies.ai_native.common.agent_eef import AgentEEFPlan, EffectorState


def _plan(position, quaternion):
    return AgentEEFPlan(
        actions=np.zeros((1, 8)), status="ok", tool_name="test", note="", feedback=None,
        metadata={}, commanded_state={"arm": EffectorState(position, quaternion, False)},
    )


@pytest.mark.parametrize("kind", ["torch", "numpy", "proxy", "warp"])
@pytest.mark.parametrize("yaw", [0., 90.])
def test_native_kinematics_keep_robot_root_control_direction(kind, yaw):
    def wrap(value):
        value = np.asarray(value, dtype=np.float32)
        if kind == "numpy":
            return value
        if kind == "warp":
            import warp as wp
            return wp.from_numpy(value, dtype=wp.float32, device="cpu")
        tensor = torch.from_numpy(value)
        return SimpleNamespace(torch=tensor) if kind == "proxy" else tensor

    root = Rotation.from_euler("z", yaw, degrees=True)
    local = Rotation.from_euler("x", 35, degrees=True)
    position = np.array([.4, .1, .3])
    root_position = np.array([1., 2., 3.])
    body_position = root_position + root.apply(position)
    jacobian = np.zeros((1, 1, 6, 8))
    jacobian[0, 0, :3, :3] = root.as_matrix()
    jacobian[0, 0, 3:, 3:6] = root.as_matrix()
    robot = SimpleNamespace(
        joint_names=[*ARM_JOINT_NAMES, "finger_joint"], body_names=["root", "base_link"],
        is_fixed_base=True, root_physx_view=SimpleNamespace(get_jacobians=lambda: wrap(jacobian)),
        data=SimpleNamespace(
            joint_pos=wrap(np.zeros((1, 8))), soft_joint_pos_limits=wrap([[[-2., 2.]] * 8]),
            root_pos_w=wrap([root_position]), root_quat_w=wrap([root.as_quat()]),
            body_pos_w=wrap([[root_position, body_position]]),
            body_quat_w=wrap([[root.as_quat(), (root * local).as_quat()]]),
        ),
    )
    adapter = DroidJointTargetAdapter()
    adapter.bind_environment(SimpleNamespace(num_envs=1, scene={"robot": robot}))
    quaternion = local.as_quat()[[3, 0, 1, 2]]
    hold, _ = adapter.action_for_plan(_plan(position, quaternion), env_id=0)
    np.testing.assert_allclose(hold, np.zeros(8), atol=1e-6)
    target = position + np.array([.01, 0., 0.])
    action, _ = adapter.action_for_plan(_plan(target, quaternion), env_id=0)
    expected = np.zeros(8)
    expected[0] = .01 / (1 + adapter.damping ** 2)
    np.testing.assert_allclose(action, expected, atol=1e-6)
    np.testing.assert_allclose(adapter.hold_action(env_id=0), np.zeros(8))


def test_droid_joint_target_runtime():
    if not torch.cuda.is_available():
        pytest.skip("Isaac Sim GPU required")
    from robolab.core.environments.runtime import create_env
    from robolab.core.utils.isaaclab_compat import as_torch
    from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

    auto_register_droid_envs(task="BananaInBowlTask")
    env, _ = create_env("BananaInBowlTask", num_envs=1, use_fabric=True)
    try:
        env.reset()
        robot = env.scene["robot"]
        body = robot.body_names.index("base_link")

        def pose():
            root = Rotation.from_quat(as_torch(robot.data.root_quat_w)[0].cpu().numpy())
            pos = as_torch(robot.data.body_pos_w)[0, body].cpu().numpy()
            root_pos = as_torch(robot.data.root_pos_w)[0].cpu().numpy()
            quat = as_torch(robot.data.body_quat_w)[0, body].cpu().numpy()
            return root.inv().apply(pos - root_pos), (root.inv() * Rotation.from_quat(quat)).as_quat()[[3, 0, 1, 2]]

        adapter = DroidJointTargetAdapter()
        adapter.bind_environment(env)
        position, quaternion = pose()
        hold, _ = adapter.action_for_plan(_plan(position, quaternion), env_id=0)
        np.testing.assert_allclose(hold[:7], adapter.hold_action(env_id=0)[:7], atol=1e-5)
        target = position + np.array([.01, 0., 0.])
        plan = _plan(target, quaternion)
        for _ in range(20):
            action, _ = adapter.action_for_plan(plan, env_id=0)
            assert np.isfinite(action).all()
            env.step(torch.tensor(action, device=env.device).reshape(1, -1))
        actual, _ = pose()
        assert np.linalg.norm(actual - target) < .003, (position, target, actual)
    finally:
        env.close()
