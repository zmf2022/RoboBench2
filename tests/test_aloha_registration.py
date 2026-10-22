# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ALOHA registration must resolve every variant's real scene configuration."""

import pytest
import torch
from isaaclab.utils.math import quat_apply

from robolab.core.environments.config import parse_env_cfg
from robolab.registrations.aloha.auto_env_registrations import auto_register_aloha_envs


@pytest.mark.parametrize("variant,postfix", [
    ("opposing", "Aloha"), ("station", "BimanualStation"),
    ("mobile", "MobileAloha"), ("gym_match", "AlohaGymMatch"),
])
def test_aloha_registration_resolves_scene(variant, postfix):
    auto_register_aloha_envs(task_dirs=["bimanual"], task="AlohaTransferCubeTask", variant=variant)
    cfg = parse_env_cfg(f"AlohaTransferCubeTask{postfix}", device="cpu", num_envs=1)
    assert cfg.scene.robot.spawn.usd_path.endswith(".usd")
    if variant == "gym_match":
        assert cfg.scene.dome_light.spawn.intensity == 0.0
        assert cfg.scene.dome_light.spawn.texture_file is None
        assert not hasattr(cfg.scene, "sphere_light")
        lights = [cfg.scene.front_left_light, cfg.scene.front_right_light, cfg.scene.back_light]
        quats = torch.tensor([light.init_state.rot for light in lights])
        actual = quat_apply(quats, torch.tensor([[0., 0., -1.]]).expand(3, -1))
        directions = torch.tensor([[1., 1., -1.], [-1., 1., -1.], [0., -1., -1.]])
        torch.testing.assert_close(actual, directions / directions.norm(dim=-1, keepdim=True))
    else:
        assert cfg.scene.sphere_light.spawn.intensity == 5000
        assert cfg.scene.dome_light.spawn.texture_file.endswith("home_office.exr")


def test_gym_match_lights_spawn_with_official_api():
    import isaaclab.sim as sim_utils
    from pxr import UsdLux

    from robolab.variations.backgrounds import DarkVoidBackgroundCfg
    from robolab.variations.lighting import GymMatchLightCfg

    sim_utils.create_new_stage()
    for config in (DarkVoidBackgroundCfg(), GymMatchLightCfg()):
        for asset in vars(config).values():
            prim = asset.spawn.func(
                asset.prim_path, asset.spawn,
                translation=asset.init_state.pos, orientation=asset.init_state.rot,
            )
            assert prim.IsValid()
            assert UsdLux.LightAPI(prim).GetIntensityAttr().Get() == asset.spawn.intensity
