# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.utils import configclass


@configclass
class GymMatchLightCfg:
    """Three directional lights for the ALOHA reference-view variant.

    Directions follow gym-aloha's ``gym_aloha/assets/scene.xml``. Native XYZW
    rotations map each USD light's -Z axis to its reference direction. Isaac
    intensity uses this project's 3000 setting; MuJoCo's diffuse values
    do not establish photometric equivalence between renderers.
    """

    front_left_light = AssetBaseCfg(
        prim_path="/World/gym_front_left_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-1.0, -1.0, 1.0),
            rot=(0.3250575837, -0.3250575837, 0.0, 0.8880738340),
        ),
    )
    front_right_light = AssetBaseCfg(
        prim_path="/World/gym_front_right_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(1.0, -1.0, 1.0),
            rot=(0.3250575837, 0.3250575837, 0.0, 0.8880738340),
        ),
    )
    back_light = AssetBaseCfg(
        prim_path="/World/gym_back_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000.0),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 1.0, 1.0),
            rot=(-0.3826834324, 0.0, 0.0, 0.9238795325),
        ),
    )


@configclass
class SphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereLightCfg(intensity=5000),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )


@configclass
class RedSphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    red_sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere_red",
        spawn=sim_utils.SphereLightCfg(intensity=100000, color=(1.0, 0.0, 0.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
@configclass
class BlueSphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    blue_sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere_blue",
        spawn=sim_utils.SphereLightCfg(intensity=100000, color=(0.0, 0.0, 1.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )
@configclass
class GreenSphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    green_sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere_green",
        spawn=sim_utils.SphereLightCfg(intensity=100000, color=(0.0, 1.0, 0.0)),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 1.0)),
    )


@configclass
class ExtremelyDimSphereLightCfg:
    """Cfg class that adds lighting to scene configurations."""
    sphere_light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/sphere",
        spawn=sim_utils.SphereLightCfg(intensity=50),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 0.7)),
    )

@configclass
class FrontDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    front_directional_light = AssetBaseCfg(
        prim_path="/World/front_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=3),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.0, 0.7071, 0.0, 0.7071)),
    )

@configclass
class BehindDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    behind_directional_light = AssetBaseCfg(
        prim_path="/World/behind_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=3),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.0, -0.7071, 0.0, 0.7071)),
    )

@configclass
class TopDownDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    top_down_directional_light = AssetBaseCfg(
        prim_path="/World/top_down_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=0.0),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.0, 0.0, 0.0, 1.0)),
    )

@configclass
class LeftDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    left_directional_light = AssetBaseCfg(
        prim_path="/World/left_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=3),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(-0.7071, 0.0, 0.0, 0.7071)),
    )

@configclass
class RightDirectionalLightCfg:
    """Cfg class that adds directional lighting to scene configurations."""
    right_directional_light = AssetBaseCfg(
        prim_path="/World/right_directional_light",
        spawn=sim_utils.DistantLightCfg(intensity=3000, angle=0.53, exposure=3),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 5.0), rot=(0.7071, 0.0, 0.0, 0.7071)),
    )
