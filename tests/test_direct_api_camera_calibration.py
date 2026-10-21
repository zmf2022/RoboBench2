# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera calibration must read the public WXYZ config, not native XYZW values.

Can also run without booting Isaac Sim:
    python -m pytest --noconftest tests/test_direct_api_camera_calibration.py
"""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import robolab.core.utils.isaaclab_compat as compat
from policies.ai_native.common.camera_geometry import DROID_CAMERA_CALIBRATION
from policies.ai_native.direct_api import client as direct_api


class _EnvironmentConfig:
    def __init__(self):
        self.sim = SimpleNamespace(dt=1 / 120)
        self.decimation = 8
        self.scene = {"robot": {"init_state": {"pos": [0, 0, 0], "rot": [1, 0, 0, 0]}}}
        for name, expected in DROID_CAMERA_CALIBRATION.items():
            suffix = ("/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam"
                      if name == "wrist_cam" else f"/{name}")
            self.scene[name] = {
                "prim_path": "/World" + suffix,
                "offset": {"convention": "opengl", "pos": expected["pos"], "rot": expected["rot"]},
                "spawn": {"focal_length": expected["focal_length"],
                          "horizontal_aperture": 5.376, "vertical_aperture": 3.024},
                "width": 640, "height": 360,
            }

    def to_dict(self):
        return {"scene": deepcopy(self.scene)}


@pytest.mark.parametrize("uses_xyzw", [False, True])
@pytest.mark.parametrize("prompt_family", ["camera_geometry_prompt", "gripper_camera_geometry_prompt"])
@pytest.mark.parametrize("mismatch", [None, "robot", "camera"])
def test_calibration_uses_public_config_without_weakening_validation(monkeypatch, uses_xyzw, prompt_family, mismatch):
    monkeypatch.setattr(compat, "ISAACLAB_USES_XYZW", uses_xyzw)
    cfg = _EnvironmentConfig()
    if mismatch == "robot":
        cfg.scene["robot"]["init_state"]["rot"] = [0, 1, 0, 0]
    elif mismatch == "camera":
        cfg.scene["wrist_cam"]["offset"]["rot"] = [1, 0, 0, 0]
    compat.prepare_env_cfg(cfg)
    native_config = cfg.to_dict()

    client = object.__new__(direct_api.AINativeClient)
    client.prompt_family = prompt_family
    client._configure_control_timing = Mock()
    artifacts = Mock()
    monkeypatch.setattr(direct_api, "write_evaluation_artifacts", artifacts)

    if mismatch:
        message = "identity DROID robot-root pose" if mismatch == "robot" else "wrist_cam"
        with pytest.raises(ValueError, match=message):
            client.configure_environment(cfg)
        artifacts.assert_not_called()
    else:
        client.configure_environment(cfg)
        client._configure_control_timing.assert_called_once_with(pytest.approx(1 / 15))
        artifacts.assert_called_once_with(client)
    assert cfg.to_dict() == native_config
