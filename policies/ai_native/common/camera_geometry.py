# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nominal DROID RGB calibration shared by geometry prompt families."""

from typing import Any, Mapping

import numpy as np

DROID_CAMERA_CALIBRATION = {
    "wrist_cam": {"parent": "flange", "pos": (0.011, -0.031, -0.074),
                  "rot": (-0.420, 0.570, 0.576, -0.409), "focal_length": 2.8},
    "over_shoulder_left_camera": {"parent": "robot_root", "pos": (0.05, 0.57, 0.66),
                                  "rot": (-0.393, -0.195, 0.399, 0.805), "focal_length": 2.1},
    "over_shoulder_right_camera": {"parent": "robot_root", "pos": (0.05, -0.57, 0.66),
                                   "rot": (0.805, 0.399, -0.195, -0.393), "focal_length": 2.1},
}


def validate_droid_camera_calibration(env_config: Mapping[str, Any]) -> None:
    """Reject a mismatched nominal configuration instead of supplying false extrinsics."""
    scene = env_config["scene"]
    robot = scene["robot"]["init_state"]
    if not np.allclose(robot["pos"], [0, 0, 0]) or not np.allclose(robot["rot"], [1, 0, 0, 0]):
        raise ValueError("camera prompt requires the calibrated identity DROID robot-root pose")
    for name, expected in DROID_CAMERA_CALIBRATION.items():
        camera = scene[name]
        offset, spawn = camera["offset"], camera["spawn"]
        valid = (
            offset["convention"] == "opengl"
            and np.allclose(offset["pos"], expected["pos"], atol=1e-8, rtol=0)
            and np.allclose(offset["rot"], expected["rot"], atol=1e-8, rtol=0)
            and np.isclose(spawn["focal_length"], expected["focal_length"])
            and np.isclose(spawn["horizontal_aperture"], 5.376)
            and np.isclose(spawn["vertical_aperture"], 3.024)
            and not spawn.get("horizontal_aperture_offset", 0)
            and not spawn.get("vertical_aperture_offset", 0)
            and spawn.get("projection_type", "pinhole") == "pinhole"
            and np.isclose(camera["width"] / camera["height"], 16 / 9)
        )
        suffix = ("/robot/Gripper/Robotiq_2F_85/base_link/wrist_cam" if name == "wrist_cam" else f"/{name}")
        if not valid or not camera["prim_path"].endswith(suffix):
            raise ValueError(f"camera prompt calibration does not match runtime configuration for {name}")
