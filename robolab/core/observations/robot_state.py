# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public robot-state observations at the native simulator boundary."""

from isaaclab.utils.math import subtract_frame_transforms

from robolab.core.utils.isaaclab_compat import as_torch, quat_isaaclab_to_wxyz


def pose_in_robot_root(robot, position_w, orientation_w):
    """Return robot-root XYZ position and public WXYZ orientation.

    Frame math consumes native simulator quaternions. Convert the result only
    when exposing it to policies, including both arms of multi-arm robots.
    """
    pos, quat = subtract_frame_transforms(
        as_torch(robot.data.root_pos_w),
        as_torch(robot.data.root_quat_w),
        as_torch(position_w),
        as_torch(orientation_w),
    )
    return pos, quat_isaaclab_to_wxyz(quat)


def body_pose_in_robot_root(robot, body_name: str):
    """Expose one articulation body's pose through the public state boundary."""
    body_idx = robot.data.body_names.index(body_name)
    return pose_in_robot_root(
        robot,
        as_torch(robot.data.body_pos_w)[:, body_idx, :],
        as_torch(robot.data.body_quat_w)[:, body_idx, :],
    )
