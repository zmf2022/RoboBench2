# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Camera extrinsics in native XYZW, independent of USD/Fabric storage.

The config owns the default local extrinsics. Runtime parent poses own the
world transform; sensor reset refreshes buffers after the public pose setter.
"""

import torch
from isaaclab.sim.views import FrameView
from isaaclab.utils.math import combine_frame_transforms, convert_camera_frame_orientation_convention

from robolab.core.utils.isaaclab_compat import as_torch


def camera_parent_world_pose(env, camera, env_ids):
    """Resolve a camera parent from live physics when it belongs to an asset."""
    if hasattr(camera, "get_mount_parent_world_pose"):
        return camera.get_mount_parent_world_pose(env_ids)

    parent_path = camera.cfg.prim_path.rsplit("/", 1)[0]
    for asset in (*env.scene.articulations.values(), *env.scene.rigid_objects.values()):
        asset_path = asset.cfg.prim_path.rstrip("/")
        if parent_path == asset_path:
            return as_torch(asset.data.root_link_pos_w)[env_ids], as_torch(asset.data.root_link_quat_w)[env_ids]
        if parent_path.startswith(asset_path + "/"):
            body_name = parent_path.rsplit("/", 1)[-1]
            if not hasattr(asset, "find_bodies"):
                raise ValueError(f"Camera parent {parent_path} is not an asset root or articulation body")
            body_ids, _ = asset.find_bodies(body_name)
            if len(body_ids) != 1:
                raise ValueError(f"Camera parent {parent_path} must identify one articulation body")
            return (
                as_torch(asset.data.body_link_pos_w)[env_ids, body_ids[0]],
                as_torch(asset.data.body_link_quat_w)[env_ids, body_ids[0]],
            )

    # Non-physics parents (env origins / static scene mounts) use the public
    # backend-dispatching view. Never read a backend's private USD fallback.
    views = getattr(camera, "_robolab_parent_views", None)
    if views is None:
        views = camera._robolab_parent_views = {}
    if parent_path not in views:
        views[parent_path] = FrameView(parent_path, device=camera.device)
    pos, quat = views[parent_path].get_world_poses()
    return as_torch(pos)[env_ids], as_torch(quat)[env_ids]


def default_camera_world_pose(env, camera, env_ids):
    """Compose immutable configured mount extrinsics with the current parent."""
    parent_pos, parent_quat = camera_parent_world_pose(env, camera, env_ids)
    offset = camera.cfg.offset
    pos = torch.tensor(offset.pos, dtype=torch.float32, device=camera.device).expand(len(env_ids), -1)
    quat = torch.tensor(offset.rot, dtype=torch.float32, device=camera.device).expand(len(env_ids), -1)
    quat = convert_camera_frame_orientation_convention(quat, origin=offset.convention, target="ros")
    return combine_frame_transforms(parent_pos, parent_quat, pos, quat)


def set_camera_world_pose(camera, env_ids, positions, orientations):
    """Set ROS camera extrinsics and refresh state through public sensor APIs."""
    if hasattr(camera, "set_mount_world_pose"):
        camera.set_mount_world_pose(positions, orientations, env_ids)
    else:
        camera.set_world_poses(positions=positions, orientations=orientations, env_ids=env_ids, convention="ros")
    camera.reset(env_ids)
