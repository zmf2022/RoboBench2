# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A tiled camera that rides a rigid body by pose, not by prim hierarchy.

Why: a camera prim parented under an articulation link only follows the link if the
renderer receives that link's physics transform through the prim hierarchy. For the
imported ALOHA 2 asset it does not — with Fabric on, the wrist cameras rendered the
USD default (MJCF zero) pose for every step of every run (docs/bimanual_plan.md,
"wrist cameras frozen"). The DROID Robotiq camera happens to work, the ALOHA one
never did, and the difference is buried in the asset. This sensor sidesteps the
question: it lives at a top-level env prim and, on every update, reads the target
body's pose straight from the PhysX rigid-body view and writes the camera's world
pose (body ∘ offset) before the frame is fetched.

Usage: identical to ``CameraCfg`` plus ``articulation_prim_path_expr`` (the
robot; ``{ENV_REGEX_NS}`` is expanded like every other prim path) and ``body_name``
(the link the camera rides). ``offset`` is the camera pose in that link's frame,
in ``offset.convention``.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    quat_apply,
    quat_mul,
    subtract_frame_transforms,
)

from robolab.core.utils.isaaclab_compat import as_torch


class BodyAttachedTiledCamera(Camera):
    cfg: "BodyAttachedTiledCameraCfg"

    def _initialize_impl(self):
        super()._initialize_impl()
        from isaacsim.core.simulation_manager import SimulationManager

        physics_sim_view = SimulationManager.get_physics_sim_view()
        # Isaac Lab expands {ENV_REGEX_NS} only in `prim_path`; expand it in the
        # articulation expression from the camera's own (already expanded) namespace.
        env_ns = self.cfg.prim_path.rsplit("/", 1)[0]
        art_expr = (self.cfg.articulation_prim_path_expr
                    .replace("{ENV_REGEX_NS}", env_ns)
                    .replace(".*", "*"))
        # A rigid-body view on an articulation link returns identity transforms
        # (probe 2026-08-25); link poses come from the articulation view, the way
        # isaaclab.assets.Articulation reads them.
        self._art_physx_view = physics_sim_view.create_articulation_view(art_expr)
        try:
            n_art = self._art_physx_view.count
        except AttributeError:
            n_art = 0
        if n_art != self._view.count:
            raise RuntimeError(
                f"BodyAttachedTiledCamera '{self.cfg.prim_path}': {self._view.count} cameras but "
                f"{n_art} articulations match '{art_expr}'."
            )
        link_names = list(self._art_physx_view.shared_metatype.link_names)
        if self.cfg.body_name not in link_names:
            raise RuntimeError(
                f"BodyAttachedTiledCamera: link '{self.cfg.body_name}' not in {link_names}")
        self._link_idx = link_names.index(self.cfg.body_name)
        self._offset_pos = torch.tensor(self.cfg.offset.pos, dtype=torch.float32,
                                        device=self._device).repeat(self._view.count, 1)
        self._offset_quat = torch.tensor(self.cfg.offset.rot, dtype=torch.float32,
                                         device=self._device).repeat(self._view.count, 1)

    def get_mount_parent_world_pose(self, env_ids):
        """Return the current mount link pose in native XYZW."""
        tf = as_torch(self._art_physx_view.get_link_transforms()).view(self._view.count, -1, 7)
        return tf[env_ids, self._link_idx, :3], tf[env_ids, self._link_idx, 3:7]

    def set_mount_world_pose(self, positions, orientations, env_ids):
        """Retain a randomized mount offset when the camera follows its body."""
        body_pos, body_quat = self.get_mount_parent_world_pose(env_ids)
        pos, quat = subtract_frame_transforms(body_pos, body_quat, positions, orientations)
        self._offset_pos[env_ids] = pos
        self._offset_quat[env_ids] = convert_camera_frame_orientation_convention(
            quat, origin="ros", target=self.cfg.offset.convention
        )
        self.set_world_poses(positions, orientations, env_ids=env_ids, convention="ros")

    def _follow_body(self, env_ids: Sequence[int]):
        # SensorBase can supply an update mask; follow all mounts before the
        # base camera updates selected image buffers.
        body_pos, body_quat = self.get_mount_parent_world_pose(slice(None))
        cam_pos = body_pos + quat_apply(body_quat, self._offset_pos)
        cam_quat = quat_mul(body_quat, self._offset_quat)
        self.set_world_poses(cam_pos, cam_quat, env_ids=None, convention=self.cfg.offset.convention)
        if not getattr(self, "_follow_logged", False):
            self._follow_logged = True
            print(f"[BodyAttachedTiledCamera] {self.cfg.prim_path}: link '{self.cfg.body_name}' at "
                  f"{body_pos[0].tolist()} -> camera at {cam_pos[0].tolist()}", flush=True)

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        self._follow_body(env_ids)
        super()._update_buffers_impl(env_ids)


@configclass
class BodyAttachedTiledCameraCfg(CameraCfg):
    """CameraCfg with a moving mount; legacy class name retained for recordings."""

    class_type: type = BodyAttachedTiledCamera
    update_latest_camera_pose: bool = True
    articulation_prim_path_expr: str = "{ENV_REGEX_NS}/robot"
    body_name: str = ""
