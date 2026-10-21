# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Action guard -- consume once, then bound the motion before anything moves.

Port of RoboJEV's ``src/jev_vla_sim/execution.py``. A guard that runs *before*
the mutation is the difference between a rejected action and a wrecked episode,
and the one-shot ``state_id`` check is what makes a decision un-replayable: a
stale or duplicated decision is an error, not a second command.

Rejected decisions are still consumed, exactly as upstream -- retrying the same
state with a fresh model call is allowed, replaying the same decision is not.
"""

from __future__ import annotations

import numpy as np

from policies.jev.types import EefDecision, PolicyError

# Divergence guards on the grasp centre, not a calibrated reachable set: wide
# enough that ordinary task motion never touches them, tight enough that a sign
# error or a diverging IK solution is caught on the first decision.
WORKSPACE_MIN = (0.10, -0.60, None)   # z floor is derived per episode, see below
WORKSPACE_MAX = (0.90, 0.60, 0.90)
# How far below the object's resting height the grasp centre may still descend.
TCP_FLOOR_MARGIN_M = 0.02
# Held objects may touch their support, but not sink into it.
SUPPORT_TOLERANCE_M = 0.002


class ActionGuard:
    """Consume a decision once, then accept or reject the motion it implies."""

    def __init__(self, workspace_min=WORKSPACE_MIN, workspace_max=WORKSPACE_MAX) -> None:
        self.workspace_min = tuple(workspace_min)
        self.workspace_max = tuple(workspace_max)
        self.consumed: set[str] = set()

    def reset(self) -> None:
        self.consumed.clear()

    def resolve(self, state: dict, decision: EefDecision, step_m: float):
        """-> ``(delta_m | None, gripper_target, reason)``.

        ``delta_m`` is None when the motion was rejected; the gripper command
        still applies, so a rejected move can still release an object.
        """
        state_id = state.get("state_id")
        if decision.state_id != state_id or decision.state_id in self.consumed:
            raise PolicyError(
                f"stale or already consumed decision: {decision.state_id!r} "
                f"against state {state_id!r}")
        self.consumed.add(decision.state_id)

        robot, relations = state["robot"], state["relations"]
        # Aim BEFORE moving, and enforce it here rather than in the criteria:
        # the ordering is safety-critical (the jaws must not arrive at the object
        # turned the wrong way) and prompt text is advisory, not binding.
        #
        # It also has to be all translation, not just the descent. A saturated
        # DLS solve buys position progress with orientation error -- the repo
        # measures the wrist barely turning while the arm translates, and only
        # catching up once it stops (vlm_agent.token_controller.
        # hold_orientation_rotvec). Letting the approach run first therefore does
        # not "aim concurrently"; it defers the whole turn to the moment the arm
        # stops, which is exactly when it is already down at the object.
        aiming = (robot["held_object"] is None
                  and not relations.get("grasp_yaw_aligned", True)
                  and not relations.get("object_settled_in_target", False))
        # 'hold' means keep the command already latched, never "do nothing".
        grip = robot["gripper_target"] if decision.gripper == "hold" else decision.gripper

        tcp = np.asarray(robot["tcp_position"], dtype=float)
        if aiming:
            return None, grip, "aiming_jaws"
        delta = decision.delta(step_m)
        target = tcp + delta
        if not np.isfinite(target).all():
            raise PolicyError("non-finite action target")

        obj_z = float(state["objects"][0]["position"][2])
        rest_z = obj_z - float(relations["object_lifted_above_rest_m"])
        floor_z = rest_z - TCP_FLOOR_MARGIN_M

        low = (self.workspace_min[0], self.workspace_min[1],
               floor_z if self.workspace_min[2] is None else self.workspace_min[2])
        if np.any(target < np.asarray(low, dtype=float)) or \
           np.any(target > np.asarray(self.workspace_max, dtype=float)):
            return None, grip, "workspace_rejected"

        if robot["held_object"] is not None and \
                obj_z + delta[2] < rest_z - SUPPORT_TOLERANCE_M:
            return None, grip, "held_object_support_rejected"

        return delta, grip, "accepted"
