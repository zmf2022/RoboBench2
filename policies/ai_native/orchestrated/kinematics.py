"""Read robot joints from USD and preview targets without stepping physics.

The FK-preview approach follows GPT-as-Policy's RoboLab integration:
https://github.com/anonymous-report-421/GPT-as-Policy
Only the robot's joint chain is inspected; no object or task state is read.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

JOINT_NAMES = tuple(f"panda_joint{i}" for i in range(1, 8))


def transform(position, quaternion_wxyz):
    position = np.asarray(position, dtype=float)
    quaternion = np.asarray(quaternion_wxyz, dtype=float)
    if (position.shape != (3,) or quaternion.shape != (4,)
            or not np.isfinite(position).all() or not np.isfinite(quaternion).all()
            or abs(np.linalg.norm(quaternion) - 1) > 1e-4):
        raise ValueError("Expected finite position and unit WXYZ quaternion")
    result = np.eye(4)
    result[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
    result[:3, 3] = position
    return result


@dataclass(frozen=True)
class JointEdge:
    name: str
    axis: str | None
    local0: np.ndarray
    local1: np.ndarray
    reverse: bool = False


class RobotFK:
    def __init__(self, chain):
        self.chain = tuple(chain)
        names = [edge.name for edge in self.chain if edge.axis is not None]
        if len(names) != 7 or set(names) != set(JOINT_NAMES):
            raise ValueError("FK chain must contain exactly the seven Panda arm joints")

    @classmethod
    def from_stage(cls, stage, robot_path="/World/envs/env_0/robot"):
        from pxr import Usd, UsdPhysics

        root = stage.GetPrimAtPath(robot_path)
        if not root.IsValid():
            raise ValueError(f"Missing DROID robot at {robot_path}")
        graph = {}
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdPhysics.Joint):
                continue
            joint = UsdPhysics.Joint(prim)
            bodies = (joint.GetBody0Rel().GetTargets(), joint.GetBody1Rel().GetTargets())
            if any(len(body) != 1 for body in bodies):
                continue
            first, second = (str(body[0]) for body in bodies)
            if not all(body.startswith(robot_path + "/") for body in (first, second)):
                continue
            if prim.IsA(UsdPhysics.RevoluteJoint):
                axis = UsdPhysics.RevoluteJoint(prim).GetAxisAttr().Get()
                if axis not in ("X", "Y", "Z"):
                    raise ValueError("Unsupported revolute axis")
            elif prim.IsA(UsdPhysics.FixedJoint):
                axis = None
            else:
                continue
            frames = []
            for index in (0, 1):
                position = getattr(joint, f"GetLocalPos{index}Attr")().Get()
                quaternion = getattr(joint, f"GetLocalRot{index}Attr")().Get()
                frames.append(transform(position, [quaternion.GetReal(), *quaternion.GetImaginary()]))
            for origin, destination, reverse in ((first, second, False), (second, first, True)):
                graph.setdefault(origin, []).append((destination, JointEdge(
                    prim.GetName(), axis, *frames, reverse=reverse,
                )))
        start = robot_path + "/panda_link0"
        end = robot_path + "/Gripper/Robotiq_2F_85/base_link"
        queue, seen = deque([(start, ())]), {start}
        while queue:
            body, chain = queue.popleft()
            if body == end:
                return cls(chain)
            for destination, edge in graph.get(body, ()):
                if destination not in seen:
                    seen.add(destination)
                    queue.append((destination, (*chain, edge)))
        raise ValueError("No Panda-to-Robotiq kinematic chain found")

    def matrix(self, joints):
        joints = np.asarray(joints, dtype=float)
        if joints.shape != (7,) or not np.isfinite(joints).all():
            raise ValueError("FK requires seven finite arm joint positions")
        positions = dict(zip(JOINT_NAMES, joints, strict=True))
        result = np.eye(4)
        for edge in self.chain:
            motion = np.eye(4)
            if edge.axis is not None:
                rotation_vector = np.zeros(3)
                rotation_vector["XYZ".index(edge.axis)] = positions[edge.name]
                motion[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
            relative = edge.local0 @ motion @ np.linalg.inv(edge.local1)
            result = result @ (np.linalg.inv(relative) if edge.reverse else relative)
        return result

    def preview(self, actions, proprio):
        """Check current FK against measurements before exposing a future trajectory."""
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (15, 8) or not np.isfinite(actions).all():
            raise ValueError("Expected a finite H15 x 8 proposal")
        predicted = self.matrix(proprio["arm_joint_pos"])
        measured = transform(proprio["ee_pos"], proprio["ee_quat"])
        position_error = float(np.linalg.norm(predicted[:3, 3] - measured[:3, 3]))
        rotation_error = float(Rotation.from_matrix(predicted[:3, :3] @ measured[:3, :3].T).magnitude())
        if position_error >= 0.002 or rotation_error >= 0.01:
            raise ValueError(f"FK disagrees with measured flange: {position_error=}, {rotation_error=}")
        trajectory = []
        for index, action in enumerate(actions):
            pose = self.matrix(action[:7])
            quaternion = Rotation.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
            trajectory.append(dict(index=index, position=pose[:3, 3].tolist(),
                                   quaternion_wxyz=quaternion.tolist(), gripper_closed=bool(action[7] > 0.5)))
        return dict(frame="base_link_in_robot_root", physical_steps=0,
                    interpretation="robot_target_kinematics_only", trajectory=trajectory,
                    measured_fk_check=dict(position_error_m=position_error, rotation_error_rad=rotation_error))


class DroidFKPreview:
    def bind_environment(self, env):
        if int(env.num_envs) != 1:
            raise ValueError("Orchestrated DROID currently requires --num-envs 1")
        import omni.usd

        self.fk = RobotFK.from_stage(omni.usd.get_context().get_stage())

    def preview(self, actions, proprio):
        return self.fk.preview(actions, proprio)


def pose(matrix):
    quaternion = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    return dict(position=matrix[:3, 3].tolist(), quaternion_wxyz=quaternion[[3, 0, 1, 2]].tolist())

def edited_targets(trajectory, steps, edit):
    """Smooth root-frame offset relative to the selected student prefix.

    Gripper overrides apply explicitly from the first step. Closure after an
    approach must be issued in a later decision using a fresh observation.
    """
    if type(steps) is not int or not 1 <= steps <= 15:
        raise ValueError('Execution prefix must be 1..15 steps')
    if not isinstance(edit, dict) or set(edit) != {'delta_position', 'delta_rotation_vector', 'gripper'}:
        raise ValueError('Edit requires delta_position, delta_rotation_vector, gripper')
    dp, dr = np.asarray(edit['delta_position'], float), np.asarray(edit['delta_rotation_vector'], float)
    if dp.shape != (3,) or dr.shape != (3,) or not np.isfinite([dp, dr]).all():
        raise ValueError('Finite 3D edit vectors required')
    if np.linalg.norm(dp) > .05 + 1e-10 or np.linalg.norm(dr) > .35 + 1e-10:
        raise ValueError('Edit exceeds 5 cm or 0.35 rad short-correction limit')
    if edit['gripper'] not in ('keep', 'open', 'closed'):
        raise ValueError('Unknown gripper override')
    targets = []
    for i, original in enumerate(trajectory[:steps]):
        t = transform(original['position'], original['quaternion_wxyz'])
        alpha = (i + 1) / steps
        t[:3, 3] += alpha * dp
        t[:3, :3] = Rotation.from_rotvec(alpha * dr).as_matrix() @ t[:3, :3]
        grip = original['gripper_closed'] if edit['gripper'] == 'keep' else edit['gripper'] == 'closed'
        targets.append(dict(index=i, **pose(t), gripper_closed=grip))
    return targets
