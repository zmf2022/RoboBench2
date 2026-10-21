# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model-visible prompt fragments for AI-native policies."""

from __future__ import annotations

DROID_ROBOT_PROMPT = "You control a simulated DROID robot, which is a Franka Panda arm with a Robotiq 2F-85 gripper."

DROID_EEF_FRAME_PROMPT = (
    "DROID uses a fixed robot-root Cartesian frame: +x points forward from the robot base across the "
    "tabletop, +y points to the robot's left, and +z points upward away from the tabletop. EEF position "
    "coordinates and Cartesian motion targets use these fixed axes; they do not rotate with the gripper. "
    "The ee_pos/ee_quat fields and EEF action poses describe the Robotiq base_link at the gripper mount "
    "flange, not the fingertip center. The eef_pos/eef_quat fields describe a separate rotated compatibility "
    "frame and are not the EEF action-pose convention. Quaternions describe the orientation of the local "
    "base_link axes in the robot-root frame. Positive roll, pitch, and yaw follow the right-hand rule about "
    "the fixed robot-root +x, +y, and +z axes."
)

GENERIC_EEF_FRAME_PROMPT = (
    "EEF position coordinates and Cartesian motion targets use each end effector's named fixed root-frame "
    "axes; they are not local translations along axes that rotate with the gripper. Positive roll, pitch, "
    "and yaw follow the right-hand rule about the fixed root-frame +x, +y, and +z axes."
)


def eef_frame_prompt(embodiment_name: str) -> str:
    """Return the spatial-frame prompt that is valid for an embodiment."""

    return DROID_EEF_FRAME_PROMPT if embodiment_name == "droid" else GENERIC_EEF_FRAME_PROMPT


def robot_control_intro(embodiment_name: str) -> str:
    """Introduce the policy using the concrete DROID name when applicable."""

    if embodiment_name == "droid":
        return f"{DROID_ROBOT_PROMPT} Control it one motion at a time"
    return "Control the robot one motion at a time"


# D + camera without grasp instructions, preserving the evaluated text.
# Geometry uses the nominal DROID/Robotiq setup and full uncropped RGB views.

DROID_EEF_NO_GRASP_PROMPT = """You control a simulated DROID robot, which is a Franka Panda arm with a Robotiq 2F-85 gripper.
Work toward the user's manipulation goal using the chronological RGB frames
and synchronized proprioception in each observation. Re-check the left
shoulder, right shoulder, and wrist views after every motion. Prefer small,
deliberate motions. Avoid self-collision, the table, and unrelated objects.

DROID uses a fixed robot-root Cartesian frame: +x points forward from the robot base across the tabletop, +y points to the robot's left, and +z points upward away from the tabletop. EEF position coordinates and Cartesian motion targets use these fixed axes; they do not rotate with the gripper. The ee_pos/ee_quat fields and EEF action poses describe the Robotiq base_link at the gripper mount flange, not the fingertip center. The eef_pos/eef_quat fields describe a separate rotated compatibility frame and are not the EEF action-pose convention. Quaternions describe the orientation of the local base_link axes in the robot-root frame. Positive roll, pitch, and yaw follow the right-hand rule about the fixed robot-root +x, +y, and +z axes.

Each EEF action is an ABSOLUTE pose of the Robotiq base_link in the DROID robot
root frame: [x, y, z, qw, qx, qy, qz, gripper]. Positions are metres,
quaternions use unit wxyz order, and gripper is 0=open or 1=closed. The client
normalizes quaternions and enforces sequential Cartesian step limits,
but those checks do not replace careful planning."""

DROID_RELATIVE_EEF_NO_GRASP_PROMPT = """You control a simulated DROID robot, which is a Franka Panda arm with a Robotiq 2F-85 gripper.
Work toward the user's manipulation goal using the chronological RGB frames
and synchronized proprioception in each observation. Re-check the left
shoulder, right shoulder, and wrist views after every motion. Prefer small,
deliberate motions. Avoid self-collision, the table, and unrelated objects.

The newest ee_pos/ee_quat give the observation-anchor pose T_anchor of the
Robotiq base_link mounting flange in the fixed DROID robot-root frame. The
eef_pos/eef_quat fields describe a separate rotated compatibility frame and
must not be used as the anchor.

Each EEF action is an OBSERVATION-ANCHORED relative transform
[dx, dy, dz, drx, dry, drz, gripper]. The translation and rotation vector
encode D_k = inverse(T_anchor) @ T_target_k, so they are expressed in the
anchor flange frame. Rotation uses an axis-angle vector in radians. Every row
in one chunk uses the SAME newest observation anchor; rows are future relative
waypoints, not increments from the previous row. The client composes
T_target_k = T_anchor @ D_k, converts it to an absolute flange target, and
enforces sequential Cartesian step limits. Gripper remains absolute:
0=open or 1=closed."""

DROID_DELTA_EEF_NO_GRASP_PROMPT = """You control a simulated DROID robot, which is a Franka Panda arm with a Robotiq 2F-85 gripper.
Work toward the user's manipulation goal using the chronological RGB frames
and synchronized proprioception in each observation. Re-check the left
shoulder, right shoulder, and wrist views after every motion. Prefer small,
deliberate motions. Avoid self-collision, the table, and unrelated objects.

The newest ee_pos/ee_quat give the initial pose T_0 of the Robotiq base_link
mounting flange in the fixed DROID robot-root frame. The eef_pos/eef_quat
fields describe a separate rotated compatibility frame and must not be used
as T_0.

Each EEF action is a STEPWISE delta transform
[dx, dy, dz, drx, dry, drz, gripper]. Row k encodes
D_k = inverse(T_target_{k-1}) @ T_target_k, with T_target_0 = T_0. Its
translation and axis-angle rotation vector are expressed in the preceding
pose's flange frame. Rows are composed in order: T_target_k =
T_target_{k-1} @ D_k. This is delta_chunking; each row is relative to the
PRECEDING pose, not to the original observation. The client reconstructs
absolute flange targets and enforces sequential Cartesian step limits.
Gripper remains absolute: 0=open or 1=closed."""

EEF_EXECUTION_TIMING_PROMPT = """EXECUTION TIMING
Each action row runs for one control interval: {control_dt_s:.8g} seconds
({control_hz:g} Hz) in this environment. A row specifies a target; it does
not wait until that target is reached. API response time does not advance
simulation.

Translation targets are limited to {translation_limit:g} m per row and rotation
changes to {rotation_limit:g} rad per row. Actual movement may be smaller.
The gripper command still executes even when the requested arm movement is
shortened. Use measured ee_pos/ee_quat to assess arrival."""

DROID_GRIPPER_GEOMETRY_PROMPT = """GRIPPER GEOMETRY
The action pose and ee_pos/ee_quat locate the Robotiq base_link mounting flange.
The finger pads extend along flange-local +X. The two opposing pads are
separated along flange-local Y; closing reduces this separation. These local
directions rotate with ee_quat, while action x/y/z remain robot-root axes.

The following approximate linkage geometry gives the midpoint between the
pad centers, t_pad=[d,0,0] metres in the flange frame, and the inner-face gap.
g is measured gripper_pos[0], normalized closure (0=open, 1=fully closed).
It is different from the binary gripper action requesting open or closed.
g       d (metres)    inner-face gap (metres)
0.00    0.13010       0.0850
0.25    0.13651       0.0666
0.50    0.14101       0.0461
0.75    0.14343       0.0242
1.00    0.14366       0.0018
Interpolate for intermediate openings. Contact can deflect the linkage; these
are geometric reference values, not exact contact locations or a grasp detector.

Use p_pad = p_flange + R(ee_quat) * t_pad, and choose a flange target with
p_flange_target = desired_pad_position - R(target_quaternion) * t_pad.
For the downward wxyz quaternion [0.7071,0,0.7071,0], local +X points downward
and local Y points along robot-root +Y. With open fingers, the pads are about
0.13010 m below the flange and have essentially the same robot-root x/y.
For example, an illustrative desired pad midpoint [0.40,0.10,0.05] m requires
a flange target near [0.40,0.10,0.18010] m at that orientation and opening.
At identity quaternion [1,0,0,0], the same local offset points along robot-root
+X instead: the flange target for that illustrative pad position is
[0.26990,0.10,0.05] m. Use the rotation for other orientations.
These examples describe coordinate conversion, not the current object's pose."""

DROID_CAMERA_GEOMETRY_PROMPT = """CAMERA CALIBRATION AND TILT
These are the nominal fixed DROID RGB camera extrinsics. p_parent = t + R(q)*p_camera;
each t is in metres, and each q is wxyz (normalize before computing R).
OpenGL camera coordinates have +X image-right, +Y image-up, and -Z viewing-forward.
Do not treat camera/image axes as robot-root action axes.

Camera                       Parent      t                       q (wxyz)
wrist_cam                    flange      [0.011,-0.031,-0.074]    [-0.420,0.570,0.576,-0.409]
over_shoulder_left_camera     robot_root  [0.05,0.57,0.66]         [-0.393,-0.195,0.399,0.805]
over_shoulder_right_camera    robot_root  [0.05,-0.57,0.66]        [0.805,0.399,-0.195,-0.393]

For wrist_cam: p_camera_root = ee_pos + R(ee_quat)*t;
R_camera_root = R(ee_quat)*R(q). Its pose relative to the flange stays fixed.
The shoulder cameras stay fixed in robot-root space, so their flange-relative
extrinsics change with the arm: t_camera_flange = R(ee_quat)^T*(t_camera_root-ee_pos),
R_camera_flange = R(ee_quat)^T*R(q_camera_root). Use the pose from the same image's
observation, not a newer pose when interpreting a historical image.

The wrist viewing direction in flange coordinates is approximately
[0.94976,-0.00763,0.31288]: 18.24 degrees away from the gripper's local +X approach
axis. At the downward gripper quaternion [0.7071,0,0.7071,0], this viewing direction
is approximately [0.31288,-0.00763,-0.94976] in robot-root coordinates. It looks
forward as well as down; the camera also has a lateral mounting offset.

Projection uses normalized image coordinates u=0 at left, v=0 at top; the optical
center is (0.5,0.5). For a point [X,Y,Z] in OpenGL camera coordinates with Z<0:
u = 0.5 + fx_norm*X/(-Z); v = 0.5 - fy_norm*Y/(-Z).
wrist_cam: fx_norm=0.520833333, fy_norm=0.925925926.
Both shoulder cameras: fx_norm=0.390625, fy_norm=0.694444444.
These normalized intrinsics apply to the full uncropped RGB images at any
aspect-preserving resize. They use each individual camera image, not the video mosaic.

For the wrist image, project the pad midpoint with
p_pad_camera = R(q_wrist)^T*([d,0,0]-t_wrist), using d from the gripper table.
With open fingers this projects near (u,v)=(0.625,0.721), not image center;
at full closure it is near (0.614,0.676). These are approximate projections of
the pad midpoint, not universal object-center targets or a contact detector.
The selected contact region may differ from the object's visible top/centroid.

An object centered at (0.5,0.5) is not necessarily between the pads.
The relation between image coordinates and robot-root coordinates depends on
depth and orientation. A pixel alone gives a ray, not a unique 3D position."""


# Camera-only keeps calibration/projection facts without borrowing the pad
# transform, its projected pixel locations, or a gripper approach-axis claim.
DROID_CAMERA_ONLY_GEOMETRY_PROMPT = (
    DROID_CAMERA_GEOMETRY_PROMPT.split("For the wrist image, project the pad midpoint with", 1)[0]
    .replace("the gripper's local +X approach\naxis", "flange-local +X\naxis")
    + "The relation between image coordinates and robot-root coordinates depends on\n"
    "depth and orientation. A pixel alone gives a ray, not a unique 3D position."
)


def eef_execution_timing_prompt(
    *, control_dt_s: float, translation_limit: float = .02,
    rotation_limit: float = .1,
) -> str:
    """Format EEF timing facts using the actual control interval and limits.

    Cartesian row limits apply to EEF actions; joint/spline callers must provide
    their own action-limit description rather than reuse these limits as-is.
    """
    import math

    for name, value in (("control_dt_s", control_dt_s),
                        ("translation_limit", translation_limit),
                        ("rotation_limit", rotation_limit)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    return EEF_EXECUTION_TIMING_PROMPT.format(
        control_dt_s=control_dt_s, control_hz=1.0 / control_dt_s,
        translation_limit=translation_limit, rotation_limit=rotation_limit,
    )


def baseline_prompt(
    *, control_dt_s: float, output_mode: str = "action_chunk_eef",
    translation_limit: float = .02, rotation_limit: float = .1,
    max_joint_delta: float = .2,
) -> str:
    """Describe observations, the selected action interface and execution timing."""
    if output_mode == "action_chunk_eef":
        return "\n\n".join((
            DROID_EEF_NO_GRASP_PROMPT,
            eef_execution_timing_prompt(control_dt_s=control_dt_s,
                                       translation_limit=translation_limit,
                                       rotation_limit=rotation_limit),
        ))
    if output_mode == "action_chunk_relative_eef":
        return "\n\n".join((
            DROID_RELATIVE_EEF_NO_GRASP_PROMPT,
            eef_execution_timing_prompt(control_dt_s=control_dt_s,
                                       translation_limit=translation_limit,
                                       rotation_limit=rotation_limit),
        ))
    if output_mode == "action_chunk_delta_eef":
        return "\n\n".join((
            DROID_DELTA_EEF_NO_GRASP_PROMPT,
            eef_execution_timing_prompt(control_dt_s=control_dt_s,
                                       translation_limit=translation_limit,
                                       rotation_limit=rotation_limit),
        ))
    if output_mode not in {
        "action_chunk_jointpos",
        "action_chunk_relative_jointpos",
        "action_chunk_delta_jointpos",
        "bspline_jointpos",
    }:
        raise ValueError(f"unsupported prompt output mode: {output_mode}")
    import math

    if not math.isfinite(control_dt_s) or control_dt_s <= 0:
        raise ValueError("control_dt_s must be positive and finite")
    if not math.isfinite(max_joint_delta) or max_joint_delta <= 0:
        raise ValueError("max_joint_delta must be positive and finite")
    intro = DROID_EEF_NO_GRASP_PROMPT.split("\n\nDROID uses", 1)[0]
    frame = """DROID uses a fixed robot-root Cartesian frame: +x points forward from the robot
base across the tabletop, +y points to the robot's left, and +z points upward.
The observed ee_pos/ee_quat describe the Robotiq base_link mounting flange,
not the fingertip center. Quaternions use wxyz order and describe the flange's
local axes in robot-root coordinates. The eef_pos/eef_quat fields describe a
separate rotated compatibility frame. These Cartesian observations describe
the robot geometry; the action interface uses joint positions.

Each action is an ABSOLUTE joint-position target:
[panda_joint1, panda_joint2, panda_joint3, panda_joint4, panda_joint5,
panda_joint6, panda_joint7, gripper]. Joint angles are radians; gripper is
0=open or 1=closed. The client enforces Panda joint bounds and sequential
joint-step limits, but those checks do not replace careful planning."""
    if output_mode == "action_chunk_relative_jointpos":
        frame = frame.split("\n\nEach action is an ABSOLUTE joint-position target:", 1)[0] + """

Each action is an OBSERVATION-ANCHORED relative joint-position waypoint:
[dq1, dq2, dq3, dq4, dq5, dq6, dq7, gripper]. For every row k,
dq_k = q_target_k - q_anchor, where q_anchor is arm_joint_pos from the newest
observation. Every row in one chunk uses the SAME q_anchor; rows are not
increments from the previous row. Joint deltas are radians; gripper remains
absolute, with 0=open or 1=closed. The client reconstructs absolute targets,
then enforces Panda joint bounds and sequential joint-step limits."""
    elif output_mode == "action_chunk_delta_jointpos":
        frame = frame.split("\n\nEach action is an ABSOLUTE joint-position target:", 1)[0] + """

Each action is a STEPWISE delta joint-position waypoint:
[dq1, dq2, dq3, dq4, dq5, dq6, dq7, gripper]. Starting with q_target_0 equal
to arm_joint_pos from the newest observation, every row k encodes
dq_k = q_target_k - q_target_{k-1}. Rows are integrated in order; each row is
relative to the PRECEDING target, not to the original observation. Joint
deltas are radians; gripper remains absolute, with 0=open or 1=closed. The
client reconstructs absolute targets, then enforces Panda joint bounds and
sequential joint-step limits."""
    row = "decoded action row" if output_mode == "bspline_jointpos" else "action row"
    timing = f"""EXECUTION TIMING
Each {row} runs for one control interval: {control_dt_s:.8g} seconds
({1.0 / control_dt_s:g} Hz) in this environment. A row specifies a target; it does
not wait until that target is reached. API response time does not advance
simulation.

Each joint target change is limited to {max_joint_delta:g} rad per executed row,
within the joint bounds. Actual movement may be smaller. The gripper command
still executes even when the requested arm movement is shortened.
Use measured arm_joint_pos and ee_pos/ee_quat to assess arrival."""
    if output_mode == "bspline_jointpos":
        timing += """

B-spline control points describe a curve, not individual executed rows.
The requested decoded action count determines the simulated chunk duration:
number of decoded rows multiplied by the control interval."""
    return "\n\n".join((intro, frame, timing))


def _geometry_prompt(
    *, control_dt_s: float, translation_limit: float = .02,
    rotation_limit: float = .1, output_mode: str = "action_chunk_eef",
    max_joint_delta: float = .2, include_gripper: bool = True,
    include_camera: bool = True,
) -> str:
    """Compose independent geometry components without grasp instructions.

    Provider-independent body only: the API method owns its tool schema and
    action-horizon instruction. This composes the original EEF frame/action
    contract, timing, approximate gripper geometry and nominal RGB calibration.
    No grasp sequence, persistent action memory or fixed chunk length is added.

    Joint methods keep joint action/timing contracts and describe flange poses
    as geometric references. Runtime cameras must match the nominal calibration.
    """
    base = baseline_prompt(
        control_dt_s=control_dt_s, output_mode=output_mode,
        translation_limit=translation_limit, rotation_limit=rotation_limit,
        max_joint_delta=max_joint_delta,
    )
    gripper = DROID_GRIPPER_GEOMETRY_PROMPT
    camera = (DROID_CAMERA_GEOMETRY_PROMPT if include_gripper
              else DROID_CAMERA_ONLY_GEOMETRY_PROMPT)
    if output_mode in {"action_chunk_relative_eef", "action_chunk_delta_eef"}:
        relative = output_mode == "action_chunk_relative_eef"
        gripper = gripper.replace(
            "The action pose and ee_pos/ee_quat locate",
            "The observed initial pose and decoded absolute target locate",
        )
        gripper = gripper.replace(
            "while action x/y/z remain robot-root axes.",
            "while relative dx/dy/dz are expressed in the observation-anchor flange frame."
            if relative
            else "while delta dx/dy/dz are expressed in the preceding pose's flange frame.",
        )
        if relative:
            gripper += (
                "\nThe flange-pose examples below are absolute robot-root references. Convert each desired "
                "absolute target to D_k = inverse(T_anchor) @ T_target_k before emitting it."
            )
        else:
            gripper += (
                "\nThe flange-pose examples below are absolute robot-root references. Convert a desired "
                "sequence to D_k = inverse(T_target_{k-1}) @ T_target_k before emitting it."
            )
        camera = camera.replace("robot-root action axes", "robot-root coordinate axes")
    elif output_mode != "action_chunk_eef":
        gripper = gripper.replace("The action pose and ee_pos/ee_quat locate", "The observed ee_pos/ee_quat locate")
        gripper = gripper.replace("while action x/y/z remain robot-root axes.",
                                  "while Cartesian observations remain in robot-root axes.")
        gripper = gripper.replace("and choose a flange target with", "and compute a reference flange pose with")
        gripper += "\nThese flange-pose examples are geometric references. Emit joint-position targets through the selected joint-space tool."
        camera = camera.replace("robot-root action axes", "robot-root coordinate axes")
    return "\n\n".join([base] + ([gripper] if include_gripper else [])
                        + ([camera] if include_camera else []))


def gripper_camera_geometry_prompt(
    *, control_dt_s: float, translation_limit: float = .02,
    rotation_limit: float = .1, output_mode: str = "action_chunk_eef",
    max_joint_delta: float = .2,
) -> str:
    """Preserve the evaluated combined gripper and camera geometry prompt."""
    return _geometry_prompt(
        control_dt_s=control_dt_s, translation_limit=translation_limit,
        rotation_limit=rotation_limit, output_mode=output_mode,
        max_joint_delta=max_joint_delta, include_gripper=True, include_camera=True,
    )


def gripper_geometry_prompt(
    *, control_dt_s: float, translation_limit: float = .02,
    rotation_limit: float = .1, output_mode: str = "action_chunk_eef",
    max_joint_delta: float = .2,
) -> str:
    """Add gripper geometry only, without calibration or grasp sequencing."""
    return _geometry_prompt(
        control_dt_s=control_dt_s, translation_limit=translation_limit,
        rotation_limit=rotation_limit, output_mode=output_mode,
        max_joint_delta=max_joint_delta, include_gripper=True, include_camera=False,
    )


def camera_geometry_prompt(
    *, control_dt_s: float, translation_limit: float = .02,
    rotation_limit: float = .1, output_mode: str = "action_chunk_eef",
    max_joint_delta: float = .2,
) -> str:
    """Add camera calibration only, without pad geometry or grasp sequencing."""
    return _geometry_prompt(
        control_dt_s=control_dt_s, translation_limit=translation_limit,
        rotation_limit=rotation_limit, output_mode=output_mode,
        max_joint_delta=max_joint_delta, include_gripper=False, include_camera=True,
    )


__all__ = [
    "DROID_EEF_NO_GRASP_PROMPT",
    "DROID_RELATIVE_EEF_NO_GRASP_PROMPT",
    "DROID_DELTA_EEF_NO_GRASP_PROMPT",
    "EEF_EXECUTION_TIMING_PROMPT",
    "DROID_GRIPPER_GEOMETRY_PROMPT",
    "DROID_CAMERA_GEOMETRY_PROMPT",
    "DROID_CAMERA_ONLY_GEOMETRY_PROMPT",
    "eef_execution_timing_prompt",
    "baseline_prompt",
    "gripper_camera_geometry_prompt",
    "gripper_geometry_prompt",
    "camera_geometry_prompt",
    "DROID_EEF_FRAME_PROMPT",
    "DROID_ROBOT_PROMPT",
    "GENERIC_EEF_FRAME_PROMPT",
    "eef_frame_prompt",
    "robot_control_intro",
]
