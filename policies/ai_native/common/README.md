# Shared API prompts

Four prompt families support all seven Direct-API output modes:

- `baseline`: action/frame contract and execution timing.
- `gripper_geometry_prompt`: baseline plus flange-to-pad and jaw geometry.
- `camera_geometry_prompt`: baseline plus camera calibration, without pad geometry.
- `gripper_camera_geometry_prompt`: baseline plus both geometry blocks.

The runners default to `gripper_camera_geometry_prompt`. Select another family with `--prompt-family`. All four omit grasp sequencing and action memory.

The shared builder returns the system body; the selected method supplies its tool schema and action-horizon instruction.

```python
from policies.ai_native.common import prompts
system_prompt = prompts.gripper_camera_geometry_prompt(
    control_dt_s=1 / 15,
    translation_limit=0.02,
    rotation_limit=0.1,
)
```

The body contains the DROID EEF frame/action contract, execution timing, opening-dependent flange-to-pad geometry, and nominal camera intrinsics, extrinsics and tilt. It has no approach/close/lift sequence or persistent action memory. The shared body does not choose or enforce a chunk length. Direct-API defaults to a minimum of three actions. The method-specific prompt, tool schema, and decoder enforce the configured `--min-action-horizon`.

Reusable components are exported separately:

- `DROID_EEF_NO_GRASP_PROMPT`: robot, observations, frame and EEF action contract.
- `DROID_RELATIVE_EEF_NO_GRASP_PROMPT`: observation-anchored relative SE(3) contract.
- `DROID_DELTA_EEF_NO_GRASP_PROMPT`: stepwise, preceding-pose SE(3) contract.
- `baseline_prompt(...)`: method-specific action/frame contract and timing.
- `eef_execution_timing_prompt(...)`: control interval and Cartesian row limits.
- `DROID_GRIPPER_GEOMETRY_PROMPT`: pad offsets, jaw axis and pose conversions.
- `DROID_CAMERA_GEOMETRY_PROMPT`: RGB camera transforms, intrinsics and projections.

Pass a relative output mode to describe all future EEF or joint waypoints against one frozen newest-observation anchor. Pass a delta output mode to make each row relative to the preceding decoded target. Absolute joint and B-spline modes retain their prior contracts. B-spline timing distinguishes control points from decoded execution rows. Geometry describes measured flange poses while joint actions remain in joint space. Nominal geometry assumes the DROID/Robotiq configuration and the same uncropped three-camera setup; Direct-API validates that configuration for every method using the geometry family.
