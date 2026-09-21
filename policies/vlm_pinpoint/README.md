# VLM Pinpoint

Pointing-based robot control: the VLM returns an **image point + phase**,
geometry turns it into metric motion. No direction tokens, no metre deltas
from the model — it only points.

```text
env obs (dict)
├─ image_obs["wrist_cam"]        (1, 720, 1280, 3) uint8    [输入]
├─ image_obs["wrist_cam_depth"]  (1, 720, 1280, 1) float32  [输入]
├─ image_obs["wrist_cam_K/pos/quat"]  内参 + 外参            [输入]
├─ proprio_obs ee_pos            (3,) float64, robot-root   [输入]
├─ proprio_obs ee_quat           (4,) wxyz                  [输入]
└─ instruction                   str, e.g. "Pick up..."     [输入]
        │ extract_views (旋转180°) + preprocess (→640×360)
        │ depth 同样旋转180° → 灰度图(越亮越近), 与 RGB 同朝向
        ▼
backend.point(frame, instruction, depth_image) → {"x", "y" 0-1000,
                                     "phase": APPROACH|…|DONE, "raw": str}
        │ frame_point_to_sensor (解旋转+缩放到传感器 px)
        │ + depth 采样 (窗口中值, 米) + 内参(K 或自带 fx=666.7)
        │ + 腕外参 (OpenGL: +X右 +Y上 −Z前)
        ▼
controller.plan() → [Waypoint(pos, gripper), ...]  一次决策 = 一整套动作
        ▼
env.step  ←  action (1, 7) float [dx,dy,dz, drx,dry,drz, gripper]  [输出]
viz      ←  640×360 红圈(point)+PHASE  ‖  深度图 panel        [输出]
```

One VLM decision expands to a whole primitive; the client servos waypoints
against live proprioception and re-queries only when the plan finishes.

## Files

| File | Role |
|------|------|
| `pointing_backend.py` | `VLLMPointingBackend`: asks the chat model for `POINT`/`PHASE`, parses + validates. Reuses `vlm_agent.vlm_backend.VLMBackend` |
| `pointing_controller.py` | `PointingController.plan()`: unprojects point+depth to a 3D target, expands the phase into waypoints (APPROACH/ALIGN/GRASP/LIFT/PLACE/DONE) |
| `client.py` | `PinpointClient` (`InferenceClient`): servos waypoints, holds latched orientation, draws viz overlays |
| `run.py` | Eval entry: `python policies/vlm_pinpoint/run.py --task BananaInBowlTask` |
| `connector.py` | Legacy adapter for the unreachable external package — kept for reference |

## Point contract

Official [RoboPoint](https://github.com/wentaoyuan/RoboPoint) format first,
ours as fallback:

```
[(x1, y1), ...]              # 0-1 normalized floats, or pixels if > 1
POINT: <x y>                 # 0-1000 ints (ours, always 0-1000)
PHASE: <APPROACH|ALIGN|GRASP|LIFT|PLACE|DONE>
```

`PHASE` is our extension (RoboPoint has none); missing phase defaults to
APPROACH. Parser takes the LAST marker.

## Cameras

`--point-camera external` (default) grounds on the fixed over-shoulder view;
`--point-camera wrist` uses eye-in-hand. External keeps target and gripper in
frame for the whole episode; the wrist view loses them once close. Depth,
`K`, pose and rotation are read from whichever camera is active.

## Model context beyond RGB

- **Gripper crosshair**: fingertip projected back into the frame (`project()`,
  exact inverse of unprojection), so "how far left to move" is visible.
- **Gripper load state**: closed-empty vs closed-loaded look identical;
  measured width is reported in words (GRASP/LIFT/DONE hinge on it).
- **Fingertip height above table**, in cm.
- **Depth map** (grayscale, brighter = closer): coarse near/far prior.
  `--no-depth-image` turns it off for A/B; numeric depth is always sampled
  geometrically regardless.

## Grasp in brief

Stage above → descend vertically → close at a height measured off the depth
map (support/top/long-axis from `object_geometry()`, not a fixed offset) →
hold → lift. Jaw yaw goes across the object's short axis, measured not asked.
Close dwell ends on measured jaw width, with stall fallback. See code comments
for the measured numbers behind each constant.

## Orientation

`action[3:6]` carries the hold correction every step — zeros do **not** hold
in relative IK mode (error compounds; ~24° drift measured upstream). The wrist
camera is this policy's only sensor, so drift invalidates the geometry.

## Status / TODO

- [x] Backend + parsing, geometric controller, depth plumbing, eval wiring
- [ ] Live-episode success-rate validation

> **Verification status (unvalidated):** unit tests green (74), static review
> done, but no full live episode has confirmed end-to-end grasping on the
> current tree. Partial live evidence only: tip-hold grasp seen once,
> descend/timeout/hesitation loops seen repeatedly. Pending live checks:
> staged-radius commit rate, closure-height vs jam boundary, yaw-hysteresis
> effect, squeeze vs piercing trade-off.

## Why pointing

Direction tokens quantise space to 6 axes and need a deadband to converge;
pointing lets the model designate pixels while geometry handles metres. The
original harness this mirrors reported 6/6 vs 0/4 (Gemini on BananaInBowl)
pointing vs Cartesian deltas.
