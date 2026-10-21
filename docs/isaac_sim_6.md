# Isaac Sim 6 support

RoboLab supports Isaac Sim 6.0.1 with the Isaac Lab 3.0 beta 2 patch 1 stack.
This is the recommended stack for RTX PRO 5000/6000 and other recent GPUs.

## Isaac Sim 6.2 upgrade status

As checked on 2026-10-21, NVIDIA pairs **Isaac Sim 6.2.0 with Isaac Lab 3.0.0 GA**.
The current `isaac60` extra remains pinned to Isaac Lab `3.0.0b2.post1` / Isaac Sim 6.0.1.
There is no runnable `isaac62` configuration yet: changing the package pin alone is insufficient.

Static source review found these migration boundaries:

| Boundary | Current project | Isaac Lab 3.0 GA | Status |
| --- | --- | --- | --- |
| Dependencies | PyTorch from the CUDA 12.8 index; Lab beta wheel | PyTorch 2.11 / CUDA 13.0, Warp 1.17 and Sim 6.2 | Requires a separate dependency resolution |
| Render configuration | `robolab/core/environments/runtime.py` reads `env_cfg.sim.render.carb_settings` on every environment creation | GA `SimulationCfg` no longer declares `render` | Direct upgrade is incompatible at this boundary |
| Application launch | Runners call `AppLauncher` | Retained as a deprecated adapter over `launch_simulation` | Adapter exists; runtime behavior has not been tested here |
| Robot state and observations | WXYZ/XYZW conversion, ProxyArray conversion and indexed state writes | GA backend integration must be checked against these contracts | Not verified on 6.2 |

Before enabling a supported 6.2 install option, adapt the renderer/launcher boundary, resolve
its official dependency set in an isolated environment, and verify rendered startup, camera
observations, gripper/contact behavior and one complete recorded task. No 6.2 packages were
installed and no 6.2 simulation was run during this review. Historical 6.0 measurements below
are not evidence of 6.2 compatibility.

Official sources:

- [Isaac Sim compatibility matrix](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/compatibility.html)
- [Isaac Lab 3.0 release](https://github.com/isaac-sim/IsaacLab/releases/tag/v3.0.0)
- [GA dependency definitions](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/pyproject.toml)
- [GA simulation configuration](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab/isaaclab/sim/simulation_cfg.py)
- [GA AppLauncher adapter](https://github.com/isaac-sim/IsaacLab/blob/v3.0.0/source/isaaclab/isaaclab/app/app_launcher.py)

## Setup

Installation and version selection are covered in the main README:
[installation](../README.md#installation), [Docker](../README.md#docker), and
[version compatibility](../README.md#version-compatibility).
The sections below retain the interface audit and historical validation evidence.

## Compatibility design

RoboLab keeps its existing external contracts while adapting the simulator
boundary:

- Configs, observations, Cartesian actions, `WorldState`, and all HDF5
  quaternion fields remain `(w, x, y, z)`. Isaac Lab 3's internal
  `(x, y, z, w)` values are converted centrally. Camera extrinsics retain ROS
  camera axes while using the same stable WXYZ component storage.
- Isaac Lab 3 `ProxyArray` state is exposed to RoboLab as zero-copy torch views.
- State writers use Isaac Lab 3's indexed APIs, with fallbacks for Isaac Lab 2.
- `sim.physics`/`isaaclab_physx.physics.PhysxCfg` and the new `FrameView` API are
  used on Isaac Lab 3.
- Removed `omni.isaac.*` utilities have Isaac Lab or USD replacements on the
  Isaac Sim 6 path.

Old recordings can be read on Isaac Sim 6 because their quaternions are
translated on restore. Exact trajectory reproduction is still only expected
on the simulator stack that created the recording because PhysX behavior
changes between releases.

## Isaac Sim 6 breaking-change audit

The [6.0 breaking-change list](https://docs.isaacsim.omniverse.nvidia.com/latest/overview/release_notes.html#breaking-changes-and-deprecations)
has been checked against RoboLab:

- The removed `omni.isaac.*` shims are not used on the Isaac Sim 6 path.
  Isaac Sim 5 compatibility branches use the renamed `isaacsim.*` modules.
- The Isaac Sim 6 path avoids the deprecated `isaacsim.core.api`,
  `isaacsim.core.prims`, and `isaacsim.core.utils` modules. It uses Isaac Lab
  simulation/view APIs or `isaacsim.core.experimental` instead. The deprecated
  imports that remain in the source are lazy Isaac Sim 5-only fallbacks.
- RoboLab task cameras use Isaac Lab's camera abstraction. Standalone screenshot
  utilities use the replacement `RtxCamera`/`CameraSensor` API from
  `isaacsim.sensors.experimental.rtx`; they do not use RTX Camera prim JSON
  configurations.
- Contact and frame sensors are Isaac Lab sensors, not the deprecated direct
  `isaacsim.sensors.physics`, `isaacsim.sensors.physx`, or
  `isaacsim.sensors.rtx` APIs.
- RoboLab has no ROS 2 OmniGraph nodes, Cortex, wheeled-robot, manipulator,
  Lula/motion-generation, MobilityGen, domain-randomization, merge-mesh,
  Replicator Agent, ML archive, app selector, benchmark examples, Scene Blox,
  or asset-browser dependency. The `PushGraph` embedded in the tools-picking
  scene is an `omni.anim.curve.core.AnimCurve`, not a ROS 2 node.

## Validation and known risk

Run at least one rendered smoke test and one contact-rich task on the target
driver/GPU before producing benchmark results:

```bash
uv run --no-sync python examples/run_kinova_jointpos.py --headless --num-steps 180
uv run --no-sync python examples/run_abs_ik_demo.py --headless
```

Isaac Lab 3.0 is still a beta line. NVIDIA's 3.0 beta release notes list a
known Robotiq 2F-85 issue, so the DROID and Kinova grippers need particular
attention in smoke testing. Validate them on the selected driver/GPU before reporting results.

## Historical validation in RoboLab Verified

The following is retained as a dated comparison, not as a current installation guide.
The old environment names and extras below are no longer supported installation options.

Checked on an A40 with both stacks installed side by side (`.venv-51`: Python 3.11,
`--extra isaac51`; `.venv-60`: Python 3.12, `--extra isaac60`) from one tree, with an open-loop
replay of the same recording on both (`examples/run_recorded.py --env-config current --num_envs 4
--max-steps 450`, GR00T `GrabAFruitTask` env 1):

| check | Isaac Sim 5.1 | Isaac Sim 6.0 |
|---|---|---|
| offline suite (no simulator) | passed | passed |
| `run_empty --friction 0.5`: request applied to 3 objects and both pads | yes, PhysX readback 0.5/0.5 | yes, PhysX readback 0.5/0.5 on every object shape (256 per object) and both pads (64 each), identical to 5.1; the readback needed two fixes for the Isaac Lab 3 backend (facade `isinstance`, warp arrays) |
| replay reproduces the recorded contact event at the recorded time (all 4 envs) | yes | yes |
| recorded quaternions stay WXYZ (end-effector and object poses) | yes | yes |
| Robotiq 2F-85 in the wrist camera | whole | whole, no detached fragments |
| replay speed, 4 envs with video | ~1.3 s/step | ~0.8 s/step |

`run_recorded.py` accepts `--max-steps`, `--no-video` and `--output-dir` so a reproducer can replay
only the stretch up to an event. Exact trajectory reproduction still holds only on the stack that
made the recording; a published number states its stack.
