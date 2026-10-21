# Jev two-stage policy

Port of [RoboJEV](https://github.com/lykycy123/RoboJEV) onto RoboLab's DROID
rel-IK harness. No images: structured state in, typed choices out, small
normalised Cartesian increments executed.

```text
proprio + gt_state ──▶ state (robot-base m, per-axis directions, contacts, held_object)
  ──▶ [Q1 INTENT]  approach/grasp/lift/carry/lower/release/withdraw/finish
  ──▶ [Q2 MOTOR]   x/y/z negative|zero|positive + gripper open|hold|close
  ──▶ EefDecision.delta()  normalised to one step ──▶ ActionGuard ──▶ rel-IK
```

## Run

```bash
# Only BananaInBowlTask is supported; select exactly one task per run.
# GT state is REQUIRED: the policy reads object poses, never pixels.
conda activate pharm_flow
python policies/jev/run.py --task BananaInBowlTask --enable-gt-state --headless
# real Jev decisions API (TypeSafe; key in the environment, never in a file)
export TYPESAFE_API_KEY=...
python policies/jev/run.py --task BananaInBowlTask --enable-gt-state --headless \
  --backend jev
```

## Files

| File | Role |
|------|------|
| `geometry.py` | `direction()` / `relation()` — the per-axis semantic layer every criterion quotes |
| `types.py` | `EefDecision`: total validation, and the one place a choice becomes metres |
| `state.py` | `build_state()` — the schema the criteria are written against |
| `prompts.py` | `INTENTS`, `INTENT_RULES`, `motor_criteria()`, `motor_rules()` |
| `execution.py` | `ActionGuard`: one-shot consume, workspace and support rejection |
| `chat_backend.py` / `jev_backend.py` | Same questions, chat server or decisions API |
| `client.py` | The control layer — the only part that is not a verbatim port |

## What the port had to change, and why

RoboJEV executes a decision as an **absolute Cartesian target with per-tick
feedback until it converges**. `DroidRelIKActionCfg` takes **open-loop relative
increments**. Five consequences, each of which silently breaks the port:

1. **The TCP is not `ee_pos`.** `ee_pos` reports `base_link`, the Robotiq mount
   flange. RoboJEV's TCP is the point between the jaws, 131 mm away
   (`vlm_agent.token_controller.FLANGE_TO_FINGERTIP`). Aiming the flange where
   the grasp centre belongs puts the fingers ~13 cm low.
2. **Relative IK achieves ~28% of what is commanded.** `step_m` is the
   *achieved* travel; the commanded total is derived, then divided across
   `control_steps`. Issuing the full delta on every control step multiplies the
   travel by `control_steps`.
3. **Zeros in the rotation slots do not hold the orientation** — in relative
   mode they mean "target = whatever you are now", so IK tilt becomes the next
   setpoint. The gripper is driven back onto a latched reference every step.
4. **The deadband is a ratio of the step, not RoboJEV's 6 mm.** Upstream pairs
   6 mm with a 10 mm step; that *ratio* (0.6) is what makes an axis converge
   instead of overshooting the band on every decision.
5. **RoboJEV grasps a cube; this grasps a banana.** A cube can be taken from any
   wrist yaw, so upstream holds one fixed downward orientation for the whole
   episode. The 2F-85 opens 85 mm and a banana is ~190 mm long: if the jaws
   close along its length the grasp does not fit — physically, not statistically.
   So the wrist is aimed across the object's long axis before the fingers close,
   and `grasp` is gated on `grasp_yaw_aligned`.

### Aiming the wrist

The angle is computed, not asked for. RoboJEV engineers its geometric helpers
explicitly and leaves the model the discrete choice, and `vlm_pinpoint` records
why for this rig: a small model has no reliable rotation channel. Only the
orientation *reference* is turned (`vlm_pinpoint.pointing_controller.
yaw_aligned_quat`, folded into ±90° because a gripper turned 180° grips
identically); the hold already running every control step drives the wrist onto
it, and the top-down approach is preserved. Aiming stops once the object is
held — spinning a loaded wrist twists the object out of the jaws.

`GRASP_YAW_TOLERANCE_RAD` is where the grasp stops fitting, not where it gets
worse: at a yaw error θ the jaws must span `190·sinθ + 35·cosθ` mm, which
reaches the ~70 mm they grip by at 10.6°.

### Numbers sized against this harness

`STEP_M` is 0.03 m and `CONTROL_STEPS` is 12. The two move together on purpose:
the commanded total is `step_m / REL_IK_ACHIEVED_FRACTION`, spread over the
control steps, so 0.03/12 asks the solver for exactly what 0.02/8 did. What
saturates a DLS solve — and a saturated solve pays for distance with
orientation error — is the per-control-step command, not the per-decision
distance. Raising `--step-m` alone commands a harder tick and lowers the
fraction actually achieved; raise both or neither.

That 0.03 exceeds RoboJEV's own `step_m <= 0.02` cap, which is not the same
constraint: RoboJEV's step is closed-loop and lands, so its cap bounds
overshoot. `BananaInBowlTask` runs `episode_length_s=50` at `dt=1/120` with
`decimation=8` — 750 control steps, so 12 steps per decision buys 62 decisions
against the ~40 a pick-and-place needs at a 3 cm step, plus the few stationary
decisions the aiming gate spends before the first approach. `SETTLE_STEPS` is 15, matching
`robolab.constants.SETTLE_WARMUP_S`, because the object's resting height is
latched from the first decision.

`container_radius_m` is a **release gate**, not the success test: the harness
decides success with a convex-hull containment check in the bowl's own frame
(`core/task/conditionals.py::object_in_container`), which this policy does not
try to reproduce.

### Still unmeasured

Two values want a probe run before a rollout is trusted:

- `grasp_hover_m` — set to the object's equator.
- `object_long_axis_local` — which local axis runs along the banana's length.
  This is only a **fallback**: if `robolab/eval/gt_state.py` exported
  `world.get_dimensions(name)` alongside pose, the long axis would be derived
  per object and the constant would stop mattering. `_long_axis_local()`
  already prefers a `dimensions` field when one is present.

## Deliberately NOT added

An automatic empty-grasp reopen (`vlm_agent` has one). RoboJEV makes this the
*model's* job — the state carries `gripper_width` and `finger_object_contacts`
and `INTENT_RULES` says to reopen and retry — so adding a reflex here would
quietly take the decision away from the policy being evaluated.

## Status

- [x] Two-stage decisions, RoboJEV schema and criteria
- [x] Normalised steps, orientation hold, action guard, measured-motion history
- [x] Offline suite incl. closed-loop pick-and-place
- [ ] Live-episode validation (incl. a `probe_axes.py` run for `grasp_hover_m`)
- [ ] More tasks (per-task intent tables + target names)
