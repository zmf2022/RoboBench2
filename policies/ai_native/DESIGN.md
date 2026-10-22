# AI-native policy design

This document describes the implemented policy inputs, outputs, and execution contracts.

The separate [Orchestrated policy](orchestrated/README.md) reviews fresh π₀.₅
H15 joint-position proposals. It retains GPT-as-Policy's `student`, `edit`, `eef`
and `stop` modes, original gate/schema and two-camera observations. Each review
selects a student prefix, a bounded trajectory edit, an EEF target, or an incomplete
stop; the shared RoboLab loop executes native joint actions.
Its host rejects interventions without an observed failure or misaligned next
intent. It does not expose native progress, success, scores or remaining steps
to the reviewer. `--review-backend codex` keeps a persistent App Server thread;
`--review-backend api` sends a full episode conversation through Chat Completions,
including review tool calls and their host validation receipts. Both use the
same execution state machine. Runtime integration is implemented; live validation
is pending.

## Policy surface

| Policy | Backend | Model prompt | Model input | Model output | Robot controller |
|---|---|---|---|---|---|
| `action_chunk_eef` | Direct-API | Prompt + absolute gripper-flange pose instructions | Task instruction + recent RGB views + robot state | 3–H explicit absolute EEF targets by default | DROID absolute differential IK |
| `action_chunk_relative_eef` | Direct-API | Prompt + observation-anchored relative flange-transform instructions | Task instruction + recent RGB views + robot state | 3–H relative SE(3) waypoints against the newest observed flange pose | Decode once to absolute flange targets; DROID absolute differential IK |
| `action_chunk_delta_eef` | Direct-API | Prompt + stepwise delta flange-transform instructions | Task instruction + recent RGB views + robot state | 3–H SE(3) deltas against the preceding target pose | Integrate to absolute flange targets; DROID absolute differential IK |
| `action_chunk_jointpos` | Direct-API | Prompt + absolute arm-joint target instructions | Task instruction + recent RGB views + robot state | 3–H explicit absolute joint targets by default | DROID joint-position control |
| `action_chunk_relative_jointpos` | Direct-API | Prompt + observation-anchored relative joint instructions | Task instruction + recent RGB views + robot state | 3–H joint displacements against the newest observed joint state | Decode once to absolute targets; DROID joint-position control |
| `action_chunk_delta_jointpos` | Direct-API | Prompt + stepwise delta joint instructions | Task instruction + recent RGB views + robot state | 3–H joint displacements against the preceding target | Integrate to absolute targets; DROID joint-position control |
| `bspline_jointpos` | Direct-API | Prompt + joint-space B-spline instructions | Task instruction + recent RGB views + robot state | Six future joint-space control points and 6–H output steps by default | DROID joint-position control |
| `agent_eef` | Codex App Server | Agent-EEF baseline or geometry prompt + Codex memory guidance | Task instruction + recent RGB views + EEF state + thread history | One fixed, complete EEF target plus 1–5 steps | Bounded DROID local DLS to joint-position control |

## Observations

Every observation pairs camera views with robot state from the same control step.

| Backend | Camera views | Robot state | History |
|---|---|---|---|
| Direct-API | DROID left shoulder, right shoulder, and wrist | Full joint, gripper, and end-effector state | Rolling window; each request is independent |
| Agent-EEF | DROID left shoulder, right shoulder, and wrist | End-effector pose and gripper state only | Persistent episode thread with saved observations |

`--history-length` sets the observation window. Direct-API can sample that window with `--history-stride`, always including the latest observation. Its default 15-step window with stride 7 sends steps `t-14`, `t-7`, and `t`.

Direct-API retains observations, but not earlier model replies. Agent-EEF keeps the thread and can reopen saved observations through its bounded memory tools. It does not receive joint state, depth, object poses, predicates, or other simulator ground truth.

## Prompt variants

### Shared prompt variants
Each method still adds its own action instructions afterward.

| Variant | Used by | Summary | Code |
|---|---|---|---|
| `baseline` | All Direct-API methods | [Robot and end-effector frame](common/prompts.py#L45) + [execution timing](common/prompts.py#L59). No grasp recipe, geometry, or action memory. | [`baseline_prompt`](common/prompts.py#L219) |
| `gripper_geometry_prompt` | All Direct-API methods | Baseline plus flange-to-pad and jaw geometry. | [`gripper_geometry_prompt`](common/prompts.py) |
| `camera_geometry_prompt` | All Direct-API methods | Baseline plus camera calibration, without pad geometry. | [`camera_geometry_prompt`](common/prompts.py) |
| `gripper_camera_geometry_prompt` | All Direct-API methods and `agent_eef` | `baseline` + [gripper geometry](common/prompts.py#L70) + [camera geometry](common/prompts.py#L101). Agent-EEF adds the geometry blocks to its own baseline. | [`gripper_camera_geometry_prompt`](common/prompts.py#L372) |

### Prompts by method

| Method | Prompt choices | Method-specific addition | Code |
|---|---|---|---|
| `action_chunk_eef` | Shared prompt family | Absolute gripper-flange pose sequence | [`_method_prompt`](direct_api/client.py#L723) |
| `action_chunk_relative_eef` | Shared prompt family | Relative SE(3) flange waypoints sharing one observation anchor | [`_method_prompt`](direct_api/client.py#L723) |
| `action_chunk_delta_eef` | Shared prompt family | Stepwise SE(3) flange deltas composed from each preceding target | [`_method_prompt`](direct_api/client.py#L723) |
| `action_chunk_jointpos` | Shared prompt family | Absolute arm-joint target sequence | [`_method_prompt`](direct_api/client.py#L723) |
| `action_chunk_relative_jointpos` | Shared prompt family | Relative arm-joint waypoints sharing one observation anchor | [`_method_prompt`](direct_api/client.py#L723) |
| `action_chunk_delta_jointpos` | Shared prompt family | Stepwise arm-joint displacements accumulated from each preceding target | [`_method_prompt`](direct_api/client.py#L723) |
| `bspline_jointpos` | Shared prompt family | Joint-space control points for a smooth trajectory | [`_method_prompt`](direct_api/client.py#L723) |
| `agent_eef` | [Agent-EEF prompt](common/agent_eef.py#L408), optionally with gripper and camera geometry, plus [Codex memory guidance](agentic/app_server.py#L656) | One complete end-effector target held for a short segment | [Request assembly](agentic/session.py#L92) |

After the prompt, each request adds the task instruction, recent measured robot state, and camera observations. Direct-API assembles this in [`AINativeClient._pack_request`](direct_api/client.py#L600). Agent-EEF assembles it in [`agentic/session.py`](agentic/session.py#L43).

## Formulations

Action-chunk terminology:

- `relative_chunking`: all future poses relative to one reference pose.
- `delta_chunking`: each future pose relative to the preceding pose.

The `action_chunk_relative_*` methods use `relative_chunking`, with the newest synchronized observation as the one reference pose or joint state. The `action_chunk_delta_*` methods use `delta_chunking`, starting from that newest observation and integrating rows sequentially.

| Method | Model output | Robot execution |
|---|---|---|
| `action_chunk_eef` | 3–15 absolute gripper-flange poses by default | Limit each pose change, execute one pose per control step, then replan |
| `action_chunk_relative_eef` | 3–15 transforms `inverse(T_anchor) @ T_target[k]` by default | Compose every row with the same newest observed flange pose, limit sequential decoded pose changes, execute, then replan |
| `action_chunk_delta_eef` | 3–15 transforms `inverse(T_target[k-1]) @ T_target[k]` by default | Compose rows sequentially from the newest observed flange pose, limit decoded pose changes, execute, then replan |
| `action_chunk_jointpos` | 3–15 absolute arm-joint targets by default | Limit each joint change, execute one target per control step, then replan |
| `action_chunk_relative_jointpos` | 3–15 displacements `q_target[k] - q_anchor` by default | Add every row to the same newest observed joint state, limit sequential decoded target changes, execute, then replan |
| `action_chunk_delta_jointpos` | 3–15 displacements `q_target[k] - q_target[k-1]` by default | Accumulate rows from the newest observed joint state, limit decoded target changes, execute, then replan |
| `bspline_jointpos` | Six joint-space control points and a 6–15 step horizon | Build a smooth trajectory from the current state, execute it, then replan |
| `agent_eef` | One absolute end-effector target and a 1–5 step duration | Validate the target, hold it for the requested duration, then replan |

The complete mathematical definition and decoding procedure are in [BSPLINE.md](direct_api/BSPLINE.md). Safety and repair behavior is described in [Validation and failure behavior](#validation-and-failure-behavior).

## Validation and failure behavior

Direct-API output is validated completely before the simulator advances. A malformed completion, malformed JSON, wrong tool, invalid shape, non-finite value, missing/extra field, or out-of-range horizon triggers another request on the unchanged observation. The default is two repair requests after the first attempt. If all attempts fail, the client emits one measured-state safe hold and replans. Partially decoded output is never executed. An API request that still fails after SDK retries produces the same one-step measured-state hold and replans instead of terminating the worker.

Agent-EEF validates each complete target in the host. Invalid or unsafe calls do not go directly to the controller. The host records feedback for the same thread and holds safely. Malformed model output never kills the evaluation job or partially steps the simulator.

## Timing and diagnostics

Direct-API `inference_calls.jsonl` records `request_build_s` for history selection and JPEG/request construction, `api_latency_s` for provider calls and repairs, and `latency_s` for their sum. It also records attempt count, repair errors, decoded actions, selected horizon, and whether safety changed the chunk. Each record contains the usage reported by all model requests for that inference, including output-repair requests.

Agent-EEF records finer client-observed phases: observation save, App Server cold start, process/RPC setup, time to first event, time to policy tool call, validation/decode, total transport, total inference, and action execution wall and simulated durations. Provider queueing, reasoning, network, and accelerator time cannot be separated without provider telemetry.

Every AI-native formulation writes an aggregate `token_usage` object into its per-environment task log (`log_<run>_env<id>.json`) and its episode summary. The fields are `inference_calls`, `model_requests`, `requests_with_usage`, `requests_missing_usage`, `input_tokens`, `cached_input_tokens`, `uncached_input_tokens`, `output_tokens`, `reasoning_tokens`, and `total_tokens`. Cached tokens are part of input tokens, and reasoning tokens are part of output tokens; neither is added again to `total_tokens`. Values come from the provider response or Codex App Server usage event. A request whose backend reports no usage increments `requests_missing_usage` instead of guessing a token count.

## Run commands

One direct method:

```bash
python policies/ai_native/direct_api/run.py \
  --headless \
  --task BananaInBowlTask \
  --history-length 15 \
  --history-stride 7 \
  --prompt-family baseline \
  --output-mode action_chunk_eef
```

Persistent Agent-EEF:

```bash
python policies/ai_native/agentic/run.py \
  --headless \
  --embodiment droid \
  --task BananaInBowlTask \
  --history-length 1 \
  --model gpt-6-astra-medium \
  --allow-model-requests
```
