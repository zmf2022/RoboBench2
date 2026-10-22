# Orchestrated Policy

This first implementation combines fresh π₀.₅ proposals with a persistent GPT
reviewer, using RoboLab's existing evaluation, recording and termination loop.
The policy follows [GPT-as-Policy's RoboLab implementation](https://github.com/anonymous-report-421/GPT-as-Policy/tree/main/hybrid_rollout/robolab).
Its gate instruction/validation, response schema and trajectory-edit algorithm
are copied from that implementation, under the [upstream MIT license](https://github.com/anonymous-report-421/GPT-as-Policy/blob/main/LICENSE).
Direct-API conventions apply only to the API reviewer transport and credentials.

CPU contract tests cover proposal freshness, gating, bounded corrections,
history isolation, failure handling and robot-only FK. Real model + OpenPI +
Isaac execution has not yet been validated; no performance result is claimed.

## Control contract

1. Capture left shoulder and wrist RGB, measured arm joints, gripper and
   Robotiq `base_link` pose in the robot-root frame. Quaternions use WXYZ.
2. Query π₀.₅ with its original instruction and existing two-camera OpenPI wire
   format. Require one finite H15 × 8 absolute joint-position chunk.
3. Preview the candidate flange trajectory through robot-only FK. Verify current
   FK against measured flange pose (less than 2 mm and 0.01 rad error) before
   sending any candidate to the reviewer. This does not simulate future objects,
   contact, tracking or collision.
4. Review the last execution outcome and the next proposal's likely task intent.
   Choose `student` (a 1–15-step prefix), `edit` (1–5 proposal-relative EEF targets),
   `eef` (one fixed target, 1–5 steps), or `stop` (incomplete episode).
   Both edit and EEF takeover require observed failure or misaligned next intent;
   uncertainty alone is insufficient. The host enforces this gate.
5. Execute one native action per shared-loop step. Absolute EEF targets are bounded
   to 5 cm / 0.35 rad from measured pose. Edits apply a root-frame translation
   and rotation-vector offset, bounded to the same magnitudes, to the proposal's
   FK targets. Offsets ramp by `(i+1)/steps`; gripper overrides apply immediately.
   Edited targets are not constrained to 5 cm of the current pose. Both correction
   modes use the existing DROID DLS adapter, recomputed
   from measured state each control step. They are targets, not arrival guarantees.
6. After execution, discard the unconsumed proposal and query π₀.₅ again from
   the new observation. A fresh proposal is required even after a correction.

Only RoboLab determines native success and termination. Model inputs exclude
object ground truth, native subtask predicates, scores, success flags, termination
flags and remaining-step counts. Reviewer task progress is its own visual
assessment. It does not access the evaluator's progress state. The student keeps
the original instruction; temporary reviewer subgoals do not replace it.

Invalid reviews are repaired on the unchanged observation and proposal, without
stepping physics. A model stop or exhausted review budget records an incomplete
episode (`success: null`) and continues with the next run/task. Repair exhaustion,
transport, FK and controller errors remain fatal. No holds or replay are inserted.

This backend supports one DROID environment and a persistent reviewer with all
four upstream decision modes. Multi-env execution, video demonstrations and
selected-history ablations are not part of this backend.

## Integration boundary

The policy retains the original two camera views, H15 proposals, outcome/intent
gate, control limits and fresh-inference requirement after every execution.
The original gate text is included unchanged in each reviewer's system prompt;
the uniform response includes `edit` and `target` in all modes, ignoring unused
fields. It does not use Direct-API's action representations, short observation
window, or 384-pixel/low-detail defaults.

Necessary differences are confined to these integration boundaries:

- The existing RoboLab loop owns stepping, reset, native termination and videos,
  replacing upstream's standalone simulator RPC. A `stop` is exported as an
  incomplete run through this host, never a native failure or success.
- For full-benchmark evaluation, the default review budget follows each task's
  maximum control steps instead of upstream's fixed 180 decisions. Set
  `--max-reviews 180` to use the upstream budget.
- Current Isaac Lab array/quaternion adapters and bounded DROID IK execute the
  same control contracts on the installed simulator version.
- The current project's model-input restrictions exclude evaluator-only native
  progress, success and remaining-step fields that upstream service packets carry.
- Codex captures one review per host turn; API replays an episode conversation
  with tool receipts. This replaces the original full Codex agent's three blocking
  service tools. API provides no shell/image-analysis tools; Codex tools remain
  optional and restricted by the existing local harness.

These differences mean this is an integration of the original policy, rather
than a reproduction of the original experiment or its full agent runtime.

## Setup and run

Use the repository `.venv`. The bundled [`openpi-client`](../../../packages/openpi-client/README.md)
is installed by the root `uv sync --extra isaac60 --extra test`; for an existing
environment use `uv pip install --python .venv/bin/python -e packages/openpi-client`
from the repository root, as described in the [π₀ family setup](../../pi0_family/README.md). OpenPI/JAX
and the `pi05_droid_jointpos` checkpoint stay in a separate server environment.
Start that server using the same instructions. Select `--review-backend codex`
(default, existing Codex login) or `--review-backend api` (direct API credentials).

### Direct API reviewer

The API reviewer calls `POST <base_url>/chat/completions` with two JPEG images,
robot state and the candidate trajectory. It uses the existing review schema as
a function tool, validates exactly one complete call, and pairs the assistant
tool call with the host's accepted/rejected receipt before the next observation.
See the official [function-calling format](https://developers.openai.com/api/docs/guides/function-calling).
No Codex executable, login, shell, image-inspection or memory-retrieval tools are
required or exposed by this backend.

```bash
source .venv/bin/activate
export OPENAI_API_KEY="your-api-key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export AI_NATIVE_MODEL="gpt-6-astra"
python policies/ai_native/orchestrated/run.py \
  --review-backend api --reasoning-effort medium \
  --max-completion-tokens 8192 \
  --headless --task RubiksCubeTask --seed 0 --num-envs 1 --num-runs 1 \
  --remote-host localhost --remote-port 8000 --allow-model-requests \
  --video-mode sensor
```

API configuration follows [Direct-API](../direct_api/README.md#setup), including
`.env` loading and `/v1/models` discovery. A single model is selected automatically;
multiple models require `--model` or `AI_NATIVE_MODEL`. Use the provider's exact
ID without a Codex effort suffix. Query failures stop startup.

For local serving, set `OPENAI_BASE_URL` to `http://host:port/v1`; use key `local`
only without authentication. `--reasoning-effort none` omits the effort field.
Default output names are `<timestamp>_orchestrated_pi05_<api-or-codex>_<model>`.

Both backends retain the upstream two views and 896-pixel maximum side
(896×504 for the standard 16:9 cameras). JPEG quality defaults to 85, and API
detail defaults to `high`; encoding these views is part of the LLM transport.
`--image-max-side`, `--jpeg-quality` and `--image-detail` can override the encoding.
`--timeout-s` sets the request timeout (default 120).

All previous observations (including images), assistant reviews and host receipts
are resent within each episode. History is neither silently truncated nor
compacted. Context or transport errors abort the evaluation. SDK transport retries
are disabled, so each request has an explicit usage record; malformed model
outputs use `--max-output-retries` on the unchanged observation/proposal.
`--optional-tools` is rejected for the API backend. API sessions are discarded
at episode reset; this backend never falls back to Codex.

### Codex reviewer

This keeps the existing Agentic model/effort options and Codex authentication.

```bash
source .venv/bin/activate
python policies/ai_native/orchestrated/run.py \
  --review-backend codex \
  --headless --task RubiksCubeTask --seed 0 --num-envs 1 --num-runs 1 \
  --remote-host localhost --remote-port 8000 \
  --model gpt-6-astra-medium --allow-model-requests \
  --video-mode sensor
```

`--remote-uri` overrides the student host/port. The default review budget is the
task's maximum control steps, with two output-repair requests per proposal.
Use `--optional-tools` to enable restricted shell/Python and image tools with the Codex
backend. History and memory remain persistent. Network, MCP and delegation tools
remain disabled.

Control timing is validated against the simulator.
The USD FK chain currently expects the standard DROID prim at
`/World/envs/env_0/robot` with `panda_link0` as its robot-root frame. A mismatched
asset/frame fails the measurement check rather than supplying a false preview.

## Artifacts and evidence

The task output directory contains:

- `orchestrated_config.json`: runtime options, reviewer prompt and tool schema.
- `orchestrated_events_<episode>.jsonl`: incremental proposals, review outcomes,
  provider usage, issued controls and next-observation confirmations.
- `orchestrated_decisions_<episode>.json`: proposals, selected actions, control
  sources, IK diagnostics and aggregate metadata, also exported after errors.
- Native RoboLab results and videos, produced by the unchanged shared loop.
- `agent_memory/orchestrated-droid/...`: Codex reviewer prompts, images, memory
  and transport artifacts, using the existing memory-root resolution.
- `agent_memory/orchestrated-api-droid/...`: API prompts, images, provider
  responses and host feedback; credentials are not archived.

`issued_control_steps` counts actions returned to the host. `observed_control_steps`
counts those followed by a new measured observation. The terminal action may
have no subsequent policy observation; the log intentionally does not manufacture
that acknowledgement. Neither counter establishes task success. Codex token
usage uses cumulative-thread differences; API usage sums each request's reported
usage. Both include output repairs and track requests with missing usage.

For a fair evaluation, compare π₀.₅, existing Agentic and Orchestrated with matched
actual initial states, instructions and simulator horizons, and record review
budgets, latency, usage and intervention frequency. Equal seeds alone do not
establish equal initial states across implementations.

## Offline tests

These tests do not load the simulator test suite's `tests/conftest.py`, start a
GPU application, contact a policy server or make model requests:

```bash
.venv/bin/python -m pytest policies/ai_native/orchestrated/tests -q
```
