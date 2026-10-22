# AI-Native Direct-API Client

This backend lets RoboLab call a remote model endpoint directly with an API key. It uses an OpenAI-compatible Chat Completions API:

```text
POST <base_url>/chat/completions
Authorization: Bearer <api_key>

{
  "model": "...",
  "messages": [...],
  "tools": [...],
  "tool_choice": "required",
  "max_completion_tokens": 8192
}
```

“OpenAI-compatible” describes this request/response format. It does not require OpenAI to host the endpoint. Configure `base_url`, the model ID, and the API key for your chosen provider.

Runs default to an **8,192-token completion budget per API call**. This is an upper bound, not a requested response length. Use `--max-completion-tokens` to override it.

When inference logging is enabled, `inference_calls_provider_requests.jsonl` preserves per-request provider usage (including cache-write TTL details), finish reasons, returned model identity, requested completion budget and reasoning effort, and latency. It includes repair requests and failed outer attempts with missing usage; prompts, images, and credentials are not stored in this sidecar.

The backend supports seven outputs:

- `action_chunk_eef`: 3–H explicit absolute end-effector targets by default.
- `action_chunk_relative_eef`: 3–H observation-anchored relative SE(3) end-effector waypoints by default.
- `action_chunk_delta_eef`: 3–H stepwise SE(3) end-effector deltas by default.
- `action_chunk_jointpos`: 3–H explicit absolute joint targets by default.
- `action_chunk_relative_jointpos`: 3–H observation-anchored relative joint waypoints by default.
- `action_chunk_delta_jointpos`: 3–H stepwise joint-position deltas by default.
- `bspline_jointpos`: six future joint-space B-spline control points decoded to 6–H actions by default.

Relative chunks use one frozen anchor from the newest observation in the API request. For joints, every row is `q_target[k] - q_anchor`. For EEF, every row is `inverse(T_anchor) @ T_target[k]`, encoded as local translation plus an axis-angle rotation vector. Rows are not cumulative increments. The client decodes the whole chunk to absolute controller targets once, then applies the same sequential motion limits used by the corresponding absolute mode. The gripper remains an absolute binary command in both relative modes.

Delta chunks instead start from the newest observation and integrate in order. For joints, `q_target[k] = q_target[k-1] + dq[k]`. For EEF, `T_target[k] = T_target[k-1] @ D[k]`. The reconstructed absolute trajectory is then passed through the same sequential motion limits; gripper remains absolute.

`--min-action-horizon` defaults to **3**, and `--action-horizon` defaults to **15**. The prompt, tool schema, and local response validation enforce these bounds. Shorter model responses enter the existing output-repair flow; they are not padded or repeated. B-splines retain their six-step default minimum, or the configured minimum if it is larger. A failed response can still produce the existing one-step safe-hold fallback, which is logged as contamination. Episode termination can also stop execution before an accepted chunk ends.

`--history-length N` defines a rolling window of the latest `N` control-step observations. `--history-stride S` samples backward from the newest observation inside that window. Every selected observation sends three JPEGs and one matching `proprio_obs` record, ordered oldest to newest. The defaults `N=15` and `S=7` select steps `t-14`, `t-7`, and `t` once the window is full: 9 images and 3 proprioception records spanning about 0.93 seconds at 15 Hz. `N=1` sends the current three-image set and current proprioception only. No video is sent.

See [DESIGN.md](../DESIGN.md) for exact request, response, and simulator input structures. The separate persistent [Agentic backend](../agentic/README.md) does not use this API-key client.

## Setup

Activate the repository environment before running: `source .venv/bin/activate`.

Configure any endpoint that implements the required Chat Completions fields:

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://your-provider.example/v1"
export AI_NATIVE_MODEL="provider/model-id"
```

`OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`. Startup queries
`/v1/models`: a single model is selected automatically; multiple models require
a matching `--model` or `AI_NATIVE_MODEL` (default `gpt-6-astra`). Requests, logs
and default output names use the selected ID. Query failures stop startup.

List models visible to the key:

```bash
python policies/ai_native/direct_api/list_models.py
```

## Run one method

```bash
python policies/ai_native/direct_api/run.py \
  --headless \
  --prompt-family gripper_camera_geometry_prompt \
  --task BananaInBowlTask \
  --num-envs 1 \
  --num-runs 1 \
  --history-length 15 \
  --history-stride 7 \
  --output-mode action_chunk_eef \
  --action-horizon 15 \
  --min-action-horizon 3 \
  --reasoning-effort medium
```

## Prompt families

The runner defaults to `gripper_camera_geometry_prompt`. Override it with `--prompt-family`. The Python client also accepts a custom `system_prompt` instead of a family.

| Family | Content |
| --- | --- |
| `baseline` | Action/frame contract and execution timing |
| `gripper_geometry_prompt` | Baseline plus gripper geometry |
| `camera_geometry_prompt` | Baseline plus camera geometry |
| `gripper_camera_geometry_prompt` | Baseline plus both geometry blocks |

All families support all seven action representations and omit grasp sequencing and action memory. Camera-geometry families validate the camera configuration before evaluation.

## Safety and logs

The complete model tool call is validated before RoboLab advances. Invalid output is repaired on the unchanged observation. If repair fails, the client sends one measured-state safe hold and replans. Partial output is never sent to the simulator. API failures (including HTTP 400/429 and timeouts) print the episode, inference call, error, and retry count to the terminal. After SDK retries, the client keeps retrying the same request with delays of 5, 10, 20, 40, then 60 seconds until the provider responds or the run is interrupted. Simulation time does not advance while waiting. This also applies to API failures during output repair; they do not consume the output-repair budget. Persistent configuration or quota errors therefore wait for operator/provider intervention instead of finishing the episode with hold actions.

Failed attempts are written immediately to `inference_calls_api_retries.jsonl`. The final inference record and episode task log include `api_retry_count`. `--max-retries` still controls the SDK's retries within each outer attempt; the outer retry loop defaults to unlimited attempts. Set `--max-api-attempts N` to fail the worker after N outer attempts for a request.

Each inference is recorded in `inference_calls.jsonl`, including provider-reported input, cached-input, output, reasoning, and total tokens. The same fields are summed across the episode under `token_usage` in each `log_<run>_env<id>.json` task log. Output-repair requests count as model requests and their tokens are included. Each inference record splits `request_build_s` (history selection and image encoding) from `api_latency_s`; `latency_s` is their sum.

## Runtime configuration

Use the following flags to configure the policy.

| Setting | Default | Flag |
| --- | --- | --- |
| Prompt | Gripper + camera geometry | `--prompt-family` |
| Actions | Absolute EEF, 3–15 targets | `--output-mode`, `--min-action-horizon`, `--action-horizon` |
| Observation history | t−14, t−7, t | `--history-offsets` or `--history-length` / `--history-stride` |
| Images | 384 pixels, JPEG 80, low detail | `--image-max-side`, `--jpeg-quality`, `--image-detail` |
| Reasoning | Medium | `--reasoning-effort` |
| Task instruction | Default wording | `--instruction-type default`, `specific`, or `vague` |
| B-spline control points / degree | 6 / 3 | `--bspline-control-points`, `--bspline-degree` |
| Completion budget | 8,192 tokens | `--max-completion-tokens` |
| Outer provider attempts | Unlimited | `--max-api-attempts` |
| SDK retries / output repairs | 2 / 2 | `--max-retries`, `--max-output-retries` |

`--history-offsets` samples control steps, including observations captured while executing a chunk. Frames are sent oldest first; unavailable startup timestamps are omitted. Explicit offsets take precedence over length/stride sampling. `--bspline-min-steps` separates decoded horizon from control-point count.

## Parallel execution

Use `--num-envs N` to run multiple environments in one simulator process with the configured API credential. Each environment keeps its own observation history, action chunk, and retry/token counters. Requests overlap; the simulator advances only after all active environments have actions, so one slow request can delay the batch. Size concurrency to GPU memory and the provider's request limits.

The Python client accepts `clients_by_env`, a mapping from environment IDs to preconstructed API clients, for custom transport routing. Set `max_concurrent_requests` to the desired concurrency; the runner uses `--num-envs`. Callers own injected clients and close them after evaluation; the policy closes only the API client it constructs itself.

Each task writes `resolved_config.json`, `prompt_variant.json`, inference logs, and episode artifacts. Logs include actual sampled timestamps, action counts, and the configuration hash. A different configuration cannot overwrite an existing configuration in the same task folder.

## Evaluation integration

The runner keeps the shared `make_client(args)` interface. The Direct-API episode wrapper calls `client.configure_environment(env_cfg)` before entering RoboLab's shared simulation and recording loop. This setup prepares prompt timing, validates the applicable camera calibration, and writes prompt/configuration artifacts. Episode metadata is collected after client cleanup and before result summarization. These lifecycle methods belong to the concrete Direct-API client.

See [DESIGN.md](../DESIGN.md) for the model input, output, and simulator action contracts and [BSPLINE.md](BSPLINE.md) for the spline formulation.
