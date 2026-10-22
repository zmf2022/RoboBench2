# Astra Agentic Policy

Astra controls a robot through RoboLab's existing `InferenceClient` and `run_evaluation` interfaces. Codex manages the model session and proposes actions; RoboLab validates and executes them and decides task termination. Each action is one absolute target held for 1–5 control steps at 15 Hz.

Model inputs include camera images, measured robot state, the task instruction, and optional measured motion feedback. Native subtask progress, status, scores, remaining-step updates, termination flags and task-specific success-criterion metadata are excluded from model inputs. RoboLab still computes and records native results.

The published original/reference runs predate this restriction and exposed native progress and success-criterion metadata. This release removes those inputs from both the original and ablation interfaces. The current evaluator also uses its updated reset/seeding protocol rather than the historical per-episode reset seeds. Published results have not been reevaluated with these changes.

## Prerequisites

Follow the repository [installation instructions](../../../README.md) on a Linux GPU machine with IsaacSim, IsaacLab and the RoboLab assets. The commands below assume Codex is already on `PATH` and signed in. The runner inherits your existing Codex configuration and login, including `CODEX_HOME` if set; no separate API key or login directory is required. This implementation was developed against Codex 0.149.1; compatibility with other versions has not been established here.

See the official [authentication instructions](https://developers.openai.com/codex/auth) and [configuration reference](https://developers.openai.com/codex/config-reference). The runner selects the `openai` provider; your account must have access to the selected model. GPT-6 Astra and GPT-6.1 Sol are supported with `low`, `medium`, `high`, `xhigh` and `max` reasoning suffixes. A bundled client model catalog is not proof of account access. Provider-specific credentials, endpoints and model metadata are not included in this release.

## Run one episode

From the repository root, using the Python environment from installation:

```bash
python policies/ai_native/agentic/run.py \
  --model gpt-6.1-sol-medium \
  --headless --embodiment droid --task RubiksCubeTask \
  --seed 0 --num-envs 1 --num-runs 1 \
  --video-mode sensor --renderer realtime \
  --output-folder-name astra-example --allow-model-requests
```

The example explicitly selects GPT-6.1 Sol with medium reasoning. Omitting `--model` uses the runner default, `gpt-6-astra-medium`, rather than the default model in your Codex configuration. Existing Codex authentication is still reused.

`--allow-model-requests` enables real model requests using your current Codex credentials. Use a distinct output folder for each parallel process. The command runs one task and seed; this release does not include a cloud scheduler.

## Runtime configuration

Use the following flags to configure the policy.

| Setting | Default | Flag |
| --- | --- | --- |
| Prompt | Gripper + camera geometry | `--prompt-family` |
| Explicit planning | Off | `--explicit-planning` / `--no-explicit-planning` |
| Measured joint-state input | Omitted for EEF actions | `--include-joint-state` / `--no-include-joint-state` |
| Model decisions | At most 150 | `--max-llm-calls` |
| Action | Absolute EEF target, held 1–5 steps at 15 Hz | `--action-space`, `--max-target-steps`, `--control-hz` |
| History | Current observation and persistent conversation | `--history-length`, `--history-offsets`, `--persistent-history` / `--no-persistent-history` |
| Text history replay | None | `--history-text none`, `full`, or `sampled` |
| Image history replay | None | `--history-images none` or `full` |
| Task instruction | Default wording | `--instruction-type default`, `specific`, or `vague` |
| Motion feedback | On | `--motion-feedback` / `--no-motion-feedback` |
| Optional shell/image tools | Off | `--optional-tools` / `--no-optional-tools` |
| Images | 640 pixels, JPEG 85 | `--image-size`, `--jpeg-quality` |
| Model | GPT-6 Astra medium | `--model` |
| Video demonstration | Off | `--demonstration-manifest` |

The geometry prompt describes DROID. Joint targets require measured joints, so `--action-space joint` always includes them. `--no-geometry` removes geometry blocks from the selected prompt while retaining its control and timing instructions. Written subgoals remain allowed unless `--no-decomposition-scaffolding` is selected; this is distinct from asking for an explicit plan.

Explicit history offsets take precedence over `--history-length`. Selected history channels require `--no-persistent-history` and disabled optional tools. Full visual history uses one current observation per decision. Demonstrations require persistent history. Invalid combinations fail before simulation starts.

### Context History

Every decision receives current camera images, measured robot state and current motion feedback. Text history retains prior assistant messages, plans, action calls and opaque native reasoning records. Visual history retains camera images from earlier decisions. Historical user robot-state and motion-feedback payloads are omitted; historical action-tool replies are neutral acknowledgements. Measurements mentioned by the assistant are not scrubbed from its messages.

Setting `--history-text` or `--history-images` to a value other than `none` creates a fresh Codex thread per decision and replays only the selected history. Notes, archive-retrieval tools and optional shell/image tools are disabled. Image offsets t−30 and t−15 are measured in simulator control steps, not model decisions. Full visual history means every decision-time observation, not every simulator frame.

`--history-text sampled` pairs each past sampled image with the latest assistant turn at or before that control step. Duplicate matches are replayed once; the current observation has no assistant action yet.

**Context capacity:** long histories require verified provider input capacity and matching Codex model metadata. Codex 0.149.1 can use a smaller fallback context window for an unknown model. Configure an appropriate `model_catalog_json` only from verified metadata; this release does not hard-code a provider-specific window. Native compaction is rejected so a run cannot silently become a different history condition. If the selected history exceeds the supported window, record an incomplete run rather than truncating history or counting it as a task failure.

### Video demonstration

Prepare sampled frames from a three-camera demonstration video. Install FFmpeg with your system package manager and ensure both `ffmpeg` and `ffprobe` are on `PATH`. Frame extraction uses those executables; RoboLab does not package or download FFmpeg.

```bash
ffmpeg -version
ffprobe -version
python policies/ai_native/agentic/tools/prepare_demonstration.py \
  --video /path/to/three-camera-rollout.mp4 \
  --control-hz 15 --output /path/to/demo
```

Add `--demonstration-manifest /path/to/demo/manifest.json` to the episode command. Frames are sampled at 1 Hz, including the final recorded frame, with at most 120 montages. The demonstration supplies visual task specification instead of task-specific text; live images, measured state and feedback remain.

This is visual in-context learning (ICL). The demonstration contains images and timestamps, without recorded robot actions. It is prepended to the first live observation and retained in the conversation, rather than appended again at every decision.

## Outputs

Native results and videos are saved under the chosen RoboLab output folder. Agent traces are stored alongside them in:

```text
agent_memory/<policy>/episode_000000/env_000/attempt_000/
```

Setup saves `system_prompt.txt` and `prompt_variant.json`, including the selected runtime options. Each session also records `options.json`.

The backend-specific episode wrapper configures the environment and binds the controller before inference, then exports `episode_metadata` and diagnostics after cleanup. Instruction fields follow RoboLab's tracked-input interface; the demonstration input does not read task-language fields.

Traces contain observations, action proposals, host results, JSON-RPC records and usage events. History decisions have separate attempt directories. Tools-enabled runs additionally expose `workspace.json`, `scratch/` and notes.

Use native results for success and completion score. Keep infrastructure errors and incomplete episodes separate from native task failures; do not retry a native failure to improve the reported score. For cost analysis, use final cumulative provider usage. Reasoning tokens are already included in output tokens.

This backend implements the Agentic Policy. Orchestrated Policy is a separate implementation and is not included here.

## Evaluation integration

The runner keeps the shared `make_client(args)` interface. The Agentic episode wrapper calls `client.configure_environment(env_cfg)` and binds the execution adapter before entering RoboLab's shared simulation and recording loop. On DROID, the adapter converts end-effector targets to bounded joint actions; `joint-target` actions bypass IK and use bounded joint targets directly.

Metadata and diagnostics are collected after client cleanup. These lifecycle methods belong to the concrete Agentic client; they do not add optional hooks to `InferenceClient`. See [DESIGN.md](../DESIGN.md) for the action contracts.

## Code layout

```text
agentic/
  run.py, run_config.py             Entry point and CLI configuration
  client.py                        RoboLab policy loop and execution
  episode.py, setup.py             Environment setup, metadata and diagnostics
  session.py, app_server.py        Model inputs, tools and Codex transport
  prompts.py                       Policy and tool instruction construction
  specs.py, droid_joint_target.py   Robot interface and EEF control
  joint_actions.py                 Joint-target validation and control
  runtime_options.py               Independent runtime options and validation
  coding_workspace.py              Optional coding workspace
  demonstration.py                 ICL demonstration loading
  history_channels.py              Conversation history selection
  tools/prepare_demonstration.py    Offline frame preparation
```
