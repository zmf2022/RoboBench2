# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a model through the Direct-API RoboLab policy client."""

import argparse
import json
import os
import re
import sys
import traceback
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

# Direct policy scripts otherwise resolve the editable ``robolab`` install
# from the main checkout. Pin imports to this worktree so code and artifacts
# come from the same repository root.
REPO_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(REPO_ROOT / ".env", override=False)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_ENV = "AI_NATIVE_MODEL"
DEFAULT_MODEL = "gpt-6-astra"


def _canonical_output_mode(value: str) -> str:
    """Accept the original CLI spelling without advertising it as canonical."""
    return {
        "action_chunk": "action_chunk_jointpos",
        "bspline": "bspline_jointpos",
        "zero_shot": "action_chunk_jointpos",
        "zero_shot_vla": "action_chunk_jointpos",
    }.get(value, value)


def _configured_model() -> str:
    return os.environ.get(MODEL_ENV) or DEFAULT_MODEL


def parse_args(argv=None) -> argparse.Namespace:
    import cv2  # noqa: F401 -- import before isaaclab
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Evaluate a model through the Direct-API /v1/chat/completions path.", allow_abbrev=False)
    parser.add_argument("--model", default=None, help="Model ID; defaults to AI_NATIVE_MODEL or the built-in model.")
    parser.add_argument("--history-offsets", nargs="+", type=int, default=None,
                        help="Exact descending control-step offsets ending in 0; overrides history length/stride.")
    parser.add_argument("--bspline-min-steps", type=int, default=None,
                        help="Decoded minimum independent of control points; omit for historical behavior.")
    parser.add_argument("--max-api-attempts", type=int, default=None,
                        help="Outer provider attempts per call; omit for unlimited attempts.")
    parser.add_argument(
        "--output-mode",
        type=_canonical_output_mode,
        choices=[
            "action_chunk_eef",
            "action_chunk_relative_eef",
            "action_chunk_delta_eef",
            "action_chunk_jointpos",
            "action_chunk_relative_jointpos",
            "action_chunk_delta_jointpos",
            "bspline_jointpos",
        ],
        default=None,
        help="Model output representation (default: action_chunk_eef).",
    )
    parser.add_argument(
        "--method",
        choices=[
            "action_chunk_eef",
            "action_chunk_relative_eef",
            "action_chunk_delta_eef",
            "action_chunk_jointpos",
            "action_chunk_relative_jointpos",
            "action_chunk_delta_jointpos",
            "bspline_jointpos",
            "action_chunk",
            "bspline",
            "zero_shot",
            "zero_shot_vla",
        ],
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the API key (default: OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="API base URL (default: OPENAI_BASE_URL, then the OpenAI API).",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=15,
        help="Maximum actions per raw chunk or B-spline call.",
    )
    parser.add_argument(
        "--min-action-horizon",
        type=int,
        default=3,
        help="Minimum model-requested actions per call (default: 3; use 1 to reproduce historical chunks).",
    )
    parser.add_argument(
        "--embodiment",
        choices=["droid"],
        default="droid",
        help="Robot/action specification. AI-native EEF output is DROID-only.",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=15,
        help="Rolling control-step observation window (default: 15).",
    )
    parser.add_argument(
        "--history-stride",
        type=int,
        default=7,
        help="Sample every Nth observation within the history window, anchored at the newest frame (default: 7).",
    )
    parser.add_argument("--image-max-side", type=int, default=None, help="Maximum encoded image side in pixels.")
    parser.add_argument(
        "--image-detail",
        choices=["low", "high", "auto"],
        default=None,
        help="Chat image detail level.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=None, help="JPEG quality for API image inputs.")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Per-request HTTP timeout.")
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries performed by the OpenAI SDK for transient failures.",
    )
    parser.add_argument(
        "--max-output-retries",
        type=int,
        default=2,
        help="Same-observation repair requests after invalid model tool output.",
    )
    parser.add_argument("--max-completion-tokens", type=int, default=None,
                        help="Output ceiling per API call (default: 8192).")
    parser.add_argument(
        "--reasoning-effort",
        default="medium",
        help="Provider-supported reasoning_effort value (default: medium).",
    )
    parser.add_argument(
        "--bspline-control-points",
        type=int,
        default=6,
        help="Future 8-D control points requested in B-spline mode.",
    )
    parser.add_argument("--bspline-degree", type=int, default=3)
    parser.add_argument(
        "--max-joint-delta",
        type=float,
        default=0.20,
        help="Safety clamp on each joint's change per control step, in radians.",
    )
    parser.add_argument(
        "--max-eef-translation-delta-m",
        type=float,
        default=0.02,
        help="Sequential translation clamp for absolute and relative EEF chunks, in metres.",
    )
    parser.add_argument(
        "--max-eef-rotation-delta-rad",
        type=float,
        default=0.10,
        help="Sequential rotation clamp for absolute and relative EEF chunks, in radians.",
    )
    parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
    parser.add_argument("--enable-debug", "--enable_debug", action="store_true")
    parser.add_argument("--record-image-data", "--record_image_data", action="store_true")
    parser.add_argument(
        "--prompt-family",
        choices=["baseline", "gripper_camera_geometry_prompt",
                 "gripper_geometry_prompt", "camera_geometry_prompt"],
        default="gripper_camera_geometry_prompt",
        help="Shared prompt family (default: gripper_camera_geometry_prompt).",
    )

    from robolab.eval.runner import add_common_eval_args

    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)

    if any(arg.split("=", 1)[0] in {"--eef-prompt-variant", "--eef_prompt_variant"} for arg in (sys.argv[1:] if argv is None else argv)):
        parser.error("--eef-prompt-variant is retired; select --prompt-family")
    args_cli = parser.parse_args(argv)
    if args_cli.num_envs < 1:
        parser.error("--num-envs must be positive")
    args_cli.enable_cameras = True
    args_cli.model = args_cli.model or _configured_model()
    legacy_output_mode = {
        "action_chunk_eef": "action_chunk_eef",
        "action_chunk_relative_eef": "action_chunk_relative_eef",
        "action_chunk_delta_eef": "action_chunk_delta_eef",
        "action_chunk_jointpos": "action_chunk_jointpos",
        "action_chunk_relative_jointpos": "action_chunk_relative_jointpos",
        "action_chunk_delta_jointpos": "action_chunk_delta_jointpos",
        "bspline_jointpos": "bspline_jointpos",
        "action_chunk": "action_chunk_jointpos",
        "bspline": "bspline_jointpos",
        "zero_shot": "action_chunk_jointpos",
        "zero_shot_vla": "action_chunk_jointpos",
    }.get(args_cli.method)
    if args_cli.output_mode is not None and legacy_output_mode is not None and args_cli.output_mode != legacy_output_mode:
        parser.error(f"--output-mode {args_cli.output_mode} conflicts with legacy --method {args_cli.method}")
    args_cli.output_mode = args_cli.output_mode or legacy_output_mode or "action_chunk_eef"
    if args_cli.history_length < 1:
        parser.error("--history-length must be a positive integer")
    if args_cli.history_stride < 1:
        parser.error("--history-stride must be a positive integer")
    if not 1 <= args_cli.min_action_horizon <= args_cli.action_horizon:
        parser.error("--min-action-horizon must be between 1 and --action-horizon")
    if args_cli.history_offsets is not None:
        offsets = args_cli.history_offsets
        if not offsets or offsets[-1] != 0 or sorted(set(offsets), reverse=True) != offsets or any(x < 0 for x in offsets):
            parser.error("--history-offsets must be unique descending nonnegative integers ending in 0")
        args_cli.history_length = max(offsets) + 1

    return args_cli


def make_client(args: argparse.Namespace, *, clients_by_env=None):
    import robolab.constants
    from policies.ai_native.direct_api import client as client_module

    client = client_module.AINativeClient(
        api_key_env=args.api_key_env,
        model=args.model,
        output_mode=args.output_mode,
        base_url=args.base_url,
        action_horizon=args.action_horizon,
        min_action_horizon=args.min_action_horizon,
        history_length=args.history_length,
        history_stride=args.history_stride,
        history_offsets=args.history_offsets,
        clients_by_env=clients_by_env,
        max_concurrent_requests=args.num_envs,
        bspline_min_steps=args.bspline_min_steps,
        max_api_attempts=args.max_api_attempts,
        image_max_side=args.image_max_side if args.image_max_side is not None else 384,
        image_detail=args.image_detail if args.image_detail is not None else "low",
        jpeg_quality=args.jpeg_quality if args.jpeg_quality is not None else 80,
        timeout_s=args.timeout_s,
        max_retries=args.max_retries,
        max_output_retries=args.max_output_retries,
        max_completion_tokens=(args.max_completion_tokens if args.max_completion_tokens is not None
                               else client_module.DEFAULT_MAX_COMPLETION_TOKENS),
        max_joint_delta=args.max_joint_delta,
        max_eef_translation_delta_m=args.max_eef_translation_delta_m,
        max_eef_rotation_delta_rad=args.max_eef_rotation_delta_rad,
        embodiment_spec=args.embodiment,
        reasoning_effort=args.reasoning_effort,
        bspline_control_points=args.bspline_control_points,
        bspline_degree=args.bspline_degree,
        inference_log_path=os.path.join(robolab.constants.get_output_dir(), "inference_calls.jsonl"),
        prompt_family=args.prompt_family,
        startup_seed=args.seed,
    )
    client.run_metadata = {
        "instruction_type": args.instruction_type,
        "renderer": args.renderer,
        "rendering_type": args.rendering_type,
        "num_envs": args.num_envs,
        "num_runs": args.num_runs,
        "run_index_offset": args.run_index_offset,
        "base_seed": args.seed,
    }
    return client


def run(args_cli: argparse.Namespace, *, clients_by_env=None) -> None:
    import cv2  # noqa: F401 -- import before isaaclab
    from isaaclab.app import AppLauncher
    import robolab.constants
    from robolab.eval.runner import run_evaluation
    from policies.ai_native.common.api_model import resolve_api_model

    args_cli.model = resolve_api_model(
        args_cli.model, base_url=args_cli.base_url,
        api_key_env=args_cli.api_key_env, timeout=args_cli.timeout_s,
    )

    # Model IDs commonly contain a provider slash. Preserve the recognizable name
    # while keeping RoboLab's output-directory and policy label path-safe.
    MODEL_POLICY = re.sub(r"[^A-Za-z0-9._-]+", "_", args_cli.model).strip("_")
    if not MODEL_POLICY:
        raise ValueError(f"{MODEL_ENV} must contain at least one letter or number")
    OUTPUT_POLICY = args_cli.output_mode
    POLICY = f"{MODEL_POLICY}_h{args_cli.history_length}_{OUTPUT_POLICY}"
    if args_cli.prompt_family is not None:
        POLICY += f"_{args_cli.prompt_family}"
    if args_cli.min_action_horizon != 1:
        POLICY += f"_amin{args_cli.min_action_horizon}"
    if args_cli.history_offsets is not None:
        POLICY += "_obs" + "-".join(map(str, args_cli.history_offsets))
    POLICY += f"_amax{args_cli.action_horizon}"

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT  # noqa: E402

        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
        robolab.constants.RECORD_IMAGE_DATA = args_cli.record_image_data
        robolab.constants.VERBOSE = args_cli.enable_verbose
        robolab.constants.DEBUG = args_cli.enable_debug

        if args_cli.output_mode in {
            "action_chunk_eef",
            "action_chunk_relative_eef",
            "action_chunk_delta_eef",
        }:
            from robolab.registrations.droid.auto_env_registrations_abs_ik import auto_register_droid_abs_ik_envs

            auto_register_droid_abs_ik_envs(
                task_dirs=args_cli.task_dirs,
                task=args_cli.task,
                cameras=WRIST_LEFT_RIGHT,
            )
        else:
            from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

            auto_register_droid_envs(
                task_dirs=args_cli.task_dirs,
                task=args_cli.task,
                cameras=WRIST_LEFT_RIGHT,
            )
        from policies.ai_native.direct_api.episode import run_direct_api_episode as episode_runner
        run_evaluation(args_cli, policy=POLICY, client_factory=partial(make_client, clients_by_env=clients_by_env), episode_runner=episode_runner)
    except BaseException as exc:
        # Isaac Sim fast shutdown may exit before the outer exception handler.
        # Save the original failure while the evaluator's output path is intact.
        failure = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        try:
            path = Path(robolab.constants.get_output_dir(), "evaluation_error.json")
            with path.open("w") as stream:
                json.dump(failure, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as save_error:
            print(f"[RoboLab] Could not save evaluation failure: {save_error}", file=sys.stderr)
        print(failure["traceback"], file=sys.stderr, flush=True)
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()


def main(argv=None) -> None:
    run(parse_args(argv))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\033[96m[RoboLab] Terminated with error: {exc}\033[0m")
        traceback.print_exc()
        sys.exit(1)
