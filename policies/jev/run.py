# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the Jev two-stage policy across registered tasks."""

import argparse
import logging
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "jev"

parser = argparse.ArgumentParser(description="Evaluate a Jev-style numeric-state policy.")
parser.add_argument("--vlm-url", "--vlm_url", default="http://localhost:8000/v1",
                    help="OpenAI-compatible chat-completions base URL (default: http://localhost:8000/v1).")
parser.add_argument("--model", default=None,
                    help="Model name (auto-detected from the server's /models if omitted; "
                         "for --backend jev, the TypeSafe model ID, default jev-1.13.0).")
parser.add_argument("--backend", choices=["chat", "jev"], default="chat",
                    help=("Decision backend: 'chat' runs intent+motor on a local "
                          "OpenAI-compatible server (no key needed); 'jev' calls "
                          "the TypeSafe decisions API (needs TYPESAFE_API_KEY)."))
parser.add_argument("--request-timeout", "--request_timeout", type=float, default=120.0,
                    help="Per-request timeout in seconds (default: 120).")
parser.add_argument("--step-m", "--step_m", type=float, default=None,
                    help="ACHIEVED translation per decision, in metres (default: 0.03). "
                         "The commanded total is derived from the harness's measured "
                         "rel-IK fraction; do not pre-compensate it here.")
parser.add_argument("--control-steps", "--control_steps", type=int, default=None,
                    help="Control steps one decision is split over (default: 12). Raise it\n                         with --step-m to keep the per-control-step command, and so\n                         the IK saturation, unchanged.")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true",
                    help="Verbose output (default: False).")

from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from policies.jev.client import CONTROL_STEPS, STEP_M, JevClient  # noqa: E402
from policies.vlm_agent.vlm_backend import VLMBackend  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_rel_ik import auto_register_droid_rel_ik_envs  # noqa: E402
from robolab.registrations.droid.camera_presets import WRIST_LEFT  # noqa: E402
from robolab.robots.droid import DroidRelIKActionCfg  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.VERBOSE = args_cli.enable_verbose

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
_jev_log = logging.getLogger("policies.jev")
_jev_log.addHandler(_handler)
_jev_log.setLevel(logging.INFO)
_jev_log.propagate = False

auto_register_droid_rel_ik_envs(
    task_dirs=args_cli.task_dirs,
    task=args_cli.task,
    cameras=WRIST_LEFT,  # recorded only; the policy decides on numbers, not pixels
)

IK_SCALE = float(DroidRelIKActionCfg().arm_action.scale)


def make_client(args: argparse.Namespace) -> JevClient:
    from policies.jev.chat_backend import ChatBackend
    # One step_m, owned here and handed to both the decision layer (which tells
    # the model how far a decision travels) and the client (which converts it to
    # commanded metres). It must not be duplicated inside a backend.
    step_m = args.step_m or STEP_M
    control_steps = args.control_steps or CONTROL_STEPS
    if args.backend == "jev":
        from policies.jev.jev_backend import DEFAULT_MODEL as JEV_DEFAULT_MODEL
        from policies.jev.jev_backend import JevBackend
        backend = JevBackend(model=args.model or JEV_DEFAULT_MODEL,
                             timeout=args.request_timeout, step_m=step_m)
        print(f"\033[96m[Jev] backend=jev API model={backend.model}\033[0m")
    else:
        chat = VLMBackend(base_url=args.vlm_url, model=args.model,
                          timeout=args.request_timeout)
        print(f"\033[96m[Jev] Waiting for the VLM server at {args.vlm_url}...\033[0m")
        chat.wait_for_server(timeout=120.0)
        backend = ChatBackend(chat, step_m=step_m)
        print(f"\033[96m[Jev] backend=chat model={chat.model} task={args.task}\033[0m")
    print(f"\033[96m[Jev] step={step_m * 100:g} cm achieved over {control_steps} "
          f"control steps (ik_scale={IK_SCALE})\033[0m")
    return JevClient(backend=backend, task=args.task, step_m=step_m,
                     control_steps=control_steps, ik_scale=IK_SCALE)


def main() -> None:
    run_evaluation(args_cli, policy=POLICY, client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\033[96m[RoboLab] Terminated with error: {e}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
