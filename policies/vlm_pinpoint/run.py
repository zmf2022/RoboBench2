# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the VLM pointing policy across registered tasks."""

import argparse
import logging
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

POLICY = "vlm_pinpoint"

parser = argparse.ArgumentParser(description="Evaluate a VLM pointing policy.")
parser.add_argument("--vlm-url", "--vlm_url", default="http://localhost:8000/v1",
                    help="OpenAI-compatible chat-completions base URL (default: http://localhost:8000/v1).")
parser.add_argument("--model", default=None,
                    help="Model name (auto-detected from the server's /models if omitted).")
parser.add_argument("--request-timeout", "--request_timeout", type=float, default=120.0,
                    help="Per-request timeout in seconds (default: 120).")
parser.add_argument("--point-every", "--point_every", type=int, default=8,
                    help="Control steps per VLM point query (default: 8).")
parser.add_argument("--step-m", "--step_m", type=float, default=None,
                    help="Per-decision travel cap in metres (default: 0.072).")
parser.add_argument("--point-camera", "--point_camera", default="external",
                    choices=["external", "wrist"],
                    help="Which view the VLM points on. 'external' is the fixed "
                         "over-shoulder camera, which keeps the whole workspace in "
                         "frame for the whole episode -- the setup every published "
                         "pointing stack uses. 'wrist' is eye-in-hand: higher "
                         "close-up detail, but the objects leave the FOV as soon as "
                         "the arm moves (default: external).")
parser.add_argument("--no-depth-image", "--no_depth_image", action="store_true",
                    help="Do not send the depth map to the VLM as a second image. "
                         "Depth is still used numerically for the unprojection; "
                         "this is the A/B switch for whether the MODEL sees it.")
parser.add_argument("--viz-dir", "--viz_dir", default=None,
                    help="Save one annotated frame per VLM decision here. The "
                         "recorded mp4 is the harness's own camera tile, and the "
                         "policy's own overlay is only drawn by cv2.imshow, so "
                         "under --headless this is the ONLY way to see where the "
                         "model actually pointed.")
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
from policies.vlm_pinpoint.client import PinpointClient  # noqa: E402
from policies.vlm_pinpoint.pointing_backend import VLLMPointingBackend  # noqa: E402
from policies.vlm_pinpoint.pointing_controller import PointingController  # noqa: E402
from robolab.registrations.droid.auto_env_registrations_rel_ik import auto_register_droid_rel_ik_envs  # noqa: E402
from robolab.robots import droid as droid_mod  # noqa: E402
from robolab.robots.droid import DroidRelIKActionCfg, WristCameraCfg  # noqa: E402
from robolab.variations.camera import OverShoulderLeftCameraCfg, with_depth  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.VERBOSE = args_cli.enable_verbose

_handler = logging.StreamHandler()
_handler.setFormatter(logging.Formatter("%(message)s"))
_pin_log = logging.getLogger("policies.vlm_pinpoint")
_pin_log.addHandler(_handler)
_pin_log.setLevel(logging.INFO)
_pin_log.propagate = False

auto_register_droid_rel_ik_envs(
    task_dirs=args_cli.task_dirs,
    task=args_cli.task,
    # Same views as vlm_agent, but the wrist camera also renders depth for
    # unprojection. with_depth() leaves shared configs untouched.
    # Both views render depth: pointing needs the depth + intrinsics + extrinsics
    # of whichever camera --point-camera selects, and the other one is still shown
    # to the human in the recording.
    cameras=[with_depth(OverShoulderLeftCameraCfg), with_depth(WristCameraCfg)],
)

IK_SCALE = float(DroidRelIKActionCfg().arm_action.scale)


# The rendered wrist sensor is attached via DroidCfg, not the camera preset
# list (WristCameraCfg is excluded from scene cameras there), so depth
# rendering is enabled here. Process-local only: each policy runs in its own
# process, so other policies keep rgb-only rendering. Read at scene-build
# time, hence configuring it at import time is early enough.
droid_mod._WRIST_CAM.data_types = ["rgb", "depth"]


def make_client(args: argparse.Namespace) -> PinpointClient:
    backend = VLLMPointingBackend(base_url=args.vlm_url, model=args.model,
                                  timeout=args.request_timeout)
    print(f"\033[96m[Pinpoint] Waiting for the VLM server at {args.vlm_url}...\033[0m")
    backend.client.wait_for_server(timeout=120.0)
    controller = PointingController(
        step_m=args.step_m if args.step_m is not None else 0.072,
    )
    print(f"\033[96m[Pinpoint] model={backend.client.model} "
          f"point_camera={args.point_camera} "
          f"point_every={args.point_every} "
          f"depth_to_vlm={not args.no_depth_image}\033[0m")
    return PinpointClient(backend=backend, controller=controller,
                          ik_scale=IK_SCALE, point_every=args.point_every,
                          send_depth_image=not args.no_depth_image,
                          point_camera=args.point_camera,
                          viz_dir=args.viz_dir)


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
