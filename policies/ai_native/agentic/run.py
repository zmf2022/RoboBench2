# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate GPT-6 Astra through Codex App Server using Agent-EEF."""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

from policies.ai_native.agentic.run_config import add_agent_eef_args, build_runtime_options
from robolab.eval.runner import add_common_eval_args

parser = argparse.ArgumentParser(description="Evaluate the robot-agnostic Codex App Server Agent-EEF policy.")
add_agent_eef_args(parser)
add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
build_runtime_options(args_cli)
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from policies.ai_native.agentic.run_config import build_client, camera_names, resolve_droid_camera_preset
from policies.ai_native.agentic.specs import make_droid_spec
from robolab.eval.runner import run_evaluation  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
names = camera_names(args_cli)

from robolab.registrations.droid import camera_presets
from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs

selected = getattr(camera_presets, resolve_droid_camera_preset(names))
auto_register_droid_envs(task_dirs=args_cli.task_dirs, task=args_cli.task, cameras=selected)
embodiment_spec = make_droid_spec()


def make_client(args: argparse.Namespace):
    return build_client(args, spec=embodiment_spec)


def main() -> None:
    policy = f"codex_h{args_cli.history_length}_{args_cli.agent_style}_{args_cli.embodiment}"
    if args_cli.prompt_family != "baseline":
        policy += f"_{args_cli.prompt_family}"
    if args_cli.action_space == "joint":
        policy += "_joint"
    from policies.ai_native.agentic.episode import run_codex_episode as episode_runner
    run_evaluation(args_cli, policy=policy, client_factory=make_client, episode_runner=episode_runner)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\033[96m[RoboLab] Terminated with error: {exc}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
