"""Evaluate pi05 proposals reviewed through Codex or a direct model API."""

import argparse
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main():
    import cv2  # noqa: F401 -- before Isaac Lab
    from isaaclab.app import AppLauncher
    from policies.ai_native.orchestrated.run_config import (
        add_orchestrated_args, build_client, check_runtime_dependencies, load_api_environment, validate_args,
    )
    from robolab.eval.runner import add_common_eval_args, run_evaluation

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_orchestrated_args(parser)
    add_common_eval_args(parser)
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    load_api_environment(args, REPO_ROOT)
    validate_args(args)
    check_runtime_dependencies(args)
    if args.review_backend == "api":
        from policies.ai_native.common.api_model import resolve_api_model

        args.model = resolve_api_model(args.model, base_url=args.base_url,
                                       api_key_env=args.api_key_env, timeout=args.decision_timeout)
    model_label = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model).strip("_")
    if not model_label:
        raise ValueError("Model ID must contain a path-safe name")
    policy = f"orchestrated_pi05_{args.review_backend}_{model_label}"
    args.enable_cameras = True
    app = AppLauncher(args).app
    try:
        import robolab.constants
        from policies.ai_native.orchestrated.episode import run_orchestrated_episode
        from robolab.registrations.droid.auto_env_registrations_jointpos import auto_register_droid_envs
        from robolab.registrations.droid.camera_presets import WRIST_LEFT

        robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args.enable_subtask
        auto_register_droid_envs(task_dirs=args.task_dirs, task=args.task, cameras=WRIST_LEFT)
        run_evaluation(args, policy=policy, client_factory=build_client,
                       episode_runner=run_orchestrated_episode)
    finally:
        app.close()


if __name__ == "__main__":
    main()
