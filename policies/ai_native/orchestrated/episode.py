"""Bind orchestration adapters while retaining the shared episode evaluator."""

import json
import math
import time

from policies.ai_native.orchestrated.client import OrchestratedEpisodeStopped
from policies.ai_native.common.episode import attach_episode_metadata
from policies.ai_native.orchestrated.protocol import review_tool
from policies.ai_native.orchestrated.session import reviewer_prompt


def configure_environment(client, env_cfg):
    dt = float(env_cfg.sim.dt * env_cfg.decimation)
    if not math.isclose(dt, 1 / client.control_hz, abs_tol=1e-9):
        raise ValueError("Reviewer control rate does not match simulator timing")
    if client.max_reviews is None:
        client.max_reviews = math.ceil(env_cfg.episode_length_s / dt)
    adapter = client.execution_adapter
    if (adapter.max_translation_step_m, adapter.max_rotation_step_rad,
            adapter.max_joint_delta_rad, adapter.damping) != (0.02, 0.1, 0.05, 0.05):
        raise ValueError("Reviewer instructions do not match DROID IK limits")
    config = dict(policy="orchestrated", runtime=client.runtime_settings, max_reviews=client.max_reviews,
                  output_tool=review_tool(client.spec), system_prompt=reviewer_prompt())
    client.output_dir.mkdir(parents=True, exist_ok=True)
    path = client.output_dir / "orchestrated_config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Cannot reuse an output directory with a different orchestration configuration")
    path.write_text(json.dumps(config, indent=2) + "\n")


def run_orchestrated_episode(env, env_cfg, episode, client, **options):
    from robolab.eval.episode import run_episode

    if options.get("enable_gt_state"):
        raise ValueError("Orchestrated excludes evaluator ground truth")
    configure_environment(client, env_cfg)
    client.bind_environment(env)
    started = time.perf_counter()
    try:
        results, messages, timing = run_episode(env, env_cfg, episode, client, **options)
    except OrchestratedEpisodeStopped:
        # Shared-loop cleanup has already closed videos and reset the client.
        # Keep native success unset and persist the partial trajectory, then let
        # the regular evaluator advance to the next run/task.
        results = env.get_env_results()
        steps = sum(client.episode_metadata(env_id=0)["issued_control_steps"].values())
        for result in results:
            if result["step"] is None:
                result["step"] = steps
        if env.recorder_manager is not None and steps:
            env.recorder_manager.export_episodes(env_ids=[0])
        messages = []
        timing = dict(wall_total_s=round(time.perf_counter() - started, 3))
    finally:
        # Also retain diagnostics for budget exhaustion and transport/physics failures.
        client.export_diagnostics()
    attach_episode_metadata(results, client.episode_metadata)
    return results, messages, timing
