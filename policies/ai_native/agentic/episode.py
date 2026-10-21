# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Codex environment binding and episode diagnostics."""

from policies.ai_native.agentic.client import CodexAgentEEFClient
from policies.ai_native.common.episode import attach_episode_metadata
from robolab.constants import get_output_dir


def run_codex_episode(env, env_cfg, episode, client: CodexAgentEEFClient, **options):
    from robolab.eval.episode import run_episode

    client.configure_environment(env_cfg)
    client.bind_environment(env)

    results, messages, timing = run_episode(
        env, env_cfg, episode, client,
        **options,
    )
    attach_episode_metadata(results, client.episode_metadata)
    client.export_diagnostics(get_output_dir(), run_idx=episode)
    return results, messages, timing
