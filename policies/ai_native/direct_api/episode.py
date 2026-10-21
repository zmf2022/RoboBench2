# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Direct-API episode execution and policy metadata."""

from policies.ai_native.direct_api.client import AINativeClient
from policies.ai_native.common.episode import attach_episode_metadata


def run_direct_api_episode(env, env_cfg, episode, client: AINativeClient, **options):
    # IsaacLab imports must wait until the application has started.
    from robolab.eval.episode import run_episode

    client.configure_environment(env_cfg)

    results, messages, timing = run_episode(env, env_cfg, episode, client, **options)
    attach_episode_metadata(results, client.episode_metadata)
    return results, messages, timing
