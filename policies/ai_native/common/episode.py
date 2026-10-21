# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Collect AI-native metadata after the simulator releases client state."""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


def attach_episode_metadata(env_results: list[dict], metadata_fn: Callable[..., dict]) -> None:
    """Keep completed results even if optional metadata collection fails.

    Clients retain completed-episode counters through reset(), until the next
    begin_episode(). The episode wrappers call this after shared-loop cleanup.
    """
    for result in env_results:
        env_id = result["env_id"]
        try:
            metadata = metadata_fn(env_id=env_id)
            if not isinstance(metadata, dict):
                raise TypeError("episode_metadata must return a dict")
            if metadata:
                result["episode_metadata"] = metadata
        except Exception:
            # Observability must not invalidate an otherwise completed robot
            # episode. Keep the result and report the metadata collection failure.
            logger.exception("Failed to collect episode metadata for env %d", env_id)
