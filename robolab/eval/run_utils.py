# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Episode identities and completion checks for evaluation runs.

These helpers inspect saved results without creating simulator environments or
writing artifacts. A run contains one episode per parallel environment.
"""

from robolab.core.utils.adaptive_sampling import count_task_episodes, should_continue_sampling


def episode_ids_for_run(run_idx: int, num_envs: int) -> list[int]:
    """Map a run to its consecutive episode IDs, including any run offset."""
    if run_idx < 0 or num_envs < 1:
        raise ValueError("run_idx must be nonnegative and num_envs must be positive")
    return [run_idx * num_envs + env_id for env_id in range(num_envs)]


def is_run_complete(results: list[dict], task: str, run_idx: int, num_envs: int) -> bool:
    """True when every episode in the run has a saved result, success or failure."""
    saved_ids = {row.get("episode") for row in results if row.get("env_name") == task}
    return all(episode_id in saved_ids for episode_id in episode_ids_for_run(run_idx, num_envs))


def is_evaluation_complete(
    results: list[dict],
    task: str,
    *,
    num_envs: int,
    num_runs: int,
    run_index_offset: int = 0,
    adaptive_max: int | None = None,
    ci_pp_width: float = 0.14,
) -> bool:
    """Check the requested fixed range or the existing adaptive stopping rule.

    Adaptive completion ignores num_runs and counts this task's saved episodes.
    It is checked between whole batches, matching the evaluator's existing
    sampling behavior. Nonzero offsets are supported only for fixed budgets.
    """
    if num_envs < 1 or run_index_offset < 0:
        raise ValueError("num_envs must be positive and run_index_offset must be nonnegative")
    if adaptive_max is not None:
        if run_index_offset != 0:
            raise ValueError("run_index_offset is not supported with adaptive sampling")
        successes, total = count_task_episodes(results, task)
        return not should_continue_sampling(
            k=successes, n=total, target_width=ci_pp_width, n_max=adaptive_max,
        )
    if num_runs < 1:
        raise ValueError("num_runs must be positive")
    return all(
        is_run_complete(results, task, run_idx, num_envs)
        for run_idx in range(run_index_offset, run_index_offset + num_runs)
    )
