# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy metadata survives evaluation result summarization."""

from types import SimpleNamespace

import pytest


def test_summary_preserves_policy_metadata(request):
    if request.config.getoption("noconftest"):
        pytest.skip("Summary integration requires initialized IsaacSim dependencies")
    from robolab.eval.summarize import build_run_summary

    cfg = SimpleNamespace(_task_name="test", instruction="test", _task_attributes=[])
    summary = build_run_summary(
        env_result={
            "success": False,
            "step": 12,
            "episode_metadata": {"token_usage": {"total_tokens": 123}},
        },
        env_id=0,
        run_idx=0,
        num_envs=1,
        run_name="test",
        task_env="test",
        env_cfg=cfg,
        policy="scripted",
        dt=1 / 15,
        traj_metrics=None,
        events={},
        events_list=[],
        final_info=None,
        enable_subtask_progress=False,
    )
    assert summary["success"] is False
    assert summary["token_usage"] == {"total_tokens": 123}
