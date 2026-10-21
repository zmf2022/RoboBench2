# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import numpy as np
import pytest

from policies.pi0_family.client import Pi0DroidJointposClient
from robolab.core.environments.config import print_env_cfg
from robolab.core.task.instruction import (
    TrackedInstructionFields,
    instruction_text,
    normalize_instruction_fields,
    primary_instruction,
    select_instruction_fields,
)
from robolab.eval.base_client import InferenceClient
from robolab.eval.summarize import build_run_summary

FIELDS = {
    "default": "Put the apple in the bowl",
    "vague": "Move the fruit",
    "video": "assets/demos/apple.mp4",
}


class _ContextClient(InferenceClient):
    open_loop_horizon = 1

    def __init__(self, requested):
        super().__init__()
        self.requested = requested
        self.available = None
        self.context_calls = []
        self.requests = []

    def requested_context_fields(self, available_fields, *, episode_idx):
        self.available = available_fields
        return self.requested

    def add_context(self, context, *, episode_idx):
        self.context_calls.append((episode_idx, dict(context)))

    def _extract_observation(self, raw_obs, *, env_id=0):
        return raw_obs

    def _pack_request(self, extracted_obs, instruction):
        # A context-aware policy deliberately omits instruction fields from
        # its per-step inference request.
        return {"observation": extracted_obs}

    def _query_server(self, request):
        self.requests.append(request)
        return {"actions": [[0.0]]}

    def _unpack_response(self, response):
        return np.asarray(response["actions"])


def test_plain_string_normalizes_to_default_field():
    assert normalize_instruction_fields("do the task") == {"default": "do the task"}


def test_no_selection_exposes_all_fields_in_order():
    assert select_instruction_fields(FIELDS) == FIELDS


def test_multiple_selected_fields_are_filtered_in_cli_order():
    assert select_instruction_fields(FIELDS, ["vague", "video"]) == {
        "vague": "Move the fruit",
        "video": "assets/demos/apple.mp4",
    }


def test_unknown_and_duplicate_selections_fail():
    with pytest.raises(ValueError, match="not found"):
        select_instruction_fields(FIELDS, ["missing"])
    with pytest.raises(ValueError, match="unique"):
        select_instruction_fields(FIELDS, ["vague", "vague"])


def test_plain_string_ignores_legacy_instruction_type():
    assert select_instruction_fields("do the task", ["vague"]) == {"default": "do the task"}


def test_tracking_records_retrieved_values_but_not_key_inspection():
    tracked = TrackedInstructionFields(FIELDS)
    assert "video" in tracked
    assert list(tracked.keys()) == ["default", "vague", "video"]
    assert tracked.instruction_fields_used == []

    assert tracked["vague"] == "Move the fruit"
    assert tracked.get("missing") is None
    assert tracked.instruction_fields_used == ["vague"]


def test_primary_text_access_is_tracked_and_returns_a_string():
    tracked = TrackedInstructionFields({"vague": "Move the fruit", "video": "demo.mp4"})
    assert instruction_text(tracked) == "Move the fruit"
    assert tracked.instruction_fields_used == ["vague"]


def test_requested_context_is_delivered_once_before_inference():
    tracked = TrackedInstructionFields(FIELDS)
    client = _ContextClient(["video", "vague"])

    client.prepare_episode(3, tracked)
    client.prepare_episode(3, tracked)
    client.infer({"step": 0}, tracked)
    client.infer({"step": 1}, tracked)

    assert client.available == ("default", "vague", "video")
    assert client.context_calls == [
        (
            3,
            {
                "video": "assets/demos/apple.mp4",
                "vague": "Move the fruit",
            },
        )
    ]
    assert tracked.instruction_fields_used == ["vague", "video"]
    assert client.requests == [
        {"observation": {"step": 0}},
        {"observation": {"step": 1}},
    ]


def test_context_can_be_sent_again_only_after_episode_reset():
    tracked = TrackedInstructionFields(FIELDS)
    client = _ContextClient("default")

    client.prepare_episode(0, tracked)
    client.reset()
    client.prepare_episode(1, tracked)

    assert client.context_calls == [
        (0, {"default": "Put the apple in the bowl"}),
        (1, {"default": "Put the apple in the bowl"}),
    ]


def test_unavailable_or_duplicate_context_requests_fail_before_delivery():
    for requested, match in [(["missing"], "not available"), (["video", "video"], "unique")]:
        client = _ContextClient(requested)
        with pytest.raises(ValueError, match=match):
            client.prepare_episode(0, TrackedInstructionFields(FIELDS))
        assert client.context_calls == []


def test_pi05_request_receives_plain_string_from_mapping():
    tracked = TrackedInstructionFields({"vague": "Move the fruit", "video": "demo.mp4"})
    client = object.__new__(Pi0DroidJointposClient)
    InferenceClient.__init__(client)
    client.prepare_episode(0, tracked)
    assert tracked.instruction_fields_used == []

    request = client._pack_request(
        {
            "right_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "wrist_image": np.zeros((8, 8, 3), dtype=np.uint8),
            "joint_position": np.zeros(7),
            "gripper_position": np.zeros(1),
        },
        tracked,
    )

    assert request["prompt"] == "Move the fruit"
    assert isinstance(request["prompt"], str)
    assert tracked.instruction_fields_used == ["vague"]


def test_serializing_entire_mapping_marks_all_fields_used():
    tracked = TrackedInstructionFields(FIELDS)
    assert json.loads(json.dumps(tracked)) == FIELDS
    assert tracked.instruction_fields_used == ["default", "vague", "video"]


def test_tracked_fields_are_read_only():
    tracked = TrackedInstructionFields(FIELDS)
    with pytest.raises(TypeError, match="read-only"):
        tracked["default"] = "changed"


def test_primary_instruction_prefers_default_else_first_field():
    assert primary_instruction(FIELDS) == ("default", "Put the apple in the bowl")
    assert primary_instruction({"vague": "Move", "video": "demo.mp4"}) == ("vague", "Move")


def test_print_env_cfg_shows_primary_instruction_and_field_names(capsys):
    cfg = SimpleNamespace(
        instruction=FIELDS,
        scene=SimpleNamespace(),
        observations=SimpleNamespace(),
        actions=SimpleNamespace(),
        terminations=SimpleNamespace(),
        contact_gripper="gripper",
        contact_object_list=["apple", "bowl"],
        episode_length_s=20,
        decimation=8,
        sim=SimpleNamespace(dt=1 / 120, render_interval=8),
        seed=0,
        num_envs=1,
        env_spacing=5.0,
    )

    print_env_cfg(cfg, prefix="  ")
    output = capsys.readouterr().out

    assert "  instruction: Put the apple in the bowl\n" in output
    assert "  instruction fields: default, vague, video\n" in output
    assert repr(FIELDS) not in output


def test_episode_summary_keeps_legacy_text_and_logs_field_usage():
    cfg = SimpleNamespace(
        instruction={"vague": "Move the fruit", "video": "demo.mp4"},
        _task_name="ExampleTask",
        _task_attributes=[],
    )
    summary = build_run_summary(
        env_result={
            "success": True,
            "step": 3,
            "instruction_fields": ["vague", "video"],
            "instruction_fields_used": ["vague", "video"],
        },
        env_id=0,
        run_idx=0,
        num_envs=1,
        run_name="ExampleTask_vague+video_0",
        task_env="ExampleTask",
        env_cfg=cfg,
        policy="pi05",
        dt=0.1,
        traj_metrics=None,
        events={},
        events_list=[],
        final_info=None,
        enable_subtask_progress=False,
        instruction_type=["vague", "video"],
    )

    assert summary["instruction"] == "Move the fruit"
    assert summary["instruction_fields"] == ["vague", "video"]
    assert summary["instruction_fields_used"] == ["vague", "video"]
    assert "instruction_type" not in summary
