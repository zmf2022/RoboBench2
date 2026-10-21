# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reconstruct model-visible history from independently selected channels.

Native reasoning items are opaque and are copied without decoding or rewriting.
Historical user/state/feedback text is never replayed. Historical images retain
only their camera/control-step labels. Tool replies are neutral protocol replies,
not motion feedback. Current state and feedback arrive through the normal input.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

ACTION_ACK = "Historical policy call recorded. Its outcome is not included in this historical tool reply."


def native_items(path: Path, turn_id: str) -> list[dict]:
    """Read the completed current turn, excluding injected earlier turns."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    start = next(
        i for i, row in enumerate(rows)
        if row.get("type") == "event_msg"
        and row["payload"].get("type") == "task_started"
        and row["payload"].get("turn_id") == turn_id
    )
    current = rows[start:]
    if any(row.get("type") == "compacted" for row in current):
        raise RuntimeError("History-channel turn compacted: full selected history was not preserved")
    return [row["payload"] for row in current if row.get("type") == "response_item"]


def assistant_items(items: list[dict], output_tool_names: frozenset[str]) -> list[dict]:
    """Allow assistant messages, native reasoning and matched action-tool pairs."""
    retained = []
    calls = set()
    outputs = set()
    for original in items:
        kind = original["type"]
        item = copy.deepcopy(original)
        if kind == "reasoning" or kind == "message" and item.get("role") == "assistant":
            retained.append(item)
        elif kind == "function_call":
            if item["name"].split(".")[-1] not in output_tool_names:
                raise ValueError("Unexpected non-action call in history-channel run")
            calls.add(item["call_id"])
            retained.append(item)
        elif kind == "function_call_output":
            outputs.add(item["call_id"])
            item["output"] = ACTION_ACK
            retained.append(item)
        elif kind != "message":
            raise ValueError(f"Unsupported native history item: {kind}")
    if calls != outputs:
        raise ValueError("Historical action calls and replies must remain paired")
    return retained


def visual_item(blocks: list[dict]) -> dict:
    """Input block zero contains state/feedback; keep only labelled images after it."""
    content = []
    for block in blocks[1:]:
        if block["type"] == "text":
            if not block["text"].startswith("Observation ") or "; Camera: " not in block["text"]:
                raise ValueError("Historical images must have only camera/control-step labels")
            content.append({"type": "input_text", "text": block["text"]})
        elif block["type"] == "image":
            content.append({"type": "input_image", "image_url": f"data:{block['mimeType']};base64,{block['data']}", "detail": "high"})
        else:
            raise ValueError("Unexpected visual-history block")
    if sum(c["type"] == "input_image" for c in content) != 3:
        raise ValueError("Full visual history requires one three-camera observation per decision")
    return {"type": "message", "role": "user", "content": content}


class ChannelHistory:
    def __init__(self, *, text: str, images: str):
        if text not in {"none", "full", "sampled"} or images not in {"none", "full"}:
            raise ValueError("Invalid history channel selection")
        self.keep_assistant = text != "none"
        self.visual_mode = images
        self.text_mode = text
        self.items: list[dict] = []
        self.decisions = 0
        self.turns: list[tuple[int, list[dict]]] = []
        self.sampled_steps: list[int] = []
        self.selected_turn_steps: list[int] = []

    def select(self, control_steps: list[int]):
        """Pair each past sampled image with the last decision at/before its step.

        Images exist at every control step, whereas outputs exist only at model
        decisions. Keep a selected native turn whole and replay duplicate matches
        only once. The current observation has no generated action yet.
        """
        if self.text_mode != "sampled":
            return
        self.sampled_steps = control_steps[:-1]
        selected = {}
        for step in self.sampled_steps:
            candidates = [(t, items) for t, items in self.turns if t <= step]
            if candidates:
                turn_step, items = candidates[-1]
                selected[turn_step] = items
        self.selected_turn_steps = list(selected)
        self.items = [item for items in selected.values() for item in items]

    def append(self, blocks: list[dict], generated: list[dict], output_tools: frozenset[str]):
        if self.visual_mode == "full":
            self.items.append(visual_item(blocks))
        if self.keep_assistant:
            retained = assistant_items(generated, output_tools)
            if self.text_mode == "sampled":
                policy = json.loads(blocks[0]["text"].split("\n")[-1])["policy"]
                self.turns.append((policy["input"]["control_steps"][-1], retained))
            else:
                self.items.extend(retained)
        self.decisions += 1

    def audit(self) -> dict:
        body = json.dumps(self.items, sort_keys=True, separators=(",", ":")).encode()
        result = {
            "history_text": self.text_mode,
            "history_images": self.visual_mode,
            "previous_decisions": self.decisions,
            "retained_item_types": dict(Counter(item["type"] for item in self.items)),
            "historical_images": sum(c.get("type") == "input_image" for item in self.items for c in item.get("content", [])),
            "encrypted_reasoning_items": sum(bool(item.get("encrypted_content")) for item in self.items if item["type"] == "reasoning"),
            "injected_sha256": hashlib.sha256(body).hexdigest(),
            "historical_feedback": False,
            "memory_tools": False,
        }
        if self.text_mode == "sampled":
            result.update(sampled_image_steps=self.sampled_steps, selected_turn_steps=self.selected_turn_steps)
        return result
