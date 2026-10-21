# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chat-model backend: the same questions, rendered for a chat server.

A chat endpoint has no structured question format, so the criteria mappings
from :mod:`policies.jev.prompts` are rendered as text -- but they are the SAME
mappings the decisions backend sends, so swapping backends does not silently
change what the model was asked. Labels are RoboJEV's
(``negative``/``zero``/``positive``), not a chat-friendly paraphrase.

One constrained call per head, mirroring the four independent heads upstream.
Unparsable replies raise :class:`PolicyError`; nothing is executed on a guess.
"""

from __future__ import annotations

import json
import logging
import re

from policies.jev.prompts import (
    AXIS_QUESTION,
    GRIPPER_CRITERIA,
    GRIPPER_QUESTION,
    INTENT_QUESTION,
    INTENT_RULES,
    INTENTS,
    motor_criteria,
    motor_rules,
)
from policies.jev.types import AXIS_CHOICES, GRIPPER_CHOICES, PolicyError

logger = logging.getLogger(__name__)

_AXES = ("x", "y", "z")


def _render(criteria: dict) -> str:
    return "\n".join(f"- {label}: {text}" for label, text in criteria.items())


class ChatBackend:
    """Intent + motor decisions over an OpenAI-compatible chat server."""

    def __init__(self, client, step_m: float = 0.02) -> None:
        self.client = client
        self.step_m = float(step_m)
        self.last_exchange: dict = {}

    def _ask(self, system: str, user: str, choices) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [{"type": "text", "text": user}]},
        ]
        response = self.client.complete(messages, choices=list(choices))
        self.last_exchange = {"request": user, "response": response}
        message = ((response.get("choices") or [{}])[0].get("message") or {})
        text = message.get("content") or message.get("reasoning_content") or ""
        if not isinstance(text, str) or not text.strip():
            raise PolicyError("empty model reply; no action executed")
        return text

    def _choose(self, system: str, user: str, choices, head: str) -> str:
        text = self._ask(system, user, choices)
        label = text.strip().strip(".").lower()
        if label in choices:
            return label
        # Constrained decoding returns the bare label; an unconstrained server
        # may still prefix it ("INTENT: grasp").
        match = re.search(r"[A-Za-z_]+\s*:\s*([A-Za-z_]+)", text)
        if match and match.group(1).strip().lower() in choices:
            return match.group(1).strip().lower()
        for candidate in choices:
            if re.fullmatch(rf"\W*{candidate}\W*", label):
                return candidate
        raise PolicyError(f"unparsable {head} reply {text[:200]!r}; no action executed")

    # ------------------------------------------------------------------

    def decide_intent(self, state: dict) -> str:
        user = (f"Measured state (robot-base metres):\n{json.dumps(state)}\n\n"
                f"{INTENT_RULES}\n\nOPTIONS:\n{_render(INTENTS)}\n\n{INTENT_QUESTION}")
        return self._choose(
            "You select one robot intent from measured state. "
            "Reply with exactly one option label and nothing else.",
            user, tuple(INTENTS), "intent")

    def decide_motor(self, state: dict, intent: str) -> dict:
        rules = motor_rules(self.step_m)
        # The intent is part of the state the motor stage reads, as upstream.
        state_json = json.dumps({**state, "jev_intent": intent})
        answers = {}
        for axis in _AXES:
            user = (f"Measured state (robot-base metres):\n{state_json}\n\n{rules}\n\n"
                    f"COMPONENT: {axis}\nOPTIONS:\n{_render(motor_criteria(axis))}\n\n"
                    f"{AXIS_QUESTION.format(axis=axis.upper())}")
            answers[axis] = self._choose(
                "You select one motion component for a robot. "
                "Reply with exactly one option label and nothing else.",
                user, AXIS_CHOICES, axis)
        user = (f"Measured state (robot-base metres):\n{state_json}\n\n{rules}\n\n"
                f"COMPONENT: gripper\nOPTIONS:\n{_render(GRIPPER_CRITERIA)}\n\n"
                f"{GRIPPER_QUESTION}")
        answers["gripper"] = self._choose(
            "You select one gripper command for a robot. "
            "Reply with exactly one option label and nothing else.",
            user, GRIPPER_CHOICES, "gripper")
        return answers

    def close(self) -> None:
        closer = getattr(self.client, "close", None)
        if callable(closer):
            closer()
