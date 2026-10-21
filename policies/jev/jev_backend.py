# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Jev decisions-API backend, ported from RoboJEV's ``JevPolicy``.

Wire format is RoboJEV's: ``{"model", "state", "questions"}`` where each
question is ``{"type": "choice", "criteria": {...}, "instructions": {...}}``
and ``instructions`` is a MAPPING (``component``/``rules``/``question``), not a
string. Replies carry a probability distribution per head which must be
normalised and whose argmax must equal the stated choice.

The backend returns **labels**, never metres: :class:`policies.jev.types.
EefDecision` owns the metric conversion so step size cannot drift between
backends. The selected intent is injected into ``state["jev_intent"]`` -- the
motor stage reads it as part of the measured state, exactly as upstream, rather
than as prose appended to the instructions.
"""

from __future__ import annotations

import json
import logging
import math
import os
import pathlib
import time
import urllib.error
import urllib.request

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

# Verified live: this endpoint answers the RoboJEV payload and returns
# {choice, confidence, probabilities} per head. OpenRouter does not serve this
# API -- it 401s on the same key -- so neither the URL nor the "typesafe/..."
# model spelling from a chat-completions catalogue applies here.
DEFAULT_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"
API_KEY_ENV = "TYPESAFE_API_KEY"

_AXES = ("x", "y", "z")
# RoboJEV's tolerance on the probability simplex.
PROBABILITY_TOLERANCE = 0.02
RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}
# Repo-root .env, read only when the variable is not already in the environment.
# Stdlib rather than python-dotenv: one policy should not add a dependency, and
# the file is gitignored so the key never reaches a commit either way.
ENV_FILE = pathlib.Path(__file__).resolve().parents[2] / ".env"


def key_from_env_file(path: pathlib.Path = ENV_FILE) -> str | None:
    """``TYPESAFE_API_KEY=...`` from a local .env, or None. Never logged."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        if name.strip() == API_KEY_ENV:
            return value.strip().strip("'\"") or None
    return None


def validate_choice(answer, options) -> str:
    """RoboJEV's ``validate_answer``: anything malformed executes no action."""
    options = tuple(options)
    try:
        if not isinstance(answer, dict):
            raise ValueError("answer is not an object")
        choice = answer["choice"]
        probabilities = answer["probabilities"]
        confidence = answer["confidence"]
        if choice not in options or set(probabilities) != set(options):
            raise ValueError("choice labels do not match the question")
        values = list(probabilities.values())
        if any(isinstance(v, bool) or not isinstance(v, (int, float))
               or not math.isfinite(v) or not 0 <= v <= 1
               for v in (confidence, *values)):
            raise ValueError("confidence/probabilities are not finite numbers in [0, 1]")
        if abs(sum(values) - 1) > PROBABILITY_TOLERANCE:
            raise ValueError("probabilities are not normalised")
        if probabilities[choice] < max(values) - 1e-6:
            raise ValueError("choice is not the argmax of the distribution")
        return choice
    except (KeyError, TypeError, ValueError) as exc:
        raise PolicyError(f"invalid decision response; no action executed ({exc})") from None


class JevBackend:
    """``typesafe/jev-1.13`` over the decisions API."""

    def __init__(self, model: str = DEFAULT_MODEL, timeout: float = 90.0,
                 api_key: str | None = None, step_m: float = 0.02,
                 url: str = DEFAULT_URL, retries: int = 2, sleep=time.sleep) -> None:
        key = api_key or os.environ.get(API_KEY_ENV) or key_from_env_file()
        if not key:
            raise ValueError(
                f"Set {API_KEY_ENV} in the environment or in {ENV_FILE.name} "
                "to use the Jev API backend")
        self.model = model
        self.timeout = float(timeout)
        self.step_m = float(step_m)
        self.url = url
        self.retries = int(retries)
        self.sleep = sleep
        self._key = key
        self.last_exchange: dict = {}

    # ------------------------------------------------------------------

    def _post(self, payload: dict) -> dict:
        request = urllib.request.Request(
            self.url, json.dumps(payload).encode(),
            headers={"Authorization": "Bearer " + self._key,
                     "Content-Type": "application/json"})
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                if exc.code in RETRYABLE_STATUS and attempt < self.retries:
                    try:
                        delay = float(exc.headers.get("retry-after", 2 ** attempt))
                    except (TypeError, ValueError):
                        delay = 2 ** attempt
                    self.sleep(min(max(delay, 0.1), 10))
                    continue
                raise PolicyError(f"decisions API returned HTTP {exc.code}; "
                                  "no action executed") from None
            except (urllib.error.URLError, TimeoutError, OSError):
                if attempt < self.retries:
                    self.sleep(min(2 ** attempt, 5))
                    continue
                raise PolicyError("decisions API transport error; no action executed") from None
        raise PolicyError("decisions API exhausted retries; no action executed")

    def _ask(self, state: dict, questions: dict) -> dict:
        payload = {"model": self.model, "state": state, "questions": questions}
        result = self._post(payload)
        self.last_exchange = {"request": payload, "response": result}
        if not isinstance(result, dict):
            raise PolicyError("malformed decisions response; no action executed")
        if result.get("model") not in (None, self.model):
            raise PolicyError(f"unexpected model {result.get('model')!r}; no action executed")
        answers = result.get("answers")
        if not isinstance(answers, dict) or set(answers) != set(questions):
            raise PolicyError("incomplete decisions response; no action executed")
        return {name: validate_choice(answers[name], question["criteria"])
                for name, question in questions.items()}

    # ------------------------------------------------------------------

    def decide_intent(self, state: dict) -> str:
        questions = {"intent": {
            "type": "choice",
            "criteria": INTENTS,
            "instructions": {"task": state.get("task", ""),
                             "rules": INTENT_RULES,
                             "question": INTENT_QUESTION},
        }}
        return self._ask(state, questions)["intent"]

    def decide_motor(self, state: dict, intent: str) -> dict:
        rules = motor_rules(self.step_m)
        questions = {axis: {
            "type": "choice",
            "criteria": motor_criteria(axis),
            "instructions": {"component": axis, "rules": rules,
                             "question": AXIS_QUESTION.format(axis=axis.upper())},
        } for axis in _AXES}
        questions["gripper"] = {
            "type": "choice",
            "criteria": GRIPPER_CRITERIA,
            "instructions": {"component": "gripper", "rules": rules,
                             "question": GRIPPER_QUESTION},
        }
        # The motor stage sees the selected intent as part of the STATE.
        answers = self._ask({**state, "jev_intent": intent}, questions)
        for axis in _AXES:
            if answers[axis] not in AXIS_CHOICES:
                raise PolicyError(f"invalid {axis} choice {answers[axis]!r}")
        if answers["gripper"] not in GRIPPER_CHOICES:
            raise PolicyError(f"invalid gripper choice {answers['gripper']!r}")
        return answers

    def close(self) -> None:
        return None
