# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize and aggregate model-reported token usage across AI-native transports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

TOKEN_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "total_tokens",
)


def _mapping(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json", exclude_none=True)
        return dumped if isinstance(dumped, Mapping) else None
    attributes = getattr(value, "__dict__", None)
    return attributes if isinstance(attributes, Mapping) else None


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number < 0:
        return None
    try:
        if float(value) != number:
            return None
    except (TypeError, ValueError, OverflowError):
        return None
    return number


def _first_int(payload: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        if name in payload:
            value = _nonnegative_int(payload[name])
            if value is not None:
                return value
    return None


def _nested_int(payload: Mapping[str, Any], containers: tuple[str, ...], names: tuple[str, ...]) -> int | None:
    for container in containers:
        nested = _mapping(payload.get(container))
        if nested is not None:
            value = _first_int(nested, *names)
            if value is not None:
                return value
    return None


def normalize_token_usage(value: Any) -> dict[str, int] | None:
    """Return one canonical usage record from OpenAI snake_case or Codex camelCase fields."""

    payload = _mapping(value)
    if payload is None:
        return None
    input_tokens = _first_int(payload, "input_tokens", "prompt_tokens", "inputTokens", "promptTokens")
    output_tokens = _first_int(
        payload,
        "output_tokens",
        "completion_tokens",
        "outputTokens",
        "completionTokens",
    )
    total_tokens = _first_int(payload, "total_tokens", "totalTokens")
    cached_input_tokens = _first_int(payload, "cached_input_tokens", "cachedInputTokens")
    if cached_input_tokens is None:
        cached_input_tokens = _nested_int(
            payload,
            ("input_tokens_details", "prompt_tokens_details", "inputTokensDetails", "promptTokensDetails"),
            ("cached_tokens", "cachedTokens"),
        )
    reasoning_tokens = _first_int(payload, "reasoning_tokens", "reasoningOutputTokens")
    if reasoning_tokens is None:
        reasoning_tokens = _nested_int(
            payload,
            ("output_tokens_details", "completion_tokens_details", "outputTokensDetails", "completionTokensDetails"),
            ("reasoning_tokens", "reasoningTokens"),
        )
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = input_tokens + output_tokens
    if input_tokens is None and output_tokens is None and total_tokens is None:
        return None
    return {
        "input_tokens": input_tokens or 0,
        "cached_input_tokens": cached_input_tokens or 0,
        "output_tokens": output_tokens or 0,
        "reasoning_tokens": reasoning_tokens or 0,
        "total_tokens": total_tokens or 0,
    }


def _select_usage(value: Any, *, codex_thread_usage: bool) -> tuple[dict[str, int] | None, bool]:
    """Select API request usage or the cumulative total of all Codex requests."""

    payload = _mapping(value)
    if payload is None:
        return None, False
    if codex_thread_usage:
        # A robot decision can contain several model requests. `last` covers
        # only the final request, so it cannot substitute for a missing total.
        return normalize_token_usage(payload.get("total")), True
    return normalize_token_usage(payload), False


@dataclass
class TokenUsageTracker:
    """Accumulate explicit model requests and their provider-reported usage."""

    inference_calls: int = 0
    model_requests: int = 0
    requests_with_usage: int = 0
    requests_missing_usage: int = 0
    totals: dict[str, int] = field(default_factory=lambda: {name: 0 for name in TOKEN_FIELDS})
    _last_cumulative: dict[str, int] | None = field(default=None, repr=False)

    def record_inference_call(self) -> None:
        self.inference_calls += 1

    def record_model_request(self, value: Any, *, codex_thread_usage: bool = False) -> dict[str, int] | None:
        """Record one explicit request and return the tokens charged to this request."""

        self.model_requests += 1
        usage, cumulative = _select_usage(value, codex_thread_usage=codex_thread_usage)
        if usage is None:
            self.requests_missing_usage += 1
            return None
        self.requests_with_usage += 1
        charged = usage
        if cumulative:
            previous = self._last_cumulative or {name: 0 for name in TOKEN_FIELDS}
            charged = {name: max(0, usage[name] - previous[name]) for name in TOKEN_FIELDS}
            self._last_cumulative = dict(usage)
        for name in TOKEN_FIELDS:
            self.totals[name] += charged[name]
        return charged

    def summary(self) -> dict[str, int]:
        input_tokens = self.totals["input_tokens"]
        cached_tokens = self.totals["cached_input_tokens"]
        return {
            "inference_calls": self.inference_calls,
            "model_requests": self.model_requests,
            "requests_with_usage": self.requests_with_usage,
            "requests_missing_usage": self.requests_missing_usage,
            **self.totals,
            "uncached_input_tokens": max(0, input_tokens - cached_tokens),
        }


__all__ = ["TOKEN_FIELDS", "TokenUsageTracker", "normalize_token_usage"]
