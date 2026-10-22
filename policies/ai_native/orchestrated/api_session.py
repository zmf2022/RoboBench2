"""Stateful proposal reviews over an OpenAI-compatible Chat Completions API."""

import copy
import json
import math
import os
import time

from openai import OpenAI
from openai.types.chat import ChatCompletion

from policies.ai_native.agentic.app_server import EpisodeMemory
from policies.ai_native.common.token_usage import normalize_token_usage
from policies.ai_native.orchestrated.protocol import review_tool
from policies.ai_native.orchestrated.session import observation_blocks, reviewer_prompt


class APIProposalReviewSession:
    """One full conversation per episode; only the host can execute actions.

    Function calls are paired with host receipts before the next user input.
    Malformed model output goes through the same host repair budget as Codex.
    Transport errors abort the episode; a potentially ambiguous action is never
    replayed, and no simulator control occurs inside this session.
    """

    def __init__(self, *, spec, model_id, memory_root, episode_idx, allow_model_requests,
                 base_url=None, api_key_env="OPENAI_API_KEY", reasoning_effort="medium",
                 max_completion_tokens=8192, timeout=120., image_size=896, jpeg_quality=85, image_detail="high",
                 memory_root_source="explicit", sdk=None):
        if allow_model_requests is not True:
            raise ValueError("Real reviews require allow_model_requests=True")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("API model ID must be nonempty")
        if type(max_completion_tokens) is not int or max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be positive")
        if not math.isfinite(timeout) or not 0 < timeout <= 900:
            raise ValueError("timeout must be in (0, 900]")
        if type(image_size) is not int or not 128 <= image_size <= 2048:
            raise ValueError("image_size must be between 128 and 2048")
        if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        if image_detail not in ("low", "high", "auto"):
            raise ValueError("image_detail must be low, high or auto")
        self._owns_sdk = sdk is None
        if sdk is None:
            key = os.environ.get(api_key_env)
            if not key or not key.strip():
                raise ValueError(f"API key is missing; set {api_key_env}")
            sdk = OpenAI(api_key=key, base_url=base_url or os.environ.get("OPENAI_BASE_URL")
                         or "https://api.openai.com/v1", timeout=timeout, max_retries=0)
        self.sdk = sdk
        self.model_id, self.reasoning_effort = model_id, reasoning_effort
        self.max_completion_tokens = max_completion_tokens
        self.image_size, self.jpeg_quality = image_size, jpeg_quality
        self.image_detail = image_detail
        self.tool = review_tool(spec)
        self.memory = EpisodeMemory(memory_root, policy="orchestrated-api-droid", episode_idx=episode_idx,
                                    env_id=0, root_source=memory_root_source)
        self.workspace = self.memory.path
        self.messages = [dict(role="system", content=reviewer_prompt())]
        self.last_metadata = {}
        self._pending_tool_ids = []
        self._awaiting_feedback = False
        self._closed = False
        self._failed = False
        self._calls = 0

    def _api_content(self, blocks):
        content = []
        for block in blocks:
            if block["type"] == "text":
                content.append(dict(type="text", text=block["text"]))
            else:
                content.append(dict(type="image_url", image_url=dict(
                    url=f"data:{block['mimeType']};base64,{block['data']}", detail=self.image_detail)))
        return content

    def _capture_message(self, message):
        """Keep a valid transcript even when a provider emits invalid function calls."""
        calls = message.get("tool_calls") or []
        structurally_valid = isinstance(calls, list) and all(
            isinstance(call, dict) and call.get("type") == "function"
            and isinstance(call.get("id"), str) and call["id"]
            and isinstance(call.get("function"), dict)
            and isinstance(call["function"].get("name"), str)
            and isinstance(call["function"].get("arguments"), str)
            for call in calls
        )
        if structurally_valid and len({call["id"] for call in calls}) == len(calls):
            saved = dict(role="assistant", content=message.get("content"))
            if calls:
                saved["tool_calls"] = [dict(id=call["id"], type="function", function=dict(
                    name=call["function"]["name"], arguments=call["function"]["arguments"])) for call in calls]
                self._pending_tool_ids = [call["id"] for call in calls]
            elif saved["content"] is None:
                saved["content"] = message.get("refusal") or "No review tool call was returned."
        else:
            # Invalid IDs/types cannot be replayed as native function calls.
            saved = dict(role="assistant", content="Malformed review response: " + json.dumps(message))
            calls = []
        self.messages.append(saved)
        self._awaiting_feedback = True
        return calls

    def query(self, request):
        if self._closed or self._failed:
            raise RuntimeError("API session is closed or failed; start a fresh episode")
        if self._awaiting_feedback:
            raise RuntimeError("Host must validate the previous review before requesting another")
        self.last_metadata = dict(backend="api", model=self.model_id, usage_is_cumulative=False)
        blocks = observation_blocks(request, image_size=self.image_size, jpeg_quality=self.jpeg_quality)
        self.memory.record_prompt(blocks)
        self.messages.append(dict(role="user", content=self._api_content(blocks)))
        payload = dict(model=self.model_id, messages=copy.deepcopy(self.messages), tools=[self.tool],
                       tool_choice="required", parallel_tool_calls=False,
                       max_completion_tokens=self.max_completion_tokens)
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        started = time.monotonic()
        try:
            # Match Direct-API's explicit-body transport: avoid SDK Omit defaults
            # leaking into request JSON under Isaac Sim's typing environment.
            completion = self.sdk.post("/chat/completions", body=payload, cast_to=ChatCompletion)
            raw = completion.model_dump(mode="json", exclude_none=True)
            usage = normalize_token_usage(raw.get("usage"))
            self.last_metadata.update(token_usage=usage, returned_model=raw.get("model"),
                                      response_id=raw.get("id"))
            self.memory.record_event("api_response", raw)
            choices = raw.get("choices") or []
            if len(choices) != 1:
                self.messages.append(dict(role="assistant", content="Expected exactly one completion choice."))
                self._awaiting_feedback = True
                return "", {}
            choice = choices[0]
            self.last_metadata["finish_reason"] = choice.get("finish_reason")
            calls = self._capture_message(choice.get("message") or {})
            if choice.get("finish_reason") != "tool_calls" or len(calls) != 1:
                self.last_metadata["output_error"] = "Expected one complete review tool call"
                return "", {}
            call = calls[0]
            try:
                arguments = json.loads(call["function"]["arguments"])
                if not isinstance(arguments, dict):
                    raise ValueError("Review arguments must be an object")
                json.dumps(arguments, allow_nan=False)
            except (ValueError, TypeError) as exc:
                self.last_metadata["output_error"] = str(exc)
                return "", {}
            return call["function"]["name"], arguments
        except BaseException as exc:
            self._failed = True
            self.last_metadata["error_type"] = type(exc).__name__
            raise
        finally:
            self._calls += 1
            self.last_metadata["call_index"] = self._calls - 1
            self.last_metadata["latency_s"] = time.monotonic() - started
            self.memory.record_event("api_request_metadata", self.last_metadata)

    def record_host_result(self, payload):
        self.memory.record_event("host_result", payload)
        if not self._awaiting_feedback:
            raise RuntimeError("No API review is awaiting host feedback")
        content = json.dumps(payload, allow_nan=False)
        if self._pending_tool_ids:
            for identifier in self._pending_tool_ids:
                self.messages.append(dict(role="tool", tool_call_id=identifier, content=content))
        else:
            self.messages.append(dict(role="user", content="Host validation: " + content))
        self._pending_tool_ids = []
        self._awaiting_feedback = False

    def close(self):
        if not self._closed:
            self._closed = True
            if self._owns_sdk:
                self.sdk.close()
