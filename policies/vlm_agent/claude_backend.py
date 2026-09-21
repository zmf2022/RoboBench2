# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local Claude Code CLI backend shaped like :class:`VLMBackend`.

Runs ``claude -p`` as a subprocess per decision and adapts the reply back to the
OpenAI-shaped dict (``choices[0].message.content``) our controllers parse. There
is no ``reasoning_content`` channel here — whatever thinking happens stays inside
the CLI session; our parsers take the LAST marker, so trailing prose is harmless.

Frames are handed over as ``image`` content blocks on a ``stream-json`` stdin
message, which is the ONLY way the CLI accepts images: trailing file paths after
the prompt are silently dropped (the ``[prompt]`` positional swallows one string
and ignores the rest), so the model sees no camera at all and answers "no images
came with this step" on every decision.

The session is stripped down to a one-shot vision call — ``--tools ""`` (no tool
schemas), ``--system-prompt`` (replaces the agent preamble rather than appending
to it) and ``--setting-sources ""`` (no user hooks/settings). That matters for
more than tidiness: the full agent context is ~34k tokens of tool definitions and
CLAUDE.md per call, versus ~700 here.

Caveats (read before putting this in a control loop):
- Each call bills the relay account and costs a few seconds of process + API
  latency. At ~170 decisions/episode it is still slow and pricey next to a local
  vLLM. Prefer it as a high-level planner, not a per-step policy.
- Requires a funded relay login (``claude`` must work non-interactively).
"""

from __future__ import annotations

import json
import logging
import subprocess

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS = 300.0
DEFAULT_MAX_RETRIES = 1


class ClaudeCliBackend:
    """Claude Code CLI (``claude -p``) chat client shaped like VLMBackend."""

    def __init__(
        self,
        model: str | None = "claude-opus-5",
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        effort: str = "high",
        cli: str = "claude",
        keep_session: bool = True,
        max_session_turns: int = 24,
    ) -> None:
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")
        if effort not in ("low", "medium", "high", "xhigh", "max"):
            raise ValueError("effort must be low|medium|high|xhigh|max")
        self.model = model or "claude-opus-5"
        self.effort = effort
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        # Accepted for interface parity; the CLI exposes neither knob.
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.cli = cli
        # One CLI session per episode instead of one per decision. Capped because the
        # session accumulates two images per turn: past the cap the context is mostly
        # stale frames, and Claude Code starts auto-compacting it anyway. Rolling over
        # costs one cold decision and restores a small context.
        self.keep_session = bool(keep_session)
        self.max_session_turns = max(1, int(max_session_turns))
        self._session_id: str | None = None
        self._turns = 0

    # ------------------------------------------------------------------
    # Translation: OpenAI shape in, OpenAI shape out.
    # ------------------------------------------------------------------

    @staticmethod
    def _image_block(url: str) -> dict | None:
        """``data:image/png;base64,...`` -> an Anthropic image content block."""
        header, _, data = url.partition("base64,")
        if not data:
            return None
        media_type = header[5:].rstrip(";") if header.startswith("data:") else "image/png"
        return {"type": "image",
                "source": {"type": "base64", "media_type": media_type, "data": data}}

    @classmethod
    def _split_messages(cls, messages: list[dict]) -> tuple[str, list[dict]]:
        """Return (system_text, user content blocks) from OpenAI messages.

        Block order is preserved exactly as the controller built it — the prompt
        identifies views by position, so collapsing images and text into separate
        groups would silently re-label the cameras.
        """
        system_parts: list[str] = []
        blocks: list[dict] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                continue
            if isinstance(content, str):
                blocks.append({"type": "text", "text": content})
                continue
            for part in content or []:
                kind = part.get("type")
                if kind == "image_url":
                    block = cls._image_block((part.get("image_url") or {}).get("url", ""))
                    if block is not None:
                        blocks.append(block)
                elif kind == "image":  # already an Anthropic block
                    blocks.append(part)
                elif kind in ("text", "input_text"):
                    blocks.append({"type": "text", "text": part.get("text", "")})
        return "\n\n".join(system_parts), blocks

    @staticmethod
    def _tools_to_schema(tools: list[dict]) -> tuple[str, dict]:
        """OpenAI tool schemas -> (prompt text, a --json-schema constraint).

        The CLI has no tool-calling hook for caller-supplied tools, but it does
        have structured output. Constraining the reply to
        ``{"name": ..., "arguments": {...}}`` produces exactly the shape
        ``ToolController._tool_call_from_message`` already accepts as its
        no-native-tool-calling fallback, so the controller needs no change. The
        schemas themselves still have to reach the model as text — the
        constraint fixes the SHAPE of the answer, not what the tools mean.
        """
        names: list[str] = []
        described: list[str] = []
        for tool in tools:
            func = tool.get("function") or tool
            name = str(func.get("name") or "")
            if not name:
                continue
            names.append(name)
            described.append(json.dumps({
                "name": name,
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {"type": "object"}),
            }, ensure_ascii=False))
        text = ("Available tools:\n" + "\n".join(described)
                + "\n\nReply with ONE tool call as a JSON object: "
                  '{"name": "<tool>", "arguments": {...}}. No prose.')
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": names},
                "arguments": {"type": "object"},
            },
            "required": ["name", "arguments"],
            "additionalProperties": False,
        }
        return text, schema

    def _run_cli(self, system: str, blocks: list[dict],
                 json_schema: dict | None = None) -> str:
        """Invoke ``claude -p`` once; return the result text."""
        cmd = [
            self.cli, "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--verbose",              # required with --output-format stream-json
            "--tools", "",            # no tool schemas: this is a one-shot vision call
            "--setting-sources", "",  # ignore user/project settings and hooks
            "--disable-slash-commands",
            "--effort", self.effort,
        ]
        if self.model:
            cmd += ["--model", self.model]
        if system:
            cmd += ["--system-prompt", system]
        if json_schema is not None:
            cmd += ["--json-schema", json.dumps(json_schema)]
        # Short-term memory across decisions. Without --resume every decision is a fresh
        # process that has never seen the episode: the model can only be told what it did
        # through the recent-units list, and re-derives the scene from two images ~90
        # times per episode. Resuming keeps the earlier turns, so "I already tried MV_BACK
        # here and the picture did not change" is something it can actually know.
        if self._session_id:
            cmd += ["--resume", self._session_id]

        stdin = json.dumps({"type": "user",
                            "message": {"role": "user", "content": blocks}}) + "\n"
        proc = subprocess.run(cmd, input=stdin, capture_output=True, text=True,
                              timeout=self.timeout)
        if proc.returncode != 0:
            raise RuntimeError(f"claude CLI failed ({proc.returncode}): "
                               f"{(proc.stderr or proc.stdout or '')[:500]}")

        result: str | None = None
        for line in proc.stdout.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get("type") != "result":
                continue
            if event.get("is_error"):
                raise RuntimeError(f"claude CLI error: {str(event.get('result'))[:500]}")
            result = event.get("result")
            if self.keep_session:
                session = event.get("session_id")
                if isinstance(session, str) and session:
                    self._session_id = session
                    self._turns += 1
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError(f"claude CLI gave no result text: {proc.stdout[-300:]!r}")
        return result

    # ------------------------------------------------------------------
    # Requests
    # ------------------------------------------------------------------

    def wait_for_server(self, timeout: float = 30.0) -> None:
        """No server; nothing to wait for (auth is checked per call)."""

    def new_session(self) -> None:
        """Forget the running CLI session; the next call starts a cold one.

        Called at episode boundaries: carrying one episode's frames into the next
        would have the model reasoning about a scene that has been reset.
        """
        self._session_id = None
        self._turns = 0

    def complete(
        self,
        messages: list[dict],
        *,
        tools: list[dict] | None = None,
        choices: list[str] | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        """Run one CLI call. ``tools``/``choices`` are accepted for interface
        parity; the CLI exposes no tool-calling or constrained-decoding hook, so
        only the reply text is parsed."""
        system, blocks = self._split_messages(messages)
        schema: dict | None = None
        if tools:
            tool_text, schema = self._tools_to_schema(tools)
            blocks.append({"type": "text", "text": tool_text})
        if not any(b.get("type") == "image" for b in blocks):
            logger.warning("[ClaudeCLI] no images in this prompt; the model is "
                           "deciding blind")
        if self.keep_session and self._turns >= self.max_session_turns:
            logger.info("[ClaudeCLI] session hit %d turns; starting a fresh one",
                        self._turns)
            self.new_session()
        last: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                result = self._run_cli(system, blocks, json_schema=schema)
                return {"choices": [{"message": {"content": result},
                                     "finish_reason": "stop"}]}
            except Exception as exc:  # noqa: BLE001 - retry transport + API errors
                last = exc
                logger.warning("[ClaudeCLI] attempt %d/%d failed: %s",
                               attempt + 1, self.max_retries, exc)
        raise RuntimeError(
            f"claude CLI failed after {self.max_retries} attempts: {last}")

    def close(self) -> None:
        return None
