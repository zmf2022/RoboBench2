# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent Codex App Server sessions for AI-native RoboLab agents.

The policy loop remains owned by RoboLab.  This module owns one local
``codex app-server`` process and one Codex thread for an environment episode,
and gives that thread a durable, host-managed observation archive.
"""

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import math
import os
import queue
import re
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

MODEL_PROVIDER = "openai"
ASTRA_MODEL = "gpt-6-astra"
ASTRA_EFFORTS = ("low", "medium", "high", "xhigh", "max")
ASTRA_MODEL_IDS = tuple(f"{ASTRA_MODEL}-{effort}" for effort in ASTRA_EFFORTS)
MAX_LINE_BYTES = 128 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_RECALL_IMAGES = 12
MAX_MEMORY_CALLS_PER_TURN = 16
MAX_NOTES_BYTES = 64 * 1024

MEMORY_LIST_TOOL = "memory_list_observations"
MEMORY_GET_TOOL = "memory_get_observation"
MEMORY_HISTORY_TOOL = "memory_read_history"
MEMORY_READ_NOTES_TOOL = "memory_read_notes"
MEMORY_WRITE_NOTES_TOOL = "memory_write_notes"
MEMORY_TOOL_NAMES = frozenset(
    {
        MEMORY_LIST_TOOL,
        MEMORY_GET_TOOL,
        MEMORY_HISTORY_TOOL,
        MEMORY_READ_NOTES_TOOL,
        MEMORY_WRITE_NOTES_TOOL,
    }
)

BUILTIN_TOOL_ITEMS = frozenset(
    {
        "commandExecution",
        "fileChange",
        "mcpToolCall",
        "collabAgentToolCall",
        "subAgentActivity",
        "webSearch",
        "imageView",
        "sleep",
        "imageGeneration",
    }
)


class AppServerError(RuntimeError):
    """Raised when the Codex App Server session is unusable."""


class MemoryToolError(ValueError):
    """A recoverable, model-visible memory-tool input error."""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(value), allow_nan=False, separators=(",", ":")) + "\n")


def _safe_component(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return result or "agent"


def resolve_agent_memory_root(
    *,
    default_output_dir: str | os.PathLike[str],
) -> tuple[Path, str]:
    """Keep durable agent traces inside the active RoboLab output tree."""

    return (Path(default_output_dir).expanduser().resolve() / "agent_memory"), "robolab_output"


def memory_tool_specs() -> list[dict[str, Any]]:
    """Return bounded dynamic tools for the host-managed observation archive."""

    return [
        {
            "type": "function",
            "name": MEMORY_LIST_TOOL,
            "description": "List saved observations from this RoboLab episode without loading image bodies.",
            "inputSchema": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": MEMORY_GET_TOOL,
            "description": (
                "Load one saved observation and selected images. Omit image_indices only when the observation "
                f"has at most {MAX_RECALL_IMAGES} images."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "observation_id": {"type": "integer", "minimum": 0},
                    "image_indices": {
                        "type": "array",
                        "items": {"type": "integer", "minimum": 0},
                        "maxItems": MAX_RECALL_IMAGES,
                        "uniqueItems": True,
                    },
                },
                "required": ["observation_id"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": MEMORY_HISTORY_TOOL,
            "description": "Read recent observation, policy-call, validation, and host-execution events.",
            "inputSchema": {
                "type": "object",
                "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 100}},
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": MEMORY_READ_NOTES_TOOL,
            "description": "Read the agent's durable notes for this episode.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "type": "function",
            "name": MEMORY_WRITE_NOTES_TOOL,
            "description": "Replace the agent's durable notes for this episode.",
            "inputSchema": {
                "type": "object",
                "properties": {"text": {"type": "string", "maxLength": MAX_NOTES_BYTES}},
                "required": ["text"],
                "additionalProperties": False,
            },
        },
    ]


class EpisodeMemory:
    """Durable observation/tool archive for one policy, episode, and environment."""

    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        policy: str,
        episode_idx: int,
        env_id: int,
        root_source: str = "explicit",
    ) -> None:
        if type(episode_idx) is not int or episode_idx < 0:
            raise ValueError("episode_idx must be a nonnegative integer")
        if type(env_id) is not int or env_id < 0:
            raise ValueError("env_id must be a nonnegative integer")
        base = (
            Path(root).expanduser().resolve()
            / _safe_component(policy)
            / f"episode_{episode_idx:06d}"
            / f"env_{env_id:03d}"
        )
        base.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while (base / f"attempt_{attempt:03d}").exists():
            attempt += 1
        self.path = base / f"attempt_{attempt:03d}"
        self.path.mkdir(parents=False)
        self.observations_path = self.path / "observations"
        self.calls_path = self.path / "calls"
        self.observations_path.mkdir()
        self.calls_path.mkdir()
        self.notes_path = self.path / "NOTES.md"
        self.notes_path.write_text("# Agent episode notes\n", encoding="utf-8")
        self._observations: list[dict[str, Any]] = []
        self._call_index = 0
        _write_json(
            self.path / "memory_manifest.json",
            {
                "schema": "robolab.codex.agent-memory.v1",
                "policy": policy,
                "episode": episode_idx,
                "env_id": env_id,
                "attempt": attempt,
                "memory_root": str(Path(root).expanduser().resolve()),
                "memory_root_source": root_source,
                "created_unix_s": time.time(),
                "observation_semantics": "exact multimodal inputs sent at inference boundaries",
            },
        )

    def record_prompt(self, blocks: Sequence[Mapping[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
        """Persist exact prompt blocks and return App Server input items."""

        observation_id = len(self._observations)
        directory = self.observations_path / f"{observation_id:06d}"
        directory.mkdir()
        app_inputs: list[dict[str, Any]] = []
        text_items: list[str] = []
        images: list[dict[str, Any]] = []
        last_label = ""
        for block in blocks:
            if not isinstance(block, Mapping):
                raise ValueError("prompt blocks must be objects")
            kind = block.get("type")
            if kind == "text" and isinstance(block.get("text"), str):
                text_value = block["text"]
                text_items.append(text_value)
                last_label = text_value[-500:]
                app_inputs.append({"type": "text", "text": text_value})
                continue
            if kind != "image" or block.get("mimeType") not in {"image/jpeg", "image/png"}:
                raise ValueError("agent prompts support text and JPEG/PNG image blocks")
            data = block.get("data")
            if not isinstance(data, str):
                raise ValueError("image block data must be base64 text")
            try:
                payload = base64.b64decode(data, validate=True)
            except (ValueError, binascii.Error) as exc:
                raise ValueError("image block contains invalid base64") from exc
            if not payload or len(payload) > MAX_IMAGE_BYTES:
                raise ValueError("image block size is invalid")
            suffix = ".jpg" if block["mimeType"] == "image/jpeg" else ".png"
            image_path = directory / f"image_{len(images):03d}{suffix}"
            image_path.write_bytes(payload)
            descriptor = {
                "index": len(images),
                "mime_type": block["mimeType"],
                "path": str(image_path.relative_to(self.path)),
                "label": last_label,
                "bytes": len(payload),
            }
            images.append(descriptor)
            app_inputs.append({"type": "localImage", "path": str(image_path), "detail": "high"})
        if not app_inputs:
            raise ValueError("agent prompt must not be empty")
        record = {
            "observation_id": observation_id,
            "created_unix_s": time.time(),
            "image_count": len(images),
            "images": images,
            "text_preview": "\n".join(text_items)[:1000],
            "directory": str(directory.relative_to(self.path)),
        }
        self._observations.append(record)
        _write_json(directory / "prompt.json", {"text": text_items, "images": images})
        _write_json(self.path / "observation_index.json", self._observations)
        _append_jsonl(self.path / "history.jsonl", {"event": "observation", **record})
        return observation_id, app_inputs

    def record_call(self, *, tool: str, arguments: Any, success: bool, result: Any) -> None:
        record = {
            "call": self._call_index,
            "created_unix_s": time.time(),
            "tool": tool,
            "arguments": arguments,
            "success": success,
            "result": result,
        }
        _write_json(self.calls_path / f"{self._call_index:06d}.json", record)
        _append_jsonl(self.path / "history.jsonl", {"event": "tool_call", **record})
        self._call_index += 1

    def record_event(self, event: str, payload: Mapping[str, Any]) -> None:
        _append_jsonl(self.path / "history.jsonl", {"event": event, "created_unix_s": time.time(), **payload})

    def _require_arguments(self, arguments: Any, allowed: set[str]) -> dict[str, Any]:
        if not isinstance(arguments, Mapping):
            raise MemoryToolError("arguments must be a JSON object")
        unexpected = set(arguments) - allowed
        if unexpected:
            raise MemoryToolError(f"unexpected argument(s): {', '.join(sorted(unexpected))}")
        return dict(arguments)

    def call(self, tool: str, arguments: Any) -> list[dict[str, Any]]:
        if tool == MEMORY_LIST_TOOL:
            values = self._require_arguments(arguments, {"limit"})
            limit = values.get("limit", 20)
            if type(limit) is not int or not 1 <= limit <= 100:
                raise MemoryToolError("limit must be an integer in [1, 100]")
            records = self._observations[-limit:]
            return [
                {
                    "type": "inputText",
                    "text": json.dumps(
                        {
                            "observations": records,
                            "notes_available": self.notes_path.exists(),
                            "workspace": str(self.path),
                        },
                        separators=(",", ":"),
                    ),
                }
            ]
        if tool == MEMORY_GET_TOOL:
            values = self._require_arguments(arguments, {"observation_id", "image_indices"})
            observation_id = values.get("observation_id")
            if type(observation_id) is not int or not 0 <= observation_id < len(self._observations):
                raise MemoryToolError("observation_id does not exist in this episode")
            directory = self.observations_path / f"{observation_id:06d}"
            manifest = json.loads((directory / "prompt.json").read_text(encoding="utf-8"))
            images = manifest["images"]
            indices = values.get("image_indices")
            if indices is None:
                if len(images) > MAX_RECALL_IMAGES:
                    raise MemoryToolError(
                        f"observation has {len(images)} images; select at most {MAX_RECALL_IMAGES} image_indices"
                    )
                indices = list(range(len(images)))
            if (
                not isinstance(indices, list)
                or len(indices) > MAX_RECALL_IMAGES
                or any(type(index) is not int or not 0 <= index < len(images) for index in indices)
                or len(set(indices)) != len(indices)
            ):
                raise MemoryToolError(f"image_indices must select at most {MAX_RECALL_IMAGES} unique images")
            content: list[dict[str, Any]] = [
                {
                    "type": "inputText",
                    "text": json.dumps(
                        {
                            "observation_id": observation_id,
                            "text": manifest["text"],
                            "selected_images": [images[index] for index in indices],
                        },
                        separators=(",", ":"),
                    ),
                }
            ]
            for index in indices:
                descriptor = images[index]
                payload = (self.path / descriptor["path"]).read_bytes()
                encoded = base64.b64encode(payload).decode("ascii")
                content.append(
                    {
                        "type": "inputImage",
                        "imageUrl": f"data:{descriptor['mime_type']};base64,{encoded}",
                    }
                )
            return content
        if tool == MEMORY_HISTORY_TOOL:
            values = self._require_arguments(arguments, {"limit"})
            limit = values.get("limit", 20)
            if type(limit) is not int or not 1 <= limit <= 100:
                raise MemoryToolError("limit must be an integer in [1, 100]")
            history_path = self.path / "history.jsonl"
            if history_path.exists():
                lines = history_path.read_text(encoding="utf-8").splitlines()[-limit:]
                events = [json.loads(line) for line in lines]
            else:
                events = []
            return [{"type": "inputText", "text": json.dumps({"events": events}, separators=(",", ":"))}]
        if tool == MEMORY_READ_NOTES_TOOL:
            self._require_arguments(arguments, set())
            return [{"type": "inputText", "text": self.notes_path.read_text(encoding="utf-8")}]
        if tool == MEMORY_WRITE_NOTES_TOOL:
            values = self._require_arguments(arguments, {"text"})
            text = values.get("text")
            if not isinstance(text, str) or not text.strip():
                raise MemoryToolError("text must be a nonempty string")
            if len(text.encode("utf-8")) > MAX_NOTES_BYTES:
                raise MemoryToolError(f"notes must be at most {MAX_NOTES_BYTES} UTF-8 bytes")
            self.notes_path.write_text(text.rstrip() + "\n", encoding="utf-8")
            return [{"type": "inputText", "text": "Episode notes saved."}]
        raise MemoryToolError(f"unknown memory tool: {tool}")


class StdioAppServer:
    """JSON-RPC transport for one local ``codex app-server`` process."""

    def __init__(self, command: Sequence[str], workspace: Path) -> None:
        if not command or not all(isinstance(item, str) and item for item in command):
            raise ValueError("app-server command must be a nonempty string sequence")
        self.command = list(command)
        self.workspace = workspace
        self._incoming: queue.Queue[Any] = queue.Queue()
        self._responses: dict[int, queue.Queue[dict[str, Any]]] = {}
        self._response_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._next_id = 0
        self._closed = False
        self._stdout_closed = threading.Event()
        self._in_log = (workspace / "rpc_in.jsonl").open("a", encoding="utf-8", buffering=1)
        self._out_log = (workspace / "rpc_out.jsonl").open("a", encoding="utf-8", buffering=1)
        self._stderr_log = (workspace / "stderr.log").open("a", encoding="utf-8", buffering=1)
        process_started = time.monotonic()
        self.process = subprocess.Popen(
            self.command,
            cwd=workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            start_new_session=True,
        )
        self.process_spawn_s = time.monotonic() - process_started
        if self.process.stdin is None or self.process.stdout is None or self.process.stderr is None:
            self.close()
            raise AppServerError("Codex app-server stdio pipes are unavailable")
        self._stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self._stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _read_stdout(self) -> None:
        try:
            for line in self.process.stdout:
                self._out_log.write(line)
                if len(line.encode("utf-8")) > MAX_LINE_BYTES:
                    self._incoming.put(AppServerError("Codex app-server message exceeds size limit"))
                    continue
                try:
                    message = json.loads(line)
                except (ValueError, UnicodeDecodeError, RecursionError) as exc:
                    self._incoming.put(AppServerError(f"Codex app-server emitted invalid JSON: {exc}"))
                    continue
                if not isinstance(message, dict):
                    self._incoming.put(AppServerError("Codex app-server emitted a non-object message"))
                    continue
                identifier = message.get("id")
                if identifier is not None and "method" not in message:
                    with self._response_lock:
                        destination = self._responses.get(identifier)
                    if destination is not None:
                        destination.put(message)
                        continue
                self._incoming.put(message)
        finally:
            self._stdout_closed.set()
            error = AppServerError("Codex app-server stdout closed; inspect stderr.log")
            with self._response_lock:
                for destination in self._responses.values():
                    try:
                        destination.put_nowait({"error": {"message": str(error)}})
                    except queue.Full:
                        pass
            self._incoming.put(error)

    def _read_stderr(self) -> None:
        for line in self.process.stderr:
            self._stderr_log.write(line)

    def _send(self, message: Mapping[str, Any]) -> None:
        if self._closed or self._stdout_closed.is_set() or self.process.poll() is not None:
            raise AppServerError("Codex app-server is not running")
        line = json.dumps(dict(message), allow_nan=False, separators=(",", ":")) + "\n"
        with self._write_lock:
            logged = message
            if message.get("method") == "thread/inject_items":
                # Replayed images are byte-identical. Store each once, while sending
                # the unmodified data URLs to Codex. This avoids quadratic video storage.
                logged = copy.deepcopy(message)
                image_store = self.workspace.parent / "history_images"
                image_store.mkdir(exist_ok=True)
                for item in logged["params"]["items"]:
                    for content in item.get("content", []):
                        if content.get("type") != "input_image":
                            continue
                        url = content["image_url"]
                        header, encoded = url.split(",", 1)
                        payload = base64.b64decode(encoded, validate=True)
                        digest = hashlib.sha256(payload).hexdigest()
                        target = image_store / digest
                        if not target.exists():
                            target.write_bytes(payload)
                        content["image_url"] = "artifact-image:../history_images/" + digest
                        content["artifact_data_url_header"] = header
            self._in_log.write(json.dumps(logged, allow_nan=False, separators=(",", ":")) + "\n")
            try:
                self.process.stdin.write(line)
                self.process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise AppServerError("Codex app-server input closed") from exc

    def request(self, method: str, params: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        with self._response_lock:
            self._next_id += 1
            identifier = self._next_id
            destination: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)
            self._responses[identifier] = destination
        try:
            self._send({"id": identifier, "method": method, "params": dict(params)})
            try:
                response = destination.get(timeout=timeout)
            except queue.Empty as exc:
                raise TimeoutError(f"Timed out waiting for Codex RPC {method}") from exc
            if "error" in response:
                error = response.get("error")
                detail = error.get("message") if isinstance(error, Mapping) else error
                raise AppServerError(f"Codex RPC {method} failed: {detail}")
            result = response.get("result")
            if not isinstance(result, dict):
                raise AppServerError(f"Codex RPC {method} returned no object result")
            return result
        finally:
            with self._response_lock:
                self._responses.pop(identifier, None)

    def notify(self, method: str, params: Mapping[str, Any] | None = None) -> None:
        message: dict[str, Any] = {"method": method}
        if params is not None:
            message["params"] = dict(params)
        self._send(message)

    def reply(self, identifier: Any, result: Mapping[str, Any]) -> None:
        self._send({"id": identifier, "result": dict(result)})

    def next_message(self, timeout: float) -> dict[str, Any]:
        try:
            message = self._incoming.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError("Timed out waiting for a Codex turn event") from exc
        if isinstance(message, BaseException):
            raise message
        if not isinstance(message, dict):
            raise AppServerError("Codex app-server emitted an invalid event")
        return message

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
        self._stdout_thread.join(timeout=2)
        self._stderr_thread.join(timeout=2)
        for stream in (self._in_log, self._out_log, self._stderr_log):
            stream.close()


@dataclass(frozen=True)
class TurnResult:
    text: str
    tool_call: dict[str, Any] | None
    observation_id: int
    turn_id: str


class PersistentAppServerSession:
    """One App Server process and one non-ephemeral Codex thread per episode."""

    def __init__(
        self,
        *,
        model_id: str,
        memory: EpisodeMemory,
        output_tools: Sequence[Mapping[str, Any]] = (),
        instructions: str,
        allow_model_requests: bool = False,
        codex_bin: str = "codex",
        app_server_command: Sequence[str] | None = None,
        timeout: float = 120.0,
        options=None,
    ) -> None:
        if allow_model_requests is not True:
            raise ValueError("Real model transport requires explicit allow_model_requests=True")
        if model_id not in ASTRA_MODEL_IDS:
            raise ValueError("Use an exact supported Astra model ID")
        if not math.isfinite(timeout) or not 0 < timeout <= 900:
            raise ValueError("timeout must be in (0, 900]")
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("instructions must be nonempty")
        if app_server_command is None:
            executable = shutil.which(codex_bin)
            if executable is None:
                raise ValueError("Codex executable not found")
            app_server_command = [executable, "app-server", "--stdio", "--strict-config"]
        converted_tools = [dict(tool) for tool in output_tools]
        names = [tool.get("name") for tool in converted_tools]
        if any(not isinstance(name, str) or not name for name in names) or len(set(names)) != len(names):
            raise ValueError("output tools must have unique nonempty names")
        if set(names) & MEMORY_TOOL_NAMES:
            raise ValueError("output tool names collide with memory tools")
        self.options = options
        self.model_id = model_id
        self.model, self.effort = model_id.rsplit("-", 1)
        self.memory = memory
        self.output_tools = converted_tools
        self.output_tool_names = frozenset(names)
        self.instructions = instructions.strip()
        self.timeout = float(timeout)
        self.command = list(app_server_command)
        self.transport: StdioAppServer | None = None
        self.thread_id: str | None = None
        self.failed = False
        self.closed = False
        self.turn_count = 0
        self.rollout_path: Path | None = None
        self.generated_items: list[dict] = []
        self.last_metadata: dict[str, Any] = {}

    @property
    def workspace(self) -> Path:
        return self.memory.path

    def _start(self) -> dict[str, Any]:
        if self.transport is not None:
            return {
                "cold_start": False,
                "process_spawn_s": 0.0,
                "initialize_rpc_s": 0.0,
                "thread_start_rpc_s": 0.0,
                "app_server_cold_start_s": 0.0,
            }
        cold_started = time.monotonic()
        transport = StdioAppServer(self.command, self.workspace)
        self.transport = transport
        try:
            initialize_started = time.monotonic()
            initialized = transport.request(
                "initialize",
                {
                    "clientInfo": {
                        "name": "robolab_codex_policy",
                        "title": "RoboLab Codex policy",
                        "version": "1",
                    },
                    "capabilities": {"experimentalApi": True},
                },
                self.timeout,
            )
            initialize_rpc_s = time.monotonic() - initialize_started
            transport.notify("initialized", {})
            memory_instructions = (
                "A durable observation archive is available through memory_* tools. Memory tools never move the "
                "robot and may be called before the final policy output. Do not use built-in shell, filesystem, "
                "web, MCP, image-view, delegation, or code-execution tools. "
            )
            if self.options:
                from policies.ai_native.agentic.prompts import tool_instructions
                memory_instructions = tool_instructions(self.options)
                if self.options.optional_tools:
                    from policies.ai_native.agentic.coding_workspace import prepare_workspace
                    prepare_workspace(self.workspace)
            if self.output_tools:
                completion = (
                    "Finish each turn with exactly one supplied non-memory policy-output tool. The policy-output "
                    "tool only proposes an action; the host validates and executes it after the turn."
                )
            else:
                completion = "After optional memory calls, return only the JSON value requested by the user prompt."
            developer_instructions = f"{self.instructions}\n\n{memory_instructions}{completion}"
            if self.options and self.options.optional_tools:
                developer_instructions += (
                    " Shell/Python, image inspection, and memory calls may precede that final proposal. "
                    "The one-action limit does not limit analysis-tool calls."
                )
            use_memory = self.options is None or self.options.memory_tools
            tools = [*(memory_tool_specs() if use_memory else []), *self.output_tools]
            if self.options and not self.options.decomposition_scaffolding:
                for tool in tools:
                    if tool["name"] == MEMORY_WRITE_NOTES_TOOL:
                        tool["description"] = (
                            "Replace factual episode notes: measurements, observed events and executed-action outcomes only. "
                            "Do not store plans, current objectives, subgoals, future actions or checklists."
                        )
            config = {'model_reasoning_effort': self.effort}
            if self.options:
                config.update({
                    'features.shell_tool': self.options.optional_tools,
                    'features.view_image': self.options.optional_tools,
                    'tools.update_plan.enabled': False,
                    'features.multi_agent': False,
                    'web_search': 'disabled',
                    'shell_environment_policy.inherit': 'all',
                    'shell_environment_policy.include_only': ['PATH', 'PYTHONPATH', 'LANG', 'LC_ALL', 'HOME', 'TMPDIR'],
                })

            thread_started = time.monotonic()
            response = transport.request(
                "thread/start",
                {
                    "cwd": str(self.workspace),
                    "model": self.model,
                    "modelProvider": MODEL_PROVIDER,
                    "config": config,
                    "approvalPolicy": "never",
                    "sandbox": "workspace-write" if self.options and self.options.optional_tools else "read-only",
                    "ephemeral": False,
                    "baseInstructions": developer_instructions,
                    "developerInstructions": developer_instructions,
                    "dynamicTools": tools,
                    "threadSource": "appServer",
                },
                self.timeout,
            )
            thread_start_rpc_s = time.monotonic() - thread_started
            thread = response.get("thread")
            thread_id = thread.get("id") if isinstance(thread, Mapping) else None
            if not isinstance(thread_id, str) or not thread_id:
                raise AppServerError("Codex App Server returned no thread ID")
            selected_model = response.get("model")
            selected_effort = response.get("reasoningEffort")
            if selected_model not in (None, self.model) or selected_effort not in (None, self.effort):
                raise AppServerError(
                    f"Codex selected {selected_model}/{selected_effort}; expected {self.model}/{self.effort}"
                )
            self.thread_id = thread_id
            self.rollout_path = Path(thread["path"]) if thread.get("path") else None
            startup_timing = {
                "cold_start": True,
                "process_spawn_s": transport.process_spawn_s,
                "initialize_rpc_s": initialize_rpc_s,
                "thread_start_rpc_s": thread_start_rpc_s,
                "app_server_cold_start_s": time.monotonic() - cold_started,
            }
            self.last_metadata.update(
                {
                    "transport": "codex-app-server-stdio",
                    "thread_id": thread_id,
                    "thread_ephemeral": False,
                    "model": self.model,
                    "effort": self.effort,
                    "model_provider": MODEL_PROVIDER,
                    "app_server_pid": transport.process.pid,
                    "app_server_initialize": initialized,
                    "workspace": str(self.workspace),
                    "startup_timing": startup_timing,
                }
            )
            _write_json(self.workspace / "thread.json", self.last_metadata)
            return startup_timing
        except BaseException:
            transport.close()
            self.transport = None
            raise

    def _decode_arguments(self, value: Any) -> dict[str, Any]:
        if isinstance(value, Mapping):
            return dict(value)
        if isinstance(value, str):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError as exc:
                raise MemoryToolError(f"tool arguments are malformed JSON: {exc}") from exc
            if isinstance(decoded, Mapping):
                return dict(decoded)
        raise MemoryToolError("tool arguments must be a JSON object")

    def query(self, blocks: Sequence[Mapping[str, Any]], *, history_items: list[dict] | None = None) -> TurnResult:
        if self.closed or self.failed:
            raise RuntimeError("Session is closed or failed; create a fresh episode session")
        started_at = time.monotonic()
        timing: dict[str, Any] = {}
        # Never attribute a previous turn's usage to a turn whose server did
        # not emit a fresh token-usage notification.
        self.last_metadata.pop("token_usage", None)
        try:
            memory_save_started = time.monotonic()
            observation_id, inputs = self.memory.record_prompt(blocks)
            timing["observation_save_s"] = time.monotonic() - memory_save_started
            if self.options and self.options.optional_tools:
                inputs.append({"type": "text", "text": (
                    f"Current observation files: observations/{observation_id:06d}/prompt.json. "
                    "Image paths in that file are relative to the workspace. "
                    "workspace.json identifies the interpreter and scratch/ directory. "
                    "Use analysis tools when useful before issuing your robot-action proposal."
                )})
            if history_items is None and (self.options is None or self.options.memory_tools):
                inputs.append(
                    {
                        "type": "text",
                        "text": (
                            f"Current persistent observation_id={observation_id}. Older observations can be recalled "
                            "with the memory_* tools. Memory lookup does not count as the one robot policy action."
                        ),
                    }
                )
            timing.update(self._start())
            assert self.transport is not None and self.thread_id is not None
            if history_items:
                self.transport.request("thread/inject_items", {"threadId": self.thread_id, "items": history_items}, self.timeout)
            turn_started_at = time.monotonic()
            turn_response = self.transport.request(
                "turn/start",
                {
                    "threadId": self.thread_id,
                    "input": inputs,
                    "model": self.model,
                    "effort": self.effort,
                    "approvalPolicy": "never",
                    "sandboxPolicy": (
                        {"type": "workspaceWrite", "writableRoots": [str(self.workspace)],
                         "networkAccess": False, "excludeTmpdirEnvVar": True, "excludeSlashTmp": True}
                        if self.options and self.options.optional_tools else
                        {"type": "readOnly", "networkAccess": False}
                    ),
                    "cwd": str(self.workspace),
                    "runtimeWorkspaceRoots": [str(self.workspace)],
                    "summary": "none",
                },
                self.timeout,
            )
            timing["turn_start_rpc_s"] = time.monotonic() - turn_started_at
            turn = turn_response.get("turn")
            turn_id = turn.get("id") if isinstance(turn, Mapping) else None
            if not isinstance(turn_id, str) or not turn_id:
                raise AppServerError("Codex App Server returned no turn ID")
            text_parts: list[str] = []
            output_call: dict[str, Any] | None = None
            memory_calls = 0
            memory_tool_s = 0.0
            output_tool_host_s = 0.0
            first_event_at: float | None = None
            policy_output_at: float | None = None
            deadline = time.monotonic() + self.timeout
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Timed out waiting for Codex to finish the policy turn")
                message = self.transport.next_message(remaining)
                received_at = time.monotonic()
                method = message.get("method")
                params = message.get("params")
                params = params if isinstance(params, Mapping) else {}
                message_thread = params.get("threadId")
                message_turn = params.get("turnId")
                if message_thread not in (None, self.thread_id) or message_turn not in (None, turn_id):
                    continue
                if first_event_at is None and message_turn == turn_id:
                    first_event_at = received_at
                if method == "item/tool/call" and "id" in message:
                    tool_started = time.monotonic()
                    tool = params.get("tool")
                    call_result: list[dict[str, Any]]
                    success = False
                    result_summary: Any
                    try:
                        if not isinstance(tool, str):
                            raise MemoryToolError("tool name is missing")
                        arguments = self._decode_arguments(params.get("arguments"))
                        if tool in MEMORY_TOOL_NAMES:
                            if self.options and not self.options.memory_tools:
                                raise MemoryToolError('Episode memory requires persistent history')
                            if output_call is not None:
                                raise MemoryToolError("memory cannot be queried after the policy output")
                            memory_calls += 1
                            if memory_calls > MAX_MEMORY_CALLS_PER_TURN:
                                raise MemoryToolError("memory-tool call limit reached for this turn")
                            call_result = self.memory.call(tool, arguments)
                            result_summary = {"content_items": len(call_result)}
                        elif tool in self.output_tool_names:
                            if output_call is not None:
                                raise MemoryToolError("a policy-output tool was already accepted")
                            output_call = {"name": tool, "arguments": arguments}
                            if policy_output_at is None:
                                policy_output_at = received_at
                            call_result = [
                                {
                                    "type": "inputText",
                                    "text": (
                                        "Policy output captured for host validation. It has not been executed yet. "
                                        "End the turn without another policy-output call."
                                    ),
                                }
                            ]
                            result_summary = "captured_not_executed"
                        else:
                            raise MemoryToolError("unknown or disabled tool")
                        success = True
                    except MemoryToolError as exc:
                        arguments = params.get("arguments")
                        result_summary = str(exc)
                        call_result = [
                            {
                                "type": "inputText",
                                "text": f"Tool input rejected; nothing was executed: {exc}",
                            }
                        ]
                    self.memory.record_call(
                        tool=str(tool),
                        arguments=arguments,
                        success=success,
                        result=result_summary,
                    )
                    self.transport.reply(message["id"], {"success": success, "contentItems": call_result})
                    tool_elapsed = time.monotonic() - tool_started
                    if tool in MEMORY_TOOL_NAMES:
                        memory_tool_s += tool_elapsed
                    else:
                        output_tool_host_s += tool_elapsed
                    deadline = time.monotonic() + self.timeout
                    continue
                if method == "item/agentMessage/delta":
                    delta = params.get("delta")
                    if isinstance(delta, str):
                        text_parts.append(delta)
                        if policy_output_at is None:
                            policy_output_at = received_at
                    continue
                if method in {"item/started", "item/completed"}:
                    item = params.get("item")
                    if isinstance(item, Mapping):
                        kind = item.get("type")
                        allowed = {'commandExecution', 'imageView', 'fileChange'} if self.options and self.options.optional_tools else set()
                        if kind in BUILTIN_TOOL_ITEMS and kind not in allowed:
                            raise AppServerError(f"Codex attempted disabled built-in tool: {kind}")
                    continue
                if method == "thread/tokenUsage/updated":
                    usage = params.get("tokenUsage")
                    if isinstance(usage, Mapping):
                        self.last_metadata["token_usage"] = _jsonable(usage)
                        _write_json(self.workspace / "token_usage.json", usage)
                    continue
                if method == "turn/completed":
                    completed = params.get("turn")
                    if not isinstance(completed, Mapping) or completed.get("id") != turn_id:
                        continue
                    if completed.get("status") != "completed":
                        error = completed.get("error")
                        raise AppServerError(f"Codex turn ended with status {completed.get('status')}: {error}")
                    if not text_parts:
                        for item in completed.get("items", []):
                            if (
                                isinstance(item, Mapping)
                                and item.get("type") == "agentMessage"
                                and isinstance(item.get("text"), str)
                            ):
                                text_parts.append(item["text"])
                                if policy_output_at is None:
                                    policy_output_at = received_at
                    break
                if "id" in message and isinstance(method, str):
                    self.transport.reply(
                        message["id"],
                        {"success": False, "contentItems": [{"type": "inputText", "text": "Capability disabled"}]},
                    )
                    raise AppServerError(f"Codex requested disabled client capability: {method}")
                if method == "error" and params.get("willRetry") is True:
                    # Reconnect notifications are not terminal turn failures.
                    # Let the server retry within the existing deadline; do not
                    # restart the turn or replay a captured robot action.
                    self.memory.record_event("turn_retry", dict(params))
                    continue
                if method in {"error", "turn/failed"}:
                    raise AppServerError(f"Codex App Server error: {params}")
            if self.output_tools and output_call is None:
                raise AppServerError("Codex completed the turn without a policy-output tool")
            text = "".join(text_parts)
            if not self.output_tools and not text:
                raise AppServerError("Codex completed the turn without a policy response")
            if history_items is not None:
                from policies.ai_native.agentic.history_channels import native_items

                # thread/read synchronizes the completed rollout before reading its native items.
                persisted = self.transport.request("thread/read", {"threadId": self.thread_id, "includeTurns": True}, self.timeout)
                path = persisted["thread"].get("path")
                if not path:
                    raise AppServerError("No persisted native rollout for history reconstruction")
                self.generated_items = native_items(Path(path), turn_id)
                _write_json(self.workspace / "generated-native-items.json", self.generated_items)
            self.turn_count += 1
            completed_at = time.monotonic()
            timing.update(
                {
                    "time_to_first_event_s": (first_event_at - turn_started_at if first_event_at is not None else None),
                    "time_to_policy_output_s": (
                        policy_output_at - turn_started_at if policy_output_at is not None else None
                    ),
                    "time_to_action_tool_s": (
                        policy_output_at - turn_started_at
                        if output_call is not None and policy_output_at is not None
                        else None
                    ),
                    "memory_tool_host_s": memory_tool_s,
                    "policy_tool_host_s": output_tool_host_s,
                    "turn_total_s": completed_at - turn_started_at,
                    "transport_total_s": completed_at - started_at,
                }
            )
            self.last_metadata.update(
                {
                    "turn_id": turn_id,
                    "turn_count": self.turn_count,
                    "observation_id": observation_id,
                    "memory_calls": memory_calls,
                    "wall_time_seconds": time.monotonic() - started_at,
                    "timing": _jsonable(timing),
                }
            )
            self.memory.record_event(
                "turn_completed",
                {
                    "turn_id": turn_id,
                    "observation_id": observation_id,
                    "memory_calls": memory_calls,
                    "output_tool": output_call["name"] if output_call else None,
                    "timing": _jsonable(timing),
                },
            )
            return TurnResult(text=text, tool_call=output_call, observation_id=observation_id, turn_id=turn_id)
        except BaseException as exc:
            self.failed = True
            timing["transport_total_s"] = time.monotonic() - started_at
            self.last_metadata["timing"] = _jsonable(timing)
            self.last_metadata["error_type"] = type(exc).__name__
            self.last_metadata["wall_time_seconds"] = time.monotonic() - started_at
            raise

    def record_host_result(self, payload: Mapping[str, Any]) -> None:
        self.memory.record_event("host_result", dict(payload))

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.transport is not None:
            self.transport.close()
            self.transport = None
        if self.options and self.options.history_channels and self.rollout_path and self.generated_items:
            # Native output and exact inputs already live in the episode artifacts.
            # A fresh thread embeds all selected history; retaining every temporary
            # Codex rollout would duplicate those images quadratically on disk.
            self.rollout_path.unlink(missing_ok=True)


__all__ = [
    "ASTRA_MODEL_IDS",
    "AppServerError",
    "EpisodeMemory",
    "MemoryToolError",
    "PersistentAppServerSession",
    "TurnResult",
    "memory_tool_specs",
    "resolve_agent_memory_root",
]
