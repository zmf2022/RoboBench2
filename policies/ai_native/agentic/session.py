# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent Agent-EEF thread through a Codex App Server subprocess."""

from __future__ import annotations

import base64
import copy
import io
import json
import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from policies.ai_native.agentic.app_server import (
    AGENTIC_MODEL_IDS,
    AppServerError,
    EpisodeMemory,
    PersistentAppServerSession,
)
from policies.ai_native.agentic.demonstration import load_demonstration
from policies.ai_native.agentic.history_channels import ChannelHistory
from policies.ai_native.agentic.joint_actions import action_tools
from policies.ai_native.agentic.prompts import policy_instructions, tool_instructions
from policies.ai_native.agentic.runtime_options import RuntimeOptions
from policies.ai_native.common.agent_eef import (
    EmbodimentSpec,
)


def _dynamic_tools(spec: EmbodimentSpec, options=None) -> list[dict]:
    result = []
    for tool in action_tools(spec, options):
        function = tool["function"]
        result.append(
            {
                "type": "function",
                "name": function["name"],
                "description": function["description"],
                "inputSchema": function["parameters"],
            }
        )
    return result


def observation_blocks(
    request: Mapping,
    *,
    spec: EmbodimentSpec,
    image_size: int = 640,
    jpeg_quality: int = 85,
    include_goal: bool = True,
    options: RuntimeOptions | None = None,
) -> list[dict]:
    """Encode a prepared policy request as direct App Server content blocks."""

    if type(image_size) is not int or not 128 <= image_size <= 2048:
        raise ValueError("image_size must be an integer in [128, 2048]")
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be an integer in [1, 100]")
    policy = request.get("policy")
    if not isinstance(policy, Mapping):
        raise ValueError("agent-eef request requires policy metadata")
    input_metadata = policy.get("input")
    if not isinstance(input_metadata, Mapping):
        raise ValueError("agent-eef request requires input metadata")
    history_length = input_metadata.get("history_length")
    if type(history_length) is not int or history_length < 1:
        raise ValueError("agent-eef history_length must be a positive integer")
    camera_names = input_metadata.get("camera_names")
    if (
        not isinstance(camera_names, list)
        or len(camera_names) != 3
        or len(set(camera_names)) != 3
        or not all(isinstance(name, str) and name for name in camera_names)
    ):
        raise ValueError("agent-eef requires exactly three unique camera names")
    control_steps = input_metadata.get("control_steps")
    if not isinstance(control_steps, list) or not all(type(step) is int and step >= 0 for step in control_steps):
        raise ValueError("agent-eef input control_steps must be nonnegative integers")
    observations = request.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("agent-eef request requires nonempty observations")
    if input_metadata.get("observation_count") != len(observations) or len(control_steps) != len(observations):
        raise ValueError("agent-eef observation metadata does not match supplied observations")
    if len(observations) > history_length:
        raise ValueError("agent-eef observations exceed history_length")
    control = policy.get("control", {})
    if not isinstance(control, Mapping):
        raise ValueError("agent-eef control metadata must be an object")
    prompt_request = copy.deepcopy({key: value for key, value in request.items() if key != "observations"})
    # Legacy callers must not expose evaluator-only task/scorer metadata.
    prompt_request["policy"].pop("task_context", None)
    prompt_request["policy"].pop("simulator_state", None)
    if options and options.history_channels:
        prompt_request["policy"].pop("measured_state_history", None)
    if not include_goal:
        prompt_request["policy"].pop("goal", None)
    control_args = dict(
        control_hz=control.get('control_hz', 15.0),
        max_target_steps=control.get('max_target_steps', 5),
        max_translation_m=control.get('max_translation_m', 0.05),
        max_rotation_rad=control.get('max_rotation_rad', 0.35),
    )
    instructions = policy_instructions(
        options, spec, prompt_family=policy.get("prompt_family", "baseline"), **control_args,
    )
    capabilities = tool_instructions(options)
    if options and options.image_demonstration:
        prompt_request['policy'].pop('goal', None)
    prompt = instructions + ' ' + capabilities + '\n' + json.dumps(prompt_request, allow_nan=False)
    blocks: list[dict] = [{"type": "text", "text": prompt}]
    for index, (observation, expected_step) in enumerate(zip(observations, control_steps, strict=True)):
        if not isinstance(observation, Mapping) or observation.get("control_step") != expected_step:
            raise ValueError("agent-eef observation control steps must match input metadata")
        images = observation.get("images")
        if not isinstance(images, Mapping) or list(images) != camera_names:
            raise ValueError("each agent-eef observation must contain the configured three cameras in order")
        for name, raw in images.items():
            array = np.asarray(raw)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] not in (3, 4) or min(array.shape[:2]) == 0:
                raise ValueError(f"camera {name} must be a nonempty uint8 RGB/RGBA image")
            image = Image.fromarray(array[..., :3])
            resampling = getattr(Image, "Resampling", Image)
            image.thumbnail((image_size, image_size), resampling.BILINEAR)
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=jpeg_quality)
            blocks.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"Observation {index + 1}/{len(observations)}, control step {expected_step}; Camera: {name}"
                        ),
                    },
                    {
                        "type": "image",
                        "mimeType": "image/jpeg",
                        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
                    },
                ]
            )
    return blocks


class CodexAgentEEFSession:
    """One process, thread, and durable observation workspace per episode."""

    def __init__(
        self,
        *,
        spec: EmbodimentSpec,
        model_id: str,
        allow_model_requests: bool = False,
        codex_bin: str = "codex",
        command: list[str] | None = None,
        timeout: float = 120.0,
        image_size: int = 640,
        jpeg_quality: int = 85,
        memory_root: str | Path = "output/codex_agent_memory",
        memory_root_source: str = "default",
        policy_name: str = "agent-eef",
        env_id: int = 0,
        episode_idx: int = 0,
        prompt_family: str = "baseline",
        control_hz: float = 15.0,
        max_target_steps: int = 5,
        max_translation_m: float = 0.05,
        max_rotation_rad: float = 0.35,
        options: RuntimeOptions | None = None,
        demonstration_path: Path | None = None,
    ) -> None:
        if not isinstance(spec, EmbodimentSpec):
            raise ValueError("spec must be an EmbodimentSpec")
        if allow_model_requests is not True:
            raise ValueError("Real model transport requires explicit allow_model_requests=True")
        if model_id not in AGENTIC_MODEL_IDS:
            raise ValueError("Use an exact supported Agentic model ID")
        if not math.isfinite(timeout) or not 0 < timeout <= 300:
            raise ValueError("timeout must be in (0, 300]")
        if type(image_size) is not int or not 128 <= image_size <= 2048:
            raise ValueError("image_size must be an integer in [128, 2048]")
        if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be an integer in [1, 100]")
        if command is not None and (
            not isinstance(command, list) or not command or not all(isinstance(item, str) and item for item in command)
        ):
            raise ValueError("command must be a nonempty list of strings")
        self.options = options
        self.prompt_family = prompt_family
        instructions = policy_instructions(
            options, spec, prompt_family=prompt_family, control_hz=control_hz,
            max_target_steps=max_target_steps, max_translation_m=max_translation_m,
            max_rotation_rad=max_rotation_rad,
        )
        self._demo_blocks = []
        self._demo_provenance = None
        if options and options.image_demonstration:
            if demonstration_path is None:
                raise ValueError('image-demonstration requires --demonstration-manifest')
            self._demo_blocks, self._demo_provenance = load_demonstration(
                demonstration_path, image_size=image_size, jpeg_quality=jpeg_quality,
            )
        elif demonstration_path is not None:
            raise ValueError('Demonstration path requires image_demonstration=True in runtime options')
        self.spec = spec
        self.model_id = model_id
        self.timeout = float(timeout)
        self.image_size = image_size
        self.jpeg_quality = jpeg_quality
        self._artifact_memory_root = Path(memory_root)
        self._neutral_root = Path(tempfile.mkdtemp(prefix="astra-visual-demo-")).resolve() if options and options.image_demonstration else None
        self._memory_root = self._neutral_root or memory_root
        self._memory_args = dict(policy=policy_name, episode_idx=episode_idx, env_id=env_id,
                                 root_source=memory_root_source)
        self._server_args = dict(
            model_id=model_id, output_tools=_dynamic_tools(spec, options), instructions=instructions,
            allow_model_requests=True, codex_bin=codex_bin, app_server_command=command, timeout=timeout,
            options=options,
        )
        self._session = self._new_session()
        self._channel_history = ChannelHistory(text=options.history_text, images=options.history_images) if options and options.history_channels else None
        self._closed = False
        self.last_metadata: dict = {}

    def _new_session(self) -> PersistentAppServerSession:
        memory = EpisodeMemory(self._memory_root, **self._memory_args)
        self.workspace = memory.path
        if self.options:
            provenance = dict(config=self.options.to_dict(), demonstration=self._demo_provenance,
                              system_prompt=self._server_args['instructions'])
            (self.workspace / 'options.json').write_text(json.dumps(provenance, indent=2) + '\n')
        return PersistentAppServerSession(memory=memory, **self._server_args)

    def query(self, request: Mapping) -> tuple[str, dict]:
        if self._closed or self._session.failed:
            raise RuntimeError("Session is closed or failed; create a fresh episode session")
        if request.get("policy", {}).get("prompt_family", "baseline") != self.prompt_family:
            raise ValueError("Agent-EEF session and observation prompt families must match")
        if self.options and not self.options.persistent_history and self._session.turn_count:
            self._session.close()
            self._session = self._new_session()
        started = time.monotonic()
        self.last_metadata = {
            "model_id": self.model_id,
            "prompt_requests": 0,
            "stage": "observation",
            "selected_model_id": self.model_id,
            "image_input_advertised": True,
            "native_image_routing_verified": False,
            "agent_context_isolation_verified": self._session.thread_id is not None,
            "agent_style": "agent-eef",
            "prompt_family": self.prompt_family,
            "history_length": request.get("policy", {}).get("input", {}).get("history_length"),
            "transport": "codex-app-server-stdio",
            "workspace": str(self.workspace),
        }
        try:
            request_started = time.monotonic()
            blocks = observation_blocks(
                request,
                spec=self.spec,
                image_size=self.image_size,
                jpeg_quality=self.jpeg_quality,
                include_goal=self._session.thread_id is None,
                options=self.options,
            )
            if self._demo_blocks and self._session.turn_count == 0:
                blocks = self._demo_blocks + blocks
            request_build_s = time.monotonic() - request_started
            self.last_metadata["stage"] = "turn/start"
            self.last_metadata["prompt_requests"] = 1
            prompt_started = time.monotonic()
            try:
                if self._channel_history is None:
                    result = self._session.query(blocks)
                else:
                    self._channel_history.select(request["policy"]["input"]["control_steps"])
                    audit = self._channel_history.audit()
                    (self.workspace / "history-channels.json").write_text(json.dumps(audit, indent=2) + "\n")
                    result = self._session.query(blocks, history_items=self._channel_history.items)
                    self._channel_history.append(blocks, self._session.generated_items, self._session.output_tool_names)
                    self._session.last_metadata["history_channels"] = audit
            finally:
                self.last_metadata["wall_time_seconds"] = time.monotonic() - prompt_started
            call = result.tool_call
            if (
                not isinstance(call, dict)
                or call.get("name") not in {tool["function"]["name"] for tool in action_tools(self.spec, self.options)}
                or not isinstance(call.get("arguments"), dict)
            ):
                raise AppServerError("Codex App Server returned an invalid Agent-EEF tool envelope")
            self.last_metadata["native_image_routing_verified"] = True
            self.last_metadata["stage"] = "complete"
            self.last_metadata.update(copy.deepcopy(self._session.last_metadata))
            timing = copy.deepcopy(self._session.last_metadata.get("timing", {}))
            timing["request_build_s"] = request_build_s
            self.last_metadata["timing"] = timing
            return call["name"], call["arguments"]
        except Exception as exc:
            self.last_metadata["error_type"] = type(exc).__name__
            raise
        finally:
            self.last_metadata["elapsed_time_seconds"] = time.monotonic() - started
            timing = copy.deepcopy(self.last_metadata.get("timing", {}))
            timing["inference_total_s"] = self.last_metadata["elapsed_time_seconds"]
            self.last_metadata["timing"] = timing
            usage = self._session.last_metadata.get("token_usage")
            self.last_metadata["usage_updates"] = [copy.deepcopy(usage)] if isinstance(usage, dict) else []

    def record_host_result(self, payload: Mapping) -> None:
        self._session.record_host_result(dict(payload))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._session.close()
        if self._neutral_root:
            destination = self._artifact_memory_root / self._neutral_root.name
            shutil.copytree(self._neutral_root, destination)
            shutil.rmtree(self._neutral_root)
            self.workspace = destination / self.workspace.relative_to(self._neutral_root)


__all__ = ["CodexAgentEEFSession", "observation_blocks"]
