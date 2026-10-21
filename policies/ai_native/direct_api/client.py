# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provider-neutral Direct-API client using the Chat Completions protocol.

Seven canonical output representations share one observation and transport
stack. Raw action chunks can contain absolute, observation-anchored relative,
or stepwise delta EEF/joint-position waypoints; ``bspline_jointpos`` emits
compact absolute joint-space B-spline control points.

The input representation is controlled by one integer history length. Every
request contains up to that many latest synchronized image/proprioception
observations. A history length of one means the current observation only.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import sys
import time
import threading
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from dotenv import load_dotenv
from openai import APIError, OpenAI
from openai.types.chat import ChatCompletion
from PIL import Image
from scipy.spatial.transform import Rotation

from robolab.core.task.instruction import InstructionInput, instruction_text
from robolab.core.utils.isaaclab_compat import env_cfg_to_recording_dict
from robolab.eval.base_client import InferenceClient

from policies.ai_native.common import prompts as _prompts
from policies.ai_native.common.agent_eef import agent_eef_state, format_agent_eef_state
from policies.ai_native.common.prompts import DROID_EEF_FRAME_PROMPT, DROID_ROBOT_PROMPT
from policies.ai_native.common.token_usage import TokenUsageTracker

from policies.ai_native.direct_api.eef_specs import get_eef_spec
from policies.ai_native.direct_api.inference_log import InferenceCallLogger
from policies.ai_native.direct_api.setup import write_evaluation_artifacts
from policies.ai_native.direct_api.trajectory import decode_anchored_bspline_trajectory

load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_BASE_URL_ENV = "OPENAI_BASE_URL"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_MODEL_ENV = "AI_NATIVE_MODEL"
DEFAULT_MODEL = "gpt-6-astra"
# Match the ablation completion budget; explicit limits preserve frozen runs.
DEFAULT_MAX_COMPLETION_TOKENS = 8192

DEFAULT_CAMERA_NAMES = (
    "over_shoulder_left_camera",
    "over_shoulder_right_camera",
    "wrist_cam",
)
OUTPUT_MODES = (
    "action_chunk_eef",
    "action_chunk_relative_eef",
    "action_chunk_delta_eef",
    "action_chunk_jointpos",
    "action_chunk_relative_jointpos",
    "action_chunk_delta_jointpos",
    "bspline_jointpos",
)
EEF_OUTPUT_MODES = (
    "action_chunk_eef",
    "action_chunk_relative_eef",
    "action_chunk_delta_eef",
)
RAW_ACTION_OUTPUT_MODES = (
    "action_chunk_eef",
    "action_chunk_relative_eef",
    "action_chunk_delta_eef",
    "action_chunk_jointpos",
    "action_chunk_relative_jointpos",
    "action_chunk_delta_jointpos",
)
PROMPT_FAMILIES = (
    "baseline", "gripper_camera_geometry_prompt",
    "gripper_geometry_prompt", "camera_geometry_prompt",
)
OUTPUT_MODE_ALIASES = {
    "action_chunk": "action_chunk_jointpos",
    "bspline": "bspline_jointpos",
    "zero_shot": "action_chunk_jointpos",
}
METHOD_TO_OUTPUT_MODE = {
    **{mode: mode for mode in OUTPUT_MODES},
    **OUTPUT_MODE_ALIASES,
    "zero_shot_vla": "action_chunk_jointpos",
}
METHODS = tuple(METHOD_TO_OUTPUT_MODE)
# Franka Panda limits in radians. The eighth DROID action is the binary
# Robotiq command used by DroidJointPositionActionCfg (0=open, 1=close).
PANDA_JOINT_LOWER = np.asarray(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973],
    dtype=np.float32,
)
PANDA_JOINT_UPPER = np.asarray(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973],
    dtype=np.float32,
)

# Shared calibration is validated by the runner for camera-geometry families.
from policies.ai_native.common.camera_geometry import DROID_CAMERA_CALIBRATION, validate_droid_camera_calibration


@dataclass(frozen=True)
class _PreparedRequest:
    payload: dict[str, Any]
    env_id: int
    current_joint_position: np.ndarray
    current_gripper_position: float
    call_index: int
    observation_metadata: dict[str, Any]
    current_proprio: Mapping[str, np.ndarray]
    request_build_s: float


@dataclass(frozen=True)
class _RawResponse:
    body: Any
    request_payload: dict[str, Any]
    env_id: int
    current_joint_position: np.ndarray
    current_gripper_position: float
    call_index: int
    observation_metadata: dict[str, Any]
    latency_s: float
    current_proprio: Mapping[str, np.ndarray]
    request_build_s: float


def _as_numpy(value: Any, env_id: int) -> np.ndarray:
    if hasattr(value, "detach"):
        return value[env_id].detach().cpu().numpy()
    array = np.asarray(value)
    return array[env_id] if array.ndim > 0 else array


def _rgb_uint8(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError(f"expected an HxWx3/4 image, got shape {array.shape}")
    if array.shape[-1] == 4:
        array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if array.size and float(np.nanmax(array)) <= 1.0:
            array = array * 255.0
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
    return np.clip(array, 0, 255).astype(np.uint8)


def _resize_for_api(image: np.ndarray, max_side: int) -> Image.Image:
    pil = Image.fromarray(_rgb_uint8(image))
    if max(pil.size) > max_side:
        scale = max_side / max(pil.size)
        size = (max(1, round(pil.width * scale)), max(1, round(pil.height * scale)))
        pil = pil.resize(size, Image.Resampling.LANCZOS)
    return pil


def _image_data_url(image: np.ndarray, *, max_side: int, jpeg_quality: int) -> str:
    buffer = io.BytesIO()
    _resize_for_api(image, max_side).save(
        buffer,
        format="JPEG",
        quality=jpeg_quality,
        optimize=True,
    )
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _resolve_api_key(api_key: str | None, api_key_env: str) -> str:
    resolved = api_key or os.environ.get(api_key_env)
    if not resolved:
        raise ValueError(f"API key is missing; set {api_key_env} or pass api_key")
    return resolved


def _resolve_base_url(base_url: str | None) -> str:
    return base_url or os.environ.get(DEFAULT_BASE_URL_ENV) or DEFAULT_BASE_URL


def resolve_model(model: str | None = None) -> str:
    """Resolve an explicit model, the configured environment value, or the default."""
    return model or os.environ.get(DEFAULT_MODEL_ENV) or DEFAULT_MODEL


def _response_mapping(response: Any) -> Mapping[str, Any]:
    if isinstance(response, Mapping):
        return response
    model_dump = getattr(response, "model_dump", None)
    if not callable(model_dump):
        raise RuntimeError("API provider returned an unsupported chat completion object")
    payload = model_dump(mode="json", exclude_none=True)
    if not isinstance(payload, Mapping):
        raise RuntimeError("API provider returned a non-object chat completion")
    return payload


def _chat_message(response: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        choices = response["choices"]
    except (KeyError, TypeError) as exc:
        raise RuntimeError("API provider returned a malformed chat completion") from exc
    if not isinstance(choices, list) or not choices:
        raise RuntimeError("API provider returned a malformed chat completion")
    text_parts: list[str] = []
    tool_calls: list[Mapping[str, Any]] = []
    for choice in choices:
        message = choice.get("message") if isinstance(choice, Mapping) else None
        if not isinstance(message, Mapping):
            raise RuntimeError("API provider returned a non-object assistant message")
        content = message.get("content")
        if isinstance(content, str) and content:
            text_parts.append(content)
        calls = message.get("tool_calls")
        if isinstance(calls, list):
            tool_calls.extend(call for call in calls if isinstance(call, Mapping))
    return {"role": "assistant", "content": "\n".join(text_parts) or None, "tool_calls": tool_calls}


def _tool_call(response: Mapping[str, Any]) -> Mapping[str, Any]:
    message = _chat_message(response)
    calls = message.get("tool_calls")
    valid_count = isinstance(calls, list) and calls and all(isinstance(call, Mapping) for call in calls)
    if not valid_count or len(calls) != 1:
        raise RuntimeError("model must return exactly one tool call")
    return calls[0]


def list_available_models(
    *,
    api_key: str | None = None,
    api_key_env: str = DEFAULT_API_KEY_ENV,
    base_url: str | None = None,
    timeout_s: float = 30.0,
    client: OpenAI | None = None,
) -> list[str]:
    """Return model IDs visible to the configured OpenAI-compatible API key."""
    resolved_key = _resolve_api_key(api_key, api_key_env)
    resolved_base_url = _resolve_base_url(base_url)
    owns_client = client is None
    sdk = client or OpenAI(
        api_key=resolved_key,
        base_url=resolved_base_url,
        timeout=timeout_s,
    )
    try:
        page = sdk.models.list()
    finally:
        if owns_client:
            sdk.close()
    try:
        return sorted(str(item.id) for item in page.data if getattr(item, "id", None))
    except (AttributeError, TypeError) as exc:
        raise RuntimeError("API provider /models returned a malformed response") from exc


class AINativeClient(InferenceClient):
    """AI-native client for direct action chunks and joint-space splines."""

    def __init__(
        self,
        *,
        model: str | None = None,
        output_mode: str | None = None,
        method: str | None = None,
        api_key: str | None = None,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        base_url: str | None = None,
        action_horizon: int = 15,
        min_action_horizon: int = 3,
        history_length: int = 15,
        history_stride: int = 7,
        history_offsets: Iterable[int] | None = None,
        bspline_min_steps: int | None = None,
        max_api_attempts: int | None = None,
        camera_names: Iterable[str] | None = None,
        image_max_side: int = 384,
        image_detail: str = "low",
        jpeg_quality: int = 80,
        timeout_s: float = 120.0,
        max_retries: int = 2,
        max_output_retries: int = 2,
        max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS,
        max_joint_delta: float = 0.20,
        max_eef_translation_delta_m: float = 0.02,
        max_eef_rotation_delta_rad: float = 0.10,
        embodiment_spec: str = "droid",
        reasoning_effort: str | None = "medium",
        bspline_control_points: int = 6,
        bspline_degree: int = 3,
        inference_log_path: str | os.PathLike[str] | None = None,
        system_prompt: str | None = None,
        prompt_family: str | None = None,
        control_dt_s: float | None = None,
        startup_seed: int | None = None,
        client: OpenAI | None = None,
        clients_by_env: Mapping[int, OpenAI] | None = None,
        max_concurrent_requests: int = 1,
    ) -> None:
        super().__init__()
        if output_mode is not None and output_mode not in (*OUTPUT_MODES, *OUTPUT_MODE_ALIASES):
            raise ValueError(f"output_mode must be one of {OUTPUT_MODES}, got {output_mode!r}")
        if method is not None and method not in METHOD_TO_OUTPUT_MODE:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
        canonical_output_mode = OUTPUT_MODE_ALIASES.get(output_mode, output_mode)
        legacy_output_mode = METHOD_TO_OUTPUT_MODE.get(method) if method is not None else None
        if (
            canonical_output_mode is not None
            and legacy_output_mode is not None
            and canonical_output_mode != legacy_output_mode
        ):
            raise ValueError(f"output_mode={output_mode!r} conflicts with legacy method={method!r}")
        resolved_output_mode = canonical_output_mode or legacy_output_mode or "action_chunk_eef"
        if prompt_family is None and system_prompt is None:
            raise ValueError("Select prompt_family explicitly (baseline or a geometry family), "
                             "or provide system_prompt; the legacy implicit baseline was retired")
        if prompt_family is not None:
            if prompt_family not in PROMPT_FAMILIES:
                raise ValueError(f"unknown prompt family: {prompt_family}; use baseline instead of new_baseline")
            if system_prompt is not None:
                raise ValueError("prompt_family cannot be combined with a custom system prompt")
            if control_dt_s is not None and (not math.isfinite(control_dt_s) or control_dt_s <= 0):
                raise ValueError("prompt families require a positive finite runtime control_dt_s")
        if type(action_horizon) is not int or action_horizon < 1:
            raise ValueError("action_horizon must be a positive integer")
        if type(min_action_horizon) is not int or not 1 <= min_action_horizon <= action_horizon:
            raise ValueError("min_action_horizon must be an integer between 1 and action_horizon")
        if type(history_length) is not int or history_length < 1:
            raise ValueError("history_length must be a positive integer")
        if type(history_stride) is not int or history_stride < 1:
            raise ValueError("history_stride must be a positive integer")
        offsets = tuple(history_offsets) if history_offsets is not None else None
        if offsets is not None:
            if (not offsets or any(type(x) is not int or x < 0 for x in offsets)
                    or tuple(sorted(set(offsets), reverse=True)) != offsets or offsets[-1] != 0):
                raise ValueError("history_offsets must be unique descending nonnegative integers ending in 0")
            history_length = offsets[0] + 1
        spline_min = (max(min_action_horizon, min(bspline_control_points, action_horizon))
                      if bspline_min_steps is None else bspline_min_steps)
        if type(spline_min) is not int or not min_action_horizon <= spline_min <= action_horizon:
            raise ValueError("bspline_min_steps must be within the configured action horizon")
        if image_max_side < 32:
            raise ValueError("image_max_side must be >= 32")
        if image_detail not in {"low", "high", "auto"}:
            raise ValueError("image_detail must be low, high, or auto")
        if not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if max_joint_delta <= 0:
            raise ValueError("max_joint_delta must be > 0")
        if max_eef_translation_delta_m <= 0:
            raise ValueError("max_eef_translation_delta_m must be > 0")
        if max_eef_rotation_delta_rad <= 0:
            raise ValueError("max_eef_rotation_delta_rad must be > 0")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if max_api_attempts is not None and (type(max_api_attempts) is not int or max_api_attempts < 1):
            raise ValueError("max_api_attempts must be a positive integer")
        if max_output_retries < 0:
            raise ValueError("max_output_retries must be >= 0")
        if max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be >= 1")
        if bspline_control_points < 1:
            raise ValueError("bspline_control_points must be >= 1")
        if bspline_degree < 0:
            raise ValueError("bspline_degree must be >= 0")

        env_clients = dict(clients_by_env) if clients_by_env is not None else {}
        if clients_by_env is not None:
            if not env_clients or any(type(i) is not int or i < 0 for i in env_clients):
                raise ValueError("clients_by_env must map nonnegative environment IDs to API clients")
            if any(transport is None for transport in env_clients.values()):
                raise ValueError("clients_by_env cannot contain a missing API client")
            if client is not None or api_key is not None:
                raise ValueError("clients_by_env cannot be combined with client or api_key")
        if type(max_concurrent_requests) is not int or max_concurrent_requests < 1:
            raise ValueError("max_concurrent_requests must be a positive integer")
        resolved_base_url = _resolve_base_url(base_url)
        resolved_model = resolve_model(model)

        self.model = resolved_model
        self.output_mode = resolved_output_mode
        # Compatibility attribute for callers that still inspect ``method``.
        self.method = resolved_output_mode
        if embodiment_spec != "droid":
            raise ValueError("AI-native EEF control supports only the DROID embodiment")
        self.embodiment_spec = get_eef_spec(embodiment_spec)
        self.open_loop_horizon = None
        self.max_action_horizon = int(action_horizon)
        self.min_action_horizon = min_action_horizon
        self.history_length = history_length
        self.history_stride = history_stride
        self.history_offsets = offsets
        self.bspline_min_steps = spline_min
        self._stop_requests = threading.Event()
        self.camera_names = tuple(DEFAULT_CAMERA_NAMES if camera_names is None else camera_names)
        if not self.camera_names:
            raise ValueError("camera_names must contain at least one camera")
        self.image_max_side = int(image_max_side)
        self.image_detail = image_detail
        self.jpeg_quality = int(jpeg_quality)
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.max_api_attempts = max_api_attempts
        self.max_output_retries = int(max_output_retries)
        self.max_completion_tokens = int(max_completion_tokens)
        self.max_joint_delta = float(max_joint_delta)
        self.max_eef_translation_delta_m = float(max_eef_translation_delta_m)
        self.max_eef_rotation_delta_rad = float(max_eef_rotation_delta_rad)
        self.reasoning_effort = reasoning_effort
        self.bspline_control_points = int(bspline_control_points)
        self.bspline_degree = int(bspline_degree)
        self.prompt_family = prompt_family
        self.control_dt_s = control_dt_s
        self.system_prompt = system_prompt
        if prompt_family is not None and control_dt_s is not None:
            self._configure_control_timing(control_dt_s)
        self.inference_logger = InferenceCallLogger(inference_log_path)
        retry_path = Path(inference_log_path) if inference_log_path is not None else None
        self.api_retry_logger = InferenceCallLogger(
            retry_path.with_name(f"{retry_path.stem}_api_retries.jsonl") if retry_path is not None else None
        )
        self.provider_request_logger = InferenceCallLogger(
            retry_path.with_name(f"{retry_path.stem}_provider_requests.jsonl") if retry_path is not None else None
        )
        self._api_retry_counts: dict[tuple[int, int], int] = {}
        self._episode_api_retries: dict[int, int] = {}
        # Injected transports belong to the caller; only close clients we create.
        self._owns_client = client is None and clients_by_env is None
        self._env_clients = env_clients
        self.client = (next(iter(env_clients.values())) if env_clients else client)
        if self.client is None:
            self.client = OpenAI(
                api_key=_resolve_api_key(api_key, api_key_env),
                base_url=resolved_base_url,
                timeout=timeout_s,
                max_retries=max_retries,
            )
        self._executor = (
            ThreadPoolExecutor(max_workers=max_concurrent_requests, thread_name_prefix="direct-api")
            if max_concurrent_requests > 1 else None
        )
        self._observation_buffers: dict[int, list[dict[str, Any]]] = {}
        self._interval_proprio_buffers: dict[int, list[dict[str, Any]]] = {}
        self._control_steps: dict[int, int] = {}
        self._inference_calls: dict[int, int] = {}
        self._token_usage: dict[int, TokenUsageTracker] = {}
        self._call_token_usage: dict[tuple[int, int], TokenUsageTracker] = {}
        self._provider_models: dict[tuple[int, int], list[str]] = {}
        self._startup_seed = startup_seed
        self.configuration_sha256: str | None = None
        self.run_metadata: dict[str, Any] | None = None

    def _configure_control_timing(self, control_dt_s: float) -> None:
        if not math.isfinite(control_dt_s) or control_dt_s <= 0:
            raise ValueError("prompt families require a positive finite runtime control_dt_s")
        self.control_dt_s = control_dt_s
        if self.prompt_family is not None:
            build_prompt = getattr(_prompts, "baseline_prompt" if self.prompt_family == "baseline" else self.prompt_family)
            self.system_prompt = build_prompt(
                control_dt_s=control_dt_s, output_mode=self.output_mode,
                translation_limit=self.max_eef_translation_delta_m,
                rotation_limit=self.max_eef_rotation_delta_rad,
                max_joint_delta=self.max_joint_delta,
            )

    def configure_environment(self, env_cfg) -> None:
        """Prepare prompts and evaluation artifacts before an episode starts."""
        if self.prompt_family in {"gripper_camera_geometry_prompt", "camera_geometry_prompt"}:
            validate_droid_camera_calibration(env_cfg_to_recording_dict(env_cfg))
        self._configure_control_timing(float(env_cfg.sim.dt * env_cfg.decimation))
        write_evaluation_artifacts(self)

    def infer_batch(self, obs: Any, instruction: InstructionInput, *, env_ids: list[int]) -> dict[int, dict]:
        """Snapshot on the simulator thread; overlap independent API calls only.

        Vectorized simulation remains synchronous: no environment advances until
        all active environments have a valid action. Each environment has its
        own history, retry counters, token tracker and action chunk.
        """
        if len(set(env_ids)) != len(env_ids):
            raise ValueError("env_ids must be unique")
        if self._env_clients and any(i not in self._env_clients for i in env_ids):
            raise ValueError("No API client assigned to an active environment")
        extracted = {i: self._extract_observation(obs, env_id=i) for i in env_ids}
        requests = {i: self._pack_request(extracted[i], instruction)
                    for i in env_ids if self._needs_refresh(i)}
        def resolve(request):
            return self._postprocess_chunk(self._unpack_response(self._query_server(request)))
        futures = {}
        try:
            if self._executor:
                futures = {i: self._executor.submit(resolve, request) for i, request in requests.items()}
                done, _ = wait(futures.values(), return_when=FIRST_EXCEPTION)
                for future in done:
                    if future.exception() is not None:
                        future.result()
                chunks = {i: future.result() for i, future in futures.items()}
            else:
                chunks = {i: resolve(request) for i, request in requests.items()}
        except BaseException:
            self._stop_requests.set()
            for future in futures.values():
                future.cancel()
            wait(futures.values())
            raise
        for i, chunk in chunks.items():
            self._set_chunk(i, chunk)
        return {i: {"action": self._next_action(i), "viz": self._build_visualization(extracted[i])}
                for i in env_ids}

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        try:
            image_obs = raw_obs["image_obs"]
            proprio_obs = raw_obs["proprio_obs"]
        except (KeyError, TypeError) as exc:
            raise KeyError("observation must contain image_obs and proprio_obs groups") from exc

        images: dict[str, np.ndarray] = {}
        for name in self.camera_names:
            if name not in image_obs:
                raise KeyError(f"camera {name!r} is missing from image_obs")
            images[name] = _rgb_uint8(_as_numpy(image_obs[name], env_id))

        proprio = {
            name: np.asarray(_as_numpy(value, env_id), dtype=np.float32).reshape(-1)
            for name, value in proprio_obs.items()
        }
        joints = proprio.get("arm_joint_pos")
        gripper = proprio.get("gripper_pos")
        if joints is None or joints.shape != (7,):
            raise ValueError("proprio_obs.arm_joint_pos must contain 7 values per environment")
        if gripper is None or gripper.size < 1:
            raise ValueError("proprio_obs.gripper_pos must contain at least one value per environment")
        if self.output_mode in EEF_OUTPUT_MODES:
            # Keep the complete DROID proprioception packet in the prompt,
            # but require the measured base_link pose used to anchor EEF
            # safety and malformed-output holds.
            format_agent_eef_state(self.embodiment_spec, proprio)
        if not all(np.all(np.isfinite(value)) for value in proprio.values()):
            raise ValueError("proprioception contains non-finite values")

        # Copy every value because simulator observation arrays may alias
        # storage that is updated in-place on the next control step.
        control_step = self._control_steps.get(env_id, 0)
        self._control_steps[env_id] = control_step + 1
        frame = {
            "control_step": control_step,
            "images": {name: image.copy() for name, image in images.items()},
            "proprio": {name: value.copy() for name, value in proprio.items()},
        }
        buffer = self._observation_buffers.setdefault(env_id, [])
        buffer.append(frame)
        if len(buffer) > self.history_length:
            del buffer[: -self.history_length]
        self._interval_proprio_buffers.setdefault(env_id, []).append(
            {
                "control_step": control_step,
                "proprio": {name: value.copy() for name, value in proprio.items()},
            }
        )
        return {
            "env_id": env_id,
            "images": images,
            "proprio": proprio,
            "current_frame": frame,
        }

    def _pack_request(self, extracted_obs: dict, instruction: InstructionInput) -> _PreparedRequest:
        instruction = instruction_text(instruction)
        request_started = time.perf_counter()
        env_id = int(extracted_obs["env_id"])
        buffered_frames = self._observation_buffers.get(env_id, [])
        history_window = (buffered_frames or [extracted_obs["current_frame"]])[-self.history_length :]
        # Anchor sampling at the newest observation so every request includes
        # the current robot state. For H=15 and stride=7, a full window sends
        # control steps t-14, t-7, and t, oldest to newest.
        if self.history_offsets is None:
            frames = list(reversed(history_window[::-1][:: self.history_stride]))
        else:
            by_step = {int(frame["control_step"]): frame for frame in history_window}
            current_step = int(extracted_obs["current_frame"]["control_step"])
            frames = [by_step[current_step - offset] for offset in self.history_offsets
                      if current_step - offset in by_step]
        interval_frames = self._interval_proprio_buffers.get(env_id, []) or [
            {
                "control_step": int(extracted_obs["current_frame"]["control_step"]),
                "proprio": extracted_obs["current_frame"]["proprio"],
            }
        ]
        interval_observation_count = len(interval_frames)
        # Keep the rolling observation history across inference calls, but
        # rotate the lightweight interval log after every request.
        self._interval_proprio_buffers[env_id] = []
        proprio_history = [
            {
                "frame": index,
                "control_step": int(frame["control_step"]),
                **{name: np.asarray(value).round(5).tolist() for name, value in frame["proprio"].items()},
            }
            for index, frame in enumerate(frames)
        ]
        interval_proprio_history = [
            {
                "frame": index,
                "control_step": int(frame["control_step"]),
                **{name: np.asarray(value).round(5).tolist() for name, value in frame["proprio"].items()},
            }
            for index, frame in enumerate(interval_frames)
        ]
        observation_text = (
            f"Observation history window: {self.history_length} control step(s).\n"
            + (f"History sample offsets: {list(self.history_offsets)}.\n" if self.history_offsets is not None
             else f"History sample stride: {self.history_stride}.\n")
            + f"Instruction: {instruction}\n"
            f"This request contains the latest {len(frames)} chronological observation(s), oldest first. "
            f"Camera order per frame: {list(self.camera_names)}.\n"
            f"Synchronized proprioception: {json.dumps(proprio_history, separators=(',', ':'))}"
        )
        content: list[dict[str, Any]] = [{"type": "text", "text": observation_text}]
        for frame_index, frame in enumerate(frames):
            for camera_name in self.camera_names:
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": (
                                f"Observation {frame_index + 1}/{len(frames)}, control step "
                                f"{int(frame['control_step'])}; Camera: {camera_name}"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": _image_data_url(
                                    frame["images"][camera_name],
                                    max_side=self.image_max_side,
                                    jpeg_quality=self.jpeg_quality,
                                ),
                                "detail": self.image_detail,
                            },
                        },
                    ]
                )

        messages = [
            {"role": "system", "content": self._method_prompt()},
            {"role": "user", "content": f"Goal: {instruction}"},
            {"role": "user", "content": content},
        ]
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "tools": self._tool_schemas(),
            "tool_choice": "required",
            "max_completion_tokens": self.max_completion_tokens,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        call_index = self._inference_calls.get(env_id, 0)
        self._inference_calls[env_id] = call_index + 1
        observation_metadata = {
            "frame_count": len(frames),
            "history_length": self.history_length,
            "history_stride": self.history_stride,
            "history_offsets": list(self.history_offsets) if self.history_offsets is not None else None,
            "control_steps": [int(frame["control_step"]) for frame in frames],
            "camera_names": list(self.camera_names),
            "media_parts": len(self.camera_names) * len(frames),
            "proprioception": proprio_history,
            "executed_interval_frame_count": interval_observation_count,
            "executed_interval_proprioception": interval_proprio_history,
        }
        return _PreparedRequest(
            payload=payload,
            env_id=env_id,
            current_joint_position=(
                np.empty(0, dtype=np.float32)
                if self.output_mode in EEF_OUTPUT_MODES
                else extracted_obs["proprio"]["arm_joint_pos"].copy()
            ),
            current_gripper_position=(
                0.0
                if self.output_mode in EEF_OUTPUT_MODES
                else float(extracted_obs["proprio"]["gripper_pos"][0])
            ),
            call_index=call_index,
            observation_metadata=observation_metadata,
            current_proprio={name: value.copy() for name, value in extracted_obs["proprio"].items()},
            request_build_s=time.perf_counter() - request_started,
        )

    def _method_prompt(self) -> str:
        if self.prompt_family is not None and self.control_dt_s is None:
            raise RuntimeError("Configure the environment before building a policy prompt")
        if self.output_mode == "action_chunk_eef":
            method = (
                f"Respond with exactly one emit_eef_actions tool call containing between {self.min_action_horizon} and "
                f"{self.max_action_horizon} consecutive absolute 8D EEF actions. Each action is "
                "[x, y, z, qw, qx, qy, qz, gripper] for the Robotiq base_link in the DROID "
                "robot-root frame. Specify every executed control target explicitly; the client does not "
                "interpolate between model-emitted actions. Choose the number of actions needed before the "
                "next visual replan, using shorter chunks near contact."
            )
        elif self.output_mode == "action_chunk_relative_eef":
            method = (
                f"Respond with exactly one emit_relative_eef_actions tool call containing between "
                f"{self.min_action_horizon} and {self.max_action_horizon} consecutive 7D relative EEF "
                "waypoints. Each action is [dx, dy, dz, drx, dry, drz, gripper] and represents "
                "inverse(T_anchor) @ T_target_k for the Robotiq base_link, where T_anchor is the "
                "newest observed ee_pos/ee_quat. Use that SAME anchor for every row; do not make a "
                "row relative to the preceding row. Specify every executed control target explicitly; "
                "the client does not interpolate between model-emitted actions. Choose the number of "
                "actions needed before the next visual replan, using shorter chunks near contact."
            )
        elif self.output_mode == "action_chunk_delta_eef":
            method = (
                f"Respond with exactly one emit_delta_eef_actions tool call containing between "
                f"{self.min_action_horizon} and {self.max_action_horizon} consecutive 7D delta EEF "
                "waypoints. Each action is [dx, dy, dz, drx, dry, drz, gripper] and represents "
                "inverse(T_target_{k-1}) @ T_target_k for the Robotiq base_link. Start with "
                "T_target_0 equal to the newest observed ee_pos/ee_quat, then make every later "
                "row relative to the PRECEDING target. Specify every executed control target "
                "explicitly; the client does not interpolate between model-emitted actions. "
                "Choose the number of actions needed before the next visual replan, using shorter "
                "chunks near contact."
            )
        elif self.output_mode == "action_chunk_jointpos":
            method = (
                f"Respond with exactly one emit_joint_actions tool call containing between {self.min_action_horizon} and "
                f"{self.max_action_horizon} consecutive absolute 8D joint-position actions. Each action is "
                "[panda_joint1..panda_joint7, gripper]. Specify every executed control target explicitly; "
                "the client does not interpolate between model-emitted actions. Choose the number of actions "
                "needed before the next visual replan, using shorter chunks near contact."
            )
        elif self.output_mode == "action_chunk_relative_jointpos":
            method = (
                f"Respond with exactly one emit_relative_joint_actions tool call containing between "
                f"{self.min_action_horizon} and {self.max_action_horizon} consecutive 8D relative "
                "joint-position waypoints. Each action is [dq1..dq7, gripper], where dq_k is the "
                "future absolute joint target minus arm_joint_pos from the newest observation. Use "
                "that SAME observed joint anchor for every row; do not make a row relative to the "
                "preceding row. Specify every executed control target explicitly; the client does not "
                "interpolate between model-emitted actions. Choose the number of actions needed before "
                "the next visual replan, using shorter chunks near contact."
            )
        elif self.output_mode == "action_chunk_delta_jointpos":
            method = (
                f"Respond with exactly one emit_delta_joint_actions tool call containing between "
                f"{self.min_action_horizon} and {self.max_action_horizon} consecutive 8D delta "
                "joint-position waypoints. Each action is [dq1..dq7, gripper], where dq_k is "
                "q_target_k minus q_target_{k-1}. Start with q_target_0 equal to arm_joint_pos "
                "from the newest observation, then make every later row relative to the PRECEDING "
                "target. Specify every executed control target explicitly; the client does not "
                "interpolate between model-emitted actions. Choose the number of actions needed "
                "before the next visual replan, using shorter chunks near contact."
            )
        else:
            minimum_steps = self.bspline_min_steps
            method = (
                "Respond with exactly one emit_joint_bspline tool call containing "
                f"exactly {self.bspline_control_points} FUTURE B-spline control points. Each control point is an "
                "absolute 8D vector [panda_joint1..panda_joint7, gripper]. The client prepends the current "
                f"robot state, decodes a clamped degree-{self.bspline_degree} B-spline into "
                f"the requested {minimum_steps} to {self.max_action_horizon} future actions, then applies local joint and "
                "gripper safety checks. Choose the shortest useful horizon and a smooth curve whose final "
                "control point makes useful task progress."
            )
        if self.prompt_family is not None:
            method = method.replace(
                "Choose the number of actions needed before the next visual replan, using shorter chunks near contact.",
                "Choose the number of actions needed before the next visual replan.",
            )
        return f"{self.system_prompt}\n\n{method}" if self.system_prompt else method

    def _tool_schemas(self) -> list[dict[str, Any]]:
        return [self._tool_schema()]

    def _tool_schema(self) -> dict[str, Any]:
        if self.output_mode in RAW_ACTION_OUTPUT_MODES:
            name, description, action_dim = {
                "action_chunk_eef": (
                    "emit_eef_actions",
                    "Emit a variable-length raw DROID absolute EEF action chunk.",
                    8,
                ),
                "action_chunk_relative_eef": (
                    "emit_relative_eef_actions",
                    "Emit observation-anchored relative DROID EEF waypoints with an absolute gripper command.",
                    7,
                ),
                "action_chunk_delta_eef": (
                    "emit_delta_eef_actions",
                    "Emit stepwise delta DROID EEF waypoints with an absolute gripper command.",
                    7,
                ),
                "action_chunk_jointpos": (
                    "emit_joint_actions",
                    "Emit a variable-length raw DROID absolute joint-position action chunk.",
                    8,
                ),
                "action_chunk_relative_jointpos": (
                    "emit_relative_joint_actions",
                    "Emit observation-anchored relative DROID joint-position waypoints with an absolute gripper command.",
                    8,
                ),
                "action_chunk_delta_jointpos": (
                    "emit_delta_joint_actions",
                    "Emit stepwise delta DROID joint-position waypoints with an absolute gripper command.",
                    8,
                ),
            }[self.output_mode]
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "actions": {
                                "type": "array",
                                "items": {
                                    "type": "array",
                                    "items": {"type": "number"},
                                    "minItems": action_dim,
                                    "maxItems": action_dim,
                                },
                                "minItems": self.min_action_horizon,
                                "maxItems": self.max_action_horizon,
                            },
                            "note": {"type": "string"},
                        },
                        "required": ["actions", "note"],
                        "additionalProperties": False,
                    },
                },
            }
        return {
            "type": "function",
            "function": {
                "name": "emit_joint_bspline",
                "description": (
                    "Emit future absolute DROID joint-space B-spline control points and a variable output horizon; "
                    "the current robot state is anchored locally."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "control_points": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number"},
                                "minItems": 8,
                                "maxItems": 8,
                            },
                            "minItems": self.bspline_control_points,
                            "maxItems": self.bspline_control_points,
                        },
                        "steps": {
                            "type": "integer",
                            "minimum": self.bspline_min_steps,
                            "maximum": self.max_action_horizon,
                        },
                        "note": {"type": "string"},
                    },
                    "required": ["control_points", "steps", "note"],
                    "additionalProperties": False,
                },
            },
        }

    def _request_with_api_retries(
        self, payload: dict[str, Any], *, env_id: int, call_index: int, stage: str
    ) -> Any:
        """Keep the observation frozen until the provider responds or the user interrupts.

        SDK retries happen inside each attempt. Persistent provider failures,
        including HTTP 400/429, must never consume simulator control steps.
        Local programming errors and KeyboardInterrupt are not swallowed.
        """
        delay_s = 5.0
        failures = 0
        while True:
            if self._stop_requests.is_set():
                raise RuntimeError("Direct-API request cancelled")
            request_started = time.perf_counter()
            try:
                transport = self._env_clients[env_id] if self._env_clients else self.client
                # The generated create() wrapper can leak SDK Omit defaults into
                # JSON under Isaac Sim's typing environment. Our payload is
                # already complete; keep SDK transport/retries and typed parsing
                # without introducing optional sentinel values.
                response = transport.post(
                    "/chat/completions", body=payload, cast_to=ChatCompletion
                )
            except (APIError, TimeoutError, ConnectionError) as exc:
                failures += 1
                self._record_model_request(env_id, call_index, None, stage=stage,
                                           latency_s=time.perf_counter() - request_started,
                                           error_type=type(exc).__name__)
                key = (env_id, call_index)
                self._api_retry_counts[key] = self._api_retry_counts.get(key, 0) + 1
                self._episode_api_retries[env_id] = self._episode_api_retries.get(env_id, 0) + 1
                message = str(exc)
                exhausted = self.max_api_attempts is not None and failures >= self.max_api_attempts
                retry_message = ("Provider attempts exhausted; stopping this worker." if exhausted
                                 else f"Simulator waiting; retrying the same request in {delay_s:g}s.")
                print(
                    f"[Direct-API {self.output_mode}] episode={self._eval_episode_idx} "
                    f"env={env_id} inference_call={call_index} stage={stage} "
                    f"API failure {failures}: {type(exc).__name__}: {message}. "
                    + retry_message,
                    file=sys.stderr,
                    flush=True,
                )
                self.api_retry_logger.write({
                    "episode": self._eval_episode_idx,
                    "env_id": env_id,
                    "inference_call": call_index,
                    "output_mode": self.output_mode,
                    "model": self.model,
                    "stage": stage,
                    "status": "api_failed" if exhausted else "api_retry",
                    "retry": self._api_retry_counts[key],
                    "retry_delay_s": 0 if exhausted else delay_s,
                    "error": {"type": type(exc).__name__, "message": message,
                              "status_code": getattr(exc, "status_code", None)},
                })
                if exhausted:
                    raise
                if self._stop_requests.wait(delay_s):
                    raise RuntimeError("Direct-API request cancelled") from exc
                delay_s = min(60.0, delay_s * 2.0)
                continue
            self._record_model_request(env_id, call_index, response, stage=stage,
                                       latency_s=time.perf_counter() - request_started)
            if failures:
                print(
                    f"[Direct-API {self.output_mode}] episode={self._eval_episode_idx} "
                    f"env={env_id} inference_call={call_index} stage={stage} "
                    f"API recovered after {failures} failure(s); validating response.",
                    file=sys.stderr,
                    flush=True,
                )
            return response

    def _query_server(self, request: _PreparedRequest) -> _RawResponse:
        started = time.perf_counter()
        response = self._request_with_api_retries(
            request.payload, env_id=request.env_id, call_index=request.call_index, stage="query"
        )
        return _RawResponse(
            # Preserve the provider object until the guarded decode stage.
            # Some OpenAI-compatible providers can return a malformed or
            # otherwise unsupported completion envelope; converting here
            # would let that error escape before output repair can run.
            body=response,
            request_payload=request.payload,
            env_id=request.env_id,
            current_joint_position=request.current_joint_position,
            current_gripper_position=request.current_gripper_position,
            call_index=request.call_index,
            observation_metadata=request.observation_metadata,
            latency_s=time.perf_counter() - started,
            current_proprio=request.current_proprio,
            request_build_s=request.request_build_s,
        )

    def _unpack_response(self, response: _RawResponse) -> np.ndarray:
        """Decode a valid chunk before allowing the simulator to step.

        Provider-side structured-output enforcement is not infallible.  Keep
        the original observation frozen while asking the model to repair an
        invalid tool call.  If every repair attempt is exhausted, cache a
        hold-position chunk so a bad response cannot terminate the episode or
        send partially decoded actions to the robot.
        """
        initial_response = response
        current_response = response
        total_latency_s = response.latency_s
        output_attempts = 1
        output_errors: list[dict[str, Any]] = []

        for repair_attempt in range(self.max_output_retries + 1):
            try:
                actions, arguments, safety_clamped = self._decode_response(current_response)
            except Exception as exc:  # Invalid model output must never escape into the control loop.
                output_errors.append(
                    {
                        "attempt": repair_attempt,
                        "stage": "decode",
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "repair_context": self._output_repair_context(current_response),
                    }
                )
                if repair_attempt >= self.max_output_retries:
                    break

                logger.warning(
                    "[%s] invalid tool output for inference call %d; requesting repair %d/%d: %s",
                    self.output_mode,
                    response.call_index,
                    repair_attempt + 1,
                    self.max_output_retries,
                    exc,
                )
                started = time.perf_counter()
                try:
                    current_response = self._query_output_repair(
                        current_response,
                        error=exc,
                        repair_attempt=repair_attempt + 1,
                    )
                except Exception as repair_exc:
                    total_latency_s += time.perf_counter() - started
                    output_attempts += 1
                    output_errors.append(
                        {
                            "attempt": repair_attempt + 1,
                            "stage": "repair_query",
                            "type": type(repair_exc).__name__,
                            "message": str(repair_exc),
                        }
                    )
                    # The next iteration will retry from the last invalid
                    # response without ever advancing the simulator.
                    continue
                total_latency_s += current_response.latency_s
                output_attempts += 1
                continue

            note = arguments.get("note")
            if isinstance(note, str) and note.strip():
                logger.info("[%s] %s", self.output_mode, note.strip())
            self._log_inference(
                env_id=response.env_id,
                call_index=response.call_index,
                observation_metadata=response.observation_metadata,
                request_build_s=response.request_build_s,
                api_latency_s=total_latency_s,
                status="ok",
                model_arguments=arguments,
                actions=actions,
                safety_clamped=safety_clamped,
                output_attempts=output_attempts,
                output_errors=output_errors,
            )
            return actions

        actions = self._safe_hold_actions(initial_response)
        logger.error(
            "[%s] inference call %d produced no valid tool output after %d API attempt(s); executing safe hold",
            self.output_mode,
            response.call_index,
            output_attempts,
        )
        self._log_inference(
            env_id=response.env_id,
            call_index=response.call_index,
            observation_metadata=response.observation_metadata,
            request_build_s=response.request_build_s,
            api_latency_s=total_latency_s,
            status="fallback",
            actions=actions,
            safety_clamped=False,
            output_attempts=output_attempts,
            output_errors=output_errors,
            fallback_reason="invalid model output; holding current robot state",
        )
        return actions

    def _decode_response(self, response: _RawResponse) -> tuple[np.ndarray, dict[str, Any], bool]:
        response_body = _response_mapping(response.body)
        call = _tool_call(response_body)
        function = call.get("function")
        if not isinstance(function, Mapping):
            raise RuntimeError("model returned a malformed function tool call")
        tool_name = function.get("name")
        expected_name = {
            "action_chunk_eef": "emit_eef_actions",
            "action_chunk_relative_eef": "emit_relative_eef_actions",
            "action_chunk_delta_eef": "emit_delta_eef_actions",
            "action_chunk_jointpos": "emit_joint_actions",
            "action_chunk_relative_jointpos": "emit_relative_joint_actions",
            "action_chunk_delta_jointpos": "emit_delta_joint_actions",
            "bspline_jointpos": "emit_joint_bspline",
        }[self.output_mode]
        if tool_name != expected_name:
            raise RuntimeError(f"model called {tool_name!r}; expected {expected_name!r}")
        raw_arguments = function.get("arguments")
        if isinstance(raw_arguments, Mapping):
            arguments = dict(raw_arguments)
        else:
            try:
                arguments = json.loads(str(raw_arguments))
            except (TypeError, ValueError) as exc:
                raise RuntimeError("model returned malformed JSON tool arguments") from exc
        if not isinstance(arguments, dict):
            raise RuntimeError("tool arguments must be a JSON object")

        required_keys = (
            {"actions", "note"}
            if self.output_mode in RAW_ACTION_OUTPUT_MODES
            else {"control_points", "steps", "note"}
        )
        missing = required_keys - set(arguments)
        extra = set(arguments) - required_keys
        if missing:
            raise ValueError(f"tool arguments are missing required field(s): {', '.join(sorted(missing))}")
        if extra:
            raise ValueError(f"tool arguments contain unexpected field(s): {', '.join(sorted(extra))}")
        note = arguments["note"]
        if not isinstance(note, str) or not note.strip():
            raise ValueError("tool argument note must be a nonempty string")

        if self.output_mode in RAW_ACTION_OUTPUT_MODES:
            self._validate_action_rows(arguments.get("actions"))

        if self.output_mode == "action_chunk_eef":
            requested = np.asarray(arguments.get("actions"), dtype=np.float32)
            actions = self._action_chunk_eef_actions(arguments, response.current_proprio)
            safety_clamped = requested.shape != actions.shape or not bool(np.allclose(requested, actions))
        elif self.output_mode == "action_chunk_relative_eef":
            decoded = self._relative_eef_actions(arguments, response.current_proprio)
            actions = self._apply_eef_action_safety(decoded, response.current_proprio)
            safety_clamped = not bool(np.allclose(decoded, actions))
        elif self.output_mode == "action_chunk_delta_eef":
            decoded = self._delta_eef_actions(arguments, response.current_proprio)
            actions = self._apply_eef_action_safety(decoded, response.current_proprio)
            safety_clamped = not bool(np.allclose(decoded, actions))
        elif self.output_mode == "action_chunk_jointpos":
            requested = np.asarray(arguments.get("actions"), dtype=np.float32)
            actions = self._action_chunk_jointpos_actions(arguments, response.current_joint_position)
            safety_clamped = requested.shape != actions.shape or not bool(np.allclose(requested, actions))
        elif self.output_mode == "action_chunk_relative_jointpos":
            decoded = self._relative_jointpos_actions(arguments, response.current_joint_position)
            actions = self._apply_action_safety(decoded, response.current_joint_position)
            safety_clamped = not bool(np.allclose(decoded, actions))
        elif self.output_mode == "action_chunk_delta_jointpos":
            decoded = self._delta_jointpos_actions(arguments, response.current_joint_position)
            actions = self._apply_action_safety(decoded, response.current_joint_position)
            safety_clamped = not bool(np.allclose(decoded, actions))
        else:
            decoded = self._bspline_jointpos_actions(arguments, response)
            actions = self._apply_action_safety(decoded, response.current_joint_position)
            safety_clamped = not bool(np.allclose(decoded, actions))
        return actions, arguments, safety_clamped

    def _validate_action_rows(self, actions: Any) -> None:
        """Explain malformed JSON rows before NumPy loses the row context."""
        schema = self._tool_schema()["function"]["parameters"]["properties"]["actions"]
        width = schema["items"]["minItems"]
        if not isinstance(actions, list):
            raise ValueError(f"actions must be a JSON array of rows, each with exactly {width} numbers")
        bad_rows = []
        for index, row in enumerate(actions):
            if not isinstance(row, list):
                bad_rows.append(f"actions[{index}] is {type(row).__name__}, not an array")
            elif len(row) != width:
                bad_rows.append(f"actions[{index}] has {len(row)} values, expected {width}")
            elif any(isinstance(value, bool) or not isinstance(value, (int, float))
                     or not math.isfinite(value) for value in row):
                bad_rows.append(f"actions[{index}] must contain only finite JSON numbers")
        if bad_rows:
            detail = "; ".join(bad_rows[:16])
            if len(bad_rows) > 16:
                detail += f"; {len(bad_rows) - 16} additional invalid rows"
            raise ValueError(detail)
        if not self.min_action_horizon <= len(actions) <= self.max_action_horizon:
            raise ValueError(f"actions has {len(actions)} rows; expected between "
                             f"{self.min_action_horizon} and {self.max_action_horizon}")

    def _output_repair_context(self, response: _RawResponse) -> str:
        """Supply the failed call as bounded data, never infer missing targets."""
        try:
            function = _tool_call(_response_mapping(response.body))["function"]
            arguments = function.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    pass
            failed = json.dumps(arguments, ensure_ascii=True)
        except (KeyError, TypeError, ValueError, RuntimeError):
            failed = "unavailable"
        if len(failed) > 16000:
            failed = failed[:16000] + " [truncated]"
        context = "\nPrevious invalid tool arguments (data to correct):\n" + failed
        if self.output_mode in RAW_ACTION_OUTPUT_MODES:
            schema = self._tool_schema()["function"]["parameters"]["properties"]["actions"]
            context += (f"\nReturn the complete actions array with {self.min_action_horizon} to "
                        f"{self.max_action_horizon} rows. Every row must contain exactly "
                        f"{schema['items']['minItems']} finite numbers; do not omit repeated coordinates "
                        "or use abbreviated rows. Check every row's length before emitting the tool call.")
            if self.output_mode == "action_chunk_eef":
                context += " The eight entries are [x, y, z, qw, qx, qy, qz, gripper]."
        return context

    def _query_output_repair(
        self,
        response: _RawResponse,
        *,
        error: Exception,
        repair_attempt: int,
    ) -> _RawResponse:
        payload = dict(response.request_payload)
        payload["messages"] = [
            *response.request_payload["messages"],
            {
                "role": "user",
                "content": (
                    "Your previous tool call could not be decoded and no robot action was executed. "
                    f"Validation error: {type(error).__name__}: {error}. "
                    "Re-evaluate the unchanged observation and respond again with exactly one valid "
                    "tool call matching the supplied schema. Do not add prose outside the tool call. "
                    f"This is output-repair attempt {repair_attempt}/{self.max_output_retries}."
                    + self._output_repair_context(response)
                ),
            },
        ]
        started = time.perf_counter()
        repaired = self._request_with_api_retries(
            payload, env_id=response.env_id, call_index=response.call_index, stage="output_repair"
        )
        return _RawResponse(
            body=repaired,
            request_payload=payload,
            env_id=response.env_id,
            current_joint_position=response.current_joint_position,
            current_gripper_position=response.current_gripper_position,
            call_index=response.call_index,
            observation_metadata=response.observation_metadata,
            latency_s=time.perf_counter() - started,
            current_proprio=response.current_proprio,
            request_build_s=response.request_build_s,
        )

    def _safe_hold_actions(self, response: _RawResponse) -> np.ndarray:
        if self.output_mode in EEF_OUTPUT_MODES:
            return self._current_eef_action(response.current_proprio)[None, :]
        hold = np.concatenate(
            [
                response.current_joint_position.astype(np.float32, copy=True),
                np.asarray([float(response.current_gripper_position >= 0.5)], dtype=np.float32),
            ]
        )
        # Variable-horizon methods replan after one safe hold instead of
        # spending the maximum horizon on an invalid response.
        return hold[None, :]

    def _current_eef_action(self, proprio: Mapping[str, Any]) -> np.ndarray:
        states = agent_eef_state(self.embodiment_spec, proprio)
        action = np.empty(self.embodiment_spec.action_dim, dtype=np.float32)
        for effector in self.embodiment_spec.effectors:
            state = states[effector.name]
            action[list(effector.position_action_indices)] = state.position_m
            offset = Rotation.from_quat(np.asarray(effector.action_quaternion_offset_wxyz)[[1, 2, 3, 0]])
            semantic = Rotation.from_quat(state.quaternion_wxyz[[1, 2, 3, 0]])
            action_quaternion = (semantic * offset.inv()).as_quat()[[3, 0, 1, 2]]
            action[list(effector.quaternion_action_indices)] = action_quaternion
            action[effector.gripper_action_index] = (
                effector.gripper_closed_action if state.gripper_closed else effector.gripper_open_action
            )
        return action

    def _action_chunk_eef_actions(
        self,
        arguments: Mapping[str, Any],
        current_proprio: Mapping[str, Any],
    ) -> np.ndarray:
        actions = np.asarray(arguments.get("actions"), dtype=np.float64)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2 or actions.shape[1:] != (8,):
            raise ValueError(f"model returned EEF action shape {actions.shape}; expected [N, 8]")
        if not self.min_action_horizon <= len(actions) <= self.max_action_horizon:
            raise ValueError(
                f"model returned {len(actions)} EEF actions; "
                f"expected between {self.min_action_horizon} and {self.max_action_horizon}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("model returned non-finite EEF actions")

        return self._apply_eef_action_safety(actions, current_proprio)

    def _relative_eef_actions(
        self,
        arguments: Mapping[str, Any],
        current_proprio: Mapping[str, Any],
    ) -> np.ndarray:
        """Decode offsets that all share the measured observation anchor."""

        return self._offset_eef_actions(
            arguments,
            current_proprio,
            cumulative=False,
            representation="relative",
        )

    def _delta_eef_actions(
        self,
        arguments: Mapping[str, Any],
        current_proprio: Mapping[str, Any],
    ) -> np.ndarray:
        """Decode offsets sequentially from the preceding decoded pose."""

        return self._offset_eef_actions(
            arguments,
            current_proprio,
            cumulative=True,
            representation="delta",
        )

    def _offset_eef_actions(
        self,
        arguments: Mapping[str, Any],
        current_proprio: Mapping[str, Any],
        *,
        cumulative: bool,
        representation: str,
    ) -> np.ndarray:
        offsets = np.asarray(arguments.get("actions"), dtype=np.float64)
        if offsets.ndim == 1:
            offsets = offsets.reshape(1, -1)
        if offsets.ndim != 2 or offsets.shape[1:] != (7,):
            raise ValueError(
                f"model returned {representation} EEF action shape {offsets.shape}; expected [N, 7]"
            )
        if not self.min_action_horizon <= len(offsets) <= self.max_action_horizon:
            raise ValueError(
                f"model returned {len(offsets)} {representation} EEF actions; "
                f"expected between {self.min_action_horizon} and {self.max_action_horizon}"
            )
        if not np.all(np.isfinite(offsets)):
            raise ValueError(f"model returned non-finite {representation} EEF actions")

        anchor = self._current_eef_action(current_proprio).astype(np.float64)
        absolute = np.empty((len(offsets), self.embodiment_spec.action_dim), dtype=np.float64)
        for effector in self.embodiment_spec.effectors:
            position_indices = list(effector.position_action_indices)
            quaternion_indices = list(effector.quaternion_action_indices)
            reference_position = anchor[position_indices]
            reference_rotation = Rotation.from_quat(anchor[quaternion_indices][[1, 2, 3, 0]])
            for index, row in enumerate(offsets):
                delta_rotation = Rotation.from_rotvec(row[3:6])
                target_position = reference_position + reference_rotation.apply(row[:3])
                target_rotation = reference_rotation * delta_rotation
                absolute[index, position_indices] = target_position
                absolute[index, quaternion_indices] = target_rotation.as_quat()[[3, 0, 1, 2]]
                absolute[index, effector.gripper_action_index] = row[6]
                if cumulative:
                    reference_position = target_position
                    reference_rotation = target_rotation
        return absolute.astype(np.float32)

    def _apply_eef_action_safety(
        self,
        actions: np.ndarray,
        current_proprio: Mapping[str, Any],
    ) -> np.ndarray:
        """Apply controller-facing limits to absolute EEF targets."""

        safe = actions.copy()
        current = self._current_eef_action(current_proprio).astype(np.float64)
        # AI-native currently exposes only the single-arm DROID adapter. Keep
        # the loop embodiment-driven so the safety logic remains unambiguous.
        for effector in self.embodiment_spec.effectors:
            position_indices = list(effector.position_action_indices)
            quaternion_indices = list(effector.quaternion_action_indices)
            previous_position = current[position_indices]
            previous_quaternion = current[quaternion_indices]
            previous_rotation = Rotation.from_quat(previous_quaternion[[1, 2, 3, 0]])
            for index, row in enumerate(safe):
                position = row[position_indices].copy()
                translation = position - previous_position
                distance = float(np.linalg.norm(translation))
                if distance > self.max_eef_translation_delta_m:
                    position = previous_position + translation * (self.max_eef_translation_delta_m / distance)

                quaternion = row[quaternion_indices]
                norm = float(np.linalg.norm(quaternion))
                if norm <= np.finfo(np.float64).eps:
                    raise ValueError(f"EEF action {index} contains a zero quaternion")
                target_rotation = Rotation.from_quat((quaternion / norm)[[1, 2, 3, 0]])
                rotation_vector = (target_rotation * previous_rotation.inv()).as_rotvec()
                rotation_angle = float(np.linalg.norm(rotation_vector))
                if rotation_angle > self.max_eef_rotation_delta_rad:
                    target_rotation = (
                        Rotation.from_rotvec(rotation_vector * (self.max_eef_rotation_delta_rad / rotation_angle))
                        * previous_rotation
                    )
                quaternion = target_rotation.as_quat()[[3, 0, 1, 2]]

                safe[index, position_indices] = position
                safe[index, quaternion_indices] = quaternion
                safe[index, effector.gripper_action_index] = (
                    effector.gripper_closed_action
                    if row[effector.gripper_action_index] >= 0.5
                    else effector.gripper_open_action
                )
                previous_position = position
                previous_rotation = target_rotation
        return safe.astype(np.float32)

    def _relative_jointpos_actions(
        self,
        arguments: Mapping[str, Any],
        current_joints: np.ndarray,
    ) -> np.ndarray:
        """Decode joint waypoints that all share the current observation anchor."""

        return self._offset_jointpos_actions(
            arguments,
            current_joints,
            cumulative=False,
            representation="relative",
        )

    def _delta_jointpos_actions(
        self,
        arguments: Mapping[str, Any],
        current_joints: np.ndarray,
    ) -> np.ndarray:
        """Decode joint offsets cumulatively from the preceding decoded target."""

        return self._offset_jointpos_actions(
            arguments,
            current_joints,
            cumulative=True,
            representation="delta",
        )

    def _offset_jointpos_actions(
        self,
        arguments: Mapping[str, Any],
        current_joints: np.ndarray,
        *,
        cumulative: bool,
        representation: str,
    ) -> np.ndarray:
        offsets = np.asarray(arguments.get("actions"), dtype=np.float32)
        if offsets.ndim == 1:
            offsets = offsets.reshape(1, -1)
        if offsets.ndim != 2 or offsets.shape[1:] != (8,):
            raise ValueError(
                f"model returned {representation} joint-position action shape {offsets.shape}; expected [N, 8]"
            )
        if not self.min_action_horizon <= len(offsets) <= self.max_action_horizon:
            raise ValueError(
                f"model returned {len(offsets)} {representation} joint-position actions; "
                f"expected between {self.min_action_horizon} and {self.max_action_horizon}"
            )
        if not np.all(np.isfinite(offsets)):
            raise ValueError(f"model returned non-finite {representation} joint-position actions")

        absolute = offsets.copy()
        joint_offsets = np.cumsum(offsets[:, :7], axis=0) if cumulative else offsets[:, :7]
        absolute[:, :7] = current_joints.astype(np.float32, copy=False)[None, :] + joint_offsets
        return absolute

    def _action_chunk_jointpos_actions(self, arguments: Mapping[str, Any], current_joints: np.ndarray) -> np.ndarray:
        actions = np.asarray(arguments.get("actions"), dtype=np.float32)
        if actions.ndim == 1:
            actions = actions.reshape(1, -1)
        if actions.ndim != 2 or actions.shape[1:] != (8,):
            raise ValueError(f"model returned joint-position action shape {actions.shape}; expected [N, 8]")
        if not self.min_action_horizon <= len(actions) <= self.max_action_horizon:
            raise ValueError(
                f"model returned {len(actions)} joint-position actions; "
                f"expected between {self.min_action_horizon} and {self.max_action_horizon}"
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("model returned non-finite actions")

        return self._apply_action_safety(actions, current_joints)

    def _bspline_jointpos_actions(self, arguments: Mapping[str, Any], response: _RawResponse) -> np.ndarray:
        control_points = np.asarray(arguments.get("control_points"), dtype=np.float32)
        expected = (self.bspline_control_points, 8)
        if control_points.shape != expected:
            raise ValueError(f"model returned B-spline control-point shape {control_points.shape}; expected {expected}")
        steps = arguments.get("steps")
        minimum_steps = self.bspline_min_steps
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or not minimum_steps <= steps <= self.max_action_horizon
        ):
            raise ValueError(
                f"B-spline steps must be an integer in [{minimum_steps}, {self.max_action_horizon}]"
            )
        current_action = np.concatenate(
            [
                response.current_joint_position.astype(np.float32, copy=True),
                np.asarray([float(response.current_gripper_position >= 0.5)], dtype=np.float32),
            ]
        )
        return decode_anchored_bspline_trajectory(
            current_action,
            control_points,
            action_horizon=steps,
            degree=self.bspline_degree,
        )

    def _apply_action_safety(self, actions: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        safe = actions.copy()
        previous = current_joints.astype(np.float32, copy=True)
        for index in range(len(safe)):
            target = np.clip(
                safe[index, :7],
                previous - self.max_joint_delta,
                previous + self.max_joint_delta,
            )
            target = np.clip(target, PANDA_JOINT_LOWER, PANDA_JOINT_UPPER)
            safe[index, :7] = target
            safe[index, 7] = float(safe[index, 7] >= 0.5)
            previous = target
        return safe

    def _log_inference(
        self,
        *,
        env_id: int,
        call_index: int,
        observation_metadata: Mapping[str, Any],
        request_build_s: float,
        api_latency_s: float,
        status: str,
        model_arguments: Mapping[str, Any] | None = None,
        actions: np.ndarray | None = None,
        safety_clamped: bool | None = None,
        output_attempts: int | None = None,
        output_errors: list[Mapping[str, Any]] | None = None,
        fallback_reason: str | None = None,
        error_stage: str | None = None,
        error: Exception | None = None,
    ) -> None:
        task_usage = self._token_usage.setdefault(env_id, TokenUsageTracker())
        task_usage.record_inference_call()
        call_usage = self._call_token_usage.pop((env_id, call_index), TokenUsageTracker())
        call_usage.record_inference_call()
        provider_models = self._provider_models.pop((env_id, call_index), [])
        record: dict[str, Any] = {
            "episode": self._eval_episode_idx,
            "env_id": env_id,
            "inference_call": call_index,
            "model": self.model,
            "provider_model": provider_models[-1] if provider_models else None,
            "provider_models": provider_models,
            "history_length": self.history_length,
            "history_stride": self.history_stride,
            "history_offsets": list(self.history_offsets) if self.history_offsets is not None else None,
            "output_mode": self.output_mode,
            "action_horizon": (int(len(actions)) if actions is not None else self.max_action_horizon),
            "min_action_horizon": self.min_action_horizon,
            "bspline_min_steps": self.bspline_min_steps,
            "configuration_sha256": self.configuration_sha256,
            "startup_seed": self._startup_seed,
            "max_action_horizon": self.max_action_horizon,
            "request_build_s": round(float(request_build_s), 6),
            "api_latency_s": round(float(api_latency_s), 6),
            "latency_s": round(float(request_build_s + api_latency_s), 6),
            "status": status,
            "api_retry_count": self._api_retry_counts.pop((env_id, call_index), 0),
            "observation": dict(observation_metadata),
            "token_usage": call_usage.summary(),
        }
        if model_arguments is not None:
            record["model_arguments"] = dict(model_arguments)
        if actions is not None:
            record["actions"] = actions
        if safety_clamped is not None:
            record["safety_clamped"] = safety_clamped
        if output_attempts is not None:
            record["output_attempts"] = output_attempts
            record["output_retries"] = max(0, output_attempts - 1)
        if output_errors:
            record["output_errors"] = output_errors
        if fallback_reason is not None:
            record["fallback_reason"] = fallback_reason
        if error_stage is not None:
            record["error_stage"] = error_stage
        if error is not None:
            record["error"] = {"type": type(error).__name__, "message": str(error)}
        self.inference_logger.write(record)

    @staticmethod
    def _response_usage(response: Any) -> Any:
        if isinstance(response, Mapping):
            return response.get("usage")
        return getattr(response, "usage", None)

    def _record_model_request(self, env_id: int, call_index: int, response: Any, *,
                              stage: str = "query", latency_s: float | None = None,
                              error_type: str | None = None) -> None:
        reported_model = response.get("model") if isinstance(response, Mapping) else getattr(response, "model", None)
        if reported_model:
            self._provider_models.setdefault((env_id, call_index), []).append(str(reported_model))
        usage = None if response is None else self._response_usage(response)
        self._token_usage.setdefault(env_id, TokenUsageTracker()).record_model_request(usage)
        self._call_token_usage.setdefault((env_id, call_index), TokenUsageTracker()).record_model_request(usage)
        raw_usage = (usage.model_dump(mode="json", exclude_none=True) if hasattr(usage, "model_dump")
                     else dict(usage) if isinstance(usage, Mapping) else None)
        choices = response.get("choices", []) if isinstance(response, Mapping) else getattr(response, "choices", [])
        self.provider_request_logger.write({
            "episode": self._eval_episode_idx, "env_id": env_id, "inference_call": call_index,
            "model": self.model, "provider_model": reported_model, "stage": stage,
            "max_completion_tokens": self.max_completion_tokens, "reasoning_effort": self.reasoning_effort,
            "api_latency_s": latency_s, "error_type": error_type,
            "finish_reasons": [choice.get("finish_reason") if isinstance(choice, Mapping)
                               else getattr(choice, "finish_reason", None) for choice in (choices or [])],
            # Preserve provider-specific cache writes and reasoning details for pricing
            # and truncation audits without storing prompts, images, or credentials.
            "raw_usage": raw_usage,
        })

    def begin_episode(self, episode_idx: int) -> None:
        super().begin_episode(episode_idx)
        self._token_usage.clear()
        self._call_token_usage.clear()
        self._provider_models.clear()
        self._api_retry_counts.clear()
        self._episode_api_retries.clear()

    def episode_metadata(self, *, env_id: int) -> dict[str, Any]:
        tracker = self._token_usage.get(env_id, TokenUsageTracker())
        return {
            "token_usage": tracker.summary(), "api_retry_count": self._episode_api_retries.get(env_id, 0),
            "startup_seed": self._startup_seed,
            "configuration_sha256": self.configuration_sha256,
        }

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        images = [np.asarray(_resize_for_api(extracted_obs["images"][name], 224)) for name in self.camera_names]
        target_height = min(image.shape[0] for image in images)
        resized = []
        for image in images:
            if image.shape[0] != target_height:
                width = max(1, round(image.shape[1] * target_height / image.shape[0]))
                image = np.asarray(Image.fromarray(image).resize((width, target_height)))
            resized.append(image)
        return np.concatenate(resized, axis=1)

    def reset(self, *, env_id: int | None = None) -> None:
        super().reset(env_id=env_id)
        if env_id is None:
            self._observation_buffers.clear()
            self._interval_proprio_buffers.clear()
            self._control_steps.clear()
            self._inference_calls.clear()
            self._call_token_usage.clear()
            self._provider_models.clear()
        else:
            self._observation_buffers.pop(env_id, None)
            self._interval_proprio_buffers.pop(env_id, None)
            self._control_steps.pop(env_id, None)
            self._inference_calls.pop(env_id, None)
            for key in [key for key in self._call_token_usage if key[0] == env_id]:
                self._call_token_usage.pop(key, None)
            for key in [key for key in self._provider_models if key[0] == env_id]:
                self._provider_models.pop(key, None)

    def close(self) -> None:
        self._stop_requests.set()
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
        if self._owns_client:
            self.client.close()
