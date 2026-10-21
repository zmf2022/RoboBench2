# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Agent-EEF controller using a persistent Codex App Server thread."""

from __future__ import annotations

import copy
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

import numpy as np

from policies.ai_native.agentic.joint_actions import decode_joint_target
from policies.ai_native.agentic.runtime_options import RuntimeOptions
from policies.ai_native.agentic.setup import configure_environment
from policies.ai_native.common.agent_eef import (
    AgentEEFPlan,
    EmbodimentSpec,
    agent_eef_state,
    build_agent_eef_policy_payload,
    decode_agent_eef_action,
    select_agent_eef_observations,
    validate_agent_eef_prompt_family,
)
from policies.ai_native.common.token_usage import TokenUsageTracker
from robolab.core.task.instruction import InstructionInput, instruction_text
from robolab.eval.base_client import InferenceClient


class DecisionSession(Protocol):
    last_metadata: dict

    def query(self, request: dict) -> tuple[str, dict]: ...

    def close(self) -> None: ...


class EEFExecutionAdapter(Protocol):
    def bind_environment(self, env: Any) -> None: ...

    def action_for_plan(self, plan: AgentEEFPlan, *, env_id: int) -> tuple[np.ndarray, Mapping[str, Any]]: ...

    def hold_action(self, *, env_id: int) -> np.ndarray: ...


@dataclass(frozen=True)
class AgentEEFControlConfig:
    control_hz: float = 15.0
    max_target_steps: int = 5
    max_translation_m: float = 0.05
    max_rotation_rad: float = 0.35
    max_invalid_actions: int = 3
    max_llm_calls: int = 150

    def __post_init__(self) -> None:
        for name in ("control_hz", "max_translation_m", "max_rotation_rad"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, lower in (
            ("max_target_steps", 1),
            ("max_invalid_actions", 0),
            ("max_llm_calls", 1),
        ):
            value = getattr(self, name)
            if type(value) is not int or value < lower:
                raise ValueError(f"{name} must be an integer >= {lower}")
        if self.max_target_steps > 5:
            raise ValueError("max_target_steps must be at most 5")


@dataclass
class _Motion:
    plan: AgentEEFPlan
    record: dict
    started_wall_s: float
    sent: int = 0


@dataclass
class _Observation:
    control_step: int
    images: dict[str, np.ndarray]
    proprio: dict[str, np.ndarray]


@dataclass
class _EnvState:
    instruction: str
    session: DecisionSession | None = None
    decisions: int = 0
    invalid_actions: int = 0
    motion: _Motion | None = None
    feedback: str | None = None
    control_step: int = 0
    interval_observation_count: int = 0
    observations: list[_Observation] = field(default_factory=list)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


class CodexAgentEEFClient(InferenceClient):
    """Execute one fixed, bounded EEF target for one to five controls."""

    def __init__(
        self,
        *,
        spec: EmbodimentSpec,
        session_factory: Callable[[int, int], DecisionSession],
        camera_names: tuple[str, ...],
        control: AgentEEFControlConfig = AgentEEFControlConfig(),
        history_length: int = 1,
        execution_adapter: EEFExecutionAdapter | None = None,
        prompt_family: str = "baseline",
        options: RuntimeOptions | None = None,
        runtime_settings: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(spec, EmbodimentSpec):
            raise ValueError("spec must be an EmbodimentSpec")
        if not callable(session_factory):
            raise ValueError("session_factory must be callable")
        if len(camera_names) != 3 or len(set(camera_names)) != 3:
            raise ValueError("agent-eef requires exactly three unique camera names")
        if not isinstance(control, AgentEEFControlConfig):
            raise ValueError("control must be AgentEEFControlConfig")
        if type(history_length) is not int or history_length < 1:
            raise ValueError("history_length must be a positive integer")
        validate_agent_eef_prompt_family(spec, prompt_family)
        self.options = options
        self.runtime_settings = dict(runtime_settings or {})
        self.prompt_family = prompt_family
        self.spec = spec
        self._session_factory = session_factory
        self.camera_names = tuple(camera_names)
        self.control = control
        self.history_length = history_length
        self.execution_adapter = execution_adapter
        self._states: dict[int, _EnvState] = {}
        self._episode_started = False
        self._closed = False
        self.decision_log: list[dict] = []
        self._token_usage: dict[int, TokenUsageTracker] = {}

    # Agent-EEF owns a variable-duration semantic control loop in ``infer``.
    # These hooks satisfy InferenceClient's transport interface and remain
    # useful to callers that want the individual preparation stages.
    def _extract_observation(self, raw_obs, *, env_id: int = 0) -> dict:
        return {"images": self._images(raw_obs, env_id), "proprioception": self._proprioception(raw_obs, env_id)}

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        observation = _Observation(
            control_step=0,
            images=extracted_obs["images"],
            proprio=extracted_obs["proprioception"],
        )
        return {
            "policy": self._policy_payload(instruction, [observation], interval_observation_count=1),
            "observations": [{"control_step": 0, "images": extracted_obs["images"]}],
        }

    def _policy_payload(
        self,
        instruction: str,
        observations: list[_Observation],
        *,
        interval_observation_count: int,
        feedback: str | None = None,
    ) -> dict:
        payload = build_agent_eef_policy_payload(
            self.spec,
            instruction,
            [(observation.control_step, observation.proprio) for observation in observations],
            history_length=self.history_length,
            prompt_family=self.prompt_family,
            camera_names=self.camera_names,
            interval_observation_count=interval_observation_count,
            control_hz=self.control.control_hz,
            max_target_steps=self.control.max_target_steps,
            max_translation_m=self.control.max_translation_m,
            max_rotation_rad=self.control.max_rotation_rad,
            feedback=feedback,
        )

        if self.options and self.options.joint_state and self.spec.name == "droid":
            payload["measured_state"]["arm_joint_pos"] = observations[-1].proprio["arm_joint_pos"].tolist()
            for entry, observation in zip(payload.get("measured_state_history", []), observations):
                entry["state"]["arm_joint_pos"] = observation.proprio["arm_joint_pos"].tolist()
            payload["action_space"] = self.options.action_space
        return payload

    def _query_server(self, request: dict):
        raise RuntimeError("Agent-EEF queries require episode state; call infer")

    def _unpack_response(self, response):
        if not isinstance(response, AgentEEFPlan):
            raise ValueError("Agent-EEF response must be an AgentEEFPlan")
        return response.actions

    def begin_episode(self, episode_idx: int) -> None:
        if self._closed:
            raise RuntimeError("Client is closed")
        if type(episode_idx) is not int or episode_idx < 0:
            raise ValueError("episode_idx must be a nonnegative integer")
        self.reset()
        super().begin_episode(episode_idx)
        self.decision_log.clear()
        self._token_usage.clear()
        self._episode_started = True

    def reset(self, *, env_id: int | None = None) -> None:
        selected = list(self._states) if env_id is None else [env_id]
        first_error: Exception | None = None
        for eid in selected:
            state = self._states.pop(eid, None)
            if state is not None:
                self._finish_motion(
                    state,
                    completed=state.motion is not None and state.motion.sent >= len(state.motion.plan.actions),
                    reason="episode_reset",
                )
            if state is not None and state.session is not None:
                try:
                    state.session.close()
                except Exception as exc:
                    first_error = first_error or exc
        super().reset(env_id=env_id)
        if env_id is None:
            self._episode_started = False
        if first_error is not None:
            raise first_error

    def close(self) -> None:
        self._closed = True
        self.reset()

    def configure_environment(self, env_cfg) -> None:
        """Validate task geometry and save prompt artifacts before inference."""
        configure_environment(self, env_cfg)

    def bind_environment(self, env: Any) -> None:
        if self.execution_adapter is not None:
            self.execution_adapter.bind_environment(env)

    def _proprioception(self, raw_obs, env_id: int) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        key_sizes: dict[str, int] = {}
        for effector in self.spec.effectors:
            for key, size in (
                (effector.position_key, 3),
                (effector.quaternion_key, 4),
                (effector.gripper_key, 1),
            ):
                previous = key_sizes.setdefault(key, size)
                if previous != size:
                    raise ValueError(f"proprioception key {key!r} has conflicting dimensions")
        if self.options and self.options.joint_state and self.spec.name == "droid":
            key_sizes["arm_joint_pos"] = 7
        for key, size in key_sizes.items():
            try:
                value = np.asarray(self._to_numpy(raw_obs["proprio_obs"][key], env_id), dtype=np.float64)
            except (KeyError, IndexError, TypeError, ValueError) as exc:
                raise ValueError(f"missing or invalid proprioception term: {key}") from exc
            if value.shape != (size,) or not np.all(np.isfinite(value)):
                raise ValueError(f"proprioception term {key} must contain {size} finite values")
            result[key] = value.copy()
        return result

    def _images(self, raw_obs, env_id: int) -> dict[str, np.ndarray]:
        result = {}
        for name in self.camera_names:
            try:
                image = np.asarray(self._to_numpy(raw_obs["image_obs"][name], env_id))
            except (KeyError, IndexError, TypeError) as exc:
                raise ValueError(f"missing camera observation: {name}") from exc
            if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (3, 4) or min(image.shape[:2]) == 0:
                raise ValueError(f"camera {name} must be a nonempty uint8 RGB/RGBA image")
            result[name] = np.array(image[..., :3], copy=True)
        return result

    def _hold_action(
        self,
        proprio: dict[str, np.ndarray],
        *,
        env_id: int,
    ) -> np.ndarray:
        if self.execution_adapter is not None:
            return np.asarray(self.execution_adapter.hold_action(env_id=env_id), dtype=np.float32)
        states = agent_eef_state(self.spec, proprio)
        arguments: dict[str, Any] = {
            "steps": 1,
            "reason": "hold current measured pose",
        }
        targets = {
            name: {
                "position": state.position_m.tolist(),
                "quaternion_wxyz": state.quaternion_wxyz.tolist(),
                "gripper_closed": state.gripper_closed,
            }
            for name, state in states.items()
        }
        if len(states) == 1:
            arguments["target"] = next(iter(targets.values()))
        else:
            arguments["targets"] = targets
        return self._decode("execute_eef_target", arguments, proprio).actions[0].copy()

    def _capture_observation(self, obs, state: _EnvState, env_id: int) -> _Observation:
        observation = _Observation(
            control_step=state.control_step,
            images=self._images(obs, env_id),
            proprio=self._proprioception(obs, env_id),
        )
        state.control_step += 1
        state.interval_observation_count += 1
        state.observations.append(observation)
        window = max(self.options.observation_offsets) + 1 if self.options and self.options.observation_offsets else self.history_length
        if len(state.observations) > window:
            del state.observations[:-window]
        return observation

    def _request(self, instruction: str, state: _EnvState) -> tuple[dict, dict]:
        if self.options and self.options.observation_offsets:
            newest = state.observations[-1].control_step
            by_step = {observation.control_step: observation for observation in state.observations}
            observations = [by_step[newest - offset] for offset in self.options.observation_offsets
                            if newest - offset in by_step]
        else:
            observations = select_agent_eef_observations(
                state.observations, history_length=self.history_length,
            )
        latest_proprio = observations[-1].proprio
        request = {
            "policy": self._policy_payload(
                instruction,
                observations,
                interval_observation_count=max(state.interval_observation_count, len(observations)),
                feedback=state.feedback,
            ),
            "observations": [
                {"control_step": observation.control_step, "images": observation.images} for observation in observations
            ],
        }
        if self.options and self.options.observation_offsets:
            # The upstream builder assumes all images are from the last interval.
            # Sampled history spans intervals, so report the actual new-step count.
            request['policy']['input']['interval_observation_count'] = state.interval_observation_count
            request['policy']['input']['observation_offsets'] = list(self.options.observation_offsets)
        state.feedback = None
        if self.options is None or self.options.observation_offsets is None:
            state.observations.clear()
        state.interval_observation_count = 0
        if self.options and self.options.image_demonstration:
            # Demonstration supplies the task; preserve live motion feedback.
            request['policy'].pop('goal', None)
        return request, latest_proprio

    def _decode(
        self,
        name: str,
        arguments: dict,
        proprio: dict[str, np.ndarray],
    ) -> AgentEEFPlan:
        if self.options and self.options.action_space == "joint":
            return decode_joint_target(self.spec, name, arguments, proprio, self.control.max_target_steps)
        return decode_agent_eef_action(
            self.spec,
            name,
            arguments,
            proprio,
            control_hz=self.control.control_hz,
            max_target_steps=self.control.max_target_steps,
            max_translation_m=self.control.max_translation_m,
            max_rotation_rad=self.control.max_rotation_rad,
        )

    def _motion_action(self, state: _EnvState, motion: _Motion, *, env_id: int, proprio: dict) -> dict:
        try:
            if self.execution_adapter is None:
                action = motion.plan.actions[motion.sent].copy()
                diagnostics: Mapping[str, Any] = {"controller": "native_absolute_ik"}
            else:
                action, diagnostics = self.execution_adapter.action_for_plan(motion.plan, env_id=env_id)
                action = np.asarray(action, dtype=np.float32)
            if action.shape != (self.spec.action_dim,) or not np.all(np.isfinite(action)):
                raise ValueError(f"execution adapter must return a finite ({self.spec.action_dim},) action")
        except ValueError as exc:
            motion.record["status"] = "execution_rejected"
            motion.record["execution_error"] = str(exc)
            self._finish_motion(state, completed=False, reason="local_ik_rejected")
            state.motion = None
            state.feedback = f"The fixed EEF target was not executed because local IK rejected it: {exc}"
            return {"action": self._hold_action(proprio, env_id=env_id), "viz": None}
        motion.sent += 1
        motion.record.setdefault("actions", []).append(action.tolist())
        motion.record.setdefault("ik_diagnostics", []).append(copy.deepcopy(dict(diagnostics)))
        recorder = getattr(state.session, "record_host_result", None)
        if callable(recorder):
            recorder(
                {
                    "event": "control_step",
                    "chunk_step": motion.sent,
                    "chunk_length": len(motion.plan.actions),
                    "executed_action": action.tolist(),
                    **({"ik_diagnostics": dict(diagnostics)} if self.options is None or self.options.motion_feedback else {}),
                }
            )
        return {"action": action, "viz": None}

    def _finish_motion(self, state: _EnvState, *, completed: bool, reason: str) -> None:
        motion = state.motion
        if motion is None:
            return
        execution = {
            "execution_completed": completed,
            "execution_end_reason": reason,
            "execution_control_steps": motion.sent,
            "execution_sim_s": motion.sent / self.control.control_hz,
            "execution_wall_s": time.perf_counter() - motion.started_wall_s,
        }
        motion.record.setdefault("timing", {}).update(execution)
        if completed and state.observations and "joint_target" not in motion.plan.metadata:
            measured = agent_eef_state(self.spec, state.observations[-1].proprio)
            tracking: dict[str, dict[str, float]] = {}
            for name, target in motion.plan.commanded_state.items():
                actual = measured[name]
                quaternion_dot = float(abs(np.dot(target.quaternion_wxyz, actual.quaternion_wxyz)))
                quaternion_dot = min(1.0, max(-1.0, quaternion_dot))
                tracking[name] = {
                    "translation_error_m": float(np.linalg.norm(target.position_m - actual.position_m)),
                    "rotation_error_rad": float(2.0 * math.acos(quaternion_dot)),
                    "measured_gripper_closed": actual.gripper_closed,
                    "commanded_gripper_closed": target.gripper_closed,
                }
            motion.record["post_chunk_tracking"] = tracking
            state.feedback = (
                (motion.plan.feedback + " " if motion.plan.feedback else "")
                + "Post-chunk measured tracking: "
                + json.dumps(tracking, allow_nan=False, separators=(",", ":"))
            )
        if completed and state.observations and "joint_target" in motion.plan.metadata:
            measured = state.observations[-1].proprio["arm_joint_pos"]
            target = np.asarray(motion.plan.metadata["joint_target"][:7])
            tracking = {"joint_error_rad": (target - measured).tolist()}
            motion.record["post_chunk_tracking"] = tracking
            state.feedback = "Post-segment measured joint tracking: " + json.dumps(tracking)
        if completed and self.options is not None and not self.options.motion_feedback:
            state.feedback = None
        recorder = getattr(state.session, "record_host_result", None)
        if callable(recorder):
            try:
                recorder({"event": "execution_timing", **execution})
            except Exception as exc:
                motion.record["execution_timing_error"] = type(exc).__name__

    def infer(self, obs, instruction: InstructionInput, *, env_id: int = 0) -> dict:
        instruction = (
            "Complete the task shown in the demonstration."
            if self.options and self.options.image_demonstration
            else instruction_text(instruction)
        )
        if not self._episode_started:
            raise RuntimeError("Call begin_episode before inference")
        if type(env_id) is not int or env_id < 0:
            raise ValueError("env_id must be a nonnegative integer")
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be nonempty")
        state = self._states.setdefault(env_id, _EnvState(instruction=instruction))
        if state.instruction != instruction:
            raise ValueError("instruction cannot change within an episode")
        observation = self._capture_observation(obs, state, env_id)
        proprio = observation.proprio
        if state.motion is not None:
            if state.motion.sent < len(state.motion.plan.actions):
                return self._motion_action(state, state.motion, env_id=env_id, proprio=proprio)
            self._finish_motion(state, completed=True, reason="target_steps_complete")
            state.motion = None
        if state.decisions >= self.control.max_llm_calls:
            return {"action": self._hold_action(proprio, env_id=env_id), "viz": None}
        request, proprio = self._request(instruction, state)
        if state.session is None:
            state.session = self._session_factory(env_id, self._eval_episode_idx)
        while True:
            inference_started = time.perf_counter()
            validation_decode_s = 0.0
            record = {
                "episode": self._eval_episode_idx,
                "env_id": env_id,
                "decision": state.decisions,
                "agent_style": "agent-eef",
                "history_length": self.history_length,
                "output_mode": "agent-joint" if self.options and self.options.action_space == "joint" else "agent-eef",
                "observation": copy.deepcopy(request["policy"]["input"]),
                "status": "transport_error",
            }
            self.decision_log.append(record)
            try:
                name, arguments = state.session.query(request)
                record["tool"] = name
                record["arguments"] = copy.deepcopy(arguments)
                validation_started = time.perf_counter()
                try:
                    plan = self._decode(name, arguments, proprio)
                finally:
                    validation_decode_s = time.perf_counter() - validation_started
                if plan.status == "refused":
                    raise ValueError(plan.feedback or "the requested operation was refused")
            except ValueError as exc:
                state.invalid_actions += 1
                hold_action = self._hold_action(proprio, env_id=env_id)
                record.update(
                    status="invalid_action",
                    error=str(exc),
                    invalid_actions=state.invalid_actions,
                    action_horizon=1,
                    actions=[hold_action.tolist()],
                    safety_rejected=True,
                )
                recorder = getattr(state.session, "record_host_result", None)
                if callable(recorder):
                    recorder(
                        {
                            "decision": state.decisions,
                            "status": "rejected",
                            "executed": False,
                            "error": str(exc),
                        }
                    )
                if state.invalid_actions >= self.control.max_invalid_actions:
                    state.invalid_actions = 0
                    state.feedback = (
                        f"The previous action was rejected before simulation: {exc}. "
                        "Generate one corrected bounded EEF target from the new observation."
                    )
                    return {"action": hold_action, "viz": None}
                request["policy"]["feedback"] = (
                    f"error: the previous tool call was not executed: {exc}. "
                    "Call exactly one valid tool for this unchanged observation."
                )
                continue
            except Exception as exc:
                record["error_type"] = type(exc).__name__
                raise
            finally:
                metadata = getattr(state.session, "last_metadata", {})
                usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
                task_usage = self._token_usage.setdefault(env_id, TokenUsageTracker())
                task_usage.record_inference_call()
                # A sampled-context query creates a new thread. Its total must not
                # be subtracted from the preceding thread's usage.
                fresh_thread = self.options is not None and not self.options.persistent_history
                if fresh_thread and isinstance(usage, dict):
                    usage = usage.get("total")
                charged_usage = task_usage.record_model_request(usage, codex_thread_usage=not fresh_thread)
                call_usage = TokenUsageTracker()
                call_usage.record_inference_call()
                call_usage.record_model_request(charged_usage)
                record["token_usage"] = call_usage.summary()
                if isinstance(metadata, dict):
                    record["transport"] = copy.deepcopy(metadata)
                    elapsed = metadata.get("elapsed_time_seconds", 0.0)
                    if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
                        record["latency_s"] = float(elapsed)
                    transport_timing = metadata.get("timing", {})
                    if isinstance(transport_timing, dict):
                        record["timing"] = copy.deepcopy(transport_timing)
                record.setdefault("timing", {}).update(
                    {
                        "validation_decode_s": validation_decode_s,
                        "inference_total_s": time.perf_counter() - inference_started,
                    }
                )
            state.invalid_actions = 0
            break

        state.decisions += 1
        record.update(
            status=plan.status,
            note=plan.note,
            motion=copy.deepcopy(dict(plan.metadata)),
            model_arguments=copy.deepcopy(arguments),
            action_horizon=len(plan.actions),
            execution_scheduled_steps=len(plan.actions),
            execution_scheduled_sim_s=len(plan.actions) / self.control.control_hz,
            model_target_actions=plan.actions.tolist(),
            actions=[],
            safety_rejected=False,
        )
        recorder = getattr(state.session, "record_host_result", None)
        if callable(recorder):
            recorder(
                {
                    "decision": state.decisions,
                    "status": plan.status,
                    "validated": True,
                    "execution_scheduled": True,
                    "action_horizon": len(plan.actions),
                    "fixed_target": copy.deepcopy(plan.metadata.get("fixed_target")),
                }
            )
        if plan.feedback is not None:
            state.feedback = plan.feedback
        state.motion = _Motion(plan=plan, record=record, started_wall_s=time.perf_counter())
        return self._motion_action(state, state.motion, env_id=env_id, proprio=proprio)

    def export_diagnostics(self, scene_output_dir: str, *, run_idx: int) -> None:
        if not self.decision_log:
            return
        path = os.path.join(scene_output_dir, f"codex_agent_eef_decisions_{run_idx}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"run": run_idx, "decisions": self.decision_log}, handle, indent=2, default=_jsonable)

    def episode_metadata(self, *, env_id: int) -> dict:
        tracker = self._token_usage.get(env_id, TokenUsageTracker())
        return {"token_usage": tracker.summary()}


__all__ = ["AgentEEFControlConfig", "CodexAgentEEFClient", "DecisionSession"]
