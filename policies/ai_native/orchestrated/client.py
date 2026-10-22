"""Sequential proposal/review/execute policy inside RoboLab's shared loop."""

from __future__ import annotations

import copy
import json
import time
import uuid
from pathlib import Path

import numpy as np

from policies.ai_native.agentic.specs import make_droid_spec
from policies.ai_native.common.token_usage import TokenUsageTracker
from policies.ai_native.orchestrated.protocol import validate_review
from policies.ai_native.orchestrated.session import CAMERA_NAMES
from robolab.core.task.instruction import instruction_text
from robolab.core.utils.isaaclab_compat import as_torch
from robolab.eval.base_client import InferenceClient


class OrchestratedEpisodeStopped(RuntimeError):
    """An intentional incomplete episode, rather than a transport/physics error."""

    def __init__(self, reason, message):
        super().__init__(message)
        self.reason = reason


class OrchestratedClient(InferenceClient):
    def __init__(self, *, student, session_factory, execution_adapter, kinematics,
                 output_dir, control_hz=15., max_reviews=None, max_output_retries=2):
        super().__init__()
        if not np.isfinite(control_hz) or control_hz <= 0:
            raise ValueError("control_hz must be positive and finite")
        if max_reviews is not None and (type(max_reviews) is not int or max_reviews < 1):
            raise ValueError("max_reviews must be positive")
        if type(max_output_retries) is not int or max_output_retries < 0:
            raise ValueError("max_output_retries must be nonnegative")
        self.student, self.session_factory = student, session_factory
        self.execution_adapter, self.kinematics = execution_adapter, kinematics
        self.output_dir = Path(output_dir)
        self.control_hz, self.max_reviews = float(control_hz), max_reviews
        self.max_output_retries = max_output_retries
        self.spec = make_droid_spec()
        self._session = None
        self._motion = None
        self._pending = None
        self._active = False
        self._closed = False
        self._failed = False
        self.decisions = []
        self.control_log = []
        self._usage = TokenUsageTracker()
        self._issued = dict(student=0, edit=0, eef=0)
        self._confirmed = dict(student=0, edit=0, eef=0)
        self._step = 0
        self._instruction = None
        self._previous_execution = None
        self._stop_reason = None

    def bind_environment(self, env):
        if int(env.num_envs) != 1:
            raise ValueError("Orchestrated currently requires --num-envs 1")
        self.execution_adapter.bind_environment(env)
        self.kinematics.bind_environment(env)

    def begin_episode(self, episode_idx):
        if self._closed:
            raise RuntimeError("Client is closed")
        self.reset()
        super().begin_episode(episode_idx)
        self.decisions.clear()
        self.control_log.clear()
        self._usage = TokenUsageTracker()
        self._issued = dict(student=0, edit=0, eef=0)
        self._confirmed = dict(student=0, edit=0, eef=0)
        self._step, self._instruction, self._previous_execution = 0, None, None
        self._stop_reason = None
        self._failed, self._active = False, True
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _event(self, event, **payload):
        record = dict(event=event, episode=self._eval_episode_idx, **payload)
        path = self.output_dir / f"orchestrated_events_{self._eval_episode_idx}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, allow_nan=False) + "\n")

    def reset(self, *, env_id=None):
        if env_id not in (None, 0):
            raise ValueError("Only env_id 0 is supported")
        session, self._session = self._session, None
        self._active = False
        self._motion = self._pending = None
        try:
            if session is not None:
                session.close()
        finally:
            self.student.reset()
            # In this single-env backend, resetting env 0 also ends its episode context.
            super().reset()

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            self.reset()
        finally:
            self.student.close()

    def _extract_observation(self, raw_obs, *, env_id=0):
        if env_id != 0:
            raise ValueError("Only env_id 0 is supported")

        def array(value):
            return as_torch(value)[0].detach().cpu().numpy().copy()

        images = {}
        for name in CAMERA_NAMES:
            image = array(raw_obs["image_obs"][name])
            if (image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] not in (3, 4)
                    or min(image.shape[:2]) == 0):
                raise ValueError(f"Invalid RGB image: {name}")
            images[name] = image[..., :3]
        proprio = {}
        for name, size in (("arm_joint_pos", 7), ("gripper_pos", 1), ("ee_pos", 3), ("ee_quat", 4)):
            value = array(raw_obs["proprio_obs"][name])
            if value.shape != (size,) or not np.isfinite(value).all():
                raise ValueError(f"Invalid measured state: {name}")
            proprio[name] = value
        return dict(images=images, proprio=proprio)

    def _pack_request(self, extracted_obs, instruction):
        raise RuntimeError("Use the stateful infer method")

    def _query_server(self, request):
        raise RuntimeError("Use the stateful infer method")

    def _unpack_response(self, response):
        raise RuntimeError("Use the stateful infer method")

    def _acknowledge(self, proprio):
        if self._pending is None:
            return
        self._step += 1
        self._confirmed[self._pending["source"]] += 1
        self._pending["confirmed_by_next_observation"] = True
        self._pending["measured_after"] = {k: v.tolist() for k, v in proprio.items()}
        self._event("control_observed", **self._pending)
        self._pending = None

    def infer(self, obs, instruction, *, env_id=0):
        if not self._active or self._failed:
            raise RuntimeError("Prepare a fresh episode before inference")
        try:
            return self._infer(obs, instruction, env_id=env_id)
        except BaseException as exc:
            self._failed = True
            if isinstance(exc, OrchestratedEpisodeStopped):
                self._stop_reason = exc.reason
                self._event("episode_stopped", reason=exc.reason)
            else:
                self._event("policy_error", error_type=type(exc).__name__, error=str(exc))
            raise

    def _infer(self, obs, instruction, *, env_id):
        observation = self._extract_observation(obs, env_id=env_id)
        proprio = observation["proprio"]
        goal = instruction_text(instruction)
        if self._instruction is None:
            self._instruction = goal
        elif self._instruction != goal:
            raise ValueError("Task instruction cannot change within an episode")
        self._acknowledge(proprio)
        if self._motion is not None and self._motion["sent"] == self._motion["steps"]:
            motion = self._motion
            self._previous_execution = dict(
                source=motion["source"], steps=motion["sent"],
                start_state=motion["start_state"], end_state={k: v.tolist() for k, v in proprio.items()},
                actions=motion["actions"],
            )
            self._event("segment_completed", **self._previous_execution)
            self._motion = None
        if self._motion is None:
            if self.max_reviews is not None and len(self.decisions) >= self.max_reviews:
                raise OrchestratedEpisodeStopped(
                    "review_budget_exhausted", "Orchestrated review budget exhausted; episode is incomplete")
            # Query the whole fresh chunk, never the student's cached single-action infer().
            started = time.monotonic()
            proposal = np.asarray(self.student.propose(obs, goal), dtype=np.float32).copy()
            if proposal.shape != (15, 8) or not np.isfinite(proposal).all():
                raise ValueError("Student must propose finite [15, 8] absolute joint targets")
            proposal[:, 7] = (proposal[:, 7] > 0.5).astype(np.float32)
            preview = self.kinematics.preview(proposal, proprio)
            request_id = uuid.uuid4().hex
            record = dict(request_id=request_id, control_step=self._step, proposal=proposal.tolist(),
                          preview=preview, student_latency_s=time.monotonic() - started, attempts=[])
            self._event("proposal", **record)
            request = dict(request_id=request_id, control_step=self._step, instruction=goal,
                           control_hz=self.control_hz, images=observation["images"],
                           measured_state={k: v.tolist() for k, v in proprio.items()},
                           proposal=dict(horizon=15, **preview),
                           previous_execution=copy.deepcopy(self._previous_execution))
            if self._session is None:
                self._session = self.session_factory(self._eval_episode_idx)
            for attempt in range(self.max_output_retries + 1):
                started = time.monotonic()
                try:
                    name, response = self._session.query(request)
                finally:
                    self._usage.record_inference_call()
                    metadata = self._session.last_metadata
                    self._usage.record_model_request(
                        metadata.get("token_usage"),
                        codex_thread_usage=metadata.get("usage_is_cumulative", True),
                    )
                    self._event("review_usage", request_id=request_id, attempt=attempt,
                                latency_s=time.monotonic() - started, transport=metadata)
                try:
                    plan = validate_review(name, response, request, spec=self.spec, proprio=proprio)
                except (ValueError, TypeError, KeyError) as exc:
                    record["attempts"].append(dict(response=copy.deepcopy(response), error=str(exc)))
                    self._session.record_host_result(dict(status="rejected", executed=False, error=str(exc)))
                    self._event("review_rejected", request_id=request_id, response=response, error=str(exc))
                    if attempt == self.max_output_retries:
                        raise RuntimeError("Review repair budget exhausted; no action executed") from exc
                    request["validation_error"] = str(exc)
                    continue
                break
            record.update(response=copy.deepcopy(response), source=response["mode"], steps=response["steps"])
            self.decisions.append(record)
            self._event("review_accepted", **record)
            if response["mode"] == "stop":
                self._session.record_host_result(dict(status="model_stop", executed=False))
                raise OrchestratedEpisodeStopped(
                    "model_stop", "Orchestrated model requested stop; episode is incomplete")
            self._motion = dict(source=response["mode"], steps=response["steps"], sent=0,
                                proposal=proposal, plan=plan, request_id=request_id, actions=[],
                                start_state={k: v.tolist() for k, v in proprio.items()})
            self._session.record_host_result(dict(status="accepted", executed=False,
                                                 source=response["mode"], steps=response["steps"]))
        motion = self._motion
        if motion["source"] == "student":
            action = motion["proposal"][motion["sent"]].copy()
            diagnostics = {}
        else:
            action, diagnostics = self.execution_adapter.action_for_plan(motion["plan"][motion["sent"]], env_id=0)
            action = np.asarray(action, dtype=np.float32)
        if action.shape != (8,) or not np.isfinite(action).all():
            raise ValueError("Controller must return a finite 8D joint action")
        self._pending = dict(control_step=self._step, request_id=motion["request_id"], source=motion["source"],
                             action=action.tolist(), diagnostics=dict(diagnostics),
                             confirmed_by_next_observation=False)
        self.control_log.append(self._pending)
        self._event("control_issued", **self._pending)
        motion["sent"] += 1
        motion["actions"].append(action.tolist())
        self._issued[motion["source"]] += 1
        return dict(action=action, viz=None)

    def episode_metadata(self, *, env_id):
        if env_id != 0:
            raise ValueError("Only env_id 0 is supported")
        return dict(policy="orchestrated", student="pi05", reviews=len(self.decisions),
                    review_backend=getattr(self, "runtime_settings", {}).get("review_backend", "codex"),
                    issued_control_steps=dict(self._issued), observed_control_steps=dict(self._confirmed),
                    episode_status="incomplete" if self._stop_reason else ("error" if self._failed else "finished"),
                    stop_reason=self._stop_reason,
                    token_usage=self._usage.summary())

    def export_diagnostics(self):
        path = self.output_dir / f"orchestrated_decisions_{self._eval_episode_idx}.json"
        path.write_text(json.dumps(dict(episode=self._eval_episode_idx, decisions=self.decisions,
                                       controls=self.control_log, metadata=self.episode_metadata(env_id=0)),
                                   indent=2, allow_nan=False) + "\n")
