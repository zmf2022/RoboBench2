# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared bounded EEF-target contract for the ``agent-eef`` policy style.

The model emits one complete absolute EEF target and chooses whether that
fixed target is applied for one to five controls.  No trajectory interpolation
or semantic motion expansion happens here.  Embodiment specs map full EEF
targets into the simulator's native action layout.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence, TypeVar

import numpy as np
from scipy.spatial.transform import Rotation

from policies.ai_native.common.prompts import (
    DROID_CAMERA_GEOMETRY_PROMPT,
    DROID_GRIPPER_GEOMETRY_PROMPT,
    eef_frame_prompt,
    robot_control_intro,
)

AGENT_EEF_PROMPT_FAMILIES = ("baseline", "gripper_camera_geometry_prompt")


def validate_agent_eef_prompt_family(spec: EmbodimentSpec, prompt_family: str) -> None:
    if prompt_family not in AGENT_EEF_PROMPT_FAMILIES:
        raise ValueError(f"Unknown Agent-EEF prompt family: {prompt_family}")
    if prompt_family != "baseline" and spec.name != "droid":
        raise ValueError("gripper_camera_geometry_prompt requires DROID")


POSITION_FIELDS = ("x", "y", "z")
ROTATION_FIELDS = ("roll", "pitch", "yaw")
MOVE_FIELDS = POSITION_FIELDS + ROTATION_FIELDS
RPY_CONVENTION = "extrinsic xyz fixed-axis roll/pitch/yaw, degrees"
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_ObservationT = TypeVar("_ObservationT")


def _finite_vector(value: Any, size: int, name: str) -> np.ndarray:
    try:
        result = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain {size} finite numbers") from exc
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain {size} finite numbers")
    return result


def _unit_quaternion(value: Any, name: str) -> np.ndarray:
    result = _finite_vector(value, 4, name)
    norm = math.hypot(*result)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must be a nonzero quaternion")
    return result / norm


def _rotation(quaternion_wxyz: np.ndarray) -> Rotation:
    return Rotation.from_quat(quaternion_wxyz[[1, 2, 3, 0]])


def _quaternion(rotation: Rotation) -> np.ndarray:
    return rotation.as_quat()[[3, 0, 1, 2]]


def _rpy_degrees(quaternion_wxyz: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return _rotation(quaternion_wxyz).as_euler("xyz", degrees=True)


def _wrap_degrees(value: float) -> float:
    """Match the reference convention: angles live in ``(-180, 180]``."""

    wrapped = (float(value) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


@dataclass(frozen=True)
class EffectorSpec:
    """Observation and action mapping for one independently rooted EEF.

    ``action_quaternion_offset_wxyz`` maps the semantic EEF orientation into
    the action controller's body frame as ``eef * inverse(offset)``. Identity
    is correct when the observed EEF and absolute-IK body frames coincide.
    Gripper observation polarity and emitted scalar values are also supplied
    here rather than assumed by the transport-independent controller.
    """

    name: str
    position_key: str
    quaternion_key: str
    gripper_key: str
    position_action_indices: tuple[int, int, int]
    quaternion_action_indices: tuple[int, int, int, int]
    gripper_action_index: int
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None
    action_quaternion_offset_wxyz: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    root_frame: str = "robot-root"
    gripper_closed_threshold: float = 0.5
    gripper_closed_above: bool = True
    gripper_open_action: float = 0.0
    gripper_closed_action: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not _NAME_RE.fullmatch(self.name):
            raise ValueError("effector name must match [A-Za-z][A-Za-z0-9_]*")
        for field_name in ("position_key", "quaternion_key", "gripper_key", "root_frame"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{field_name} must be a nonempty string")
        position_indices = tuple(self.position_action_indices)
        quaternion_indices = tuple(self.quaternion_action_indices)
        if len(position_indices) != 3 or len(quaternion_indices) != 4:
            raise ValueError("pose action indices must contain three position and four quaternion indices")
        indices = position_indices + quaternion_indices + (self.gripper_action_index,)
        if any(type(index) is not int or index < 0 for index in indices) or len(set(indices)) != 8:
            raise ValueError("effector action indices must be eight unique nonnegative integers")
        if (self.workspace_min is None) != (self.workspace_max is None):
            raise ValueError("workspace_min and workspace_max must both be provided or both be None")
        if self.workspace_min is not None:
            lower = _finite_vector(self.workspace_min, 3, "workspace_min")
            upper = _finite_vector(self.workspace_max, 3, "workspace_max")
            if np.any(lower >= upper):
                raise ValueError("workspace_min must be strictly below workspace_max")
            object.__setattr__(self, "workspace_min", tuple(lower.tolist()))
            object.__setattr__(self, "workspace_max", tuple(upper.tolist()))
        offset = _unit_quaternion(self.action_quaternion_offset_wxyz, "action_quaternion_offset_wxyz")
        if type(self.gripper_closed_above) is not bool:
            raise ValueError("gripper_closed_above must be bool")
        for field_name in ("gripper_closed_threshold", "gripper_open_action", "gripper_closed_action"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
                raise ValueError(f"{field_name} must be a finite number")
            object.__setattr__(self, field_name, float(value))
        if math.isclose(self.gripper_open_action, self.gripper_closed_action):
            raise ValueError("gripper open and closed action values must differ")
        object.__setattr__(self, "position_action_indices", position_indices)
        object.__setattr__(self, "quaternion_action_indices", quaternion_indices)
        object.__setattr__(self, "action_quaternion_offset_wxyz", tuple(offset.tolist()))


@dataclass(frozen=True)
class EmbodimentSpec:
    """Complete absolute-IK action layout for an arbitrary embodiment."""

    name: str
    action_dim: int
    effectors: tuple[EffectorSpec, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("embodiment name must be nonempty")
        if type(self.action_dim) is not int or self.action_dim <= 0:
            raise ValueError("action_dim must be a positive integer")
        effectors = tuple(self.effectors)
        if not effectors or len({item.name for item in effectors}) != len(effectors):
            raise ValueError("effectors must be nonempty and uniquely named")
        used: list[int] = []
        for item in effectors:
            if not isinstance(item, EffectorSpec):
                raise ValueError("effectors must contain EffectorSpec values")
            used.extend(item.position_action_indices)
            used.extend(item.quaternion_action_indices)
            used.append(item.gripper_action_index)
        if len(set(used)) != len(used):
            raise ValueError("effector action layouts overlap")
        if set(used) != set(range(self.action_dim)):
            raise ValueError("effectors must describe every action dimension exactly once")
        object.__setattr__(self, "effectors", effectors)


@dataclass(frozen=True)
class EffectorState:
    position_m: np.ndarray
    quaternion_wxyz: np.ndarray
    gripper_closed: bool

    def __post_init__(self) -> None:
        position = _finite_vector(self.position_m, 3, "position_m")
        quaternion = _unit_quaternion(self.quaternion_wxyz, "quaternion_wxyz")
        if type(self.gripper_closed) is not bool:
            raise ValueError("gripper_closed must be bool")
        position.setflags(write=False)
        quaternion.setflags(write=False)
        object.__setattr__(self, "position_m", position)
        object.__setattr__(self, "quaternion_wxyz", quaternion)


@dataclass(frozen=True)
class AgentEEFPlan:
    """Validated result of one model tool call."""

    actions: np.ndarray
    status: str
    tool_name: str
    note: str
    feedback: str | None
    metadata: Mapping[str, Any]
    commanded_state: Mapping[str, EffectorState]

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.float32)
        if actions.ndim != 2 or not np.all(np.isfinite(actions)):
            raise ValueError("plan actions must be a finite 2D array")
        actions.setflags(write=False)
        object.__setattr__(self, "actions", actions)
        expected = set(self.commanded_state)
        if not expected or any(not isinstance(state, EffectorState) for state in self.commanded_state.values()):
            raise ValueError("commanded_state must contain EffectorState values")


def _state_map(spec: EmbodimentSpec, proprio: Mapping[str, Any]) -> dict[str, EffectorState]:
    if not isinstance(proprio, Mapping):
        raise ValueError("proprioception must be a mapping")
    result: dict[str, EffectorState] = {}
    for effector in spec.effectors:
        try:
            raw_gripper = np.asarray(proprio[effector.gripper_key], dtype=np.float64)
            gripper = float(raw_gripper.reshape(-1)[0])
            position = proprio[effector.position_key]
            quaternion = proprio[effector.quaternion_key]
        except (KeyError, IndexError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"missing or invalid proprioception for {effector.name}") from exc
        if not math.isfinite(gripper):
            raise ValueError(f"invalid gripper proprioception for {effector.name}")
        closed = gripper >= effector.gripper_closed_threshold
        if not effector.gripper_closed_above:
            closed = gripper <= effector.gripper_closed_threshold
        result[effector.name] = EffectorState(position, quaternion, bool(closed))
    return result


def agent_eef_state(spec: EmbodimentSpec, proprio: Mapping[str, Any]) -> dict[str, EffectorState]:
    """Extract the measured EEF state used to initialize commanded state."""

    return _state_map(spec, proprio)


def _anchor_state(
    spec: EmbodimentSpec,
    measured: Mapping[str, EffectorState],
    commanded: Mapping[str, EffectorState] | None,
) -> dict[str, EffectorState]:
    if commanded is None:
        return dict(measured)
    if set(commanded) != {effector.name for effector in spec.effectors}:
        raise ValueError("commanded_state must contain every configured effector exactly once")
    if any(not isinstance(state, EffectorState) for state in commanded.values()):
        raise ValueError("commanded_state must contain EffectorState values")
    return dict(commanded)


def format_agent_eef_state(
    spec: EmbodimentSpec, proprio: Mapping[str, Any], *, prompt_family: str = "baseline",
) -> dict[str, Any]:
    """Return the non-privileged numeric state placed in a model prompt."""

    validate_agent_eef_prompt_family(spec, prompt_family)
    states = _state_map(spec, proprio)
    result = {
        "embodiment": spec.name,
        "orientation_convention": RPY_CONVENTION,
        "effectors": {
            effector.name: {
                "root_frame": effector.root_frame,
                "position_m": states[effector.name].position_m.round(6).tolist(),
                "roll_pitch_yaw_deg": _rpy_degrees(states[effector.name].quaternion_wxyz).round(4).tolist(),
                "gripper": "closed" if states[effector.name].gripper_closed else "open",
                **(
                    {
                        "workspace_min_m": list(effector.workspace_min),
                        "workspace_max_m": list(effector.workspace_max),
                    }
                    if effector.workspace_min is not None else {}
                ),
            }
            for effector in spec.effectors
        },
    }

    if prompt_family == "gripper_camera_geometry_prompt":
        effector = spec.effectors[0]
        state = states[effector.name]
        result["geometry_state"] = {
            "ee_pos": state.position_m.tolist(),
            "ee_quat": state.quaternion_wxyz.tolist(),
            "gripper_pos": [float(np.asarray(proprio[effector.gripper_key]).reshape(-1)[0])],
        }
    return result



def _eef_target_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "position": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
                "description": "Absolute EEF position [x, y, z] in metres in its fixed robot-root frame.",
            },
            "quaternion_wxyz": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
                "description": "Absolute unit EEF orientation quaternion [w, x, y, z].",
            },
            "gripper_closed": {"type": "boolean"},
        },
        "required": ["position", "quaternion_wxyz", "gripper_closed"],
        "additionalProperties": False,
    }


def build_agent_eef_tools(spec: EmbodimentSpec) -> list[dict[str, Any]]:
    """Generate the single bounded EEF-target tool for an embodiment."""

    if len(spec.effectors) == 1:
        target_property: dict[str, Any] = _eef_target_schema()
        target_name = "target"
        target_description = f"Complete absolute target for {spec.effectors[0].name}."
    else:
        target_property = {
            "type": "object",
            "properties": {effector.name: _eef_target_schema() for effector in spec.effectors},
            "required": [effector.name for effector in spec.effectors],
            "additionalProperties": False,
        }
        target_name = "targets"
        target_description = "Complete absolute target for every named end effector."
    target_property = dict(target_property)
    target_property["description"] = target_description
    return [
        {
            "type": "function",
            "function": {
                "name": "execute_eef_target",
                "description": (
                    "Apply one fixed absolute EEF target for 1 to 5 simulator control steps. "
                    "The host replans local IK from measured robot state before every step."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        target_name: target_property,
                        "steps": {"type": "integer", "minimum": 1, "maximum": 5},
                        "reason": {
                            "type": "string",
                            "description": "Concise visible evidence and purpose of this target.",
                        },
                    },
                    "required": [target_name, "steps", "reason"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def _hold_action(spec: EmbodimentSpec, states: Mapping[str, EffectorState]) -> np.ndarray:
    action = np.empty(spec.action_dim, dtype=np.float32)
    for effector in spec.effectors:
        state = states[effector.name]
        action[list(effector.position_action_indices)] = state.position_m
        offset_inverse = _rotation(np.asarray(effector.action_quaternion_offset_wxyz)).inv()
        action[list(effector.quaternion_action_indices)] = _quaternion(
            _rotation(state.quaternion_wxyz) * offset_inverse
        )
        action[effector.gripper_action_index] = (
            effector.gripper_closed_action if state.gripper_closed else effector.gripper_open_action
        )
    return action


def _note_and_keys(arguments: Mapping[str, Any], allowed: set[str]) -> tuple[str, list[str]]:
    if not isinstance(arguments, Mapping):
        raise ValueError("tool arguments must be an object")
    unknown = set(arguments) - allowed
    if unknown:
        raise ValueError(f"unexpected tool argument(s): {', '.join(sorted(unknown))}")
    note = arguments.get("note")
    if not isinstance(note, str) or not note.strip():
        raise ValueError("note must be a nonempty string")
    semantic = [key for key in arguments if key != "note"]
    if not semantic:
        raise ValueError("tool call must specify at least one target")
    return note.strip(), semantic


def _positive(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)) or float(value) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def agent_eef_policy_instructions(
    spec: EmbodimentSpec,
    *,
    prompt_family: str = "baseline",
    control_hz: float = 15.0,
    max_target_steps: int = 5,
    max_translation_m: float = 0.05,
    max_rotation_rad: float = 0.35,
) -> str:
    """Build the transport-independent prompt contract for this policy style."""

    if not isinstance(spec, EmbodimentSpec):
        raise ValueError("spec must be an EmbodimentSpec")
    validate_agent_eef_prompt_family(spec, prompt_family)
    control_hz = _positive(control_hz, "control_hz")
    max_translation_m = _positive(max_translation_m, "max_translation_m")
    max_rotation_rad = _positive(max_rotation_rad, "max_rotation_rad")
    if type(max_target_steps) is not int or not 1 <= max_target_steps <= 5:
        raise ValueError("max_target_steps must be an integer in [1, 5]")
    names = ", ".join(effector.name for effector in spec.effectors)
    bounds = "; ".join(
        f"{effector.name}: x [{effector.workspace_min[0]}, {effector.workspace_max[0]}], "
        f"y [{effector.workspace_min[1]}, {effector.workspace_max[1]}], "
        f"z [{effector.workspace_min[2]}, {effector.workspace_max[2]}]"
        for effector in spec.effectors
        if effector.workspace_min is not None
    )
    bounds_prompt = f"Position bounds are {bounds}. " if bounds else ""
    prompt = (
        f"{robot_control_intro(spec.name)} using the supplied RGB views and measured end-effector state. "
        f"The named end effectors are: {names}. Call exactly one tool per turn, then you will see the result. "
        "Each end effector has its own fixed root frame, so the same world direction can have different signs. "
        f"{eef_frame_prompt(spec.name)} "
        "Positions are metres and orientations are unit quaternions in wxyz order. Grippers are explicit booleans. "
        f"{bounds_prompt}Call execute_eef_target with one COMPLETE absolute target and 1 to "
        f"{max_target_steps} control steps at {control_hz:g} Hz. The fixed target is reused for every selected step; "
        "you do not provide a trajectory and the host does not interpolate. The target must be within "
        f"{max_translation_m:g} m translation and {max_rotation_rad:g} rad rotation of the latest measured EEF pose. "
        "After the short segment you will receive fresh RGB views and measured state. "
        "Use shorter segments near contact. A gripper command or valid IK result is not proof "
        "of grasp or task success. Only RoboLab decides termination; there is no model-controlled done action."
    )

    if prompt_family == "baseline":
        return prompt
    prompt = prompt.replace("Use shorter segments near contact. ", "")
    timing = (
        "EXECUTION TIMING\n"
        f"Each selected control step advances simulation by {1 / control_hz:.9g} seconds. "
        "The simulator is paused during model inference. execute_eef_target holds the same "
        "absolute target for the selected number of steps; it does not wait until the pose is reached. "
        "At each step the host recomputes local damped least-squares IK from measured state, "
        "limiting the position error used by IK to 0.02 m, the rotation error to 0.1 rad, "
        "and the joint-position increment to 0.05 rad per joint. Actual movement may be smaller. "
        "The requested gripper command is applied at every selected step even if the arm has "
        "not reached the target. Use fresh measured state to assess arrival.\n\n"
        "The geometry_state object in measured_state supplies measured ee_pos, ee_quat (wxyz), "
        "and gripper_pos for the current observation; any measured_state_history entries supply "
        "the corresponding values for their own images. These are robot proprioception, "
        "not object ground-truth poses."
    )
    return "\n\n".join((prompt, timing, DROID_GRIPPER_GEOMETRY_PROMPT, DROID_CAMERA_GEOMETRY_PROMPT))


def build_agent_eef_policy_payload(
    spec: EmbodimentSpec,
    goal: str,
    observations: Sequence[tuple[int, Mapping[str, Any]]],
    *,
    history_length: int,
    prompt_family: str = "baseline",
    camera_names: Sequence[str],
    interval_observation_count: int | None = None,
    control_hz: float = 15.0,
    max_target_steps: int = 5,
    max_translation_m: float = 0.05,
    max_rotation_rad: float = 0.35,
    feedback: str | None = None,
) -> dict[str, Any]:
    """Build the backend-neutral numeric/text request metadata."""

    if not isinstance(goal, str) or not goal.strip():
        raise ValueError("goal must be a nonempty string")
    if type(history_length) is not int or history_length < 1:
        raise ValueError("history_length must be a positive integer")
    names = tuple(camera_names)
    if len(names) != 3 or len(set(names)) != 3 or not all(isinstance(name, str) and name for name in names):
        raise ValueError("agent-eef requires exactly three unique camera names")
    samples = list(observations)
    if not samples:
        raise ValueError("agent-eef requires at least one measured observation")
    if len(samples) > history_length:
        raise ValueError("observations exceed history_length")
    if interval_observation_count is None:
        interval_observation_count = len(samples)
    if type(interval_observation_count) is not int or interval_observation_count < len(samples):
        raise ValueError("interval_observation_count must be an integer covering every supplied observation")
    steps: list[int] = []
    states: list[dict[str, Any]] = []
    for step, proprio in samples:
        if type(step) is not int or step < 0:
            raise ValueError("observation control steps must be nonnegative integers")
        if steps and step <= steps[-1]:
            raise ValueError("observation control steps must be strictly increasing")
        steps.append(step)
        states.append(format_agent_eef_state(spec, proprio, prompt_family=prompt_family))
    control_hz = _positive(control_hz, "control_hz")
    max_translation_m = _positive(max_translation_m, "max_translation_m")
    max_rotation_rad = _positive(max_rotation_rad, "max_rotation_rad")
    if type(max_target_steps) is not int or not 1 <= max_target_steps <= 5:
        raise ValueError("max_target_steps must be an integer in [1, 5]")
    if feedback is not None and (not isinstance(feedback, str) or not feedback.strip()):
        raise ValueError("feedback must be None or a nonempty string")
    result = {
        "style": "agent-eef",
        "goal": goal.strip(),
        "input": {
            "history_length": history_length,
            "camera_names": list(names),
            "observation_count": len(samples),
            "interval_observation_count": interval_observation_count,
            "control_steps": steps,
            "order": "oldest_first",
        },
        "measured_state": states[-1],
        "control": {
            "control_hz": control_hz,
            "max_target_steps": max_target_steps,
            "max_translation_m": max_translation_m,
            "max_rotation_rad": max_rotation_rad,
        },
    }
    if prompt_family != "baseline":
        result["prompt_family"] = prompt_family
    if len(samples) > 1:
        result["measured_state_history"] = [
            {"control_step": step, "state": state} for step, state in zip(steps, states, strict=True)
        ]
    if feedback is not None:
        result["feedback"] = feedback.strip()
    return result


def select_agent_eef_observations(
    observations: Sequence[_ObservationT],
    *,
    history_length: int,
) -> list[_ObservationT]:
    """Select the latest bounded chronological image/proprioception history."""

    if type(history_length) is not int or history_length < 1:
        raise ValueError("history_length must be a positive integer")
    items = list(observations)
    if not items:
        raise ValueError("observations must be nonempty")
    return items[-history_length:]


def decode_agent_eef_action(
    spec: EmbodimentSpec,
    tool_name: str,
    arguments: Mapping[str, Any],
    proprio: Mapping[str, Any],
    *,
    control_hz: float = 15.0,
    max_target_steps: int = 5,
    max_translation_m: float = 0.05,
    max_rotation_rad: float = 0.35,
) -> AgentEEFPlan:
    """Validate one fixed EEF target and repeat it for the requested controls."""

    control_hz = _positive(control_hz, "control_hz")
    max_translation_m = _positive(max_translation_m, "max_translation_m")
    max_rotation_rad = _positive(max_rotation_rad, "max_rotation_rad")
    if type(max_target_steps) is not int or not 1 <= max_target_steps <= 5:
        raise ValueError("max_target_steps must be an integer in [1, 5]")
    if tool_name != "execute_eef_target":
        raise ValueError(f"unexpected agent-eef tool: {tool_name!r}")
    if not isinstance(arguments, Mapping):
        raise ValueError("tool arguments must be an object")

    measured_states = _state_map(spec, proprio)
    target_key = "target" if len(spec.effectors) == 1 else "targets"
    if set(arguments) != {target_key, "steps", "reason"}:
        raise ValueError(f"execute_eef_target requires exactly {target_key}, steps, and reason")
    steps = arguments["steps"]
    if type(steps) is not int or not 1 <= steps <= max_target_steps:
        raise ValueError(f"steps must be an integer in [1, {max_target_steps}]")
    reason = arguments["reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a nonempty string")
    raw_targets = (
        {spec.effectors[0].name: arguments[target_key]}
        if len(spec.effectors) == 1
        else arguments[target_key]
    )
    expected_names = {effector.name for effector in spec.effectors}
    if not isinstance(raw_targets, Mapping) or set(raw_targets) != expected_names:
        raise ValueError("targets must contain every configured end effector exactly once")

    next_states: dict[str, EffectorState] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    for effector in spec.effectors:
        raw = raw_targets[effector.name]
        if not isinstance(raw, Mapping) or set(raw) != {"position", "quaternion_wxyz", "gripper_closed"}:
            raise ValueError(
                f"{effector.name} target requires exactly position, quaternion_wxyz, and gripper_closed"
            )
        position = _finite_vector(raw["position"], 3, f"{effector.name}.position")
        quaternion = _unit_quaternion(raw["quaternion_wxyz"], f"{effector.name}.quaternion_wxyz")
        gripper_closed = raw["gripper_closed"]
        if type(gripper_closed) is not bool:
            raise ValueError(f"{effector.name}.gripper_closed must be boolean")
        if effector.workspace_min is not None:
            lower = np.asarray(effector.workspace_min)
            upper = np.asarray(effector.workspace_max)
            if np.any(position < lower) or np.any(position > upper):
                raise ValueError(f"{effector.name} target lies outside the configured workspace")
        measured = measured_states[effector.name]
        translation_m = float(np.linalg.norm(position - measured.position_m))
        rotation_rad = float((_rotation(quaternion) * _rotation(measured.quaternion_wxyz).inv()).magnitude())
        if translation_m > max_translation_m + 1e-12:
            raise ValueError(
                f"{effector.name} target translation {translation_m:.6g} m exceeds {max_translation_m:g} m"
            )
        if rotation_rad > max_rotation_rad + 1e-12:
            raise ValueError(
                f"{effector.name} target rotation {rotation_rad:.6g} rad exceeds {max_rotation_rad:g} rad"
            )
        next_states[effector.name] = EffectorState(position, quaternion, gripper_closed)
        diagnostics[effector.name] = {
            "target_translation_m": translation_m,
            "target_rotation_rad": rotation_rad,
        }

    action = _hold_action(spec, next_states)
    actions = np.repeat(action[None, :], steps, axis=0)
    feedback = (
        f"Executed fixed EEF target for {steps} control step{'s' if steps != 1 else ''}; "
        "inspect the fresh measured pose and RGB views before the next target."
    )
    return AgentEEFPlan(
        actions=actions,
        status="executing",
        tool_name=tool_name,
        note=reason.strip(),
        feedback=feedback,
        metadata={
            "steps": steps,
            "fixed_target": {
                name: {
                    "position": state.position_m.tolist(),
                    "quaternion_wxyz": state.quaternion_wxyz.tolist(),
                    "gripper_closed": state.gripper_closed,
                }
                for name, state in next_states.items()
            },
            "validation": diagnostics,
            "max_translation_m": max_translation_m,
            "max_rotation_rad": max_rotation_rad,
            "control_hz": control_hz,
            "interpolation": False,
        },
        commanded_state=next_states,
    )


__all__ = [
    "AgentEEFPlan",
    "EffectorSpec",
    "EffectorState",
    "EmbodimentSpec",
    "MOVE_FIELDS",
    "POSITION_FIELDS",
    "ROTATION_FIELDS",
    "RPY_CONVENTION",
    "agent_eef_policy_instructions",
    "agent_eef_state",
    "build_agent_eef_policy_payload",
    "build_agent_eef_tools",
    "decode_agent_eef_action",
    "format_agent_eef_state",
    "select_agent_eef_observations",
]
