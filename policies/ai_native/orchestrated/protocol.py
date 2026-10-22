"""Upstream hybrid response contract with a RoboLab execution-plan adapter."""

from collections.abc import Mapping

import numpy as np

from policies.ai_native.common.agent_eef import AgentEEFPlan, EffectorState, decode_agent_eef_action
from policies.ai_native.orchestrated.gate_assessment import validate_assessment
from policies.ai_native.orchestrated.kinematics import edited_targets
from policies.ai_native.orchestrated.schema import response_schema

TOOL_NAME = "review_policy_proposal"


def review_tool(spec):
    return dict(type="function", function=dict(
        name=TOOL_NAME,
        description="Review the fresh proposal and select student/edit/eef/stop.",
        parameters=response_schema(),
    ))


def _edited_plan(target, reason):
    state = EffectorState(np.asarray(target['position']), np.asarray(target['quaternion_wxyz']),
                          target['gripper_closed'])
    action = np.array([[*target['position'], *target['quaternion_wxyz'], float(target['gripper_closed'])]])
    return AgentEEFPlan(actions=action, status="executing", tool_name=TOOL_NAME, note=reason,
                        feedback=None, metadata=dict(fixed_target=dict(arm=target)),
                        commanded_state=dict(arm=state))


def validate_review(name, response, request, *, spec, proprio):
    """Apply the original outcome/intent gate and mode semantics before execution."""
    if name != TOOL_NAME:
        raise ValueError("Expected the proposal-review tool")
    if not isinstance(response, Mapping) or set(response) != set(response_schema()['required']):
        raise ValueError("Review must match the upstream response schema")
    if response['request_id'] != request['request_id']:
        raise ValueError("Stale review: request_id does not match the current proposal")
    if not isinstance(response['reason'], str) or not response['reason'].strip():
        raise ValueError("A visible observation/decision explanation is required")
    mode = response['mode']
    if mode not in ('student', 'edit', 'eef', 'stop'):
        raise ValueError("Expected student/edit/eef/stop")
    validate_assessment(response, dict(step_id=request['control_step']))
    if mode == 'stop':
        return None
    steps = response['steps']
    cap = request['proposal']['horizon'] if mode == 'student' else 5
    if type(steps) is not int or not 1 <= steps <= cap:
        raise ValueError(f"Expected 1..{cap} execution steps")
    if mode == 'student':
        return None
    if mode == 'edit':
        # Offset is bounded relative to the proposed trajectory, not to the
        # current measured pose. Each edited target still uses bounded local IK.
        targets = edited_targets(request['proposal']['trajectory'], steps, response['edit'])
        return [_edited_plan(target, response['reason']) for target in targets]
    target = response['target']
    if not isinstance(target, Mapping):
        raise ValueError("eef requires an explicit target")
    quaternion = np.asarray(target.get('quaternion_wxyz'), dtype=float)
    if (quaternion.shape != (4,) or not np.isfinite(quaternion).all()
            or abs(np.linalg.norm(quaternion) - 1) > 1e-4):
        raise ValueError("EEF target requires a unit WXYZ quaternion")
    plan = decode_agent_eef_action(spec, 'execute_eef_target', dict(
        target=target, steps=steps, reason=response['reason'],
    ), proprio, control_hz=request['control_hz'], max_target_steps=5,
        max_translation_m=0.05, max_rotation_rad=0.35)
    return [plan] * steps
