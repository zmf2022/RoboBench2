# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the Jev two-stage policy (no server, no sim).

These pin the invariants that make the RoboJEV port a port rather than a
lookalike: normalised steps, criteria that name fields the state actually
measures, a physical grasp test, and a guard that consumes each decision once.
"""

import re

import numpy as np
import pytest

from policies.jev.chat_backend import ChatBackend
from policies.jev.execution import ActionGuard
from policies.jev.geometry import DIRECTION_TOLERANCE_M, direction, relation
from policies.jev.jev_backend import validate_choice
from policies.jev.prompts import (
    GRIPPER_CRITERIA,
    INTENT_RULES,
    INTENTS,
    motor_criteria,
    motor_rules,
)
from policies.jev.state import build_state
from policies.jev.types import AXIS_CHOICES, EefDecision, PolicyError
from robolab.core.task.instruction import TrackedInstructionFields


def _state(**overrides):
    kwargs = dict(
        task="Pick up the banana and place it in the bowl",
        task_id="BananaInBowlTask",
        state_id="episode_0000:0:3",
        step_id=3,
        tcp=[0.36, 0.0, 0.30],
        tcp_quaternion=[1.0, 0.0, 0.0, 0.0],
        object_pos=[0.35, 0.19, 0.08],
        object_quaternion=[1.0, 0.0, 0.0, 0.0],
        object_velocity=[0.0, 0.0, 0.0],
        container_pos=[0.33, -0.1, 0.11],
        gripper_width=0.085,
        gripper_target="open",
        objects_in_contact=[],
        object_label="banana",
        container_label="bowl",
        grasp_hover_m=0.0,
        release_hover_m=0.10,
        container_radius_m=0.05,
        object_rest_z=0.08,
        transport_clearance_m=0.08,
        object_long_axis_local=(1.0, 0.0, 0.0),
    )
    kwargs.update(overrides)
    return build_state(**kwargs)


# ---------------------------------------------------------------------------
# Action semantics
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("choices", [
    ("positive", "zero", "zero"),
    ("positive", "negative", "zero"),
    ("negative", "positive", "positive"),
])
def test_nonzero_axes_are_normalised_to_one_step(choices):
    """A diagonal decision travels exactly as far as a straight one."""
    delta = EefDecision("s", *choices, gripper="hold").delta(0.02)
    assert np.linalg.norm(delta) == pytest.approx(0.02)


def test_all_zero_stays_still():
    delta = EefDecision("s", "zero", "zero", "zero", gripper="close").delta(0.02)
    assert not delta.any()


@pytest.mark.parametrize("bad", [
    {"x": "hold"}, {"z": "up"}, {"gripper": "zero"}, {"gripper": "release"},
])
def test_invalid_labels_cannot_be_constructed(bad):
    kwargs = {"x": "zero", "y": "zero", "z": "zero", "gripper": "hold"}
    kwargs.update(bad)
    with pytest.raises(PolicyError):
        EefDecision("s", **kwargs)


def test_empty_state_id_rejected():
    with pytest.raises(PolicyError):
        EefDecision("", "zero", "zero", "zero", gripper="hold")


def test_direction_deadband():
    assert direction(DIRECTION_TOLERANCE_M * 0.9) == "zero"
    assert direction(DIRECTION_TOLERANCE_M * 1.1) == "positive"
    assert direction(-DIRECTION_TOLERANCE_M * 1.1) == "negative"
    assert relation([0.0, 0.0, -0.4])["xy_aligned"] is True


# ---------------------------------------------------------------------------
# The criteria and the state schema must agree
# ---------------------------------------------------------------------------

# Tokens that look like field paths (they carry an underscore or a dot) rather
# than prose. Anything matching this must resolve in a real state.
_FIELD_TOKEN = re.compile(r"\b[a-z][a-z0-9]*(?:[_.][a-z0-9]+)+\b")
# Prose that happens to match the shape above.
_NOT_FIELDS = {"robot-base", "e.g", "i.e"}


_MISSING = object()


def _resolve(state, path):
    """Walk a dotted path from the state root or one of its sections.

    A field whose measured value is ``None`` (``held_object`` with an empty
    gripper) still exists, so presence is tested by key, not by truthiness.
    """
    # RoboJEV's rules address the payload as ``state.<field>``.
    path = path[len("state."):] if path.startswith("state.") else path
    for root in (state, state["robot"], state["relations"], state["target"]):
        node = root
        for segment in path.split("."):
            if not isinstance(node, dict) or segment not in node:
                node = _MISSING
                break
            node = node[segment]
        if node is not _MISSING:
            return True
    return False


def _cited_fields(text):
    return {t for t in _FIELD_TOKEN.findall(text) if t not in _NOT_FIELDS}


def test_every_intent_criterion_names_a_measured_field():
    """The bug this suite exists for: criteria that cite facts nothing measures.

    An INTENT whose condition mentions e.g. whether a grasp is "secured" cannot
    be evaluated, and the model falls back to guessing at the exact moment the
    task is decided.
    """
    state = {**_state(), "jev_intent": "approach"}
    missing, prose = {}, {}
    for intent, criterion in INTENTS.items():
        cited = _cited_fields(criterion)
        unresolved = {f for f in cited if not _resolve(state, f)}
        if unresolved:
            missing[intent] = sorted(unresolved)
        # Both halves are needed. Checking only that cited fields resolve lets a
        # criterion written entirely in prose ("held AND secured") pass, which is
        # how the condition that decides the grasp became unevaluable.
        if len(cited) < 2:
            prose[intent] = criterion
    assert not missing, f"criteria cite fields the state does not measure: {missing}"
    assert not prose, f"criteria are prose, not measured conditions: {prose}"


def test_intent_rules_and_motor_criteria_name_measured_fields():
    state = {**_state(), "jev_intent": "approach"}
    texts = [INTENT_RULES, motor_rules(0.02), *GRIPPER_CRITERIA.values()]
    for axis in ("x", "y", "z"):
        texts.extend(motor_criteria(axis).values())
    unresolved = sorted({f for text in texts for f in _cited_fields(text)
                         if not _resolve(state, f)})
    assert not unresolved, f"criteria cite unmeasured fields: {unresolved}"


def test_motor_criteria_are_conditioned_on_the_intent():
    """Stage 2 must constrain axes per intent, not describe directions generically."""
    for axis in ("x", "y", "z"):
        criteria = motor_criteria(axis)
        assert set(criteria) == set(AXIS_CHOICES)
        assert all("ntent" in text for text in criteria.values())
    assert all("ntent" in text for text in GRIPPER_CRITERIA.values())


def test_every_intent_has_a_motor_rule():
    rules = motor_rules(0.02)
    for intent in INTENTS:
        assert f"{intent}:" in rules


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def test_held_object_needs_contact_and_a_plausible_width():
    closed_on_nothing = _state(gripper_width=0.001, objects_in_contact=[])
    assert closed_on_nothing["robot"]["held_object"] is None

    touching_but_open = _state(gripper_width=0.085, objects_in_contact=["banana"])
    assert touching_but_open["robot"]["held_object"] is None

    gripping = _state(gripper_width=0.03, objects_in_contact=["banana"])
    assert gripping["robot"]["held_object"] == "banana"


def test_placement_target_compensates_for_the_object_in_the_jaws():
    """Aim the OBJECT at the container, not the TCP."""
    free = _state()
    held = _state(gripper_width=0.03, objects_in_contact=["banana"],
                  object_pos=[0.36, 0.02, 0.28], tcp=[0.36, 0.0, 0.30])
    # The object hangs 2 cm to the robot's left of the TCP, so the TCP must end
    # up 2 cm to the right of the container centre.
    assert held["relations"]["placement_tcp_from_tcp"]["delta_m"][1] == pytest.approx(
        free["relations"]["placement_tcp_from_tcp"]["delta_m"][1] - 0.02, abs=1e-6)


def test_transport_and_settle_booleans_track_the_object():
    resting = _state()
    assert resting["relations"]["object_clear_for_transport"] is False
    lifted = _state(object_pos=[0.35, 0.19, 0.17], gripper_width=0.03,
                    objects_in_contact=["banana"])
    assert lifted["relations"]["object_clear_for_transport"] is True

    dropped_in = _state(object_pos=[0.33, -0.1, 0.12])
    assert dropped_in["relations"]["object_inside_target_xy"] is True
    assert dropped_in["relations"]["object_settled_in_target"] is True
    still_falling = _state(object_pos=[0.33, -0.1, 0.12], object_velocity=[0, 0, -0.5])
    assert still_falling["relations"]["object_settled_in_target"] is False


def test_state_declares_its_frame_and_units():
    state = _state()
    assert (state["frame"], state["units"]) == ("robot_base", "m")
    assert state["recent_actions"] == []


# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

def test_guard_consumes_each_decision_once():
    guard, state = ActionGuard(), _state()
    decision = EefDecision(state["state_id"], "positive", "zero", "zero", gripper="hold")
    guard.resolve(state, decision, 0.02)
    with pytest.raises(PolicyError):
        guard.resolve(state, decision, 0.02)


def test_guard_rejects_a_stale_decision():
    guard, state = ActionGuard(), _state()
    stale = EefDecision("episode_0000:0:2", "positive", "zero", "zero", gripper="hold")
    with pytest.raises(PolicyError):
        guard.resolve(state, stale, 0.02)


def test_hold_keeps_the_latched_gripper_command():
    guard, state = ActionGuard(), _state(gripper_target="close")
    _, grip, _ = guard.resolve(
        state, EefDecision(state["state_id"], "zero", "zero", "zero", gripper="hold"), 0.02)
    assert grip == "close"


def test_guard_rejects_motion_out_of_the_workspace():
    guard = ActionGuard()
    state = _state(tcp=[0.895, 0.0, 0.30])
    delta, _, reason = guard.resolve(
        state, EefDecision(state["state_id"], "positive", "zero", "zero", gripper="hold"), 0.02)
    assert (delta, reason) == (None, "workspace_rejected")


def test_guard_will_not_push_a_held_object_through_its_support():
    guard = ActionGuard()
    state = _state(gripper_width=0.03, objects_in_contact=["banana"],
                   object_pos=[0.35, 0.19, 0.08], object_rest_z=0.08)
    delta, _, reason = guard.resolve(
        state, EefDecision(state["state_id"], "zero", "zero", "negative", gripper="hold"), 0.02)
    assert (delta, reason) == (None, "held_object_support_rejected")


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _StubChat:
    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, messages, choices=None):
        self.prompts.append((messages[-1]["content"][0]["text"], tuple(choices or ())))
        return {"choices": [{"message": {"content": self.replies.pop(0)}}]}


def test_chat_backend_parses_bare_and_prefixed_labels():
    assert ChatBackend(_StubChat(["grasp"])).decide_intent(_state()) == "grasp"
    assert ChatBackend(_StubChat(["INTENT: finish"])).decide_intent(_state()) == "finish"


def test_chat_backend_returns_labels_not_metres():
    chat = _StubChat(["positive", "zero", "negative", "close"])
    answers = ChatBackend(chat, step_m=0.02).decide_motor(_state(), "approach")
    assert answers == {"x": "positive", "y": "zero", "z": "negative", "gripper": "close"}
    # The axis vocabulary is RoboJEV's, and the intent reaches the model as state.
    assert chat.prompts[0][1] == AXIS_CHOICES
    assert '"jev_intent": "approach"' in chat.prompts[0][0]


def test_chat_backend_refuses_to_guess():
    with pytest.raises(PolicyError):
        ChatBackend(_StubChat(["sideways", "zero", "zero", "close"])).decide_motor(
            _state(), "approach")
    with pytest.raises(PolicyError):
        ChatBackend(_StubChat([""])).decide_intent(_state())


def test_validate_choice_accepts_a_normalised_argmax():
    # Shape verified against the live API: {choice, confidence, probabilities}.
    assert validate_choice(
        {"choice": "grasp", "confidence": 0.29,
         "probabilities": {"grasp": 0.8, "lift": 0.2}},
        ["grasp", "lift"]) == "grasp"


@pytest.mark.parametrize("answer", [
    {"choice": "grasp", "confidence": 0.3,
     "probabilities": {"grasp": 0.5, "lift": 0.2}},                      # unnormalised
    {"choice": "lift", "confidence": 0.3,
     "probabilities": {"grasp": 0.8, "lift": 0.2}},                      # not the argmax
    {"choice": "grasp", "confidence": 0.3, "probabilities": {"grasp": 1.0}},  # wrong labels
    {"choice": "grasp", "confidence": 0.3,
     "probabilities": {"grasp": True, "lift": False}},                   # bools
    {"choice": "grasp", "confidence": 1.4,
     "probabilities": {"grasp": 0.8, "lift": 0.2}},                      # confidence > 1
    {"choice": "grasp", "probabilities": {"grasp": 0.8, "lift": 0.2}},   # no confidence
    {"choice": "grasp", "confidence": 0.3},                              # no distribution
    "grasp",                                                             # not an object
])
def test_validate_choice_executes_nothing_on_a_malformed_reply(answer):
    with pytest.raises(PolicyError):
        validate_choice(answer, ["grasp", "lift"])


# ---------------------------------------------------------------------------
# Control layer (needs the sim packages; skipped on a bare checkout)
# ---------------------------------------------------------------------------

def test_one_decision_commands_exactly_one_step_of_travel(stubbed_client):
    """The whole decision's travel is spread across its control steps.

    Issuing the full delta on every control step instead multiplies the travel
    by ``control_steps`` -- it looks correct in the logs and overshoots by 4-8x.
    """
    JevClient, _EnvState = stubbed_client.JevClient, stubbed_client._EnvState
    from policies.vlm_agent.token_controller import REL_IK_ACHIEVED_FRACTION

    client = JevClient(backend=None, task="BananaInBowlTask", step_m=0.02,
                       control_steps=8, ik_scale=0.5)
    env = _EnvState()
    env.quat_ref = None  # skip the orientation hold; it needs the simulator's math
    delta = EefDecision("s", "positive", "negative", "zero", gripper="hold").delta(0.02)
    env.per_step_cmd = delta / REL_IK_ACHIEVED_FRACTION / client.control_steps
    env.steps_left = client.control_steps

    pose = {"ee_quat": np.array([1.0, 0.0, 0.0, 0.0]), "tcp": np.zeros(3)}
    steps = np.stack([client._action_for_step(env, pose)[:3]
                      for _ in range(client.control_steps)])
    commanded = steps.sum(axis=0) * client.ik_scale
    assert np.linalg.norm(commanded) * REL_IK_ACHIEVED_FRACTION == pytest.approx(0.02)
    assert env.steps_left == 0
    # The plan is exhausted: further control steps must not keep moving.
    assert not client._action_for_step(env, pose)[:3].any()


def test_unknown_task_is_rejected_not_defaulted(stubbed_client):
    with pytest.raises(ValueError):
        stubbed_client.JevClient(backend=None, task="NoSuchTask")




# ---------------------------------------------------------------------------
# Grasp yaw: which way round the jaws have to be
# ---------------------------------------------------------------------------

def _yaw_quat(rad):
    """WXYZ rotation about +Z."""
    return np.array([np.cos(rad / 2), 0.0, 0.0, np.sin(rad / 2)])


def test_jaws_must_close_across_an_elongated_object():
    """A cube can be grasped from any yaw; a 190 mm banana cannot.

    The 2F-85 opens 85 mm, so closing along the object's length does not fit --
    this is a prerequisite for the grasp, not a refinement of it.
    """
    # Gripper at identity: its jaw axis (base_link +y) runs along world +Y.
    across = _state(object_quaternion=_yaw_quat(0.0))          # banana along +X
    assert across["relations"]["grasp_yaw_aligned"] is True
    along = _state(object_quaternion=_yaw_quat(np.pi / 2))     # banana along +Y
    assert along["relations"]["grasp_yaw_aligned"] is False
    assert abs(along["relations"]["grasp_yaw_error_rad"]) == pytest.approx(np.pi / 2, abs=1e-3)  # state rounds to 4 dp


def test_grasp_yaw_tolerance_is_where_the_grasp_stops_fitting():
    from policies.jev.state import GRASP_YAW_TOLERANCE_RAD
    # 190 mm long, 35 mm thick, jaws gripping by ~70 mm.
    span = lambda t: 0.190 * np.sin(t) + 0.035 * np.cos(t)  # noqa: E731
    assert span(GRASP_YAW_TOLERANCE_RAD) == pytest.approx(0.070, abs=0.002)
    assert span(GRASP_YAW_TOLERANCE_RAD * 2) > 0.085        # past the aperture


def test_an_axis_has_no_head_or_tail():
    """A banana pointing north-east and one pointing south-west are one pose."""
    ne, sw = _state(object_quaternion=_yaw_quat(0.4)), _state(object_quaternion=_yaw_quat(0.4 + np.pi))
    assert (ne["relations"]["object_long_axis_rad"]
            == pytest.approx(sw["relations"]["object_long_axis_rad"]))


def test_a_round_object_needs_no_aiming():
    """No long axis declared -> the held orientation stands, as for RoboJEV's cube."""
    kwargs = dict(object_long_axis_local=None)
    assert _state(**kwargs)["relations"]["object_long_axis_rad"] is None
    assert _state(**kwargs)["relations"]["grasp_yaw_aligned"] is True


# ---------------------------------------------------------------------------
# Closed loop against a toy plant
# ---------------------------------------------------------------------------

def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def _hold_rotvec(ref, cur, max_rad=0.15):
    """``ref ⊗ cur⁻¹`` as a world-frame axis-angle -- the real correction's shape."""
    err = _qmul(np.asarray(ref, float), np.asarray(cur, float) * [1, -1, -1, -1])
    if err[0] < 0:
        err = -err
    angle = 2 * np.arccos(np.clip(err[0], -1.0, 1.0))
    axis = err[1:] / (np.linalg.norm(err[1:]) + 1e-12)
    rotvec = axis * angle
    magnitude = float(np.linalg.norm(rotvec))
    return rotvec * (max_rad / magnitude) if magnitude > max_rad else rotvec


def _integrate(quat, rotvec):
    """Apply a world-frame axis-angle to a WXYZ quaternion."""
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return quat
    axis = np.asarray(rotvec, float) / angle
    delta = np.array([np.cos(angle / 2), *(axis * np.sin(angle / 2))])
    out = _qmul(delta, quat)
    return out / np.linalg.norm(out)


@pytest.fixture
def stubbed_client(monkeypatch):
    """Import ``policies.jev.client`` with the simulator packages stubbed.

    The control layer is the half that cannot be ported verbatim, so it gets a
    closed-loop test rather than only unit checks: this is what catches a step
    that travels ``control_steps`` times too far, a deadband the step cannot
    converge into, jaws that close along the object, and an intent table with a
    dead end in it.
    """
    import sys
    import types

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    class _Base:
        def __init__(self):
            self._chunks, self._counters, self._eval_episode_idx = {}, {}, 0

        def begin_episode(self, idx):
            self._eval_episode_idx = idx

        def reset(self, *, env_id=None):
            pass

        def close(self):
            pass

        @staticmethod
        def _get_env_gt_state(raw_obs, env_id):
            gt = raw_obs.get("gt_state") if isinstance(raw_obs, dict) else None
            return gt[env_id] if isinstance(gt, dict) and env_id in gt else None

    for name in ("robolab", "robolab.core", "robolab.core.observations", "robolab.eval"):
        module(name)
    module("robolab.core.observations.observation_utils",
           unpack_proprio_obs=lambda obs, env_id=0: obs["proprio_obs"][env_id])
    module("robolab.eval.base_client", InferenceClient=_Base)
    # Only the sim-bound helpers are stubbed. ``yaw_aligned_quat`` is pure numpy
    # and is exercised for real -- it is the thing under test here.
    module("policies.vlm_agent.token_controller",
           GRIPPER_CLOSE=1.0, GRIPPER_OPEN=0.0, REL_IK_ACHIEVED_FRACTION=0.28,
           fingertip_position=lambda p, q: np.asarray(p, float) + np.array([0.0, 0.0, -0.131]),
           gripper_width_m=lambda g: 0.085 * (1.0 - float(np.clip(g, 0.0, 1.0))),
           hold_orientation_rotvec=_hold_rotvec)
    monkeypatch.delitem(sys.modules, "policies.jev.client", raising=False)
    import policies.jev.client as client_module
    yield client_module
    sys.modules.pop("policies.jev.client", None)


class _CriteriaBackend:
    """Answers exactly as the published criteria say to. No model involved."""

    def __init__(self):
        self.intents = []

    def decide_intent(self, state):
        r, held = state["relations"], state["robot"]["held_object"]
        if r["object_settled_in_target"]:
            intent = "withdraw" if r["tcp_above_object_m"] < r["withdraw_clearance_m"] else "finish"
        elif held:
            if not r["target_from_object"]["xy_aligned"]:
                intent = "carry" if r["object_clear_for_transport"] else "lift"
            else:
                intent = ("release" if r["placement_tcp_from_tcp"]["directions"]["z"] == "zero"
                          else "lower")
        else:
            ready = (set(r["grasp_tcp_from_tcp"]["directions"].values()) == {"zero"}
                     and r["grasp_yaw_aligned"])
            intent = "grasp" if ready else "approach"
        self.intents.append(intent)
        return intent

    def decide_motor(self, state, intent):
        r = state["relations"]
        grasp = r["grasp_tcp_from_tcp"]
        target = r["target_from_object"]["directions"]
        place = r["placement_tcp_from_tcp"]["directions"]
        # approach holds height until the XY is aligned AND the wrist is across.
        descend = grasp["directions"]["z"] if (grasp["xy_aligned"] and r["grasp_yaw_aligned"]) else "zero"
        table = {
            "approach": ({**grasp["directions"], "z": descend}, "open"),
            "grasp": ({"x": "zero", "y": "zero", "z": "zero"}, "close"),
            "lift": ({"x": "zero", "y": "zero", "z": "positive"}, "hold"),
            "carry": ({**target, "z": "zero"}, "hold"),
            "lower": ({"x": "zero", "y": "zero", "z": place["z"]}, "hold"),
            "release": ({"x": "zero", "y": "zero", "z": "zero"}, "open"),
            "withdraw": ({"x": "zero", "y": "zero", "z": "positive"}, "open"),
            "finish": ({"x": "zero", "y": "zero", "z": "zero"}, "open"),
        }
        axes, gripper = table[intent]
        return {**axes, "gripper": gripper}

    def close(self):
        pass


def _run_episode(client, backend, banana_yaw_rad, budget=750):
    """Toy plant: rel-IK achieves 28%, the wrist integrates what it is commanded."""
    flange = np.array([0.40, 0.05, 0.45])
    ee_quat = np.array([1.0, 0.0, 0.0, 0.0])
    banana, bowl = np.array([0.35, 0.19, 0.08]), np.array([0.33, -0.10, 0.11])
    banana_quat = _yaw_quat(banana_yaw_rad)
    grip, contacts = 0.0, []
    jaw_error = []

    for _ in range(budget):
        obs = {"proprio_obs": {0: {"ee_pos": flange.copy(), "ee_quat": ee_quat.copy(),
                                   "gripper_pos": np.array([grip])}},
               "gt_state": {0: {"objects": {
                   "banana": {"pos": banana.copy(), "quat": banana_quat.copy(),
                              "vel": np.zeros(6)},
                   "bowl": {"pos": bowl.copy(), "quat": np.array([1.0, 0.0, 0.0, 0.0]),
                            "vel": np.zeros(6)}},
                   "robot": {"objects_in_contact": list(contacts)}}}}
        action = client.infer(obs, "Pick up the banana and place it in the bowl", env_id=0)["action"]
        assert action.shape == (7,)

        flange = flange + action[:3] * client.ik_scale * 0.28
        ee_quat = _integrate(ee_quat, action[3:6] * client.ik_scale * 0.28)
        tcp = flange + np.array([0.0, 0.0, -0.131])
        # A 35 mm banana stops the fingers; jaws that closed to zero width would
        # make the width-band grasp test unsatisfiable.
        ceiling = 1.0 - 0.035 / 0.085 if contacts else 1.0
        grip = min(ceiling, grip + 0.02) if action[6] > 0.5 else max(0.0, grip - 0.02)
        if action[6] > 0.5 and np.linalg.norm(tcp - banana) < 0.03:
            jaw_error.append(_state(tcp_quaternion=ee_quat, object_quaternion=banana_quat)
                             ["relations"]["grasp_yaw_error_rad"])
            contacts = ["banana"]
        if contacts and grip > 0.1:
            banana = tcp.copy()
        elif contacts and action[6] <= 0.5:
            contacts = []
            banana[2] = 0.12 if np.linalg.norm(banana[:2] - bowl[:2]) < 0.05 else 0.08
    return banana, bowl, jaw_error


@pytest.mark.parametrize("banana_yaw_deg", [0, 45, 90, 135])
def test_the_criteria_drive_a_pick_and_place_to_completion(stubbed_client, banana_yaw_deg):
    """Whatever way round the banana lies, the wrist aims and the task completes."""
    backend = _CriteriaBackend()
    client = stubbed_client.JevClient(backend=backend, task="BananaInBowlTask", ik_scale=0.5)
    client.begin_episode(0)

    banana, bowl, jaw_error = _run_episode(client, backend, np.radians(banana_yaw_deg))

    assert np.linalg.norm(banana[:2] - bowl[:2]) < 0.05, "the banana never reached the bowl"
    # The fingers only ever closed while the jaws were across the object.
    from policies.jev.state import GRASP_YAW_TOLERANCE_RAD
    assert jaw_error and max(abs(e) for e in jaw_error) <= GRASP_YAW_TOLERANCE_RAD
    # Every phase has to be reachable: a criterion no state can satisfy shows up
    # here as a missing intent, not as a silently worse success rate.
    assert set(backend.intents) == set(INTENTS)
    # ``finish`` is an intention, not a verdict -- it must stop costing calls.
    assert backend.intents.count("finish") <= stubbed_client.FINISH_HOLD_STREAK


def test_backend_targets_the_endpoint_that_actually_serves_this_api():
    """Verified live. OpenRouter 401s on the same key and does not serve it.

    The catalogue-style model id ``typesafe/jev-1.13`` belongs to a chat
    completions listing, not to this decisions API, and the key is a TypeSafe
    key -- all three have to move together or none of them work.
    """
    from policies.jev.jev_backend import API_KEY_ENV, DEFAULT_MODEL, DEFAULT_URL
    assert DEFAULT_URL == "https://api.typesafe.ai/v1/systemone"
    assert DEFAULT_MODEL == "jev-1.13.0"
    assert API_KEY_ENV == "TYPESAFE_API_KEY"
    assert DEFAULT_URL.startswith("https://")


def test_key_is_read_from_a_local_env_file(tmp_path):
    from policies.jev.jev_backend import key_from_env_file
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "OTHER_KEY=not-this-one\n"
        "TYPESAFE_API_KEY='apikey_example'\n",
        encoding="utf-8")
    assert key_from_env_file(env) == "apikey_example"


def test_missing_env_file_is_not_an_error(tmp_path):
    from policies.jev.jev_backend import key_from_env_file
    assert key_from_env_file(tmp_path / "nope.env") is None
    blank = tmp_path / ".env"
    blank.write_text("TYPESAFE_API_KEY=\n", encoding="utf-8")
    assert key_from_env_file(blank) is None


def test_a_stalled_descent_counts_as_having_arrived():
    """The 2F-85's fingers dip as they close, so a tip near the support jams the
    arc on the table with the jaws still open. ``vlm_pinpoint``'s grasp waypoint
    carries ``on_stall="advance"`` for the same reason: below the fingertip floor
    no further downward command helps, and pushing anyway freezes the episode.
    """
    descending = dict(tcp=[0.35, 0.19, 0.12], object_pos=[0.35, 0.19, 0.08])
    assert _state(**descending)["relations"]["at_grasp_pose"] is False
    assert _state(**descending, previous_action_stalled=True)["relations"]["at_grasp_pose"] is True


def test_a_stall_while_still_far_in_xy_is_not_a_grasp_pose():
    """Blocked on the way in is an obstruction, not arrival."""
    blocked = _state(tcp=[0.50, 0.30, 0.12], object_pos=[0.35, 0.19, 0.08],
                     previous_action_stalled=True)
    assert blocked["relations"]["at_grasp_pose"] is False


def test_guard_refuses_to_translate_before_the_jaws_are_aimed():
    """Ordering enforced in code, not asked for in prompt text.

    A saturated DLS solve buys position progress with orientation error, so
    "approach while the wrist turns" does not turn the wrist -- it defers the
    whole rotation to the moment the arm stops, which is when it is already down
    at the object. Holding still is what makes the turn happen.
    """
    guard = ActionGuard()
    misaimed = _state(object_quaternion=_yaw_quat(np.pi / 2))
    assert misaimed["relations"]["grasp_yaw_aligned"] is False
    delta, grip, reason = guard.resolve(
        misaimed,
        EefDecision(misaimed["state_id"], "positive", "positive", "negative", gripper="open"),
        0.02)
    assert (delta, reason) == (None, "aiming_jaws")
    assert grip == "open"          # the gripper command still applies


def test_guard_stops_blocking_once_aimed():
    guard = ActionGuard()
    aimed = _state(object_quaternion=_yaw_quat(0.0))
    delta, _, reason = guard.resolve(
        aimed, EefDecision(aimed["state_id"], "positive", "zero", "zero", gripper="open"), 0.02)
    assert reason == "accepted" and delta is not None


def test_a_held_object_is_never_blocked_for_aiming():
    """Spinning a loaded wrist twists the object out, so carrying is exempt."""
    guard = ActionGuard()
    carrying = _state(object_quaternion=_yaw_quat(np.pi / 2), gripper_width=0.03,
                      objects_in_contact=["banana"], object_pos=[0.35, 0.19, 0.20])
    delta, _, reason = guard.resolve(
        carrying, EefDecision(carrying["state_id"], "zero", "negative", "zero", gripper="hold"), 0.02)
    assert reason == "accepted" and delta is not None


# Regression checks at the client boundary (simulator dependencies stubbed).
def _client_observation():
    return {
        "gt_state": {0: {
            "objects": {
                "banana": {"pos": np.array([0.35, 0.19, 0.08])},
                "bowl": {"pos": np.array([0.33, -0.10, 0.11])},
            },
            "robot": {"objects_in_contact": []},
        }},
    }


def test_measured_stall_reaches_the_model_state(stubbed_client):
    client = stubbed_client.JevClient(backend=None, task="BananaInBowlTask")
    env = client._env_state(0)
    env.prev_tip = np.array([0.35, 0.19, 0.12])
    env.pending = {"delta_requested_m": [0.0, 0.0, -0.03]}
    pose = {"tcp": env.prev_tip.copy(), "ee_quat": _yaw_quat(0), "gripper_width": 0.085}
    client._close_out_history(env, pose)
    state = client._measured_state(_client_observation(), "pick and place", env, pose, 0)
    assert state["relations"]["previous_action_stalled"] is True
    assert state["relations"]["at_grasp_pose"] is True


def test_env_reset_does_not_reuse_consumed_decision_ids(stubbed_client):
    client = stubbed_client.JevClient(backend=None, task="BananaInBowlTask")
    pose = {"tcp": np.array([0.35, 0.19, 0.12]), "ee_quat": _yaw_quat(0),
            "gripper_width": 0.085}
    def state():
        return client._measured_state(_client_observation(), "pick and place",
                                      client._env_state(0), pose, 0)
    before = state()
    old = EefDecision(before["state_id"], "zero", "zero", "zero", "open")
    client.guard.resolve(before, old, client.step_m)
    other = client._env_state(1)
    client.reset(env_id=0)
    after = state()
    assert after["state_id"] != before["state_id"]
    assert client._env_state(1) is other
    with pytest.raises(PolicyError):
        client.guard.resolve(after, old, client.step_m)
    client.guard.resolve(after, EefDecision(after["state_id"], "zero", "zero", "zero", "open"),
                         client.step_m)


def test_multiple_tasks_are_rejected_instead_of_using_the_first(stubbed_client):
    with pytest.raises(ValueError, match="one task"):
        stubbed_client.JevClient(backend=None, task=["BananaInBowlTask", "NoSuchTask"])


def test_jev_reads_selected_instruction_text_and_records_usage(stubbed_client):
    client = stubbed_client.JevClient(backend=None, task="BananaInBowlTask")
    pose = {"tcp": np.array([0.35, 0.19, 0.12]), "ee_quat": _yaw_quat(0),
            "gripper_width": 0.085}
    instruction = TrackedInstructionFields({"vague": "Move the fruit", "video": "demo.mp4"})
    state = client._measured_state(_client_observation(), instruction,
                                   client._env_state(0), pose, 0)
    assert state["task"] == "Move the fruit"
    assert instruction.instruction_fields_used == ["vague"]
