"""CPU contract tests. No Isaac application, policy server or model is started."""

import argparse
import copy
import json
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from policies.ai_native.orchestrated.client import OrchestratedClient
from policies.ai_native.orchestrated.kinematics import JointEdge, JOINT_NAMES, RobotFK, transform
from policies.ai_native.orchestrated.protocol import TOOL_NAME, validate_review
from policies.ai_native.orchestrated.session import CAMERA_NAMES, observation_blocks


def observation():
    return dict(
        image_obs={name: np.zeros((1, 8, 12, 3), np.uint8) for name in CAMERA_NAMES},
        proprio_obs=dict(arm_joint_pos=np.zeros((1, 7)), gripper_pos=np.zeros((1, 1)),
                         ee_pos=np.zeros((1, 3)), ee_quat=np.array([[1., 0., 0., 0.]])),
        gt_state={0: dict(secret_object_pose="not-for-model")},
        task_context=dict(success_criteria="not-for-model"), remaining_steps=999,
    )


def response(request, mode="student", steps=1):
    return dict(request_id=request["request_id"], mode=mode, steps=steps, reason="Observed approach is appropriate",
                edit=dict(delta_position=[0, 0, 0], delta_rotation_vector=[0, 0, 0], gripper="keep"),
                target=dict(
                    position=[.01, 0, 0], quaternion_wxyz=[1, 0, 0, 0], gripper_closed=False),
                assessment=dict(
                    task_progress=dict(verified_completed=[], currently_attempting="approach", remaining=["place"]),
                    current_subgoal="approach", execution_evidence="Measured motion and RGB",
                    expected_next_intent="approach", predicted_next_intent="approach", intent_evidence="FK targets",
                    execution_status="not_started" if request["control_step"] == 0 else "progressing",
                    intent_status="aligned" if mode == "student" else "misaligned"))


class Student:
    def __init__(self):
        self.requests = []
        self.resets = 0
        self.closed = False

    def propose(self, obs, instruction):
        self.requests.append(instruction)
        chunk = np.full((15, 8), len(self.requests), dtype=np.float32)
        chunk[:, 0] += np.arange(15) / 100
        chunk[:, 7] = .6
        return chunk

    def reset(self):
        self.resets += 1

    def close(self):
        self.closed = True


class Reviewer:
    def __init__(self, choices):
        self.choices = iter(choices)
        self.requests = []
        self.results = []
        self.closed = False
        self.last_metadata = {}

    def query(self, request):
        self.requests.append(copy.deepcopy(request))
        self.last_metadata = dict(token_usage=dict(total=dict(inputTokens=10*len(self.requests),
                                                             outputTokens=len(self.requests))))
        choice = next(self.choices)
        if isinstance(choice, BaseException):
            raise choice
        return TOOL_NAME, choice(request)

    def record_host_result(self, record):
        self.results.append(record)

    def close(self):
        self.closed = True


class Adapter:
    def __init__(self):
        self.calls = 0

    def action_for_plan(self, plan, *, env_id):
        self.calls += 1
        return np.full(8, -.1 * self.calls), dict(recomputed=True)


class FK:
    def preview(self, actions, proprio):
        return dict(trajectory=[dict(position=[float(a[0]), 0, 0], quaternion_wxyz=[1, 0, 0, 0],
                                     gripper_closed=bool(a[7] > .5)) for a in actions],
                    measured_fk_check=dict(position_error_m=0., rotation_error_rad=0.), physical_steps=0)


def make_client(tmp_path, choices, **kwargs):
    reviewer = Reviewer(choices)
    student, adapter = Student(), Adapter()
    client = OrchestratedClient(student=student, session_factory=lambda episode: reviewer,
                                execution_adapter=adapter, kinematics=FK(), output_dir=tmp_path, **kwargs)
    client.prepare_episode(0, "put the cube in the bowl")
    return client, student, reviewer, adapter


def test_prefix_takeover_and_fresh_handback_discard_old_tail(tmp_path):
    client, student, reviewer, adapter = make_client(tmp_path, [
        lambda r: response(r, steps=2), lambda r: response(r, "eef", steps=2), response,
    ])
    obs = observation()
    actions = [client.infer(obs, "put the cube in the bowl")["action"] for _ in range(5)]
    assert len(student.requests) == 3 and len(reviewer.requests) == 3
    assert student.requests == ["put the cube in the bowl"] * 3
    np.testing.assert_allclose([a[0] for a in actions], [1., 1.01, -.1, -.2, 3.])
    assert adapter.calls == 2  # Local IK is recalculated for every correction control.
    assert reviewer.requests[1]["previous_execution"]["source"] == "student"
    assert reviewer.requests[2]["previous_execution"]["source"] == "eef"
    assert [r["control_step"] for r in reviewer.requests] == [0, 2, 4]
    assert len({r["request_id"] for r in reviewer.requests}) == 3
    assert client.episode_metadata(env_id=0)["issued_control_steps"] == dict(student=3, edit=0, eef=2)
    assert client.episode_metadata(env_id=0)["observed_control_steps"] == dict(student=2, edit=0, eef=2)
    assert client.episode_metadata(env_id=0)["token_usage"]["input_tokens"] == 30
    assert not client.control_log[-1]["confirmed_by_next_observation"]
    client.reset()
    assert reviewer.closed
    assert client.episode_metadata(env_id=0)["reviews"] == 3
    client.export_diagnostics()
    saved = json.loads((tmp_path / "orchestrated_decisions_0.json").read_text())
    assert len(saved["controls"]) == 5


def test_invalid_gate_repairs_same_proposal_without_step(tmp_path):
    def invalid(r):
        value = response(r, "eef")
        value["assessment"]["intent_status"] = "uncertain"
        return value

    client, student, reviewer, adapter = make_client(tmp_path, [invalid, response])
    action = client.infer(observation(), "put the cube in the bowl")["action"]
    assert action[0] == 1. and len(student.requests) == 1 and adapter.calls == 0
    assert len(reviewer.requests) == 2
    assert reviewer.requests[0]["request_id"] == reviewer.requests[1]["request_id"]
    assert reviewer.requests[1]["control_step"] == 0
    assert "validation_error" in reviewer.requests[1]
    assert reviewer.results[0]["executed"] is False
    assert len(client.control_log) == 1


def test_stale_output_exhaustion_does_not_execute_or_retry_physics(tmp_path):
    def stale(r):
        value = response(r)
        value["request_id"] = "previous-proposal"
        return value

    client, student, reviewer, adapter = make_client(tmp_path, [stale, stale], max_output_retries=1)
    with pytest.raises(RuntimeError, match="repair budget"):
        client.infer(observation(), "put the cube in the bowl")
    assert len(student.requests) == 1 and not client.control_log and adapter.calls == 0
    with pytest.raises(RuntimeError, match="fresh episode"):
        client.infer(observation(), "put the cube in the bowl")


def test_review_budget_aborts_incomplete_without_consuming_tail(tmp_path):
    client, student, _, _ = make_client(tmp_path, [response], max_reviews=1)
    client.infer(observation(), "put the cube in the bowl")
    with pytest.raises(RuntimeError, match="incomplete"):
        client.infer(observation(), "put the cube in the bowl")
    assert len(student.requests) == 1 and len(client.control_log) == 1


@pytest.mark.parametrize("failure", [RuntimeError("transport failed"), ValueError("controller failed")])
def test_failures_do_not_generate_hold_or_replay(tmp_path, failure):
    client, _, _, adapter = make_client(tmp_path, [failure])
    with pytest.raises(type(failure), match=str(failure)):
        client.infer(observation(), "put the cube in the bowl")
    assert not client.control_log and adapter.calls == 0


def test_controller_failure_is_fatal_without_fallback(tmp_path):
    client, _, _, adapter = make_client(tmp_path, [lambda r: response(r, "eef")])
    adapter.action_for_plan = lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad Jacobian"))
    with pytest.raises(ValueError, match="Jacobian"):
        client.infer(observation(), "put the cube in the bowl")
    assert not client.control_log


def test_instruction_and_episode_boundaries(tmp_path):
    client, student, reviewer, _ = make_client(tmp_path, [response])
    client.infer(observation(), "put the cube in the bowl")
    with pytest.raises(ValueError, match="instruction cannot change"):
        client.infer(observation(), "different goal")
    fresh = Reviewer([response])
    client.session_factory = lambda episode: fresh
    client.reset()
    client.prepare_episode(1, "new task")
    client.infer(observation(), "new task")
    assert reviewer.closed and len(client.decisions) == 1
    assert fresh.requests[0]["control_step"] == 0
    assert fresh.requests[0]["previous_execution"] is None
    assert client.episode_metadata(env_id=0)["token_usage"]["input_tokens"] == 10
    client.close()
    assert student.closed and fresh.closed


def test_prompt_excludes_evaluator_state_and_keeps_upstream_two_views(tmp_path):
    client, _, reviewer, _ = make_client(tmp_path, [response])
    client.infer(observation(), "put the cube in the bowl")
    request = reviewer.requests[0]
    request["gt_state"] = "DO_NOT_SERIALIZE"
    request["native_success"] = True
    blocks = observation_blocks(request, image_size=128, jpeg_quality=85)
    payload = json.loads(blocks[0]["text"])
    assert len([b for b in blocks if b["type"] == "image"]) == 2
    assert not {"gt_state", "task_context", "remaining_steps", "native_success"} & payload.keys()
    assert "not-for-model" not in json.dumps(blocks)
    assert "DO_NOT_SERIALIZE" not in json.dumps(blocks)


@pytest.mark.parametrize("change", [
    lambda r: r.update(steps=True),
    lambda r: r.update(steps=16),
    lambda r: r.update(mode="unknown"),
    lambda r: r.update(extra="unknown"),
    lambda r: r["assessment"].update(execution_status="not_started"),
    lambda r: r["target"].update(position=[.051, 0, 0]),
    lambda r: r["target"].update(quaternion_wxyz=[0, 0, 0, 0]),
    lambda r: r["target"].update(quaternion_wxyz=[2, 0, 0, 0]),
    lambda r: r["target"].update(gripper_closed=1),
])
def test_host_rejects_invalid_review_and_eef_bounds(tmp_path, change):
    client, _, reviewer, _ = make_client(tmp_path, [response])
    client.infer(observation(), "put the cube in the bowl")
    request = reviewer.requests[0]
    request["control_step"] = 1
    value = response(request, "eef")
    change(value)
    with pytest.raises(ValueError):
        validate_review(TOOL_NAME, value, request, spec=client.spec,
                        proprio=client._extract_observation(observation())["proprio"])


def chain():
    edges = []
    for index, name in enumerate(JOINT_NAMES):
        frame = np.eye(4)
        frame[0, 3] = .1 if index == 0 else 0
        edges.append(JointEdge(name, "Z", frame, np.eye(4)))
    return edges


def test_fk_preview_matches_measurement_and_does_not_mutate_targets():
    fk = RobotFK(chain())
    proprio = dict(arm_joint_pos=np.zeros(7), ee_pos=[.1, 0, 0], ee_quat=[1, 0, 0, 0])
    actions = np.zeros((15, 8))
    actions[:, 0] = np.pi / 2
    saved = actions.copy()
    preview = fk.preview(actions, proprio)
    assert preview["physical_steps"] == 0
    np.testing.assert_allclose(preview["trajectory"][0]["position"], [.1, 0, 0])
    np.testing.assert_allclose(preview["trajectory"][0]["quaternion_wxyz"], [2**-.5, 0, 0, 2**-.5])
    np.testing.assert_array_equal(actions, saved)
    proprio["ee_pos"] = [.2, 0, 0]
    with pytest.raises(ValueError, match="disagrees"):
        fk.preview(actions, proprio)


def test_fk_handles_reverse_joints_and_local_body_frames():
    edges = chain()
    first = edges[0]
    edges[0] = JointEdge(first.name, "Z", first.local0, first.local1, reverse=True)
    fk = RobotFK(edges)
    joints = np.zeros(7)
    joints[0] = np.pi / 2
    result = fk.matrix(joints)
    expected = np.linalg.inv(transform([.1, 0, 0], [2**-.5, 0, 0, 2**-.5]))
    np.testing.assert_allclose(result, expected, atol=1e-15)


def test_usd_joint_chain_loading_without_simulator():
    pxr = pytest.importorskip("pxr")
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.CreateInMemory()
    root = "/World/envs/env_0/robot"
    UsdGeom.Xform.Define(stage, root)
    names = [root + "/panda_link0"] + [root + f"/link{i}" for i in range(1, 7)]
    names.append(root + "/Gripper/Robotiq_2F_85/base_link")
    for path in names:
        UsdGeom.Xform.Define(stage, path)
    for index, name in enumerate(JOINT_NAMES):
        joint = UsdPhysics.RevoluteJoint.Define(stage, root + "/" + name)
        joint.CreateBody0Rel().SetTargets([names[index]])
        joint.CreateBody1Rel().SetTargets([names[index + 1]])
        joint.CreateAxisAttr("Z")
        joint.CreateLocalPos0Attr(Gf.Vec3f(.1, 0, 0))
    fk = RobotFK.from_stage(stage)
    np.testing.assert_allclose(fk.matrix(np.zeros(7))[:3, 3], [.7, 0, 0], atol=1e-7)


def test_cli_rejects_unsupported_modes_before_connecting():
    from policies.ai_native.orchestrated.run_config import add_orchestrated_args, validate_args
    from robolab.eval.runner import add_common_eval_args

    parser = argparse.ArgumentParser()
    add_orchestrated_args(parser)
    add_common_eval_args(parser)
    args = parser.parse_args(["--allow-model-requests"])
    validate_args(args)
    for field, value in (("num_envs", 2), ("enable_gt_state", True), ("allow_model_requests", False),
                         ("max_reviews", 0), ("image_size", 1), ("control_hz", float("nan"))):
        invalid = copy.deepcopy(args)
        setattr(invalid, field, value)
        with pytest.raises(ValueError):
            validate_args(invalid)


def test_openpi_adapter_reuses_wire_contract_but_not_chunk_cache():
    # Exercise the bundled client/image utilities; replace only the server query.
    from policies.ai_native.orchestrated.student import Pi05ProposalClient

    student = Pi05ProposalClient.__new__(Pi05ProposalClient)
    calls = []

    def infer(request):
        calls.append(request)
        return dict(actions=np.zeros((15, 8)))

    student._query_server = infer
    import torch
    obs = observation()
    obs["image_obs"] = {k: torch.from_numpy(v) for k, v in obs["image_obs"].items()}
    obs["proprio_obs"] = {k: torch.from_numpy(v) for k, v in obs["proprio_obs"].items()}
    for _ in range(2):
        assert student.propose(obs, "unchanged goal").shape == (15, 8)
    assert len(calls) == 2
    assert calls[0]["prompt"] == "unchanged goal"
    assert set(calls[0]) == {"observation/exterior_image_1_left", "observation/wrist_image_left",
                             "observation/joint_position", "observation/gripper_position", "prompt"}


def test_review_session_uses_existing_transport_with_own_output_schema(tmp_path, monkeypatch):
    from policies.ai_native.orchestrated import session
    from policies.ai_native.agentic.specs import make_droid_spec

    servers = []

    class Server:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.last_metadata = dict(token_usage=dict(total=dict(inputTokens=8)))
            self.closed = False
            servers.append(self)

        def query(self, blocks):
            self.blocks = blocks
            return SimpleNamespace(tool_call=dict(name=TOOL_NAME, arguments=dict(request_id="id")))

        def close(self):
            self.closed = True

    monkeypatch.setattr(session, "PersistentAppServerSession", Server)
    worker = session.ProposalReviewSession(
        spec=make_droid_spec(), model_id="gpt-6-astra-medium", memory_root=tmp_path,
        episode_idx=0, allow_model_requests=True,
    )
    client, _, reviewer, _ = make_client(tmp_path, [response])
    client.infer(observation(), "put the cube in the bowl")
    name, _ = worker.query(reviewer.requests[0])
    assert name == TOOL_NAME
    assert servers[0].kwargs["output_tools"][0]["name"] == TOOL_NAME
    assert servers[0].kwargs["options"].persistent_history
    assert not servers[0].kwargs["options"].optional_tools
    assert len([b for b in servers[0].blocks if b["type"] == "image"]) == 2
    assert worker.last_metadata["token_usage"]["total"]["inputTokens"] == 8
    worker.close()
    assert servers[0].closed


@pytest.mark.parametrize("physics_error", [False, True])
def test_episode_wrapper_preserves_native_result_and_exports_after_cleanup(tmp_path, monkeypatch, physics_error):
    from policies.ai_native.orchestrated.episode import run_orchestrated_episode

    client, _, reviewer, adapter = make_client(tmp_path, [response])
    client.runtime_settings = {}
    adapter.max_translation_step_m, adapter.max_rotation_step_rad = .02, .1
    adapter.max_joint_delta_rad, adapter.damping = .05, .05
    bindings = []
    client.bind_environment = bindings.append
    env = object()
    native = [dict(env_id=0, success=False, step=1)]

    def shared_episode(env, cfg, episode, actual_client, **options):
        assert actual_client is client
        assert not options.get("enable_gt_state")
        try:
            client.infer(observation(), "put the cube in the bowl")
            if physics_error:
                raise RuntimeError("physics error after issued action")
            return native, ["native messages"], dict(native_timing=True)
        finally:
            client.reset()

    module = ModuleType("robolab.eval.episode")
    module.run_episode = shared_episode
    monkeypatch.setitem(sys.modules, "robolab.eval.episode", module)
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=1 / 120), decimation=8, episode_length_s=40)
    if physics_error:
        with pytest.raises(RuntimeError, match="physics error"):
            run_orchestrated_episode(env, cfg, 0, client, enable_gt_state=False)
    else:
        results, messages, timing = run_orchestrated_episode(env, cfg, 0, client, enable_gt_state=False)
        assert results is native and results[0]["success"] is False
        assert results[0]["episode_metadata"]["reviews"] == 1
        assert messages == ["native messages"] and timing["native_timing"]
    assert bindings == [env] and reviewer.closed
    assert (tmp_path / "orchestrated_config.json").is_file()
    saved = json.loads((tmp_path / "orchestrated_decisions_0.json").read_text())
    assert len(saved["controls"]) == 1
    assert saved["metadata"]["observed_control_steps"] == dict(student=0, edit=0, eef=0)


def test_configuration_rejects_wrong_timing(tmp_path):
    from policies.ai_native.orchestrated.episode import configure_environment

    client, _, _, _ = make_client(tmp_path, [response])
    client.runtime_settings = {}
    with pytest.raises(ValueError, match="control rate"):
        configure_environment(client, SimpleNamespace(sim=SimpleNamespace(dt=.1), decimation=1))


@pytest.mark.parametrize("budget", [None, 1])
def test_incomplete_episode_is_saved_and_next_run_can_proceed(tmp_path, monkeypatch, budget):
    from policies.ai_native.orchestrated.episode import run_orchestrated_episode

    choices = ([response, lambda r: response(r, mode="stop"), response]
               if budget is None else [response, response])
    client, _, _, adapter = make_client(tmp_path, choices, max_reviews=budget)
    client.runtime_settings = {}
    adapter.max_translation_step_m, adapter.max_rotation_step_rad = .02, .1
    adapter.max_joint_delta_rad, adapter.damping = .05, .05
    client.bind_environment = lambda env: None
    exports = []
    env = SimpleNamespace(
        recorder_manager=SimpleNamespace(export_episodes=lambda **kw: exports.append(kw)),
        get_env_results=lambda: [dict(env_id=0, success=None, step=None)],
    )

    def shared_episode(env, cfg, episode, actual_client, **options):
        actual_client.prepare_episode(episode, "put the cube in the bowl")
        try:
            actual_client.infer(observation(), "put the cube in the bowl")
            if episode == 0:
                actual_client.infer(observation(), "put the cube in the bowl")
            return [dict(env_id=0, success=False, step=1)], [], {}
        finally:
            actual_client.reset()

    module = ModuleType("robolab.eval.episode")
    module.run_episode = shared_episode
    monkeypatch.setitem(sys.modules, "robolab.eval.episode", module)
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=1 / 120), decimation=8, episode_length_s=240)
    first, _, _ = run_orchestrated_episode(env, cfg, 0, client)
    assert first[0]["success"] is None and first[0]["step"] == 1
    metadata = first[0]["episode_metadata"]
    assert metadata["episode_status"] == "incomplete"
    assert metadata["stop_reason"] == ("model_stop" if budget is None else "review_budget_exhausted")
    assert exports == [dict(env_ids=[0])]
    assert client.max_reviews == (3600 if budget is None else 1)
    from robolab.eval.summarize import build_run_summary
    from robolab.eval.run_utils import is_run_complete

    summary_cfg = SimpleNamespace(instruction="put the cube in the bowl",
                                  _task_name="synthetic", _task_attributes=[])
    row = build_run_summary(
        env_result=first[0], env_id=0, run_idx=0, num_envs=1, run_name="synthetic_0",
        task_env="synthetic", env_cfg=summary_cfg, policy="orchestrated", dt=1 / 15,
        traj_metrics=None, events={}, events_list=[], final_info=None, enable_subtask_progress=False,
    )
    assert row["success"] is None and row["episode_status"] == "incomplete"
    assert is_run_complete([row], "synthetic", 0, 1)
    second, _, _ = run_orchestrated_episode(env, cfg, 1, client)
    assert second[0]["success"] is False
    assert second[0]["episode_metadata"]["episode_status"] == "finished"
    assert second[0]["episode_metadata"]["stop_reason"] is None


def test_default_budget_allows_every_step_of_a_long_task(tmp_path):
    from policies.ai_native.orchestrated.episode import configure_environment

    client, _, _, adapter = make_client(tmp_path, [])
    client.runtime_settings = {}
    adapter.max_translation_step_m, adapter.max_rotation_step_rad = .02, .1
    adapter.max_joint_delta_rad, adapter.damping = .05, .05
    cfg = SimpleNamespace(sim=SimpleNamespace(dt=1 / 120), decimation=8, episode_length_s=240)
    configure_environment(client, cfg)
    assert client.max_reviews == 3600


def test_env_zero_reset_releases_once_only_context(tmp_path):
    client, _, reviewer, _ = make_client(tmp_path, [response])
    client.infer(observation(), "put the cube in the bowl")
    client.reset(env_id=0)
    client.prepare_episode(1, "new task")
    assert reviewer.closed and client._active
    assert client._prepared_episode_idx == 1


def test_original_edit_ramps_offsets_in_root_frame_and_overrides_gripper_immediately():
    from scipy.spatial.transform import Rotation
    from policies.ai_native.orchestrated.kinematics import edited_targets

    initial = Rotation.from_euler('x', 90, degrees=True)
    trajectory = [dict(position=[.2, .1, .3], quaternion_wxyz=initial.as_quat()[[3, 0, 1, 2]],
                       gripper_closed=True)] * 2
    targets = edited_targets(trajectory, 2, dict(delta_position=[.04, 0, 0],
                                               delta_rotation_vector=[0, 0, .2], gripper='open'))
    for index, target in enumerate(targets):
        alpha = (index + 1) / 2
        np.testing.assert_allclose(target['position'], [.2 + alpha * .04, .1, .3])
        actual = Rotation.from_quat(np.asarray(target['quaternion_wxyz'])[[1, 2, 3, 0]])
        expected = Rotation.from_rotvec([0, 0, alpha * .2]) * initial
        assert (actual * expected.inv()).magnitude() < 1e-12
        assert target['gripper_closed'] is False


def test_edit_executes_proposal_relative_targets_then_discards_old_tail(tmp_path):
    def edit(request):
        result = response(request, mode='edit', steps=2)
        result['assessment']['intent_status'] = 'misaligned'
        result['edit'] = dict(delta_position=[.04, 0, 0], delta_rotation_vector=[0, 0, 0], gripper='open')
        return result

    client, student, _, adapter = make_client(tmp_path, [edit, response])
    targets = []

    def control(plan, *, env_id):
        state = plan.commanded_state['arm']
        targets.append(state)
        return np.zeros(8), {}

    adapter.action_for_plan = control
    first = client.infer(observation(), 'put the cube in the bowl')
    second = client.infer(observation(), 'put the cube in the bowl')
    handback = client.infer(observation(), 'put the cube in the bowl')
    # FK targets can be farther than 5 cm from the current pose. The original
    # edit limit applies to the offset, while bounded local IK handles tracking.
    np.testing.assert_allclose([state.position_m[0] for state in targets], [1.02, 1.05])
    assert all(not state.gripper_closed for state in targets)
    assert len(student.requests) == 2 and handback['action'][0] == 2.
    assert [row['source'] for row in client.control_log] == ['edit', 'edit', 'student']
    assert client.episode_metadata(env_id=0)['issued_control_steps']['edit'] == 2


@pytest.mark.parametrize('mode', ['edit', 'eef'])
def test_original_gate_requires_failure_or_wrong_intent_for_both_corrections(tmp_path, mode):
    client, _, reviewer, _ = make_client(tmp_path, [response])
    client.infer(observation(), 'put the cube in the bowl')
    request = reviewer.requests[0]
    value = response(request, mode=mode)
    value['assessment']['intent_status'] = 'uncertain'
    with pytest.raises(ValueError, match='Takeover requires'):
        validate_review(TOOL_NAME, value, request, spec=client.spec,
                        proprio=client._extract_observation(observation())['proprio'])


def test_model_stop_preserves_incomplete_artifacts_without_issuing_action(tmp_path):
    client, student, reviewer, _ = make_client(tmp_path, [lambda r: response(r, mode='stop')])
    with pytest.raises(RuntimeError, match='model requested stop.*incomplete'):
        client.infer(observation(), 'put the cube in the bowl')
    assert len(student.requests) == 1 and not client.control_log
    assert client.decisions[0]['source'] == 'stop'
    assert reviewer.results[-1] == dict(status='model_stop', executed=False)
    client.reset()
    client.export_diagnostics()
    saved = json.loads((tmp_path / 'orchestrated_decisions_0.json').read_text())
    assert saved['decisions'][0]['response']['mode'] == 'stop'
    assert sum(saved['metadata']['issued_control_steps'].values()) == 0


def test_review_uses_original_gate_text_schema_and_camera_pair():
    from policies.ai_native.orchestrated.gate_assessment import GATE_INSTRUCTION
    from policies.ai_native.orchestrated.schema import response_schema
    from policies.ai_native.orchestrated.session import reviewer_prompt
    from policies.ai_native.orchestrated.protocol import review_tool
    from policies.ai_native.agentic.specs import make_droid_spec

    assert GATE_INSTRUCTION in reviewer_prompt()
    assert review_tool(make_droid_spec())['function']['parameters'] == response_schema()
    assert response_schema()['properties']['mode']['enum'] == ['student', 'edit', 'eef', 'stop']
    assert CAMERA_NAMES == ('over_shoulder_left_camera', 'wrist_cam')


@pytest.mark.parametrize('changes', [
    dict(edit=dict(delta_position=[.051, 0, 0], delta_rotation_vector=[0, 0, 0], gripper='keep')),
    dict(edit=dict(delta_position=[0, 0, 0], delta_rotation_vector=[0, 0, .351], gripper='keep')),
    dict(edit=dict(delta_position=[0, 0, 0], delta_rotation_vector=[0, 0, 0], gripper='unknown')),
    dict(steps=6),
])
def test_edit_limit_rejection_executes_nothing(tmp_path, changes):
    def invalid(req):
        value = response(req, mode='edit')
        value['assessment']['intent_status'] = 'misaligned'
        value.update(changes)
        return value

    client, _, _, adapter = make_client(tmp_path, [invalid], max_output_retries=0)
    with pytest.raises(RuntimeError, match='repair budget'):
        client.infer(observation(), 'put the cube in the bowl')
    assert not client.control_log and adapter.calls == 0
