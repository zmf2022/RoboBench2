"""Direct API contract tests using the real SDK with a mock HTTP transport."""

import argparse
import copy
import json

import httpx
import pytest
from openai import APIStatusError, OpenAI

from policies.ai_native.agentic.specs import make_droid_spec
from policies.ai_native.orchestrated.api_session import APIProposalReviewSession
from policies.ai_native.orchestrated.protocol import TOOL_NAME
from policies.ai_native.orchestrated.tests.test_orchestrated import make_client, observation, response


@pytest.fixture
def transport():
    clients = []

    def make(outputs):
        outputs = iter(outputs)
        captured = []

        def handler(request):
            assert request.url.path == "/v1/chat/completions"
            payload = json.loads(request.content)
            captured.append(payload)
            item = next(outputs)
            if isinstance(item, int):
                return httpx.Response(item, json=dict(error=dict(message="Synthetic provider failure")))
            latest = json.loads(payload["messages"][-1]["content"][0]["text"])
            value = item(latest) if callable(item) else copy.deepcopy(item)
            if "choices" in value:
                packet = value
            else:
                packet = dict(choices=[dict(index=0, finish_reason="tool_calls", message=dict(
                    role="assistant", tool_calls=[dict(id=f"call_{len(captured)}", type="function",
                        function=dict(name=TOOL_NAME, arguments=json.dumps(value)))], content=None))])
            packet.update(id=f"completion_{len(captured)}", object="chat.completion", created=0, model="test-model")
            packet.setdefault("usage", dict(prompt_tokens=10, completion_tokens=3, total_tokens=13,
                                            prompt_tokens_details=dict(cached_tokens=2),
                                            completion_tokens_details=dict(reasoning_tokens=1)))
            return httpx.Response(200, json=packet)

        sdk = OpenAI(api_key="synthetic-test-key", base_url="https://api.example.test/v1", max_retries=0,
                     http_client=httpx.Client(transport=httpx.MockTransport(handler)))
        clients.append(sdk)
        return sdk, captured

    yield make
    for sdk in clients:
        sdk.close()


def worker(tmp_path, sdk, **kwargs):
    return APIProposalReviewSession(spec=make_droid_spec(), model_id="test-model", memory_root=tmp_path,
                                   episode_idx=0, allow_model_requests=True, sdk=sdk, **kwargs)


def request(tmp_path):
    client, _, reviewer, _ = make_client(tmp_path, [response])
    client.infer(observation(), "put the cube in the bowl")
    return reviewer.requests[0]


def test_real_sdk_wire_format_and_full_episode_history(tmp_path, transport):
    sdk, captured = transport([response, response])
    session = worker(tmp_path, sdk, reasoning_effort="high", max_completion_tokens=2048)
    req = request(tmp_path)
    assert session.query(req)[0] == TOOL_NAME
    session.record_host_result(dict(status="accepted", executed=False))
    session.query(req)
    first, second = captured
    assert first["model"] == "test-model" and first["reasoning_effort"] == "high"
    assert first["max_completion_tokens"] == 2048
    assert first["tool_choice"] == "required" and first["parallel_tool_calls"] is False
    assert first["tools"][0]["function"]["name"] == TOOL_NAME
    assert first["tools"][0]["function"]["parameters"]["properties"]["mode"]["enum"] == [
        "student", "edit", "eef", "stop"]
    images = [part for part in first["messages"][-1]["content"] if part["type"] == "image_url"]
    assert len(images) == 2
    assert all(image["image_url"]["url"].startswith("data:image/jpeg;base64,") for image in images)
    assert all(image["image_url"]["detail"] == "high" for image in images)
    assert [message["role"] for message in second["messages"]] == ["system", "user", "assistant", "tool", "user"]
    assert second["messages"][3]["tool_call_id"] == "call_1"
    assert session.last_metadata["usage_is_cumulative"] is False
    assert session.last_metadata["token_usage"]["cached_input_tokens"] == 2
    assert session.last_metadata["token_usage"]["reasoning_tokens"] == 1
    session.close()
    assert not sdk.is_closed()  # Injected transports remain caller-owned.
    for path in session.workspace.rglob("*.json*"):
        assert "synthetic-test-key" not in path.read_text()


def test_api_gate_repair_takeover_and_handback_share_client_state_machine(tmp_path, transport):
    def uncertainty(req):
        result = response(req, "eef")
        result["assessment"]["intent_status"] = "uncertain"
        return result

    sdk, captured = transport([uncertainty, lambda r: response(r, steps=2),
                               lambda r: response(r, "eef"), response])
    session = worker(tmp_path, sdk)
    client, student, _, adapter = make_client(tmp_path, [])
    client.session_factory = lambda episode: session
    client.runtime_settings = dict(review_backend="api")
    actions = [client.infer(observation(), "put the cube in the bowl")["action"][0] for _ in range(4)]
    assert actions == pytest.approx([1., 1.01, -.1, 3.])
    assert len(student.requests) == 3 and len(captured) == 4 and adapter.calls == 1
    second_input = json.loads(captured[1]["messages"][-1]["content"][0]["text"])
    first_input = json.loads(captured[0]["messages"][-1]["content"][0]["text"])
    assert second_input["request_id"] == first_input["request_id"]
    assert "validation_error" in second_input
    metadata = client.episode_metadata(env_id=0)
    assert metadata["review_backend"] == "api"
    assert metadata["token_usage"]["input_tokens"] == 40  # Per-request totals must be summed.
    assert metadata["token_usage"]["total_tokens"] == 52
    assert metadata["token_usage"]["model_requests"] == 4
    client.reset()
    assert session._closed


@pytest.mark.parametrize("bad_message,finish_reason", [
    (dict(role="assistant", content="No tool"), "stop"),
    (dict(role="assistant", content=None, refusal="Synthetic refusal"), "stop"),
    (dict(role="assistant", content=None, tool_calls=[dict(id="bad", type="function",
        function=dict(name=TOOL_NAME, arguments="{bad json"))]), "tool_calls"),
    (dict(role="assistant", content=None, tool_calls=[dict(id="bad", type="function",
        function=dict(name=TOOL_NAME, arguments="[]"))]), "tool_calls"),
    (dict(role="assistant", content=None, tool_calls=[dict(id="bad", type="function",
        function=dict(name=TOOL_NAME, arguments="{}"))]), "length"),
])
def test_malformed_refused_or_truncated_output_uses_host_repair(tmp_path, transport, bad_message, finish_reason):
    sdk, captured = transport([dict(choices=[dict(index=0, message=bad_message, finish_reason=finish_reason)]), response])
    session = worker(tmp_path, sdk)
    client, student, _, _ = make_client(tmp_path, [])
    client.session_factory = lambda episode: session
    action = client.infer(observation(), "put the cube in the bowl")["action"]
    assert action[0] == 1. and len(student.requests) == 1
    assert len(captured) == 2 and len(client.control_log) == 1
    assert len(client.decisions[0]["attempts"]) == 1
    if bad_message.get("tool_calls"):
        receipts = [msg for msg in captured[1]["messages"] if msg["role"] == "tool"]
        assert len(receipts) == 1 and receipts[0]["tool_call_id"] == "bad"


def test_multiple_calls_are_rejected_and_all_receive_receipts(tmp_path, transport):
    calls = [dict(id=f"bad_{i}", type="function", function=dict(name=TOOL_NAME, arguments="{}")) for i in range(2)]
    sdk, captured = transport([dict(choices=[dict(index=0, finish_reason="tool_calls",
                                                message=dict(role="assistant", tool_calls=calls))]), response])
    session = worker(tmp_path, sdk)
    client, _, _, _ = make_client(tmp_path, [])
    client.session_factory = lambda episode: session
    client.infer(observation(), "put the cube in the bowl")
    assert [msg["tool_call_id"] for msg in captured[1]["messages"] if msg["role"] == "tool"] == ["bad_0", "bad_1"]
    assert len(client.control_log) == 1


def test_api_failure_is_fatal_and_never_retries_or_replays_an_action(tmp_path, transport):
    sdk, captured = transport([503])
    session = worker(tmp_path, sdk)
    client, _, _, _ = make_client(tmp_path, [])
    client.session_factory = lambda episode: session
    with pytest.raises(APIStatusError):
        client.infer(observation(), "put the cube in the bowl")
    assert len(captured) == 1 and not client.control_log
    assert client.episode_metadata(env_id=0)["token_usage"]["requests_missing_usage"] == 1
    with pytest.raises(RuntimeError, match="failed"):
        session.query(request(tmp_path))


def test_api_none_reasoning_omits_field_and_requires_host_feedback(tmp_path, transport):
    sdk, captured = transport([response])
    session = worker(tmp_path, sdk, reasoning_effort=None)
    req = request(tmp_path)
    session.query(req)
    assert "reasoning_effort" not in captured[0]
    with pytest.raises(RuntimeError, match="Host must validate"):
        session.query(req)
    assert len(captured) == 1


def args_for(*extra):
    from policies.ai_native.orchestrated.run_config import add_orchestrated_args
    from robolab.eval.runner import add_common_eval_args

    parser = argparse.ArgumentParser()
    add_orchestrated_args(parser)
    add_common_eval_args(parser)
    return parser.parse_args(["--allow-model-requests", *extra])


def test_api_cli_and_dependency_checks_do_not_require_codex(monkeypatch):
    from policies.ai_native.orchestrated import run_config

    args = args_for("--review-backend", "api", "--model", "provider/custom-gpt6")
    run_config.validate_args(args)
    monkeypatch.setattr(run_config.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(run_config.shutil, "which", lambda name: pytest.fail("API backend looked for Codex"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    run_config.check_runtime_dependencies(args)
    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        run_config.check_runtime_dependencies(args)
    args.optional_tools = True
    with pytest.raises(ValueError, match="optional-tools"):
        run_config.validate_args(args)


@pytest.mark.parametrize("backend", ["codex", "api"])
def test_session_factory_selects_backend_and_preserves_settings(tmp_path, monkeypatch, backend):
    from policies.ai_native.orchestrated import api_session, run_config, session

    captured = []
    monkeypatch.setattr(api_session, "APIProposalReviewSession", lambda **kw: captured.append(("api", kw)))
    monkeypatch.setattr(session, "ProposalReviewSession", lambda **kw: captured.append(("codex", kw)))
    args = args_for("--review-backend", backend)
    factory = run_config.build_session_factory(args, spec=make_droid_spec(), memory_root=tmp_path,
                                               memory_root_source="test")
    factory(3)
    assert captured[0][0] == backend and captured[0][1]["episode_idx"] == 3
    assert captured[0][1]["allow_model_requests"]
    if backend == "api":
        assert captured[0][1]["model_id"] == args.model
        assert captured[0][1]["api_key_env"] == "OPENAI_API_KEY"
        assert "codex_bin" not in captured[0][1]
    else:
        assert captured[0][1]["model_id"] == args.model


def test_owned_sdk_uses_env_auth_no_hidden_retries_and_is_closed(tmp_path, monkeypatch):
    from policies.ai_native.orchestrated import api_session

    captured = []
    fake_sdk = type("SDK", (), {"closed": False, "close": lambda self: setattr(self, "closed", True)})()
    monkeypatch.setenv("CUSTOM_REVIEW_KEY", "synthetic-private-key")
    monkeypatch.setattr(api_session, "OpenAI", lambda **kwargs: (captured.append(kwargs), fake_sdk)[1])
    session = APIProposalReviewSession(spec=make_droid_spec(), model_id="gpt-6-astra", memory_root=tmp_path,
        episode_idx=0, allow_model_requests=True, api_key_env="CUSTOM_REVIEW_KEY", base_url="https://api.example.test/v1")
    assert captured[0]["max_retries"] == 0
    assert captured[0]["api_key"] == "synthetic-private-key"
    session.close()
    assert fake_sdk.closed
    assert "synthetic-private-key" not in "".join(p.read_text() for p in tmp_path.rglob("*.json*"))


def test_direct_api_style_configuration_precedence_and_defaults(tmp_path, monkeypatch):
    from policies.ai_native.orchestrated.run_config import load_api_environment, validate_args

    for name in ("OPENAI_BASE_URL", "AI_NATIVE_MODEL", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "OPENAI_BASE_URL=http://127.0.0.1:8080/v1\nAI_NATIVE_MODEL=local-vision-model\nOPENAI_API_KEY=local\n")
    args = args_for("--review-backend", "api")
    load_api_environment(args, tmp_path)
    validate_args(args)
    assert args.model == "local-vision-model" and args.base_url == "http://127.0.0.1:8080/v1"
    assert (args.image_size, args.jpeg_quality, args.image_detail) == (896, 85, "high")
    monkeypatch.setenv("AI_NATIVE_MODEL", "shell-model")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:8081/v1")
    from_shell = args_for("--review-backend", "api")
    load_api_environment(from_shell, tmp_path)
    validate_args(from_shell)
    assert from_shell.model == "shell-model" and from_shell.base_url.endswith("8081/v1")
    explicit = args_for("--review-backend", "api", "--model", "explicit-model", "--base-url",
                        "http://127.0.0.1:8082/v1", "--timeout-s", "30", "--image-max-side", "512")
    load_api_environment(explicit, tmp_path)
    validate_args(explicit)
    assert explicit.model == "explicit-model" and explicit.base_url.endswith("8082/v1")
    assert explicit.decision_timeout == 30 and explicit.image_size == 512


def test_codex_configuration_does_not_inherit_api_env(tmp_path, monkeypatch):
    from policies.ai_native.orchestrated.run_config import load_api_environment, validate_args

    monkeypatch.setenv("AI_NATIVE_MODEL", "local-vision-model")
    (tmp_path / ".env").write_text("AI_NATIVE_MODEL=another-api-model\n")
    args = args_for()
    load_api_environment(args, tmp_path)
    validate_args(args)
    assert args.model == "gpt-6-astra-medium"
    assert (args.image_size, args.jpeg_quality) == (896, 85)
    with pytest.raises(ValueError, match="model-effort"):
        validate_args(args_for("--model", "local-vision-model"))


@pytest.mark.parametrize('mode', ['edit', 'stop'])
def test_api_transport_preserves_original_edit_and_stop_modes(tmp_path, transport, mode):
    def choose(req):
        result = response(req, mode=mode)
        if mode == 'edit':
            result['assessment']['intent_status'] = 'misaligned'
        return result

    sdk, captured = transport([choose])
    session = worker(tmp_path, sdk)
    client, _, _, adapter = make_client(tmp_path, [])
    client.session_factory = lambda episode: session
    if mode == 'stop':
        with pytest.raises(RuntimeError, match='model requested stop'):
            client.infer(observation(), 'put the cube in the bowl')
        assert not client.control_log and adapter.calls == 0
    else:
        client.infer(observation(), 'put the cube in the bowl')
        assert client.control_log[0]['source'] == 'edit' and adapter.calls == 1
    assert len(captured) == 1
    assert client.decisions[0]['source'] == mode
