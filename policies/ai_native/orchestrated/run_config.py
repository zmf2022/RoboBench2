"""CLI configuration and construction, without simulator startup."""

import argparse
import importlib.util
import math
import os
import shutil

from policies.ai_native.agentic.app_server import AGENTIC_MODEL_IDS, resolve_agent_memory_root


def add_orchestrated_args(parser):
    parser.add_argument("--review-backend", choices=("codex", "api"), default="codex",
                        help="GPT reviewer transport; default preserves the Codex implementation.")
    parser.add_argument("--remote-host", default="localhost")
    parser.add_argument("--remote-port", type=int, default=8000)
    parser.add_argument("--remote-uri", default=None)
    parser.add_argument("--model", default=None,
                        help="API: model ID (AI_NATIVE_MODEL or gpt-6-astra). Codex: model-effort ID.")
    parser.add_argument("--allow-model-requests", action="store_true")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--base-url", default=None,
                        help="API base URL; defaults to OPENAI_BASE_URL, then the OpenAI API.")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high", "xhigh", "max", "none"),
                        default="medium", help="API reasoning effort; none omits the field.")
    parser.add_argument("--max-completion-tokens", type=int, default=8192, help="API completion budget per review.")
    parser.add_argument("--timeout-s", dest="decision_timeout", type=float, default=120.)
    parser.add_argument("--max-reviews", type=int, default=None,
                        help="Review budget per episode; defaults to the task's maximum control steps.")
    parser.add_argument("--max-output-retries", type=int, default=2)
    parser.add_argument("--control-hz", type=float, default=15.)
    parser.add_argument("--image-max-side", dest="image_size", type=int, default=896,
                        help="Maximum model image side; preserve the upstream 896x504 views by default.")
    parser.add_argument("--image-detail", choices=("low", "high", "auto"), default="high")
    parser.add_argument("--jpeg-quality", type=int, default=85)
    parser.add_argument("--optional-tools", action=argparse.BooleanOptionalAction, default=False)


def validate_args(args):
    if args.review_backend == "api":
        args.model = args.model or os.environ.get("AI_NATIVE_MODEL") or "gpt-6-astra"
        args.base_url = args.base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
    else:
        args.model = args.model or "gpt-6-astra-medium"
        if args.model not in AGENTIC_MODEL_IDS:
            raise ValueError("Codex requires a supported model-effort ID")
    if args.num_envs != 1:
        raise ValueError("Orchestrated currently requires --num-envs 1")
    if args.enable_gt_state:
        raise ValueError("Orchestrated does not accept --enable-gt-state")
    if not args.allow_model_requests:
        raise ValueError("Real reviews require --allow-model-requests")
    for field in ("control_hz", "decision_timeout"):
        value = getattr(args, field)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{field} must be finite and positive")
    if args.decision_timeout > 900:
        raise ValueError("decision_timeout must be at most 900 seconds")
    if (args.max_reviews is not None and args.max_reviews < 1) or args.max_output_retries < 0:
        raise ValueError("Invalid review or output-repair budget")
    if not 128 <= args.image_size <= 2048 or not 1 <= args.jpeg_quality <= 100:
        raise ValueError("Invalid image size or JPEG quality")
    if not 1 <= args.remote_port <= 65535:
        raise ValueError("Invalid student server port")
    if args.review_backend == "api":
        if args.optional_tools:
            raise ValueError("--optional-tools is available only with --review-backend codex")
        for field in ("model", "base_url", "api_key_env"):
            if not isinstance(getattr(args, field), str) or not getattr(args, field).strip():
                raise ValueError(f"{field} must be nonempty")
        if args.max_completion_tokens < 1:
            raise ValueError("max_completion_tokens must be positive")


def build_session_factory(args, *, spec, memory_root, memory_root_source):
    validate_args(args)
    shared = dict(spec=spec, memory_root=memory_root, memory_root_source=memory_root_source,
                  allow_model_requests=args.allow_model_requests, timeout=args.decision_timeout,
                  image_size=args.image_size, jpeg_quality=args.jpeg_quality)
    if args.review_backend == "api":
        from policies.ai_native.orchestrated.api_session import APIProposalReviewSession

        return lambda episode: APIProposalReviewSession(
            **shared, episode_idx=episode, model_id=args.model, base_url=args.base_url,
            api_key_env=args.api_key_env,
            image_detail=args.image_detail,
            reasoning_effort=None if args.reasoning_effort == "none" else args.reasoning_effort,
            max_completion_tokens=args.max_completion_tokens,
        )
    from policies.ai_native.orchestrated.session import ProposalReviewSession

    return lambda episode: ProposalReviewSession(
        **shared, episode_idx=episode, model_id=args.model, codex_bin=args.codex_bin,
        optional_tools=args.optional_tools,
    )


def build_client(args):
    # Delay OpenPI imports until an authorized evaluation actually constructs a client.
    from policies.ai_native.agentic.droid_joint_target import DroidJointTargetAdapter
    from policies.ai_native.agentic.specs import make_droid_spec
    from policies.ai_native.orchestrated.client import OrchestratedClient
    from policies.ai_native.orchestrated.kinematics import DroidFKPreview
    from policies.ai_native.orchestrated.student import Pi05ProposalClient
    from robolab.constants import get_output_dir

    validate_args(args)
    root, root_source = resolve_agent_memory_root(default_output_dir=get_output_dir())
    student = Pi05ProposalClient(remote_host=args.remote_host, remote_port=args.remote_port,
                                remote_uri=args.remote_uri, policy_variant="pi05")
    client = OrchestratedClient(
        student=student, session_factory=build_session_factory(
            args, spec=make_droid_spec(), memory_root=root, memory_root_source=root_source,
        ),
        execution_adapter=DroidJointTargetAdapter(), kinematics=DroidFKPreview(),
        output_dir=get_output_dir(), control_hz=args.control_hz, max_reviews=args.max_reviews,
        max_output_retries=args.max_output_retries,
    )
    client.runtime_settings = {key: getattr(args, key) for key in (
        "review_backend", "model", "remote_host", "remote_port", "remote_uri", "decision_timeout", "max_reviews",
        "max_output_retries", "control_hz", "image_size", "jpeg_quality", "optional_tools",
    )}
    if args.review_backend == "api":
        client.runtime_settings.update({key: getattr(args, key) for key in (
            "base_url", "api_key_env", "reasoning_effort", "max_completion_tokens", "image_detail",
        )})
    return client


def check_runtime_dependencies(args):
    """Report missing external runtimes before starting the simulator."""
    if importlib.util.find_spec("openpi_client") is None:
        raise RuntimeError("Missing openpi-client in the RoboLab environment; follow policies/pi0_family/README.md")
    if args.review_backend == "api":
        if not os.environ.get(args.api_key_env, "").strip():
            raise RuntimeError(f"API key is missing; set {args.api_key_env}")
    elif shutil.which(args.codex_bin) is None:
        raise RuntimeError(f"Codex executable not found: {args.codex_bin}")


def load_api_environment(args, repo_root):
    """Match Direct-API's .env loading; shell environment values take precedence."""
    if args.review_backend == "api":
        from dotenv import load_dotenv

        load_dotenv(repo_root / ".env", override=False)
