"""Resolve the served model before naming or starting an API evaluation."""

import logging
import os


def list_available_models(*, api_key=None, api_key_env="OPENAI_API_KEY", base_url=None,
                          timeout_s=30., client=None):
    """Query model identity once for both API backends and the listing command."""
    from openai import OpenAI

    key = api_key or os.environ.get(api_key_env, "").strip()
    if not key:
        raise RuntimeError(f"API key is missing; set {api_key_env}")
    sdk = client or OpenAI(api_key=key, base_url=base_url or os.environ.get("OPENAI_BASE_URL")
                           or "https://api.openai.com/v1", timeout=timeout_s, max_retries=0)
    try:
        return sorted({entry.id for entry in sdk.models.list()
                       if isinstance(entry.id, str) and entry.id.strip()})
    finally:
        if client is None:
            sdk.close()


def resolve_api_model(model, *, base_url=None, api_key_env="OPENAI_API_KEY", timeout=120.):
    models = list_available_models(base_url=base_url, api_key_env=api_key_env, timeout_s=timeout)
    if len(models) == 1:
        if model != models[0]:
            logging.warning("Using served API model %s instead of configured model %s", models[0], model)
        return models[0]
    if model in models:
        return model
    raise ValueError("API model discovery requires one served model or --model matching /v1/models; "
                     f"received {models}")
