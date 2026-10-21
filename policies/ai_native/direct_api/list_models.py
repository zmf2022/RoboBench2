# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""List models visible to the configured Direct-API key."""

import argparse

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policies.ai_native.direct_api.client import DEFAULT_API_KEY_ENV, list_available_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-key-env", default=DEFAULT_API_KEY_ENV)
    parser.add_argument("--base-url", default=None)
    parser.add_argument(
        "--contains",
        default=None,
        help="Only print model IDs containing this case-insensitive text.",
    )
    args = parser.parse_args()

    models = list_available_models(api_key_env=args.api_key_env, base_url=args.base_url)
    if args.contains:
        needle = args.contains.casefold()
        models = [model for model in models if needle in model.casefold()]
    for model in models:
        print(model)


if __name__ == "__main__":
    main()
