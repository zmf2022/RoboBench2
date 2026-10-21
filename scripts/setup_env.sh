#!/usr/bin/env bash
# Install a project-local RoboLab environment. Does not launch simulation or inference.
set -euo pipefail

usage() {
    echo "Usage: $0"
    echo "Install Isaac Sim 6.0.1 / Isaac Lab 3.0 beta with Python 3.12 in .venv."
}
if (( $# == 1 )) && [[ "$1" == "--help" || "$1" == "-h" ]]; then
    usage
    exit 0
fi
if (( $# != 0 )); then
    usage >&2
    exit 2
fi
command -v uv >/dev/null 2>&1 || { echo "Install uv before running this script." >&2; exit 1; }
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$project_root"
UV_PROJECT_ENVIRONMENT="$project_root/.venv" uv sync --python 3.12 --extra isaac60
printf '\nEnvironment installed. Activate with:\n  source %q/bin/activate\n' "$project_root/.venv"
