# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Discoverable analysis workspace for the tools-enabled diagnostic."""

import json
import sys
from pathlib import Path

CODING_GUIDANCE = """Use your normal file, image, shell/Python tools to inspect observations,
write and run calculations, crop images, calculate poses, and maintain notes
when useful. You are an autonomous robot agent, not a JSON-only reviewer.
The robot-action tool is additional to your analysis tools, not your entire toolset.

Read workspace.json for the Python interpreter and exact observation/image paths.
Keep scripts, crops and calculations in scratch/ and confirmed progress, geometry
estimates, tracking errors and corrections in NOTES.md. Distinguish estimates from
measured state. When alignment or orientation is uncertain, consider inspecting a
crop or calculating a transform rather than relying solely on a visual guess.
Tool use is not mandatory for every action.

Only use supplied observations and same-episode action evidence. Do not inspect
simulator internals, hidden object states, or open another simulator connection.
Do not edit recorded observations, action logs, or executing source files.
Analysis tools do not move the robot. After analysis, finish the turn with exactly
one robot-action proposal; the host executes it and supplies the next observation.
"""


def prepare_workspace(workspace):
    workspace = Path(workspace)
    (workspace / "scratch").mkdir(exist_ok=True)
    data = {
        "python_executable": sys.executable,
        "observations_path": str(workspace / "observations"),
        "observation_index": str(workspace / "observation_index.json"),
        "action_history": str(workspace / "history.jsonl"),
        "notes_path": str(workspace / "NOTES.md"),
        "scratch": str(workspace / "scratch"),
        "image_paths": "observations/NNNNNN/prompt.json lists image paths relative to this workspace",
    }
    (workspace / "workspace.json").write_text(json.dumps(data, indent=2) + "\n")
    (workspace / "AGENTS.md").write_text(CODING_GUIDANCE)
    notes = workspace / "NOTES.md"
    if notes.read_text() == "# Agent episode notes\n":
        notes.write_text(
            "# Episode working memory\n\n"
            "Maintain confirmed progress, current objective, geometry estimates, "
            "tracking errors and corrections as useful. No observations yet.\n"
        )
