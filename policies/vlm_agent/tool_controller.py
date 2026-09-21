# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Function-calling controller: the same motion contract, expressed as tools.

Produces the same :class:`~policies.vlm_agent.token_controller.Decision` as the token
controller, so the client is mode-agnostic and both modes inherit one implementation of
the ik-scale division, the displacement cap, the orientation hold and the gripper latch.

Three things here are deliberate, and each one is a bug the naive version had:

* **The tool is named ``move_by`` and takes NAMED dimensions.** A tool called
  ``move_joints`` whose description says "target joint positions in radians" but whose
  handler treats the numbers as Cartesian metres is not a schema mismatch the model can
  work around — it will dutifully send joint angles of 1-2 rad and the handler will ask
  the IK for a 1-2 metre step. Relative and absolute control get different names for the
  same reason.
* **The bounds are in the description, and enforced on the way out.** The model is told
  the per-call limit, and anything outside it is clipped rather than forwarded.
* **A malformed call returns an error to the model, it does not raise.** Raising kills the
  whole eval sweep over one bad reply; returning the error gives the model the chance to
  correct itself on the next step.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np

from policies.vlm_agent.token_controller import (
    REL_IK_ACHIEVED_FRACTION,
    DONE_UNIT,
    GIVE_UP_UNIT,
    GRASP_UNIT,
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    RELEASE_UNIT,
    STEP_M,
    Decision,
    TokenController,
    format_state,
    image_to_data_url,
)

logger = logging.getLogger(__name__)

# Per-call displacement bound, in metres of COMMANDED motion. Sized so one call is a
# comfortable multiple of the calibrated 2 cm unit step without letting the model ask for
# a lunge; see the token controller's module docstring for where 0.072 comes from.
MAX_MOVE_M = 4 * STEP_M
MAX_YAW_RAD = 0.6

DIMENSIONS = ("x", "y", "z", "yaw")


class ToolController(TokenController):
    """Tool-calling front end over the token controller's motion contract."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    # ------------------------------------------------------------------
    # Schemas + prompt
    # ------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict]:
        bounds = (
            f"Axes are the robot base frame, in metres: +x away from the base, "
            f"+y to the robot's left, +z up. yaw is in radians about the base +z axis "
            f"(right-hand rule). "
            f"Per call: |x|,|y|,|z| <= {MAX_MOVE_M:.3f} m and |yaw| <= {MAX_YAW_RAD:.2f} rad; "
            f"larger values are clipped to those bounds. Omitted dimensions do not move. "
            f"IMPORTANT: this controller achieves only about "
            f"{REL_IK_ACHIEVED_FRACTION * 100:.0f}% of the displacement you request, so ask "
            f"for about {1.0 / REL_IK_ACHIEVED_FRACTION:.1f}x the distance you actually want "
            f"to travel: {STEP_M:.3f} m requested lands on about "
            f"{STEP_M * REL_IK_ACHIEVED_FRACTION * 100:.0f} cm, and the maximum "
            f"{MAX_MOVE_M:.3f} m lands on about "
            f"{MAX_MOVE_M * REL_IK_ACHIEVED_FRACTION * 100:.0f} cm."
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": "move_by",
                    "description": (
                        "Move the gripper BY the given displacement, holding its current "
                        "orientation except for any yaw you ask for. " + bounds
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "deltas": {
                                "type": "object",
                                "description": (
                                    "Map of dimension name to displacement. Valid names: "
                                    + ", ".join(DIMENSIONS)
                                ),
                                "additionalProperties": {"type": "number"},
                            },
                            "note": {
                                "type": "string",
                                "description": (
                                    "What you see right now and why you chose this motion, "
                                    "in one plain sentence. It is written to the run log."
                                ),
                            },
                        },
                        "required": ["deltas", "note"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "grasp",
                    "description": (
                        "Close the gripper. The command is held long enough for the fingers "
                        "to travel and load up on the object; do not repeat it while closed."
                    ),
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "release",
                    "description": "Open the gripper to release the held object.",
                    "parameters": {"type": "object", "properties": {}},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "done",
                    "description": (
                        "Declare the task complete. Only call this when the goal state is "
                        "already visible in the images. Stops the run."
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {"summary": {"type": "string"}},
                        "required": ["summary"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "give_up",
                    "description": "The task cannot be completed. Stops the run.",
                    "parameters": {
                        "type": "object",
                        "properties": {"reason": {"type": "string"}},
                        "required": ["reason"],
                    },
                },
            },
        ]

    def system_prompt(self, view_labels: list[str]) -> str:
        wrist = next((label for label in view_labels if label == "WristView"), None)
        agent = next((label for label in view_labels if label != "WristView"), None)
        lines = [
            "You control a Franka arm with a parallel gripper by calling tools. Call exactly "
            "one tool per reply.",
            "",
            f"VIEWS, in the order they are attached: {', '.join(view_labels) or 'none'}.",
        ]
        if agent:
            lines.append(
                f"- {agent}: fixed third-person view. In it, the robot's +x runs toward the "
                "top of the image and +y toward the image left (both lean leftward, so "
                "use vertical to tell them apart: up = +x, down = +y)."
            )
        if wrist:
            lines.append(
                f"- {wrist}: bolted to the flange and looking out ALONG the jaw axis, "
                "with the fingertips about 15 cm ahead of it. What sits between the "
                "fingers is what fills the middle of this view, so this is the view that "
                "tells you whether the object is lined up to be grasped. The fingertips "
                "enter from the top of the frame, so a target you are correctly centred "
                "on sits high in the view, partly behind them. Take the remaining height "
                "from the fingertip height in the state block, not from this view."
            )
        lines += [
            "",
            "Approach in stages: line the gripper up above the target in x and y, lower it "
            "in z until the fingertip height in the state block is within a few cm of the "
            "table with the target still centred left/right, then grasp. "
            "Lift before translating a held object, and release only once it is over and "
            "lowered onto its destination.",
            "",
            "Prefer several small moves you can check against the images over one large one.",
        ]
        return "\n".join(lines)

    def build_prompt(
        self,
        obs: dict,
        *,
        gripper_cmd: float,
        recent: list[str] | None = None,
    ) -> list[dict]:
        views = obs.get("views") or []
        content: list[dict] = []
        for label, image in views:
            content.append({"type": "text", "text": f"{label}:"})
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(image)}})

        history = ", ".join(recent) if recent else "none"
        content.append({
            "type": "text",
            "text": (
                f"Task: {obs.get('instruction', '')}\n\n"
                f"Robot state:\n"
                f"{format_state(obs.get('state', {}), gripper_cmd, travel_cm=self.travel_cm)}\n\n"
                f"Your last calls (oldest first): {history}\n\n"
                "Call one tool."
            ),
        })
        return [
            {"role": "system", "content": self.system_prompt([label for label, _ in views])},
            {"role": "user", "content": content},
        ]

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _tool_call_from_message(message: dict) -> tuple[str, dict]:
        """Return ``(name, args)``. Falls back to a JSON object in ``content``.

        Some OpenAI-compatible servers emit the call as plain text when the model is not
        reliably tool-trained; recovering that is cheap and avoids discarding a decision
        the model did make.
        """
        tool_calls = message.get("tool_calls") or []
        if not tool_calls:
            raw = message.get("content") or message.get("reasoning") or ""
            try:
                parsed = json.loads(raw)
            except (TypeError, ValueError):
                parsed = None
            if isinstance(parsed, dict) and "name" in parsed:
                tool_calls = [{"function": parsed}]
        if not tool_calls:
            raise ValueError("reply contained no tool call")

        func = tool_calls[0].get("function") or {}
        name = str(func.get("name") or "")
        raw_args = func.get("arguments")
        if isinstance(raw_args, str):
            try:
                args = json.loads(raw_args or "{}")
            except ValueError as exc:
                raise ValueError(f"tool arguments were not valid JSON: {exc}") from exc
        else:
            args = raw_args or {}
        if not isinstance(args, dict):
            raise ValueError(f"tool arguments must be an object, got {type(args).__name__}")
        return name, args

    def _decision_from_move(self, args: dict) -> Decision:
        deltas = args.get("deltas")
        if isinstance(deltas, (list, tuple)):
            # Tolerate a positional array; pad/truncate explicitly rather than letting a
            # short list silently produce an under-length action.
            values = list(deltas)[: len(DIMENSIONS)]
            deltas = dict(zip(DIMENSIONS, values))
        if not isinstance(deltas, dict):
            raise ValueError(f"'deltas' must be an object keyed by {', '.join(DIMENSIONS)}")

        unknown = [key for key in deltas if key not in DIMENSIONS]
        if unknown:
            raise ValueError(
                f"unknown dimension(s) {', '.join(map(repr, unknown))}; "
                f"valid names: {', '.join(DIMENSIONS)}"
            )

        try:
            xyz = np.array(
                [float(deltas.get(axis, 0.0) or 0.0) for axis in ("x", "y", "z")],
                dtype=np.float64,
            )
            yaw = float(deltas.get("yaw", 0.0) or 0.0)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'deltas' values must be numbers: {exc}") from exc

        if not np.all(np.isfinite(xyz)) or not np.isfinite(yaw):
            raise ValueError("'deltas' contained a non-finite value")

        xyz = np.clip(xyz, -MAX_MOVE_M, MAX_MOVE_M)
        yaw = float(np.clip(yaw, -MAX_YAW_RAD, MAX_YAW_RAD))

        # Split the request over control steps the same way a token decision is split, so
        # the per-step command — and with it the orientation transient — stays small.
        steps = self.control_steps
        return Decision(
            unit="move_by",
            steps=steps,
            move_m=xyz / steps,
            yaw_rad=yaw,
            note=str(args.get("note") or "")[:200],
        )

    def decision_from_message(self, message: dict) -> Decision:
        """Assistant message -> :class:`Decision`. Raises ``ValueError`` with a message
        that is safe to hand back to the model."""
        name, args = self._tool_call_from_message(message)

        if name == "move_by":
            return self._decision_from_move(args)
        if name == "grasp":
            return Decision(unit=GRASP_UNIT, steps=self.gripper_hold_steps, gripper=GRIPPER_CLOSE)
        if name == "release":
            return Decision(unit=RELEASE_UNIT, steps=self.gripper_hold_steps, gripper=GRIPPER_OPEN)
        if name == "done":
            return Decision(unit=DONE_UNIT, steps=1, terminal=True, note=str(args.get("summary") or "")[:200])
        if name == "give_up":
            return Decision(unit=GIVE_UP_UNIT, steps=1, terminal=True, note=str(args.get("reason") or "")[:200])
        raise ValueError(
            f"unknown tool {name!r}; valid tools: move_by, grasp, release, done, give_up"
        )
