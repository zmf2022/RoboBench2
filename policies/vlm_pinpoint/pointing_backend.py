# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pointing backend on top of the shared vLLM chat service.

The external ``vlm-pinpoint`` package is unreachable (its GitHub repo is gone),
so this module implements the same role in-repo: ask the VLM for an image point
plus a phase, parse and validate the reply. Geometry (point + depth ->
metric motion) lives in ``pointing_controller.py``.

Point contract (official RoboPoint format first, ours as fallback):
    [(x1, y1), ...]              # 0-1 normalized floats, or pixels if > 1
    [{"point_2d": [x, y], ...}]  # Qwen-VL grounding style, pixels
    POINT: <x y>                 # 0-1000 ints, origin top-left
    PHASE: <APPROACH|ALIGN|GRASP|LIFT|PLACE|DONE>

All forms are origin-top-left and are normalized to 0-1000 on the way out.
"""

from __future__ import annotations

import base64
import io
import logging
import re

import numpy as np
from PIL import Image

from policies.vlm_agent.vlm_backend import VLMBackend

logger = logging.getLogger(__name__)

PHASES = ("APPROACH", "ALIGN", "GRASP", "LIFT", "PLACE", "DONE")

# Grayscale, not a colour map. The depth image is only a coarse near/far prior
# for choosing a pixel — the metric depth the controller servos on is sampled
# numerically from wrist_cam_depth. A monotonic brightness ramp plus a one-line
# legend is read reliably; turbo/jet hue order is not (pretraining has it both
# ways round) and buys nothing here.
DEPTH_LEGEND = "brighter means closer, black means no reading (sky)"

_NUM = r"(\d*\.?\d+)"
_POINT_RE = re.compile(r"POINT:\s*(\d{1,4})\s+(\d{1,4})", re.IGNORECASE)
_PHASE_RE = re.compile(r"PHASE:\s*([A-Za-z_]+)", re.IGNORECASE)
# Qwen-VL grounding style: [{"point_2d": [x, y], "label": ...}], pixels.
_POINT2D_RE = re.compile(r"point_2d\D{0,4}" + _NUM + r"\s*,\s*" + _NUM, re.IGNORECASE)
# Official RoboPoint format: [(x1, y1), ...].
_TUPLE_RE = re.compile(r"\(\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\)")
# Bare JSON pair: [x, y].
_BRACKET_RE = re.compile(r"\[\s*" + _NUM + r"\s*,\s*" + _NUM + r"\s*\]")


def _pair_to_1000(sx: str, sy: str, frame_size: tuple[int, int] | None) -> tuple[int, int]:
    """One (x, y) reply pair -> our 0-1000 frame coords.

    RoboPoint's convention, which Qwen-family grounding also follows: a value in
    0-1 is a normalized fraction, anything larger is a pixel in the image the
    model was shown. Pixels need ``frame_size`` to normalize; without it we can
    only assume the reply already used our 0-1000 scale.
    """
    x, y = float(sx), float(sy)
    if x <= 1.0 and y <= 1.0 and ("." in sx or "." in sy):
        return int(round(x * 1000)), int(round(y * 1000))
    if frame_size is not None:
        w, h = frame_size
        return int(round(x / w * 1000)), int(round(y / h * 1000))
    return int(round(x)), int(round(y))


def _to_data_url(image: np.ndarray) -> str:
    array = np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[:, :, :3]
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def encode_depth_for_vlm(depth, max_m: float = 2.0) -> np.ndarray:
    """Depth (m) -> grayscale uint8 image for the VLM, brighter = closer.

    Fixed range (no per-frame auto-normalize, so values stay comparable across
    steps). Invalid readings (inf/nan/<=0, e.g. skybox) go black. See
    :data:`DEPTH_LEGEND` for why this is grayscale and not a colour map.
    """
    d = np.asarray(depth, dtype=float)
    while d.ndim > 2:
        d = d[..., 0]
    norm = np.clip(d / max_m, 0.0, 1.0)
    valid = np.isfinite(d) & (d > 0)
    out = np.repeat((((1.0 - norm) * 255.0).astype(np.uint8))[:, :, None], 3, axis=2)
    out[~valid] = 0
    return out


_VIEW_PROMPTS = {
    # Fixed over-shoulder camera. The robot is IN frame, which is the whole
    # difficulty and the whole opportunity: it can see the spatial relationship
    # between its own hand and the target, but it can also mistake the arm for
    # an object.
    "external": (
        "This is a fixed third-person camera looking down at a table. The white "
        "robot arm and its black two-finger gripper are visible in the frame; a "
        "green crosshair marks the gripper's fingertips, so you can see how far "
        "the hand still has to travel and in which direction. "
    ),
    # Eye-in-hand. Measured on BananaInBowl: the objects sit near the TOP edge,
    # level with the fingertips, and the lower half of the frame is bare table.
    "wrist": (
        "The black gripper fingers enter from the TOP of the image. The table "
        "objects sit NEAR THE TOP of the frame too, level with the fingertips "
        "or just beyond them; the lower half of the frame is usually bare "
        "table with nothing on it. Point at the object itself wherever it "
        "actually is — do not avoid the region near or above the fingertips, "
        "and never point at bare table just because it is below them. "
    ),
}


class VLLMPointingBackend:
    """Ask a chat-completions VLM for ``POINT`` + ``PHASE`` on one RGB frame."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        model: str | None = None,
        timeout: float = 120.0,
        # VLMBackend sends enable_thinking + reasoning_effort="xhigh", so the
        # reasoning trace is billed against this budget before a single answer
        # token is emitted. 512 was exhausted by the trace alone: replies came
        # back finish_reason="length" with empty content and every decision
        # degraded to a hold. Match vlm_agent's headroom.
        max_tokens: int = 8192,
        temperature: float = 0.0,
    ) -> None:
        self.client = VLMBackend(
            base_url=base_url,
            model=model,
            timeout=timeout,
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def system_prompt(self, view: str = "wrist", has_depth: bool = True) -> str:
        # Mirrors the official RoboPoint demo suffix (gradio_web_server.py):
        # free-form instruction + fixed format string asking for normalized tuples.
        # PHASE is our controller extension (RoboPoint itself has no phases).
        return (
            "Locate the task target in the image. Your answer should be formatted "
            "as a list of tuples, i.e. [(x1, y1), ...], where each tuple contains "
            "the x and y coordinates of a point satisfying the task. "
            "The coordinates should be between 0 and 1, indicating the normalized "
            "pixel locations of the points in the image. "
            "The point must lie ON the object you intend to act on: the object to "
            "pick for APPROACH/ALIGN/GRASP/LIFT, the destination surface for PLACE. "
            "Never point at the robot itself (white arm, links, or the black "
            "fingertips): only table objects count. "
            + _VIEW_PROMPTS.get(view, _VIEW_PROMPTS["wrist"])
            + ("A second image shows the depth map from the same viewpoint: "
               + DEPTH_LEGEND + ". Prefer close regions for grasp targets; never "
               "point into black areas. " if has_depth else "") +
            "Then on a new line write PHASE: <APPROACH|ALIGN|GRASP|LIFT|PLACE|DONE> "
            "— APPROACH while far, ALIGN when close, GRASP to close the fingers, "
            "LIFT after grasping, PLACE over the destination, DONE when the goal "
            "state is already visible. "
            "You MUST always output both the point list and the PHASE line, even "
            "if the target is not visible — then answer [(0.5, 0.5)] with "
            "PHASE: APPROACH."
        )

    def point(self, image: np.ndarray, instruction: str, max_retries: int = 3,
              depth_image: np.ndarray | None = None,
              state_text: str | None = None, view: str = "wrist") -> dict:
        """Query the VLM; return ``{"x", "y", "phase", "raw"}`` in 0-1000 coords.

        ``image`` is the wrist frame the point is reported in; ``depth_image``
        is the grayscale wrist depth map (brighter = closer) helping far-vs-near
        judgment, and must be in the SAME orientation as ``image``. Retries
        garbage replies; raises the last :class:`ValueError` when exhausted.
        """
        image = np.asarray(image)
        frame_size = (int(image.shape[1]), int(image.shape[0]))
        user_content: list[dict] = [
            {"type": "image_url", "image_url": {"url": _to_data_url(image)}},
        ]
        task_text = f"Task: {instruction}"
        if depth_image is not None:
            user_content.append(
                {"type": "image_url",
                 "image_url": {"url": _to_data_url(depth_image)}})
            task_text += (" The SECOND image is the wrist depth map from the same "
                          "viewpoint as the first image: " + DEPTH_LEGEND
                          + ". Prefer close regions for grasp targets; never "
                          "point into black areas.")
        if state_text:
            # Proprioception the picture cannot show: a closed EMPTY gripper and
            # a closed LOADED one are the same image, and that is exactly what
            # the GRASP/LIFT/DONE decision turns on.
            task_text += " " + state_text
        user_content.append({"type": "text", "text": task_text})
        messages = [
            {"role": "system",
             "content": self.system_prompt(view, has_depth=depth_image is not None)},
            {"role": "user", "content": user_content},
        ]
        last: ValueError | None = None
        for attempt in range(max(1, max_retries)):
            # Re-sending an identical payload at temperature 0 returns an
            # identical reply, so a retry only means something if it samples.
            # Nudged on the shared client rather than through complete(), which
            # takes no per-call temperature — restored in the finally below so
            # nothing else sees the change.
            base_temp = self.client.temperature
            if attempt:
                self.client.temperature = 0.2 * attempt
            try:
                response = self.client.complete(messages)
            finally:
                self.client.temperature = base_temp
            try:
                return parse_point_response(response, frame_size=frame_size)
            except ValueError as exc:
                last = exc
                logger.warning(
                    "[Pinpoint] unparsable reply (attempt %d/%d, finish_reason=%s): %s",
                    attempt + 1, max_retries, _finish_reason(response), exc)
        raise last  # type: ignore[misc]

    def close(self) -> None:
        closer = getattr(self.client, "close", None)
        if callable(closer):
            closer()


def _finish_reason(response: dict) -> str | None:
    """Why the server stopped. ``"length"`` means the budget ran out mid-reply,
    which looks exactly like a model that will not point — worth logging apart."""
    choices = response.get("choices") or []
    return choices[0].get("finish_reason") if choices else None


def _message_text(response: dict) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise ValueError("VLM response has no choices")
    message = choices[0].get("message") or {}
    for key in ("content", "reasoning_content", "reasoning"):
        text = message.get(key)
        if isinstance(text, str) and text.strip():
            return text
    raise ValueError("VLM reply has no text content")


def parse_point_response(response: dict, frame_size: tuple[int, int] | None = None) -> dict:
    """Parse a point + phase from a chat-completions response.

    Accepts our ``POINT: x y`` (0-1000) marker, Qwen-VL's ``"point_2d": [x, y]``
    grounding form, and RoboPoint's ``[(x, y), ...]`` tuples / bare ``[x, y]``
    pairs. For the latter three, ``frame_size`` (w, h of the image the model was
    shown) lets a pixel answer be told apart from a normalized one — without it
    a pixel answer is silently misread, so pass it whenever you have it.

    Takes the LAST match of each marker so leading chatter cannot hijack the
    decision. A reply without PHASE defaults to APPROACH. Raises
    :class:`ValueError` on anything unusable.
    """
    text = _message_text(response)
    points = _POINT_RE.findall(text)
    phases = _PHASE_RE.findall(text)
    if points:
        # Our own marker is 0-1000 by definition, no frame_size needed.
        x, y = (int(v) for v in points[-1])
    else:
        for pattern in (_POINT2D_RE, _TUPLE_RE, _BRACKET_RE):
            pairs = pattern.findall(text)
            if pairs:
                x, y = _pair_to_1000(*pairs[-1], frame_size)
                break
        else:
            raise ValueError(f"reply has no point: {text[:200]!r}")
    phase = phases[-1].strip().upper() if phases else "APPROACH"
    if not (0 <= x <= 1000 and 0 <= y <= 1000):
        logger.warning("[Pinpoint] POINT %s out of range, clamping to frame", (x, y))
        x, y = min(max(x, 0), 1000), min(max(y, 0), 1000)
    if phase not in PHASES:
        # Unknown phase words (e.g. "THE ...") default to APPROACH: servoing on
        # is always safe, while a wrong GRASP/DONE would be destructive. The
        # radius auto-grasp still closes the fingers when actually close.
        logger.warning("[Pinpoint] unknown PHASE %r, defaulting to APPROACH", phase)
        phase = "APPROACH"
    return {"x": x, "y": y, "phase": phase, "raw": text}
