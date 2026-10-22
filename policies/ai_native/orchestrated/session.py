"""Proposal reviews through the existing persistent Codex transport."""

import base64
import io
import json

from PIL import Image

from policies.ai_native.agentic.app_server import EpisodeMemory, PersistentAppServerSession
from policies.ai_native.agentic.runtime_options import RuntimeOptions
from policies.ai_native.common.prompts import (
    DROID_EEF_FRAME_PROMPT, DROID_ROBOT_PROMPT,
)
from policies.ai_native.orchestrated.gate_assessment import GATE_INSTRUCTION
from policies.ai_native.orchestrated.protocol import review_tool

CAMERA_NAMES = ("over_shoulder_left_camera", "wrist_cam")


def reviewer_prompt():
    return DROID_ROBOT_PROMPT + "\n" + DROID_EEF_FRAME_PROMPT + "\n" + GATE_INSTRUCTION + "\n" + """
Use review_policy_proposal exactly once per turn, with the current request_id.
The response includes edit and target for every mode; unused fields are ignored.
For student, choose a prefix of 1..15 controls. For eef, choose
one absolute flange target within 0.05 m and 0.35 rad of the measured pose, held
for 1..5 controls. For edit, choose 1..5 FK targets and root-frame offsets up to
0.05 m and 0.35 rad. Offset at row i is (i+1)/steps times the requested offset;
the keep/open/closed gripper override applies immediately. Root-frame rotation
offsets left-multiply the proposal orientation. Stop saves an incomplete episode;
it is not native success or failure. Unit quaternion is WXYZ; gripper_closed is boolean. The host
recomputes bounded DLS at each step (0.02 m, 0.1 rad, 0.05 rad per-joint limits).
It does not wait for arrival; assess actual tracking at the next observation.
Simulation pauses during model inference. control_hz gives execution timing.
After any execution a fresh student proposal is required; discarded tails never run.
The nominal open-gripper pad center is +0.1311 m along base_link-local X. This
is a geometry reference, not measured contact or proof of a grasp.
Only the evaluator decides success and termination. Do not request rewards,
object ground truth, task predicates, resets, rollback or hypothetical physics.
Give concise observable evidence and action purpose, not private reasoning.
请用简体中文输出 reason 和 assessment 中的公开说明，保留原始任务指令、JSON 字段名和枚举值。
"""


def observation_blocks(request, *, image_size, jpeg_quality):
    # The client constructs this whitelist. Never serialize the simulator obs.
    payload = {key: request[key] for key in (
        "request_id", "control_step", "instruction", "control_hz", "measured_state",
        "proposal", "previous_execution",
    )}
    if "validation_error" in request:
        payload["validation_error"] = request["validation_error"]
    blocks = [dict(type="text", text=json.dumps(payload, allow_nan=False))]
    for name in CAMERA_NAMES:
        image = Image.fromarray(request["images"][name])
        image.thumbnail((image_size, image_size), Image.Resampling.BILINEAR)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=jpeg_quality)
        blocks.extend([dict(type="text", text=f"Current control step {request['control_step']}; camera {name}"),
                       dict(type="image", mimeType="image/jpeg",
                            data=base64.b64encode(buffer.getvalue()).decode("ascii"))])
    return blocks


class ProposalReviewSession:
    def __init__(self, *, spec, model_id, memory_root, episode_idx, allow_model_requests,
                 codex_bin="codex", timeout=120., image_size=896, jpeg_quality=85,
                 optional_tools=False, memory_root_source="explicit"):
        if type(image_size) is not int or not 128 <= image_size <= 2048:
            raise ValueError("image_size must be between 128 and 2048")
        if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be between 1 and 100")
        self.image_size, self.jpeg_quality = image_size, jpeg_quality
        memory = EpisodeMemory(memory_root, policy="orchestrated-droid", episode_idx=episode_idx,
                               env_id=0, root_source=memory_root_source)
        self.workspace = memory.path
        tool = review_tool(spec)["function"]
        self.server = PersistentAppServerSession(
            model_id=model_id, memory=memory, allow_model_requests=allow_model_requests,
            codex_bin=codex_bin, timeout=timeout, instructions=reviewer_prompt(),
            output_tools=[dict(type="function", name=tool["name"], description=tool["description"],
                               inputSchema=tool["parameters"])],
            options=RuntimeOptions(optional_tools=optional_tools),
        )
        self.last_metadata = {}

    def query(self, request):
        self.last_metadata = {}
        try:
            result = self.server.query(observation_blocks(
                request, image_size=self.image_size, jpeg_quality=self.jpeg_quality,
            ))
        finally:
            self.last_metadata = dict(self.server.last_metadata)
            self.last_metadata["usage_is_cumulative"] = True
        return result.tool_call["name"], result.tool_call["arguments"]

    def record_host_result(self, payload):
        self.server.record_host_result(payload)

    def close(self):
        self.server.close()
