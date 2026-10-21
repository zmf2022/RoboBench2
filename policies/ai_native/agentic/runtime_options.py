# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independent Agentic runtime settings; no report-condition presets."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RuntimeOptions:
    motion_feedback: bool = True
    optional_tools: bool = False
    persistent_history: bool = True
    observation_offsets: tuple[int, ...] | None = None
    geometry: bool = True
    explicit_planning: bool = False
    image_demonstration: bool = False
    decomposition_scaffolding: bool = True
    action_space: str = "eef"
    include_joint_state: bool = False
    history_text: str = "none"
    history_images: str = "none"

    def __post_init__(self):
        offsets = self.observation_offsets
        if offsets is not None and (
            not offsets or any(type(x) is not int or x < 0 for x in offsets)
            or offsets[-1] != 0 or tuple(sorted(set(offsets), reverse=True)) != offsets
        ):
            raise ValueError("history-offsets must be unique descending nonnegative integers ending in 0")
        if self.action_space not in {"eef", "joint"}:
            raise ValueError("action-space must be eef or joint")
        if self.history_text not in {"none", "full", "sampled"} or self.history_images not in {"none", "full"}:
            raise ValueError("Invalid history channel selection")
        if self.history_channels and self.persistent_history:
            raise ValueError("Selected history channels require --no-persistent-history")
        if self.history_channels and self.optional_tools:
            raise ValueError("Selected history channels require --no-optional-tools")
        if self.history_text == "sampled" and (offsets is None or len(offsets) < 2):
            raise ValueError("Sampled text history requires multiple --history-offsets")
        if self.history_images == "full" and offsets not in (None, (0,)):
            raise ValueError("Full image history requires only the current observation per decision")
        if self.image_demonstration and not self.persistent_history:
            raise ValueError("Demonstrations require persistent history")
        if self.explicit_planning and not self.decomposition_scaffolding:
            raise ValueError("Explicit planning conflicts with --no-decomposition-scaffolding")

    @property
    def history_channels(self) -> bool:
        return self.history_text != "none" or self.history_images != "none"

    @property
    def memory_tools(self) -> bool:
        return self.persistent_history

    @property
    def joint_state(self) -> bool:
        return self.include_joint_state or self.action_space == "joint"

    def to_dict(self) -> dict:
        result = asdict(self)
        if self.observation_offsets is not None:
            result["observation_offsets"] = list(self.observation_offsets)
        result["memory_tools"] = self.memory_tools
        return result
