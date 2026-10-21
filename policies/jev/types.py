# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decision types and the one place a choice becomes metres.

Port of RoboJEV's ``src/jev_vla_sim/types.py``. Two invariants live here and
nowhere else:

* **Validation is total.** A decision that is not three valid axis labels plus
  a valid gripper label cannot be constructed, so no malformed model reply can
  reach the robot.
* **Non-zero axes are NORMALISED.** ``delta()`` returns a vector whose norm is
  exactly ``step_m`` regardless of how many axes are non-zero. Giving each axis
  the full step instead -- the obvious-looking alternative -- makes a diagonal
  move sqrt(2) or sqrt(3) times longer than a straight one, so the model's
  notion of "one step" silently depends on how many axes it picked.

Backends therefore return *labels*, never metres: the metric conversion has a
single owner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

AXIS_CHOICES = ("negative", "zero", "positive")
GRIPPER_CHOICES = ("open", "hold", "close")


class PolicyError(RuntimeError):
    """No action may be executed from this failed decision."""


@dataclass(frozen=True)
class EefDecision:
    """One two-stage decision: a direction per axis plus a gripper command."""

    state_id: str
    x: str
    y: str
    z: str
    gripper: str
    intent: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.state_id, str) or not self.state_id:
            raise PolicyError("decision has no state ID")
        if any(v not in AXIS_CHOICES for v in (self.x, self.y, self.z)):
            raise PolicyError(f"invalid XYZ choice: {(self.x, self.y, self.z)!r}")
        if self.gripper not in GRIPPER_CHOICES:
            raise PolicyError(f"invalid gripper choice: {self.gripper!r}")

    def delta(self, step_m: float) -> np.ndarray:
        """Normalised translation for this decision, in metres.

        ``[negative, zero, positive] -> [-1, 0, +1]``, then scaled so the total
        displacement is ``step_m``. All-zero stays all-zero.
        """
        if not math.isfinite(step_m) or step_m <= 0:
            raise ValueError(f"invalid step_m: {step_m!r}")
        d = np.array([AXIS_CHOICES.index(v) - 1 for v in (self.x, self.y, self.z)],
                     dtype=float)
        return d * (step_m / np.linalg.norm(d)) if d.any() else d

    def action_dict(self) -> dict:
        return {k: getattr(self, k) for k in ("x", "y", "z", "gripper")}
