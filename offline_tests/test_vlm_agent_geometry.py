# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The one geometric fact the vlm_agent policy rests on: where a grasp closes.

``ee_pos``/``ee_quat`` report the Robotiq 2F-85 *mount flange* (base_link), not the
fingers, and every number the VLM aligns against -- the fingertip position, the height
above the table, the descent countdown, the GRASP gate -- is that pose plus
``FLANGE_TO_FINGERTIP``. Get it wrong and the model centres a point that is not where its
jaws are: it descends forever and the object is never between the fingers.

This has now been wrong twice, each time for a different reason, so both are pinned here:

* ``(0, 0, 0.1628)`` -- wrong AXIS. Off the Robotiq spec sheet, along local +z, without
  checking which way base_link's z points. The fingers run along +X.
* ``(0.15, 0.03, 0)`` -- right axis, wrong POINT. That is
  ``robolab/eval/phases.py::TCP_OFFSET``, the fingertip midpoint the repo grades contact
  against. It is 36 mm from where the jaws actually close, 30 mm of it straight off the
  jaw centreline -- so the aim point lands beside the jaws rather than between them.

Runs without Isaac.
"""
import numpy as np

# PharmFlow's measured DROID contract, arena/biomedical_contract.py::
# DROID_BASE_TO_CLOSED_GRASP_CENTER -- "offset from the DROID base_link to the point where
# closed fingers meet". Copied rather than imported: that is a separate repository, and
# this test must run here. Independently reproduced on this rig from the pad colliders
# (robolab/eval/tows.py::PAD_BOX plus the live finger-link poses, an FK validated to
# 0.00 mm against recorded finger poses): (0.1301, 0.0, 0.0), agreeing to 1.0 mm in x and
# 0.14 mm in y. See tests/test_droid_banana_grasp.py, which measures it live.
PHARMFLOW_GRASP_CENTRE = (0.13108637, 0.00014421, 0.0)


def test_agent_aims_at_the_point_the_jaws_close_on():
    from policies.vlm_agent.token_controller import FLANGE_TO_FINGERTIP

    assert np.allclose(FLANGE_TO_FINGERTIP, PHARMFLOW_GRASP_CENTRE, atol=2e-4)


def test_the_aim_point_is_on_the_jaw_centreline():
    """y must be exactly 0.

    The two pad colliders are symmetric about base_link's y=0 (``PAD_BOX`` puts their
    faces at +-42.5 mm), so the centreline y is 0 by construction. A non-zero y here is
    the whole "gripper straddles nothing" failure: the model aims that far to one side.
    """
    from policies.vlm_agent.token_controller import FLANGE_TO_FINGERTIP

    assert FLANGE_TO_FINGERTIP[1] == 0.0
    assert FLANGE_TO_FINGERTIP[2] == 0.0


def test_the_grading_tcp_is_a_different_point_and_must_not_be_reused_here():
    """Guards against "fixing" this back to ``TCP_OFFSET``.

    Grading a contact and aiming a grasp are different points on this gripper, 36 mm
    apart. ``TCP_OFFSET`` is the fingertip midpoint and is correct for what phases.py
    does with it; it is wrong as an aim point.
    """
    from robolab.eval.phases import TCP_OFFSET

    from policies.vlm_agent.token_controller import FLANGE_TO_FINGERTIP

    separation = np.linalg.norm(np.subtract(TCP_OFFSET, FLANGE_TO_FINGERTIP))
    assert 0.030 < separation < 0.042, (
        f"the grading TCP and the grasp centre are {separation * 1000:.0f} mm apart; "
        "if this collapsed to ~0 somebody has aimed the policy at the grading point again"
    )


def test_approach_axis_is_the_direction_the_fingers_point():
    """Tilt is measured against the jaw axis, so it has to BE the jaw axis.

    With the old ``(0, 0, 1)`` a gripper hanging perfectly straight down reported 90 deg
    of tilt, and that bogus number was read as evidence the orientation hold had failed.
    Measured live after the fix: 0 deg (tests/test_droid_banana_grasp.py).
    """
    from policies.vlm_agent.token_controller import (
        _APPROACH_AXIS_LOCAL,
        FLANGE_TO_FINGERTIP,
    )

    tip_dir = np.asarray(FLANGE_TO_FINGERTIP, dtype=float)
    tip_dir /= np.linalg.norm(tip_dir)
    axis = np.asarray(_APPROACH_AXIS_LOCAL, dtype=float)
    axis /= np.linalg.norm(axis)
    cos = float(tip_dir @ axis)
    assert cos > 0.99, (
        f"approach axis {axis} is {np.degrees(np.arccos(cos)):.0f} deg off the "
        f"flange -> grasp-centre direction {np.round(tip_dir, 3)}"
    )


def test_pointing_straight_down_reads_as_zero_tilt_and_a_lower_grasp_point():
    """The end-to-end symptom, without Isaac: flange 30 cm up, jaws 17 cm up."""
    from policies.vlm_pinpoint.pointing_controller import quat_rotate

    from policies.vlm_agent.token_controller import (
        _APPROACH_AXIS_LOCAL,
        FLANGE_TO_FINGERTIP,
    )

    # +90 deg about +y maps local +x onto world -z: the gripper hangs straight down.
    half = np.pi / 4
    quat = (np.cos(half), 0.0, np.sin(half), 0.0)
    flange = np.array([0.45, 0.0, 0.30])

    tip = flange + quat_rotate(quat, FLANGE_TO_FINGERTIP)
    assert tip[2] < flange[2], "the jaws must be BELOW the flange when pointing down"
    assert np.isclose(tip[2], flange[2] - FLANGE_TO_FINGERTIP[0])
    assert np.allclose(tip[:2], flange[:2]), "pointing straight down adds no lateral offset"

    down = quat_rotate(quat, _APPROACH_AXIS_LOCAL)
    down /= np.linalg.norm(down)
    tilt = np.degrees(np.arccos(np.clip(down @ np.array([0.0, 0.0, -1.0]), -1.0, 1.0)))
    assert tilt < 1.0, f"a gripper pointing straight down reported {tilt:.0f} deg of tilt"
