# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the VLM pointing backend + geometric controller (no server, no sim)."""

import numpy as np
import pytest

from policies.vlm_pinpoint.pointing_backend import (
    PHASES,
    VLLMPointingBackend,
    parse_point_response,
)
from policies.vlm_pinpoint.pointing_controller import (
    PREGRASP_HEIGHT_M,
    MissingDepthError,
    PointingController,
    frame_point_to_sensor,
    intrinsics,
    unproject,
)


def _reply(text):
    return {"choices": [{"message": {"content": text}}]}


def test_parse_point_and_phase():
    out = parse_point_response(_reply("POINT: 350 420\nPHASE: ALIGN"))
    assert out == {"x": 350, "y": 420, "phase": "ALIGN", "raw": out["raw"]}


def test_parse_takes_last_marker():
    out = parse_point_response(
        _reply("POINT: 100 100\nPHASE: APPROACH\nno wait\nPOINT: 351 421\nPHASE: GRASP")
    )
    assert (out["x"], out["y"], out["phase"]) == (351, 421, "GRASP")


def test_parse_reads_reasoning_channel():
    out = parse_point_response(
        {"choices": [{"message": {"reasoning_content": "POINT: 10 20\nPHASE: DONE"}}]}
    )
    assert (out["x"], out["y"], out["phase"]) == (10, 20, "DONE")


@pytest.mark.parametrize("text", ["no point here", ""])
def test_parse_rejects_garbage(text):
    with pytest.raises(ValueError):
        parse_point_response(_reply(text))


def test_parse_defaults_phase_to_approach():
    out = parse_point_response(_reply("POINT: 10 20"))
    assert (out["x"], out["y"], out["phase"]) == (10, 20, "APPROACH")


def test_parse_official_normalized_tuples():
    out = parse_point_response(_reply("[(0.35, 0.42)]\nPHASE: ALIGN"))
    assert (out["x"], out["y"], out["phase"]) == (350, 420, "ALIGN")


def test_parse_official_tuples_take_last():
    out = parse_point_response(_reply("[(0.1, 0.1), (0.9, 0.8)]"))
    assert (out["x"], out["y"], out["phase"]) == (900, 800, "APPROACH")


@pytest.mark.parametrize("text", ["POINT: 1001 0\nPHASE: ALIGN"])
def test_parse_clamps_slightly_out_of_range(text):
    out = parse_point_response(_reply(text))
    assert (out["x"], out["y"]) == (1000, 0)


def test_parse_bad_phase_defaults_to_approach():
    out = parse_point_response(_reply("POINT: 10 20\nPHASE: FLY"))
    assert (out["x"], out["y"], out["phase"]) == (10, 20, "APPROACH")


class _StubSession:
    """Stand-in for requests.Session returning one canned completion."""

    def __init__(self, text):
        self._text = text

    def post(self, url, json=None, timeout=None):
        return _StubResponse(self._text)


class _StubResponse:
    status_code = 200

    def __init__(self, text):
        self._text = text

    def json(self):
        return {"choices": [{"message": {"content": self._text}}]}


def test_point_end_to_end_with_stubbed_server():
    backend = VLLMPointingBackend.__new__(VLLMPointingBackend)
    from policies.vlm_agent.vlm_backend import VLMBackend

    backend.client = VLMBackend(session=_StubSession("POINT: 500 600\nPHASE: LIFT"))
    out = backend.point(np.zeros((64, 64, 3), dtype=np.uint8), "lift it")
    assert (out["x"], out["y"], out["phase"]) == (500, 600, "LIFT")
    assert set(PHASES) == {"APPROACH", "ALIGN", "GRASP", "LIFT", "PLACE", "DONE"}


# ------------------------------------------------------------------
# Controller: transform + phase machine.


def _obs(depth_value=0.5):
    return {
        "depth": np.full((720, 1280), depth_value),
        "K": None, "cam_pos": None, "cam_quat": None,
        "ee_pos": np.array([0.36, 0.0, 0.47]),
        "ee_quat": np.array([1.0, 0.0, 0.0, 0.0]),
    }


def test_frame_point_unrotates_180():
    u, v = frame_point_to_sensor(0, 0)
    assert (u, v) == (1280, 720)
    u, v = frame_point_to_sensor(1000, 1000)
    assert (u, v) == (0, 0)
    u, v = frame_point_to_sensor(500, 500)
    assert (u, v) == (640, 360)


def test_unproject_optical_center():
    fx, fy, cx, cy = intrinsics()
    target = unproject(cx, cy, 0.5, ee_pos=[0.0, 0.0, 0.0],
                       ee_quat=[1.0, 0.0, 0.0, 0.0])
    # Identity EE pose: target = mount_pos + mount_rot * (0, 0, -0.5).
    assert target.shape == (3,)
    assert np.all(np.isfinite(target))


def test_unproject_lands_in_front_of_the_camera():
    """The OpenGL/OpenCV axis mix-up put the target BEHIND the lens.

    Gripper pointing straight down, so the wrist camera looks forward-and-down:
    whatever it sees at the image centre must be further along +x and LOWER than
    the camera. Reading depth as +z instead of -z mirrors it through the lens and
    the arm servos away from the object.
    """
    from policies.vlm_pinpoint.pointing_controller import wrist_extrinsics

    ee_pos, ee_quat = [0.45, 0.0, 0.35], [0.0, 1.0, 0.0, 0.0]
    cam_pos, _ = wrist_extrinsics(ee_pos, ee_quat)
    _, _, cx, cy = intrinsics()
    target = unproject(cx, cy, 0.30, ee_pos=ee_pos, ee_quat=ee_quat)
    assert target[0] > cam_pos[0]      # in front of the camera
    assert target[2] < cam_pos[2]      # and below it, on the table


def test_step_servos_and_caps():
    ctrl = PointingController(step_m=0.072)
    delta, gripper, done = ctrl.step(
        _obs(), {"x": 500, "y": 500, "phase": "APPROACH"}, np.zeros(3))
    assert np.linalg.norm(delta) <= 0.072 + 1e-9
    assert gripper == 0.0 and done is False


def test_step_auto_grasps_when_close():
    ctrl = PointingController()
    target = unproject(640, 360, 0.5, ee_pos=[0.36, 0.0, 0.47],
                       ee_quat=[1.0, 0.0, 0.0, 0.0])
    obs = _obs()
    delta, gripper, done = ctrl.step(
        obs, {"x": 500, "y": 500, "phase": "ALIGN"}, target)
    assert gripper == 1.0 and done is False


def test_place_releases_on_arrival():
    ctrl = PointingController()
    target = unproject(640, 360, 0.5, ee_pos=[0.36, 0.0, 0.47],
                       ee_quat=[1.0, 0.0, 0.0, 0.0])
    delta, gripper, done = ctrl.step(
        _obs(), {"x": 500, "y": 500, "phase": "PLACE"}, target)
    assert gripper == 0.0 and done is False  # fingers open, episode continues


def test_done_terminates():
    ctrl = PointingController()
    delta, gripper, done = ctrl.step(
        _obs(), {"x": 0, "y": 0, "phase": "DONE"}, np.zeros(3))
    assert done is True and np.all(delta == 0)


def test_controller_is_stateless_across_decisions():
    """A GRASP must not stall the NEXT decisions.

    The caller replays the closed gripper for gripper_hold_steps CONTROL steps;
    counting the same hold down inside the controller stalled it for that many
    DECISIONS instead (8x longer by default, each a wasted VLM call).
    """
    ctrl = PointingController()
    assert ctrl.step(_obs(), {"x": 500, "y": 500, "phase": "GRASP"}, np.zeros(3))[1] == 1.0
    delta, _, _ = ctrl.step(_obs(), {"x": 500, "y": 500, "phase": "LIFT"}, np.zeros(3))
    assert np.linalg.norm(delta) > 0  # moves on the very next decision


def test_missing_and_invalid_depth_raise():
    ctrl = PointingController()
    with pytest.raises(MissingDepthError):
        ctrl.step({}, {"x": 500, "y": 500, "phase": "ALIGN"}, np.zeros(3))
    with pytest.raises(MissingDepthError):
        ctrl.step(_obs(np.inf), {"x": 500, "y": 500, "phase": "ALIGN"}, np.zeros(3))


def test_frame_point_round_trip():
    """VLM-frame 0-1000 point -> sensor px must invert rotate-180 + halving."""
    from policies.vlm_pinpoint.pointing_controller import frame_point_to_sensor
    for x1000, y1000 in [(0, 0), (1000, 1000), (500, 500), (590, 380), (123, 987)]:
        su, sv = frame_point_to_sensor(x1000, y1000)
        # forward: sensor -> VLM frame
        u_vlm = (1280.0 - su) / 2.0
        v_vlm = (720.0 - sv) / 2.0
        assert abs(u_vlm / 640.0 * 1000 - x1000) < 2
        assert abs(v_vlm / 360.0 * 1000 - y1000) < 2


def test_encode_depth_brighter_is_closer():
    from policies.vlm_pinpoint.pointing_backend import encode_depth_for_vlm
    d = np.array([[0.2, 1.0], [np.inf, 0.0]])
    out = encode_depth_for_vlm(d, max_m=2.0)
    assert out.shape == (2, 2, 3) and out.dtype == np.uint8
    assert out[0, 0, 0] > out[0, 1, 0]  # nearer brighter
    assert out[1, 0, 0] == 0 and out[1, 1, 0] == 0  # inf/zero -> black


def test_point_sends_rgb_plus_depth():
    from policies.vlm_agent.vlm_backend import VLMBackend
    from policies.vlm_pinpoint.pointing_backend import VLLMPointingBackend

    seen = {}

    class CapSession(_StubSession):
        def post(self, url, json=None, timeout=None):
            seen["n_images"] = sum(
                1 for p in json["messages"][1]["content"] if p["type"] == "image_url")
            seen["has_legend"] = "brighter" in json["messages"][1]["content"][-1]["text"]
            return _StubResponse("[(0.5, 0.5)]\nPHASE: ALIGN")

    backend = VLLMPointingBackend.__new__(VLLMPointingBackend)
    backend.client = VLMBackend(session=CapSession("x"))
    img = np.zeros((36, 64, 3), dtype=np.uint8)
    out = backend.point(img, "t", depth_image=img)
    assert seen["n_images"] == 2 and seen["has_legend"] is True
    assert (out["x"], out["y"]) == (500, 500)


# ------------------------------------------------------------------
# Reply formats. The prompt asks for 0-1 normalized tuples, but a served model
# answers how it was trained to: Qwen-family grounding emits absolute pixels.


def test_parse_qwen_point_2d_pixels():
    """Qwen-VL grounding style: JSON brackets, absolute pixels of the shown frame."""
    out = parse_point_response(
        _reply('[{"point_2d": [320, 180], "label": "banana"}]\nPHASE: ALIGN'),
        frame_size=(640, 360),
    )
    assert (out["x"], out["y"], out["phase"]) == (500, 500, "ALIGN")


def test_parse_pixel_tuples_need_frame_size():
    """Integer tuples are pixels (RoboPoint's own convention), not 0-1000."""
    out = parse_point_response(_reply("[(160, 90)]"), frame_size=(640, 360))
    assert (out["x"], out["y"]) == (250, 250)


def test_parse_normalized_tuples_ignore_frame_size():
    out = parse_point_response(_reply("[(0.35, 0.42)]"), frame_size=(640, 360))
    assert (out["x"], out["y"]) == (350, 420)


def test_parse_point_marker_wins_over_tuples():
    """Our own marker is 0-1000 by definition and must not be rescaled."""
    out = parse_point_response(
        _reply("maybe (0.9, 0.9)?\nPOINT: 350 420"), frame_size=(640, 360))
    assert (out["x"], out["y"]) == (350, 420)


def test_point_passes_frame_size_from_the_image():
    """The backend knows the frame it sent; a pixel reply must resolve against it."""
    backend = VLLMPointingBackend.__new__(VLLMPointingBackend)
    from policies.vlm_agent.vlm_backend import VLMBackend

    backend.client = VLMBackend(session=_StubSession('[{"point_2d": [320, 180]}]'))
    out = backend.point(np.zeros((360, 640, 3), dtype=np.uint8), "point at it")
    assert (out["x"], out["y"]) == (500, 500)


def test_retry_varies_temperature():
    """At temperature 0 an identical payload returns an identical reply, so a
    retry that does not sample is a no-op."""
    from policies.vlm_agent.vlm_backend import VLMBackend

    temps = []

    class TempSession(_StubSession):
        def post(self, url, json=None, timeout=None):
            temps.append(json["temperature"])
            # Unparsable until the last allowed attempt.
            return _StubResponse("POINT: 1 2" if len(temps) == 3 else "no point")

    backend = VLLMPointingBackend.__new__(VLLMPointingBackend)
    backend.client = VLMBackend(session=TempSession("x"))
    backend.point(np.zeros((36, 64, 3), dtype=np.uint8), "t")
    assert temps[0] == 0.0 and len(set(temps)) == len(temps)


def test_point_to_servo_round_trip():
    """Project a known 3D point into the wrist camera, answer that pixel as the
    VLM would, and check the controller servos straight back at it.

    This is the end-to-end guard on the camera convention. Reading depth as +z
    (OpenCV) while the extrinsics are OpenGL mirrors the target through the lens
    and sends the arm the other way, which every single-stage test still passes.
    """
    from policies.vlm_pinpoint.connector import fingertip_position, quat_rotate
    from policies.vlm_pinpoint.pointing_controller import (
        _SENSOR_H,
        _SENSOR_W,
        wrist_extrinsics,
    )

    ee_pos = np.array([0.45, 0.0, 0.35])
    ee_quat = np.array([0.0, 1.0, 0.0, 0.0])  # gripper pointing straight down
    cam_pos, cam_quat = wrist_extrinsics(ee_pos, ee_quat)
    cam_quat_inv = np.array([cam_quat[0], -cam_quat[1], -cam_quat[2], -cam_quat[3]])

    forward = quat_rotate(cam_quat, (0.0, 0.0, -1.0))  # OpenGL: -Z is forward
    right = quat_rotate(cam_quat, (1.0, 0.0, 0.0))
    down = quat_rotate(cam_quat, (0.0, -1.0, 0.0))
    target = cam_pos + 0.35 * forward + 0.05 * right + 0.06 * down

    # World -> OpenGL camera -> pinhole pixel.
    p_gl = quat_rotate(cam_quat_inv, target - cam_pos)
    x_cv, y_cv, z_cv = p_gl[0], -p_gl[1], -p_gl[2]
    fx, fy, cx, cy = intrinsics()
    u, v = fx * x_cv / z_cv + cx, fy * y_cv / z_cv + cy
    assert 0 <= u < _SENSOR_W and 0 <= v < _SENSOR_H, "fixture must be in frame"

    # The VLM sees the frame rotated 180°, so it reports the flipped pixel.
    point = {"x": round((_SENSOR_W - u) / _SENSOR_W * 1000),
             "y": round((_SENSOR_H - v) / _SENSOR_H * 1000),
             "phase": "APPROACH"}

    depth = np.full((_SENSOR_H, _SENSOR_W), np.inf)  # only the object reads back
    ui, vi = int(round(u)), int(round(v))
    depth[vi - 3:vi + 4, ui - 3:ui + 4] = z_cv
    obs = {"depth": depth, "K": None, "cam_pos": cam_pos, "cam_quat": cam_quat,
           "ee_pos": ee_pos, "ee_quat": ee_quat}

    tip = fingertip_position(ee_pos, ee_quat)
    delta, _, _ = PointingController().step(obs, point, tip)
    ideal = target - tip
    cos = float(delta @ ideal / (np.linalg.norm(delta) * np.linalg.norm(ideal)))
    assert cos > 0.999, f"servoing {np.degrees(np.arccos(cos)):.1f}° off target"


# ------------------------------------------------------------------
# Grasp yaw. A Robotiq 2F-85 opens 85 mm; a banana is ~190 mm long, so jaws
# that close along the long axis cannot grip it at all. The model has no
# rotation channel, so the angle is measured off the depth map.


def _nadir_scene(length=0.18, width=0.04, axis_deg=30.0, height=0.04):
    """Depth map of one elongated object on a table, seen from straight above."""
    from policies.vlm_pinpoint.pointing_controller import unproject

    cam_pos = np.array([0.5, 0.0, 1.0])
    cam_quat = np.array([1.0, 0.0, 0.0, 0.0])   # looking down world -Z
    fx = fy = 600.0
    K = np.array([[fx, 0.0, 160.0], [0.0, fy, 120.0], [0.0, 0.0, 1.0]])
    table_d, obj_d = 1.0, 1.0 - height

    depth = np.full((240, 320), table_d)
    th = np.radians(axis_deg)
    long_ax = np.array([np.cos(th), np.sin(th)])
    short_ax = np.array([-np.sin(th), np.cos(th)])
    centre = np.array([0.5, 0.0])
    for v in range(240):
        for u in range(320):
            p = unproject(u, v, obj_d, K=K, cam_pos=cam_pos, cam_quat=cam_quat)
            d = p[:2] - centre
            if abs(d @ long_ax) <= length / 2 and abs(d @ short_ax) <= width / 2:
                depth[v, u] = obj_d
    return depth, K, cam_pos, cam_quat, np.array([0.5, 0.0, height])


def test_long_axis_is_recovered_from_depth():
    from policies.vlm_pinpoint.pointing_controller import object_geometry

    for axis_deg in (0.0, 30.0, 75.0, 120.0):
        depth, K, cam_pos, cam_quat, target = _nadir_scene(axis_deg=axis_deg)
        geom = object_geometry(depth, 160, 120, target, K=K,
                               cam_pos=cam_pos, cam_quat=cam_quat)
        assert geom is not None and geom.long_axis_rad is not None, (
            f"no axis found at {axis_deg} deg"
        )
        err = (np.degrees(geom.long_axis_rad) - axis_deg + 90) % 180 - 90
        assert abs(err) < 5.0, f"axis off by {err:.1f} deg at {axis_deg}"


def test_object_height_is_measured_not_assumed():
    """support_z/top_z are what the closure height is derived from."""
    from policies.vlm_pinpoint.pointing_controller import object_geometry

    depth, K, cam_pos, cam_quat, target = _nadir_scene(height=0.04)
    geom = object_geometry(depth, 160, 120, target, K=K,
                           cam_pos=cam_pos, cam_quat=cam_quat)
    assert geom is not None
    assert abs(geom.support_z - 0.0) < 0.005, geom.support_z
    assert abs((geom.top_z - geom.support_z) - 0.04) < 0.01


def test_no_axis_for_a_round_object():
    """A ball has no long axis; inventing one would spin the wrist on noise."""
    from policies.vlm_pinpoint.pointing_controller import object_geometry

    depth, K, cam_pos, cam_quat, target = _nadir_scene(length=0.06, width=0.06)
    geom = object_geometry(depth, 160, 120, target, K=K,
                           cam_pos=cam_pos, cam_quat=cam_quat)
    assert geom is not None, "a round object is still measurable, just not oriented"
    assert geom.long_axis_rad is None


def test_no_geometry_when_nothing_stands_above_the_table():
    from policies.vlm_pinpoint.pointing_controller import object_geometry

    depth, K, cam_pos, cam_quat, target = _nadir_scene()
    flat = np.full_like(depth, 1.0)
    assert object_geometry(flat, 160, 120, target, K=K,
                           cam_pos=cam_pos, cam_quat=cam_quat) is None


def test_the_robots_own_hand_is_not_measured_as_the_object():
    """The gripper hovers over the target exactly when the grasp is planned.

    Unmasked it is segmented as part of the object: measured at 15.3 cm of
    "banana" on a 4 cm banana, which both corrupts the long axis and lifts
    ``top_z``, and the closure height is derived from that cloud.
    """
    from policies.vlm_pinpoint.pointing_controller import object_geometry

    depth, K, cam_pos, cam_quat, target = _nadir_scene(height=0.04)
    # A hand hovering 10 cm over the object, seen by the same nadir camera.
    tip = np.array([0.5, 0.0, 0.14])
    flange = tip + np.array([0.0, 0.0, 0.15])
    hand = depth.copy()
    hand[110:130, 150:170] = 1.0 - 0.14      # fingers, much nearer the camera

    # No target ceiling and no hand mask: nothing keeps the gripper out.
    dirty = object_geometry(hand, 160, 120, None, K=K,
                            cam_pos=cam_pos, cam_quat=cam_quat)
    clean = object_geometry(hand, 160, 120, target, K=K,
                            cam_pos=cam_pos, cam_quat=cam_quat,
                            flange_pos=flange, tip_pos=tip)
    assert dirty is not None and clean is not None
    assert dirty.top_z > 0.10, "precondition: the hand contaminates the cloud"
    assert abs((clean.top_z - clean.support_z) - 0.04) < 0.015, (
        f"hand still in the cloud: {100 * (clean.top_z - clean.support_z):.1f} cm"
    )


def test_closure_height_reaches_past_the_widest_part():
    """Closing above the equator squeezes a round object out of the jaws.

    Regression for a grasp that closed at 65% of a banana's height and came up
    empty: the depth sample is the VISIBLE TOP, and a fixed offset below it does
    not know how tall the object is.
    """
    from policies.vlm_pinpoint.pointing_controller import (
        GRASP_CLEARANCE_M,
        GRASP_DEPTH_M,
        MAX_BITE_M,
        ObjectGeometry,
        PointingController,
    )

    surface = 0.0426                      # measured banana top, robot frame
    geom = ObjectGeometry(support_z=0.0012, top_z=surface, long_axis_rad=None)

    fixed = surface - GRASP_DEPTH_M
    measured = PointingController._closure_height(surface, geom)
    height = geom.top_z - geom.support_z

    assert measured < fixed, "measured closure must go lower than the constant"
    # Below the equator (past the widest part): closing above it squeezes a
    # round object out of the jaws -- the regression above. Deliberately NOT
    # pinned to an exact height: the floor lives in FINGERTIP_FLOOR_M (8 mm
    # jammed in measurement), and this test owns the direction, not the constant.
    assert measured < 0.5 * (geom.support_z + geom.top_z)
    assert measured >= geom.support_z, "never through the table"
    assert (measured - geom.support_z) < 0.75 * height, "too high to hold"
    # Bounded: a bad support estimate cannot drive the hand through the table.
    wild = ObjectGeometry(support_z=-5.0, top_z=surface, long_axis_rad=None)
    assert PointingController._closure_height(surface, wild) >= surface - MAX_BITE_M
    # No measurement -> the old fixed behaviour, unchanged.
    assert PointingController._closure_height(surface, None) == fixed


def test_yaw_only_spins_the_wrist():
    """The top-down approach direction must survive the yaw correction.

    If aiming the jaws also tilted the approach axis, the fingers would stop
    coming down on the object and the orientation hold would fight the plan.
    """
    from policies.vlm_pinpoint.connector import quat_rotate
    from policies.vlm_pinpoint.pointing_controller import yaw_aligned_quat

    # base_link +X (fingers) -> world -Z, i.e. a real top-down pick.
    top_down = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    approach = quat_rotate(top_down, (1.0, 0.0, 0.0))
    assert np.allclose(approach, (0.0, 0.0, -1.0), atol=1e-9)

    for want in np.radians([0.0, 45.0, 120.0, 161.9, -80.0]):
        q = yaw_aligned_quat(top_down, want)
        assert np.allclose(quat_rotate(q, (1.0, 0.0, 0.0)), approach, atol=1e-9), (
            "yaw correction tilted the approach axis"
        )
        jaw = quat_rotate(q, (0.0, 1.0, 0.0))
        got = np.arctan2(jaw[1], jaw[0])
        # The jaw axis is symmetric: 180 deg apart is the same grasp.
        assert abs((np.degrees(got - want) + 90) % 180 - 90) < 1e-6


def test_yaw_correction_never_takes_the_long_way_round():
    """A 170 deg "correction" is really -10 deg: same grasp, no wrist windup."""
    from policies.vlm_pinpoint.connector import quat_rotate
    from policies.vlm_pinpoint.pointing_controller import yaw_aligned_quat

    top_down = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])
    jaw0 = quat_rotate(top_down, (0.0, 1.0, 0.0))
    start = np.arctan2(jaw0[1], jaw0[0])

    for want in np.radians(np.arange(-180.0, 181.0, 15.0)):
        q = yaw_aligned_quat(top_down, want)
        jaw = quat_rotate(q, (0.0, 1.0, 0.0))
        moved = np.degrees(np.arctan2(jaw[1], jaw[0]) - start)
        moved = (moved + 90) % 180 - 90
        assert abs(moved) <= 90.0 + 1e-6, f"wrist swung {moved:.1f} deg"


def test_grasp_plan_carries_the_yaw_on_every_leg():
    """Including the close and the lift: a wrist that unwinds mid-grasp twists
    the object out of the jaws."""
    obs, _, target = _grasp_obs()
    ctrl = PointingController()
    above = target + np.array([0.0, 0.0, PREGRASP_HEIGHT_M])

    from policies.vlm_pinpoint.pointing_controller import ObjectGeometry

    geom = ObjectGeometry(support_z=0.0, top_z=0.04, long_axis_rad=1.23)
    plan = ctrl._grasp_plan(target, above, geom)
    assert [wp.yaw_rad for wp in plan] == [1.23 + np.pi / 2] * len(plan)

    held = ctrl._grasp_plan(target, above, None)
    assert all(wp.yaw_rad is None for wp in held), "None must mean 'keep the latch'"

    # The descent -- and ONLY the descent -- treats contact as arrival. Without
    # this wiring the hand presses on the object with the jaws open until the
    # step budget runs out, then lifts away empty.
    above_wp, descend, close, lift = plan
    assert descend.on_stall == "advance"
    assert [above_wp.on_stall, close.on_stall, lift.on_stall] == ["replan"] * 3


def test_approach_descends_once_the_fingers_are_over_the_target():
    """A model that never says GRASP must not be able to stall the episode.

    Regression for a live-lock: APPROACH staged the fingers 10 cm over the object
    and then waited for the VLM to emit GRASP. A local Qwen emitted APPROACH 26
    times and GRASP zero times, so every decision re-planned the waypoint the arm
    was already standing on and the hand never closed. Arriving is the trigger.
    """
    obs, _, target = _grasp_obs()
    ctrl = PointingController()

    # Still on the way in: stage overhead, fingers open, nothing else.
    far = target + np.array([0.30, 0.0, 0.20])
    waypoints, done = ctrl.plan(obs, {"x": 500, "y": 500, "phase": "APPROACH"}, far)
    assert not done
    assert len(waypoints) == 1, "approaching from afar should only stage"
    assert waypoints[0].gripper == 0.0

    # Already over it: descend, close, lift -- without another round trip.
    staged = target + np.array([0.01, 0.0, PREGRASP_HEIGHT_M])
    waypoints, done = ctrl.plan(obs, {"x": 500, "y": 500, "phase": "APPROACH"}, staged)
    assert not done
    grippers = [wp.gripper for wp in waypoints]
    assert 1.0 in grippers, f"staged APPROACH never closes the hand: {grippers}"
    assert any(wp.pos is None for wp in waypoints), "no hold to let the fingers load"
    lowest = min(wp.pos[2] for wp in waypoints if wp.pos is not None)
    assert lowest < target[2], "descends to the object, not just to its surface"


def test_staged_approach_and_explicit_grasp_agree():
    """The auto-promoted plan is the GRASP primitive, not a second copy of it."""
    obs, _, target = _grasp_obs()
    ctrl = PointingController()
    staged = target + np.array([0.0, 0.0, PREGRASP_HEIGHT_M])

    auto, _ = ctrl.plan(obs, {"x": 500, "y": 500, "phase": "APPROACH"}, staged)
    told, _ = ctrl.plan(obs, {"x": 500, "y": 500, "phase": "GRASP"}, staged)

    assert len(auto) == len(told)
    for a, b in zip(auto, told, strict=True):
        assert a.gripper == b.gripper
        assert (a.pos is None) == (b.pos is None)
        if a.pos is not None:
            assert np.allclose(a.pos, b.pos)


def test_staging_latch_survives_pointing_noise():
    """The latch commits when truly overhead and servos otherwise.

    The staging point is recomputed from a fresh VLM point every decision, and
    a point carries centimetres of error at this range -- so the latch cannot
    be as tight as WAYPOINT_TOL_M (1.5 cm) or it never fires. But 0.06 committed
    with up to 6 cm of horizontal error on a 4 cm banana: measured air-closes in
    a descend-lift loop. The boundary is 0.03: inside it commits, outside it
    keeps servoing at height instead of spending a full descend on noise.
    """
    from policies.vlm_pinpoint.pointing_controller import (
        STAGED_RADIUS_M,
        WAYPOINT_TOL_M,
    )

    assert STAGED_RADIUS_M >= 2 * WAYPOINT_TOL_M

    obs, _, target = _grasp_obs()
    ctrl = PointingController()
    # 2 cm off, above the target: close enough to commit a grasp plan.
    near = target + np.array([0.02, 0.0, PREGRASP_HEIGHT_M])
    waypoints, _ = ctrl.plan(obs, {"x": 500, "y": 500, "phase": "APPROACH"}, near)
    assert 1.0 in [wp.gripper for wp in waypoints], "2 cm off must latch"
    # 5 cm off: genuinely not overhead -- stage at height, fingers open, no
    # grasp primitive spent on what is probably jitter.
    far = target + np.array([0.05, 0.0, PREGRASP_HEIGHT_M])
    waypoints, _ = ctrl.plan(obs, {"x": 500, "y": 500, "phase": "APPROACH"}, far)
    assert 1.0 not in [wp.gripper for wp in waypoints], "5 cm off must not commit"


def test_a_second_external_camera_cannot_be_silently_substituted():
    """The RGB frame and the unprojection frustum must be the same camera.

    ``camera_label`` names every non-wrist camera "AgentView (...)", so matching
    the view on that prefix would take whichever external camera sorts first
    while the geometry kept coming from ``over_shoulder_left_camera``. The model
    would be pointing on one image and the arm servoing through another -- silent,
    and indistinguishable from bad grounding.
    """
    from policies.vlm_agent.token_controller import camera_label

    external = ("over_shoulder_left_camera", "over_shoulder_left_cam")
    wanted = {camera_label(key) for key in external}

    # A different fixed camera in the same scene.
    intruder = camera_label("head_camera")
    assert intruder.startswith("AgentView"), "precondition: both are AgentView views"
    assert intruder not in wanted, (
        f"{intruder!r} would be accepted in place of the over-shoulder view"
    )
    assert camera_label("over_shoulder_left_camera") in wanted


def test_fingertip_offset_points_where_the_wrist_camera_looks():
    """The fingertip offset must run along base_link's +X, not +Z.

    This is the one geometric fact the whole policy rests on: ``ee_pos``/``ee_quat``
    report the Robotiq 2F-85 *mount flange*, and every waypoint is servoed against
    ``fingertip_position`` of it. The offset was once (0, 0, 0.1034) -- the Franka
    panda_hand TCP, a different robot's hand -- which put the estimated tip 18.5 cm
    off in a fixed direction, so the arm descended forever and never arrived.

    Asserted against a physical invariant rather than the constant itself: wrist_cam
    is bolted to base_link to watch the fingers, so its optical axis and the flange
    -> fingertip vector have to point essentially the same way.
    """
    from policies.vlm_pinpoint.connector import FLANGE_TO_FINGERTIP
    from policies.vlm_pinpoint.pointing_controller import _MOUNT_QUAT_WXYZ, quat_rotate

    tip_dir = np.asarray(FLANGE_TO_FINGERTIP, dtype=float)
    tip_dir /= np.linalg.norm(tip_dir)
    # OpenGL cameras look down their own -Z.
    cam_forward = quat_rotate(_MOUNT_QUAT_WXYZ, (0.0, 0.0, -1.0))
    cam_forward /= np.linalg.norm(cam_forward)

    cos = float(tip_dir @ cam_forward)
    assert cos > 0.9, (
        f"fingertip offset {tip_dir} is {np.degrees(np.arccos(cos)):.0f}° off the "
        f"wrist camera's optical axis {np.round(cam_forward, 3)} -- one of them is "
        "using the wrong body frame"
    )


def test_fingertip_offset_matches_the_measured_tcp():
    """Magnitude and frame agree with what the rest of the repo grades against."""
    from policies.vlm_pinpoint.connector import FLANGE_TO_FINGERTIP

    # robolab/eval/phases.py::TCP_OFFSET, "base_link -> fingertip midpoint,
    # base_link frame (measured)". Duplicated there and here on purpose: this
    # module must import without Isaac. The test is what keeps the two in step.
    assert np.allclose(FLANGE_TO_FINGERTIP, (0.15, 0.03, 0.0))


def test_the_marked_gripper_lands_on_the_real_fingertip():
    """Project the fingertip into a fixed external view and back out again.

    The green GRIPPER crosshair and the servo target come off the *same* tip
    estimate, so a crosshair that visibly misses the hand in the saved frames is
    the arm missing the object by the same vector. Round-tripping catches that
    without needing a rendered scene.
    """
    from policies.vlm_pinpoint.connector import fingertip_position
    from policies.vlm_pinpoint.pointing_controller import (
        project,
        quat_rotate,
        unproject,
    )

    # Over-shoulder left camera (robolab/variations/camera.py), which is what
    # --point-camera external uses.
    cam_pos = np.array([0.05, 0.57, 0.66])
    cam_quat = np.array([-0.393, -0.195, 0.399, 0.805])
    cam_quat = cam_quat / np.linalg.norm(cam_quat)
    fx = fy = 2.1 * 1280 / 5.376
    K = np.array([[fx, 0.0, 640.0], [0.0, fy, 360.0], [0.0, 0.0, 1.0]])

    ee_pos = np.array([0.45, 0.0, 0.35])
    ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
    tip = fingertip_position(ee_pos, ee_quat)

    uv = project(tip, K=K, cam_pos=cam_pos, cam_quat=cam_quat)
    assert uv is not None, "fingertip projected behind the external camera"
    u, v = uv
    assert 0 <= u < 1280 and 0 <= v < 720, f"fingertip off-frame at ({u:.0f}, {v:.0f})"

    # distance_to_image_plane, not ray length: project along the optical axis.
    axis = quat_rotate(cam_quat, (0.0, 0.0, -1.0))
    depth = float((tip - cam_pos) @ axis)
    back = unproject(u, v, depth, K=K, cam_pos=cam_pos, cam_quat=cam_quat)
    assert np.allclose(back, tip, atol=1e-6), f"{back} != {tip}"


# ------------------------------------------------------------------
# Primitives: one VLM decision -> a whole waypoint plan, so the model is not
# inside the control loop.


def _grasp_obs():
    """Wrist looking forward-and-down at an object 0.35 m out."""
    from policies.vlm_pinpoint.connector import fingertip_position, quat_rotate
    from policies.vlm_pinpoint.pointing_controller import wrist_extrinsics

    ee_pos = np.array([0.45, 0.0, 0.35])
    ee_quat = np.array([0.0, 1.0, 0.0, 0.0])
    cam_pos, cam_quat = wrist_extrinsics(ee_pos, ee_quat)
    target = cam_pos + 0.35 * quat_rotate(cam_quat, (0.0, 0.0, -1.0))

    obs = {"depth": np.full((720, 1280), 0.35), "K": None,
           "cam_pos": cam_pos, "cam_quat": cam_quat,
           "ee_pos": ee_pos, "ee_quat": ee_quat}
    tip = fingertip_position(ee_pos, ee_quat)
    return obs, tip, target


def test_grasp_plan_goes_over_the_object_before_descending():
    """A top-down pick must stage above the target, not drive down the
    line-of-sight into the table or the side of the object."""
    from policies.vlm_pinpoint.pointing_controller import (
        GRASP_DEPTH_M,
        PREGRASP_HEIGHT_M,
    )

    obs, tip, _ = _grasp_obs()
    ctrl = PointingController()
    waypoints, done = ctrl.plan(obs, {"x": 500, "y": 500, "phase": "GRASP"}, tip)

    assert done is False and len(waypoints) == 4
    above, descend, close, lift = waypoints
    assert above.gripper == 0.0 and descend.gripper == 0.0   # fingers open on the way in
    assert above.pos[2] > descend.pos[2]                      # comes down, not across
    assert np.allclose(above.pos[:2], descend.pos[:2])        # purely vertical descent
    # The descent is at LEAST the staging height plus the old fixed offset: the
    # closure height is measured off the object now, and measuring sends the
    # fingers lower than the constant ever did (it closed above the widest part).
    assert (above.pos[2] - descend.pos[2]) >= PREGRASP_HEIGHT_M + GRASP_DEPTH_M - 1e-9
    assert close.pos is None and close.gripper == 1.0
    assert close.hold_steps == ctrl.gripper_hold_steps        # binary joint must be HELD
    assert lift.gripper == 1.0 and lift.pos[2] > descend.pos[2]


def test_grasp_waypoint_sits_below_the_sampled_surface():
    """Depth gives the VISIBLE SURFACE; the fingers have to straddle the object."""
    from policies.vlm_pinpoint.pointing_controller import GRASP_DEPTH_M, MAX_BITE_M

    obs, tip, target = _grasp_obs()
    waypoints, _ = PointingController().plan(
        obs, {"x": 500, "y": 500, "phase": "GRASP"}, tip)
    z = waypoints[1].pos[2]
    # Never above the old fixed offset, never past the safety floor. Where it
    # lands between the two is whatever the object turned out to be.
    assert z <= target[2] - GRASP_DEPTH_M + 1e-9
    assert z >= target[2] - MAX_BITE_M - 1e-9


def test_approach_stages_above_and_keeps_fingers_open():
    from policies.vlm_pinpoint.pointing_controller import PREGRASP_HEIGHT_M

    obs, tip, target = _grasp_obs()
    waypoints, done = PointingController().plan(
        obs, {"x": 500, "y": 500, "phase": "APPROACH"}, tip)
    assert done is False and len(waypoints) == 1
    assert waypoints[0].gripper == 0.0
    assert waypoints[0].pos[2] == pytest.approx(target[2] + PREGRASP_HEIGHT_M)


def test_place_releases_only_after_arriving():
    obs, tip, _ = _grasp_obs()
    waypoints, _ = PointingController().plan(
        obs, {"x": 500, "y": 500, "phase": "PLACE"}, tip)
    assert [wp.gripper for wp in waypoints] == [1.0, 1.0, 0.0]  # holds, then opens
    assert waypoints[-1].pos is None                            # opens in place


def test_lift_needs_no_depth():
    """LIFT is 'straight up from here', so it must not require a valid reading."""
    _, tip, _ = _grasp_obs()
    waypoints, done = PointingController().plan(
        {}, {"x": 500, "y": 500, "phase": "LIFT"}, tip)
    assert done is False and waypoints[0].pos[2] > tip[2]
    assert np.allclose(waypoints[0].pos[:2], tip[:2])


def test_done_plan_is_terminal():
    _, tip, _ = _grasp_obs()
    waypoints, done = PointingController().plan(
        {}, {"x": 0, "y": 0, "phase": "DONE"}, tip)
    assert done is True and waypoints == []


# ------------------------------------------------------------------
# Client: waypoint servo + orientation hold.


def _client(**kw):
    from policies.vlm_pinpoint.client import PinpointClient

    class _Backend:
        def point(self, *a, **k):
            return {"x": 500, "y": 500, "phase": "APPROACH", "raw": ""}

    return PinpointClient(backend=_Backend(), ik_scale=0.5, **kw)


def test_servo_moves_toward_the_waypoint_at_a_bounded_speed():
    from policies.vlm_pinpoint.client import _EnvState
    from policies.vlm_pinpoint.pointing_controller import (
        SERVO_M_PER_STEP,
        Waypoint,
    )

    client = _client()
    state = _EnvState(plan=[Waypoint(np.array([1.0, 0.0, 0.0]), 0.0)])
    move = client._advance(state, np.zeros(3), 0)
    assert np.allclose(move, [SERVO_M_PER_STEP, 0.0, 0.0])
    assert state.gripper_cmd == 0.0


def test_servo_recomputes_against_the_live_tip():
    """The delta is re-measured every step, so IK slip cannot accumulate into
    the target the way a replayed precomputed delta does."""
    from policies.vlm_pinpoint.client import _EnvState
    from policies.vlm_pinpoint.pointing_controller import Waypoint

    client = _client()
    state = _EnvState(plan=[Waypoint(np.array([0.0, 1.0, 0.0]), 0.0)])
    client._advance(state, np.zeros(3), 0)
    # The arm slipped sideways; the command must point back at the waypoint.
    move = client._advance(state, np.array([0.3, 0.0, 0.0]), 0)
    assert move[0] < 0 and move[1] > 0


def test_servo_advances_on_arrival_and_dwells_for_the_grasp():
    from policies.vlm_pinpoint.client import _EnvState
    from policies.vlm_pinpoint.pointing_controller import Waypoint

    client = _client()
    state = _EnvState(plan=[Waypoint(np.zeros(3), 0.0),
                            Waypoint(None, 1.0, hold_steps=3)])
    assert np.allclose(client._advance(state, np.zeros(3), 0), 0)  # already there
    assert state.wp_idx == 1
    for _ in range(3):
        assert np.allclose(client._advance(state, np.zeros(3), 0), 0)
        assert state.gripper_cmd == 1.0  # closed command HELD across the dwell
    assert state.wp_idx == 2  # plan exhausted -> next infer() re-queries


def test_servo_gives_up_on_an_unreachable_waypoint():
    """An unreachable waypoint must not pin the episode."""
    from policies.vlm_pinpoint.client import _EnvState
    from policies.vlm_pinpoint.pointing_controller import (
        MAX_STEPS_PER_WAYPOINT,
        Waypoint,
    )

    client = _client()
    state = _EnvState(plan=[Waypoint(np.array([99.0, 0.0, 0.0]), 0.0)])
    for _ in range(MAX_STEPS_PER_WAYPOINT):
        client._advance(state, np.zeros(3), 0)
    assert state.plan == []  # re-plans from wherever the arm actually is


def test_infer_holds_the_orientation_it_latched():
    """Zeros in the rotation slots do NOT hold orientation in relative mode --
    they make the current (drifting) pose the next setpoint. See
    vlm_agent.kinematics.hold_orientation_rotvec."""
    from policies.vlm_pinpoint.client import PinpointClient

    latched = np.array([0.0, 1.0, 0.0, 0.0])
    drifted = np.array([0.0, 0.995, 0.0, 0.0998])  # ~11 deg off

    class _Stub(PinpointClient):
        quat = latched

        def _extract_observation(self, raw_obs, *, env_id: int = 0):
            return {"views": {}, "frame": np.zeros((36, 64, 3), np.uint8),
                    "tip": np.zeros(3), "ee_quat": self.quat, "harness_obs": {}}

        def _plan_next(self, state, *a, **k):
            state.plan = []  # no motion; this test is about the rotation slots

    client = _Stub.__new__(_Stub)
    PinpointClient.__init__(client, backend=_client().backend, ik_scale=0.5)

    assert np.allclose(client.infer({}, "t")["action"][3:6], 0)  # latch step
    client.quat = drifted
    assert np.linalg.norm(client.infer({}, "t")["action"][3:6]) > 0


# ------------------------------------------------------------------
# Fixed external view. CAMERA_ROTATIONS only turns the wrist, so flipping this
# one anyway would resolve every point to the opposite corner of the table.


def _external_camera():
    """The real OverShoulderLeftCameraCfg placement (robolab/variations/camera.py)."""
    cam_pos = np.array([0.05, 0.57, 0.66])
    cam_quat = np.array([-0.393, -0.195, 0.399, 0.805])  # opengl wxyz
    sw, sh = 1280, 720
    fx, fy = 2.1 * sw / 5.376, 2.1 * sh / 3.024
    K = np.array([[fx, 0, sw / 2], [0, fy, sh / 2], [0, 0, 1]])
    return cam_pos, cam_quat, K, sw, sh


def test_external_view_is_not_flipped():
    from policies.vlm_pinpoint.pointing_controller import frame_point_to_sensor

    assert frame_point_to_sensor(0, 0, 1280, 720, rot_deg=0) == (0.0, 0.0)
    assert frame_point_to_sensor(1000, 1000, 1280, 720, rot_deg=0) == (1280.0, 720.0)
    # ...while the wrist view still inverts, as before.
    assert frame_point_to_sensor(0, 0, 1280, 720, rot_deg=180) == (1280.0, 720.0)
    with pytest.raises(ValueError):
        frame_point_to_sensor(0, 0, 1280, 720, rot_deg=90)


def test_external_camera_point_round_trip():
    """A table object -> pixel -> the point a VLM would answer -> back to 3D."""
    from policies.vlm_pinpoint.connector import quat_rotate
    from policies.vlm_pinpoint.pointing_controller import (
        frame_point_to_sensor,
        project,
        sensor_point_to_frame,
    )

    cam_pos, cam_quat, K, sw, sh = _external_camera()
    banana = np.array([0.55, 0.05, 0.02])

    u, v = project(banana, K=K, cam_pos=cam_pos, cam_quat=cam_quat)
    assert 0 <= u < sw and 0 <= v < sh, "the fixed view must actually see the table"

    fx_n, fy_n = sensor_point_to_frame(u, v, sw, sh, rot_deg=0)
    su, sv = frame_point_to_sensor(round(fx_n * 1000), round(fy_n * 1000),
                                   sw, sh, rot_deg=0)
    cq_inv = np.array([cam_quat[0], -cam_quat[1], -cam_quat[2], -cam_quat[3]])
    z = -quat_rotate(cq_inv, banana - cam_pos)[2]
    back = unproject(su, sv, z, K=K, cam_pos=cam_pos, cam_quat=cam_quat)
    assert np.linalg.norm(back - banana) < 2e-3  # sub-millimetre, bar rounding


def test_plan_uses_the_view_rotation_from_the_observation():
    """The same point must resolve differently for a flipped and an upright view."""
    cam_pos, cam_quat, K, sw, sh = _external_camera()
    base = {"depth": np.full((sh, sw), 1.2), "K": K,
            "cam_pos": cam_pos, "cam_quat": cam_quat}
    point = {"x": 250, "y": 250, "phase": "APPROACH"}
    ctrl = PointingController()
    upright = ctrl.plan({**base, "view_rot_deg": 0}, point, np.zeros(3))[0][0].pos
    flipped = ctrl.plan({**base, "view_rot_deg": 180}, point, np.zeros(3))[0][0].pos
    assert not np.allclose(upright, flipped)


def test_state_text_calls_out_an_empty_grasp():
    """A closed empty gripper and a closed loaded one are the SAME picture, and
    that is exactly what the GRASP/LIFT/DONE decision turns on."""
    from policies.vlm_pinpoint.client import PinpointClient

    hobs = {"tip_pos": np.array([0.4, 0.0, 0.25]), "gripper_pos": 1.0}
    assert "EMPTY" in PinpointClient._state_text(hobs, None)
    assert "holding something" in PinpointClient._state_text(
        {**hobs, "gripper_pos": 0.6}, None)
    assert "OPEN" in PinpointClient._state_text({**hobs, "gripper_pos": 0.0}, None)
    assert "25.0 cm above the table" in PinpointClient._state_text(hobs, None)


def test_waypoint_budget_scales_with_distance():
    """A flat step cap starves long legs and not short ones.

    Relative IK achieves ~28% of what is commanded, so at SERVO_M_PER_STEP a
    0.10 m descent needs ~40 control steps but a 0.25 m lateral move needs ~100.
    A single 60-step cap let the descent finish and timed the travel out every
    time — an arm that goes down but never goes forward.
    """
    from policies.vlm_pinpoint.pointing_controller import (
        IK_ACHIEVED_FRACTION,
        SERVO_M_PER_STEP,
        waypoint_budget,
    )

    achieved = SERVO_M_PER_STEP * IK_ACHIEVED_FRACTION
    for dist in (0.05, 0.10, 0.25, 0.40):
        assert waypoint_budget(dist) > dist / achieved, f"{dist} m leg is starved"
    assert waypoint_budget(0.25) > waypoint_budget(0.10)


def test_contact_on_a_grasp_descent_closes_the_hand():
    """Fingers that stop making progress have ARRIVED, not failed.

    Regression for an empty pick: the descent pressed the open jaws onto the
    banana, made no further ground, burned its whole step budget standing there,
    then gave up and lifted away. On a grasp descent contact IS the arrival
    condition -- the hand must close where it stopped.
    """
    from policies.vlm_pinpoint.client import _EnvState
    from policies.vlm_pinpoint.pointing_controller import STALL_STEPS, Waypoint

    goal = np.array([0.0, 0.0, -0.10])
    blocked_at = -0.06          # the table: the fingers get no lower
    plan = [
        Waypoint(goal, 0.0, tol_m=0.008, on_stall="advance"),
        Waypoint(None, 1.0, 10),          # close and hold
    ]
    client = _client()
    state = _EnvState(plan=plan)
    tip, steps, touched = np.zeros(3), 0, None
    while state.wp_idx == 0 and steps < 500:
        tip = tip + client._advance(state, tip, 0) * 0.28
        if tip[2] <= blocked_at and touched is None:
            touched = steps
        tip[2] = max(tip[2], blocked_at)
        steps += 1

    assert state.plan, "the grasp was abandoned instead of closing"
    assert state.wp_idx == 1, "never advanced to the close"
    assert state.plan[state.wp_idx].gripper == 1.0
    # And it noticed promptly after contact, rather than standing on the object
    # for a whole step budget with the jaws open.
    assert touched is not None
    assert steps - touched <= STALL_STEPS + 2, (
        f"{steps - touched} steps of pressing before closing"
    )


def test_the_lift_waits_for_the_jaws_to_stop_moving():
    """Close first, THEN lift. Lifting mid-travel drags the object out.

    Judged relative to the jaws' own fastest window, never against an absolute
    speed: this gripper closes at 0.16 mm per control step, slower than any
    constant worth writing, so an absolute test calls a steadily closing gripper
    "stopped" and starts the lift mid-close -- which is how the banana kept
    sliding out.
    """
    from policies.vlm_pinpoint.client import _EnvState
    from policies.vlm_pinpoint.pointing_controller import (
        GRIPPER_SETTLE_STEPS,
        Waypoint,
    )

    rate = 0.01                       # normalised finger joint per control step
    client = _client()
    state = _EnvState(plan=[Waypoint(None, 1.0, 10), Waypoint(np.zeros(3), 1.0)])
    g = 0.0
    for _ in range(40):               # well past hold_steps, still travelling
        g += rate
        client._advance(state, np.zeros(3), 0, g)
    assert state.wp_idx == 0, "lifted while the jaws were still closing"

    # Now they stall on the object -- width stops changing -- so the lift may go.
    for _ in range(3 * GRIPPER_SETTLE_STEPS):
        client._advance(state, np.zeros(3), 0, g)
        if state.wp_idx == 1:
            break
    assert state.wp_idx == 1, "never lifted even once the jaws had stopped"


def test_the_dwell_ends_on_the_object_and_not_a_margin_short_of_it():
    """The primary close condition is a measurement, not a stopwatch.

    Four absolute "have they stopped?" thresholds in a row got this wrong, all
    for the same reason: at 0.18 mm per step the jaws look stationary long before
    they touch anything. The point cloud already knows how thick the banana is.

    And the target has to be reached from the CLOSED side. A rollout that ended
    the dwell 8 mm wider than the measured width stopped with the jaws around
    the banana but not on it -- and since the close command is held through the
    lift, the rest of the travel happened on the way up, which is the slip.
    """
    from policies.vlm_agent.token_controller import gripper_width_m
    from policies.vlm_pinpoint.client import _EnvState
    from policies.vlm_pinpoint.pointing_controller import Waypoint

    banana = 0.0428                        # as measured off the depth map
    client = _client()
    state = _EnvState(
        plan=[Waypoint(None, 1.0, 10, close_to_m=banana), Waypoint(np.zeros(3), 1.0)]
    )

    g = 0.0
    for _ in range(400):
        if state.wp_idx:
            break
        g += 0.002                         # ~0.17 mm of width per control step
        client._advance(state, np.zeros(3), 0, g)

    assert state.wp_idx == 1, "never finished closing"
    assert gripper_width_m(g) <= banana, (
        "lifted with the jaws still wider than the object: they never touched it"
    )


def test_the_measured_width_is_carried_into_the_grasp_plan():
    """A width measured off the depth map is useless if the plan drops it."""
    from policies.vlm_pinpoint.pointing_controller import (
        PointingController,
        object_geometry,
    )

    banana = 0.04
    depth, K, cam_pos, cam_quat, target = _nadir_scene(width=banana)
    geom = object_geometry(depth, 160.0, 120.0, target=target, K=K,
                           cam_pos=cam_pos, cam_quat=cam_quat)
    assert geom is not None
    assert geom.width_m == pytest.approx(banana, abs=0.01), (
        "the minor-axis extent is what the jaws have to close to"
    )

    plan = PointingController()._grasp_plan(target, target + [0, 0, 0.1], geom)
    dwell = [wp for wp in plan if wp.pos is None]
    assert dwell, "no dwell waypoint: nothing ever closes"
    assert dwell[0].close_to_m == pytest.approx(geom.width_m), (
        "closed to a guess instead of the measurement"
    )

    # Without a measurement it must say so rather than invent a number, so the
    # client falls back to the relative stop test instead of closing to a guess.
    assert PointingController()._grasp_plan(
        target, target + [0, 0, 0.1], None
    )[2].close_to_m is None


def test_jaws_that_have_not_started_moving_are_not_mistaken_for_finished():
    """The bug that kept sliding the banana out of the gripper.

    For the first steps after the close command the joint has not begun to
    travel, so the measured width sits constant at fully open -- indistinguishable
    from a gripper that has finished closing. Every stillness test said "settled",
    the lift started, and the fingers shut on the way up. Stillness may only mean
    "done" once the jaws have actually moved.
    """
    from policies.vlm_pinpoint.client import _EnvState
    from policies.vlm_pinpoint.pointing_controller import (
        GRIPPER_SETTLE_STEPS,
        Waypoint,
    )

    client = _client()
    state = _EnvState(plan=[Waypoint(None, 1.0, 10), Waypoint(np.zeros(3), 1.0)])

    # 25 steps of actuator latency: commanded, but not moving yet.
    for _ in range(25):
        client._advance(state, np.zeros(3), 0, 0.0)
    assert state.wp_idx == 0, "lifted before the jaws had even started to close"

    # They start travelling. Still not done.
    g = 0.0
    for _ in range(20):
        g += 0.02
        client._advance(state, np.zeros(3), 0, g)
    assert state.wp_idx == 0, "lifted while the jaws were still travelling"

    # They stall on the object: now the lift may go.
    for _ in range(3 * GRIPPER_SETTLE_STEPS):
        client._advance(state, np.zeros(3), 0, g)
        if state.wp_idx == 1:
            break
    assert state.wp_idx == 1, "never lifted after the jaws closed and stopped"


def test_a_fully_shut_gripper_replans_instead_of_lifting():
    """Closed on nothing must not lift: drop the plan so the next decision
    re-queries, instead of carrying empty jaws up and restarting the loop."""
    from policies.vlm_pinpoint.client import _EnvState
    from policies.vlm_pinpoint.pointing_controller import Waypoint

    client = _client()
    state = _EnvState(plan=[Waypoint(None, 1.0, 10), Waypoint(np.zeros(3), 1.0)])
    for _ in range(12):
        client._advance(state, np.zeros(3), 0, 1.0)   # fully shut
    assert state.plan == [], "miss must clear the plan, not advance to lift"


def test_a_loaded_gripper_still_advances_to_lift():
    """The counterpart: fingers stopped ON something (40 mm) keep the plan."""
    from policies.vlm_pinpoint.client import _EnvState
    from policies.vlm_pinpoint.pointing_controller import Waypoint

    client = _client()
    state = _EnvState(plan=[Waypoint(None, 1.0, 10), Waypoint(np.zeros(3), 1.0)])
    # Close 85 -> 40 mm across AND past the hold window (the settle detector
    # needs post-hold travel before stillness means anything), then sit still.
    for i in range(50):
        g = min(0.53, 0.53 * (i + 1) / 25)
        client._advance(state, np.zeros(3), 0, g)
    assert state.wp_idx >= 1


def test_jaws_blocked_wide_open_still_time_out():
    """The case the ceiling exists for, and it is not hypothetical.

    Fingers wedged on the table cannot move at all, so the width stays at fully
    open forever. The travel gate above would then wait forever -- which is a
    hang, not a grasp. The ceiling has to let the episode continue.
    """
    from policies.vlm_pinpoint.client import _EnvState
    from policies.vlm_pinpoint.pointing_controller import (
        GRIPPER_SETTLE_MAX_STEPS,
        Waypoint,
    )

    client = _client()
    state = _EnvState(
        plan=[Waypoint(None, 1.0, 10), Waypoint(np.array([9.0, 0.0, 0.0]), 1.0)])
    left_at = None
    for i in range(GRIPPER_SETTLE_MAX_STEPS + 5):
        client._advance(state, np.zeros(3), 0, 0.0)   # jammed wide open
        if state.wp_idx >= 1:
            left_at = i
            break
    assert left_at is not None, "a jammed gripper pinned the plan forever"
    assert left_at >= GRIPPER_SETTLE_MAX_STEPS - 1, "gave up on the close too early"


def test_a_stalled_leg_that_is_not_a_grasp_still_replans():
    """Only the grasp descent treats contact as arrival; everything else asks
    the model again rather than pretending it arrived."""
    from policies.vlm_pinpoint.client import _EnvState
    from policies.vlm_pinpoint.pointing_controller import Waypoint

    client = _client()
    state = _EnvState(plan=[Waypoint(np.array([0.30, 0.0, 0.0]), 0.0)])
    tip, steps = np.zeros(3), 0
    while state.plan and steps < 500:
        client._advance(state, tip, 0)       # tip never moves: fully blocked
        steps += 1
    assert not state.plan, "a blocked travel leg must force a re-plan"


def test_long_lateral_leg_completes_at_the_real_ik_rate():
    """Replay the servo against the measured 28% tracking, not perfect tracking."""
    from policies.vlm_pinpoint.client import _EnvState
    from policies.vlm_pinpoint.pointing_controller import (
        IK_ACHIEVED_FRACTION,
        Waypoint,
    )

    for goal in (np.array([0.25, 0.0, 0.0]), np.array([0.0, 0.0, -0.10])):
        client = _client()
        state = _EnvState(plan=[Waypoint(goal, 0.0)])
        tip, steps = np.zeros(3), 0
        while state.plan and state.wp_idx < len(state.plan) and steps < 500:
            tip = tip + client._advance(state, tip, 0) * IK_ACHIEVED_FRACTION
            steps += 1
        assert state.plan, f"{goal} leg was abandoned after {steps} steps"
        assert state.wp_idx >= 1


# ------------------------------------------------------------------
# Client re-query cadence helpers (pure numpy, no sim).


def _make_state(plan=(), plan_kind="", wp_idx=0):
    import dataclasses
    from policies.vlm_pinpoint.client import _EnvState
    st = _EnvState()
    st.plan, st.plan_kind, st.wp_idx = list(plan), plan_kind, wp_idx
    return st


def test_target_close_keeps_stable_plan():
    import numpy as np
    from policies.vlm_pinpoint.client import PinpointClient
    from policies.vlm_pinpoint.pointing_controller import Waypoint
    st = _make_state(plan=[Waypoint(np.array([0.5, 0.0, 0.1]), 0.0)],
                     plan_kind="APPROACH")
    assert PinpointClient._target_close(
        st, "ALIGN", np.array([0.51, 0.005, 0.1])) is True


def test_target_close_rejects_far_or_atomic():
    import numpy as np
    from policies.vlm_pinpoint.client import PinpointClient
    from policies.vlm_pinpoint.pointing_controller import Waypoint
    st = _make_state(plan=[Waypoint(np.array([0.5, 0.0, 0.1]), 0.0)],
                     plan_kind="APPROACH")
    assert PinpointClient._target_close(
        st, "ALIGN", np.array([0.9, 0.0, 0.1])) is False
    st_atomic = _make_state(plan=[Waypoint(np.array([0.5, 0.0, 0.1]), 1.0)],
                            plan_kind="GRASP")
    assert PinpointClient._target_close(
        st_atomic, "GRASP", np.array([0.5, 0.0, 0.1])) is False
    st_empty = _make_state(plan=[], plan_kind="")
    assert PinpointClient._target_close(
        st_empty, "APPROACH", np.array([0.5, 0.0, 0.1])) is False


def test_staged_grasp_plan_is_atomic_despite_approach_phase():
    """Regression: a staged auto-grasp under APPROACH must not be interrupted.

    Live symptom was tip bobbing 0.10-0.14 m forever: every descend died
    halfway when the next query window replaced the multi-leg grasp plan with
    a fresh single staging waypoint.
    """
    import numpy as np
    from policies.vlm_pinpoint.client import PinpointClient
    from policies.vlm_pinpoint.pointing_controller import Waypoint

    grasp_plan = [Waypoint(np.zeros(3), 0.0),
                  Waypoint(np.zeros(3), 0.0),
                  Waypoint(None, 1.0, 10),
                  Waypoint(np.zeros(3), 1.0)]
    assert PinpointClient._is_atomic("APPROACH", grasp_plan) is True
    servo_plan = [Waypoint(np.zeros(3), 0.0)]
    assert PinpointClient._is_atomic("APPROACH", servo_plan) is False
    assert PinpointClient._is_atomic("GRASP", servo_plan) is True
    assert PinpointClient._is_atomic("APPROACH", []) is False
