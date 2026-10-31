"""
Unit tests for the student components (src/student). The tests the course shipped are
kept unchanged; the three it marked TODO are written below, and further tests cover the
remaining components (visibility/design, data collection and its simulated-time cost,
the parameter model, identifiability, and an end-to-end calibration against the
handout's targets).

Run from the assignment2 folder:  python -m pytest tests -q
"""

import numpy as np
import pytest

from src import simulation, vct3, Rot, Frame
from src.simulation import clock, constants
from src.student import metrics, registration, workflow, kinematics, design


def random_theta(rng):
    return np.array([rng.uniform(0, 600), rng.uniform(-300, 300), rng.uniform(0, 600),
                     rng.uniform(-1.3, 1.3), rng.uniform(-1.3, 1.3), rng.uniform(-2.5, 2.5)])


def random_frame(rng, angle=1.0, dist=100.0):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    return Frame(Rot(vct3(axis), angle=rng.uniform(-angle, angle)), vct3(rng.uniform(-dist, dist, 3)))


def build_scene(seed=0):
    """Our Step-1 design wired through the simulator (fresh clock)."""
    clock.reset()
    rob = simulation.create_robot("rob", design.robot_pose(), seed=seed)
    trk = simulation.create_tracker("trk", design.tracker_pose(), seed=seed)
    mbee = simulation.create_robot_calibration_body(rob, design.mbee_layout())
    simulation.add_marker_bodies(trk, [mbee])
    return rob, trk, mbee


# ---------------------------------------------------------------- metrics

def test_frame_error_known_values():
    F_a = Frame(Rot(), vct3(1, 2, 3))
    F_b = Frame(Rot('z', 0.1), vct3(1, 2, 3))
    angle, dist = metrics.frame_error(F_a, F_b)
    assert np.isclose(angle, 0.1) and np.isclose(dist, 0.0)
    angle, dist = metrics.frame_error(F_a, Frame(Rot(), vct3(4, 6, 3)))
    assert np.isclose(angle, 0.0) and np.isclose(dist, 5.0)


def test_frame_error_is_invariant_to_common_left_factor():
    rng = np.random.default_rng(1)
    G, F_a, F_b = random_frame(rng), random_frame(rng), random_frame(rng)
    assert np.allclose(metrics.frame_error(F_a, F_b), metrics.frame_error(G * F_a, G * F_b))


def test_frame_error_large_angle_and_translation_in_reference_frame():
    # the angle is the true rotation angle (log map), not a small-angle approximation
    F_a = Frame(Rot(), vct3())
    F_b = Frame(Rot('y', 2.5), vct3())
    assert np.isclose(metrics.frame_error(F_a, F_b)[0], 2.5)
    # the translation is measured in F_a's frame: a pure offset gives the same distance whatever F_a's rotation
    F_a = Frame(Rot('x', 1.0), vct3(5, 5, 5))
    F_b = F_a * Frame(Rot(), vct3(0.3, 0.4, 0.0))
    assert np.isclose(metrics.frame_error(F_a, F_b)[1], 0.5)


def test_summarize_errors():
    s = metrics.summarize_errors([(0.0, 3.0), (0.0, 4.0)])
    assert s["n"] == 2 and np.isclose(s["rms_pos"], np.sqrt(12.5)) and np.isclose(s["max_pos"], 4.0)


def test_summarize_errors_angles_and_target_fraction():
    s = metrics.summarize_errors([(1e-4, 0.1), (3e-3, 0.1), (2e-4, 0.5)])
    assert np.isclose(s["max_angle"], 3e-3) and np.isclose(s["rms_angle"], np.sqrt((1e-8 + 9e-6 + 4e-8) / 3))
    assert np.isclose(s["frac_within_target"], 1 / 3)
    assert metrics.summarize_errors([])["n"] == 0


# ---------------------------------------------------------------- registration

def test_register_points_recovers_known_transform():
    rng = np.random.default_rng(2)
    F = random_frame(rng)
    a = [vct3(rng.uniform(-50, 50, 3)) for _ in range(6)]
    b = [F * p for p in a]
    F_est = registration.register_points(a, b)
    assert np.allclose(F_est.R.matrix, F.R.matrix, atol=1e-9)
    assert np.allclose(F_est.p.vec, F.p.vec, atol=1e-9)


def test_register_points_rejects_degenerate_input():
    with pytest.raises(ValueError):
        registration.register_points([vct3(0, 0, 0), vct3(1, 0, 0)], [vct3(0, 0, 0), vct3(1, 0, 0)])


def test_register_points_rejects_collinear_and_mismatched_input():
    line = [vct3(0, 0, 0), vct3(10, 0, 0), vct3(20, 0, 0), vct3(30, 0, 0)]
    with pytest.raises(ValueError):
        registration.register_points(line, line)
    with pytest.raises(ValueError):
        registration.register_points(design.mbee_layout(), design.mbee_layout()[:5])


def test_register_points_noise_level_and_proper_rotation():
    """With sigma = 0.1 mm per marker on our 8-marker body the registered position is off
    by ~0.035 mm and the orientation by ~6e-4 rad (1 sigma); the result is a proper rotation."""
    rng = np.random.default_rng(3)
    a = design.mbee_layout()
    pos, ang = [], []
    for _ in range(300):
        F = random_frame(rng, angle=0.5, dist=50.0)
        b = [F * p + vct3(rng.normal(0, 0.1, 3)) for p in a]
        F_est = registration.register_points(a, b)
        R = F_est.R.matrix
        assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)
        an, di = metrics.frame_error(F, F_est)
        pos.append(di)
        ang.append(an)
    assert 0.03 < np.sqrt(np.mean(np.square(pos))) < 0.09      # 0.1/sqrt(8) per axis -> ~0.06 mm norm
    assert 5e-4 < np.sqrt(np.mean(np.square(ang))) < 1.5e-3


def test_average_frames_of_identical_frames():
    rng = np.random.default_rng(3)
    F = random_frame(rng)
    F_avg = registration.average_frames([F] * 5)
    assert np.allclose(F_avg.R.matrix, F.R.matrix, atol=1e-9)
    assert np.allclose(F_avg.p.vec, F.p.vec, atol=1e-9)


def test_average_frames_is_a_proper_rotation():
    rng = np.random.default_rng(4)
    F_avg = registration.average_frames([random_frame(rng, angle=0.2) for _ in range(10)])
    R = F_avg.R.matrix
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-9) and np.isclose(np.linalg.det(R), 1.0)


def test_average_frames_reduces_noise():
    rng = np.random.default_rng(5)
    F = random_frame(rng)
    noisy = [F * Frame(Rot(vct3(rng.normal(size=3)), angle=rng.normal(0, 1e-3)), vct3(rng.normal(0, 0.1, 3))) for _ in range(100)]
    ang, dist = metrics.frame_error(F, registration.average_frames(noisy))
    assert dist < 0.06 and ang < 6e-4           # ~ 1/sqrt(100) of the single-frame noise


def test_refine_marker_body_reduces_layout_error():
    """A user-defined body has a manufacturing error of ~0.7 mm per marker axis. Refining
    it from 30 readings must bring the nominal layout to within ~0.1 mm of the true one
    (up to a rigid transform), and cost 30 * delta_t_samp of simulated time."""
    clock.reset()
    trk = simulation.create_tracker("trk", Frame(Rot(), vct3(0, 0, -1600)), seed=0)
    mb = simulation.define_marker_body("mb", [vct3(0, 0, 0), vct3(60, 0, 0), vct3(0, 60, 0), vct3(0, 0, 60), vct3(60, 60, 0)])
    placed = simulation.place_marker_body(mb, Frame(Rot('z', 0.3), vct3(100, 50, 0)))
    simulation.add_marker_bodies(trk, [placed])
    actual = [u.p for u in mb.get_actual_marker_positions()]          # ground truth, allowed in a test

    def layout_error(nominal):
        F = registration.register_points(nominal, actual)            # best rigid alignment nominal -> actual
        return np.sqrt(np.mean([(F * n - a).norm() ** 2 for n, a in zip(nominal, actual)]))

    before = layout_error(list(mb.nominal_marker_positions))
    t0 = clock.elapsed
    new = registration.refine_marker_body(trk, placed, n_readings=30)
    assert np.isclose(clock.elapsed - t0, 30 * constants.SAMPLE_DELTA_T_SEC)
    after = layout_error(new)
    assert before > 0.3                      # the manufacturing error is real
    assert after < before / 3 and after < 0.15
    assert all(np.allclose(n.vec, m.vec) for n, m in zip(mb.nominal_marker_positions, new))  # stored on the body


# ---------------------------------------------------------------- design / visibility

def test_design_envelope_is_visible_with_margin():
    """Every corner of the marker envelope (every marker of every reachable pose) is inside
    the frustum, with more margin than the worst-case robot placement error (~83 mm)."""
    trk = simulation.create_tracker("trk", design.tracker_pose())
    corners = design.envelope_corners()
    assert trk.is_visible(corners)
    shift = constants.ALPHA_MAX_ROB * max(c.norm() for c in corners) + constants.EPS_MAX_ROB
    assert shift < 85.0
    for d in (vct3(85, 0, 0), vct3(-85, 0, 0), vct3(0, 85, 0), vct3(0, -85, 0), vct3(0, 0, 85), vct3(0, 0, -85)):
        assert trk.is_visible([c + d for c in corners])
    assert all(74.9 < a.norm() <= 75.0 for a in design.mbee_layout())


def test_planned_poses_are_visible_and_within_limits():
    rob, trk, mbee = build_scene(seed=0)
    poses, info = workflow.plan_calibration_poses(rob, trk, design.mbee_layout(), desired_accuracy=1e-3, return_info=True)
    assert len(poses) == info["n_poses"] >= 30
    for th in poses:
        assert rob.within_limits(th)
        assert workflow.pose_is_visible(rob, trk, th, design.mbee_layout(), margin=85.0)
    # every joint is swept: at least 4 distinct values per joint
    P = np.array(poses)
    assert all(len(np.unique(np.round(P[:, k], 6))) >= 4 for k in range(6))
    # sizing: 1e-3 rad target with a 10% share of the budget for tracker noise -> 34 readings,
    # taken as 5 re-commands x 7 readings; 37 distinct poses -> 185 commands
    assert info["readings_requested"] == 34 and info["commands_per_pose"] == 5
    assert info["readings_per_command"] == 7 and info["readings_per_pose"] == 35
    assert info["n_distinct"] == 40 and len(poses) == 5 * 40        # home + 3 x 9 slide stops + 3 x 4 wrist stops
    assert np.isclose(info["slide_steps_mm"][0], 75.0)
    # the plan must fail loudly if asked for poses the tracker cannot see
    with pytest.raises(ValueError):
        workflow.plan_calibration_poses(rob, trk, design.mbee_layout(), desired_accuracy=1e-3, wrist_sweep=2.0,
                                        rest=np.array([0, 0, 0, 0, 0, 0.0]), visibility_margin=2000.0)


def test_plan_sizing_scales_with_desired_accuracy():
    rob, trk, mbee = build_scene(seed=0)
    _, loose = workflow.plan_calibration_poses(rob, trk, design.mbee_layout(), desired_accuracy=2e-3, return_info=True)
    _, tight = workflow.plan_calibration_poses(rob, trk, design.mbee_layout(), desired_accuracy=5e-4, return_info=True)
    assert tight["readings_per_pose"] > 10 * loose["readings_per_pose"]      # ~ (2e-3 / 5e-4)^2 = 16x, before rounding
    assert tight["predicted_time"]["total_s"] > loose["predicted_time"]["total_s"]


# ---------------------------------------------------------------- workflow

def test_order_poses_by_motion_time_reduces_travel():
    rob = simulation.create_robot("rob", Frame.eye(), seed=0)
    rng = np.random.default_rng(21)
    poses = [random_theta(rng) for _ in range(12)]
    ordered = workflow.order_poses_by_motion_time(rob, poses, start=np.zeros(6))

    def travel(seq):
        cur, t = np.zeros(6), 0.0
        for p in seq:
            t += rob.motion_time(cur, p)
            cur = p
        return t

    assert len(ordered) == 12 and {tuple(p) for p in ordered} == {tuple(p) for p in poses}
    assert travel(ordered) < travel(poses)

def test_measure_home_offsets_matches_encoders():
    rob = simulation.create_robot("rob", Frame.eye(), seed=9)
    offsets = workflow.measure_home_offsets(rob, n_homings=20)
    enc = rob.home_robot()
    assert offsets.shape == (6,)
    assert np.allclose(-offsets, enc, atol=4 * simulation.constants.SIGMA_RND.max())


def test_measure_home_offsets_cost_and_averaging():
    clock.reset()
    rob = simulation.create_robot("rob", Frame.eye(), seed=11)
    rob.move_joints([300, 0, 0, 0, 0, 0])
    t0 = clock.elapsed
    offsets = workflow.measure_home_offsets(rob, n_homings=5)
    # one real move home (~6 s) + four zero-length homings at t_settle each
    assert clock.elapsed - t0 < rob.motion_time([300, 0, 0, 0, 0, 0], np.zeros(6)) + 5 * rob.t_settle + 0.1
    # ground truth (test only): offsets ~= delta_theta_sys at home within the averaged random error
    assert np.allclose(offsets, rob._delta_theta_sys(np.zeros(6)), atol=4 * constants.SIGMA_RND / np.sqrt(5) + 1e-6)


def test_collect_calibration_data_counts_and_time():
    rob, trk, mbee = build_scene(seed=0)
    poses = [np.array([100, 0, 100, 0, 0, 0.0]), np.array([200, 50, 100, 0.2, 0, 0.0]), np.array([200, 50, 100, 0.2, 0, 0.0])]
    start = rob.theta_cmd.copy()
    expected_motion = sum(rob.motion_time(a, b) for a, b in zip([start] + poses[:-1], poses))
    t0 = clock.elapsed
    data = workflow.collect_calibration_data(rob, trk, mbee, poses, repeats=4)
    # the repeated third command is merged into the second sample: 2 samples, 4 + 8 readings, 3 moves
    assert len(data) == 2 and data.readings == [4, 8]
    assert np.isclose(data.t_collect, expected_motion + 3 * 4 * constants.SAMPLE_DELTA_T_SEC)
    assert np.isclose(clock.elapsed - t0, data.t_collect)
    assert all(np.allclose(th, p) for th, p in zip(data.thetas, poses[:2]))  # encoder values = commands
    # each stored frame is close to the truth (0.1 mm noise / 4 readings + <= 0.1 mm systematic field)
    F_trk = trk.nominal_pose()
    for th, F in zip(data.thetas, data.frames):
        ang, dist = metrics.frame_error(F_trk.inv() * rob.actual_ee_pose(th), F)
        assert dist < 0.3 and ang < 2e-3


def test_collect_skips_invisible_poses():
    clock.reset()
    rob = simulation.create_robot("rob", Frame.eye(), seed=0)
    trk = simulation.create_tracker("trk", Frame(Rot(), vct3(0, 0, -1600)), seed=0)   # robot's reach is mostly outside
    mbee = simulation.create_robot_calibration_body(rob, design.mbee_layout())
    simulation.add_marker_bodies(trk, [mbee])
    data = workflow.collect_calibration_data(rob, trk, mbee, [np.array([600, 300, 600, 0, 0, 0.0])], repeats=2)
    assert len(data) == 0 and len(data.skipped) == 1


def test_estimate_base_frame_on_synthetic_data():
    """With perfect nominal data, the estimate must reproduce F_X exactly."""
    rob = simulation.create_robot("rob", Frame.eye(), seed=0)
    rng = np.random.default_rng(5)
    F_X = random_frame(rng, angle=0.3, dist=300.0)
    data = workflow.CalibrationData()
    for _ in range(10):
        th = random_theta(rng)
        data.thetas.append(th)
        data.frames.append(F_X * rob.fwd_kins(th))
    F_est = workflow.estimate_base_frame(data, rob)
    assert np.allclose(F_est.R.matrix, F_X.R.matrix, atol=1e-9)
    assert np.allclose(F_est.p.vec, F_X.p.vec, atol=1e-6)


def test_estimate_base_frame_with_improved_model_and_merge():
    rob = simulation.create_robot("rob", Frame.eye(), seed=0)
    rng = np.random.default_rng(12)
    F_X = random_frame(rng, angle=0.3, dist=300.0)
    improved = kinematics.ImprovedKinematics(rob, Frame.eye())
    improved.params = 1e-3 * rng.normal(size=improved.params.shape)
    d1, d2 = workflow.CalibrationData(), workflow.CalibrationData()
    for d in (d1, d2):
        for _ in range(5):
            th = random_theta(rng)
            d.thetas.append(th)
            d.frames.append(F_X * improved.fwd_kins(th))
        d.t_collect = 1.5
    data = workflow.merge_data(d1, d2)
    assert len(data) == 10 and np.isclose(data.t_collect, 3.0)
    F_est = workflow.estimate_base_frame(data, improved)
    assert np.allclose(F_est.R.matrix, F_X.R.matrix, atol=1e-9) and np.allclose(F_est.p.vec, F_X.p.vec, atol=1e-6)


# ---------------------------------------------------------------- kinematics

def test_improved_kinematics_initial_params_match_nominal():
    rob = simulation.create_robot("rob", Frame.eye(), seed=0)
    rng = np.random.default_rng(6)
    F_X = random_frame(rng, angle=0.3, dist=300.0)
    improved = kinematics.ImprovedKinematics(rob, F_X)
    for _ in range(10):
        th = random_theta(rng)
        F_nom, F_imp = rob.fwd_kins(th), improved.fwd_kins(th)
        assert np.allclose(F_nom.R.matrix, F_imp.R.matrix, atol=1e-9)
        assert np.allclose(F_nom.p.vec, F_imp.p.vec, atol=1e-9)
    assert np.allclose(improved.base_frame().p.vec, F_X.p.vec, atol=1e-9)


def test_improved_inv_kins_round_trip():
    rob = simulation.create_robot("rob", Frame.eye(), seed=0)
    improved = kinematics.ImprovedKinematics(rob, Frame.eye())
    improved.params = improved.initial_params() + 1e-3 * np.random.default_rng(7).normal(size=improved.params.shape)
    rng = np.random.default_rng(8)
    for _ in range(10):
        th = random_theta(rng)
        F = improved.fwd_kins(th)
        th_back = improved.inv_kins(F, seed=th)
        F_back = improved.fwd_kins(th_back)
        assert metrics.frame_error(F, F_back)[1] < 1e-6
        assert metrics.frame_error(F, F_back)[0] < 1e-8


def test_parameter_layout_offsets_and_tilts():
    rob = simulation.create_robot("rob", Frame.eye(), seed=0)
    offsets = np.array([0.1, -0.1, 0.05, 0.005, -0.004, 0.003])
    improved = kinematics.ImprovedKinematics(rob, Frame.eye(), joint_offsets=offsets, n_poly=(2, 2, 2, 1, 1, 1))
    assert improved.params.shape == (22 + 9,) and len(improved.param_names) == 31
    # at home the distortion is exactly the homing offset; the polynomial adds nothing at theta = 0
    assert np.allclose(improved.joint_distortion(np.zeros(6)), offsets)
    p = improved.initial_params()
    p[improved.param_names.index("poly_j1_c1")] = 0.12            # 0.12 mm at theta_1 = 600 -> slope 2e-4
    assert np.isclose(improved.joint_distortion([600, 0, 0, 0, 0, 0], p)[0], 0.1 + 0.12)
    assert np.isclose(improved.model.joint_scale(p)[0], 0.12 / 600.0)
    # a tilt parameter rotates the axis by exactly that angle, about a perpendicular nominal axis
    p = improved.initial_params()
    p[improved.param_names.index("tilt_axis2_z")] = 0.01          # rotate d_2 = y about z
    d2 = improved.axes(p)[1]
    assert np.isclose(np.degrees(np.arccos(d2 @ [0, 1, 0])), np.degrees(0.01)) and np.isclose(d2[2], 0.0)
    assert np.allclose(improved.axes(p)[0], [1, 0, 0])            # d_1 is never tilted
    # the tilted model moves the end effector the way the real robot would: a 0.01 rad tilt of d_2
    # displaces the real travel theta*_2 = 300 + offset along y by theta*_2 sin(0.01) in -x
    F0, F1 = improved.fwd_kins([0, 300, 0, 0, 0, 0], improved.initial_params()), improved.fwd_kins([0, 300, 0, 0, 0, 0], p)
    assert np.isclose((F1.p - F0.p).x, -(300 + offsets[1]) * np.sin(0.01), atol=1e-6)


def test_residuals_vanish_on_self_generated_data_and_weigh_rotation():
    rob, trk, mbee = build_scene(seed=0)
    rng = np.random.default_rng(13)
    improved = kinematics.ImprovedKinematics(rob, random_frame(rng, 0.2, 200.0), rot_weight=200.0)
    improved.params = 2e-3 * rng.normal(size=improved.params.shape)
    data = workflow.CalibrationData()
    for _ in range(6):
        th = random_theta(rng)
        data.thetas.append(th)
        data.frames.append(improved.predict_in_tracker(th))
    r = improved.residuals(improved.params, data)
    assert r.shape == (36,) and np.allclose(r, 0.0, atol=1e-9)
    # a 1e-3 rad rotation of one observation shows up as 0.2 mm-equivalent in the residual
    data.frames[0] = data.frames[0] * Frame(Rot('z', 1e-3), vct3())
    r = improved.residuals(improved.params, data)
    assert np.isclose(np.linalg.norm(r[:3]), 0.2, rtol=1e-3) and np.allclose(r[3:], 0.0, atol=1e-9)


def test_fit_recovers_synthetic_parameters():
    """Noise-free data from our own model with a known, non-zero parameter vector (including
    a non-zero gauge component), fitted from zero: the fitted model must reproduce the
    generating model's poses on new joint vectors even though its parameters differ in the
    gauge direction."""
    rob, trk, mbee = build_scene(seed=0)
    rng = np.random.default_rng(14)
    F_X_init = design.tracker_pose().inv() * rob.F_rob
    truth = kinematics.ImprovedKinematics(rob, F_X_init, joint_offsets=np.array([0.1, -0.1, 0.05, 0.005, -0.004, 0.003]),
                                          n_poly=(2, 2, 2, 1, 1, 1))
    truth.params = np.concatenate([rng.normal(0, 0.02, 3), rng.normal(0, 3.0, 3),          # base twist (gauge included)
                                   rng.normal(0, 0.01, 10),                                 # tilts
                                   rng.normal(0, 0.005, 3), rng.normal(0, 0.3, 3),          # tool twist
                                   rng.normal(0, 0.1, 6), rng.normal(0, 2e-4, 3)])          # polynomials
    poses = workflow.plan_calibration_poses(rob, trk, design.mbee_layout(), desired_accuracy=1e-3)
    data = workflow.CalibrationData()
    for th in poses:
        data.thetas.append(th)
        data.frames.append(truth.predict_in_tracker(th))
    est = kinematics.ImprovedKinematics(rob, F_X_init, joint_offsets=truth.joint_offsets, n_poly=(2, 2, 2, 1, 1, 1))
    info = est.fit(data)
    assert info["rms_pos_residual_mm"] < 1e-6 and info["rms_rot_residual_rad"] < 1e-8
    for _ in range(20):
        th = random_theta(rng)
        ang, dist = metrics.frame_error(truth.predict_in_tracker(th), est.predict_in_tracker(th))
        assert dist < 1e-5 and ang < 1e-7
    # the parameters are NOT equal (the gauge direction differs) but the gauge-free quantities are
    assert not np.allclose(est.params, truth.params, atol=1e-6)
    assert np.allclose(est.parameters_report()["axis_angles_deg"], truth.parameters_report()["axis_angles_deg"], atol=1e-7)
    assert np.allclose(est.parameters_report()["joint_scale"], truth.parameters_report()["joint_scale"], atol=1e-9)


def test_jacobian_rank_report_finds_the_gauge_null_direction():
    rob, trk, mbee = build_scene(seed=0)
    improved = kinematics.ImprovedKinematics(rob, design.tracker_pose().inv() * rob.F_rob, n_poly=(2, 2, 2, 1, 1, 1))
    poses = workflow.plan_calibration_poses(rob, trk, design.mbee_layout(), desired_accuracy=1e-3)
    data = workflow.CalibrationData()
    for th in poses:
        data.thetas.append(th)
        data.frames.append(improved.predict_in_tracker(th))
    rep = improved.jacobian_rank_report(data)
    assert rep["n_params"] == 31 and rep["rank"] == 30                    # exactly one unobservable direction ...
    names = {n for n, _ in rep["weakest_directions"][0]["direction"]}
    gauge = {"base_alpha_x", "tilt_axis2_x", "tilt_axis3_x", "tilt_axis5_x", "tilt_axis6_x", "tool_alpha_x"}
    assert len(names & gauge) >= 5                                        # the six equal-weight gauge components
    assert rep["free"]["rank"] == rep["free"]["n_params"] == 30            # ... which fixing the gauge removes
    assert rep["free"]["condition_number"] < 1e4


# ---------------------------------------------------------------- end to end

def test_end_to_end_calibration_meets_targets():
    """The full pipeline on one robot seed: the calibrated model must predict the robot's
    actual end-effector pose (ground truth, allowed here) within the handout's targets
    over random joint vectors, in well under the 900 s soft time budget."""
    rob, trk, mbee = build_scene(seed=0)
    home_offsets = workflow.measure_home_offsets(rob, n_homings=5)
    poses, info = workflow.plan_calibration_poses(rob, trk, design.mbee_layout(), desired_accuracy=1e-3, return_info=True)
    data = workflow.collect_calibration_data(rob, trk, mbee, poses, repeats=info["readings_per_command"])
    # one sample per run of identical consecutive commands (the home pose and the first stop of joint 1 coincide)
    n_groups = 1 + sum(not np.array_equal(a, b) for a, b in zip(poses[:-1], poses[1:]))
    assert len(data) == n_groups and len(data.skipped) == 0
    assert sum(data.readings) == len(poses) * info["readings_per_command"]
    F_X_nom = workflow.estimate_base_frame(data, rob)
    improved = kinematics.ImprovedKinematics(rob, F_X_nom, joint_offsets=home_offsets, n_poly=(2, 2, 2, 1, 1, 1))
    fit = improved.fit(data)
    assert fit["status"] > 0 and fit["rms_pos_residual_mm"] < 0.1
    t_cal = clock.elapsed
    assert t_cal < constants.TARGET_TIME_SEC
    F_trk = trk.nominal_pose()
    rng = np.random.default_rng(20)
    errs_nom, errs_imp = [], []
    for _ in range(150):
        th = random_theta(rng)
        truth = F_trk.inv() * rob.actual_ee_pose(th)
        errs_nom.append(metrics.frame_error(truth, F_X_nom * rob.fwd_kins(th)))
        errs_imp.append(metrics.frame_error(truth, improved.predict_in_tracker(th)))
    s_nom, s_imp = metrics.summarize_errors(errs_nom), metrics.summarize_errors(errs_imp)
    assert s_nom["max_pos"] > 1.0                                        # the nominal model really is off
    assert s_imp["max_pos"] < constants.TARGET_POS_ERR_MM and s_imp["max_angle"] < constants.TARGET_ANG_ERR_RAD
    # the parameters are close to ground truth (gauge-free comparison)
    cmp = improved.compare_with_ground_truth(rob.ground_truth_parameters(), F_trk)
    assert np.all(np.abs(cmp["axis_angles_deg_err"]) < np.degrees(5e-4))
    assert np.all(cmp["axis_direction_err_mrad"] < 0.5)
    # the distortion functions agree over the whole travel to within the tracker's systematic field
    # (<= 0.1 mm, which the fit cannot tell apart from a slide's scale error) and to ~1e-4 rad on the wrist
    assert np.all(cmp["joint_distortion_err_max"][:3] < 0.1) and np.all(cmp["joint_distortion_err_max"][3:] < 5e-4)
    assert np.all(np.abs(cmp["joint_scale_err_effective"][:3]) < 3e-4)
