"""
Data-collection workflow: which robot poses to visit, how to read them, and
how to register the robot base to the tracker.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from data_types.nominal_types import vct3, Frame

from ..robot import Robot
from assignments.assignment1.src.tracker import Tracker
from ..simulation import PlacedBody
from .. import simulation
from ..simulation import constants, clock
from . import _core
from ._convert import to_np, to_frame, pts_to_np


@dataclass
class CalibrationData:
    """One calibration data set.

    thetas[j] is the commanded (encoder) joint vector and frames[j] the
    calibration body's observed pose relative to the tracker, F_ee,j (this
    handout's tracker has no jiggle, so the tracker frame is a fixed
    reference). t_collect is the simulated time the collection took.
    readings[j] is how many tracker readings were averaged into frames[j];
    skipped lists the commanded poses at which the body was never visible.
    """
    thetas: List[np.ndarray] = field(default_factory=list)
    frames: List[Frame] = field(default_factory=list)
    t_collect: float = 0.0
    readings: List[int] = field(default_factory=list)
    skipped: List[np.ndarray] = field(default_factory=list)

    def __len__(self):
        return len(self.thetas)

    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        """(N,6) joint vectors and (N,4,4) observed frames, for the numpy core."""
        return np.array(self.thetas, dtype=np.float64), np.array([to_np(F) for F in self.frames])


def merge_data(*datasets: CalibrationData) -> CalibrationData:
    """Concatenate data sets (e.g. the Step-6 baseline and the Step-7 sweeps); the time
    already spent on each is summed."""
    out = CalibrationData()
    for d in datasets:
        out.thetas += [np.asarray(t, dtype=np.float64) for t in d.thetas]
        out.frames += list(d.frames)
        out.readings += list(d.readings) if d.readings else [1] * len(d.thetas)
        out.skipped += list(d.skipped)
        out.t_collect += d.t_collect
    return out


# --------------------------------------------------------------------------------- homing

def measure_home_offsets(rob: Robot, n_homings: int = 5) -> np.ndarray:
    """
    HomeRobot() drives the robot to theta* = 0 and returns the encoder values there,
    i.e. -delta_theta_sys(home) - delta_theta_rnd. Repeating it and averaging
    estimates the systematic joint offsets at home with the random part reduced by
    sqrt(n_homings) (0.005 mm / 1e-4 rad -> 0.002 mm / 4.5e-5 rad for n = 5). After
    the first homing each repeat is a zero-length move and costs only t_settle.

    input:
        rob - the robot (rob.home_robot())
        n_homings - how many homings to average
    output: estimated delta_theta_sys at home, np.ndarray of 6
    """
    if n_homings < 1:
        raise ValueError("measure_home_offsets: n_homings must be >= 1")
    enc = np.array([np.asarray(rob.home_robot(), dtype=np.float64).reshape(6) for _ in range(n_homings)])
    return -enc.mean(axis=0)


# --------------------------------------------------------------------------------- planning

LAST_PLAN_INFO: Dict = {}


def _markers_world(rob: Robot, theta, mbee_local: np.ndarray) -> np.ndarray:
    F = to_np(rob.F_rob) @ to_np(rob.fwd_kins(theta))
    return mbee_local @ F[:3, :3].T + F[:3, 3]


def pose_is_visible(rob: Robot, trk: Tracker, theta, mbee_local_positions, margin: float = 0.0) -> bool:
    """All markers of the calibration body visible at the NOMINAL pose, with every
    marker also displaced by +-margin along each axis (covers the real base's
    placement error, which is at most ~83 mm at the envelope corners)."""
    a = pts_to_np(mbee_local_positions)
    pts = _markers_world(rob, theta, a)
    if margin > 0:
        shifts = margin * np.vstack([np.eye(3), -np.eye(3)])
        pts = np.vstack([pts] + [pts + s for s in shifts])
    return trk.is_visible([vct3(p) for p in pts])


def predicted_collection_time(rob: Robot, poses: List[np.ndarray], readings_per_pose: int = 1,
                              start=None) -> Dict[str, float]:
    """Simulated time the plan will cost: motion (incl. t_settle per command) + sampling."""
    cur = np.asarray(rob.theta_cmd if start is None else start, dtype=np.float64)
    t_move = 0.0
    for th in poses:
        t_move += rob.motion_time(cur, th)
        cur = np.asarray(th, dtype=np.float64)
    t_samp = len(poses) * readings_per_pose * constants.SAMPLE_DELTA_T_SEC
    return {"motion_s": t_move, "sampling_s": t_samp, "total_s": t_move + t_samp}


def order_poses_by_motion_time(rob: Robot, poses: List[np.ndarray], start=None) -> List[np.ndarray]:
    """Greedy nearest-neighbour ordering of a set of poses by rob.motion_time (slowest joint), so
    that a test or calibration run spends as little simulated time travelling as possible."""
    remaining = [np.asarray(p, dtype=np.float64) for p in poses]
    cur = np.asarray(rob.theta_cmd if start is None else start, dtype=np.float64)
    ordered = []
    while remaining:
        j = int(np.argmin([rob.motion_time(cur, p) for p in remaining]))
        cur = remaining.pop(j)
        ordered.append(cur)
    return ordered


def plan_calibration_poses(rob: Robot, trk: Tracker, mbee_local_positions: List[vct3], desired_accuracy: float,
                           **kwargs) -> List[np.ndarray]:
    """
    Method 1 of the handout: home, then sweep each joint on its own while the other
    joints rest at their nominal home value. Joints 1-3 (translation) are decoupled
    from 4-6 (rotation), so every parameter of the model is excited by exactly one
    sweep: slide sweeps give the slide directions and distortion, wrist sweeps give the
    wrist axes, the wrist distortion and the tool frame, and the home pose anchors
    the base frame. The robot does not return home between sweeps: the first stop of
    the next sweep moves the previous joint back in the same command (motion time is
    set by the slowest joint), which saves ~85 s with no loss of accuracy (tested).

    Sizing from desired_accuracy (target RMS orientation error, rad):
      * readings per pose M: one registration of the body has orientation noise
        sigma_1 = sigma_b / sqrt(lambda_min(layout)) (5.8e-4 rad for our 8-marker, 75 mm
        cube); we give tracker noise a fraction f = budget_fraction of the budget, so
        M = ceil((sigma_1 / (f * desired_accuracy))^2) = 34 for f = 0.1. The readings are
        spread over commands_per_pose = 5 re-commands of the same pose (ceil(34/5) = 7 readings
        each): re-commanding costs only t_settle = 0.05 s and re-rolls the joint repeatability
        noise (1e-4 rad per wrist joint, the same size as the averaged tracker noise), which
        measurably improves the orientation accuracy (report, trade-off table).
      * slide step h: linear interpolation of the joint distortion between stops is off
        by at most omega_sys h / 2 (slope bound), so h <= 2 f pos_accuracy / omega_sys =
        80 mm -> 75 mm (9 stops over 600 mm). Our polynomial model is not an interpolator,
        but the same spacing keeps the stops dense enough to resolve the distortion's
        slowest ripple (period >= 180 mm) and averages tracker noise along the axis.
      * wrist stops: +-w and +-w/2 with w = wrist_sweep (default 2.0 rad, inside the
        +-0.75 pi limits). The wrist axis direction error is ~ sigma_theta / w, and
        the linear wrist distortion term needs a large lever; +-1 rad was measured to
        be marginal, +-2 rad comfortable (report, trade-off table).
      * the wrist distortion's slope bound gives |d(dtheta_sys)/dtheta| * w <= 5e-5 * 2
        = 1e-4 rad between home and the sweep ends, so no finer wrist sampling is needed.

    kwargs (all optional):
        pos_accuracy (0.2 mm), budget_fraction (0.1), slide_step (mm), wrist_sweep (rad),
        wrist_stops (explicit list), readings_per_pose (override M, the total per distinct pose),
        commands_per_pose (list each pose this many times in a row, 5 by default; the readings are
        split over the commands: readings_per_command = ceil(M / commands)), rest (joint vector the non-swept joints hold,
        default zeros), include_home (True), visibility_margin (85 mm), return_info (False:
        return only the list; True: return (poses, info dict)).
    output: ordered List of joint vectors (np.ndarray of 6)
    """
    pos_accuracy = float(kwargs.get("pos_accuracy", constants.TARGET_POS_ERR_MM))
    f = float(kwargs.get("budget_fraction", 0.1))
    a = pts_to_np(mbee_local_positions)
    sigma1 = _core.orientation_noise_per_reading(a, constants.SENSOR_NOISE_STD)
    M_total = int(kwargs.get("readings_per_pose") or _core.readings_needed(sigma1, desired_accuracy, f))
    commands = max(int(kwargs.get("commands_per_pose", 5)), 1)
    per_command = int(np.ceil(M_total / commands))
    M = per_command * commands                          # readings actually taken per distinct pose
    rest = np.zeros(6) if kwargs.get("rest") is None else np.asarray(kwargs["rest"], dtype=np.float64).reshape(6)
    margin = float(kwargs.get("visibility_margin", 85.0))

    # slide stops
    h = kwargs.get("slide_step")
    slide_stops = []
    for k in range(3):
        travel = rob.theta_max[k] - rob.theta_min[k]
        hk = float(h) if h else _core.slide_step_from_bounds(constants.OMEGA_SYS[k], pos_accuracy, f, travel)
        n = max(int(round(travel / hk)), 1)
        slide_stops.append(np.linspace(rob.theta_min[k], rob.theta_max[k], n + 1))
    # wrist stops
    w = float(kwargs.get("wrist_sweep", 2.0))
    wrist_stops = kwargs.get("wrist_stops")
    wrist_stops = np.array(sorted({-w, -w / 2, w / 2, w})) if wrist_stops is None else np.asarray(wrist_stops, dtype=np.float64)

    distinct = []
    if kwargs.get("include_home", True):
        distinct.append(rest.copy())
    for k in range(3):
        for t in slide_stops[k]:
            th = rest.copy()
            th[k] = t
            distinct.append(th)
    for k in range(3, 6):
        for t in wrist_stops:
            th = rest.copy()
            th[k] = np.clip(rest[k] + t, rob.theta_min[k], rob.theta_max[k])
            distinct.append(th)

    bad = []
    for th in distinct:
        if not rob.within_limits(th):
            bad.append(("limits", th))
        elif not pose_is_visible(rob, trk, th, mbee_local_positions, margin):
            bad.append(("visibility", th))
    if bad:
        raise ValueError("plan_calibration_poses: %d poses violate %s, e.g. %s" % (len(bad), bad[0][0], np.round(bad[0][1], 3).tolist()))

    poses = [th.copy() for th in distinct for _ in range(commands)]
    timing = predicted_collection_time(rob, poses, readings_per_pose=per_command)
    info = {
        "readings_per_pose": M, "commands_per_pose": commands, "readings_per_command": per_command,
        "readings_requested": M_total,
        "sigma_orientation_per_reading_rad": sigma1, "sigma_position_per_reading_mm": _core.position_noise_per_reading(a, constants.SENSOR_NOISE_STD),
        "expected_orientation_noise_rad": sigma1 / np.sqrt(M), "budget_fraction": f,
        "slide_steps_mm": [float(s[1] - s[0]) for s in slide_stops], "slide_stops": [s.tolist() for s in slide_stops],
        "wrist_stops_rad": wrist_stops.tolist(), "n_poses": len(poses), "n_distinct": len(distinct),
        "predicted_time": timing,
    }
    LAST_PLAN_INFO.clear()
    LAST_PLAN_INFO.update(info)
    if kwargs.get("return_info", False):
        return poses, info
    return poses


# --------------------------------------------------------------------------------- collection

def collect_calibration_data(rob: Robot, trk: Tracker, mbee_placed: PlacedBody,
                             thetas: List[np.ndarray], repeats: int = 1) -> CalibrationData:
    """
    Visit each joint vector with rob.move_joints, take `repeats` tracker readings and
    store the calibration body's frame relative to the tracker. The raw marker
    positions of the readings are averaged first and registered once with our own
    register_points (noise falls as 1/sqrt(repeats); a registration of averaged
    markers is slightly better than an average of registrations). A pose listed
    several times IN A ROW is re-commanded each time (t_settle, and the joint
    repeatability noise is re-rolled) and all its readings are averaged into ONE
    stored sample, so the data set stays small for the fit. A pose at which the
    body is never visible is skipped (recorded in data.skipped) rather than stored.

    input:
        rob, trk - robot and tracker
        mbee_placed - the robot calibration body (attached)
        thetas - joint vectors to visit, in order
        repeats - readings per command (each costs delta_t_samp)
    output: CalibrationData (readings[j] = total readings behind frames[j])
    """
    if repeats < 1:
        raise ValueError("collect_calibration_data: repeats must be >= 1")
    body = mbee_placed.body
    if mbee_placed not in trk.tracked_bodies:
        simulation.add_marker_bodies(trk, [mbee_placed])
    nominal = pts_to_np(body.nominal_marker_positions)
    data = CalibrationData()
    t0 = clock.elapsed
    groups = []                                   # consecutive identical commands -> one group
    for th in thetas:
        th = np.asarray(th, dtype=np.float64).reshape(6)
        if groups and np.array_equal(groups[-1][0], th):
            groups[-1][1] += 1
        else:
            groups.append([th, 1])
    for th, n_cmd in groups:
        B = []
        for _ in range(n_cmd):
            rob.move_joints(th)
            for _ in range(repeats):
                obs = simulation.sample_marker_bodies(trk).get(body.name)
                if obs is not None:
                    B.append(pts_to_np(obs.markers))
        enc = np.asarray(rob.state().theta, dtype=np.float64).reshape(6).copy()
        if not B:
            data.skipped.append(enc)
            continue
        F = _core.register_points(nominal, np.mean(B, axis=0))
        data.thetas.append(enc)
        data.frames.append(to_frame(F))
        data.readings.append(len(B))
    data.t_collect = clock.elapsed - t0
    return data


# --------------------------------------------------------------------------------- base frame

def estimate_base_frame(data: CalibrationData, kins: Robot) -> Frame:
    """
    Every sample satisfies F_ee,j = F_X * kins.fwd_kins(theta_j) for one constant
    unknown F_X (the robot base in the tracker frame), so each sample gives an estimate
    F_X,j = F_ee,j * fwd_kins(theta_j)^-1 and we average them (average_frames). With the
    nominal kinematics this is only a first guess (the robot's mm-level errors leak into
    it); the least-squares fit of ImprovedKinematics refines the base frame as one of
    its parameters.

    input:
        data - CalibrationData
        kins - the kinematics to use (the nominal Robot at first; later ImprovedKinematics)
    output: F_X
    """
    if len(data) == 0:
        raise ValueError("estimate_base_frame: no data")
    ests = [to_np(F) @ _core.inv_frame(to_np(kins.fwd_kins(th))) for th, F in zip(data.thetas, data.frames)]
    return to_frame(_core.average_frames(ests))
