"""
The calibrated kinematic model: ImprovedFwdKins / ImprovedInvKins.
"""

from typing import Dict, Optional

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame
from uncertainty_networks.se3 import log_so3

from ..robot import Robot
from .workflow import CalibrationData
from . import _core
from ._convert import to_np, to_frame


class ImprovedKinematics(Robot):
    """
    A Robot whose fwd_kins / inv_kins account for the calibrated error model.

    Parameter vector (see _core.ParamModel; every entry zero = the nominal robot at F_X_init):
        params[0:6]     xi_X          twist of the base frame about F_X_init, F_X = F_X_init * Exp(xi_X)
        params[6:16]    (t1, t2) x 5  tilt of d_2, d_3, n_4, n_5, n_6, two components each in the
                                      plane perpendicular to the nominal axis (the component along
                                      the axis would not change it and is therefore not a parameter)
        params[16:22]   xi_we         twist of the tool frame, F_we = F_WE * Exp(xi_we)
        params[22:]     c_k,j         per-joint distortion polynomial coefficients
                                      delta_theta_k,sys = offset_k + sum_j c_k,j (theta_k / scale_k)^j,
                                      j = 1..n_poly_k (default degrees (2, 2, 2, 1, 1, 1): the handout's
                                      distortion is offset + scale + a ripple of 0.01 mm / 1e-3 rad
                                      with slope <= 5e-5 on the wrist, so a wrist quadratic only
                                      fits noise -- measured, see report)
    offset_k comes from HomeRobot() (workflow.measure_home_offsets) and is NOT fitted: a
    joint-1..3 offset is the same thing as a shift of the base along that axis.

    Identifiability: the model has exactly one direction the tracker cannot see -- a
    rotation of the base about d_1 together with the opposite rotation of d_2, d_3, n_5,
    n_6 and the tool frame about d_1 (jacobian_rank_report shows it as the zero singular
    value of the full Jacobian). fit() therefore freezes xi_X[0] (fix_gauge=True), which
    defines the base-frame gauge as "x-rotation of F_X_init"; predictions in the tracker
    frame (predict_in_tracker) do not depend on this choice.
    """

    def __init__(self, robot: Robot, F_X_init: Frame, joint_offsets=None, n_poly=2, rot_weight: float = 200.0):
        super().__init__(robot.name + "_improved", robot.F_rob, robot.theta_min, robot.theta_max,
                         robot.t_settle, robot.theta_dot_max, robot.theta_ddot_max)
        self.F_X_init = F_X_init  # starting estimate of the base frame (from the nominal fit)
        self.joint_offsets = np.zeros(6) if joint_offsets is None else np.asarray(joint_offsets, dtype=np.float64).reshape(6)
        self.n_poly = n_poly      # degree(s) of the per-joint distortion polynomial
        self.rot_weight = float(rot_weight)   # mm per rad in the residual: 1e-3 rad counts like 0.2 mm
        self.model = _core.ParamModel(to_np(F_X_init), self.joint_offsets, n_poly,
                                      theta_scale=np.maximum(np.abs(self.theta_min), np.abs(self.theta_max)))
        self.params = self.initial_params()
        self.fit_info: Dict = {}

    # ------------------------------------------------------------------ parameterization

    def initial_params(self) -> np.ndarray:
        """The parameter vector that makes this model identical to the nominal kinematics
        with base frame F_X_init: all zeros, length 22 + sum(n_poly)."""
        return self.model.initial_params()

    @property
    def param_names(self):
        return list(self.model.names)

    def base_frame(self, params: Optional[np.ndarray] = None) -> Frame:
        """F_X (robot base relative to the tracker) = F_X_init * Exp(params[0:6])."""
        return to_frame(self.model.base_frame(self.params if params is None else params))

    def tool_frame(self, params: Optional[np.ndarray] = None) -> Frame:
        return to_frame(self.model.tool_frame(self.params if params is None else params))

    def axes(self, params: Optional[np.ndarray] = None) -> np.ndarray:
        """(6,3) unit axes d_1..3, n_4..6 in the (calibrated) base frame."""
        return self.model.axes(self.params if params is None else params)

    def joint_distortion(self, theta, params: Optional[np.ndarray] = None) -> np.ndarray:
        """delta_theta_sys(theta) for all six joints (offsets + polynomial)."""
        return self.model.joint_distortion(theta, self.params if params is None else params)

    # ------------------------------------------------------------------ kinematics

    def fwd_kins(self, theta, params: Optional[np.ndarray] = None) -> Frame:
        """
        ImprovedFwdKins(theta).

        input: theta - 6 encoder joint values; params - defaults to self.params
        output: F_ee relative to the robot base frame, with the calibrated error model
                (joint offsets + distortion, tilted axes, tool frame offset) applied
        """
        return to_frame(self.model.fwd(theta, self.params if params is None else params))

    def predict_in_tracker(self, theta, params: Optional[np.ndarray] = None) -> Frame:
        """The end-effector pose in the tracker frame: base_frame * fwd_kins."""
        return to_frame(self.model.predict_tracker(theta, self.params if params is None else params))

    def inv_kins(self, F: Frame, seed=None, tol: float = 1e-10, max_iter: int = 50) -> np.ndarray:
        """
        ImprovedInvKins(F) -- provided.

        There is no closed form once the axes are tilted, so this iterates
        on your fwd_kins: both nominal wrist branches (Robot.inv_kins'
        closed form) are refined with damped Gauss-Newton / Levenberg-
        Marquardt steps on the pose error [alpha; epsilon] of
        fwd_kins(theta)^-1 F, and the in-limit solution nearest `seed`
        (default: the current command) is returned. Works as soon as your
        fwd_kins does.

        input: F - desired F_ee relative to the robot base frame
        output: theta with fwd_kins(theta) ~= F; raises ValueError if no
                branch satisfies the joint limits
        """
        seed = self.theta_cmd if seed is None else np.asarray(seed, dtype=np.float64).reshape(6)
        p_w = F.p - F.R * self.F_WE.p - self.F_CW.p   # nominal wrist center -> seeds for the linear joints
        step_max = np.array([50.0] * 3 + [0.3] * 3)
        h = np.array([1e-3] * 3 + [1e-6] * 3)

        def pose_delta(theta):
            dF = self.fwd_kins(theta).inv() * F
            return np.concatenate([log_so3(dF.R.matrix), dF.p.vec.ravel()])

        candidates = []
        for wrist in self.wrist_solutions(F.R):
            theta = np.concatenate([[p_w.x, p_w.y, p_w.z], wrist])
            lam, delta = 1e-4, pose_delta(theta)
            for _ in range(max_iter):
                cost = float(delta @ delta)
                if np.sqrt(cost) < tol:
                    break
                J = np.zeros((6, 6))
                for k in range(6):
                    th = theta.copy()
                    th[k] += h[k]
                    J[:, k] = (pose_delta(th) - delta) / h[k]
                while True:  # Levenberg-Marquardt: shrink the step until the cost decreases
                    JtJ = J.T @ J
                    step = -np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), J.T @ delta)
                    trial = theta + np.clip(step, -step_max, step_max)
                    delta_trial = pose_delta(trial)
                    if float(delta_trial @ delta_trial) < cost or lam > 1e6:
                        theta, delta, lam = trial, delta_trial, max(lam / 10.0, 1e-12)
                        break
                    lam *= 10.0
            theta[3] = (theta[3] + np.pi) % (2 * np.pi) - np.pi
            theta[5] = (theta[5] + np.pi) % (2 * np.pi) - np.pi
            if self.within_limits(theta):
                candidates.append(theta)
        if not candidates:
            raise ValueError(f"inv_kins: no joint solution within limits for {F!r}")
        return min(candidates, key=lambda th: float(np.linalg.norm(th - seed)))

    # ------------------------------------------------------------------ fitting

    def free_mask(self, fix_gauge: bool = True, fixed: Optional[list] = None) -> np.ndarray:
        """Boolean mask of the parameters the fit may change."""
        free = np.ones(self.model.n_params, dtype=bool)
        if fix_gauge:
            free[0] = False                  # base rotation about d_1: the unobservable gauge direction
        for name in fixed or []:
            free[self.model.names.index(name)] = False
        return free

    def residuals(self, params: np.ndarray, data: CalibrationData) -> np.ndarray:
        """
        Flat residual vector, 6 per sample: for dF = predict_in_tracker(theta_j)^-1 F_obs,j
        the rotation vector of dF scaled by rot_weight (mm/rad, default 200 so that the
        handout's 1e-3 rad counts like its 0.2 mm) followed by the translation of dF (mm).
        """
        thetas, frames = data.as_arrays()
        return self.model.residuals(np.asarray(params, dtype=np.float64), thetas, frames, self.rot_weight)

    def fit(self, data: CalibrationData, fix_gauge: bool = True, fixed: Optional[list] = None,
            x0: Optional[np.ndarray] = None, max_nfev: int = 200, **kwargs) -> Dict:
        """
        Non-linear least squares over the free parameters (scipy.optimize.least_squares,
        Levenberg-Marquardt, numerical central-difference Jacobian). Sets self.params and
        returns diagnostics: final cost, number of function evaluations, rms / max residuals.
        """
        thetas, frames = data.as_arrays()
        free = self.free_mask(fix_gauge, fixed)
        x0 = self.params if x0 is None else np.asarray(x0, dtype=np.float64)
        params, info = self.model.fit(thetas, frames, x0=x0, free=free, rot_weight=self.rot_weight, max_nfev=max_nfev)
        self.params = params
        info["fix_gauge"] = bool(fix_gauge)
        info["fixed_params"] = [n for n, f in zip(self.model.names, free) if not f]
        self.fit_info = info
        return info

    def jacobian_rank_report(self, data: CalibrationData, params: Optional[np.ndarray] = None,
                             all_params: bool = True, rel_tol: float = 1e-6) -> dict:
        """
        Singular values of the numerically differentiated residual Jacobian and the
        parameter directions they belong to. With all_params=True (default) the FULL
        parameter vector is analysed, which exposes the model's one null direction (rank
        = n_params - 1); the report also contains the analysis for the free parameters of
        fit() under key 'free' (full rank). Columns are scaled to a common unit (rad-like
        parameters x rot_weight) so singular values are comparable.
        """
        thetas, frames = data.as_arrays()
        p = self.params if params is None else np.asarray(params, dtype=np.float64)
        full = self.model.rank_report(p, thetas, frames, self.rot_weight, free=None, rel_tol=rel_tol)
        free = self.model.rank_report(p, thetas, frames, self.rot_weight, free=self.free_mask(True), rel_tol=rel_tol)
        if all_params:
            out = dict(full)
            out["free"] = free
        else:
            out = dict(free)
            out["full"] = full
        return out

    # ------------------------------------------------------------------ reporting

    def parameters_report(self) -> Dict[str, np.ndarray]:
        """
        The calibrated parameters in physical terms: unit axes and tilt vectors, the
        angles between the axes (gauge-free: compare these with ground truth), the
        per-joint offsets (from homing) and scale factors (linear distortion coefficient),
        the full polynomial coefficients, the tool-frame twist / Frame and the base frame.
        Axes, tilts, the tool twist and F_X are only defined up to the base-frame gauge
        (a rotation about d_1); see compare_with_ground_truth for a gauge-free comparison.
        """
        r = self.model.report(self.params)
        r["F_we"] = to_frame(r["F_we"])
        r["F_X"] = to_frame(r["F_X"])
        r["param_names"] = list(self.model.names)
        r["params"] = self.params.copy()
        return r

    def compare_with_ground_truth(self, truth: Dict, F_tracker: Optional[Frame] = None) -> Dict:
        """
        Debugging/report helper (handout, method 1 (e)): our parameters next to
        rob.ground_truth_parameters(). Angles between axes, joint scales and the
        end-effector pose predictions are gauge-free. If F_tracker is given, the true
        base frame F_X* = F_tracker^-1 F*_rob is used to remove the gauge rotation
        R_g = R_X*^T R_X_hat, so axes, tilts and the tool frame can be compared directly.
        """
        rep = self.model.report(self.params)
        axes_true = np.asarray(truth["axes"], dtype=np.float64)
        ang = lambda u, v: float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))
        pairs = [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]
        out = {
            "axis_angle_pairs": rep["axis_angle_pairs"],
            "axis_angles_deg_est": rep["axis_angles_deg"],
            "axis_angles_deg_true": np.array([ang(axes_true[i], axes_true[j]) for i, j in pairs]),
            "joint_scale_est": rep["joint_scale"], "joint_scale_true": np.asarray(truth["joint_scale"], dtype=np.float64),
            "joint_offset_est": rep["joint_offset"],
            # the true delta_theta_sys at home is c + sigma sin(B) (the ripple's value at 0)
            "joint_offset_true_at_home": np.asarray(truth["joint_offset"]) + np.asarray(truth["ripple_std"]) * np.sin(np.asarray(truth["ripple_phase"])),
        }
        out["axis_angles_deg_err"] = out["axis_angles_deg_est"] - out["axis_angles_deg_true"]
        out["joint_scale_err"] = out["joint_scale_est"] - out["joint_scale_true"]
        out["joint_offset_err"] = out["joint_offset_est"] - out["joint_offset_true_at_home"]
        # The true distortion is c + s theta + sigma sin(A theta + B); its ripple has a local slope of up
        # to sigma*A (3.5e-4 on the slides), so our polynomial's linear coefficient is not the same thing
        # as s. Compare the distortion FUNCTIONS over the travel instead, and s with the slope of the
        # best straight line through the true function (the "effective" scale).
        c, s_, sig = (np.asarray(truth[k], dtype=np.float64) for k in ("joint_offset", "joint_scale", "ripple_std"))
        A, B = (np.asarray(truth[k], dtype=np.float64) for k in ("ripple_freq", "ripple_phase"))
        rms, mx, eff = np.zeros(6), np.zeros(6), np.zeros(6)
        for k in range(6):
            g = np.linspace(self.theta_min[k], self.theta_max[k], 241)
            true_d = c[k] + s_[k] * g + sig[k] * np.sin(A[k] * g + B[k])
            est_d = np.array([self.joint_distortion(np.eye(6)[k] * t)[k] for t in g])
            rms[k], mx[k] = np.sqrt(np.mean((est_d - true_d) ** 2)), np.max(np.abs(est_d - true_d))
            eff[k] = np.polyfit(g, true_d, 1)[0]
        out["joint_distortion_err_rms"], out["joint_distortion_err_max"] = rms, mx
        out["joint_scale_true_effective"] = eff
        out["joint_scale_err_effective"] = out["joint_scale_est"] - eff
        if F_tracker is not None:
            F_X_true = to_np(F_tracker.inv() * truth["F_rob_actual"])
            R_g = F_X_true[:3, :3].T @ rep["F_X"][:3, :3]          # our base = true base rotated by R_g
            axes_true_g = axes_true @ R_g                               # R_g^T d*_k
            out["axis_direction_err_mrad"] = np.array([1e3 * np.arccos(np.clip(rep["axes"][k] @ axes_true_g[k], -1, 1)) for k in range(6)])
            F_we_true = _core.F_WE @ _core.exp_se3(np.asarray(truth["F_we_twist"], dtype=np.float64))
            F_we_true_g = _core.make_frame(R_g.T, np.zeros(3)) @ F_we_true
            out["tool_frame_err_rad_mm"] = _core.frame_error(F_we_true_g, rep["F_we"])
            out["base_frame_err_rad_mm"] = _core.frame_error(F_X_true, rep["F_X"])
            out["gauge_rotation_rad"] = float(np.linalg.norm(_core.log_so3(R_g)))
        return out
