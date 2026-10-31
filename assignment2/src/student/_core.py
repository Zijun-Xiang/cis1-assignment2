"""
Pure-numpy core of our calibration: the parameterized error model, its residuals and
Jacobian, the least-squares fit, the identifiability (rank) analysis and the plan-sizing
formulas. Everything here works on plain 4x4 numpy frames so it can be unit-tested
without the course's Frame/Rot/vct3 classes; kinematics.py / workflow.py are thin
wrappers that convert to and from those classes.

Conventions (same as the course's uncertainty_networks.se3): a twist is xi = [alpha; eps]
(rotation first), Exp(xi) is the 4x4 matrix with rotation Exp(alpha) and translation
J_l(alpha) eps, and frames are perturbed on the right: F = F_nominal * Exp(xi).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

# ----------------------------------------------------------------------------- se3 math

_I3 = np.eye(3)


def skew(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(3)
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def exp_so3(phi) -> np.ndarray:
    """Rodrigues: rotation vector -> rotation matrix."""
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(phi))
    K = skew(phi)
    if th < 1e-12:
        return _I3 + K
    return _I3 + (np.sin(th) / th) * K + ((1.0 - np.cos(th)) / th ** 2) * (K @ K)


def log_so3(R) -> np.ndarray:
    """Rotation matrix -> rotation vector (angle = norm)."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    c = float(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    th = float(np.arccos(c))
    K = R - R.T
    vee = np.array([K[2, 1], K[0, 2], K[1, 0]])
    if th < 1e-12:
        return 0.5 * vee
    if np.pi - th < 1e-7:                       # near pi: use the symmetric part
        A = (R + _I3) / 2.0
        k = int(np.argmax(np.diag(A)))
        axis = A[:, k] / np.sqrt(max(A[k, k], 1e-18))
        return axis * th
    return (th / (2.0 * np.sin(th))) * vee


def _left_jacobian_so3(phi) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64).reshape(3)
    th = float(np.linalg.norm(phi))
    K = skew(phi)
    if th < 1e-6:
        return _I3 + 0.5 * K + (K @ K) / 6.0
    return _I3 + ((1.0 - np.cos(th)) / th ** 2) * K + ((th - np.sin(th)) / th ** 3) * (K @ K)


def exp_se3(xi) -> np.ndarray:
    """Twist [alpha; eps] -> 4x4 rigid transform."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    T = np.eye(4)
    T[:3, :3] = exp_so3(xi[:3])
    T[:3, 3] = _left_jacobian_so3(xi[:3]) @ xi[3:]
    return T


def make_frame(R, p) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=np.float64).reshape(3)
    return T


def inv_frame(T) -> np.ndarray:
    R = T[:3, :3]
    return make_frame(R.T, -R.T @ T[:3, 3])


def frame_error(F_a, F_b) -> Tuple[float, float]:
    """(angle [rad], distance [mm]) of dF = F_a^-1 F_b."""
    D = inv_frame(F_a) @ F_b
    return float(np.linalg.norm(log_so3(D[:3, :3]))), float(np.linalg.norm(D[:3, 3]))


def project_rotation(M) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    if np.linalg.det(U @ Vt) < 0:
        U = U.copy()
        U[:, -1] *= -1
    return U @ Vt


def average_frames(Ts: Sequence[np.ndarray]) -> np.ndarray:
    Ts = np.asarray(Ts, dtype=np.float64)
    return make_frame(project_rotation(Ts[:, :3, :3].sum(axis=0)), Ts[:, :3, 3].mean(axis=0))


def register_points(a, b) -> np.ndarray:
    """Arun's method: least-squares rigid F with b_i ~= F a_i. a, b: (N,3)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != 2 or a.shape != b.shape or a.shape[0] < 3 or a.shape[1] != 3:
        raise ValueError("register_points needs >= 3 corresponding 3-D points")
    ac, bc = a.mean(axis=0), b.mean(axis=0)
    A, B = a - ac, b - bc
    sv = np.linalg.svd(A, compute_uv=False)
    if sv[1] <= 1e-9 * max(sv[0], 1e-12):
        raise ValueError("register_points: points are collinear (or coincident), the rotation is not determined")
    H = A.T @ B
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T     # d = -1 would be a reflection: fix the last axis
    return make_frame(R, bc - R @ ac)


# ----------------------------------------------------------------------------- the model

NOMINAL_AXES = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]] * 2, dtype=np.float64)   # d_1..3, n_4..6
F_CW = make_frame(_I3, [500.0, 0.0, 0.0])
F_WE = make_frame(_I3, [100.0, 0.0, 0.0])


class ParamModel:
    """
    Parameter vector (all zeros = the nominal robot with base frame F_X_init):

        params[0:6]    xi_X    twist of the base frame:        F_X  = F_X_init * Exp(xi_X)
        params[6:16]   (t1,t2) tilt of d_2, d_3, n_4, n_5, n_6: axis = Exp(t1 e_a + t2 e_b) axis_nom,
                               with (e_a, e_b) the two nominal axes perpendicular to axis_nom
                               (a component along the axis itself would not change it)
        params[16:22]  xi_we   twist of the tool frame:        F_we = F_WE * Exp(xi_we)
        params[22:]    c_kj    per-joint distortion polynomial, j = 1..n_poly_k:
                               dtheta_k,sys(theta) = offset_k + sum_j c_kj (theta_k / scale_k)^j
                               (offset_k is fixed from HomeRobot and NOT fitted: it is not separable
                               from the base position for joints 1-3)
    """

    def __init__(self, F_X_init: np.ndarray, joint_offsets=None, n_poly=2, theta_scale=None):
        self.F_X_init = np.asarray(F_X_init, dtype=np.float64).reshape(4, 4)
        self.joint_offsets = np.zeros(6) if joint_offsets is None else np.asarray(joint_offsets, dtype=np.float64).reshape(6)
        n_poly = (n_poly,) * 6 if np.isscalar(n_poly) else tuple(int(n) for n in n_poly)
        if len(n_poly) != 6:
            raise ValueError("n_poly must be an int or a sequence of 6 ints")
        self.n_poly = n_poly
        self.theta_scale = np.array([600.0, 300.0, 600.0, 0.75 * np.pi, 0.75 * np.pi, np.pi]) if theta_scale is None \
            else np.asarray(theta_scale, dtype=np.float64).reshape(6)
        self.n_params = 22 + int(sum(n_poly))
        # names, for reports
        names = [f"base_alpha_{c}" for c in "xyz"] + [f"base_eps_{c}" for c in "xyz"]
        for k in range(1, 6):
            ea, eb = self._perp_basis(k)
            names += [f"tilt_axis{k+1}_{'xyz'[ea]}", f"tilt_axis{k+1}_{'xyz'[eb]}"]
        names += [f"tool_alpha_{c}" for c in "xyz"] + [f"tool_eps_{c}" for c in "xyz"]
        for k in range(6):
            names += [f"poly_j{k+1}_c{j}" for j in range(1, n_poly[k] + 1)]
        self.names = names
        # slices into the parameter vector
        self.sl_base, self.sl_tilt, self.sl_tool = slice(0, 6), slice(6, 16), slice(16, 22)
        self.sl_poly = []
        i = 22
        for k in range(6):
            self.sl_poly.append(slice(i, i + n_poly[k]))
            i += n_poly[k]

    # -- pieces -------------------------------------------------------------------------
    @staticmethod
    def _perp_basis(k: int):
        """Indices of the two nominal unit vectors perpendicular to nominal axis k (k = 0..5)."""
        a = k % 3
        return (a + 1) % 3, (a + 2) % 3

    def initial_params(self) -> np.ndarray:
        return np.zeros(self.n_params)

    def base_frame(self, params) -> np.ndarray:
        return self.F_X_init @ exp_se3(params[self.sl_base])

    def tool_frame(self, params) -> np.ndarray:
        return F_WE @ exp_se3(params[self.sl_tool])

    def tilt_vectors(self, params) -> np.ndarray:
        """(6,3) rotation vectors phi_k (row 0 is zero: d*_1 = d_1)."""
        phi = np.zeros((6, 3))
        t = params[self.sl_tilt]
        for k in range(1, 6):
            ea, eb = self._perp_basis(k)
            phi[k, ea] = t[2 * (k - 1)]
            phi[k, eb] = t[2 * (k - 1) + 1]
        return phi

    def axes(self, params) -> np.ndarray:
        phi = self.tilt_vectors(params)
        return np.array([exp_so3(phi[k]) @ NOMINAL_AXES[k] for k in range(6)])

    def joint_distortion(self, theta, params) -> np.ndarray:
        theta = np.asarray(theta, dtype=np.float64).reshape(6)
        out = self.joint_offsets.copy()
        for k in range(6):
            c = params[self.sl_poly[k]]
            if c.size:
                u = theta[k] / self.theta_scale[k]
                out[k] += np.polyval(np.concatenate([c[::-1], [0.0]]), u)
        return out

    def joint_scale(self, params) -> np.ndarray:
        """Linear coefficient of dtheta_sys(theta) in physical units (mm/mm, rad/rad)."""
        s = np.zeros(6)
        for k in range(6):
            c = params[self.sl_poly[k]]
            if c.size:
                s[k] = c[0] / self.theta_scale[k]
        return s

    # -- kinematics -----------------------------------------------------------------------
    def prepare(self, params):
        """The theta-independent pieces of the model (axes, tool frame, base frame), computed once."""
        return self.axes(params), self.tool_frame(params), self.base_frame(params)

    def fwd_prepared(self, theta, params, prep) -> np.ndarray:
        ax, F_tool, _ = prep
        ts = np.asarray(theta, dtype=np.float64).reshape(6) + self.joint_distortion(theta, params)
        p_c = ts[0] * ax[0] + ts[1] * ax[1] + ts[2] * ax[2]
        R_w = exp_so3(ax[3] * ts[3]) @ exp_so3(ax[4] * ts[4]) @ exp_so3(ax[5] * ts[5])
        T = make_frame(R_w, p_c + F_CW[:3, 3])          # Trans(p_c) F_CW Frame(R_w, 0)
        return T @ F_tool

    def fwd(self, theta, params) -> np.ndarray:
        """F_ee in the robot base frame for encoder values theta."""
        return self.fwd_prepared(theta, params, self.prepare(params))

    def predict_tracker(self, theta, params) -> np.ndarray:
        prep = self.prepare(params)
        return prep[2] @ self.fwd_prepared(theta, params, prep)

    # -- least squares ----------------------------------------------------------------------
    def residuals(self, params, thetas, frames, rot_weight: float = 200.0) -> np.ndarray:
        """Flat residual: for each sample dF = predict^-1 F_obs -> [rot_weight * log(dF.R); dF.p].
        rot_weight (mm per rad) makes 1e-3 rad count like 0.2 mm, the handout's two tolerances."""
        prep = self.prepare(params)
        F_X = prep[2]
        out = np.empty(6 * len(thetas))
        for j, (th, F_obs) in enumerate(zip(thetas, frames)):
            D = inv_frame(F_X @ self.fwd_prepared(th, params, prep)) @ F_obs
            out[6 * j:6 * j + 3] = rot_weight * log_so3(D[:3, :3])
            out[6 * j + 3:6 * j + 6] = D[:3, 3]
        return out

    def param_steps(self) -> np.ndarray:
        """Finite-difference steps per parameter (rad-like 1e-6, mm-like 1e-3)."""
        h = np.full(self.n_params, 1e-6)
        h[3:6] = 1e-3
        h[19:22] = 1e-3
        for k in range(3):
            h[self.sl_poly[k]] = 1e-3
        return h

    def jacobian(self, params, thetas, frames, rot_weight=200.0, free=None) -> np.ndarray:
        """Numerical (central-difference) Jacobian of residuals w.r.t. the free parameters."""
        free = np.ones(self.n_params, bool) if free is None else np.asarray(free, bool)
        idx = np.flatnonzero(free)
        h = self.param_steps()
        J = np.empty((6 * len(thetas), idx.size))
        for c, i in enumerate(idx):
            dp = np.zeros(self.n_params)
            dp[i] = h[i]
            J[:, c] = (self.residuals(params + dp, thetas, frames, rot_weight) -
                       self.residuals(params - dp, thetas, frames, rot_weight)) / (2 * h[i])
        return J

    def fit(self, thetas, frames, x0=None, free=None, rot_weight=200.0, max_nfev=200, verbose=0) -> Tuple[np.ndarray, dict]:
        """Gauss-Newton / Levenberg-Marquardt least squares over the free parameters."""
        from scipy.optimize import least_squares
        x0 = self.initial_params() if x0 is None else np.asarray(x0, dtype=np.float64).copy()
        free = np.ones(self.n_params, bool) if free is None else np.asarray(free, bool)
        idx = np.flatnonzero(free)

        def full(xf):
            x = x0.copy()
            x[idx] = xf
            return x

        def fun(xf):
            return self.residuals(full(xf), thetas, frames, rot_weight)

        def jac(xf):
            return self.jacobian(full(xf), thetas, frames, rot_weight, free)

        res = least_squares(fun, x0[idx], jac=jac, method="lm" if 6 * len(thetas) >= idx.size else "trf",
                            x_scale="jac", max_nfev=max_nfev, xtol=1e-12, ftol=1e-12, gtol=1e-12, verbose=verbose)
        params = full(res.x)
        r = res.fun.reshape(-1, 6)
        info = {
            "status": int(res.status), "message": res.message, "nfev": int(res.nfev),
            "cost": float(res.cost), "n_samples": len(thetas), "n_free_params": int(idx.size),
            "rms_rot_residual_rad": float(np.sqrt(np.mean(np.sum((r[:, :3] / rot_weight) ** 2, axis=1)))),
            "rms_pos_residual_mm": float(np.sqrt(np.mean(np.sum(r[:, 3:] ** 2, axis=1)))),
            "max_pos_residual_mm": float(np.max(np.linalg.norm(r[:, 3:], axis=1))),
        }
        return params, info

    def rank_report(self, params, thetas, frames, rot_weight=200.0, free=None, rel_tol=1e-6, n_weak=4) -> dict:
        """Singular values of the Jacobian, numerical rank and the weakest parameter directions."""
        free = np.ones(self.n_params, bool) if free is None else np.asarray(free, bool)
        idx = np.flatnonzero(free)
        J = self.jacobian(params, thetas, frames, rot_weight, free)
        # scale columns to a common unit so rad-like and mm-like parameters are comparable
        scale = np.ones(idx.size)
        for c, i in enumerate(idx):
            scale[c] = rot_weight if ("alpha" in self.names[i] or "tilt" in self.names[i] or
                                      (self.names[i].startswith("poly_j") and int(self.names[i][6]) >= 4)) else 1.0
        Js = J / scale
        U, s, Vt = np.linalg.svd(Js, full_matrices=False)
        rank = int(np.sum(s > rel_tol * s[0]))
        weak = []
        for r in range(1, min(n_weak, len(s)) + 1):
            v = Vt[-r]
            order = np.argsort(-np.abs(v))[:6]
            weak.append({"singular_value": float(s[-r]), "relative": float(s[-r] / s[0]),
                         "direction": [(self.names[idx[o]], round(float(v[o]), 3)) for o in order]})
        return {"rank": rank, "n_params": int(idx.size), "singular_values": s.tolist(),
                "condition_number": float(s[0] / s[-1]) if s[-1] > 0 else float("inf"),
                "weakest_directions": weak, "names": [self.names[i] for i in idx]}

    # -- reporting -----------------------------------------------------------------------------
    def report(self, params) -> dict:
        ax = self.axes(params)
        ang = lambda u, v: float(np.degrees(np.arccos(np.clip(u @ v, -1.0, 1.0))))
        return {
            "axes": ax, "axis_tilts": self.tilt_vectors(params),
            "axis_angles_deg": np.array([ang(ax[0], ax[1]), ang(ax[0], ax[2]), ang(ax[1], ax[2]),
                                         ang(ax[3], ax[4]), ang(ax[3], ax[5]), ang(ax[4], ax[5])]),
            "axis_angle_pairs": ["d1-d2", "d1-d3", "d2-d3", "n4-n5", "n4-n6", "n5-n6"],
            "joint_offset": self.joint_offsets.copy(), "joint_scale": self.joint_scale(params),
            "joint_poly": [params[self.sl_poly[k]].copy() for k in range(6)],
            "F_we_twist": params[self.sl_tool].copy(), "F_we": self.tool_frame(params),
            "base_twist": params[self.sl_base].copy(), "F_X": self.base_frame(params),
        }


# ----------------------------------------------------------------------------- plan sizing

def orientation_noise_per_reading(markers, sigma_b: float) -> float:
    """Worst-axis 1-sigma orientation error of one registration of this marker layout
    (isotropic marker noise sigma_b): sigma_b / sqrt(min_u sum_k |a_k x u|^2) = sigma_b / sqrt(lambda_min),
    lambda_min the smallest eigenvalue of sum_k (|a_k|^2 I - a_k a_k^T) about the centroid."""
    a = np.asarray(markers, dtype=np.float64)
    a = a - a.mean(axis=0)
    M = sum(np.dot(x, x) * _I3 - np.outer(x, x) for x in a)
    lam = np.linalg.eigvalsh(M).min()
    return float(sigma_b / np.sqrt(lam))


def position_noise_per_reading(markers, sigma_b: float) -> float:
    return float(sigma_b / np.sqrt(len(markers)))


def readings_needed(sigma_per_reading: float, target: float, budget_fraction: float) -> int:
    """Smallest M with sigma_per_reading / sqrt(M) <= budget_fraction * target."""
    return int(np.ceil((sigma_per_reading / (budget_fraction * target)) ** 2))


def slide_step_from_bounds(omega_sys: float, pos_target: float, budget_fraction: float, travel: float) -> float:
    """Linear interpolation between samples spaced h apart is off by at most omega*h/2, so
    h <= 2 * (budget_fraction * pos_target) / omega; rounded down to divide the travel evenly."""
    h_max = 2.0 * budget_fraction * pos_target / omega_sys
    n = int(np.ceil(travel / h_max))
    return travel / n
