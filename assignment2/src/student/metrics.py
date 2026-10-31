"""
Pose-error metrics used to report calibration accuracy.
"""

from typing import Dict, Sequence, Tuple

import numpy as np

from data_types.nominal_types import Frame

from . import _core
from ._convert import to_np


def frame_error(F_a: Frame, F_b: Frame) -> Tuple[float, float]:
    """
    The mismatch between two frames, dF = F_a^-1 F_b ~= Frame(I + sk(alpha), epsilon).

    input:
        F_a - reference Frame (e.g. ground truth)
        F_b - Frame to compare (e.g. your prediction)
    output:
        (angle_rad, dist_mm) - ||alpha|| (the rotation angle of dF.R, from the log map,
            so it is exact for large angles too) and ||epsilon|| (the norm of dF.p)
    """
    return _core.frame_error(to_np(F_a), to_np(F_b))


def summarize_errors(errors: Sequence[Tuple[float, float]]) -> Dict[str, float]:
    """
    input:
        errors - list of (angle_rad, dist_mm) pairs from frame_error
    output:
        dict with n, rms_angle, max_angle, rms_pos, max_pos, plus mean_*, p95_* and
        the fraction of samples within the handout's targets (0.2 mm, 1e-3 rad)
    """
    e = np.asarray(list(errors), dtype=np.float64).reshape(-1, 2)
    if e.shape[0] == 0:
        return {"n": 0, "rms_angle": float("nan"), "max_angle": float("nan"),
                "rms_pos": float("nan"), "max_pos": float("nan")}
    ang, pos = e[:, 0], e[:, 1]
    return {
        "n": int(e.shape[0]),
        "rms_angle": float(np.sqrt(np.mean(ang ** 2))), "max_angle": float(ang.max()),
        "mean_angle": float(ang.mean()), "p95_angle": float(np.percentile(ang, 95)),
        "rms_pos": float(np.sqrt(np.mean(pos ** 2))), "max_pos": float(pos.max()),
        "mean_pos": float(pos.mean()), "p95_pos": float(np.percentile(pos, 95)),
        "frac_within_target": float(np.mean((ang <= 1e-3) & (pos <= 0.2))),
    }


def format_summary(s: Dict[str, float]) -> str:
    """One readable line for notebooks / reports."""
    if not s or s.get("n", 0) == 0:
        return "n = 0"
    return (f"n={s['n']}  angle rms {s['rms_angle']:.2e} / max {s['max_angle']:.2e} rad   "
            f"pos rms {s['rms_pos']:.3f} / max {s['max_pos']:.3f} mm   "
            f"within target {100 * s.get('frac_within_target', float('nan')):.1f}%")
