"""
Rigid registration helpers: point-set registration, frame averaging and
marker-body refinement.
"""

from typing import List

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame

from assignments.assignment1.src.tracker import Tracker
from ..simulation import PlacedBody
from .. import simulation
from . import _core
from ._convert import to_np, to_frame, pts_to_np, np_to_pts


def register_points(a: List[vct3], b: List[vct3]) -> Frame:
    """
    Rigid point-set registration (Arun's SVD method, our own implementation): the
    least-squares Frame F with F * a[i] ~= b[i] for every corresponding pair.

    Fewer than 3 pairs, or collinear points, do not determine the rotation and raise
    ValueError. A reflection (det = -1 from the SVD) is corrected by flipping the
    singular direction with the smallest singular value.

    input:
        a - points in the source frame (e.g. a marker body's local layout)
        b - the same points observed in the target frame, same order
    output: Frame mapping a -> b
    """
    A, B = pts_to_np(a), pts_to_np(b)
    if len(A) != len(B):
        raise ValueError("register_points: a and b must have the same number of points")
    return to_frame(_core.register_points(A, B))


def average_frames(frames: List[Frame]) -> Frame:
    """
    Average of several noisy estimates of the same Frame: the positions are averaged
    element-wise; the rotation is the nearest proper rotation (SVD projection) to the
    element-wise mean of the rotation matrices, which is the chordal (Frobenius) mean.

    input:
        frames - List[Frame], all estimates of the same transform
    output: the averaged Frame
    """
    if len(frames) == 0:
        raise ValueError("average_frames: need at least one frame")
    return to_frame(_core.average_frames([to_np(F) for F in frames]))


def refine_marker_body(trk: Tracker, placed: PlacedBody, n_readings: int = 30) -> List[vct3]:
    """
    Improve a user-defined marker body's layout by averaging tracker observations:
    each reading's marker positions b_m,k (tracker frame) are brought back into the
    body's own frame with our register_points, averaged over n_readings and pushed
    into the body with update_marker_body. Costs n_readings * delta_t_samp of
    simulated time (readings of a body that is not visible are skipped).

    Not needed for the robot calibration body (a*_k == a_k by construction).

    input:
        trk - Tracker the body is tracked by (it is added if it is not)
        placed - PlacedBody whose body's nominal marker positions should be refined
        n_readings - number of readings to average
    output: the new List[vct3] of marker positions (also stored on the body)
    """
    body = placed.body
    if placed not in trk.tracked_bodies:
        simulation.add_marker_bodies(trk, [placed])
    nominal = pts_to_np(body.nominal_marker_positions)
    acc, n_ok = np.zeros_like(nominal), 0
    for _ in range(n_readings):
        simulation.sample_marker_bodies(trk)
        b = simulation.most_recent_observed_markers(body)
        if b is None:
            continue
        B = pts_to_np(b)
        F = _core.register_points(nominal, B)            # body frame defined by the current layout
        acc += (B - F[:3, 3]) @ F[:3, :3]                 # F^-1 b: markers back in the body frame
        n_ok += 1
    if n_ok == 0:
        raise RuntimeError("refine_marker_body: the body was never visible")
    new_positions = np_to_pts(acc / n_ok)
    simulation.update_marker_body(body, new_positions)
    return new_positions
