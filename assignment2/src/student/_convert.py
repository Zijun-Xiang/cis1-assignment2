"""Conversions between the course's Frame / Rot / vct3 classes and plain numpy arrays."""
from typing import Iterable, List

import numpy as np

from data_types.nominal_types import vct3, Rot, Frame


def to_np(F: Frame) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = F.R.matrix
    T[:3, 3] = F.p.vec.ravel()
    return T


def to_frame(T: np.ndarray) -> Frame:
    T = np.asarray(T, dtype=np.float64)
    return Frame(Rot(matrix=T[:3, :3].copy()), vct3(T[:3, 3]))


def pts_to_np(points: Iterable) -> np.ndarray:
    """List of vct3 (or uvct3, which carries .p) -> (N,3)."""
    rows = []
    for p in points:
        if hasattr(p, "vec"):
            rows.append(np.asarray(p.vec, dtype=np.float64).ravel())
        elif hasattr(p, "p"):
            rows.append(np.asarray(p.p.vec, dtype=np.float64).ravel())
        else:
            rows.append(np.asarray(p, dtype=np.float64).ravel())
    return np.array(rows, dtype=np.float64).reshape(-1, 3)


def np_to_pts(a: np.ndarray) -> List[vct3]:
    return [vct3(row) for row in np.asarray(a, dtype=np.float64).reshape(-1, 3)]
